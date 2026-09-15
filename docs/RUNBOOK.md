# Runbook: ONNX → TensorRT → Triton

Every command here was run end to end on an NVIDIA L4 against the `greek_1_5`
checkpoint. Substitute your own `<model_id>`.

- [Prerequisites](#prerequisites)
- [Step 0 — Export the ONNX graphs](#step-0--export-the-onnx-graphs)
- [Step 1 — Build the TensorRT engines](#step-1--build-the-tensorrt-engines)
- [Step 2 — Serve with Triton](#step-2--serve-with-triton)
- [Step 3 — Verify](#step-3--verify)
- [Troubleshooting](#troubleshooting)
- [Raw trtexec reference](#raw-trtexec-reference)
- [Running this in production](#running-this-in-production)

---

## Prerequisites

| | |
|---|---|
| GPU | Any CUDA GPU with enough memory to build the engines (~6 GB headroom). Verified on L4 23 GB. |
| Docker | With the NVIDIA container runtime (`nvidia-ctk`, `--gpus all` works). |
| Image | `nvcr.io/nvidia/tritonserver:24.06-py3` — ~24 GB, ships both `tritonserver` and `/usr/src/tensorrt/bin/trtexec`. |
| Disk | ~3 GB per checkpoint cached, ~700 MB per built model repository. |

```bash
docker pull nvcr.io/nvidia/tritonserver:24.06-py3
```

The exporter itself runs on the host, not in the container. Install its
dependencies per the README.

---

## Step 0 — Export the ONNX graphs

```bash
python convert.py configs/<model_id>.yaml --dry-run   # validate, fetch nothing
python convert.py configs/<model_id>.yaml
```

Produces:

```
triton_repos/<model_id>/
├── triton_model_repository/     # 16 models: 13 .onnx + config.pbtxt + 2 python backends
└── reference_styles/            # <speaker>.npy (+ .pt) per speaker
```

Twelve graphs are exported here. `int_steps_2` is pure tensor arithmetic with no
weights, so it ships pre-traced in the template and is copied, not re-exported.

---

## Step 1 — Build the TensorRT engines

A `.plan` is tied to the **GPU architecture, the TensorRT version and the CUDA
version** that produced it. It cannot be built in CI and shipped to a
heterogeneous fleet — it has to be built on a machine with the same GPU as the
one that will serve it.

```bash
# A container that has trtexec, with the repo bind-mounted
docker run -d --name trt --gpus all --shm-size=1g \
  -v "$PWD":/workspace/repo -w /workspace/repo \
  nvcr.io/nvidia/tritonserver:24.06-py3 sleep infinity

# See what would be built, without building it
docker exec trt python3 tools/build_trt_engines.py \
  triton_repos/<model_id>/triton_model_repository --print

# Build (about 6-10 min for all eight on an L4)
docker exec trt python3 tools/build_trt_engines.py \
  triton_repos/<model_id>/triton_model_repository --run --workspace-mb 3072
```

Expected output:

```
# utterance budget: 0.1s / 6.4s / 19.2s  ->  4 / 256 / 768 alignment frames
# phoneme budget:   4 / 128 / 512 tokens
# precision:        fp16 permitted, except ['generator'] (fp32: fp16 overflows SineGen's phase)

[trt] building bert (fp16) ...                 [trt] built bert  (14 MB)
[trt] building prosody_text_encoder (fp16) ... [trt] built prosody_text_encoder  (12 MB)
[trt] building prosody_lstm (fp16) ...         [trt] built prosody_lstm  (4 MB)
[trt] building prosody_dur_proj (fp16) ...     [trt] built prosody_dur_proj  (0 MB)
[trt] building diffusion_steps (fp16) ...      [trt] built diffusion_steps  (58 MB)
[trt] building prosody_predictor_ftrain (fp16) [trt] built prosody_predictor_ftrain  (18 MB)
[trt] building decoder (fp16) ...              [trt] built decoder  (71 MB)
[trt] building generator (fp32) ...            [trt] built generator  (137 MB)
```

### Sizing the shape profiles

Every profile is derived from **one utterance budget**, because the graphs are
links in one chain and have to describe the same sentence. One alignment frame
is 600 output samples at 24 kHz — exactly 25 ms.

```bash
--max-seconds 19.2    # longest utterance the fleet must synthesize
--opt-seconds 6.4     # length to tune kernels for; pick your median request
--max-tokens  512     # longest phoneme sequence (PLBERT's limit is 512)
--opt-tokens  128
```

`opt` is not a bound — it is where TensorRT times its kernel candidates. Shapes
far from it still run, just on kernels chosen for a different length. Set it to
your actual median utterance, not the midpoint of the range.

Raising `--max-seconds` raises engine build time and memory. The ceiling is
whatever you set; the phoneme ceiling is a hard 512 from PLBERT.

### Other flags

| Flag | Effect |
|---|---|
| `--only bert,decoder` | Build a subset. |
| `--no-fp16` | Build everything in fp32. Slow; use to isolate a precision bug. |
| `--generator-fp16` | Allow fp16 in the generator. **Produces NaN audio** — see below. |
| `--workspace-mb N` | TensorRT scratch memory. Lower it if a build OOMs. |

### The generator is built in fp32 on purpose

HiFi-GAN's `SineGen` accumulates an unbounded oscillator phase, then scales it
by the upsample factor before interpolating:

```
phase = cumsum(f0 / 24000) * 2*pi                  # grows with utterance length
phase = interpolate(phase * upsample_scale, ...)   # upsample_scale = 300
sines = sin(phase)
```

`harmonic_num` is 8, so the eighth harmonic accumulates eight times faster than
the fundamental. On ordinary speech that product passes **65504**, the largest
finite fp16 value: the tensor becomes `inf`, `sin(inf)` is `NaN`, and the entire
waveform comes back NaN with **no error anywhere** — HTTP 200, correct shape,
all NaN. The corruption then persists for later requests on the same engine
context until Triton restarts.

The threshold is **data-dependent, not a length cutoff** — higher F0 reaches it
sooner, so a short utterance from a high-pitched voice triggers it too.

Measured on `Multilingual`, melina, peak F0 308 Hz:

| | harmonic 1 | harmonic 8 |
|---|---:|---:|
| 263 frames (6.6 s) | 12,716 | 101,727 |
| 370 frames (9.3 s) | 17,889 | 143,113 |

StyleTTS2's `SineGen` has a phase-wrapping guard for exactly this, commented out
upstream with the note *"To prevent torch.cumsum numerical overflow"*. Until
that is restored, fp32 is the only safe setting for this engine. It costs
roughly 2–3× on the vocoder, which is the heaviest stage — so it is worth fixing
at the source.

---

## Step 2 — Serve with Triton

```bash
docker run -d --name serve --gpus all --shm-size=2g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD":/workspace \
  -v "$PWD/triton_repos/<model_id>/triton_model_repository":/models \
  -v "$PWD/triton_repos/<model_id>/reference_styles":/workspace/reference_styles \
  -w /workspace \
  nvcr.io/nvidia/tritonserver:24.06-py3 sleep infinity

# The frontend python backend needs a phonemizer. The stock image has none.
docker exec serve bash -lc '
  apt-get update -qq && apt-get install -y -qq espeak-ng espeak-ng-data
  pip install -q --no-cache-dir phonemizer nltk
  python3 -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab'

docker exec -d serve bash -lc '
  export PYTHONPATH=/workspace
  tritonserver --model-repository=/models \
    --strict-readiness=true --exit-on-error=false --log-info=false \
    > /tmp/triton.log 2>&1'

# Wait for readiness
until curl -sf http://127.0.0.1:8000/v2/health/ready; do sleep 2; done
```

Ports: **8000** HTTP, **8001** gRPC, **8002** Prometheus metrics.

### What the mounts are for

| Mount | Why |
|---|---|
| repo at `/workspace` | The `frontend` backend imports `common_code`. It searches two and three levels above the model repository, then `/workspace`. |
| repository at `/models` | `--model-repository` points here. |
| styles at `/workspace/reference_styles` | The path in `frontend/config.pbtxt`'s `REFERENCE_STYLES_DIR` parameter. |

`PYTHONPATH=/workspace` makes `common_code` and `config/json_files/` importable
from inside the backend process.

**The serving container does not need PyTorch.** The frontend reads
`<speaker>.npy` with numpy. Importing torch there would pull a multi-gigabyte
CUDA-linked dependency into every backend process to read 256 floats.

---

## Step 3 — Verify

```bash
# Config vs. graph: names, dtypes, ranks, and ensemble wiring. Exits non-zero
# on any mismatch, so it works as a CI gate.
python tools/check_triton_contract.py triton_repos/<model_id>/triton_model_repository

# End-to-end request, with per-model latency from Triton's own counters
python tools/smoke_test_triton.py \
  --speaker chloe --language el \
  --text "Καλημέρα, πώς είστε σήμερα;" \
  --out /tmp/out.wav --repeat 5 --stats
```

Measured on greek_1_5, single L4, batch 1, 3 diffusion steps:

| Utterance | Speaker | Audio | p50 | RTF |
|---|---|---:|---:|---:|
| `Ναι.` | chloe | 0.65 s | 30 ms | 0.046 |
| `Καλησπέρα σας.` | chloe | 1.40 s | 41 ms | 0.029 |
| One sentence | chloe | 4.08 s | 96 ms | 0.024 |
| One sentence | dora | 4.05 s | 94 ms | 0.023 |
| Three sentences | chloe | 9.65 s | 226 ms | 0.023 |

**Always check the output is finite**, not just that the request returned 200 —
the fp16 failure above is silent. `smoke_test_triton.py` warns on a silent
waveform; for NaN, check `np.isfinite(audio).all()`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `at least one version must be available under the version policy of model 'styletts2_ensemble'` | Ensembles need a version directory; git cannot track an empty one. | The exporter creates it. If you assembled the repo by hand: `mkdir -p <repo>/styletts2_ensemble/1` |
| Audio returns as all NaN, and stays NaN for later requests | `generator` built with `--fp16`. | Rebuild it fp32 (the default). Restart Triton — the context stays poisoned. |
| `request specifies invalid shape for input 'TOKENS'` | Phoneme sequence exceeds `--max-tokens` (512). | Split the text upstream, or raise the budget — PLBERT's own limit is 512. |
| `failed to load model 'X': backend 'tensorrt' ... no model.plan` | Engines not built for this model. | Run Step 1. |
| `No module named 'phonemizer'` in `frontend` | Serving container missing deps. | Step 2's `docker exec ... pip install`. |
| `no style vector for speaker 'X'` | Styles not mounted, or speaker absent from the export config. | Check the `reference_styles` mount and the config's `speakers:` block. |
| `sending FP16 data via JSON is not supported` | Calling an fp16-output model directly over HTTP JSON. | Use the ensemble, or the binary tensor protocol / gRPC. |
| Engine build OOMs | Workspace too large for free GPU memory. | Lower `--workspace-mb`. |

Server logs: `docker exec serve cat /tmp/triton.log`.
Model states: `curl -s -X POST localhost:8000/v2/repository/index`.

---

## Raw trtexec reference

`tools/build_trt_engines.py --print` emits these. Shown for a 19.2 s / 512-token
budget, so you can run one by hand when debugging a single engine.

```bash
TRT=/usr/src/tensorrt/bin/trtexec
REPO=triton_repos/<model_id>/triton_model_repository

$TRT --onnx=$REPO/bert/1/model.onnx --saveEngine=$REPO/bert/1/model.plan --fp16 \
  --minShapes=TOKENS:1x4,ATTENTION_MASK:1x4 \
  --optShapes=TOKENS:1x128,ATTENTION_MASK:1x128 \
  --maxShapes=TOKENS:1x512,ATTENTION_MASK:1x512

$TRT --onnx=$REPO/prosody_text_encoder/1/model.onnx --saveEngine=$REPO/prosody_text_encoder/1/model.plan --fp16 \
  --minShapes=D_EN:1x512x4,S_OUT:1x128,TEXT_MASK_PRED:1x4 \
  --optShapes=D_EN:1x512x128,S_OUT:1x128,TEXT_MASK_PRED:1x128 \
  --maxShapes=D_EN:1x512x512,S_OUT:1x128,TEXT_MASK_PRED:1x512

$TRT --onnx=$REPO/prosody_lstm/1/model.onnx --saveEngine=$REPO/prosody_lstm/1/model.plan --fp16 \
  --minShapes=D_OUT:1x4x640 --optShapes=D_OUT:1x128x640 --maxShapes=D_OUT:1x512x640

$TRT --onnx=$REPO/prosody_dur_proj/1/model.onnx --saveEngine=$REPO/prosody_dur_proj/1/model.plan --fp16 \
  --minShapes=X_OUT:1x4x512 --optShapes=X_OUT:1x128x512 --maxShapes=X_OUT:1x512x512

$TRT --onnx=$REPO/diffusion_steps/1/model.onnx --saveEngine=$REPO/diffusion_steps/1/model.plan --fp16 \
  --minShapes=BERT_DUR:1x4x768,REF_S:1x256 \
  --optShapes=BERT_DUR:1x128x768,REF_S:1x256 \
  --maxShapes=BERT_DUR:1x512x768,REF_S:1x256

$TRT --onnx=$REPO/prosody_predictor_ftrain/1/model.onnx --saveEngine=$REPO/prosody_predictor_ftrain/1/model.plan --fp16 \
  --minShapes=EN:1x640x4,S_OUT:1x128 \
  --optShapes=EN:1x640x256,S_OUT:1x128 \
  --maxShapes=EN:1x640x768,S_OUT:1x128

$TRT --onnx=$REPO/decoder/1/model.onnx --saveEngine=$REPO/decoder/1/model.plan --fp16 \
  --minShapes=ASR:1x512x4,F0_OUT:1x1x4,N_OUT:1x1x4,REF:1x128 \
  --optShapes=ASR:1x512x256,F0_OUT:1x1x256,N_OUT:1x1x256,REF:1x128 \
  --maxShapes=ASR:1x512x768,F0_OUT:1x1x768,N_OUT:1x1x768,REF:1x128

# NOTE: no --fp16 here. See "The generator is built in fp32 on purpose".
$TRT --onnx=$REPO/generator/1/model.onnx --saveEngine=$REPO/generator/1/model.plan \
  --minShapes=X_OUT:1x512x8,REF:1x128,F0_PRED:1x8 \
  --optShapes=X_OUT:1x512x512,REF:1x128,F0_PRED:1x512 \
  --maxShapes=X_OUT:1x512x1536,REF:1x128,F0_PRED:1x1536
```

Not built as engines: `text_encoder`, `bert_encoder`, `int_steps_1`,
`int_steps_2`, `prep_decoder` run on ONNX Runtime; `frontend` and `last_step`
are Python backends. `int_steps_2` cannot be a TensorRT engine at all — its
output length depends on input *values*, not input shapes.

---

## Running this in production

The reference deployment (`VoicingAI/PR-TTS-Service`, branch
`client/telmex-tts`) builds engines at **container start**, not build time, and
shares one EFS volume across four replicas. Three details are worth copying:

```bash
flock -x                              # exclusive lock per engine
  re-check if [ -f "$ENGINE_PATH" ]   # a peer may have finished while we waited
  trtexec --saveEngine=$ENGINE.tmp    # build to a temp name
  mv $ENGINE.tmp $ENGINE              # atomic rename: never a partial .plan
flock -u
```

- The **double-check inside the lock** stops N replicas building the same engine N times.
- The **temp-then-rename** stops a crashed build leaving a truncated `.plan` that every later boot treats as valid.
- A **persistent volume** with `CLEANUP_ENABLED=false` and `FORCE_REBUILD=false` turns a pod restart into a download rather than a rebuild.

`tools/build_trt_engines.py` already does the temp-then-rename. It does not take
a cross-process lock — add one if several replicas share a volume.

Also worth knowing: TTS latency varies by an order of magnitude with text
length, so load-balance on **outstanding requests**, not round-robin. The
reference deployment uses `least_outstanding_requests` on its ALB.
