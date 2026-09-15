# StyleTTS2 → ONNX → Triton

Exports a StyleTTS2 checkpoint into the twelve ONNX graphs that the Triton
ensemble in `triton_model_repoistory_with_diffusion/` is wired for, computes a
style vector per speaker, and assembles a ready-to-serve model repository.

One YAML file describes one export. Nothing else is needed:

```bash
python convert.py configs/examples/local.yaml
```

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [How the pipeline fits together](#how-the-pipeline-fits-together)
- [Config reference](#config-reference)
- [CLI reference](#cli-reference)
- [Adding a new model](#adding-a-new-model)
- [Serving it: TensorRT + Triton](#serving-it-tensorrt--triton)
- [Tools](#tools)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)
- [Further reading](#further-reading)

---

## Install

```bash
# Install the torch build matching your CUDA driver first
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# The phonemizer needs espeak-ng as a system package
sudo apt-get install -y espeak-ng
```

Credentials for object stores go in `.env` (gitignored) — copy `.env.example`
and fill it in. They are read only from the environment; nothing in the repo
reads keys from a file.

**Supported torch versions:** 2.1 and up. Exports are pinned to the TorchScript
exporter (`dynamo=False`). torch 2.9 changed `torch.onnx.export` to default to
the dynamo exporter, which handles this model's `dynamic_axes`, LSTMs and
`pack_padded_sequence` very differently; pinning keeps a torch upgrade from
silently changing the graphs.

---

## Quick start

Validate a config without touching the network:

```bash
python convert.py configs/examples/local.yaml --dry-run
```

Run the export:

```bash
python convert.py configs/examples/local.yaml
```

Output lands in two directories:

```
triton_repos/<model_id>/
├── triton_model_repository/     # mount this as Triton's --model-repository
│   ├── bert/1/model.onnx
│   ├── decoder/1/model.onnx
│   ├── ...  (+ config.pbtxt for each, copied from the template)
│   └── styletts2_ensemble/config.pbtxt
└── reference_styles/
    └── <speaker>.pt             # mount at /workspace/reference_styles
```

plus `model_registry.json`, which records what every export produced.

Check the result before deploying:

```bash
python tools/check_triton_contract.py triton_repos/<model_id>/triton_model_repository
```

---

## How the pipeline fits together

A request flows through fifteen Triton steps. Twelve of them are ONNX graphs
this exporter produces; three are fixed assets from the template.

```
TEXT ─► frontend ────────────────────────► phoneme tokens, mask, REF_S, speed
         (python: phonemize + tokenize + load the speaker's style vector)
          │
          ├─► text_encoder ──────────────► T_EN      [B, 512, tokens]
          │
          └─► bert ──► bert_encoder ─────► D_EN      [B, 512, tokens]  (fp16)
                 │
                 └──► diffusion_steps ───► S_PRED    [B, 256]   sampled style
                            │
                            ▼
                      int_steps_1 ───────► REF, S_OUT, TEXT_MASK_PRED
                      (blend sampled style with the speaker's, weights α/β)
                            │
                            ▼
       prosody_text_encoder ──► prosody_lstm ──► prosody_dur_proj
                                                   │  DURATION [B, tokens, 50]
                                                   ▼
                                            int_steps_2
                          (durations ► alignment ► expand to frames)
                                   │  EN [B, 640, frames]
                                   │  ASR [B, 512, frames]
                                   ▼
                     prosody_predictor_ftrain ──► F0_PRED, N_PRED
                                   │
                                   ▼
                             prep_decoder ──► F0_OUT, N_OUT   (strided convs)
                                   │
                                   ▼
                               decoder ──► X_OUT   (AdaIN residual stack)
                                   │
                                   ▼
                              generator ──► AUDIO_OUT   (HiFi-GAN vocoder)
                                   │
                                   ▼
                              last_step ──► AUDIO   [samples]
```

**Why the decoder is split three ways.** The training-time decoder is one
module. Here it becomes `prep_decoder` (the strided F0/energy convolutions),
`decoder` (the AdaIN residual stack) and `generator` (the vocoder), so the
three can be scheduled — and quantised — independently. `model_loader.py`
routes a checkpoint's single `decoder` state dict across all three and fails
loudly if any parameter is left unfilled.

**Which graphs are verified.** After each export the graph is re-run under
onnxruntime and compared against PyTorch. Ten of the twelve are deterministic
and get a full value comparison; `diffusion_steps` and `generator` contain
`Random*` nodes (ancestral noise, and `SineGen`'s random initial phase) so only
their output shapes and finiteness are checked.

---

## Config reference

```yaml
model_id: SpanishV2          # required. Names the output directory.

stores:                      # optional. Omit to read every path off disk.
  <name>:
    type: s3 | local
    # --- type: s3 ---
    bucket: my-bucket
    endpoint_url: https://...        # omit for AWS; set for B2/MinIO/R2
    region: us-east-1                # optional
    access_key_env: AWS_ACCESS_KEY   # env var holding the key
    secret_key_env: AWS_SECRET_KEY
    # --- type: local ---
    root: /data/tts                  # relative paths resolve under this

defaults:
  store: <name>              # store used by any path without an explicit one

checkpoint:                  # required
  store: <name>              # optional, overrides defaults.store
  model: path/to/model.pth
  config: path/to/config_ft.yml

plbert:                      # required. A directory: config.yml + step_*.t7
  store: <name>
  path: PLBERTs/multi

speakers:                    # required, at least one
  <speaker>:
    store: <name>            # optional
    reference: path/to/reference.wav
    language: es             # or: languages: [de, en, fr]

export:
  device: cuda:0             # falls back to CPU when no GPU is present
  opset: 17
  modules: all               # or a list, e.g. [bert, text_encoder, generator]
  overwrite: false           # re-export graphs and styles that already exist
  verify: true               # compare each graph against PyTorch
  strict: true               # stop on the first failure
  diffusion_num_steps: 3     # unrolled into diffusion_steps.onnx (>= 2)

triton:
  template: triton_model_repoistory_with_diffusion
  output_root: triton_repos

style:
  alpha: 0.3                 # acoustic style: sampled vs reference, in [0, 1]
  beta: 0.7                  # prosodic style: same blend
  cache_dir: style_cache

cache_root: raw_models       # where downloaded assets are cached
```

A bare string is shorthand for a path in `defaults.store`; an **absolute** path
is always read off local disk, whatever the default store is.

Unknown keys are rejected with the offending key named, so a typo fails before
anything is downloaded.

### Stores

`type: s3` covers AWS S3 and every S3-compatible store. For Backblaze B2, find
your endpoint with:

```bash
curl -s -u "<keyId>:<appKey>" https://api.backblazeb2.com/b2api/v3/b2_authorize_account \
  | python -c "import json,sys; print(json.load(sys.stdin)['apiInfo']['storageApi']['s3ApiUrl'])"
```

Downloads are cached under `cache_root/` and keyed by `<bucket>/<key>`, so
re-running an export costs nothing. Files are written to a `.part` name and
renamed only on completion — an interrupted run never leaves a truncated file
that a later run mistakes for a cache hit.

---

## CLI reference

```
python convert.py <config.yaml> [options]

  --speaker NAME     only build styles for this speaker (repeatable).
                     ONNX graphs are shared across a model's speakers regardless.
  --modules LIST     comma-separated subset to export
  --device DEV       override export.device
  --overwrite        re-export graphs and recompute styles that exist
  --no-verify        skip the onnxruntime check (faster, much less safe)
  --keep-going       log export failures and continue
  --dry-run          validate the config and print the plan, fetch nothing
  --list-modules     print exportable module names and what each one is
```

---

## Adding a new model

1. Write a config. Start from `configs/examples/local.yaml` for files already
   on disk, or `configs/examples/mixed_stores.yaml` for object stores.

2. Check it resolves:

   ```bash
   python convert.py configs/my-model.yaml --dry-run
   ```

3. Export. Start with the cheap graphs to shake out config problems before
   committing to the vocoder:

   ```bash
   python convert.py configs/my-model.yaml --modules bert,text_encoder
   python convert.py configs/my-model.yaml
   ```

4. Verify the repository is internally consistent:

   ```bash
   python tools/check_triton_contract.py triton_repos/my-model/triton_model_repository
   ```

The checkpoint must be a StyleTTS2 training checkpoint — a `.pth` with a `net`
key holding one state dict per submodule, and a `decoder` entry containing
`F0_conv.*` / `N_conv.*`. A `module.` prefix (from DDP) is stripped if present.
If `model_params` in the paired `config_ft.yml` disagrees with the checkpoint,
the load fails with the names of the parameters that would have been left
randomly initialised, rather than producing silent noise.

## Serving it: TensorRT + Triton

The ONNX graphs are the input to engine building, not a substitute for it. Eight
models declare `backend: "tensorrt"` and need a `model.plan` built **on a machine
with the same GPU that will serve them** — a `.plan` is tied to the GPU
architecture, the TensorRT version and the CUDA version that produced it.

```bash
# 1. A container that has trtexec
docker run -d --name trt --gpus all -v "$PWD":/workspace/repo -w /workspace/repo \
  nvcr.io/nvidia/tritonserver:24.06-py3 sleep infinity

# 2. Build the engines. Every shape profile is derived from one utterance
#    budget, so the graphs in the chain agree about the same sentence.
docker exec trt python3 tools/build_trt_engines.py \
  triton_repos/<model_id>/triton_model_repository --run

# 3. Serve
docker run -d --name serve --gpus all -p 8000:8000 -p 8001:8001 \
  -v "$PWD":/workspace \
  -v "$PWD/triton_repos/<model_id>/triton_model_repository":/models \
  -v "$PWD/triton_repos/<model_id>/reference_styles":/workspace/reference_styles \
  nvcr.io/nvidia/tritonserver:24.06-py3 \
  bash -lc 'PYTHONPATH=/workspace tritonserver --model-repository=/models'

# 4. Check it
python tools/check_triton_contract.py triton_repos/<model_id>/triton_model_repository
python tools/smoke_test_triton.py --speaker <name> --language <lang> --text "..." --stats
```

**[docs/RUNBOOK.md](docs/RUNBOOK.md) has the full procedure**: every `trtexec`
invocation, what the mounts are for, how to size the shape budget, a
troubleshooting table, and how the production deployment coordinates engine
builds across replicas.

Two things that bite people:

- **The generator must be built in fp32.** Its oscillator phase overflows fp16
  and the whole waveform comes back as NaN — with no error, and it stays broken
  for later requests until Triton restarts. `build_trt_engines.py` does this by
  default; the arithmetic is in the runbook.
- **The serving container does not need PyTorch.** The frontend reads
  `<speaker>.npy` with numpy. It does need `espeak-ng`, `phonemizer` and `nltk`.

### Measured

greek_1_5, single L4, batch 1, 3 diffusion steps:

| Utterance | Audio | p50 | RTF |
|---|---:|---:|---:|
| `Ναι.` | 0.65 s | 30 ms | 0.046 |
| `Καλησπέρα σας.` | 1.40 s | 41 ms | 0.029 |
| One sentence | 4.08 s | 96 ms | 0.024 |
| Three sentences | 9.65 s | 226 ms | 0.023 |

Phoneme ceiling is 512 tokens (PLBERT's limit, ~400 characters of Greek); past
it Triton rejects the request with a shape error naming the input.

---


## Tools

| Command | What it does |
| --- | --- |
| `python tools/generate_configs.py` | Regenerates `configs/generated/` from the legacy speaker JSON files. Migration aid — once the generated configs are reviewed, they are the source of truth. |
| `python tools/check_triton_contract.py <repo>` | Compares every `config.pbtxt` against its ONNX graph (names, dtypes, ranks) and checks the ensemble wiring for orphaned tensors. Exits non-zero on any mismatch, so it works as a CI gate. |
| `python tools/build_trt_engines.py <repo> --print` | Prints the `trtexec` commands for every engine, with shape profiles derived from one utterance budget. `--run` executes them. |
| `python tools/smoke_test_triton.py --speaker X --language Y` | Sends a request through the ensemble, saves the audio, and reports per-model latency from Triton's own counters. |
| `python convert.py --list-modules` | Lists the exportable graphs with a note on each. |

`tools/reference/int_steps_2_source.py` is the readable source for the
`int_steps_2` graph, which ships pre-traced in the template. It is not a live
backend; see its docstring.

---

## Known limitations

**Diffusion steps are fixed at export time.** `ADPM2ONNX` loops in Python, so
tracing unrolls it and the step count becomes part of the graph's structure.
Set `export.diffusion_num_steps` and re-export to change it. Making it a
genuine runtime input requires rewriting the loop as an ONNX `Loop` op.

`NUM_STEPS` used to be declared as an input on `diffusion_steps` and routed
through the ensemble, but the exported graph never had such an input — the
tracer folded it away. Triton would refuse to load the model. It has been
removed.

**Classifier-free guidance is not implemented.** `EMBEDDING_SCALE` was likewise
accepted from clients and silently dropped: the frontend never forwarded it and
`DiffusionONNX` never consumed it. It has been removed rather than left as a
no-op knob. Restoring it means running `StyleTransformer1d` twice per denoise
step — conditional and masked — and blending, roughly doubling diffusion cost.

**The generator runs in fp32, so the vocoder is over half the latency budget.**
fp16 overflows its oscillator phase and returns NaN audio silently. StyleTTS2's
`SineGen` has a phase-wrapping guard for this commented out upstream; restoring
it is the single biggest win available. See
[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

**Everything is batch 1.** Every engine profile is built for a batch of one, so
there is no batched inference to be had without rebuilding across a batch axis
*and* solving padding for the alignment step, whose output length differs per
request.

**Triton and the torch reference path differ for hifigan decoders.** The
one-frame shift applied to `en` and `asr` is present in
`inference_core.StyleTTS2Synth.synthesize` but not in the traced `int_steps_2`
graph, where the guard is `self.hifigan_mode == "hifigan"` against a value
assigned `True`. Fixing it means re-tracing that graph; see
`tools/reference/int_steps_2_source.py`.

**`common_code/inference_core.py` is untested.** It is the pure-PyTorch
reference synthesizer, useful for A/B-ing against Triton output. Several
crash-on-first-call bugs are fixed, but it has not been run end to end against
a real checkpoint — and note the hifigan shift above means it would not agree
with Triton even if it ran.

**There are no automated tests.** See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)
for the three levels that would pay for themselves fastest.

**`int_steps_2` has no exporter here.** Its graph is pure tensor arithmetic
with no weights, so it is checkpoint-independent and ships pre-traced in the
template rather than being re-exported per model.

---

## Repository layout

```
convert.py                  entry point: python convert.py <config.yaml>

styletts2_export/           the exporter
  config.py                 YAML schema, validation, defaults
  assets.py                 S3 / S3-compatible / local fetching, with caching
  checkpoint.py             build the model, route a checkpoint into it
  models.py                 ONNX-friendly prosody stack + model factory
  styles.py                 per-speaker style vectors, content-hash cached
  triton.py                 template copy + model_registry.json
  cli.py                    argument parsing and the run sequence
  onnx/
    wrappers.py             nn.Modules reshaped so the exporter can trace them
    harness.py              how to export and verify ONE module
    pipeline.py             WHAT to export, in what order

configs/
  examples/                 hand-written, annotated starting points
  generated/                one per model, from the legacy JSON files
  <model_id>.yaml           your own

config/json_files/          speaker / language / PLBERT static config
common_code/                vendored StyleTTS2 model definitions
triton_model_repoistory_with_diffusion/
                            config.pbtxt template + python backends

tools/
  build_trt_engines.py      ONNX -> TensorRT, profiles from one utterance budget
  check_triton_contract.py  config.pbtxt vs. graph; CI gate
  smoke_test_triton.py      end-to-end request + per-model latency
  generate_configs.py       migrate the legacy speaker JSON to YAML
  reference/                source for graphs that ship pre-traced

docs/
  RUNBOOK.md                TensorRT + Triton, every command
  NEXT_STEPS.md             open defects and where to go next
  optimization.md           how the pipeline got this shape, and why
```

The package is imported by path from the repo root — there is no
`pyproject.toml` yet, so run `convert.py` from the repository directory.

---

## Further reading

| Document | What is in it |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Every TensorRT and Triton command, mounts, shape budgets, troubleshooting, production notes. |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | What works, every defect found and fixed, what is still open, and a suggested order of work. |
| [docs/optimization.md](docs/optimization.md) | Why the pipeline has this shape: the graph breaks, the five optimization stages, precision and shape theory. |
