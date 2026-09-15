# StyleTTS2 Inference Optimization

How a single PyTorch speech model became sixteen Triton models, eight TensorRT engines and a fixed shape budget — what each step bought, what it cost, and which knobs turned out to be connected to nothing.

| | |
|---|---|
| **Triton models** | 16 |
| **TensorRT engines** | 8 |
| **Ensemble hops per request** | 15 |
| **Serving fleet** | 4 × NVIDIA L4 |
| **Max utterance** | 19.2 s |
| **Timeline** | Feb 2025 → Jun 2026 |

---

## 1. Why StyleTTS2 resists being one graph

Every graph compiler — ONNX Runtime, TensorRT, TorchInductor — makes the same bargain: give me a computation whose *structure* is fixed and whose *shapes* are derivable from input shapes, and I will fuse kernels, pick tuned implementations and eliminate the Python interpreter. StyleTTS2 cannot make good on either half of that bargain end to end.

The model is not a network so much as a small pipeline of eight sub-networks with two non-differentiable joints in the middle. The blocking joint is duration prediction. The prosody head emits a per-phoneme duration; those durations are summed into a cumulative index, and an alignment matrix of shape `[tokens, frames]` is scattered from it. The output length is a function of the input **values**, not the input **shapes**.

That distinction is the whole reason this system has the shape it has. A compiler can propagate `[1, 512, T] → [1, 512, 2T]`. It cannot propagate `[1, T, 50] → [1, 640, sum(round(σ(x).sum(-1)))]`, because the answer is not in the shape algebra. So the graph gets cut there, and the pieces on either side get compiled independently.

### Three seams, three different answers

There are exactly three places where the computation stops being dense tensor arithmetic, and each got a different treatment:

1. **Text → phoneme tokens.** String normalisation, espeak-ng phonemization, dictionary lookup, per-speaker pronunciation overrides. Not tensor math at all. Runs as a Triton *Python backend* (`frontend`), which also loads the speaker's 256-d style vector from disk.
2. **Durations → alignment → frames.** The value-dependent one. This *was* expressed in ONNX — `Range`, `CumSum`, a broadcast comparison and three `ScatterND`s build the alignment matrix on-GPU without a host round-trip. It stays on ONNX Runtime because TensorRT requires output shapes computable from input shapes, and this one is not.
3. **Audio → client bytes.** Squeeze, trim, resample, encode. Python backend (`last_step`) plus the FastAPI gateway.

> **The generalisable principle.** Don't ask "can I export this model?" Ask "where does shape stop being a function of shape?" Those points are your graph boundaries, and they are decided by the model's semantics, not by the exporter's limitations. Everything between two boundaries is a compilation unit and should be as large as possible; everything at a boundary needs an interpreter.

---

## 2. The ladder: five steps, sixteen months

Each stage kept the stage before it working. Numbered because the order was forced — you cannot profile a decomposition you have not built, and you cannot build engines for graphs you have not exported.

### Stage 01 — Sep 2025 — In-process fp16 and admission control

The model ran as PyTorch inside the FastAPI worker. The cheapest wins first: half precision on the weights, and a micro-batching queue in front of the endpoint that collects up to 8 requests within a time window before dispatching.

This is the stage everyone starts at, and it is worth naming what it actually does. Half precision on an L4 roughly halves weight-memory traffic and unlocks the tensor cores for the convolutional stack. The queue does not make any single request faster — it bounds how many requests can be in flight, which converts tail-latency blowup under load into a bounded wait.

- **Bought:** memory bandwidth, load stability
- **Cost:** nothing structural
- **Ceiling:** Python still in the inner loop

### Stage 02 — Oct 2025 — Decomposition into a Triton ensemble

The structural step. The model was cut at the three seams above and exported as a set of ONNX graphs wired together by a Triton `ensemble_scheduling` block. The single training-time decoder was itself split three ways — `prep_decoder` for the strided F0/energy convolutions, `decoder` for the AdaIN residual stack, `generator` for the HiFi-GAN vocoder — so each could be scheduled and quantised independently.

Decomposition is not free. Fifteen ensemble steps means fifteen scheduler hops, fifteen tensor handoffs, and no fusion across a boundary. You pay that overhead to buy the right to compile each piece with the runtime that suits it. The bet only pays if the pieces are large enough that per-hop overhead is noise — which is why the split is at semantic joints, not arbitrary layer counts.

- **Bought:** per-graph runtime choice
- **Cost:** 15 scheduler hops per request
- **Unlocked:** TensorRT, later

### Stage 03 — Dec 2025 — Style diffusion moved onto Triton

The style sampler is a loop: a Karras noise schedule, then *n* ancestral DPM-Solver steps, each calling a style transformer twice. Loops are a compiler's problem child. The choice made here was to **unroll the loop at trace time** — the Python `for` becomes *n* copies of the denoiser in the graph.

The upside is total: no control flow, full fusion across steps, and TensorRT sees one straight-line graph it can tune end to end. The downside is equally total: **the step count is now part of the graph's structure.** Changing it is a re-export, not a different input value. This is the single most consequential trade in the whole pipeline, and section 7 returns to what it broke.

- **Bought:** full fusion, no control flow
- **Cost:** step count frozen into the engine

### Stage 04 — Jan 2026 — TensorRT engines and shape profiles

Eight of the twelve exported graphs were compiled to TensorRT engines with `trtexec --fp16` and explicit `min`/`opt`/`max` shape profiles. Warmup sentences landed in the same commit, and that is not a coincidence — the two are the same idea seen from opposite ends.

The four graphs left on ONNX Runtime were left there deliberately: one has value-dependent output shape, one is dominated by an LSTM that TensorRT handles poorly, and two are small enough that a compiled engine would buy less than the extra handoff costs.

- **Bought:** kernel autotuning, layer fusion
- **Cost:** a hard shape ceiling, minutes of build time

### Stage 05 — Feb → Jun 2026 — Making it survive production

Nothing in this stage makes a single inference faster. All of it makes the *system* faster or cheaper: a shared EFS volume so four replicas build engines once rather than four times, a pre-recorded audio cache that answers common utterances without touching the GPU, a warmup sidecar, Prometheus high-water-mark concurrency metrics, and load balancing by outstanding request count rather than round-robin.

That last one matters more than it looks. TTS request duration varies by an order of magnitude with text length. Round-robin across replicas assumes uniform cost and will pile short requests behind long ones; `least_outstanding_requests` does not.

- **Bought:** cold-start amortisation, GPU-free cache hits
- **Cost:** a stateful shared volume

---

## 3. Topology: the sixteen-model pipeline

One request, fifteen scheduled steps.

```
 TEXT        frontend [PY]
             └─> TOKENS · TEXT_MASK · REF_S[1,256] · SPEED

 ENCODE ∥    text_encoder [ORT] ──────────────> T_EN[1,512,t]
             bert [TRT] ─> bert_encoder [ORT] ─> D_EN[1,512,t] fp16

 STYLE       diffusion_steps [TRT] ─> S_PRED[1,256]
             └─> int_steps_1 [ORT] ─> REF · S_OUT        (α/β blend)

 DURATION    prosody_text_encoder [TRT]
             └─> prosody_lstm [TRT]
                 └─> prosody_dur_proj [TRT] ─> DURATION[1,t,50] fp16

 ALIGN ⚑     int_steps_2 [ORT] ─> EN[1,640,A] · ASR[1,512,A]
             ^^^ tokens → frames, output length depends on VALUES

 PROSODY     prosody_predictor_ftrain [TRT] ─> F0_PRED · N_PRED [1,2A]
             └─> prep_decoder [ORT] ─────────> F0_OUT · N_OUT [1,1,A]

 VOCODE      decoder [TRT] ─> X_OUT[1,512,2A]
             └─> generator [TRT] ─> AUDIO[1,1,600A]

 EMIT        last_step [PY] ─> AUDIO[600A] @ 24 kHz
```

`t` = phoneme tokens, `A` = alignment frames. One alignment frame is 600 output samples, or exactly **25 ms of audio** — the decoder doubles the frame rate and the vocoder's upsample rates `[10,5,3,2]` multiply by 300.

### Why each graph sits on the runtime it does

| Model | Runtime | Character of the work | Reason for the assignment |
|---|---|---|---|
| `frontend` | Python | Strings, dictionaries, file I/O | Not tensor math. Also caches style vectors in-process. |
| `bert` | **TensorRT** | 12-layer ALBERT, attention-bound | Dense, shape-static, the largest single graph. Prime fusion target. |
| `text_encoder` | ORT | Conv stack + BiLSTM with packing | `pack_padded_sequence` and recurrence; TRT gains little and risks a lot. |
| `bert_encoder` | ORT | One Linear 768→512 | Too small to repay an engine handoff. Emits fp16 to pin the next boundary. |
| `diffusion_steps` | **TensorRT** | *n* unrolled transformer denoise steps | Unrolling made it straight-line; that is exactly what TRT wants. |
| `int_steps_1` | ORT | Two lerps and a mask | Elementwise, microseconds. Engine overhead would dominate. |
| `prosody_text_encoder` | **TensorRT** | Stacked BiLSTM + AdaLayerNorm | Deep enough that fusion across the norm/LSTM alternation pays. |
| `prosody_lstm` | **TensorRT** | Single BiLSTM | fp32 weights, fp16 output — narrows the tensor crossing the next hop. |
| `prosody_dur_proj` | **TensorRT** | Linear 512→50 | fp16 in, fp16 out; rides the same engine context as its neighbours. |
| `int_steps_2` ⚑ | ORT | Cumsum → scatter → matmul | **Output length depends on input values.** TRT cannot build a profile for it. |
| `prosody_predictor_ftrain` | **TensorRT** | Shared LSTM + two AdaIN stacks | Two parallel residual branches; heavy, static, fuses well. |
| `prep_decoder` | ORT | Two strided 1-D convolutions | Split out so the decoder engine has a clean static profile. Tiny. |
| `decoder` | **TensorRT** | 4 AdaIN residual blocks, 1024 ch | The second-heaviest graph. Large channel counts, ideal for tensor cores. |
| `generator` | **TensorRT** | HiFi-GAN, 300× upsampling | The heaviest by output volume — one frame becomes 300 samples. |
| `last_step` | Python | Squeeze to 1-D float32 | Keeps the ensemble's output shape independent of the vocoder's. |

---

## 4. Numerics: where precision is spent

Two different mechanisms are at work, and conflating them is a common source of confusion.

**Inside an engine**, `trtexec --fp16` does not force half precision. It *permits* it: TensorRT autotunes layer by layer, timing fp16 and fp32 kernels and keeping whichever wins, subject to its own accuracy heuristics. You are granting a search space, not making a numerical decision.

**At a boundary**, the ONNX graph's declared dtypes are a hard contract, because Triton copies the tensor between models. Three boundaries are pinned to fp16 on purpose: `D_EN` leaving `bert_encoder`, `X_OUT` leaving `prosody_lstm`, and `DURATION` leaving `prosody_dur_proj`. Each halves the bytes moved across a scheduler hop.

The pattern in the recurrent parts is deliberate and worth copying: **compute in fp32, hand off in fp16.** An LSTM accumulates error along the time axis — each step's output feeds the next, so a half-precision rounding error at *t* is still present at *t+200*. Convolutions and attention have no such accumulation path. So `prosody_lstm` keeps fp32 weights and casts only its result, while the convolutional decoder and vocoder are let loose in fp16 throughout.

> One consequence is invisible unless you look for it. `D_EN` arrives at `prosody_text_encoder` as fp16, but the first operation there concatenates it with an fp32 style vector — which silently promotes it straight back to fp32. The fp16 boundary saves bandwidth on the hop and buys nothing computationally. That is a defensible trade, but it should be a known one.

---

## 5. Shape budget: profiles are a promise about the future

A TensorRT engine is built for a shape *range*, declared as three points. `min` and `max` are correctness bounds — outside them the engine refuses to run. `opt` is the tuning target: TensorRT times its kernel candidates at that shape and picks winners there. Shapes far from `opt` still work, just on kernels chosen for someone else.

Because every graph in the chain has its own profile, the profiles have to agree about the same utterance. Reading them back through the frame arithmetic — one alignment frame is 25 ms — tells you what the deployed fleet can actually say:

| Engine | Profile axis | min | opt | max | max as audio |
|---|---|---:|---:|---:|---|
| `bert` | TOKENS | 1 | 128 | 512 | 512 phonemes |
| `prosody_text_encoder` | D_EN length | 4 | 128 | 512 | 512 phonemes |
| `prosody_lstm` | D_OUT length | 1 | 128 | 512 | 512 phonemes |
| `prosody_dur_proj` | X_OUT length | 4 | 192 | 1024 | over-provisioned |
| `prosody_predictor_ftrain` | EN frames | 1 | 192 | 768 | **19.2 s** |
| `decoder` | ASR frames | 4 | 256 | 1024 | 25.6 s |
| `generator` | X_OUT frames | 4 | 512 | 1536 | **19.2 s** |

The fleet's real ceiling is **19.2 seconds of audio per request**, set jointly by `prosody_predictor_ftrain` and `generator`. The `decoder` engine is built for 25.6 s it can never be handed, and `prosody_dur_proj` for 1024 tokens where `bert` stops at 512 — both are wasted engine capacity rather than bugs, but they are the signature of profiles set per-graph rather than derived from one utterance budget.

The `opt` points disagree more interestingly. `decoder` and `generator` are both tuned for a 6.4-second utterance; `prosody_predictor_ftrain` is tuned for 4.8 seconds. They are describing different sentences. Nothing fails — but two of the three heavy engines are running kernels selected for a length the third never produces.

### Why warmup sentences exist

The warmup list in the service is forty-odd Spanish sentences, and the striking thing about it is the distribution: production-length utterances of 200+ characters sitting next to four-character fragments like `'¿Con'` and `'Sin'`.

That is the shape profile seen from the serving side. An engine's kernel choice is baked at build time, but the *first* execution at a given shape still pays for context memory allocation, workspace binding and library-level heuristic caching. Warming only with long sentences leaves the short-utterance path cold — and short utterances are exactly what a conversational agent emits when it backchannels. The list is a deliberate sweep of the profile range so that every bucket a real caller will hit has been hit once already, at startup, by nobody.

---

## 6. Build vs. run: engines cannot be shipped

A `.plan` file is not portable. It is tied to the GPU's compute capability, the TensorRT version and the CUDA version that produced it. An engine built on an L4 will not load on an A10, and one built against TensorRT 8.6 will not load on 10.x. This single fact dictates the entire deployment shape.

The consequence is that engine construction is a **runtime** step, not a build step. The container image ships ONNX; the download script explicitly refuses to fetch `.plan` files from S3 even if they are there. Engines are compiled on first boot on the machine that will run them.

Compiling eight engines takes minutes, and there are four replicas sharing one EFS volume. The entrypoint therefore implements a small distributed build protocol:

```bash
# per engine, on the shared volume
flock -x                              # acquire exclusive lock on <engine>.lock
  re-check if [ -f "$ENGINE_PATH" ]   # a peer may have finished while we waited
  trtexec --saveEngine=$ENGINE.tmp    # build to a temp name
  mv $ENGINE.tmp $ENGINE              # atomic rename: never a partial .plan
flock -u
```

Three properties are being bought here, and each maps to a failure that would otherwise happen:

- The **double-check inside the lock** stops four replicas from building the same engine four times.
- The **temp-file-then-rename** stops a crashed or OOM-killed build from leaving a truncated `.plan` that every subsequent boot would treat as valid.
- The **persistent EFS volume**, with `CLEANUP_ENABLED=false` and `FORCE_REBUILD=false` in production, means a pod restart or a scale-up event costs a download, not a rebuild.

> The trade being made is image portability against cold-start time. Because the fleet is pinned to L4 by node selector, the engines *could* have been baked into the image — turning a multi-minute first boot into a longer CI build. The current choice keeps the image architecture-agnostic and pushes the cost to a one-time event amortised across the volume's lifetime. Both are defensible; the one that is not is building engines in CI and shipping them to heterogeneous nodes.

---

## 7. Honest accounting: knobs connected to nothing

Reading the pipeline end to end turns up several controls that are accepted, transported and then discarded. None of them produce an error; each simply has no effect. They are grouped here because they share a root cause — a graph boundary silently dropped something, and nothing in the system checks that a declared contract matches the graph behind it.

### `NUM_STEPS` — diffusion step count

The client sends it, the ensemble routes it, `config.pbtxt` declares it as a shape tensor, and the `trtexec` profile reserves a range of 1 to 10 for it. But the sampler loop is unrolled in Python during tracing, so the exporter folds the tensor away and the graph never had such an input. The value is transported across the whole stack and read by nothing.

Downstream, the service maps its user-facing `expressive_level` parameter onto this input. **That parameter therefore has no audible effect.**

### `EMBEDDING_SCALE` — classifier-free guidance

Accepted by the ensemble and passed to the `frontend`, which does not forward it; the diffusion graph never consumed it either. Real CFG would require running the style transformer twice per denoise step — conditional and masked — and blending, roughly doubling diffusion cost. That work was never done, but the parameter was left in the interface.

### The hifigan frame shift

The reference PyTorch path applies a one-frame shift to the expanded features before vocoding when the decoder is `hifigan`. The traced `int_steps_2` graph guards the same shift on a flag compared against a string it is never set to, so the shift never runs. Triton and the PyTorch reference therefore produce measurably different audio for the same input — which makes A/B validation against the reference misleading.

### fp16 overflows the vocoder's oscillator phase

Not a dead knob — a live numerical failure, found by running the exported
pipeline end to end on Triton. HiFi-GAN's source module accumulates an unbounded
oscillator phase, then scales it by the upsample factor before interpolating:

```
phase = cumsum(f0 / 24000) * 2*pi                  # grows with utterance length
phase = interpolate(phase * upsample_scale, ...)   # upsample_scale = 300
sines = sin(phase)
```

`harmonic_num` is 8, so the eighth harmonic accumulates eight times faster than
the fundamental. On a real Greek utterance at peak F0 308 Hz that product
reaches 143,113 — past **65504**, the largest finite fp16 value. The tensor
becomes `inf`, `sin(inf)` is `NaN`, and the whole waveform returns as NaN.

Three properties make this worse than a normal precision bug:

- **Silent.** No error at any layer. Triton returns HTTP 200 and a correctly
  shaped float array that happens to be entirely NaN.
- **Persistent.** Once it fires, later requests on the same engine context also
  return NaN until Triton is restarted. One bad utterance takes the replica out.
- **Data-dependent.** It is not a length cutoff. The threshold is reached sooner
  at higher F0, so a short utterance from a high-pitched voice triggers it just
  as a long one does — which is why length-based testing can miss it entirely.

StyleTTS2's `SineGen` carries a phase-wrapping guard for precisely this,
commented out upstream with the note *"To prevent torch.cumsum numerical
overflow"*. Until it is restored, or the source module's layers are pinned with
`--precisionConstraints=obey --layerPrecisions=...`, the generator has to be
built in fp32. That costs roughly 2-3x on the vocoder stage, which is the single
heaviest engine — so it is worth fixing at the source rather than paying for.

### The micro-batcher does not batch

The queue collects up to 8 requests, then issues them through `asyncio.gather` as 8 independent batch-of-1 Triton calls. That is a concurrency limiter, not batched inference — and it could not be otherwise, because every engine profile in the fleet is built for batch 1.

Two implementation details compound it: the queue lock is held across the entire in-flight batch, so no new request can enqueue while one is running; and the loop's wait interval is computed as `int(100/1000)`, which truncates to zero, turning the worker into a busy spin on a pod limited to a single CPU core.

---

The generalisable lesson is that **an ensemble's declared interface and its graphs' real interfaces drift silently apart**. Triton validates a model's config against its own engine at load time, but nothing validates that an ensemble input reaches a consumer, or that a tensor a client sends is read by anybody. That check is cheap to write and belongs in CI.

---

## 8. Where the remaining gains are

Ordered by expected return against effort, given the topology as it stands.

1. **Restore SineGen's phase wrapping.** The generator is the heaviest engine in
   the pipeline and is currently stuck in fp32 because fp16 overflows its
   oscillator phase. Fixing the accumulation — or pinning just those layers —
   buys back fp16 on the one graph where it matters most. Nothing else on this
   list is worth more per line changed.
2. **Collapse the prosody chain.** `prosody_text_encoder → prosody_lstm → prosody_dur_proj` is three scheduler hops and two tensor copies for what is one computation with no branch between them. Exporting them as a single graph removes two hops per request with no numerical change. This is the cheapest structural win left.
3. **Fix the admission path before adding capacity.** The lock-held-across-inference and the zero-length sleep are both a few lines, and on a single-core pod the spin is not free. Neither requires touching a model.
4. **Decide what `expressive_level` means.** Either re-export `diffusion_steps` as an ONNX `Loop` so the step count becomes a real runtime input, or remove the parameter from the public interface. Shipping a knob that does nothing is worse than shipping neither.
5. **Derive the shape profiles from one utterance budget.** Pick a maximum supported duration, propagate it through the frame arithmetic, and generate every `min`/`opt`/`max` from it. That removes the current over-provisioning and the mismatched `opt` points, and makes the ceiling a stated property rather than an emergent one.
6. **Then consider true batching.** Real batch inference needs `max_batch_size > 0`, engines rebuilt across a batch axis, and — the hard part — a padding strategy for the alignment step, whose output length differs per request in the batch. This is the largest available win and also the largest amount of work; it should not be started before the four items above are done, because each of them changes the number it would be measured against.

> **A note on measurement.** Every item above should be evaluated against the same fixed set of utterances spanning the profile range — the warmup list is already close to being that set. Optimisation work on a pipeline with this many stages is very easy to misattribute; a change that helps 6-second utterances and hurts 1-second ones will look like a win on any benchmark that only uses paragraphs.

---

*Sources: `VoicingAI/PR-TTS-Service` @ `client/telmex-tts` (entrypoint.sh, app/, helm-charts3/) and `VoicingAI/styletts2_triton_model_registry` (export pipeline, ensemble template). Frame arithmetic derived from the vocoder upsample rates and verified against the deployed shape profiles. Timeline dates taken from commit history, Feb 2025 – Jun 2026.*
