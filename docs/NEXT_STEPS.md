# Where this repo stands, and what to do next

Written after taking the exporter from a 1,300-line script to a config-driven
package, and running two checkpoints end to end through ONNX → TensorRT →
Triton on an L4.

- [What works now](#what-works-now)
- [Open defects, ranked](#open-defects-ranked)
- [Architecture: what to change and why](#architecture-what-to-change-and-why)
- [Code structure: what is still awkward](#code-structure-what-is-still-awkward)
- [Testing](#testing)
- [Suggested sequence](#suggested-sequence)

---

## What works now

Two checkpoints exported and served, verified end to end:

| Model | Speakers | Graphs | Engines | Triton |
|---|---|---|---|---|
| `greek_1_5` | chloe, dora (el) | 12/12 verified | 8 built | 16/16 READY |
| `Multilingual` | melina, marios (el) | 12/12 verified | 8 built | 16/16 READY |

Measured on greek_1_5, single L4, batch 1, 3 diffusion steps:

| Utterance | Audio | p50 | RTF |
|---|---:|---:|---:|
| `Ναι.` | 0.65 s | 30 ms | 0.046 |
| `Καλησπέρα σας.` | 1.40 s | 41 ms | 0.029 |
| One sentence | 4.08 s | 96 ms | 0.024 |
| Three sentences | 9.65 s | 226 ms | 0.023 |

Steady-state per-model means, greek_1_5: `generator` 37 ms, `last_step` 16 ms,
`text_encoder` 9 ms, everything else under 4 ms, ensemble total ~77 ms. **The
vocoder is over half the budget**, and it is the one engine stuck in fp32.

### Defects found and fixed

| | Was |
|---|---|
| Triton template path | Pointed at a directory that does not exist; every run produced a repo with `.onnx` files and **no `config.pbtxt`**, silently. |
| Checkpoint loading | `strict=False` with the result discarded, plus an unconditional `k[7:]` prefix strip — a non-DDP checkpoint loaded **zero decoder weights** and produced noise with no error. |
| `frontend` import path | Inserted the package directory instead of its parent; `ImportError` on load. |
| `common_code/config.py` | Read from `./config/json_files` relative to cwd, and two of the four JSON files never existed. All four loaded as `{}`, swallowed. |
| Style vector filename | Exporter wrote `reference_style_<spk>.pt`; frontend looked for `<spk>_style.pt`. |
| `build_model_onnx` | `from Modules.istftnet import Decoder` — guaranteed `ImportError`; returned a Munch with no `bert` key that callers indexed. |
| Dtype mutation | `.half()` / `.float()` applied in place to shared submodules, making exports order-dependent and non-repeatable. |
| `NUM_STEPS` | Declared on `diffusion_steps`, routed through the ensemble, reserved in the TRT profile — **never an input of the exported graph**. Triton refuses to load. |
| `EMBEDDING_SCALE` | Accepted from clients, dropped in the frontend, never consumed. |
| Ensemble version dir | Git cannot track an empty directory, so `styletts2_ensemble/1/` never survived a clone and Triton refused to load. |
| `generator` fp16 | Produced **all-NaN audio** and poisoned the engine context for later requests. Now fp32 by default. |
| `build_model` | Defined twice in one file; which one won depended on star-import ordering across four modules. |
| Registry | A partial run (`--modules X`) overwrote the model's whole record. |
| `frontend` torch dependency | Imported all of PyTorch to read 256 floats. Now numpy-only; the serving image no longer needs torch. |

---

## Open defects, ranked

### 1. Restore `SineGen`'s phase wrapping — the largest single win

The generator is 37 ms of a 77 ms budget and is the only engine forced to fp32,
because its oscillator phase overflows fp16 (see `docs/RUNBOOK.md`). Fixing the
accumulation — or pinning just those layers with
`--precisionConstraints=obey --layerPrecisions=...` — buys fp16 back on the one
graph where it matters most.

The guard already exists upstream, commented out, with a note saying what it is
for. Restoring it changes the numerics slightly, so it needs an A/B listen
against the current fp32 output before it ships.

**Nothing else on this list is worth more per line changed.**

### 2. Triton and the PyTorch reference produce different audio

`int_steps_2` guards the hifigan one-frame shift on `self.hifigan_mode ==
"hifigan"` against a flag assigned `True`, so the shift never runs. The
reference path in `common_code/inference_core.py` *does* apply it. Any A/B
validation against the reference is therefore measuring a difference that is
not the one you are looking for.

`int_steps_2` has no exporter in this repo — its source is preserved at
`tools/reference/int_steps_2_source.py`. Giving it a real exporter, and fixing
the flag, are the same piece of work.

### 3. `expressive_level` does nothing

The service maps it to `num_steps` and `embedding_scale`; neither reaches a
graph. Either make the step count a real runtime input (rewrite the sampler loop
as an ONNX `Loop`) or remove the parameter from the public API. A knob that
silently does nothing is worse than no knob.

### 4. The micro-batcher in the serving repo does not batch

Not this repo, but it bounds what this repo's engines can deliver. It issues N
independent batch-1 calls, holds its queue lock across the whole in-flight
batch, and computes its wait interval as `int(100/1000)` — which truncates to
zero and busy-spins a single-core pod.

### 5. `int_steps_2`'s `ScatterND` indices can collide

It writes `pred_dur[0]`, `pred_dur[-1]` and `pred_dur[-2]`. For a one- or
two-token sequence those indices overlap, and ONNX Runtime warns that
`ScatterND` with `reduction='none'` is only well-defined for unique indices.
Very short inputs are exactly what a conversational agent emits.

---

## Architecture: what to change and why

### Collapse the prosody chain

`prosody_text_encoder → prosody_lstm → prosody_dur_proj` is three Triton models,
three scheduler hops and two tensor copies for one computation with no branch
between them. Exporting them as a single graph removes two hops per request with
no numerical change.

A wrapper for this already exists but is unused: `StyleTTS2ProsodyBlock` in the
original `onnx_exporter_modules.py`. This is the cheapest structural win left.

### Derive the shape budget from the model, not by hand

`tools/build_trt_engines.py` now derives every profile from one utterance
budget. The remaining gap is that the budget is typed in rather than measured.
Deriving `--opt-seconds` from the actual request-length distribution (Triton
exports it) would put kernels where the traffic is.

### Make the step count a real input

The sampler loop is unrolled at trace time, so `diffusion_steps` is one
straight-line graph — which is exactly why TensorRT likes it. Rewriting it as an
ONNX `Loop` makes the step count dynamic but gives up cross-step fusion. Measure
before committing: if 3 steps is what production always uses, the current shape
is the better trade and the right fix is to delete the knob.

### Batching is the big one, and it is last for a reason

Every engine profile is batch 1. Real batching needs `max_batch_size > 0`,
engines rebuilt across a batch axis, and — the hard part — a padding strategy
for the alignment step, whose output length differs per request in the batch.

Do not start this before items 1–3 above: each of them changes the number it
would be measured against.

---

## Code structure: what is still awkward

The exporter is now a package:

```
styletts2_export/
  config.py       YAML schema, validation, defaults
  assets.py       S3 / S3-compatible / local fetching + cache
  checkpoint.py   build the model, route a checkpoint into it
  models.py       ONNX-friendly prosody stack + model factory
  styles.py       per-speaker style vectors
  triton.py       template copy + model_registry.json
  cli.py          argument parsing and the run sequence
  onnx/
    wrappers.py   nn.Modules reshaped so the exporter can trace them
    harness.py    how to export and verify ONE module
    pipeline.py   WHAT to export, in what order
```

What is still not right:

**`common_code/` is vendored upstream code with local edits mixed in.** It
carries dead code (`inference_core.StyleTTS2Synth` has never been run end to
end), commented-out alternative implementations, and files the export path never
touches (`slmadv.py`, `discriminators.py`, the ASR and JDC utilities). It should
either be pinned to an upstream commit with the local edits isolated as a small
patch set, or pruned to just what the exporter imports. Right now nobody can
tell which lines are upstream and which are ours.

**`triton_model_repoistory_with_diffusion/` is misspelled** and doubles as both
the config template and a shipped artifact (`int_steps_2/1/model.onnx`). Splitting
"config templates" from "prebuilt weights" would make `copy_template`'s ignore
rules unnecessary. Renaming it is a one-line change here and a coordinated one
in the serving repo, so it is worth doing deliberately rather than casually.

**The legacy JSON config is still the source of truth for languages.**
`config/json_files/` drives `common_code/config.py`, which the Triton frontend
imports at runtime, while `configs/*.yaml` drives the exporter. Two config
systems describing overlapping facts. The YAML should win; the JSON should be
generated from it, or the frontend should read the export's
`model_registry.json`.

**`conversion_script_latest.py` is a deprecation shim.** Delete it once nothing
calls `--speaker <name>` directly.

**No packaging.** There is no `pyproject.toml`, so the package is import-by-path
from the repo root. That is fine for a repo you clone, and worth changing the
moment anything else wants to `import styletts2_export`.

---

## Testing

There are no automated tests. Given the failure modes seen here, three levels
would pay for themselves quickly:

1. **Config unit tests.** `load_config` already rejects a dozen classes of
   malformed input with key-qualified messages. Pin that behaviour — it is pure,
   fast and needs no GPU.

2. **A synthetic-checkpoint export test.** Build a small random-weight model,
   export all twelve graphs, assert every one verifies. This runs in about two
   minutes on a GPU and would have caught the checkpoint-routing, dtype-mutation
   and dynamic-axis bugs. The fixture-building code used during this work is a
   starting point.

3. **A contract gate in CI.** `tools/check_triton_contract.py` already exits
   non-zero on a mismatch. Run it on every exported repo before publishing.

The one check that cannot be skipped: **assert the output audio is finite.** The
fp16 failure returned HTTP 200 with a correctly shaped, entirely-NaN array. A
test that only asserts "request succeeded" passes straight through it.

---

## Suggested sequence

1. Restore `SineGen`'s phase wrapping; A/B the audio; re-enable fp16 on the generator.
2. Give `int_steps_2` an exporter and fix the hifigan shift flag, so Triton and the reference path agree.
3. Add the synthetic-checkpoint export test and wire the contract check into CI.
4. Collapse the prosody chain into one graph.
5. Decide `expressive_level`: implement it or remove it.
6. Prune or pin `common_code/`.
7. Only then, look at batching.

Steps 1–3 are worth doing before anything else: 1 is the biggest latency win
available, 2 makes correctness measurable, and 3 stops the whole list from
regressing.
