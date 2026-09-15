#!/usr/bin/env python3
"""Build TensorRT engines for an exported Triton repository.

Every engine's shape profile is derived from a single utterance budget rather
than set per-graph, because the graphs are links in one chain and their
profiles have to describe the same sentence. Setting them independently is how
a repository ends up with a decoder built for 25.6 s feeding a generator that
accepts 19.2 s.

The chain, for an utterance of `t` phoneme tokens and `A` alignment frames:

    bert                      TOKENS    [1, t]
    prosody_text_encoder      D_EN      [1, 512, t]
    prosody_lstm              D_OUT     [1, t, 640]
    prosody_dur_proj          X_OUT     [1, t, 512]
    ---- int_steps_2 expands tokens to frames (value-dependent) ----
    prosody_predictor_ftrain  EN        [1, 640, A]      -> F0/N [1, 2A]
    decoder                   ASR       [1, 512, A]      -> X_OUT [1, 512, 2A]
    generator                 X_OUT     [1, 512, 2A]     -> audio [1, 1, 600A]

One alignment frame is 600 output samples at 24 kHz, i.e. exactly 25 ms, so
the audio budget in seconds fixes A and every profile downstream of it.

    python tools/build_trt_engines.py triton_repos/<id>/triton_model_repository --print
    python tools/build_trt_engines.py triton_repos/<id>/triton_model_repository --run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 600          # decoder 2x upsample * vocoder 300x
SECONDS_PER_FRAME = SAMPLES_PER_FRAME / SAMPLE_RATE      # 0.025

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# The generator is built in fp32 by default. Its HiFi-GAN source module
# (SineGen) accumulates an unbounded oscillator phase:
#
#     phase = cumsum(f0 / 24000) * 2*pi          # grows with utterance length
#     phase = interpolate(phase * upsample_scale, ...)   # upsample_scale = 300
#     sines = sin(phase)
#
# with harmonic_num = 8, so the 8th harmonic accumulates eight times faster.
# For a few hundred alignment frames of ordinary speech that product passes
# 65504 -- the largest finite fp16 value -- so the tensor becomes inf and
# sin(inf) is NaN. The whole waveform then comes back as NaN, and the failure
# persists for later requests on the same engine context until Triton restarts.
#
# StyleTTS2's SineGen has a phase-wrapping guard for precisely this, commented
# out upstream ("To prevent torch.cumsum numerical overflow"). Until that is
# restored or the source module's layers are pinned with
# --precisionConstraints=obey --layerPrecisions=..., fp32 is the only safe
# setting for this engine. It is worth roughly 2-3x on the generator stage.
#
# The threshold is data-dependent, not a clean length cutoff: it is reached
# sooner for higher F0, so a short utterance from a high-pitched voice can
# trigger it just as a long one can.
FP32_ONLY_BY_DEFAULT = ("generator",)


def frames(seconds: float) -> int:
    return max(1, int(round(seconds / SECONDS_PER_FRAME)))


def profiles(min_tok: int, opt_tok: int, max_tok: int,
             min_a: int, opt_a: int, max_a: int) -> Dict[str, Dict[str, str]]:
    """Shape strings for every TensorRT model, keyed by model then min/opt/max."""
    def spec(**axes) -> Dict[str, str]:
        return {k: ",".join(f"{n}:{d}" for n, d in v.items()) for k, v in axes.items()}

    return {
        "bert": spec(
            min={"TOKENS": f"1x{min_tok}", "ATTENTION_MASK": f"1x{min_tok}"},
            opt={"TOKENS": f"1x{opt_tok}", "ATTENTION_MASK": f"1x{opt_tok}"},
            max={"TOKENS": f"1x{max_tok}", "ATTENTION_MASK": f"1x{max_tok}"},
        ),
        "prosody_text_encoder": spec(
            min={"D_EN": f"1x512x{min_tok}", "S_OUT": "1x128", "TEXT_MASK_PRED": f"1x{min_tok}"},
            opt={"D_EN": f"1x512x{opt_tok}", "S_OUT": "1x128", "TEXT_MASK_PRED": f"1x{opt_tok}"},
            max={"D_EN": f"1x512x{max_tok}", "S_OUT": "1x128", "TEXT_MASK_PRED": f"1x{max_tok}"},
        ),
        "prosody_lstm": spec(
            min={"D_OUT": f"1x{min_tok}x640"},
            opt={"D_OUT": f"1x{opt_tok}x640"},
            max={"D_OUT": f"1x{max_tok}x640"},
        ),
        "prosody_dur_proj": spec(
            min={"X_OUT": f"1x{min_tok}x512"},
            opt={"X_OUT": f"1x{opt_tok}x512"},
            max={"X_OUT": f"1x{max_tok}x512"},
        ),
        "diffusion_steps": spec(
            min={"BERT_DUR": f"1x{min_tok}x768", "REF_S": "1x256"},
            opt={"BERT_DUR": f"1x{opt_tok}x768", "REF_S": "1x256"},
            max={"BERT_DUR": f"1x{max_tok}x768", "REF_S": "1x256"},
        ),
        "prosody_predictor_ftrain": spec(
            min={"EN": f"1x640x{min_a}", "S_OUT": "1x128"},
            opt={"EN": f"1x640x{opt_a}", "S_OUT": "1x128"},
            max={"EN": f"1x640x{max_a}", "S_OUT": "1x128"},
        ),
        "decoder": spec(
            min={"ASR": f"1x512x{min_a}", "F0_OUT": f"1x1x{min_a}",
                 "N_OUT": f"1x1x{min_a}", "REF": "1x128"},
            opt={"ASR": f"1x512x{opt_a}", "F0_OUT": f"1x1x{opt_a}",
                 "N_OUT": f"1x1x{opt_a}", "REF": "1x128"},
            max={"ASR": f"1x512x{max_a}", "F0_OUT": f"1x1x{max_a}",
                 "N_OUT": f"1x1x{max_a}", "REF": "1x128"},
        ),
        "generator": spec(
            min={"X_OUT": f"1x512x{2*min_a}", "REF": "1x128", "F0_PRED": f"1x{2*min_a}"},
            opt={"X_OUT": f"1x512x{2*opt_a}", "REF": "1x128", "F0_PRED": f"1x{2*opt_a}"},
            max={"X_OUT": f"1x512x{2*max_a}", "REF": "1x128", "F0_PRED": f"1x{2*max_a}"},
        ),
    }


def command(repo: Path, model: str, shapes: Dict[str, str], *, fp16: bool,
            workspace_mb: int) -> List[str]:
    onnx = repo / model / "1" / "model.onnx"
    plan = repo / model / "1" / "model.plan"
    cmd = [
        TRTEXEC,
        f"--onnx={onnx}",
        f"--saveEngine={plan}",
        f"--minShapes={shapes['min']}",
        f"--optShapes={shapes['opt']}",
        f"--maxShapes={shapes['max']}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", help="Triton model repository containing the exported .onnx files")
    p.add_argument("--max-seconds", type=float, default=19.2,
                   help="longest utterance the fleet must synthesize (default: 19.2)")
    p.add_argument("--opt-seconds", type=float, default=6.4,
                   help="utterance length to tune kernels for (default: 6.4)")
    p.add_argument("--min-seconds", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="longest phoneme sequence (default: 512)")
    p.add_argument("--opt-tokens", type=int, default=128)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--no-fp16", action="store_true", help="build every engine in fp32")
    p.add_argument("--generator-fp16", action="store_true",
                   help="allow fp16 in the generator. Produces NaN audio on many "
                        "utterances -- see FP32_ONLY_BY_DEFAULT above.")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--only", help="comma-separated subset of models to build")
    p.add_argument("--run", action="store_true", help="execute the builds")
    p.add_argument("--print", dest="show", action="store_true",
                   help="print the trtexec commands and exit (default)")
    args = p.parse_args()

    repo = Path(args.repo)
    if not repo.is_dir():
        print(f"not a directory: {repo}", file=sys.stderr)
        return 2

    min_a, opt_a, max_a = (frames(s) for s in
                           (args.min_seconds, args.opt_seconds, args.max_seconds))
    table = profiles(args.min_tokens, args.opt_tokens, args.max_tokens, min_a, opt_a, max_a)

    if args.only:
        wanted = {m.strip() for m in args.only.split(",") if m.strip()}
        unknown = wanted - set(table)
        if unknown:
            print(f"unknown model(s) {sorted(unknown)}; known: {sorted(table)}", file=sys.stderr)
            return 2
        table = {k: v for k, v in table.items() if k in wanted}

    print(f"# utterance budget: {args.min_seconds}s / {args.opt_seconds}s / {args.max_seconds}s"
          f"  ->  {min_a} / {opt_a} / {max_a} alignment frames", file=sys.stderr)
    print(f"# phoneme budget:   {args.min_tokens} / {args.opt_tokens} / {args.max_tokens} tokens",
          file=sys.stderr)
    forced = [m for m in FP32_ONLY_BY_DEFAULT
              if m in table and not args.generator_fp16 and not args.no_fp16]
    print(f"# precision:        {'fp16 permitted' if not args.no_fp16 else 'fp32 only'}"
          + (f", except {forced} (fp32: fp16 overflows SineGen's phase)" if forced else ""),
          file=sys.stderr)
    print(file=sys.stderr)

    failures = []
    for model, shapes in table.items():
        onnx = repo / model / "1" / "model.onnx"
        if not onnx.is_file():
            print(f"# SKIP {model}: no {onnx}", file=sys.stderr)
            continue

        fp16 = not args.no_fp16
        if model in FP32_ONLY_BY_DEFAULT and not args.generator_fp16:
            fp16 = False
        cmd = command(repo, model, shapes, fp16=fp16, workspace_mb=args.workspace_mb)
        if not args.run:
            print(" \\\n  ".join(shlex.quote(c) if " " in c else c for c in cmd))
            print()
            continue

        plan = repo / model / "1" / "model.plan"
        tmp = plan.with_suffix(".plan.tmp")
        precision = "fp16" if fp16 else "fp32"
        print(f"[trt] building {model} ({precision}) ...", file=sys.stderr, flush=True)
        cmd = [c.replace(str(plan), str(tmp)) for c in cmd]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-12:])
            print(f"[trt] FAILED {model}\n{tail}\n", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            failures.append(model)
            continue
        tmp.replace(plan)                       # atomic: never a partial .plan
        size_mb = plan.stat().st_size / 1e6
        print(f"[trt] built {model}  ({size_mb:.0f} MB)", file=sys.stderr, flush=True)

    if failures:
        print(f"\n[trt] {len(failures)} engine(s) failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
