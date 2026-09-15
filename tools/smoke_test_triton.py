#!/usr/bin/env python3
"""Send a request through the styletts2_ensemble and save the audio.

Speaks to Triton's HTTP endpoint directly so it needs no tritonclient install.

    python tools/smoke_test_triton.py --speaker melina --language el \
        --text "Καλημέρα, πώς είστε σήμερα;" --out /tmp/melina.wav

Reports per-model latency from Triton's own statistics endpoint, which is the
only way to see where the time goes inside an ensemble.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

SAMPLE_RATE = 24_000


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def wait_ready(base: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/v2/health/ready", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def model_states(base: str) -> List[dict]:
    try:
        return _post(f"{base}/v2/repository/index", {})
    except Exception as exc:
        print(f"  could not read repository index: {exc}", file=sys.stderr)
        return []


def infer(base: str, model: str, text: str, speaker: str,
          language: str, speed: float) -> Dict:
    payload = {
        "inputs": [
            {"name": "TEXT", "shape": [1, 1], "datatype": "BYTES", "data": [text]},
            {"name": "SPEAKER", "shape": [1, 1], "datatype": "BYTES", "data": [speaker]},
            {"name": "LANGUAGE", "shape": [1, 1], "datatype": "BYTES", "data": [language]},
            {"name": "SPEED", "shape": [1, 1], "datatype": "FP32", "data": [speed]},
        ],
        "outputs": [{"name": "AUDIO"}],
    }
    return _post(f"{base}/v2/models/{model}/infer", payload)


def per_model_latency(base: str) -> List[tuple]:
    """(model, executions, mean ms) from Triton's own counters."""
    try:
        stats = _get(f"{base}/v2/models/stats")
    except Exception:
        return []
    rows = []
    for entry in stats.get("model_stats", []):
        inference = entry.get("inference_stats", {})
        success = inference.get("success", {})
        count = int(success.get("count", 0))
        if not count:
            continue
        total_ns = int(success.get("ns", 0))
        rows.append((entry["name"], count, total_ns / count / 1e6))
    return sorted(rows, key=lambda r: -r[2])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="styletts2_ensemble")
    p.add_argument("--text", default="Καλημέρα, πώς είστε σήμερα;")
    p.add_argument("--speaker", default="melina")
    p.add_argument("--language", default="el")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default="/tmp/styletts2_smoke.wav")
    p.add_argument("--wait", type=int, default=0, help="seconds to wait for readiness")
    p.add_argument("--repeat", type=int, default=1, help="requests to send (first is a warmup)")
    p.add_argument("--stats", action="store_true", help="print per-model latency afterwards")
    args = p.parse_args()

    base = args.url.rstrip("/")

    if args.wait and not wait_ready(base, args.wait):
        print(f"Triton did not become ready within {args.wait}s", file=sys.stderr)
        for m in model_states(base):
            if m.get("state") != "READY":
                print(f"  {m.get('name')}: {m.get('state')} {m.get('reason','')}", file=sys.stderr)
        return 1

    ready = [m for m in model_states(base) if m.get("state") == "READY"]
    print(f"models READY: {len(ready)}")

    audio: Optional[List[float]] = None
    timings = []
    for i in range(max(1, args.repeat)):
        started = time.perf_counter()
        try:
            result = infer(base, args.model, args.text, args.speaker,
                           args.language, args.speed)
        except urllib.error.HTTPError as exc:
            print(f"\ninference failed ({exc.code}):\n{exc.read().decode()[:2000]}",
                  file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - started
        timings.append(elapsed)

        output = next(o for o in result["outputs"] if o["name"] == "AUDIO")
        audio = output["data"]
        seconds = len(audio) / SAMPLE_RATE
        label = "warmup " if i == 0 and args.repeat > 1 else "request"
        print(f"  {label} {i+1}: {elapsed*1000:7.1f} ms  ->  {len(audio):>7,} samples "
              f"({seconds:.2f} s audio, RTF {elapsed/max(seconds,1e-9):.3f})")

    if audio is None:
        return 1

    try:
        import numpy as np
        import soundfile as sf
        wav = np.asarray(audio, dtype=np.float32)
        sf.write(args.out, wav, SAMPLE_RATE)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        print(f"\nwrote {args.out}  ({wav.size/SAMPLE_RATE:.2f} s, peak {peak:.3f})")
        if peak < 1e-4:
            print("  WARNING: output is silent", file=sys.stderr)
    except ImportError:
        print("\n(soundfile/numpy not installed — audio not written)", file=sys.stderr)

    if args.repeat > 1:
        steady = timings[1:]
        print(f"steady-state: min {min(steady)*1000:.1f} ms  "
              f"median {sorted(steady)[len(steady)//2]*1000:.1f} ms  "
              f"max {max(steady)*1000:.1f} ms")

    if args.stats:
        rows = per_model_latency(base)
        if rows:
            print("\nper-model mean latency (Triton counters):")
            for name, count, mean_ms in rows:
                print(f"  {name:<28} {count:>4} exec  {mean_ms:7.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
