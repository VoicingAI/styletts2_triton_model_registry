#!/usr/bin/env python3
"""Entry point for the StyleTTS2 -> ONNX -> Triton exporter.

    python convert.py configs/greek_1_5.yaml
    python convert.py --list-modules

The implementation lives in the `styletts2_export` package; this file exists so
the tool can be run from a clone without installing anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from styletts2_export.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
