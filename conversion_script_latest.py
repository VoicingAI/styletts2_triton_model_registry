#!/usr/bin/env python3
"""Deprecated entry point, kept so `--speaker <name>` keeps working.

The export is now driven by a YAML config instead of the speaker JSON files:

    python convert.py configs/generated/<model_id>.yaml

This shim looks the speaker up in config/json_files/speaker_languages.json,
finds the generated config for the model it belongs to, and hands off. It will
be removed once callers have moved over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SPEAKER_LANGUAGES = REPO_ROOT / "config" / "json_files" / "speaker_languages.json"
GENERATED_CONFIGS = REPO_ROOT / "configs" / "generated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args, passthrough = parser.parse_known_args()

    print(
        "conversion_script_latest.py is deprecated; use convert.py with a "
        "config from configs/. Continuing.\n",
        file=sys.stderr,
    )

    if not SPEAKER_LANGUAGES.is_file():
        print(f"error: {SPEAKER_LANGUAGES} not found", file=sys.stderr)
        return 2

    by_speaker = {
        speaker: model_id
        for speakers in json.loads(SPEAKER_LANGUAGES.read_text()).values()
        for speaker, model_id in speakers.items()
    }
    if args.speaker not in by_speaker:
        print(
            f"error: speaker '{args.speaker}' is not in {SPEAKER_LANGUAGES.name}",
            file=sys.stderr,
        )
        return 2

    model_id = by_speaker[args.speaker]
    config_path = GENERATED_CONFIGS / f"{model_id}.yaml"
    if not config_path.is_file():
        print(
            f"error: no config for model '{model_id}' at {config_path}\n"
            f"Generate them with: python tools/generate_configs.py",
            file=sys.stderr,
        )
        return 2

    argv = [str(config_path), "--speaker", args.speaker, *passthrough]
    if args.overwrite:
        argv.append("--overwrite")

    print(f"-> python convert.py {' '.join(argv)}\n", file=sys.stderr)

    from styletts2_export.cli import main as convert_main
    return convert_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
