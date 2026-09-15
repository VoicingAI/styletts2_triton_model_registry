#!/usr/bin/env python3
"""Export a StyleTTS2 checkpoint to ONNX and lay out a Triton model repository.

    python convert.py configs/examples/local.yaml
    python convert.py configs/examples/multilingual.yaml --speaker amelie --overwrite
    python convert.py --list-modules

Everything about a run — where the checkpoint lives, which speakers to build
styles for, which graphs to emit — comes from the YAML file. Flags only
override; see README.md for the field reference.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import ALL_MODULES, ConfigError, ExportConfig, load_config

__all__ = ["main", "run"]

_REPO_ROOT = Path(__file__).resolve().parent


def _seed_everything(seed: int = 0) -> None:
    """Make exports reproducible.

    The diffusion graph still contains a RandomNormal node — its *output* is
    stochastic by design — but tracing, weight init and style computation are
    all pinned.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("high")


def _banner(title: str) -> None:
    print(f"\n{title}\n{'─' * max(len(title), 60)}")


def run(config: ExportConfig) -> int:
    """Execute one export. Returns a process exit code."""
    # Imported lazily so `--list-modules` and config validation work without
    # torch/onnx fully set up.
    from .assets import AssetError, AssetResolver
    from .checkpoint import CheckpointError, build_and_load
    from .onnx.pipeline import run_exports
    from .styles import compute_reference_style, style_path_for
    from .triton import copy_template, write_registry

    _banner(f"StyleTTS2 export — {config.model_id}")
    print(f"config      {config.source_path}")
    print(f"speakers    {', '.join(s.name for s in config.speakers)}")
    print(f"modules     {', '.join(config.export.modules)}")
    print(f"output      {config.repo_dir}")

    resolver = AssetResolver(cache_root=config.cache_root)

    _banner("1. Assets")
    checkpoint_path = resolver.resolve_file(config.checkpoint, "models", filename=None)
    model_config_path = resolver.resolve_file(config.model_config, "configs")
    plbert_dir = resolver.resolve_dir(config.plbert, "plbert")
    references: Dict[str, Path] = {
        speaker.name: resolver.resolve_file(speaker.reference, "references")
        for speaker in config.speakers
    }

    _banner("2. Model")
    model, model_params = build_and_load(
        model_config_path,
        checkpoint_path,
        plbert_dir,
        device=config.export.device,
    )

    _banner("3. ONNX export")
    outcomes = run_exports(
        model,
        config.repo_dir,
        module_names=config.export.modules,
        opset=config.export.opset,
        overwrite=config.export.overwrite,
        verify=config.export.verify,
        strict=config.export.strict,
        context={
            "model_params": model_params,
            "style": config.style,
            "device": config.export.device,
            "diffusion_num_steps": config.export.diffusion_num_steps,
        },
    )

    _banner("4. Reference styles")
    style_paths: Dict[str, Path] = {}
    for speaker in config.speakers:
        compute_reference_style(
            model,
            references[speaker.name],
            speaker.name,
            config.style_dir,
            cache_dir=config.style.cache_dir,
            overwrite=config.export.overwrite,
        )
        style_paths[speaker.name] = style_path_for(config.style_dir, speaker.name)

    _banner("5. Triton repository")
    template = config.triton.template
    if template is not None and not Path(template).is_absolute():
        # Templates are given relative to the repo, not the caller's cwd.
        candidate = _REPO_ROOT / template
        template = str(candidate if candidate.exists() else Path(template))
    copy_template(
        template,
        config.repo_dir,
        exported=[o.name for o in outcomes if o.exported or o.skipped],
    )
    write_registry(config, outcomes, style_paths)

    _banner("Summary")
    failed = [o for o in outcomes if not o.ok]
    exported = [o for o in outcomes if o.exported]
    print(f"  model_id     {config.model_id}")
    print(f"  exported     {len(exported)} / {len(outcomes)} module(s)")
    print(f"  styles       {len(style_paths)} speaker(s) -> {config.style_dir}")
    print(f"  triton repo  {config.repo_dir}")
    if failed:
        print(f"\n  {len(failed)} module(s) FAILED:")
        for outcome in failed:
            print(f"    {outcome.name}: {outcome.error}")
        return 1
    print("\n  all good ✓")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", nargs="?", help="path to an export config YAML (see configs/)"
    )
    parser.add_argument(
        "--speaker", action="append", metavar="NAME",
        help="only build styles for this speaker (repeatable); "
             "ONNX graphs are shared across a model's speakers either way",
    )
    parser.add_argument(
        "--modules", metavar="LIST",
        help=f"comma-separated subset to export. Choices: {','.join(ALL_MODULES)}",
    )
    parser.add_argument(
        "--device", metavar="DEV", help="override export.device (e.g. cpu, cuda:0)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-export graphs and recompute styles that already exist",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip the onnxruntime-vs-PyTorch check (faster, much less safe)",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="log export failures and continue instead of stopping",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the config and print the plan without downloading anything",
    )
    parser.add_argument(
        "--list-modules", action="store_true", help="print exportable module names and exit",
    )
    return parser


def _apply_overrides(config: ExportConfig, args: argparse.Namespace) -> ExportConfig:
    export = config.export
    if args.modules:
        names = tuple(n.strip() for n in args.modules.split(",") if n.strip())
        unknown = [n for n in names if n not in ALL_MODULES]
        if unknown:
            raise ConfigError(
                f"--modules names unknown module(s) {unknown}; "
                f"valid: {list(ALL_MODULES)}"
            )
        export = replace(export, modules=names)
    if args.device:
        export = replace(export, device=args.device)
    if args.overwrite:
        export = replace(export, overwrite=True)
    if args.no_verify:
        export = replace(export, verify=False)
    if args.keep_going:
        export = replace(export, strict=False)

    config = replace(config, export=export)
    if args.speaker:
        config = config.with_speakers(args.speaker)
    return config


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_modules:
        from .onnx.pipeline import EXPORTERS
        print("Exportable modules (in dependency order):\n")
        for spec in EXPORTERS:
            print(f"  {spec.name:<26} {spec.notes}")
        return 0

    if not args.config:
        parser.error("a config file is required (or pass --list-modules)")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # credentials can still come from the real environment

    try:
        config = _apply_overrides(load_config(args.config), args)
    except (ConfigError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _banner(f"Plan — {config.model_id}  (dry run, nothing fetched)")
        print(f"  checkpoint   {config.checkpoint}")
        print(f"  model config {config.model_config}")
        print(f"  plbert       {config.plbert}")
        print(f"  device       {config.export.device}   opset {config.export.opset}")
        print(f"  modules      {', '.join(config.export.modules)}")
        print(f"  repo dir     {config.repo_dir}")
        print(f"  style dir    {config.style_dir}")
        print("  speakers:")
        for speaker in config.speakers:
            langs = f"  [{', '.join(speaker.languages)}]" if speaker.languages else ""
            print(f"    {speaker.name:<20} {speaker.reference}{langs}")
        return 0

    _seed_everything()
    try:
        return run(config)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nexport failed: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
