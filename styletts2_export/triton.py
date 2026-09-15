"""Assembling the Triton model repository and its registry.

An export produces two things per model: the ONNX graphs under
``<output_root>/<model_id>/triton_model_repository/`` (laid out as Triton wants,
``<model>/<version>/model.onnx``) and the reference style vectors beside them.
The ``config.pbtxt`` files come from a template directory that is copied in —
without it Triton sees a directory of ``.onnx`` files and loads nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional

__all__ = ["TritonRepoError", "copy_template", "write_registry"]

_REGISTRY_VERSION = 1


class TritonRepoError(RuntimeError):
    """Raised when the Triton repository cannot be assembled."""


def copy_template(
    template_dir: Optional[str | Path],
    repo_dir: str | Path,
    exported: Iterable[str] = (),
) -> None:
    """Copy ``config.pbtxt`` files and Python backends into the output repo.

    *exported* names the models this run produced ONNX for. Their ``.onnx``
    files are skipped so a stale graph in the template cannot shadow a freshly
    exported one; every other model's weights (``int_steps_2``, which is pure
    tensor arithmetic and identical for all speakers) are copied through.
    """
    if template_dir is None:
        print("[triton] no template configured — skipping config.pbtxt copy")
        return

    source = Path(template_dir)
    if not source.is_dir():
        raise TritonRepoError(
            f"Triton template directory not found: {source}\n"
            f"Set `triton.template` in the config to the directory holding the "
            f"config.pbtxt files (the repo ships one at "
            f"'triton_model_repoistory_with_diffusion')."
        )

    destination = Path(repo_dir)
    destination.mkdir(parents=True, exist_ok=True)

    exported = set(exported)

    def _ignore(directory, names):
        # Only shield the models this run exported; their graphs are newer.
        model = Path(directory).relative_to(source).parts
        if model and model[0] in exported:
            return [n for n in names if n.endswith(".onnx")]
        return []

    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore)

    configs = sorted(p.parent.name for p in destination.glob("*/config.pbtxt"))
    if not configs:
        raise TritonRepoError(
            f"copied '{source}' but it produced no config.pbtxt files — "
            f"the template directory looks wrong"
        )

    # An ensemble has no weights, so its version directory is empty — and git
    # cannot track an empty directory, so it never survives a clone of the
    # template. Triton still requires one ("at least one version must be
    # available under the version policy"), so create it here rather than
    # leaving every deploy script to remember.
    for config_path in sorted(destination.glob("*/config.pbtxt")):
        if "ensemble" not in config_path.read_text():
            continue
        model_dir = config_path.parent
        if not any(child.is_dir() and child.name.isdigit() for child in model_dir.iterdir()):
            (model_dir / "1").mkdir(exist_ok=True)
            print(f"[triton] created empty version dir {model_dir.name}/1 (ensemble)")

    print(f"[triton] template -> {destination}  ({len(configs)} model config(s))")


def write_registry(
    config,
    outcomes: Iterable,
    style_paths: Mapping[str, Path],
    output_path: str | Path = "model_registry.json",
) -> Path:
    """Record what this export produced, for downstream serving code to read."""
    path = Path(output_path)

    registry = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, Mapping) and existing.get("version") == _REGISTRY_VERSION:
                registry = dict(existing.get("models", {}))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[registry] ignoring unreadable {path}: {exc}")

    # A partial run (`--modules X`, `--speaker Y`) must not erase what earlier
    # runs recorded for the same model: merge module and speaker entries into
    # the existing record rather than replacing it wholesale.
    previous = registry.get(config.model_id, {})
    merged_modules = dict(previous.get("modules", {}))
    merged_speakers = dict(previous.get("speakers", {}))

    registry[config.model_id] = {
        "config": config.source_path,
        "repo_dir": str(config.repo_dir),
        "style_dir": str(config.style_dir),
        "checkpoint": str(config.checkpoint),
        "model_config": str(config.model_config),
        "plbert": str(config.plbert),
        "modules": {
            **merged_modules,
            **{
                outcome.name: {
                    "path": str(outcome.path),
                    "verified": outcome.verified,
                    "max_abs_error": outcome.max_abs_error,
                    "skipped": outcome.skipped,
                }
                for outcome in outcomes
            },
        },
        "speakers": {
            **merged_speakers,
            **{
                name: {
                    "style": str(style_path),
                    "languages": list(config.speaker(name).languages),
                    "reference": str(config.speaker(name).reference),
                }
                for name, style_path in sorted(style_paths.items())
            },
        },
    }

    path.write_text(
        json.dumps({"version": _REGISTRY_VERSION, "models": registry}, indent=2) + "\n"
    )
    print(f"[registry] {path}  ({len(registry)} model(s))")
    return path
