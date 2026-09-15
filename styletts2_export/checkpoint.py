"""Building a StyleTTS2 model and loading a training checkpoint into it.

The export graph splits the monolithic training-time decoder into three Triton
models — ``prep_decoder`` (the strided F0/energy convs), ``decoder`` (the AdaIN
residual stack) and ``generator`` (the HiFi-GAN vocoder) — so a checkpoint's
single ``decoder`` state dict has to be routed to three modules. That routing,
and verifying it actually landed, is what this module is for.

The original script loaded the decoder and generator with ``strict=False`` and
never inspected the result, so a checkpoint whose keys did not match produced a
randomly-initialised vocoder and no error at all. :func:`load_checkpoint` fails
loudly instead.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
import yaml
from munch import Munch

from common_code.styletts2.Utils.PLBERT.util import load_plbert
from .models import build_model

__all__ = [
    "CheckpointError",
    "recursive_munch",
    "load_model_params",
    "load_plbert_from_dir",
    "build_and_load",
    "load_checkpoint",
]

# Keys inside the checkpoint's `decoder` state dict that belong to prep_decoder.
_PREP_DECODER_PREFIXES = ("F0_conv.", "N_conv.")

# Sub-models that get their weights from the checkpoint's `decoder` entry
# rather than from a same-named entry.
_DECODER_DERIVED = ("prep_decoder", "decoder", "generator")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be mapped onto the built model."""


def recursive_munch(value):
    """Turn nested dicts into attribute-accessible ``Munch`` objects."""
    if isinstance(value, Mapping):
        return Munch({k: recursive_munch(v) for k, v in value.items()})
    if isinstance(value, list):
        return [recursive_munch(v) for v in value]
    return value


def load_model_params(config_path: str | Path):
    """Read ``model_params`` out of a StyleTTS2 training config."""
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or "model_params" not in config:
        raise CheckpointError(f"{config_path} has no 'model_params' section")
    return recursive_munch(config["model_params"])


def load_plbert_from_dir(plbert_dir: str | Path):
    """Load PLBERT, with a readable error when the directory is wrong."""
    directory = Path(plbert_dir)
    if not directory.is_dir():
        raise CheckpointError(f"PLBERT directory does not exist: {directory}")
    if not (directory / "config.yml").is_file():
        raise CheckpointError(
            f"{directory} has no config.yml — this does not look like a PLBERT "
            f"checkpoint directory. Expected config.yml plus a step_*.t7 file."
        )
    return load_plbert(str(directory))


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Drop a ``module.`` prefix, but only from keys that actually have one.

    Checkpoints saved from a ``DistributedDataParallel`` wrapper carry the
    prefix; ones saved from a bare model do not. The original code sliced
    ``k[7:]`` unconditionally, which silently mangled every key of a non-DDP
    checkpoint.
    """
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[len("module."):] if key.startswith("module.") else key] = value
    return cleaned


def _report(name: str, result) -> Tuple[List[str], List[str]]:
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return missing, unexpected


def _load_into(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    *,
    allow_unexpected: bool = False,
) -> None:
    """Load *state_dict* into *module*, raising if anything is left unfilled.

    ``allow_unexpected`` is for the decoder/generator split, where each module
    is handed the full decoder state dict and legitimately ignores the half
    that belongs to the other one. Missing keys are never acceptable: they mean
    a parameter kept its random initialisation.
    """
    result = module.load_state_dict(state_dict, strict=False)
    missing, unexpected = _report(name, result)

    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise CheckpointError(
            f"'{name}': {len(missing)} parameter(s) were not present in the "
            f"checkpoint and kept their random initialisation: {preview}{more}.\n"
            f"This usually means the checkpoint was trained with a different "
            f"model_params (hidden_dim, n_layer, decoder.type ...) than the "
            f"config.yml paired with it."
        )
    if unexpected and not allow_unexpected:
        preview = ", ".join(unexpected[:5])
        more = f" (+{len(unexpected) - 5} more)" if len(unexpected) > 5 else ""
        raise CheckpointError(
            f"'{name}': checkpoint contains {len(unexpected)} key(s) the model "
            f"has no home for: {preview}{more}"
        )

    extra = f", {len(unexpected)} unused key(s)" if unexpected else ""
    print(f"  loaded  {name:<20} {len(state_dict)} key(s){extra}")


def _split_decoder_state(
    decoder_state: Mapping[str, torch.Tensor],
) -> Tuple["OrderedDict[str, torch.Tensor]", "OrderedDict[str, torch.Tensor]"]:
    """Split the checkpoint's decoder weights into prep-decoder and the rest."""
    prep = OrderedDict()
    rest = OrderedDict()
    for key, value in decoder_state.items():
        target = prep if key.startswith(_PREP_DECODER_PREFIXES) else rest
        target[key] = value
    if not prep:
        raise CheckpointError(
            "checkpoint's 'decoder' state dict has no F0_conv./N_conv. keys, so "
            "prep_decoder cannot be populated. Expected a hifigan-style decoder."
        )
    return prep, rest


def load_checkpoint(model: Mapping[str, torch.nn.Module], checkpoint_path: str | Path) -> None:
    """Load a StyleTTS2 ``.pth`` into a model built by :func:`build_model`."""
    path = Path(checkpoint_path)
    print(f"[checkpoint] {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    if "net" not in raw:
        raise CheckpointError(
            f"{path} has no 'net' key (found: {sorted(raw)[:10]}). Expected a "
            f"StyleTTS2 training checkpoint."
        )
    state: Dict[str, Mapping[str, torch.Tensor]] = raw["net"]

    if "decoder" not in state:
        raise CheckpointError(f"{path} has no 'decoder' weights")

    decoder_state = _strip_module_prefix(state["decoder"])
    prep_state, rest_state = _split_decoder_state(decoder_state)

    _load_into(model["prep_decoder"], prep_state, "prep_decoder")
    # decoder and generator each take their own slice of the remainder.
    _load_into(model["decoder"], rest_state, "decoder", allow_unexpected=True)
    _load_into(model["generator"], rest_state, "generator", allow_unexpected=True)

    for name in model:
        if name in _DECODER_DERIVED:
            continue
        if name not in state:
            print(f"  skip    {name:<20} (absent from checkpoint)")
            continue
        _load_into(model[name], _strip_module_prefix(state[name]), name)


def build_and_load(
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    plbert_dir: str | Path,
    *,
    device: str = "cpu",
):
    """Build the model from config, load weights, and return ``(model, params)``.

    The model is returned on *device* in eval mode. Individual exports move
    what they need and put it back (see ``onnx_export.module_as``).
    """
    model_params = load_model_params(model_config_path)
    plbert = load_plbert_from_dir(plbert_dir)

    model = build_model(model_params, plbert)
    load_checkpoint(model, checkpoint_path)

    target = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    for key in model:
        model[key].to(target).eval()
        for parameter in model[key].parameters():
            parameter.requires_grad_(False)

    print(f"[checkpoint] model ready on {target}")
    return model, model_params
