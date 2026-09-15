"""Per-speaker reference style vectors.

A speaker's identity is a 256-d vector: the acoustic style encoder's 128-d
output concatenated with the prosodic encoder's. It is derived once from a few
seconds of reference audio and then shipped alongside the ONNX graphs — the
Triton frontend loads the ``.pt`` and feeds it in as ``REF_S``.

Vectors are cached by the reference WAV's content hash, so re-running an export
never recomputes a style for audio that has not changed, even if the file moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import librosa
import numpy as np
import torch
import torchaudio

from .assets import file_digest

__all__ = [
    "STYLE_FILENAME",
    "STYLE_FILENAME_NPY",
    "MelConfig",
    "compute_reference_style",
    "style_path_for",
]

# Triton's frontend looks for these names; keep the three in step.
#
# The .npy is what the frontend actually loads. It exists so the serving
# container needs numpy and nothing else -- loading a 256-float vector from a
# .pt would drag all of PyTorch into the Python backend process. The .pt is
# kept for the torch-side reference path and for older deployments.
STYLE_FILENAME = "{speaker}.pt"
STYLE_FILENAME_NPY = "{speaker}.npy"

_SAMPLE_RATE = 24_000
_TRIM_TOP_DB = 30


class MelConfig:
    """Mel front-end used to train StyleTTS2. These values are not tunable —
    they must match the training-time transform or the style vector is garbage.
    """

    n_mels = 80
    n_fft = 2048
    win_length = 1200
    hop_length = 300
    mean = -4.0
    std = 4.0


def _mel_transform(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        n_mels=MelConfig.n_mels,
        n_fft=MelConfig.n_fft,
        win_length=MelConfig.win_length,
        hop_length=MelConfig.hop_length,
    ).to(device)


def style_path_for(style_dir: str | Path, speaker: str) -> Path:
    return Path(style_dir) / STYLE_FILENAME.format(speaker=speaker)


def _write_style(style: torch.Tensor, style_dir: Path, speaker: str) -> Path:
    """Write both representations; returns the .pt path."""
    output_path = style_path_for(style_dir, speaker)
    torch.save(style, str(output_path))
    np.save(
        str(style_dir / STYLE_FILENAME_NPY.format(speaker=speaker)),
        style.numpy().astype(np.float32),
    )
    return output_path


def _load_mel(audio_path: str | Path, device: torch.device) -> torch.Tensor:
    # librosa.load resamples to `sr` itself, so the audio is already at 24 kHz
    # here — the original code's post-hoc `if sr != 24000: resample` was dead.
    wave, _ = librosa.load(str(audio_path), sr=_SAMPLE_RATE)
    if wave.size == 0:
        raise ValueError(f"reference audio is empty: {audio_path}")

    trimmed, _ = librosa.effects.trim(wave, top_db=_TRIM_TOP_DB)
    if trimmed.size < MelConfig.n_fft:
        # Trimming can eat a very short or very quiet clip entirely.
        trimmed = wave

    wave_tensor = torch.from_numpy(np.ascontiguousarray(trimmed)).float().to(device)
    mel = _mel_transform(device)(wave_tensor.unsqueeze(0))
    return (torch.log(1e-5 + mel) - MelConfig.mean) / MelConfig.std


def compute_reference_style(
    model: Mapping[str, torch.nn.Module],
    audio_path: str | Path,
    speaker: str,
    style_dir: str | Path,
    cache_dir: str | Path = "style_cache",
    *,
    overwrite: bool = False,
) -> torch.Tensor:
    """Return the 256-d style vector for *speaker*, writing it to *style_dir*.

    Cached under ``cache_dir/<wav content hash>.pt``; the copy in *style_dir*
    is what gets deployed.
    """
    style_dir = Path(style_dir)
    cache_dir = Path(cache_dir)
    style_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = style_path_for(style_dir, speaker)
    cache_path = cache_dir / f"{file_digest(audio_path)}.pt"

    if cache_path.is_file() and not overwrite:
        style = torch.load(str(cache_path), map_location="cpu", weights_only=True)
        _write_style(style, style_dir, speaker)
        print(f"  cached  {speaker:<20} -> {output_path}  {tuple(style.shape)}")
        return style

    device = next(model["style_encoder"].parameters()).device
    mel = _load_mel(audio_path, device)

    with torch.no_grad():
        mel_batched = mel.unsqueeze(1)                      # [1, 1, n_mels, T]
        acoustic = model["style_encoder"](mel_batched)      # [1, 128]
        prosodic = model["predictor_encoder"](mel_batched)  # [1, 128]

    style = torch.cat([acoustic, prosodic], dim=1).cpu()

    torch.save(style, str(cache_path))
    _write_style(style, style_dir, speaker)
    print(f"  built   {speaker:<20} -> {output_path}  {tuple(style.shape)}")
    return style
