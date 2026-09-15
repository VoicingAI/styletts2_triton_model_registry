"""Triton python backend: text -> phoneme tokens, plus the speaker's style.

Runs first in the ensemble. Everything downstream is a fixed-shape tensor
graph; all the string handling, phonemization and per-speaker asset loading
happens here.

Deliberately numpy-only: style vectors are loaded from .npy, and the padding
mask is a comparison. Importing torch here would pull a multi-gigabyte,
CUDA-linked dependency into every Python backend process to read 256 floats.
A .pt file is still accepted, but only if torch happens to be installed.

Configuration comes from `parameters` in config.pbtxt:
  REFERENCE_STYLES_DIR  directory of <speaker>.npy style vectors
                        (default /workspace/reference_styles)
  DEFAULT_LANGUAGE      language used when a request omits one (default "en")
"""

import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from nltk.tokenize import word_tokenize

# common_code is a package, so its *parent* has to be importable — inserting
# the package directory itself makes `import common_code.x` fail.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.abspath(os.path.join(_THIS_DIR, "../../..")),      # repo checked out next to the model repo
    os.path.abspath(os.path.join(_THIS_DIR, "../../../..")),   # model repo nested one deeper
    "/workspace",
):
    if os.path.isdir(os.path.join(_candidate, "common_code")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

from common_code.config import SPEAKER_MODEL_MAP, get_supported_languages
from common_code.phonemizer_utils import PhonemizerManager
from common_code.text_utils import TextCleaner

DEFAULT_STYLES_DIR = "/workspace/reference_styles"

# Token id 16 is the phoneme-sequence boundary marker in the symbol table.
BOUNDARY_TOKEN = 16

# These voices were trained with an explicit trailing boundary token; the rest
# were not, and adding one changes their prosody at the end of an utterance.
SPEAKERS_NEEDING_TRAILING_BOUNDARY = frozenset({"chloe", "dora", "regina", "isha"})


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameters = config.get("parameters", {})

        def param(key, default):
            entry = parameters.get(key)
            value = entry.get("string_value") if isinstance(entry, dict) else entry
            return value or default

        self.styles_dir = param("REFERENCE_STYLES_DIR", DEFAULT_STYLES_DIR)
        self.default_language = param("DEFAULT_LANGUAGE", "en")

        self.cleaner = TextCleaner()
        self.phoneme_cleanup = re.compile(r"\([^)]*\)")
        self.reference_styles: Dict[str, np.ndarray] = {}

        languages = {lang for _, lang in SPEAKER_MODEL_MAP} or set(get_supported_languages())
        if not languages:
            languages = {self.default_language}
        self.phonemizer_manager = PhonemizerManager(languages=languages)

        # Fail at load time rather than on the first request.
        if not os.path.isdir(self.styles_dir):
            raise pb_utils.TritonModelException(
                f"REFERENCE_STYLES_DIR '{self.styles_dir}' does not exist. Mount "
                f"the export's reference_styles/ directory there, or set the "
                f"parameter in config.pbtxt."
            )

    # ── assets ───────────────────────────────────────────────────────────
    def _load_reference_style(self, speaker: str) -> np.ndarray:
        """Return the speaker's [1, 256] style vector, cached after first use."""
        if speaker in self.reference_styles:
            return self.reference_styles[speaker]

        # <speaker>.npy is what convert.py writes and the only format that
        # loads without torch. The .pt names are accepted so an existing
        # deployment keeps working mid-migration.
        candidates = [
            f"{speaker}.npy",
            f"{speaker}.pt",
            f"{speaker}_style.pt",
            f"reference_style_{speaker}.pt",
        ]
        for candidate in candidates:
            path = os.path.join(self.styles_dir, candidate)
            if os.path.isfile(path):
                break
        else:
            available = sorted(
                f for f in os.listdir(self.styles_dir) if f.endswith((".npy", ".pt"))
            )
            raise ValueError(
                f"no style vector for speaker '{speaker}' in {self.styles_dir} "
                f"(looked for {candidates}); available: {available}"
            )

        if path.endswith(".npy"):
            array = np.load(path).astype(np.float32)
        else:
            try:
                import torch
            except ImportError as exc:
                raise ValueError(
                    f"only a .pt style vector exists for '{speaker}' and torch is not "
                    f"installed. Re-run the exporter to produce {speaker}.npy, which "
                    f"loads without torch."
                ) from exc
            array = (
                torch.load(path, map_location="cpu", weights_only=True)
                .detach().float().numpy().astype(np.float32)
            )

        if array.ndim == 1:
            array = array[None, :]
        if array.shape[-1] != 256:
            raise ValueError(
                f"style vector for '{speaker}' has {array.shape[-1]} dims, expected 256"
            )

        self.reference_styles[speaker] = array
        return array

    # ── text ─────────────────────────────────────────────────────────────
    def _text_to_tokens(self, text: str, language: str, speaker: str) -> List[int]:
        backend = (
            self.phonemizer_manager.get_backend(language)
            or self.phonemizer_manager.get_backend(self.default_language)
        )
        if backend is None:
            raise ValueError(
                f"no phonemizer backend for language '{language}' or fallback "
                f"'{self.default_language}'"
            )

        phonemes = backend.phonemize([text.strip()])[0]
        try:
            phonemes = " ".join(word_tokenize(phonemes))
        except LookupError:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            phonemes = " ".join(word_tokenize(phonemes))
        phonemes = self.phoneme_cleanup.sub("", phonemes)

        token_ids = self.cleaner(phonemes)
        if not token_ids:
            raise ValueError(f"text produced no usable phonemes: {text!r}")

        # Leading pad, then the boundary marker.
        token_ids = [0, BOUNDARY_TOKEN] + token_ids
        if speaker in SPEAKERS_NEEDING_TRAILING_BOUNDARY and token_ids[-1] != BOUNDARY_TOKEN:
            token_ids.append(BOUNDARY_TOKEN)
        return token_ids

    @staticmethod
    def _length_to_mask(lengths: np.ndarray) -> np.ndarray:
        """True at padded positions, shape [batch, max(lengths)]."""
        positions = np.arange(int(lengths.max()), dtype=np.int64)[None, :]
        return positions + 1 > lengths[:, None]

    # ── request handling ─────────────────────────────────────────────────
    @staticmethod
    def _scalar_string(request, name: str, default=None):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return tensor.as_numpy()[0][0].decode("utf-8")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                text = self._scalar_string(request, "TEXT")
                speaker = self._scalar_string(request, "SPEAKER")
                if not text or not speaker:
                    raise ValueError("TEXT and SPEAKER are required and must be non-empty")

                language = (self._scalar_string(request, "LANGUAGE") or self.default_language).lower()

                speed_tensor = pb_utils.get_input_tensor_by_name(request, "SPEED")
                speed = speed_tensor.as_numpy().astype(np.float32) if speed_tensor is not None \
                    else np.ones((1, 1), dtype=np.float32)

                ref_s = self._load_reference_style(speaker)
                token_ids = self._text_to_tokens(text, language, speaker)

                tokens = np.asarray(token_ids, dtype=np.int64)[None, :]
                input_lengths = np.array([tokens.shape[-1]], dtype=np.int64)
                text_mask = self._length_to_mask(input_lengths)
                attention_mask = (~text_mask).astype(np.int64)

                # Style-blend weights. Baked in here because int_steps_1 takes
                # them as graph inputs; expose them as ensemble inputs if they
                # ever need to be per-request.
                alpha = np.array([0.3], dtype=np.float32)
                beta = np.array([0.7], dtype=np.float32)

                responses.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("ALPHA", alpha),
                    pb_utils.Tensor("BETA", beta),
                    pb_utils.Tensor("TOKENS", tokens),
                    pb_utils.Tensor("INPUT_LENGTHS", input_lengths),
                    pb_utils.Tensor("TEXT_MASK", text_mask),
                    pb_utils.Tensor("ATTENTION_MASK", attention_mask),
                    pb_utils.Tensor("REF_S", ref_s),
                    pb_utils.Tensor("SPEED_OUT", speed),
                ]))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"frontend failed: {exc}"),
                ))

        return responses

    def finalize(self):
        pass
