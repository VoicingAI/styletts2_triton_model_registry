"""Static JSON configuration: speakers, languages and PLBERT variants.

Loaded once at import. The files live in ``config/json_files/`` and are resolved
relative to *this file*, not the working directory — the Triton python backends
import this module from inside a model repository, where the cwd is Triton's.

  speaker_configs.json    per-speaker object-store keys for model/config/audio
  speaker_languages.json  language -> {speaker: model_id}
  language_codes.json     language -> {espeak: <espeak-ng voice>}
  language_plbert.json    PLBERT variant -> {plbert: <path>, languages: [...]}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

__all__ = [
    "CONFIG_BASE_DIR",
    "SPEAKER_S3_CONFIG",
    "SPEAKER_LANGS_CONFIG",
    "LANGUAGE_CODES_CONFIG",
    "LANGUAGE_PLBERT_CONFIG",
    "SPEAKER_MODEL_MAP",
    "get_supported_languages",
    "get_espeak_voice",
    "get_plbert_group_for_language",
    "get_plbert_relative_path",
    "get_plbert_dir",
]

# common_code/config.py -> common_code/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_BASE_DIR = Path(
    os.getenv("STYLETTS2_CONFIG_DIR", _REPO_ROOT / "config" / "json_files")
)


def load_json_config(file_path: str | os.PathLike, *, required: bool = True) -> dict:
    """Read a JSON config.

    Missing or malformed files raise by default. The original swallowed both
    and returned ``{}``, which turned a typo'd path into an empty speaker map
    and a confusing failure several layers away.
    """
    path = Path(file_path)
    try:
        with path.open() as handle:
            return json.load(handle)
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(
                f"config file not found: {path}\n"
                f"Expected it under {CONFIG_BASE_DIR}. Set STYLETTS2_CONFIG_DIR "
                f"to point somewhere else."
            ) from None
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


SPEAKER_S3_CONFIG: Dict = load_json_config(CONFIG_BASE_DIR / "speaker_configs.json")
SPEAKER_LANGS_CONFIG: Dict = load_json_config(CONFIG_BASE_DIR / "speaker_languages.json")
LANGUAGE_CODES_CONFIG: Dict = load_json_config(CONFIG_BASE_DIR / "language_codes.json")
LANGUAGE_PLBERT_CONFIG: Dict = load_json_config(CONFIG_BASE_DIR / "language_plbert.json")

# (speaker, language) -> model_id
SPEAKER_MODEL_MAP: Dict[Tuple[str, str], str] = {
    (speaker, language): model_id
    for language, speakers in SPEAKER_LANGS_CONFIG.items()
    for speaker, model_id in speakers.items()
}


def get_supported_languages() -> Iterable[str]:
    return LANGUAGE_CODES_CONFIG.keys()


def get_espeak_voice(language: str) -> str:
    """espeak-ng voice name for a language code (falls back to the code itself)."""
    entry = LANGUAGE_CODES_CONFIG.get((language or "").lower(), {})
    return entry.get("espeak") or language


def _resolve_plbert_entry(language: str) -> Tuple[str, Dict]:
    """Return ``(group name, group config)`` for a language.

    A language may either name its own PLBERT directly or belong to a grouped
    one (``multi``, ``multi_indic``). Falls back to ``multi``.
    """
    language = (language or "").lower()

    entry = LANGUAGE_PLBERT_CONFIG.get(language, {})
    if entry.get("plbert") and "languages" not in entry:
        return language, entry

    for group, config in LANGUAGE_PLBERT_CONFIG.items():
        if language in config.get("languages", []):
            return group, config

    if entry.get("plbert"):
        return language, entry

    return "multi", LANGUAGE_PLBERT_CONFIG.get("multi", {})


def get_plbert_group_for_language(language: str) -> str:
    return _resolve_plbert_entry(language)[0]


def get_plbert_relative_path(language: str) -> str:
    """Store-relative path to the PLBERT for a language, e.g. ``PLBERTs/multi``.

    This is what goes in an export config's ``plbert.path``.
    """
    group, entry = _resolve_plbert_entry(language)
    path = (entry.get("plbert") or "").strip("/")
    if not path:
        raise KeyError(
            f"no PLBERT configured for language '{language}' (resolved to group "
            f"'{group}'). Add it to {CONFIG_BASE_DIR / 'language_plbert.json'}."
        )
    return path


def get_plbert_dir(base_dir: str | os.PathLike, language: str) -> str:
    """Locate a language's PLBERT directory beneath *base_dir*.

    A directory qualifies only if it contains ``config.yml``. Tries the
    configured path first, then the bare group name — the original tried 100+
    case and prefix permutations, which turned a wrong ``base_dir`` into an
    unreadable error listing every one of them.
    """
    base = Path(base_dir)
    group = get_plbert_group_for_language(language)
    relative = get_plbert_relative_path(language)

    candidates = [base / relative, base / group, base / "PLBERTs" / group, base / "PLBERT" / group]

    for candidate in candidates:
        if (candidate / "config.yml").is_file():
            return str(candidate)

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no PLBERT for language '{language}' (group '{group}') under {base}.\n"
        f"Looked for a directory containing config.yml at:\n  {tried}"
    )
