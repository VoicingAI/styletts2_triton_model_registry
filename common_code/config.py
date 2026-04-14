# common_code/config.py
import json
import os
from pathlib import Path

from typing import Dict, Iterable, Tuple

def load_json_config(file_path: str) -> dict:
    """Loads a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load {file_path} - {e}")
        return {}

# Define base directory for config files
# This assumes the script is run from the project root.
CONFIG_BASE_DIR = Path("./config/json_files")

# Load all necessary JSON configurations
SPEAKER_S3_CONFIG = load_json_config(CONFIG_BASE_DIR / "speaker_configs.json")
SPEAKER_LANGS_CONFIG = load_json_config(CONFIG_BASE_DIR / "speaker_languages.json")
LANGUAGE_CODES_CONFIG = load_json_config(CONFIG_BASE_DIR / "language_codes.json")
LANGUAGE_PLBERT_CONFIG = load_json_config(CONFIG_BASE_DIR / "language_plbert.json")

# Create a master mapping for easy lookup
# Maps (speaker, lang) -> model_id
SPEAKER_MODEL_MAP = {}
for lang, speakers in SPEAKER_LANGS_CONFIG.items():
    for speaker, model_id in speakers.items():
        SPEAKER_MODEL_MAP[(speaker, lang)] = model_id


def get_supported_languages() -> Iterable[str]:
    return LANGUAGE_CODES_CONFIG.keys()


def _resolve_plbert_entry(lang: str) -> Tuple[str, Dict]:
    lang = (lang or "").lower()

    entry = LANGUAGE_PLBERT_CONFIG.get(lang, {})
    if entry and "plbert" in entry and "languages" not in entry:
        return lang, entry

    for group_key in ("multi", "multi_indic"):
        group_cfg = LANGUAGE_PLBERT_CONFIG.get(group_key, {})
        languages = group_cfg.get("languages", [])
        if lang in languages:
            return group_key, group_cfg

    if entry and "plbert" in entry:
        return lang, entry

    fallback = LANGUAGE_PLBERT_CONFIG.get("multi", {})
    return "multi", fallback


def get_plbert_group_for_language(lang: str) -> str:
    group, _ = _resolve_plbert_entry(lang)
    return group


def _case_variants(segment: str) -> Iterable[str]:
    entries = {segment, segment.lower(), segment.upper(), segment.capitalize()}
    return [s for s in entries if s]


def _expand_path_variants(parts: Iterable[str]) -> Iterable[Tuple[str, ...]]:
    variants = [tuple()]
    for part in parts:
        new_variants = []
        for existing in variants:
            for variant in _case_variants(part):
                new_variants.append(existing + (variant,))
        variants = new_variants
    return variants


def get_plbert_dir(base_dir: str, lang: str) -> str:
    group, entry = _resolve_plbert_entry(lang)
    rel_path = (entry.get("plbert") or "").strip("/")

    prefixes = [
        (),
        ("tts", "PLBERTs"),
        ("tts", "plberts"),
        ("tts", "PLBERT"),
        ("tts", "plbert"),
        ("PLBERT",),
        ("PLBERTs",),
        ("plbert",),
        ("plberts",),
    ]

    group_variants = {g for g in _case_variants(group)} if group else set()
    rel_parts = tuple(part for part in rel_path.split('/') if part)

    search_order = []
    seen = set()

    def _register(path_parts: Tuple[str, ...]):
        if not path_parts:
            return
        full_path = os.path.join(base_dir, *path_parts)
        if full_path not in seen:
            seen.add(full_path)
            search_order.append(full_path)

    if rel_parts:
        for prefix in prefixes:
            for variant in _expand_path_variants(prefix + rel_parts):
                _register(variant)

    for prefix in prefixes:
        for variant in group_variants:
            _register(tuple(prefix) + (variant,))
        if rel_parts:
            for variant in group_variants:
                _register(tuple(prefix) + rel_parts + (variant,))

    if not search_order:
        candidate = os.path.join(base_dir, rel_path or group)
        raise FileNotFoundError(f"Unable to resolve PLBERT directory for language '{lang}' (tried '{candidate}').")

    for candidate in search_order:
        if os.path.isfile(os.path.join(candidate, "config.yml")):
            return candidate

    first = search_order[0]
    raise FileNotFoundError(
        f"Unable to locate PLBERT assets for language '{lang}'. Looked in: {', '.join(search_order)}."
    )
