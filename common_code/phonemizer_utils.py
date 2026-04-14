"""Utility helpers for language-specific phonemization."""

from typing import Dict, Iterable

from common_code.config import get_supported_languages

try:  # pragma: no cover - preferred API when available
    from phonemizer.backend import EspeakBackend  # type: ignore
except Exception:  # pragma: no cover - fall back to functional API
    from phonemizer import phonemize as _phonemize  # type: ignore

    class EspeakBackend:  # type: ignore
        """Minimal shim that mirrors the backend interface we need."""

        def __init__(self, language: str, preserve_punctuation: bool = True, with_stress: bool = True):
            self.language = language
            self._preserve_punctuation = preserve_punctuation
            self._with_stress = with_stress

        def phonemize(self, texts):
            return _phonemize(
                texts,
                language=self.language,
                preserve_punctuation=self._preserve_punctuation,
                with_stress=self._with_stress,
            )


_PHONEMIZER_LANG_OVERRIDES = {
    "en": "en-us",
    "fr": "fr-fr",
    "en-uk": "en-gb",
}


class PhonemizerManager:
    """Initialize and cache phonemizer backends for supported languages."""

    def __init__(self, languages: Iterable[str] = None):
        self._languages = set(languages) if languages else set(get_supported_languages())
        self.phonemizer_cache: Dict[str, EspeakBackend] = {}
        self._initialize()

    def _resolve_backend_lang(self, lang: str) -> str:
        return _PHONEMIZER_LANG_OVERRIDES.get(lang, lang)

    def _initialize(self) -> None:
        for lang in sorted(self._languages):
            backend_lang = self._resolve_backend_lang(lang)
            try:
                backend = EspeakBackend(
                    language=backend_lang,
                    preserve_punctuation=True,
                    with_stress=True,
                )
                self.phonemizer_cache[lang] = backend
            except Exception as exc:  # pragma: no cover - logging path only
                print(f"Warning: Unable to initialize phonemizer for language '{lang}': {exc}")

    def get_backend(self, lang: str):
        return self.phonemizer_cache.get(lang)

    def phonemize(self, lang: str, text: str) -> str:
        backend = self.get_backend(lang)
        if backend is None:
            raise ValueError(f"No phonemizer backend available for language '{lang}'.")
        result = backend.phonemize([text])
        return result[0] if result else ""
