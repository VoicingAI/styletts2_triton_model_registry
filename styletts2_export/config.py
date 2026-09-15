"""Declarative configuration for a StyleTTS2 -> ONNX -> Triton export.

One YAML file fully describes one export run: where the checkpoint, PLBERT
weights and per-speaker reference audio live, which ONNX modules to emit, and
where to write the resulting Triton model repository.

See ``configs/examples/`` for annotated samples and the README for the full
field reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

__all__ = [
    "ConfigError",
    "StoreConfig",
    "AssetRef",
    "SpeakerConfig",
    "ExportOptions",
    "TritonOptions",
    "StyleOptions",
    "ExportConfig",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a config file is malformed. Message names the offending key."""


# ─────────────────────────────────────────────────────────────────────────────
#  Small parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{where}' must be a mapping, got {type(value).__name__}")
    return dict(value)


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{where}' must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, where)


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(
            f"unknown key(s) {unknown} in '{where}'; expected one of {sorted(allowed)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Stores
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StoreConfig:
    """Where a group of assets lives.

    ``type: s3`` covers AWS S3 and every S3-compatible store — Backblaze B2,
    MinIO, Cloudflare R2 — by pointing ``endpoint_url`` at the right host.
    ``type: local`` reads straight off the filesystem.
    """

    name: str
    type: str                                  # "s3" | "local"
    bucket: Optional[str] = None               # s3 only
    endpoint_url: Optional[str] = None         # s3 only; omit for AWS
    region: Optional[str] = None               # s3 only
    access_key_env: str = "AWS_ACCESS_KEY"     # s3 only
    secret_key_env: str = "AWS_SECRET_KEY"     # s3 only
    root: Optional[str] = None                 # local only; paths resolve under it

    _ALLOWED = (
        "type", "bucket", "endpoint_url", "region",
        "access_key_env", "secret_key_env", "root",
    )

    @classmethod
    def parse(cls, name: str, raw: Any) -> "StoreConfig":
        data = _require_mapping(raw, f"stores.{name}")
        _reject_unknown(data, cls._ALLOWED, f"stores.{name}")

        store_type = _require_str(data.get("type"), f"stores.{name}.type").lower()
        if store_type not in ("s3", "local"):
            raise ConfigError(
                f"stores.{name}.type must be 's3' or 'local', got '{store_type}'"
            )

        if store_type == "s3":
            if not data.get("bucket"):
                raise ConfigError(f"stores.{name}.bucket is required for type 's3'")
            if data.get("root"):
                raise ConfigError(f"stores.{name}.root is only valid for type 'local'")
        else:
            for s3_only in ("bucket", "endpoint_url", "region"):
                if data.get(s3_only):
                    raise ConfigError(
                        f"stores.{name}.{s3_only} is only valid for type 's3'"
                    )

        return cls(
            name=name,
            type=store_type,
            bucket=_optional_str(data.get("bucket"), f"stores.{name}.bucket"),
            endpoint_url=_optional_str(data.get("endpoint_url"), f"stores.{name}.endpoint_url"),
            region=_optional_str(data.get("region"), f"stores.{name}.region"),
            access_key_env=_optional_str(data.get("access_key_env"), f"stores.{name}.access_key_env")
            or "AWS_ACCESS_KEY",
            secret_key_env=_optional_str(data.get("secret_key_env"), f"stores.{name}.secret_key_env")
            or "AWS_SECRET_KEY",
            root=_optional_str(data.get("root"), f"stores.{name}.root"),
        )


# The implicit store used when a config never declares ``stores:`` and every
# path is a plain filesystem path.
LOCAL_STORE = StoreConfig(name="local", type="local")


@dataclass(frozen=True)
class AssetRef:
    """One file (or directory) to fetch: a store plus a path within it."""

    store: StoreConfig
    path: str

    def __str__(self) -> str:
        if self.store.type == "local":
            return self.path
        return f"{self.store.name}://{self.store.bucket}/{self.path}"


# ─────────────────────────────────────────────────────────────────────────────
#  Speakers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpeakerConfig:
    name: str
    reference: AssetRef
    # Informational, recorded in the registry. A list because a multilingual
    # voice is served for several languages from one reference recording.
    languages: tuple = ()

    @property
    def language(self) -> Optional[str]:
        """First language, for display. Prefer :attr:`languages`."""
        return self.languages[0] if self.languages else None


# ─────────────────────────────────────────────────────────────────────────────
#  Option blocks
# ─────────────────────────────────────────────────────────────────────────────

# Export module names, in dependency order. ``convert.py --list-modules``
# prints this list.
ALL_MODULES: tuple = (
    "int_steps_1",
    "bert",
    "text_encoder",
    "bert_encoder",
    "prosody_text_encoder",
    "prosody_lstm",
    "prosody_dur_proj",
    "prosody_predictor_ftrain",
    "diffusion_steps",
    "prep_decoder",
    "decoder",
    "generator",
)


@dataclass(frozen=True)
class ExportOptions:
    device: str = "cuda:0"
    opset: int = 17
    modules: tuple = ALL_MODULES
    overwrite: bool = False
    verify: bool = True
    # Fail the run on the first export or verification error instead of
    # logging it and carrying on. The original script always carried on,
    # which is how half-built repos got shipped.
    strict: bool = True
    diffusion_num_steps: int = 3      # traced into diffusion_steps.onnx

    _ALLOWED = (
        "device", "opset", "modules", "overwrite",
        "verify", "strict", "diffusion_num_steps",
    )

    @classmethod
    def parse(cls, raw: Any) -> "ExportOptions":
        data = _require_mapping(raw, "export")
        _reject_unknown(data, cls._ALLOWED, "export")
        defaults = cls()

        modules_raw = data.get("modules", "all")
        if isinstance(modules_raw, str):
            if modules_raw.strip().lower() != "all":
                raise ConfigError(
                    "export.modules must be the string 'all' or a list of module names"
                )
            modules = ALL_MODULES
        elif isinstance(modules_raw, Sequence):
            modules = tuple(str(m) for m in modules_raw)
            unknown = [m for m in modules if m not in ALL_MODULES]
            if unknown:
                raise ConfigError(
                    f"export.modules contains unknown module(s) {unknown}; "
                    f"valid names: {list(ALL_MODULES)}"
                )
            if not modules:
                raise ConfigError("export.modules is empty; omit it or use 'all'")
        else:
            raise ConfigError("export.modules must be 'all' or a list of module names")

        opset = data.get("opset", defaults.opset)
        if not isinstance(opset, int) or opset < 11:
            raise ConfigError("export.opset must be an integer >= 11")

        steps = data.get("diffusion_num_steps", defaults.diffusion_num_steps)
        if not isinstance(steps, int) or steps < 2:
            # The Karras schedule divides by (num_steps - 1).
            raise ConfigError("export.diffusion_num_steps must be an integer >= 2")

        return cls(
            device=_optional_str(data.get("device"), "export.device") or defaults.device,
            opset=opset,
            modules=modules,
            overwrite=bool(data.get("overwrite", defaults.overwrite)),
            verify=bool(data.get("verify", defaults.verify)),
            strict=bool(data.get("strict", defaults.strict)),
            diffusion_num_steps=steps,
        )


@dataclass(frozen=True)
class TritonOptions:
    template: Optional[str] = "triton_model_repoistory_with_diffusion"
    output_root: str = "triton_repos"

    _ALLOWED = ("template", "output_root")

    @classmethod
    def parse(cls, raw: Any) -> "TritonOptions":
        data = _require_mapping(raw, "triton")
        _reject_unknown(data, cls._ALLOWED, "triton")
        defaults = cls()

        template = data.get("template", defaults.template)
        if template is not None:
            template = _require_str(template, "triton.template")

        return cls(
            template=template,
            output_root=_optional_str(data.get("output_root"), "triton.output_root")
            or defaults.output_root,
        )


@dataclass(frozen=True)
class StyleOptions:
    """Blend weights baked into ``int_steps_1``.

    ``alpha`` mixes the diffusion-predicted acoustic style with the reference
    speaker's; ``beta`` does the same for the prosodic half.
    """

    alpha: float = 0.3
    beta: float = 0.7
    cache_dir: str = "style_cache"

    _ALLOWED = ("alpha", "beta", "cache_dir")

    @classmethod
    def parse(cls, raw: Any) -> "StyleOptions":
        data = _require_mapping(raw, "style")
        _reject_unknown(data, cls._ALLOWED, "style")
        defaults = cls()

        def _unit(key: str, fallback: float) -> float:
            value = data.get(key, fallback)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"style.{key} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigError(f"style.{key} must be within [0.0, 1.0], got {value}")
            return float(value)

        return cls(
            alpha=_unit("alpha", defaults.alpha),
            beta=_unit("beta", defaults.beta),
            cache_dir=_optional_str(data.get("cache_dir"), "style.cache_dir")
            or defaults.cache_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExportConfig:
    model_id: str
    checkpoint: AssetRef
    model_config: AssetRef
    plbert: AssetRef
    speakers: tuple                      # tuple[SpeakerConfig, ...]
    export: ExportOptions = field(default_factory=ExportOptions)
    triton: TritonOptions = field(default_factory=TritonOptions)
    style: StyleOptions = field(default_factory=StyleOptions)
    cache_root: str = "raw_models"
    source_path: Optional[str] = None    # the YAML this came from

    _ALLOWED = (
        "model_id", "stores", "defaults", "checkpoint", "plbert",
        "speakers", "export", "triton", "style", "cache_root",
    )

    @property
    def repo_dir(self) -> Path:
        """Triton model repository for this model."""
        return Path(self.triton.output_root) / self.model_id / "triton_model_repository"

    @property
    def style_dir(self) -> Path:
        """Where per-speaker ``<name>.pt`` style vectors are written."""
        return Path(self.triton.output_root) / self.model_id / "reference_styles"

    def speaker(self, name: str) -> SpeakerConfig:
        for spk in self.speakers:
            if spk.name == name:
                return spk
        known = ", ".join(s.name for s in self.speakers)
        raise KeyError(f"speaker '{name}' is not in this config. Known: {known}")

    def with_speakers(self, names: Sequence[str]) -> "ExportConfig":
        """Narrow the config to a subset of speakers (used by ``--speaker``)."""
        selected = tuple(self.speaker(n) for n in names)
        return replace(self, speakers=selected)


def _parse_asset(
    raw: Any,
    where: str,
    stores: Mapping[str, StoreConfig],
    default_store: Optional[StoreConfig],
    key: str = "path",
) -> AssetRef:
    """Parse an asset reference.

    Accepts either a bare string (a plain local path, or a path in the default
    store) or a mapping ``{store: <name>, <key>: <path>}``.
    """
    if isinstance(raw, str):
        path = _require_str(raw, where)
        store = default_store or LOCAL_STORE
        # An absolute path is unambiguous: always read it off disk.
        if os.path.isabs(path):
            store = LOCAL_STORE
        return AssetRef(store=store, path=path)

    data = _require_mapping(raw, where)
    _reject_unknown(data, ("store", key), where)

    store_name = data.get("store")
    if store_name is None:
        store = default_store or LOCAL_STORE
    else:
        store_name = _require_str(store_name, f"{where}.store")
        if store_name not in stores:
            raise ConfigError(
                f"{where}.store references undeclared store '{store_name}'; "
                f"declared: {sorted(stores) or '(none)'}"
            )
        store = stores[store_name]

    path = _require_str(data.get(key), f"{where}.{key}")
    if os.path.isabs(path) and store.type != "local":
        raise ConfigError(
            f"{where}.{key} is an absolute path but store '{store.name}' is type "
            f"'{store.type}'; object-store keys must be relative"
        )
    return AssetRef(store=store, path=path)


def load_config(path: str | os.PathLike) -> ExportConfig:
    """Read and validate an export config.

    Raises ``ConfigError`` with a key-qualified message on any problem, so a
    typo is reported before anything is downloaded.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open() as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    _reject_unknown(raw, ExportConfig._ALLOWED, str(config_path))

    model_id = _require_str(raw.get("model_id"), "model_id")

    stores = {
        name: StoreConfig.parse(name, value)
        for name, value in _require_mapping(raw.get("stores"), "stores").items()
    }

    defaults = _require_mapping(raw.get("defaults"), "defaults")
    _reject_unknown(defaults, ("store",), "defaults")
    default_store: Optional[StoreConfig] = None
    if defaults.get("store") is not None:
        default_name = _require_str(defaults["store"], "defaults.store")
        if default_name not in stores:
            raise ConfigError(
                f"defaults.store references undeclared store '{default_name}'; "
                f"declared: {sorted(stores) or '(none)'}"
            )
        default_store = stores[default_name]

    checkpoint_raw = _require_mapping(raw.get("checkpoint"), "checkpoint")
    if not checkpoint_raw:
        raise ConfigError("'checkpoint' is required (needs 'model' and 'config')")
    _reject_unknown(checkpoint_raw, ("store", "model", "config"), "checkpoint")
    ckpt_store_name = checkpoint_raw.get("store")
    ckpt_default = default_store
    if ckpt_store_name is not None:
        ckpt_store_name = _require_str(ckpt_store_name, "checkpoint.store")
        if ckpt_store_name not in stores:
            raise ConfigError(
                f"checkpoint.store references undeclared store '{ckpt_store_name}'; "
                f"declared: {sorted(stores) or '(none)'}"
            )
        ckpt_default = stores[ckpt_store_name]

    # Inside `checkpoint:`, `model` and `config` are each either a bare path
    # (resolved against checkpoint.store / defaults.store) or {store, path}.
    checkpoint = _parse_asset(
        checkpoint_raw.get("model"), "checkpoint.model", stores, ckpt_default
    )
    model_config = _parse_asset(
        checkpoint_raw.get("config"), "checkpoint.config", stores, ckpt_default
    )

    plbert = _parse_asset(raw.get("plbert"), "plbert", stores, default_store, key="path")

    speakers_raw = _require_mapping(raw.get("speakers"), "speakers")
    if not speakers_raw:
        raise ConfigError("'speakers' must list at least one speaker")

    speakers: List[SpeakerConfig] = []
    for name, value in speakers_raw.items():
        where = f"speakers.{name}"
        if isinstance(value, str):
            reference = _parse_asset(value, where, stores, default_store)
            languages: tuple = ()
        else:
            data = _require_mapping(value, where)
            _reject_unknown(data, ("store", "reference", "language", "languages"), where)
            if "language" in data and "languages" in data:
                raise ConfigError(f"{where}: set either 'language' or 'languages', not both")
            reference = _parse_asset(
                {k: v for k, v in data.items() if k in ("store", "reference")},
                where, stores, default_store, key="reference",
            )
            raw_languages = data.get("languages", data.get("language"))
            if raw_languages is None:
                languages = ()
            elif isinstance(raw_languages, str):
                languages = (_require_str(raw_languages, f"{where}.language"),)
            elif isinstance(raw_languages, Sequence):
                languages = tuple(
                    _require_str(item, f"{where}.languages[{i}]")
                    for i, item in enumerate(raw_languages)
                )
            else:
                raise ConfigError(f"{where}.languages must be a string or a list of strings")
        speakers.append(
            SpeakerConfig(name=str(name), reference=reference, languages=languages)
        )

    return ExportConfig(
        model_id=model_id,
        checkpoint=checkpoint,
        model_config=model_config,
        plbert=plbert,
        speakers=tuple(speakers),
        export=ExportOptions.parse(raw.get("export")),
        triton=TritonOptions.parse(raw.get("triton")),
        style=StyleOptions.parse(raw.get("style")),
        cache_root=_optional_str(raw.get("cache_root"), "cache_root") or "raw_models",
        source_path=str(config_path),
    )
