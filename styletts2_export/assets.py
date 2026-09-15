"""Fetching model assets from S3-compatible object stores or local disk.

Downloads are content-addressed under ``<cache_root>/`` and skipped when the
file is already there, so re-running an export costs nothing. Local assets are
never copied — they are used in place.

Credentials are read from the environment variables named by the store's
``access_key_env`` / ``secret_key_env`` fields. They are never read from, or
written to, any file in the repo.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

from .config import AssetRef, StoreConfig

__all__ = ["AssetError", "AssetResolver", "file_digest"]

_LOG_PREFIX = "[assets]"


class AssetError(RuntimeError):
    """Raised when an asset cannot be resolved or downloaded."""


def file_digest(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """First 12 hex chars of the SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def _key_digest(bucket: str, key: str) -> str:
    """Stable short hash of an object's fully-qualified location."""
    return hashlib.sha256(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:12]


class AssetResolver:
    """Resolves :class:`AssetRef` values to local filesystem paths.

    One resolver per export run. It memoises boto3 clients per store so a
    config with many speakers opens one connection, not one per file.
    """

    def __init__(self, cache_root: str | os.PathLike = "raw_models", quiet: bool = False):
        self.cache_root = Path(cache_root)
        self.quiet = quiet
        self._clients: dict = {}

    # ── logging ──────────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        if not self.quiet:
            print(f"{_LOG_PREFIX} {message}")

    # ── boto3 ────────────────────────────────────────────────────────────
    def _client(self, store: StoreConfig):
        if store.name in self._clients:
            return self._clients[store.name]

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - import guard
            raise AssetError(
                f"store '{store.name}' is type 's3' but boto3 is not installed. "
                "Run: pip install boto3"
            ) from exc

        access_key = os.getenv(store.access_key_env)
        secret_key = os.getenv(store.secret_key_env)
        missing = [
            name for name, value in
            ((store.access_key_env, access_key), (store.secret_key_env, secret_key))
            if not value
        ]
        if missing:
            raise AssetError(
                f"store '{store.name}' needs credentials in environment variable(s) "
                f"{missing}. Set them in your shell or in a .env file "
                f"(see .env.example), then re-run."
            )

        client = boto3.client(
            "s3",
            endpoint_url=store.endpoint_url,
            region_name=store.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._clients[store.name] = client
        return client

    # ── local ────────────────────────────────────────────────────────────
    @staticmethod
    def _local_path(ref: AssetRef) -> Path:
        path = Path(ref.path).expanduser()
        if ref.store.root and not path.is_absolute():
            path = Path(ref.store.root).expanduser() / path
        return path

    # ── public API ───────────────────────────────────────────────────────
    def resolve_file(self, ref: AssetRef, category: str, filename: Optional[str] = None) -> Path:
        """Return a local path to the file named by *ref*, downloading if needed.

        ``category`` groups downloads in the cache (``models``, ``configs``,
        ``references``, ...). ``filename`` overrides the cached basename.
        """
        if ref.store.type == "local":
            path = self._local_path(ref)
            if not path.is_file():
                raise AssetError(f"local asset not found: {path}  (from config: {ref})")
            return path

        bucket, key = ref.store.bucket, ref.path
        stem = _key_digest(bucket, key)
        suffix = Path(key).suffix
        name = filename or f"{stem}{suffix}"
        destination = self.cache_root / category / name

        if destination.is_file():
            self._log(f"cached  {destination}  ({ref})")
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        self._log(f"fetch   {ref}  ->  {destination}")
        try:
            self._client(ref.store).download_file(bucket, key, str(partial))
        except AssetError:
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise AssetError(f"failed to download {ref}: {exc}") from exc
        # Rename only after a complete download, so an interrupted run never
        # leaves a truncated file that a later run would treat as cached.
        partial.replace(destination)
        return destination

    def resolve_dir(self, ref: AssetRef, category: str) -> Path:
        """Return a local directory for *ref*, mirroring it from the store if needed.

        Used for PLBERT, which is a directory of ``config.yml`` + ``step_*.t7``.
        """
        if ref.store.type == "local":
            path = self._local_path(ref)
            if not path.is_dir():
                raise AssetError(f"local directory not found: {path}  (from config: {ref})")
            return path

        bucket = ref.store.bucket
        prefix = ref.path.rstrip("/") + "/"
        destination = self.cache_root / category / _key_digest(bucket, prefix)
        marker = destination / ".complete"

        if marker.is_file():
            self._log(f"cached  {destination}/  ({ref})")
            return destination

        if destination.exists():
            shutil.rmtree(destination)      # previous run died partway through
        destination.mkdir(parents=True, exist_ok=True)

        client = self._client(ref.store)
        self._log(f"mirror  {ref}/  ->  {destination}/")
        downloaded = 0
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    relative = key[len(prefix):]
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(target))
                    downloaded += 1
        except AssetError:
            raise
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise AssetError(f"failed to mirror {ref}/: {exc}") from exc

        if downloaded == 0:
            shutil.rmtree(destination, ignore_errors=True)
            raise AssetError(
                f"no objects found under {ref}/ — check the prefix and that the "
                f"key has list permission on bucket '{bucket}'"
            )

        marker.touch()
        self._log(f"mirror  done ({downloaded} file(s))")
        return destination
