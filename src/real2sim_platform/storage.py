from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import PlatformConfig


ALLOWED_AREAS = {"uploads", "datasets", "scenes", "models", "runs", "backups"}
_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9._-]+")


class StorageUnavailable(RuntimeError):
    pass


class StorageBoundaryError(ValueError):
    pass


class BosStorage:
    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.bucket = config.bos_bucket.strip()
        self.prefix = normalize_prefix(config.bos_prefix)
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.bucket and self.config.bos_access_key_id and self.config.bos_secret_access_key)

    def health(self) -> dict[str, Any]:
        result = {"configured": self.configured, "bucket": self.bucket, "prefix": self.prefix, "endpoint": self.config.bos_endpoint}
        if not self.configured:
            result["ready"] = False
            result["reason"] = "BOS credentials or bucket are not configured"
            return result
        try:
            self.list_objects(area="uploads", relative_prefix="", max_keys=1)
            result["ready"] = True
        except Exception as exc:
            result["ready"] = False
            result["reason"] = str(exc)
        return result

    def object_key(self, area: str, relative_path: str) -> str:
        if area not in ALLOWED_AREAS:
            raise StorageBoundaryError(f"unsupported BOS area: {area}")
        relative = normalize_relative_path(relative_path)
        return f"{self.prefix}/{area}/{relative}" if relative else f"{self.prefix}/{area}/"

    def uri_for_key(self, key: str) -> str:
        checked = self.assert_owned_key(key)
        return f"bos://{self.bucket}/{checked}"

    def assert_owned_key(self, key: str) -> str:
        normalized = str(PurePosixPath(str(key).strip().lstrip("/")))
        owned_root = f"{self.prefix}/"
        if not normalized.startswith(owned_root) or normalized == self.prefix:
            raise StorageBoundaryError("object key is outside the Real2Sim BOS prefix")
        area = normalized[len(owned_root):].split("/", 1)[0]
        if area not in ALLOWED_AREAS:
            raise StorageBoundaryError("object key is outside an approved Real2Sim area")
        if ".." in PurePosixPath(normalized).parts:
            raise StorageBoundaryError("parent traversal is not allowed")
        return normalized

    def upload_file(self, local_path: Path, *, area: str, relative_path: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        key = self.object_key(area, relative_path)
        client = self._get_client()
        kwargs: dict[str, Any] = {}
        if metadata:
            kwargs["user_metadata"] = {str(k): str(v) for k, v in metadata.items()}
        client.put_object_from_file(self.bucket, key, str(local_path), **kwargs)
        return {
            "key": key,
            "storage_uri": self.uri_for_key(key),
            "size_bytes": local_path.stat().st_size,
            "checksum": sha256_file(local_path),
        }

    def download_file(self, storage_uri: str, target_path: Path) -> Path:
        key = self.key_from_uri(storage_uri)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._get_client().get_object_to_file(self.bucket, key, str(target_path))
        return target_path

    def list_objects(self, *, area: str, relative_prefix: str = "", max_keys: int = 1000) -> list[dict[str, Any]]:
        prefix = self.object_key(area, relative_prefix)
        response = self._get_client().list_objects(self.bucket, prefix=prefix, max_keys=max(1, min(1000, int(max_keys))))
        contents: Iterable[Any] = getattr(response, "contents", None) or []
        return [
            {
                "key": self.assert_owned_key(str(getattr(item, "key", ""))),
                "storage_uri": self.uri_for_key(str(getattr(item, "key", ""))),
                "size_bytes": int(getattr(item, "size", 0) or 0),
                "etag": str(getattr(item, "e_tag", getattr(item, "etag", "")) or "").strip('"'),
                "last_modified": str(getattr(item, "last_modified", "") or ""),
            }
            for item in contents
            if getattr(item, "key", None)
        ]

    def delete_object(self, storage_uri: str) -> None:
        key = self.key_from_uri(storage_uri)
        self._get_client().delete_object(self.bucket, key)

    def key_from_uri(self, storage_uri: str) -> str:
        prefix = f"bos://{self.bucket}/"
        if not storage_uri.startswith(prefix):
            raise StorageBoundaryError("BOS URI bucket does not match configured bucket")
        return self.assert_owned_key(storage_uri[len(prefix):])

    def _get_client(self):
        if not self.configured:
            raise StorageUnavailable("BOS storage is not configured")
        if self._client is not None:
            return self._client
        try:
            from baidubce.auth.bce_credentials import BceCredentials
            from baidubce.bce_client_configuration import BceClientConfiguration
            from baidubce.services.bos.bos_client import BosClient
        except ImportError as exc:
            raise StorageUnavailable("bce-python-sdk is not installed") from exc
        credentials = BceCredentials(self.config.bos_access_key_id, self.config.bos_secret_access_key)
        configuration = BceClientConfiguration(credentials=credentials, endpoint=self.config.bos_endpoint)
        self._client = BosClient(configuration)
        return self._client


def normalize_prefix(value: str) -> str:
    parts = PurePosixPath(str(value).strip().strip("/"))
    if not str(parts) or str(parts) == "." or ".." in parts.parts:
        raise StorageBoundaryError("BOS prefix must be a non-empty relative path")
    return str(parts)


def normalize_relative_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip().lstrip("/")
    if not raw:
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise StorageBoundaryError("BOS path traversal is not allowed")
    return str(path)


def safe_object_name(filename: str) -> str:
    name = Path(filename).name.strip()
    cleaned = _SAFE_COMPONENT.sub("-", name).strip(".-")
    if not cleaned:
        raise ValueError("filename is empty after sanitization")
    return cleaned[:180]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_part(path: Path, chunks: Iterable[bytes]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return size, digest.hexdigest()
