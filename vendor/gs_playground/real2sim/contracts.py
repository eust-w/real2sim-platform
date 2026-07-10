"""Artifact contracts for Real2Sim stage boundaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import StageArtifactError


def read_json(path: Path) -> Any:
    if not path.exists():
        raise StageArtifactError(f"missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise StageArtifactError(f"missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise StageArtifactError(f"missing {description}: {path}")
    return path


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_json(cls, path: Path) -> "Pose":
        data = read_json(path)
        if not isinstance(data, Mapping):
            raise StageArtifactError(f"pose artifact must be a JSON object: {path}")

        pos = data.get("position", data.get("translation", [0.0, 0.0, 0.0]))
        quat = data.get("quat_wxyz", data.get("quaternion_wxyz"))
        if quat is None and "quaternion_xyzw" in data:
            qx, qy, qz, qw = _float_list(data["quaternion_xyzw"], 4, "quaternion_xyzw", path)
            quat = [qw, qx, qy, qz]
        if quat is None:
            quat = [1.0, 0.0, 0.0, 0.0]

        scale_raw = data.get("scale", [1.0, 1.0, 1.0])
        if isinstance(scale_raw, (int, float)):
            scale = (float(scale_raw), float(scale_raw), float(scale_raw))
        else:
            scale = tuple(_float_list(scale_raw, 3, "scale", path))

        return cls(
            position=tuple(_float_list(pos, 3, "position", path)),
            quat_wxyz=tuple(_float_list(quat, 4, "quat_wxyz", path)),
            scale=scale,
        )

    def to_json(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "quat_wxyz": list(self.quat_wxyz),
            "scale": list(self.scale),
        }


@dataclass(frozen=True)
class Instance:
    instance_id: str
    label: str
    score: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], base_dir: Path) -> "Instance":
        raw_id = raw.get("id", raw.get("instance_id", raw.get("name")))
        label = raw.get("label", raw.get("class", raw_id))
        if raw_id is None or label is None:
            raise StageArtifactError(f"instance entry requires id/name and label: {raw}")
        bbox = raw.get("bbox_xyxy", raw.get("bbox"))
        mask = raw.get("mask_path", raw.get("mask"))
        return cls(
            instance_id=str(raw_id),
            label=str(label),
            score=float(raw["score"]) if raw.get("score") is not None else None,
            bbox_xyxy=tuple(_float_list(bbox, 4, "bbox", base_dir)) if bbox is not None else None,
            mask_path=(base_dir / str(mask)).resolve() if mask else None,
        )


@dataclass(frozen=True)
class Real2SimAsset:
    name: str
    label: str
    gaussian_ply: Path
    mesh_obj: Path | None = None
    pose: Pose = field(default_factory=Pose)
    dynamic: bool = False
    source_instance_id: str | None = None

    def to_json(self, root: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "gaussian_ply": _relative_or_absolute(self.gaussian_ply, root),
            "mesh_obj": _relative_or_absolute(self.mesh_obj, root) if self.mesh_obj else None,
            "pose": self.pose.to_json(),
            "dynamic": self.dynamic,
            "source_instance_id": self.source_instance_id,
        }


@dataclass(frozen=True)
class Real2SimManifest:
    source: Path
    prompt: str
    scene: Real2SimAsset
    objects: tuple[Real2SimAsset, ...] = ()
    camera_artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, root: Path) -> dict[str, Any]:
        return {
            "source": _relative_or_absolute(self.source, root),
            "prompt": self.prompt,
            "scene": self.scene.to_json(root),
            "objects": [obj.to_json(root) for obj in self.objects],
            "camera_artifacts": {
                key: _relative_or_absolute(value, root)
                for key, value in sorted(self.camera_artifacts.items())
            },
            "metadata": self.metadata,
        }


def load_instances(path: Path) -> list[Instance]:
    data = read_json(path)
    entries = data.get("instances", data) if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise StageArtifactError(f"instances artifact must be a list or object with instances list: {path}")
    return [Instance.from_mapping(entry, path.parent) for entry in entries]


def validate_detections(path: Path) -> None:
    data = read_json(path)
    entries = data.get("detections", data) if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise StageArtifactError(f"detections artifact must be a list or object with detections list: {path}")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise StageArtifactError(f"detection entries must be JSON objects: {path}")
        if "bbox" not in entry and "bbox_xyxy" not in entry:
            raise StageArtifactError(f"detection entry missing bbox/bbox_xyxy: {entry}")


def _float_list(value: Any, size: int, field_name: str, path: Path) -> list[float]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise StageArtifactError(f"{field_name} must be a list with {size} values in {path}")
    vals = [float(item) for item in value]
    if len(vals) != size:
        raise StageArtifactError(f"{field_name} must have {size} values in {path}, got {len(vals)}")
    return vals


def _relative_or_absolute(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
