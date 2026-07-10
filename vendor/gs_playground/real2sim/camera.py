"""Camera utilities shared by Real2Sim packaging and viewers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import StageArtifactError


@dataclass(frozen=True)
class SourceCamera:
    position: np.ndarray
    xmat: np.ndarray
    fovy_degrees: float
    width: int
    height: int
    y_up: bool = False


def load_anysplat_source_camera(path: Path) -> SourceCamera:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise StageArtifactError(f"camera JSON must be an object: {path}")

    intrinsics = payload.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise StageArtifactError(f"camera JSON missing intrinsics: {path}")
    for key in ("width", "height", "fy"):
        if key not in intrinsics:
            raise StageArtifactError(f"camera JSON intrinsics missing {key}: {path}")

    extrinsic_value: Any = payload.get("pred_context_pose", {}).get("extrinsic")
    if extrinsic_value is None:
        extrinsic_value = payload.get("extrinsic")
    extrinsic = _first_matrix(extrinsic_value, rows=4, cols=4, path=path)
    c2w = np.linalg.inv(extrinsic).astype(np.float32)

    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fy = float(intrinsics["fy"])
    if width <= 0 or height <= 0 or fy <= 0:
        raise StageArtifactError(f"camera JSON has invalid dimensions or focal length: {path}")

    fovy = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return SourceCamera(
        position=c2w[:3, 3].astype(np.float32),
        xmat=c2w[:3, :3].reshape(9).astype(np.float32),
        fovy_degrees=fovy,
        width=width,
        height=height,
        y_up=False,
    )


def _first_matrix(value: Any, *, rows: int, cols: int, path: Path) -> np.ndarray:
    if value is None:
        raise StageArtifactError(f"camera JSON missing extrinsic matrix: {path}")
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[1:] == (rows, cols):
        arr = arr[0]
    if arr.shape != (rows, cols):
        raise StageArtifactError(f"camera JSON extrinsic must have shape {(rows, cols)}, got {arr.shape}: {path}")
    return arr
