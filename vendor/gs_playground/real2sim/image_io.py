"""Small image I/O helpers used by the built-in Real2Sim baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import PipelineError


def read_rgb(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return _read_ppm(path)

    try:
        import matplotlib.image as mpimg
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise PipelineError(
            f"reading {suffix or 'this image'} requires matplotlib in this environment. "
            "Use a .ppm input or run through the project uv environment."
        ) from exc

    arr = np.asarray(mpimg.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        alpha = arr[:, :, 3:4]
        arr = arr[:, :, :3] * alpha + (1.0 - alpha)
    else:
        arr = arr[:, :, :3]
    if arr.dtype.kind in {"u", "i"} or arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def read_gray(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pgm":
        return _read_pgm(path)

    try:
        import matplotlib.image as mpimg
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise PipelineError(
            f"reading {suffix or 'this mask'} requires matplotlib in this environment. "
            "Use a .pgm mask or run through the project uv environment."
        ) from exc

    arr = np.asarray(mpimg.imread(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    if arr.dtype.kind in {"u", "i"} or arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    if path.suffix.lower() == ".ppm":
        _write_ppm(path, arr)
        return
    try:
        import matplotlib.image as mpimg

        mpimg.imsave(path, arr)
    except Exception:
        _write_ppm(path, arr)


def write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    if path.suffix.lower() == ".pgm":
        _write_pgm(path, arr)
        return
    try:
        import matplotlib.image as mpimg

        mpimg.imsave(path, arr, cmap="gray")
    except Exception:
        _write_pgm(path, arr)


def _read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise PipelineError(f"only binary P6 PPM input is supported without matplotlib: {path}")
        dims = _next_ppm_token_line(f)
        width, height = [int(x) for x in dims.split()]
        maxval = int(_next_ppm_token_line(f))
        if maxval <= 0 or maxval > 255:
            raise PipelineError(f"unsupported PPM max value {maxval}: {path}")
        data = np.frombuffer(f.read(width * height * 3), dtype=np.uint8)
    if data.size != width * height * 3:
        raise PipelineError(f"truncated PPM input: {path}")
    return data.reshape(height, width, 3).astype(np.float32) / float(maxval)


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise PipelineError(f"only binary P5 PGM input is supported without matplotlib: {path}")
        dims = _next_ppm_token_line(f)
        width, height = [int(x) for x in dims.split()]
        maxval = int(_next_ppm_token_line(f))
        if maxval <= 0 or maxval > 255:
            raise PipelineError(f"unsupported PGM max value {maxval}: {path}")
        data = np.frombuffer(f.read(width * height), dtype=np.uint8)
    if data.size != width * height:
        raise PipelineError(f"truncated PGM input: {path}")
    return data.reshape(height, width).astype(np.float32) / float(maxval)


def _write_ppm(path: Path, image: np.ndarray) -> None:
    arr = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    with path.open("wb") as f:
        f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        f.write(arr[:, :, :3].tobytes())


def _write_pgm(path: Path, image: np.ndarray) -> None:
    arr = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    with path.open("wb") as f:
        f.write(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def _next_ppm_token_line(f) -> str:
    while True:
        line = f.readline()
        if not line:
            raise PipelineError("unexpected EOF while reading PPM header")
        line = line.strip()
        if line and not line.startswith(b"#"):
            return line.decode("ascii")
