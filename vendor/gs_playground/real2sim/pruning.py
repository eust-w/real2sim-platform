"""3DGS pruning helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence, Union

from .commands import format_command
from .errors import PipelineError


CommandValue = Union[str, Sequence[str]]


@dataclass(frozen=True)
class PruneOptions:
    enabled: bool = True
    keep_ratio: float = 0.3
    keep_count: int | None = None
    mode: str = "opacity_area"
    external_command: CommandValue | None = None


def prune_ply(
    src: Path,
    dst: Path,
    options: PruneOptions,
    repo_root: Path,
    *,
    extra_context: Mapping[str, Any] | None = None,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not options.enabled or options.keep_ratio >= 1.0:
        shutil.copy2(src, dst)
        return dst

    if options.mode.replace("-", "_").lower() in {"speedy_splat", "speedysplat"}:
        return prune_with_speedy_splat(src, dst, options, repo_root, extra_context=extra_context)

    script = repo_root / "benchmark" / "scripts" / "prune_gaussians.py"
    if not script.is_file():
        raise PipelineError(f"pruning script not found: {script}")

    cmd = [sys.executable, script.as_posix(), "--input", src.as_posix(), "--output", dst.as_posix(), "--mode", options.mode]
    if options.keep_count is not None:
        cmd.extend(["--keep-count", str(options.keep_count)])
    else:
        cmd.extend(["--keep-ratio", str(options.keep_ratio)])

    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise PipelineError(f"3DGS pruning failed with exit code {completed.returncode}: {cmd}")
    return dst


def prune_with_speedy_splat(
    src: Path,
    dst: Path,
    options: PruneOptions,
    repo_root: Path,
    *,
    extra_context: Mapping[str, Any] | None = None,
) -> Path:
    if options.external_command is None:
        raise PipelineError(
            "Speedy-Splat pruning requires pruning.external_command or an equivalent "
            "configured command. The built-in opacity_area pruner is not paper-equivalent."
        )
    extras = dict(extra_context or {})
    context = {
        "input": src,
        "output": dst,
        "keep_ratio": options.keep_ratio,
        "keep_count": options.keep_count or "",
        "mode": options.mode,
        "python": sys.executable,
        "source": extras.get("source", ""),
        "source_image": extras.get("source_image", extras.get("image", "")),
        "image": extras.get("image", extras.get("source_image", "")),
        "cameras_json": extras.get("cameras_json", extras.get("cameras", "")),
        "cameras": extras.get("cameras", extras.get("cameras_json", "")),
    }
    argv = format_command(options.external_command, context)
    completed = subprocess.run(argv, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise PipelineError(f"Speedy-Splat pruning failed with exit code {completed.returncode}: {argv}")
    if not dst.exists():
        raise PipelineError(f"Speedy-Splat command did not write output PLY: {dst}")
    return dst
