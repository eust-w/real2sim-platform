from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


SOURCE_MARKER = ".real2sim-source.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_source_revision(root: Path) -> str:
    """Return the pinned revision from Git metadata or a verified archive marker."""

    root = root.resolve()
    if not root.is_dir():
        return ""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip().lower()
        if _COMMIT_RE.fullmatch(commit):
            return commit
    except Exception:
        pass
    marker = read_source_marker(root)
    return str(marker.get("commit", ""))


def read_source_marker(root: Path) -> dict[str, Any]:
    marker_path = root.resolve() / SOURCE_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if marker.get("schema") != "real2sim.source.v1":
        return {}
    commit = str(marker.get("commit", "")).lower()
    archive_sha256 = str(marker.get("archive_sha256", "")).lower()
    source = str(marker.get("source", ""))
    if not _COMMIT_RE.fullmatch(commit) or not _SHA256_RE.fullmatch(archive_sha256) or not source.startswith("https://"):
        return {}
    return {"schema": marker["schema"], "source": source, "commit": commit, "archive_sha256": archive_sha256}
