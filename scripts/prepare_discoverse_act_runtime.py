#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OPTIONAL_IMPORT = "from airbot_data_collection.tools.av_coder import AvCoder"
OPTIONAL_REPLACEMENT = '''try:
    from airbot_data_collection.tools.av_coder import AvCoder
except ModuleNotFoundError as exc:
    _AIRBOT_DATA_COLLECTION_ERROR = str(exc)

    class AvCoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "airbot_data_collection is required only for MCAP decoding or AIRBOT hardware evaluation; "
                "it is not required for Real2Sim HDF5 ACT training"
            ) from RuntimeError(_AIRBOT_DATA_COLLECTION_ERROR)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare pinned DISCOVERSE ACT for server-side HDF5 training.")
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root = args.act_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise SystemExit(f"ACT revision mismatch: expected {args.expected_commit}, got {commit}")
    target = root / "utils" / "utils.py"
    changed = patch_optional_airbot_dependency(target)
    print(json.dumps({"ok": True, "act_root": str(root), "commit": commit, "patched": changed}, indent=2))
    return 0


def patch_optional_airbot_dependency(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if OPTIONAL_REPLACEMENT in source:
        return False
    if OPTIONAL_IMPORT not in source:
        raise RuntimeError(f"expected AIRBOT dependency import was not found in {path}")
    path.write_text(source.replace(OPTIONAL_IMPORT, OPTIONAL_REPLACEMENT, 1), encoding="utf-8")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
