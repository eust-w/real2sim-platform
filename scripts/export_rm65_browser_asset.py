#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from real2sim_platform.motrix_scene import export_mujoco_body_glb


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the registered RM65 MJCF visual geometry as a browser GLB")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_mujoco_body_glb(args.source.resolve(), args.output.resolve(), root_body="base_link", relative_to_root=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
