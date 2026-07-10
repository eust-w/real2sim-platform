#!/usr/bin/env python3
"""Sequential LaMa inpainting adapter for Real2Sim.

Preferred runtime is the lightweight simple-lama-inpainting package, which is
listed by the official LaMa repository as a third-party app. This adapter keeps
the Real2Sim contract stable: image + instances.json -> background image.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.commands import format_command
from real2sim.contracts import load_instances, write_json
from real2sim.image_io import read_gray, read_rgb, write_gray, write_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LaMa object removal from Real2Sim instance masks")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=7)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--redetect-command", default="")
    parser.add_argument("--segment-command", default="")
    parser.add_argument("--redetect-command-env", default="")
    parser.add_argument("--segment-command-env", default="")
    parser.add_argument("--require-redetect-loop", action="store_true")
    parser.add_argument("--max-recovery-passes", type=int, default=1)
    parser.add_argument("--max-area-growth", type=float, default=0.5)
    args = parser.parse_args()

    instances = load_instances(args.instances)
    image = read_rgb(args.image)
    redetect_command = args.redetect_command or (os.environ.get(args.redetect_command_env, "") if args.redetect_command_env else "")
    segment_command = args.segment_command or (os.environ.get(args.segment_command_env, "") if args.segment_command_env else "")
    require_recovery_commands(redetect_command, segment_command, strict=args.require_redetect_loop)

    try:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
    except Exception as exc:
        raise SystemExit(
            "simple-lama-inpainting is not installed. Install it in the external "
            "model environment or use the built-in backend."
        ) from exc

    lama = SimpleLama()
    current = image
    steps = []
    recovery_events = []
    work_dir = args.work_dir or args.output.parent
    instance_masks = {
        instance.instance_id: read_gray(instance.mask_path) > 0.5
        for instance in instances
        if instance.mask_path and instance.mask_path.exists()
    }
    ordered = sorted(
        [instance for instance in instances if instance.mask_path],
        key=lambda inst: (-(inst.score if inst.score is not None else 0.0), inst.instance_id),
    )
    for instance in ordered:
        mask = read_gray(instance.mask_path) > 0.5
        if mask.shape != current.shape[:2]:
            continue
        expanded = dilate(mask, args.dilate)
        pil_image = Image.fromarray((np.clip(current, 0, 1) * 255).astype(np.uint8))
        pil_mask = Image.fromarray((expanded.astype(np.uint8) * 255), mode="L")
        result = lama(pil_image, pil_mask)
        current = np.asarray(result.convert("RGB"), dtype=np.float32) / 255.0
        if redetect_command and segment_command:
            recovery_events.extend(
                recover_occluded_regions(
                    current,
                    instances=instances,
                    instance_masks=instance_masks,
                    instances_json=args.instances,
                    prompt=args.prompt,
                    work_dir=work_dir,
                    pass_index=len(steps),
                    redetect_command=redetect_command,
                    segment_command=segment_command,
                    max_area_growth=args.max_area_growth,
                )
            )
        steps.append(
            {
                "instance_id": instance.instance_id,
                "label": instance.label,
                "mask_pixels": int(mask.sum()),
                "expanded_mask_pixels": int(expanded.sum()),
            }
        )
    write_rgb(args.output, current)
    write_json(
        args.output.with_suffix(args.output.suffix + ".json"),
        {
            "backend": "simple_lama_sequential",
            "strategy": "one_object_at_a_time_mask_expansion",
            "steps": steps,
            "recovery_events": recovery_events,
            "redetect_loop": bool(redetect_command and segment_command),
        },
    )
    return 0


def require_recovery_commands(redetect_command: str, segment_command: str, *, strict: bool) -> None:
    if strict and (not redetect_command or not segment_command):
        raise SystemExit(
            "paper-strict sequential inpainting requires both a re-detection command "
            "and a segmentation command for newly exposed region recovery"
        )


def recover_occluded_regions(
    image: np.ndarray,
    *,
    instances,
    instance_masks: dict[str, np.ndarray],
    instances_json: Path,
    prompt: str,
    work_dir: Path,
    pass_index: int,
    redetect_command,
    segment_command,
    max_area_growth: float,
) -> list[dict]:
    recovery_dir = work_dir / "03_inpainting" / "recovery" / f"pass_{pass_index:03d}"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    image_path = recovery_dir / "image.ppm"
    detections_json = recovery_dir / "detections.json"
    masks_dir = recovery_dir / "masks"
    recovered_instances_json = recovery_dir / "instances.json"
    write_rgb(image_path, image)

    base_context = {
        "python": sys.executable,
        "image": image_path,
        "input": image_path,
        "prompt": prompt,
        "detections_json": detections_json,
        "instances_json": recovered_instances_json,
        "masks_dir": masks_dir,
    }
    run_nested_command(redetect_command, base_context)
    run_nested_command(segment_command, base_context)

    existing = {instance.instance_id: instance for instance in instances}
    events = []
    for candidate in load_instances(recovered_instances_json):
        if candidate.mask_path is None or not candidate.mask_path.exists():
            continue
        candidate_mask = read_gray(candidate.mask_path) > 0.5
        for instance in instances:
            if instance.label != candidate.label or instance.mask_path is None:
                continue
            base_mask = instance_masks.get(instance.instance_id)
            if base_mask is None:
                continue
            merged = merge_recovered_mask(base_mask, candidate_mask, same_label=True, max_area_growth=max_area_growth)
            if int(merged.sum()) == int(base_mask.sum()):
                continue
            write_gray(instance.mask_path, merged.astype(np.float32))
            instance_masks[instance.instance_id] = merged
            events.append(
                {
                    "target_instance_id": instance.instance_id,
                    "candidate_instance_id": candidate.instance_id,
                    "label": instance.label,
                    "old_mask_pixels": int(base_mask.sum()),
                    "new_mask_pixels": int(merged.sum()),
                }
            )
            break
    return events


def run_nested_command(command, context: dict) -> None:
    argv = format_command(command, context)
    completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"nested recovery command failed with exit code {completed.returncode}: {argv}")


def merge_recovered_mask(base: np.ndarray, candidate: np.ndarray, *, same_label: bool, max_area_growth: float) -> np.ndarray:
    if not same_label or base.shape != candidate.shape:
        return base
    new_region = candidate & ~base
    if not new_region.any():
        return base
    if int(new_region.sum()) > int(base.sum()) * max_area_growth:
        return base
    if not (dilate(base, 1) & new_region).any():
        return base
    return base | new_region


def dilate(mask: np.ndarray, passes: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
