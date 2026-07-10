#!/usr/bin/env python3
"""SAM2 segmentation adapter for Real2Sim.

Consumes detections.json boxes and writes instances.json plus per-instance PGM
masks. This uses Meta's official SAM2 image predictor API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import read_json, write_json
from real2sim.image_io import write_gray


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM2 masks from GroundingDINO boxes")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("REAL2SIM_SAM2_CHECKPOINT", ""))
    parser.add_argument("--model-cfg", default=os.environ.get("REAL2SIM_SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if not args.checkpoint:
        raise SystemExit("SAM2 checkpoint is required via --checkpoint or REAL2SIM_SAM2_CHECKPOINT")

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    detections_raw = read_json(args.detections)
    detections = detections_raw.get("detections", detections_raw) if isinstance(detections_raw, dict) else detections_raw
    if not detections:
        write_json(args.output, {"instances": [], "metadata": {"backend": "sam2"}})
        return 0

    boxes = np.array([entry.get("bbox_xyxy", entry.get("bbox")) for entry in detections], dtype=np.float32)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    image = np.array(Image.open(args.image).convert("RGB"))
    model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    args.masks_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = predict_masks(predictor, image, boxes)
        else:
            masks, scores, _ = predict_masks(predictor, image, boxes)

    masks = normalize_masks(masks)
    instances = []
    for idx, (detection, mask) in enumerate(zip(detections, masks)):
        instance_id = str(detection.get("id", f"det_{idx:03d}"))
        mask_path = args.masks_dir / f"{safe_name(instance_id)}.pgm"
        write_gray(mask_path, mask.astype(np.float32))
        segmentation_score = None
        if scores is not None and idx < len(scores):
            segmentation_score = float(np.asarray(scores[idx]).reshape(-1)[0])
        instances.append(
            instance_record(
                detection,
                mask_path=mask_path,
                output_parent=args.output.parent,
                segmentation_score=segmentation_score,
            )
        )

    write_json(args.output, {"instances": instances, "metadata": {"backend": "sam2", "model_cfg": args.model_cfg}})
    return 0


def predict_masks(predictor, image, boxes):
    predictor.set_image(image)
    return predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)


def normalize_masks(masks):
    import numpy as np

    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return masks > 0


def instance_record(detection: dict, *, mask_path: Path, output_parent: Path, segmentation_score: float | None) -> dict:
    instance_id = str(detection.get("id", "instance"))
    detection_score = detection.get("score")
    score = float(detection_score) if detection_score is not None else segmentation_score
    record = {
        "id": instance_id,
        "label": str(detection.get("label", instance_id)),
        "score": score,
        "bbox_xyxy": detection.get("bbox_xyxy", detection.get("bbox")),
        "mask_path": mask_path.relative_to(output_parent).as_posix(),
    }
    if detection_score is not None:
        record["detection_score"] = float(detection_score)
    if segmentation_score is not None:
        record["segmentation_score"] = float(segmentation_score)
    for key in ("prompt_index", "prompt_query"):
        if key in detection:
            record[key] = detection[key]
    return record


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "instance"


if __name__ == "__main__":
    raise SystemExit(main())
