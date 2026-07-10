#!/usr/bin/env python3
"""GroundingDINO detection adapter for Real2Sim.

This uses the Hugging Face GroundingDINO interface used by the official
Grounded-SAM-2 demos. It writes the Real2Sim detections.json contract.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GroundingDINO via Hugging Face and write detections.json")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    detections = []
    queries = prompt_queries(args.prompt)
    for prompt_index, prompt in enumerate(queries):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        for idx, box in enumerate(boxes):
            label = labels[idx] if idx < len(labels) else prompt.rstrip(".")
            score = float(scores[idx].detach().cpu().item()) if idx < len(scores) else None
            bbox = [float(v) for v in box.detach().cpu().tolist()]
            detections.append(
                {
                    "id": f"det_{prompt_index:03d}_{idx:03d}",
                    "label": str(label).strip().rstrip(".") or prompt.rstrip("."),
                    "score": score,
                    "bbox_xyxy": bbox,
                    "prompt_query": prompt,
                    "prompt_index": prompt_index,
                }
            )

    detections = nms_detections(detections, args.nms_threshold)[: max(0, args.max_detections)]
    write_json(
        args.output,
        {
            "detections": detections,
            "metadata": {
                "backend": "groundingdino_hf",
                "model_id": args.model_id,
                "prompt": args.prompt,
                "prompt_queries": queries,
                "detection_scheme": "prompt_wise_independent",
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "nms_threshold": args.nms_threshold,
                "max_detections": args.max_detections,
            },
        },
    )
    return 0


def normalize_prompt(prompt: str) -> str:
    text = " ".join(prompt.strip().lower().split())
    if not text:
        text = "object"
    return text if text.endswith(".") else f"{text}."


def prompt_queries(prompt: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[.,;\n]+", prompt) if part.strip()]
    if not parts:
        parts = ["object"]
    return [normalize_prompt(part) for part in parts]


def nms_detections(detections: list[dict], threshold: float) -> list[dict]:
    pending = sorted(detections, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    kept: list[dict] = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        pending = [
            candidate
            for candidate in pending
            if current.get("label") != candidate.get("label")
            or iou(current["bbox_xyxy"], candidate["bbox_xyxy"]) < threshold
        ]
    for idx, item in enumerate(kept):
        item["id"] = f"det_{idx:03d}"
    return kept


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
