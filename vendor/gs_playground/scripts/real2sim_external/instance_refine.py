#!/usr/bin/env python3
"""Paper-style visual instance refinement for Real2Sim.

The GS-Playground paper avoids semantic-similarity de-duplication after
open-vocabulary detection. This stage uses only visual mask criteria and a
composite detection/segmentation confidence score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, REPO_ROOT.as_posix())

from real2sim.contracts import read_json, write_json
from real2sim.image_io import read_gray, read_rgb, write_gray


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine SAM/SAM2 instances using visual de-duplication")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="Optional source RGB image for accessory mask recovery")
    parser.add_argument("--mask-iou-threshold", type=float, default=0.75)
    parser.add_argument("--inclusion-threshold", type=float, default=0.92)
    parser.add_argument("--boundary-overlap-threshold", type=float, default=0.85)
    parser.add_argument("--merge-saturated-accessories", action="store_true")
    parser.add_argument("--accessory-search-margin-ratio", type=float, default=0.25)
    parser.add_argument("--accessory-saturation-threshold", type=float, default=0.45)
    parser.add_argument("--accessory-contact-passes", type=int, default=4)
    parser.add_argument("--accessory-max-area-growth", type=float, default=0.15)
    parser.add_argument("--accessory-min-component-area-ratio", type=float, default=0.001)
    parser.add_argument("--accessory-max-components", type=int, default=8)
    args = parser.parse_args()

    raw = read_json(args.instances)
    entries = raw.get("instances", raw) if isinstance(raw, dict) else raw
    image = read_rgb(args.image) if args.image is not None and args.image.exists() else None
    refined = refine_instance_entries(
        entries,
        args.instances.parent,
        image=image,
        merge_saturated_accessories=args.merge_saturated_accessories,
        mask_iou_threshold=args.mask_iou_threshold,
        inclusion_threshold=args.inclusion_threshold,
        boundary_overlap_threshold=args.boundary_overlap_threshold,
        accessory_search_margin_ratio=args.accessory_search_margin_ratio,
        accessory_saturation_threshold=args.accessory_saturation_threshold,
        accessory_contact_passes=args.accessory_contact_passes,
        accessory_max_area_growth=args.accessory_max_area_growth,
        accessory_min_component_area_ratio=args.accessory_min_component_area_ratio,
        accessory_max_components=args.accessory_max_components,
    )
    metadata = dict(raw.get("metadata", {})) if isinstance(raw, dict) else {}
    metadata["instance_refinement"] = {
        "backend": "visual_mask_dedup",
        "input_instances": len(entries),
        "output_instances": len(refined),
        "mask_iou_threshold": args.mask_iou_threshold,
        "inclusion_threshold": args.inclusion_threshold,
        "boundary_overlap_threshold": args.boundary_overlap_threshold,
        "merge_saturated_accessories": args.merge_saturated_accessories,
        "accessory_search_margin_ratio": args.accessory_search_margin_ratio,
        "accessory_saturation_threshold": args.accessory_saturation_threshold,
        "accessory_contact_passes": args.accessory_contact_passes,
        "accessory_max_area_growth": args.accessory_max_area_growth,
        "accessory_min_component_area_ratio": args.accessory_min_component_area_ratio,
        "accessory_max_components": args.accessory_max_components,
        "accessory_pixels_added": sum(int(entry.get("accessory_pixels_added", 0)) for entry in refined),
    }
    write_json(args.output, {"instances": refined, "metadata": metadata})
    return 0


def refine_instance_entries(
    entries: list[dict[str, Any]],
    base_dir: Path,
    *,
    image: np.ndarray | None = None,
    merge_saturated_accessories: bool = False,
    mask_iou_threshold: float = 0.75,
    inclusion_threshold: float = 0.92,
    boundary_overlap_threshold: float = 0.85,
    accessory_search_margin_ratio: float = 0.25,
    accessory_saturation_threshold: float = 0.45,
    accessory_contact_passes: int = 4,
    accessory_max_area_growth: float = 0.15,
    accessory_min_component_area_ratio: float = 0.001,
    accessory_max_components: int = 8,
) -> list[dict[str, Any]]:
    prepared = []
    for index, entry in enumerate(entries):
        mask_path = entry.get("mask_path", entry.get("mask"))
        if not mask_path:
            continue
        path = Path(str(mask_path))
        path = path if path.is_absolute() else (base_dir / path).resolve()
        if not path.exists():
            continue
        mask = read_gray(path) > 0.5
        if merge_saturated_accessories and image is not None and image.shape[:2] == mask.shape:
            mask, merge_event = merge_saturated_accessories_into_mask(
                mask,
                image,
                bbox_xyxy=entry.get("bbox_xyxy"),
                search_margin_ratio=accessory_search_margin_ratio,
                saturation_threshold=accessory_saturation_threshold,
                contact_passes=accessory_contact_passes,
                max_area_growth=accessory_max_area_growth,
                min_component_area_ratio=accessory_min_component_area_ratio,
                max_components=accessory_max_components,
            )
            if merge_event["added_pixels"] > 0:
                write_gray(path, mask.astype(np.float32))
                entry = dict(entry)
                entry["accessory_pixels_added"] = merge_event["added_pixels"]
                entry["accessory_components_added"] = merge_event["components_added"]
        score = composite_score(entry)
        item = dict(entry)
        item["composite_score"] = score
        item["_input_index"] = index
        prepared.append((item, mask))

    prepared.sort(key=lambda pair: (-float(pair[0]["composite_score"]), int(pair[0]["_input_index"])))
    kept: list[tuple[dict[str, Any], np.ndarray]] = []
    for item, mask in prepared:
        duplicate = False
        for kept_item, kept_mask in kept:
            if is_visual_duplicate(
                mask,
                kept_mask,
                mask_iou_threshold=mask_iou_threshold,
                inclusion_threshold=inclusion_threshold,
                boundary_overlap_threshold=boundary_overlap_threshold,
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append((item, mask))

    kept.sort(key=lambda pair: int(pair[0]["_input_index"]))
    output = []
    for idx, (item, _) in enumerate(kept):
        cleaned = dict(item)
        cleaned.pop("_input_index", None)
        cleaned["id"] = str(cleaned.get("id", cleaned.get("instance_id", f"inst_{idx:03d}")))
        output.append(cleaned)
    return output


def composite_score(entry: dict[str, Any]) -> float:
    detection = entry.get("detection_score", entry.get("score", 0.0))
    segmentation = entry.get("segmentation_score", entry.get("mask_score", detection))
    detection = 0.0 if detection is None else float(detection)
    segmentation = 0.0 if segmentation is None else float(segmentation)
    return 0.5 * detection + 0.5 * segmentation


def is_visual_duplicate(
    mask: np.ndarray,
    reference: np.ndarray,
    *,
    mask_iou_threshold: float,
    inclusion_threshold: float,
    boundary_overlap_threshold: float,
) -> bool:
    if mask.shape != reference.shape:
        return False
    intersection = int((mask & reference).sum())
    area = int(mask.sum())
    reference_area = int(reference.sum())
    if area == 0 or reference_area == 0:
        return False
    union = area + reference_area - intersection
    if union > 0 and intersection / union >= mask_iou_threshold:
        return True
    smaller = mask if area <= reference_area else reference
    larger = reference if area <= reference_area else mask
    inclusion = int((smaller & larger).sum()) / float(int(smaller.sum()))
    boundary_overlap = int((boundary(smaller) & larger).sum()) / max(1, int(boundary(smaller).sum()))
    return inclusion >= inclusion_threshold and boundary_overlap >= boundary_overlap_threshold


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded &= padded[:-2, 1:-1]
    eroded &= padded[2:, 1:-1]
    eroded &= padded[1:-1, :-2]
    eroded &= padded[1:-1, 2:]
    return mask & ~eroded


def merge_saturated_accessories_into_mask(
    mask: np.ndarray,
    image: np.ndarray,
    *,
    bbox_xyxy: Any | None = None,
    search_margin_ratio: float = 0.25,
    saturation_threshold: float = 0.45,
    contact_passes: int = 4,
    max_area_growth: float = 0.15,
    min_component_area_ratio: float = 0.001,
    max_components: int = 8,
) -> tuple[np.ndarray, dict[str, int]]:
    base = np.asarray(mask, dtype=bool)
    if not base.any():
        return base, {"added_pixels": 0, "components_added": 0}

    search = expanded_search_region(base, bbox_xyxy, margin_ratio=search_margin_ratio)
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    saturation = rgb_saturation(rgb)
    candidates = (saturation >= saturation_threshold) & search & ~base
    if not candidates.any():
        return base, {"added_pixels": 0, "components_added": 0}

    contact = dilate(base, max(0, int(contact_passes)))
    max_added = max(1, int(base.sum() * max(0.0, max_area_growth)))
    min_component_area = max(1, int(np.ceil(base.sum() * max(0.0, min_component_area_ratio))))
    merged = base.copy()
    added_pixels = 0
    components_added = 0
    components = connected_components(candidates)
    components.sort(key=lambda comp: int(comp.sum()), reverse=True)
    for component in components:
        if max_components > 0 and components_added >= max_components:
            break
        area = int(component.sum())
        if area < min_component_area or added_pixels + area > max_added:
            continue
        if not (component & contact).any():
            continue
        merged |= component
        contact = dilate(merged, max(0, int(contact_passes)))
        added_pixels += area
        components_added += 1
    return merged, {"added_pixels": added_pixels, "components_added": components_added}


def expanded_search_region(mask: np.ndarray, bbox_xyxy: Any | None, *, margin_ratio: float) -> np.ndarray:
    h, w = mask.shape
    y0, y1, x0, x1 = mask_bounds(mask)
    if bbox_xyxy is not None:
        try:
            bx0, by0, bx1, by1 = [float(v) for v in bbox_xyxy]
            x0 = min(x0, int(np.floor(bx0)))
            y0 = min(y0, int(np.floor(by0)))
            x1 = max(x1, int(np.ceil(bx1)))
            y1 = max(y1, int(np.ceil(by1)))
        except (TypeError, ValueError):
            pass
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    margin_x = max(1, int(round(box_w * max(0.0, margin_ratio))))
    margin_y = max(1, int(round(box_h * max(0.0, margin_ratio))))
    x0 = max(0, x0 - margin_x)
    x1 = min(w - 1, x1 + margin_x)
    y0 = max(0, y0 - margin_y)
    y1 = min(h - 1, y1 + margin_y)
    region = np.zeros((h, w), dtype=bool)
    region[y0 : y1 + 1, x0 : x1 + 1] = True
    return region


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def rgb_saturation(rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(rgb[:, :, :3], 0.0, 1.0)
    maxc = clipped.max(axis=2)
    minc = clipped.min(axis=2)
    return (maxc - minc) / np.maximum(maxc, 1e-6)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx] or not mask[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros_like(mask, dtype=bool)
        for y, x in pixels:
            component[y, x] = True
        components.append(component)
    return components


def dilate(mask: np.ndarray, passes: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = out.copy()
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = expanded
    return out


if __name__ == "__main__":
    raise SystemExit(main())
