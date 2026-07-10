#!/usr/bin/env python3
"""Expand one RGB reference image into synthetic multi-view frames.

This is an optional Real2Sim adapter. It calls an OpenAI-style image edit API
with one reference image, saves the returned contact sheet, and slices it into
ordered view images that AnySplat can consume through ``--frames-dir``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any


VIEW_NAMES = [
    "01_front_main",
    "02_front_left_45",
    "03_left_90",
    "04_rear_left_135",
    "05_rear_back",
    "06_rear_right_225",
    "07_right_270",
    "08_top_oblique",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-view frames from one image")
    parser.add_argument("--image", type=Path, required=True, help="Reference RGB image")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated metadata and contact sheet")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory where sliced views are written")
    parser.add_argument("--prompt", default="", help="Image edit prompt. A real2sim prompt is built when omitted.")
    parser.add_argument("--mode", choices=["object", "scene"], default="object", help="Default prompt mode when --prompt is omitted")
    parser.add_argument("--api-url", default=os.environ.get("REAL2SIM_IMAGE_EDIT_API_URL", "https://chatapi.applyma.com/v1/images/edits"))
    parser.add_argument("--api-key-env", default="REAL2SIM_IMAGE_EDIT_API_KEY", help="Environment variable that contains the API key")
    parser.add_argument("--model", default=os.environ.get("REAL2SIM_IMAGE_EDIT_MODEL", "gpt-image-2"))
    parser.add_argument("--size", default=os.environ.get("REAL2SIM_IMAGE_EDIT_SIZE", "1536x1024"))
    parser.add_argument("--quality", default=os.environ.get("REAL2SIM_IMAGE_EDIT_QUALITY", "auto"))
    parser.add_argument("--response-format", default="b64_json")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument(
        "--cell-crop-ratio",
        type=float,
        default=float(os.environ.get("REAL2SIM_IMAGE_EDIT_CELL_CROP_RATIO", "0.015")),
        help="Small per-cell edge crop used to remove generated contact-sheet gutters",
    )
    parser.add_argument(
        "--keep-label-bars",
        action="store_true",
        help="Keep generated panel label bars instead of automatically cropping dark bottom captions",
    )
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("REAL2SIM_IMAGE_EDIT_TIMEOUT", "300")))
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"reference image not found: {args.image}")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key: set {args.api_key_env}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.strip() or build_real2sim_multiview_prompt("the reference object", mode=args.mode)
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    payload = request_image_edit(
        api_url=args.api_url,
        api_key=api_key,
        image=args.image,
        prompt=prompt,
        model=args.model,
        size=args.size,
        quality=args.quality,
        response_format=args.response_format,
        timeout=args.timeout,
    )
    write_response_metadata(payload, args.output_dir / "response_meta.json")
    contact_sheet = args.output_dir / "contact_sheet.png"
    save_first_image_payload(payload, contact_sheet, timeout=args.timeout)
    views = split_contact_sheet(
        contact_sheet,
        args.frames_dir,
        cols=args.cols,
        rows=args.rows,
        cell_crop_ratio=args.cell_crop_ratio,
        remove_label_bars=not args.keep_label_bars,
    )
    manifest = {
        "backend": "synthetic_multiview_expand",
        "reference_image": args.image.as_posix(),
        "contact_sheet": contact_sheet.as_posix(),
        "frames_dir": args.frames_dir.as_posix(),
        "views": [path.as_posix() for path in views],
        "grid": {"cols": args.cols, "rows": args.rows},
        "cell_crop_ratio": args.cell_crop_ratio,
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def build_real2sim_multiview_prompt(subject: str, *, mode: str = "object") -> str:
    if mode == "scene":
        return (
            "Create a single photorealistic real2sim scene-completion multiview contact sheet from the reference image. "
            f"Scene identity to preserve: {subject}. Treat the reference as the canonical source view and preserve the exact visible layout, "
            "floor plane, walls, doors, windows, static furniture, fixtures, material colors, lighting direction, and object identities. "
            "Output exactly 8 equal portrait-photo panels in a clean 4 columns x 2 rows grid, filling one 1536x1024 image. "
            "Use thin white gutters between panels. No panel titles, labels, captions, numbers, UI icons, or watermarks. "
            "Views, left to right, top row then bottom row: source/front view, front-left 45 degree view, left side 90 degree view, "
            "rear-left 135 degree view, rear/back view, rear-right 225 degree view, right side 270 degree view, and top-oblique "
            "30 to 45 degree view. Infer unseen regions conservatively as physically plausible continuations of the visible scene, "
            "keeping camera height, focal length, scale, horizon, navigable ground continuity, occlusion relationships, and shadows consistent. "
            "Do not turn the scene into an isolated object turntable, do not invent a different room, and do not add people."
        )
    return (
        "Create a single photorealistic real2sim multiview capture contact sheet from the reference image. "
        f"Subject identity to preserve: {subject}. Keep the same object geometry, material, colors, labels, markings, and scale. "
        "Output exactly 8 equal portrait-photo panels in a clean 4 columns x 2 rows grid, filling one 1536x1024 image. "
        "Use thin white gutters between panels. No panel titles, labels, captions, numbers, UI icons, or watermarks. "
        "Views, left to right, top row then bottom row: front main view, front-left 45 degree view, left side 90 degree view, "
        "rear-left 135 degree view, rear/back view, rear-right 225 degree view, right side 270 degree view, and top-oblique "
        "30 to 45 degree view. Keep the object stationary on the same tabletop, consistent focal length, consistent soft lighting, "
        "sharp focus, no motion blur, no hands, no people blocking the object, no play button overlays. Keep the object fully inside "
        "every panel with 10 to 15 percent silhouette margin and preserve reconstruction-useful texture and contour details."
    )


def request_image_edit(
    *,
    api_url: str,
    api_key: str,
    image: Path,
    prompt: str,
    model: str,
    size: str,
    quality: str,
    response_format: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        raise SystemExit(f"synthetic multiview expansion requires requests: {exc}") from exc

    with image.open("rb") as image_file:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "response_format": response_format,
                "n": "1",
            },
            files=[("image", (image.name, image_file, mime_type_for_image(image)))],
            timeout=timeout,
        )
    if response.status_code >= 400:
        raise SystemExit(f"image edit API failed with status {response.status_code}: {response.text[:1000]}")
    try:
        payload = response.json()
    except Exception as exc:
        raise SystemExit(f"image edit API returned non-JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("image edit API response must be a JSON object")
    return payload


def write_response_metadata(payload: dict[str, Any], path: Path) -> None:
    redacted = {key: value for key, value in payload.items() if key != "data"}
    if isinstance(payload.get("data"), list):
        redacted["data"] = [
            {key: value for key, value in item.items() if key != "b64_json"}
            for item in payload["data"]
            if isinstance(item, dict)
        ]
    path.write_text(json.dumps(redacted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_first_image_payload(payload: dict[str, Any], output: Path, *, timeout: float) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise SystemExit("image edit API response missing data[0]")
    item = data[0]
    if item.get("b64_json"):
        output.write_bytes(base64.b64decode(str(item["b64_json"])))
        return
    if item.get("url"):
        try:
            import requests
        except Exception as exc:
            raise SystemExit(f"downloading URL response requires requests: {exc}") from exc
        response = requests.get(str(item["url"]), timeout=timeout)
        if response.status_code >= 400:
            raise SystemExit(f"failed to download generated image: HTTP {response.status_code}")
        output.write_bytes(response.content)
        return
    raise SystemExit("image edit API response data[0] must contain b64_json or url")


def split_contact_sheet(
    contact_sheet: Path,
    frames_dir: Path,
    *,
    cols: int = 4,
    rows: int = 2,
    cell_crop_ratio: float = 0.015,
    remove_label_bars: bool = True,
) -> list[Path]:
    from PIL import Image

    if cols <= 0 or rows <= 0:
        raise ValueError("contact sheet grid must have positive cols and rows")
    if not 0 <= cell_crop_ratio < 0.25:
        raise ValueError("cell_crop_ratio must be in [0, 0.25)")
    image = Image.open(contact_sheet).convert("RGB")
    width, height = image.size
    cell_w = width // cols
    cell_h = height // rows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(f"contact sheet is too small for {cols}x{rows}: {width}x{height}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    views: list[Path] = []
    for index in range(cols * rows):
        col = index % cols
        row = index // cols
        left = col * cell_w
        top = row * cell_h
        right = width if col == cols - 1 else (col + 1) * cell_w
        bottom = height if row == rows - 1 else (row + 1) * cell_h
        crop_x = int((right - left) * cell_crop_ratio)
        crop_y = int((bottom - top) * cell_crop_ratio)
        if right - left > (2 * crop_x + 8) and bottom - top > (2 * crop_y + 8):
            left += crop_x
            right -= crop_x
            top += crop_y
            bottom -= crop_y
        name = VIEW_NAMES[index] if index < len(VIEW_NAMES) else f"{index + 1:02d}_view"
        path = frames_dir / f"{name}.jpg"
        panel = image.crop((left, top, right, bottom))
        if remove_label_bars:
            panel = crop_dark_bottom_label_bar(panel)
        panel.save(path, quality=95)
        views.append(path)
    return views


def crop_dark_bottom_label_bar(image: "Image.Image") -> "Image.Image":
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = gray.shape
    if height < 80 or width < 80:
        return image

    dark_fraction = np.mean(gray < 0.10, axis=1)
    row_mean = np.mean(gray, axis=1)
    start = height
    scan_limit = int(height * 0.55)
    for row in range(height - 1, scan_limit, -1):
        if dark_fraction[row] > 0.55 and row_mean[row] < 0.22:
            start = row
            continue
        if start < height:
            break

    if start == height:
        return image

    while start > scan_limit and dark_fraction[start - 1] > 0.35 and row_mean[start - 1] < 0.35:
        start -= 1

    bar_height = height - start
    if bar_height < max(12, int(height * 0.035)):
        return image
    if start < int(height * 0.60):
        return image
    return image.crop((0, 0, width, start))


def mime_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


if __name__ == "__main__":
    raise SystemExit(main())
