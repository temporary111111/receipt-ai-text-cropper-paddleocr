"""Run a detector-only text-crop pilot on fake and real receipt images.

This is an evaluation utility, not a replacement for the current cropper.
It uses PaddleOCR text detection only; it does not run OCR recognition.
Every detected region is saved with a manifest row and visual overlay so the
post-processing can be reviewed before any pipeline decision is made.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import TextDetection


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmax(diffs)]
    ordered[3] = points[np.argmin(diffs)]
    return ordered


def warp_polygon(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rect = order_points(polygon)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(2, int(round(max(width_a, width_b))))
    height = max(2, int(round(max(height_a, height_b))))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def tight_crop(warped: np.ndarray, padding_ratio: float = 0.12) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray.std() < 3:
        return warped, "uniform"
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return warped, "no_ink"
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ink_ratio = float(np.mean(mask > 0))
    if ink_ratio > 0.65:
        return warped, "dense_fallback"
    pad = max(2, int(round(max(y1 - y0 + 1, 1) * padding_ratio)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(warped.shape[1] - 1, x1 + pad)
    y1 = min(warped.shape[0] - 1, y1 + pad)
    return warped[y0 : y1 + 1, x0 : x1 + 1], "otsu_ink_bounds"


def barcode_candidate(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h == 0 or w == 0 or w / h < 5.0:
        return False
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    column_density = (mask > 0).mean(axis=0)
    alternating = np.mean(np.abs(np.diff(column_density)) > 0.18) if w > 1 else 0.0
    # Keep this deliberately conservative.  Long receipt text lines often
    # have a high aspect ratio too, so weak barcode signals must stay review
    # flags rather than causing automatic crop rejection.
    return bool(alternating > 0.30 and column_density.mean() > 0.12)


def contact_sheet(crops: list[tuple[str, np.ndarray, bool]], output: Path) -> None:
    thumb_w, thumb_h = 300, 110
    columns = 3
    rows = max(1, math.ceil(len(crops) / columns))
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, crop, accepted) in enumerate(crops):
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image.thumbnail((thumb_w - 10, thumb_h - 28))
        x = (index % columns) * thumb_w
        y = (index // columns) * thumb_h
        sheet.paste(image, (x + 5, y + 22))
        label = f"{name} {'REVIEW' if accepted else ''}".strip()
        draw.text((x + 5, y + 4), label, fill="red" if accepted else "black")
    sheet.save(output, quality=95)


def process_image(detector: TextDetection, image_path: Path, output_dir: Path) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    result = detector.predict(str(image_path))[0]
    polygons = np.asarray(result["dt_polys"])
    scores = list(result["dt_scores"])
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    crops_for_sheet: list[tuple[str, np.ndarray, bool]] = []
    rows: list[dict[str, object]] = []
    for index, (polygon, score) in enumerate(zip(polygons, scores), start=1):
        polygon = np.asarray(polygon, dtype=np.float32)
        warped = warp_polygon(image, polygon)
        crop, trim_mode = tight_crop(warped)
        barcode = barcode_candidate(crop)
        # Detection crops remain available by default.  The barcode heuristic
        # is only a review flag because long text lines can look barcode-like.
        accepted = True
        crop_id = f"crop_{index:03d}"
        crop_path = output_dir / f"{crop_id}.png"
        cv2.imwrite(str(crop_path), crop)
        color = (0, 0, 255) if barcode else (40, 180, 40)
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, color, 2)
        anchor = tuple(polygon[0].astype(int))
        cv2.putText(overlay, str(index), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        crops_for_sheet.append((crop_id, crop, barcode))
        rows.append(
            {
                "source_image": image_path.name,
                "crop_id": crop_id,
                "score": float(score),
                "source_polygon": polygon.astype(int).tolist(),
                "crop_file": str(crop_path),
                "trim_mode": trim_mode,
                "barcode_candidate": int(barcode),
                "accepted_default": int(accepted),
            }
        )
    cv2.imwrite(str(output_dir / "source_with_boxes.jpg"), overlay)
    contact_sheet(crops_for_sheet, output_dir / "contact_sheet.jpg")
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_image",
            "crop_id",
            "score",
            "source_polygon",
            "crop_file",
            "trim_mode",
            "barcode_candidate",
            "accepted_default",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_complete_rows(output_dir: Path) -> list[dict[str, str]] | None:
    """Return saved rows only when a receipt output is complete and reusable."""
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.is_file():
        return None
    if not (output_dir / "source_with_boxes.jpg").is_file():
        return None
    if not (output_dir / "contact_sheet.jpg").is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    required = {
        "source_image", "crop_id", "score", "source_polygon", "crop_file",
        "trim_mode", "barcode_candidate", "accepted_default",
    }
    if rows and not required.issubset(rows[0]):
        return None
    if any(not (output_dir / f"{row.get('crop_id', '')}.png").is_file() for row in rows):
        return None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-count", type=int, default=10)
    parser.add_argument("--real-count", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or gpu:0.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete receipt outputs and process only missing/incomplete receipts.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit("Output directory is not empty. Use --resume to keep completed receipts.")

    aggregate: list[dict[str, object]] = []
    targets: list[tuple[str, Path, int, Path, list[dict[str, str]] | None]] = []
    groups = [("fake", args.fake_root, 1, args.fake_count), ("real", args.real_root, 0, args.real_count)]
    for group, root, label, count in groups:
        images = sorted(root.glob("*.png" if group == "fake" else "*.jpg"))[:count]
        for image_path in images:
            receipt_output = args.output_dir / group / image_path.stem
            saved = load_complete_rows(receipt_output) if args.resume else None
            targets.append((group, image_path, label, receipt_output, saved))

    pending = [target for target in targets if target[4] is None]
    detector = None
    if pending:
        detector = TextDetection(
            model_name="PP-OCRv5_server_det",
            engine="paddle_static",
            device=args.device,
            enable_mkldnn=False,
        )
    processed = 0
    skipped = 0
    try:
        for index, (group, image_path, label, receipt_output, saved) in enumerate(targets, start=1):
            if saved is not None:
                rows: list[dict[str, object]] = list(saved)
                skipped += 1
                action = "skipped"
            else:
                if detector is None:
                    raise RuntimeError("Detector was not initialized for pending work")
                rows = process_image(detector, image_path, receipt_output)
                processed += 1
                action = "processed"
            for row in rows:
                row.update({"group": group, "label": label})
            aggregate.extend(rows)
            print(f"[{index}/{len(targets)}] {action} {group}/{image_path.name} crops={len(rows)}", flush=True)
    finally:
        if detector is not None:
            detector.close()
    aggregate_path = args.output_dir / "aggregate_manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["group", "label"] + [
            "source_image",
            "crop_id",
            "score",
            "source_polygon",
            "crop_file",
            "trim_mode",
            "barcode_candidate",
            "accepted_default",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    print(
        f"images={len(targets)} processed={processed} skipped={skipped} "
        f"detections={len(aggregate)}"
    )


if __name__ == "__main__":
    main()
