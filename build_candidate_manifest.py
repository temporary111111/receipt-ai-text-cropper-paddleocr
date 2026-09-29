from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic balanced receipt candidate manifest.")
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=650)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    if args.per_class <= 0:
        raise SystemExit("--per-class must be positive")

    groups = [
        ("ai_generated", 1, args.fake_root),
        ("non_ai_generated", 0, args.real_root),
    ]
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []

    for class_name, label, root in groups:
        files = image_files(root)
        if len(files) < args.per_class:
            raise SystemExit(f"{root} contains {len(files)} images; need {args.per_class}")
        shuffled = files[:]
        rng.shuffle(shuffled)
        selected = sorted(shuffled[: args.per_class], key=lambda p: p.name.lower())
        for rank, path in enumerate(selected, start=1):
            rows.append(
                {
                    "candidate_id": f"{class_name}_{rank:04d}",
                    "class_name": class_name,
                    "label": label,
                    "source_image": path.name,
                    "source_path": str(path.resolve()),
                    "candidate_rank": rank,
                    "selection_seed": args.seed,
                    "selected_for_pool": 1,
                    "review_status": "pending",
                }
            )

    rows.sort(key=lambda row: (int(row["label"]), int(row["candidate_rank"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"created={len(rows)} rows")
    print(f"ai_generated={sum(row['label'] == 1 for row in rows)}")
    print(f"non_ai_generated={sum(row['label'] == 0 for row in rows)}")
    print(f"seed={args.seed}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
