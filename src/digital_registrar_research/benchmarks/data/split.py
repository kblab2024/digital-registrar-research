"""
Deterministic stratified 100/51 split of the 151 doctor-validated gold
annotations.

Stratifies by cancer_category so every organ appears in both folds.
Writes splits.json next to this file.

Usage:
    python data/split.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from ...paths import GOLD_ANNOTATIONS, RAW_REPORTS

GOLD_ROOT = GOLD_ANNOTATIONS
REPORT_ROOT = RAW_REPORTS
OUT_PATH = Path(__file__).parent / "splits.json"

TEST_SIZE = 51
SEED = 20251117


def locate_report(stem: str) -> Path | None:
    """stem = 'tcga1_10_annotation' -> find tcga1_10.txt under tcga_dataset_.../tcga1/"""
    base = stem.replace("_annotation", "")
    prefix = base.split("_")[0]
    candidate = REPORT_ROOT / prefix / f"{base}.txt"
    return candidate if candidate.exists() else None


def load_cases() -> list[dict]:
    cases = []
    for ann_file in GOLD_ROOT.rglob("*_annotation.json"):
        with ann_file.open(encoding="utf-8") as f:
            ann = json.load(f)
        report_path = locate_report(ann_file.stem)
        if report_path is None:
            print(f"[warn] no matching report for {ann_file}")
            continue
        cases.append({
            "id": ann_file.stem.replace("_annotation", ""),
            "annotation_path": str(ann_file),
            "report_path": str(report_path),
            "cancer_category": ann.get("cancer_category") or "null",
        })
    return cases


def stratified_split(cases: list[dict], test_size: int, seed: int) -> dict:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["cancer_category"]].append(c)

    total = len(cases)
    train, test = [], []
    for _cat, items in by_cat.items():
        rng.shuffle(items)
        # proportional allocation to test
        n_test_cat = max(1, round(len(items) * test_size / total)) if len(items) >= 2 else 0
        test.extend(items[:n_test_cat])
        train.extend(items[n_test_cat:])

    # Adjust to hit exact test_size (trim/pad from largest category)
    while len(test) > test_size:
        largest_cat = max(
            {c["cancer_category"] for c in test},
            key=lambda k: sum(1 for c in test if c["cancer_category"] == k),
        )
        for i, c in enumerate(test):
            if c["cancer_category"] == largest_cat:
                train.append(test.pop(i))
                break
    while len(test) < test_size:
        largest_cat = max(
            {c["cancer_category"] for c in train},
            key=lambda k: sum(1 for c in train if c["cancer_category"] == k),
        )
        for i, c in enumerate(train):
            if c["cancer_category"] == largest_cat:
                test.append(train.pop(i))
                break

    return {"train": train, "test": test, "seed": seed, "total": total}


def main() -> None:
    cases = load_cases()
    print(f"Loaded {len(cases)} gold cases")
    split = stratified_split(cases, TEST_SIZE, SEED)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"Wrote split to {OUT_PATH}")
    print(f"  train: {len(split['train'])}   test: {len(split['test'])}")

    def dist(fold):
        d: dict[str, int] = defaultdict(int)
        for c in fold:
            d[c["cancer_category"]] += 1
        return dict(sorted(d.items()))

    print(f"  train dist: {dist(split['train'])}")
    print(f"  test dist:  {dist(split['test'])}")


if __name__ == "__main__":
    main()
