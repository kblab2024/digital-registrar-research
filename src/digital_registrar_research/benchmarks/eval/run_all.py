"""
Runs the shared scoring harness across every method and writes the
result tables.

Assumes each method has already produced per-case JSON predictions in
`../results/<method>/<case_id>.json`. This script just aggregates.

Usage:
    python eval/run_all.py
    python eval/run_all.py --methods digital_registrar gpt4_dspy rules
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import aggregate_to_csv, summary_table
from ...paths import BENCHMARKS_RESULTS, GOLD_ANNOTATIONS, SPLITS_JSON

RESULTS = BENCHMARKS_RESULTS
SPLITS = SPLITS_JSON
GOLD_ROOT = GOLD_ANNOTATIONS

DEFAULT_METHODS = [
    "digital_registrar",   # baseline (reuse existing runs)
    "gpt4_dspy",
    "clinicalbert_merged", # clinicalbert_cls + clinicalbert_qa merged offline
    "rules",
]


def merge_clinicalbert_outputs() -> None:
    """Combine the categorical (cls) and span (qa) ClinicalBERT outputs into
    a single `clinicalbert_merged/` folder per case id. Preserves both."""
    cls_dir = RESULTS / "clinicalbert_cls"
    qa_dir = RESULTS / "clinicalbert_qa"
    out = RESULTS / "clinicalbert_merged"
    if not cls_dir.exists() or not qa_dir.exists():
        print("[skip] clinicalbert_{cls,qa} outputs not found; skip merge")
        return
    out.mkdir(parents=True, exist_ok=True)

    import json
    for cls_file in cls_dir.glob("*.json"):
        cid = cls_file.stem
        with cls_file.open(encoding="utf-8") as f:
            a = json.load(f)
        qa_file = qa_dir / f"{cid}.json"
        if qa_file.exists():
            with qa_file.open(encoding="utf-8") as f:
                b = json.load(f)
            # QA populates only cancer_data scalars — merge those in.
            a_data = a.setdefault("cancer_data", {})
            for k, v in (b.get("cancer_data") or {}).items():
                a_data.setdefault(k, v)
        with (out / f"{cid}.json").open("w", encoding="utf-8") as f:
            json.dump(a, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--out", default=str(RESULTS / "by_method.csv"))
    args = ap.parse_args()

    merge_clinicalbert_outputs()

    method_to_preds = {m: (RESULTS / m) for m in args.methods
                       if (RESULTS / m).exists()}
    missing = set(args.methods) - set(method_to_preds)
    if missing:
        print(f"[warn] skipping methods with no predictions: {sorted(missing)}")

    df = aggregate_to_csv(method_to_preds, GOLD_ROOT, SPLITS, Path(args.out))
    print(f"Wrote {args.out} ({len(df)} rows)")

    summary = summary_table(df)
    summary_path = Path(args.out).with_name("summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Pivot to a publication-ready per-field × per-method table.
    pivot = summary.pivot_table(
        index="field", columns="method",
        values="accuracy_attempted", aggfunc="first",
    )
    pivot.to_csv(Path(args.out).with_name("comparison_table.csv"))
    print(pivot.to_string())


if __name__ == "__main__":
    main()
