"""
Runs the shared scoring harness across every method and writes the
result tables, with per-dataset stratification and pluggable scope.

Each method's predictions live under ``BENCHMARKS_RESULTS / <method> /``,
either flat (``<method>/<case_id>.json``) or per-dataset
(``<method>/<dataset>/<case_id>.json``). The aggregator probes both.

Usage:
    # Default: BERT-eligible scope, both datasets, dummy data root.
    python -m digital_registrar_research.benchmarks.eval.run_all

    # Production data (TCGA only — no CMUH splits today), FAIR_SCOPE:
    python -m digital_registrar_research.benchmarks.eval.run_all \\
        --data-root data --datasets tcga --scope fair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...paths import BENCHMARKS_RESULTS
from ..baselines._data import load_cases
from .bert_scope import bert_scope_for_organ
from .metrics import aggregate_cases_to_df, summary_table

RESULTS = BENCHMARKS_RESULTS

DEFAULT_METHODS = [
    "digital_registrar",
    "gpt4_dspy",
    "clinicalbert_merged",
    "rule_based",
]
DEFAULT_DATASETS = ["cmuh", "tcga"]
DEFAULT_ORGANS = ["breast", "colorectal", "esophagus", "liver", "stomach"]


def merge_clinicalbert_outputs(datasets: list[str]) -> None:
    """Combine ``clinicalbert_cls/<dataset>/`` and ``clinicalbert_qa/<dataset>/``
    into a single ``clinicalbert_merged/<dataset>/`` per case.

    Falls back to flat ``clinicalbert_{cls,qa}/<id>.json`` when no
    per-dataset subdirs exist (legacy layout)."""
    cls_root = RESULTS / "clinicalbert_cls"
    qa_root = RESULTS / "clinicalbert_qa"
    out_root = RESULTS / "clinicalbert_merged"
    if not cls_root.exists():
        print(f"[skip] {cls_root} not found; skip clinicalbert merge")
        return

    def _merge_one_dir(cls_dir: Path, qa_dir: Path, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for cls_file in cls_dir.glob("*.json"):
            cid = cls_file.stem
            with cls_file.open(encoding="utf-8") as f:
                a = json.load(f)
            qa_file = qa_dir / f"{cid}.json"
            if qa_file.exists():
                with qa_file.open(encoding="utf-8") as f:
                    b = json.load(f)
                a_data = a.setdefault("cancer_data", {})
                for k, v in (b.get("cancer_data") or {}).items():
                    a_data.setdefault(k, v)
            with (out_dir / f"{cid}.json").open("w", encoding="utf-8") as f:
                json.dump(a, f, ensure_ascii=False, indent=2)
            n += 1
        return n

    any_per_dataset = any((cls_root / ds).exists() for ds in datasets)
    if any_per_dataset:
        for ds in datasets:
            cls_dir = cls_root / ds
            qa_dir = qa_root / ds
            if not cls_dir.exists():
                continue
            n = _merge_one_dir(cls_dir, qa_dir, out_root / ds)
            print(f"[merged] clinicalbert {ds}: {n} cases")
    else:
        n = _merge_one_dir(cls_root, qa_root, out_root)
        print(f"[merged] clinicalbert (flat): {n} cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--out", default=str(RESULTS / "by_method.csv"))
    ap.add_argument("--data-root", default="dummy",
                    help="Root containing data/<dataset>/ subtrees (default: dummy).")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                    help="Datasets to evaluate (default: cmuh tcga).")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS),
                    help="CSV of cancer_category values to keep.")
    ap.add_argument("--scope", choices=["fair", "bert"], default="bert",
                    help="Field set to score: fair (FAIR_SCOPE + nested + biomarkers) "
                         "or bert (BERT-eligible per-organ scope).")
    args = ap.parse_args()

    merge_clinicalbert_outputs(args.datasets)

    organs = {x.strip() for x in args.organs.split(",") if x.strip()}
    cases = load_cases(
        datasets=args.datasets,
        split="test",
        root=Path(args.data_root),
        organs=organs,
    )
    if not cases:
        raise SystemExit("no test cases found — check --data-root, --datasets, --organs")
    print(f"Scoring {len(cases)} test cases across datasets={args.datasets}")

    method_to_preds = {m: (RESULTS / m) for m in args.methods
                       if (RESULTS / m).exists()}
    missing = set(args.methods) - set(method_to_preds)
    if missing:
        print(f"[warn] skipping methods with no predictions: {sorted(missing)}")

    scope = bert_scope_for_organ if args.scope == "bert" else None
    df = aggregate_cases_to_df(cases, method_to_preds, scope=scope)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    summary = summary_table(df, by=["method", "dataset", "field"])
    summary_path = out_csv.with_name("summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    pivot = summary.pivot_table(
        index="field", columns=["method", "dataset"],
        values="accuracy_attempted", aggfunc="first",
    )
    pivot.to_csv(out_csv.with_name("comparison_table.csv"))
    print(pivot.to_string())


if __name__ == "__main__":
    main()
