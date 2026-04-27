"""
Head-to-head paired comparison between two methods (e.g. ClinicalBERT vs LLM).

For every (dataset, organ, field) cell:
  * acc_a, acc_b with 95% Wilson CIs (per-method marginal accuracy)
  * delta = acc_a - acc_b with 95% paired-bootstrap CI
  * McNemar p-value on the paired binary disagreements
  * coverage rates (% of cases each method attempted)

The paired contract: only count cases where BOTH methods attempted the
field. That keeps the comparison honest — neither method gets credit
nor penalty for not having tried.

Usage:
    python -m digital_registrar_research.benchmarks.eval.pairwise_compare \\
        --a clinicalbert_merged --b gpt4_dspy \\
        --scope bert --datasets cmuh tcga \\
        --n-boot 2000 --stratify-by organ
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ...paths import BENCHMARKS_RESULTS
from ..baselines._data import load_cases
from .bert_scope import bert_scope_for_organ
from .ci import mcnemar_test, paired_bootstrap_diff, wilson_ci
from .metrics import aggregate_cases_to_df

DEFAULT_DATASETS = ["cmuh", "tcga"]
DEFAULT_ORGANS = ["breast", "colorectal", "esophagus", "liver", "stomach"]


def _wide_paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two long-form correctness frames on (case, field)."""
    keep = ["case_id", "dataset", "organ", "field", "correct", "attempted"]
    a = df_a[keep].rename(columns={"correct": "correct_a", "attempted": "attempted_a"})
    b = df_b[keep].rename(columns={"correct": "correct_b", "attempted": "attempted_b"})
    merged = a.merge(b, on=["case_id", "dataset", "organ", "field"], how="inner")
    return merged


def _cell_stats(cell: pd.DataFrame, n_boot: int, stratify_by: str | None) -> dict:
    """Compute paired stats for one (dataset, organ, field) slice."""
    n_total = len(cell)
    paired_mask = cell["attempted_a"] & cell["attempted_b"]
    paired = cell[paired_mask]
    n_paired = len(paired)

    out = {
        "n_total": n_total,
        "n_paired": n_paired,
        "attempted_a_pct": float(cell["attempted_a"].mean()) if n_total else float("nan"),
        "attempted_b_pct": float(cell["attempted_b"].mean()) if n_total else float("nan"),
    }

    if n_paired == 0:
        for k in ("acc_a", "acc_a_lo", "acc_a_hi",
                  "acc_b", "acc_b_lo", "acc_b_hi",
                  "delta", "delta_lo", "delta_hi",
                  "mcnemar_b", "mcnemar_c", "mcnemar_p"):
            out[k] = float("nan")
        return out

    a_correct = paired["correct_a"].astype(float).to_numpy()
    b_correct = paired["correct_b"].astype(float).to_numpy()

    k_a = int(a_correct.sum())
    k_b = int(b_correct.sum())
    out["acc_a"] = k_a / n_paired
    out["acc_b"] = k_b / n_paired
    out["acc_a_lo"], out["acc_a_hi"] = wilson_ci(k_a, n_paired)
    out["acc_b_lo"], out["acc_b_hi"] = wilson_ci(k_b, n_paired)

    # Delta with paired bootstrap. Stratify_by would require sampling
    # within stratum — we already split per cell, so the cell IS the
    # stratum; pass None to paired_bootstrap_diff (it pairs by index).
    boot = paired_bootstrap_diff(a_correct, b_correct, n_boot=n_boot)
    out["delta"], out["delta_lo"], out["delta_hi"] = boot.point, boot.lo, boot.hi

    # McNemar table:
    # b = a correct, b wrong; c = a wrong, b correct
    b_count = int(((a_correct == 1) & (b_correct == 0)).sum())
    c_count = int(((a_correct == 0) & (b_correct == 1)).sum())
    mc = mcnemar_test(b_count, c_count)
    out["mcnemar_b"] = b_count
    out["mcnemar_c"] = c_count
    out["mcnemar_p"] = mc["p_value"]
    return out


def pairwise_table(
    method_a: str, method_b: str,
    cases: list[dict],
    method_to_preds: dict[str, Path],
    scope=bert_scope_for_organ,
    n_boot: int = 2000,
    stratify_by: str | None = "organ",
) -> pd.DataFrame:
    df_a = aggregate_cases_to_df(
        cases, {method_a: method_to_preds[method_a]}, scope=scope)
    df_b = aggregate_cases_to_df(
        cases, {method_b: method_to_preds[method_b]}, scope=scope)
    wide = _wide_paired(df_a, df_b)

    rows = []
    # Per (dataset, organ, field) cell.
    for (ds, org, field), cell in wide.groupby(["dataset", "organ", "field"]):
        stats = _cell_stats(cell, n_boot, stratify_by)
        rows.append({"dataset": ds, "organ": org, "field": field,
                     "method_a": method_a, "method_b": method_b, **stats})

    # Per (dataset, field) cell — pooled across organs (ALL).
    for (ds, field), cell in wide.groupby(["dataset", "field"]):
        stats = _cell_stats(cell, n_boot, stratify_by)
        rows.append({"dataset": ds, "organ": "ALL", "field": field,
                     "method_a": method_a, "method_b": method_b, **stats})

    # Per dataset (overall, ALL fields).
    for ds, cell in wide.groupby("dataset"):
        stats = _cell_stats(cell, n_boot, stratify_by)
        rows.append({"dataset": ds, "organ": "ALL", "field": "ALL",
                     "method_a": method_a, "method_b": method_b, **stats})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Method A name (e.g. clinicalbert_merged)")
    ap.add_argument("--b", required=True, help="Method B name (e.g. gpt4_dspy)")
    ap.add_argument("--data-root", default="dummy")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--scope", choices=["fair", "bert"], default="bert")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--stratify-by", default="organ", choices=["organ", "none"])
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: pairwise_<a>_vs_<b>.csv).")
    args = ap.parse_args()

    organs = {x.strip() for x in args.organs.split(",") if x.strip()}
    cases = load_cases(
        datasets=args.datasets, split="test",
        root=Path(args.data_root), organs=organs,
    )
    if not cases:
        raise SystemExit("no test cases — check --data-root, --datasets, --organs")

    pred_root_a = BENCHMARKS_RESULTS / args.a
    pred_root_b = BENCHMARKS_RESULTS / args.b
    for name, p in [(args.a, pred_root_a), (args.b, pred_root_b)]:
        if not p.exists():
            raise SystemExit(f"no predictions found at {p} for method {name!r}")

    scope = bert_scope_for_organ if args.scope == "bert" else None
    stratify_by = None if args.stratify_by == "none" else args.stratify_by

    df = pairwise_table(
        args.a, args.b, cases,
        {args.a: pred_root_a, args.b: pred_root_b},
        scope=scope, n_boot=args.n_boot, stratify_by=stratify_by,
    )

    out_path = Path(args.out) if args.out else (
        BENCHMARKS_RESULTS / f"pairwise_{args.a}_vs_{args.b}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")

    # Print the headline ALL-organs ALL-fields rows for quick scan.
    headline = df[(df["organ"] == "ALL") & (df["field"] == "ALL")][
        ["dataset", "n_paired", "acc_a", "acc_b", "delta",
         "delta_lo", "delta_hi", "mcnemar_p"]
    ]
    print("\nHeadline (ALL organs, ALL fields):")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
