#!/usr/bin/env python3
"""Side-by-side comparison of non_nested eval outputs across methods.

Joins per-method ``correctness_table.parquet`` files (produced by
``scripts.eval.cli non_nested``) into a single wide table indexed on
``(case_id, organ, field)``, then emits per-field accuracy / coverage,
pairwise paired-bootstrap deltas with Wilson CIs and McNemar p-values,
and a headline summary across all fields.

Designed so the rule, BERT, and LLM baselines can be compared like-for-
like — all consume the same canonical predictions tree and the same
shared eval primitives, so the only source of variation is the method
itself.

Usage
-----
    python -m scripts.eval.compare.run_compare \\
        --inputs rule_based:workspace/results/eval/non_nested_rule \\
                 clinicalbert:workspace/results/eval/non_nested_bert_merged \\
                 llm:workspace/results/eval/non_nested_llm_gptoss \\
        --out workspace/results/eval/compare/rule_bert_llm

Each input is ``LABEL:DIR`` where ``DIR`` contains a
``correctness_table.parquet`` file. Labels are arbitrary; they appear
in the output column names.

Output tree
-----------
    {--out}/
        manifest.json
        wide.csv               row per (case_id, organ, field); columns per label
        per_field.csv          long form: (label, organ, field) → accuracy, coverage
        pairwise.csv           every label pair × (organ, field) → delta + CI + p
        headline.csv           summary across all fields per (label[, dataset])
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Opt into the future fillna behavior to silence the FutureWarning emitted
# by ``.fillna(0).astype(int)`` on object-dtype columns holding bool/None.
pd.set_option("future.no_silent_downcasting", True)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logger = logging.getLogger("scripts.eval.compare")


# --- IO ---------------------------------------------------------------------


def parse_inputs(specs: list[str]) -> dict[str, Path]:
    """Parse ``LABEL:DIR`` specs into ``{label: Path(parquet_file)}``."""
    out: dict[str, Path] = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(
                f"--inputs must be LABEL:DIR (got {spec!r}). DIR is the "
                f"non_nested output directory containing correctness_table.parquet."
            )
        label, raw = spec.split(":", 1)
        label = label.strip()
        d = Path(raw.strip())
        parquet = d / "correctness_table.parquet"
        if not parquet.is_file():
            raise SystemExit(f"missing correctness_table.parquet under {d} (resolved: {parquet})")
        if label in out:
            raise SystemExit(f"duplicate label {label!r} in --inputs")
        out[label] = parquet
    if len(out) < 2:
        raise SystemExit("--inputs needs at least two LABEL:DIR specs to compare")
    return out


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Stats primitives -------------------------------------------------------


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 95% CI for a proportion. Returns (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def _mcnemar(b: int, c: int) -> float:
    """Two-sided McNemar p-value for paired binary outcomes (b, c are
    the discordant cell counts).

    Uses exact binomial when b+c is small, normal approximation
    otherwise. Returns 1.0 when b+c==0 (no disagreement).
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n=n, p=0.5).pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    return float(1.0 - chi2_dist.cdf(chi2, df=1))


def _paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha: float,
) -> tuple[float, float, float]:
    """Bootstrap (mean(a) - mean(b)) and return (delta, lo, hi)."""
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return (float(a.mean() - b.mean()), lo, hi)


# --- Aggregation ------------------------------------------------------------


def _load_atomic(label: str, parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    keep = ["case_id", "organ", "organ_idx", "field", "subgroup",
            "attempted", "correct", "wrong", "field_missing",
            "parse_error", "gold_value", "pred_value", "run_id", "method"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["label"] = label
    return df


def build_wide(per_label: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivot per-label atomic tables into a wide (case_id, organ, field) table.

    Multi-run methods (LLM with multiple run_ids) are collapsed by
    majority vote on ``correct``: if any run for a (case, field) was
    correct, the cell is marked correct (lenient). Use the long
    per-field tables for run-level granularity.
    """
    keys = ["case_id", "organ", "field"]
    out = None
    for label, df in per_label.items():
        # Collapse runs: per (case, organ, field), take the "best" outcome.
        agg = df.groupby(keys, as_index=False).agg({
            "attempted": "max",
            "correct": "max",
            "pred_value": "first",
        })
        agg = agg.rename(columns={
            "attempted": f"{label}_attempted",
            "correct": f"{label}_correct",
            "pred_value": f"{label}_pred",
        })
        out = agg if out is None else out.merge(agg, on=keys, how="outer")
    return out.sort_values(keys).reset_index(drop=True)


def per_field_long(per_label: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-form per (label, organ, field) accuracy + coverage."""
    rows = []
    for label, df in per_label.items():
        # Collapse runs first.
        keys = ["case_id", "organ", "field"]
        collapsed = df.groupby(keys, as_index=False).agg({
            "attempted": "max",
            "correct": "max",
        })
        for (organ, field), g in collapsed.groupby(["organ", "field"]):
            n = len(g)
            attempted = int(g["attempted"].sum())
            correct = int(g["correct"].sum())
            cov = attempted / n if n else float("nan")
            acc_attempted = correct / attempted if attempted else float("nan")
            acc_lo, acc_hi = (
                _wilson_ci(correct, attempted) if attempted else (float("nan"),) * 2
            )
            rows.append({
                "label": label, "organ": organ, "field": field,
                "n_cases": n, "n_attempted": attempted, "n_correct": correct,
                "coverage": cov, "accuracy_attempted": acc_attempted,
                "ci_lo": acc_lo, "ci_hi": acc_hi,
            })
    return pd.DataFrame(rows)


def pairwise_table(
    wide: pd.DataFrame, labels: list[str], n_boot: int, seed: int, alpha: float,
) -> pd.DataFrame:
    rows = []
    for i, a_label in enumerate(labels):
        for b_label in labels[i + 1:]:
            mask = (wide[f"{a_label}_attempted"] == True) & (  # noqa: E712
                wide[f"{b_label}_attempted"] == True              # noqa: E712
            )
            for (organ, field), g in wide[mask].groupby(["organ", "field"]):
                a_correct = g[f"{a_label}_correct"].fillna(0).infer_objects(copy=False).astype(int).to_numpy()
                b_correct = g[f"{b_label}_correct"].fillna(0).infer_objects(copy=False).astype(int).to_numpy()
                n = len(g)
                if n == 0:
                    continue
                delta, ci_lo, ci_hi = _paired_bootstrap_delta(
                    a_correct, b_correct, n_boot=n_boot, seed=seed, alpha=alpha,
                )
                discord_a_only = int(((a_correct == 1) & (b_correct == 0)).sum())
                discord_b_only = int(((a_correct == 0) & (b_correct == 1)).sum())
                p = _mcnemar(discord_a_only, discord_b_only)
                rows.append({
                    "a_label": a_label, "b_label": b_label,
                    "organ": organ, "field": field,
                    "n_paired": n,
                    "acc_a": float(a_correct.mean()),
                    "acc_b": float(b_correct.mean()),
                    "delta": delta, "delta_lo": ci_lo, "delta_hi": ci_hi,
                    "discord_a_only": discord_a_only,
                    "discord_b_only": discord_b_only,
                    "mcnemar_p": p,
                })
            # ALL-fields rollup row for this pair.
            mask_all = (wide[f"{a_label}_attempted"] == True) & (  # noqa: E712
                wide[f"{b_label}_attempted"] == True                  # noqa: E712
            )
            ga = wide.loc[mask_all, f"{a_label}_correct"].fillna(0).infer_objects(copy=False).astype(int).to_numpy()
            gb = wide.loc[mask_all, f"{b_label}_correct"].fillna(0).infer_objects(copy=False).astype(int).to_numpy()
            if ga.size:
                delta, ci_lo, ci_hi = _paired_bootstrap_delta(
                    ga, gb, n_boot=n_boot, seed=seed, alpha=alpha,
                )
                rows.append({
                    "a_label": a_label, "b_label": b_label,
                    "organ": "ALL", "field": "ALL",
                    "n_paired": int(ga.size),
                    "acc_a": float(ga.mean()),
                    "acc_b": float(gb.mean()),
                    "delta": delta, "delta_lo": ci_lo, "delta_hi": ci_hi,
                    "discord_a_only": int(((ga == 1) & (gb == 0)).sum()),
                    "discord_b_only": int(((ga == 0) & (gb == 1)).sum()),
                    "mcnemar_p": _mcnemar(
                        int(((ga == 1) & (gb == 0)).sum()),
                        int(((ga == 0) & (gb == 1)).sum()),
                    ),
                })
    return pd.DataFrame(rows)


def headline_table(per_field: pd.DataFrame) -> pd.DataFrame:
    """Per-method aggregate across all (organ, field) cells."""
    rows = []
    for label, g in per_field.groupby("label"):
        n_cases = int(g["n_cases"].sum())
        n_attempted = int(g["n_attempted"].sum())
        n_correct = int(g["n_correct"].sum())
        cov = n_attempted / n_cases if n_cases else float("nan")
        acc = n_correct / n_attempted if n_attempted else float("nan")
        lo, hi = (
            _wilson_ci(n_correct, n_attempted) if n_attempted else (float("nan"),) * 2
        )
        rows.append({
            "label": label,
            "n_cells_total": n_cases,
            "n_cells_attempted": n_attempted,
            "n_cells_correct": n_correct,
            "coverage": cov,
            "accuracy_attempted": acc,
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values("accuracy_attempted", ascending=False)


# --- Entry point ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="Method specs as LABEL:DIR (>=2). DIR contains correctness_table.parquet.",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="Output directory.",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    inputs = parse_inputs(args.inputs)
    logger.info("loading %d inputs: %s", len(inputs), list(inputs.keys()))
    per_label = {label: _load_atomic(label, p) for label, p in inputs.items()}
    for label, df in per_label.items():
        logger.info("  %s: %d atomic rows", label, len(df))

    args.out.mkdir(parents=True, exist_ok=True)

    wide = build_wide(per_label)
    wide.to_csv(args.out / "wide.csv", index=False)
    logger.info("wrote wide.csv (%d rows × %d cols)", len(wide), len(wide.columns))

    per_field = per_field_long(per_label)
    per_field.to_csv(args.out / "per_field.csv", index=False)
    logger.info("wrote per_field.csv (%d rows)", len(per_field))

    labels = list(inputs.keys())
    pairwise = pairwise_table(
        wide, labels, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha,
    )
    pairwise.to_csv(args.out / "pairwise.csv", index=False)
    logger.info("wrote pairwise.csv (%d rows)", len(pairwise))

    headline = headline_table(per_field)
    headline.to_csv(args.out / "headline.csv", index=False)
    logger.info("wrote headline.csv:\n%s", headline.to_string(index=False))

    manifest = {
        "subcommand": "compare",
        "created_at": _utc_now_iso(),
        "inputs": {label: str(p) for label, p in inputs.items()},
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "seed": args.seed,
        "outputs": {
            "wide": str(args.out / "wide.csv"),
            "per_field": str(args.out / "per_field.csv"),
            "pairwise": str(args.out / "pairwise.csv"),
            "headline": str(args.out / "headline.csv"),
        },
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
