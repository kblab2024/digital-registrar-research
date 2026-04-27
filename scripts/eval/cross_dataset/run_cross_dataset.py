"""Cross-dataset comparison.

Takes two ``--non-nested-out`` directories and emits per-field Δ +
distribution-shift indicators.

Output tree:
    manifest.json
    per_field_delta.csv
    distribution_shift.csv
    transferability.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.ci import paired_bootstrap_diff

from .._common.reporting import setup_logging, write_csv, write_manifest
from .._common.stats_extra import (
    jensen_shannon, kl_divergence, wasserstein,
)

logger = logging.getLogger("scripts.eval.cross_dataset")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cross_dataset",
        help="Cross-dataset Δ accuracy + distribution-shift indicators.",
        description=__doc__,
    )
    parser.add_argument(
        "--left", type=Path, required=True,
        help="Left non_nested output directory (typically CMUH).",
    )
    parser.add_argument(
        "--right", type=Path, required=True,
        help="Right non_nested output directory (typically TCGA).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=2000,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    parser.set_defaults(_handler=_main)


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    args.out.mkdir(parents=True, exist_ok=True)

    left_atomic = pd.read_parquet(args.left / "correctness_table.parquet")
    right_atomic = pd.read_parquet(args.right / "correctness_table.parquet")
    left_label = _label(args.left)
    right_label = _label(args.right)

    # --- Per-field Δ ---------------------------------------------------
    per_field_delta = _per_field_delta(
        left_atomic, right_atomic,
        left_label=left_label, right_label=right_label,
        n_boot=args.n_boot, seed=args.seed,
    )
    write_csv(per_field_delta, args.out / "per_field_delta.csv")

    # --- Transferability ------------------------------------------------
    trans = _transferability(per_field_delta)
    write_csv(trans, args.out / "transferability.csv")

    # --- Distribution shift --------------------------------------------
    shift = _distribution_shift(left_atomic, right_atomic,
                                left_label=left_label, right_label=right_label)
    write_csv(shift, args.out / "distribution_shift.csv")

    write_manifest(
        args.out, args, subcommand="cross_dataset",
        extra={
            "left": str(args.left), "right": str(args.right),
            "left_label": left_label, "right_label": right_label,
            "n_left_rows": int(len(left_atomic)),
            "n_right_rows": int(len(right_atomic)),
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


def _label(path: Path) -> str:
    """Infer a short label from the manifest, falling back to the dir name."""
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        import json
        try:
            with manifest_path.open(encoding="utf-8") as f:
                m = json.load(f)
            return str(m.get("args", {}).get("dataset") or path.name)
        except Exception:
            pass
    return path.name


def _per_field_delta(
    left: pd.DataFrame, right: pd.DataFrame,
    *, left_label: str, right_label: str,
    n_boot: int, seed: int,
) -> pd.DataFrame:
    """Per (organ, field) Δ-accuracy with bootstrap CI on the difference.

    Note: cases are NOT paired across datasets (different patients), so
    we use independent bootstrap on each side and report the Δ of means
    with the unpaired-bootstrap CI on the difference.
    """
    rows: list[dict] = []
    fields = sorted(set(left["field"].dropna()) & set(right["field"].dropna()))
    organs = sorted(set(left["organ"].dropna()) & set(right["organ"].dropna())) + ["ALL"]
    for organ in organs:
        for field in fields:
            l = _accuracy_vec(left, organ=organ, field=field)
            r = _accuracy_vec(right, organ=organ, field=field)
            if l.size == 0 or r.size == 0:
                continue
            mean_l = float(l.mean())
            mean_r = float(r.mean())
            delta = mean_l - mean_r
            # Bootstrap with independent resamples of equal size.
            rng = np.random.default_rng(seed)
            min_n = min(l.size, r.size)
            boot = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                idx_l = rng.integers(0, l.size, size=min_n)
                idx_r = rng.integers(0, r.size, size=min_n)
                boot[i] = float(l[idx_l].mean() - r[idx_r].mean())
            lo = float(np.quantile(boot, 0.025))
            hi = float(np.quantile(boot, 0.975))
            rows.append({
                "organ": organ, "field": field,
                "left_label": left_label, "right_label": right_label,
                "n_left": int(l.size), "n_right": int(r.size),
                "left_accuracy": mean_l,
                "right_accuracy": mean_r,
                "delta": delta,
                "delta_ci_lo": lo, "delta_ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _accuracy_vec(df: pd.DataFrame, *, organ: str, field: str) -> np.ndarray:
    sub = df[(df["field"] == field) & df["attempted"]]
    if organ != "ALL":
        sub = sub[sub["organ"] == organ]
    return sub["correct"].astype(float).to_numpy()


def _transferability(per_field_delta: pd.DataFrame) -> pd.DataFrame:
    """Per-organ summary: median |Δ|, mean Δ, fraction of fields where
    one dataset is ahead."""
    rows: list[dict] = []
    for organ, sub in per_field_delta.groupby("organ"):
        rows.append({
            "organ": organ,
            "n_fields": int(len(sub)),
            "mean_delta": float(sub["delta"].mean()),
            "median_abs_delta": float(sub["delta"].abs().median()),
            "frac_left_ahead": float((sub["delta"] > 0).mean()),
            "frac_right_ahead": float((sub["delta"] < 0).mean()),
        })
    return pd.DataFrame(rows)


def _distribution_shift(
    left: pd.DataFrame, right: pd.DataFrame,
    *, left_label: str, right_label: str,
) -> pd.DataFrame:
    """Per-field gold-class distribution shift indicators.

    For categorical fields: Jensen-Shannon distance, KL divergence,
    chi-square p-value. For continuous fields: Wasserstein-1 distance.
    """
    rows: list[dict] = []
    fields = sorted(set(left["field"].dropna()) & set(right["field"].dropna()))
    for field in fields:
        l = left[(left["field"] == field) & left["gold_present"]]["gold_value"].astype(str)
        r = right[(right["field"] == field) & right["gold_present"]]["gold_value"].astype(str)
        if l.empty or r.empty:
            continue
        # Try numeric first.
        try:
            l_num = l.astype(float).to_numpy()
            r_num = r.astype(float).to_numpy()
            wd = wasserstein(l_num, r_num)
            rows.append({
                "field": field, "kind": "continuous",
                "left_label": left_label, "right_label": right_label,
                "n_left": int(l.size), "n_right": int(r.size),
                "wasserstein": wd,
                "js_distance": float("nan"), "kl": float("nan"),
                "chi2_p": float("nan"),
            })
            continue
        except (ValueError, TypeError):
            pass
        # Categorical
        cats = sorted(set(l) | set(r))
        l_counts = np.array([float((l == c).sum()) for c in cats])
        r_counts = np.array([float((r == c).sum()) for c in cats])
        l_p = l_counts / l_counts.sum()
        r_p = r_counts / r_counts.sum()
        # Avoid zeros for KL.
        eps = 1e-9
        kl = float(kl_divergence(l_p + eps, r_p + eps))
        js = jensen_shannon(l_p, r_p)
        from scipy.stats import chi2_contingency
        try:
            _, p_chi, _, _ = chi2_contingency([l_counts, r_counts])
            p_val = float(p_chi)
        except Exception:
            p_val = float("nan")
        rows.append({
            "field": field, "kind": "categorical",
            "left_label": left_label, "right_label": right_label,
            "n_left": int(l.size), "n_right": int(r.size),
            "js_distance": js, "kl": kl, "chi2_p": p_val,
            "wasserstein": float("nan"),
        })
    return pd.DataFrame(rows)


__all__ = ["register"]
