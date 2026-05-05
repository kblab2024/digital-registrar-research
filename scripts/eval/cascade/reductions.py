"""Reductions for the cascade chapter outputs.

Builds the chapter1 / chapter2 / chapter3 CSVs from the cascade atomic
table. Where the legacy non_nested / nested orchestrators have proven
reductions, we delegate to them rather than re-implement.

The atomic-table column contract (built by :mod:`run_cascade`):

    run_id, model, dataset, case_id, organ_idx, organ, subgroup,
    cascade_stage, gate_pass,
    others_disposition,
    field, field_kind,
    gold_present, attempted, correct, wrong, field_missing, parse_error,
    error_mode, gold_value, pred_value
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.stats import (
    accuracy_flip_rate,
    bca_bootstrap_ci,
    cohens_kappa,
    cronbach_alpha,
    icc_2_1,
    icc_3_k,
    krippendorff_alpha,
    matthews_corrcoef,
    missing_flip_rate,
    per_case_run_sd,
    spearman_brown,
    weighted_kappa,
    wilson_ci,
)


def _safe_proportion(k: int, n: int) -> float:
    return float(k / n) if n > 0 else float("nan")


def _wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    return wilson_ci(k, n, alpha)


# --- Chapter 1: eligibility triage ----------------------------------------

def chapter1_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Eligibility triage headline metrics.

    Columns: model, dataset, n_total, n_correct, accuracy, ci_lo, ci_hi,
    n_positive_gold, n_negative_gold, sensitivity, specificity, mcc,
    cohens_kappa.
    """
    stage_a = atomic[atomic["cascade_stage"] == "A"]
    if stage_a.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_a.groupby(["model", "dataset"], dropna=False):
        n_total = len(sub)
        n_correct = int(sub["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_total)
        lo, hi = _wilson(n_correct, n_total, alpha)
        # Sens/spec on the binary eligibility decision.
        gold_pos = sub["gold_value"].astype(str).isin(["True", "true", "1"])
        pred_pos = sub["pred_value"].astype(str).isin(["True", "true", "1"])
        tp = int(((gold_pos) & (pred_pos)).sum())
        fn = int(((gold_pos) & (~pred_pos)).sum())
        fp = int(((~gold_pos) & (pred_pos)).sum())
        tn = int(((~gold_pos) & (~pred_pos)).sum())
        sens = _safe_proportion(tp, tp + fn)
        spec = _safe_proportion(tn, tn + fp)
        mcc_val = float("nan")
        try:
            mcc_val = matthews_corrcoef(
                gold_pos.tolist(), pred_pos.tolist(),
            )
        except Exception:
            pass
        kappa_val = float("nan")
        try:
            kappa_val = cohens_kappa(
                gold_pos.tolist(), pred_pos.tolist(),
            )
        except Exception:
            pass
        rows.append({
            "model": model, "dataset": dataset,
            "n_total": n_total, "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "n_positive_gold": int(gold_pos.sum()),
            "n_negative_gold": int((~gold_pos).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec,
            "mcc": mcc_val, "cohens_kappa": kappa_val,
        })
    return pd.DataFrame(rows)


def chapter1_confusion(atomic: pd.DataFrame) -> pd.DataFrame:
    """2x2 confusion of eligibility, per (model, dataset)."""
    stage_a = atomic[atomic["cascade_stage"] == "A"]
    if stage_a.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_a.groupby(["model", "dataset"], dropna=False):
        gold_pos = sub["gold_value"].astype(str).isin(["True", "true", "1"])
        pred_pos = sub["pred_value"].astype(str).isin(["True", "true", "1"])
        tp = int(((gold_pos) & (pred_pos)).sum())
        fn = int(((gold_pos) & (~pred_pos)).sum())
        fp = int(((~gold_pos) & (pred_pos)).sum())
        tn = int(((~gold_pos) & (~pred_pos)).sum())
        rows.append({
            "model": model, "dataset": dataset,
            "TP_eligible_predicted_eligible": tp,
            "FP_ineligible_predicted_eligible": fp,
            "FN_eligible_predicted_ineligible": fn,
            "TN_ineligible_predicted_ineligible": tn,
            "n_total": tp + fp + fn + tn,
        })
    return pd.DataFrame(rows)


# --- Chapter 2: organ classification --------------------------------------

def chapter2_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Organ-classification headline. Restricted to Stage-A passers.

    Columns: model, dataset, n_eligible_for_b, n_correct, accuracy,
    ci_lo, ci_hi, macro_f1, weighted_kappa, cohens_kappa.
    """
    stage_b = atomic[atomic["cascade_stage"] == "B"]
    if stage_b.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset), sub in stage_b.groupby(["model", "dataset"], dropna=False):
        n_total = len(sub)
        n_correct = int(sub["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_total)
        lo, hi = _wilson(n_correct, n_total, alpha)
        gold = sub["gold_value"].dropna().tolist()
        pred = sub["pred_value"].dropna().tolist()
        # Pair-by-row.
        paired = list(zip(sub["gold_value"].tolist(), sub["pred_value"].tolist()))
        paired = [(g, p) for g, p in paired if g is not None and p is not None]
        kappa_val = float("nan")
        try:
            if paired:
                gv, pv = zip(*paired)
                kappa_val = cohens_kappa(list(gv), list(pv))
        except Exception:
            pass
        # Macro F1 via per-class P/R/F1 helper from stats_extra.
        macro_f1 = float("nan")
        try:
            from scripts.eval._common.stats_extra import per_class_prf1
            if paired:
                gv, pv = zip(*paired)
                pr = per_class_prf1(list(gv), list(pv))
                macro_f1 = pr.get("macro_avg", {}).get("f1", float("nan"))
        except Exception:
            pass
        rows.append({
            "model": model, "dataset": dataset,
            "n_eligible_for_b": n_total,
            "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "cohens_kappa": kappa_val,
            "macro_f1": macro_f1,
        })
    return pd.DataFrame(rows)


def chapter2_confusion_per_class(atomic: pd.DataFrame) -> pd.DataFrame:
    """Per-class P/R/F1 + support for the 11-class organ classifier."""
    stage_b = atomic[atomic["cascade_stage"] == "B"]
    if stage_b.empty:
        return pd.DataFrame()
    from scripts.eval._common.stats_extra import per_class_prf1
    rows: list[dict] = []
    for (model, dataset), sub in stage_b.groupby(["model", "dataset"], dropna=False):
        paired = [(g, p) for g, p in zip(sub["gold_value"], sub["pred_value"])
                  if g is not None and p is not None]
        if not paired:
            continue
        gv, pv = zip(*paired)
        pr = per_class_prf1(list(gv), list(pv))
        for label, vals in pr.items():
            if not isinstance(vals, dict):
                continue
            rows.append({
                "model": model, "dataset": dataset,
                "label": label,
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1": vals.get("f1"),
                "support": vals.get("support", float("nan")),
            })
    return pd.DataFrame(rows)


# --- Chapter 3: field extraction ------------------------------------------

def chapter3_per_field_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Stage-C scalar field accuracy with Wilson CI and Cohen's kappa.

    One row per (model, dataset, field).
    """
    stage_c = atomic[atomic["cascade_stage"] == "C"]
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, field), sub in stage_c.groupby(
        ["model", "dataset", "field"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        n_total = len(sub)
        coverage = _safe_proportion(n_attempted, n_total)
        # Kappa per field for paired non-null gold/pred.
        kappa_val = float("nan")
        paired = [(g, p) for g, p in zip(attempted["gold_value"],
                                          attempted["pred_value"])
                  if g is not None and p is not None]
        try:
            if paired:
                gv, pv = zip(*paired)
                kappa_val = cohens_kappa(list(gv), list(pv))
        except Exception:
            pass
        rows.append({
            "model": model, "dataset": dataset, "field": field,
            "n_total": n_total, "n_attempted": n_attempted,
            "n_correct": n_correct,
            "coverage": coverage,
            "accuracy_attempted": acc,
            "ci_lo": lo, "ci_hi": hi,
            "cohens_kappa": kappa_val,
        })
    return pd.DataFrame(rows)


def chapter3_per_field_by_organ(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Stage-C per-field accuracy stratified by organ."""
    stage_c = atomic[atomic["cascade_stage"] == "C"]
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ, field), sub in stage_c.groupby(
        ["model", "dataset", "organ", "field"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        rows.append({
            "model": model, "dataset": dataset,
            "organ": organ, "field": field,
            "n_attempted": n_attempted,
            "n_correct": n_correct,
            "accuracy": acc,
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def chapter3_per_organ_overall(
    atomic: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mean accuracy across all fields per organ (Stage C only)."""
    stage_c = atomic[atomic["cascade_stage"] == "C"]
    if stage_c.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (model, dataset, organ), sub in stage_c.groupby(
        ["model", "dataset", "organ"], dropna=False,
    ):
        attempted = sub[sub["attempted"] == True]  # noqa: E712
        n_attempted = len(attempted)
        n_correct = int(attempted["correct"].fillna(False).astype(bool).sum())
        acc = _safe_proportion(n_correct, n_attempted)
        lo, hi = _wilson(n_correct, n_attempted, alpha)
        rows.append({
            "model": model, "dataset": dataset, "organ": organ,
            "n_attempted": n_attempted, "n_correct": n_correct,
            "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# --- Multi-run reliability per chapter ------------------------------------

def _correctness_matrix(
    sub: pd.DataFrame,
    *,
    case_col: str = "case_id",
    run_col: str = "run_id",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """Pivot a long-form DF to (case_id × run_id) of correctness values."""
    sub = sub.copy()
    sub[correct_col] = pd.to_numeric(sub[correct_col], errors="coerce")
    return sub.pivot_table(
        index=case_col, columns=run_col, values=correct_col, aggfunc="first",
    )


def chapter_multirun_reliability(
    atomic: pd.DataFrame,
    *,
    stage: str,
    field: str | None = None,
) -> pd.DataFrame:
    """Multi-run reliability for one cascade stage.

    Computes ICC(2,1), ICC(3,k), Cronbach alpha, accuracy_flip_rate per
    (model, dataset[, field]) over the case × run correctness matrix.

    For Stage A and B, set ``field=None`` (only one field per stage).
    For Stage C, pass a specific ``field`` or call once per field.
    """
    sub = atomic[atomic["cascade_stage"] == stage]
    if field is not None:
        sub = sub[sub["field"] == field]
    if sub.empty or sub["run_id"].nunique() < 2:
        return pd.DataFrame()
    rows: list[dict] = []
    by = ["model", "dataset"]
    if field is not None:
        by.append("field")
    for keys, grp in sub.groupby(by, dropna=False):
        mat = _correctness_matrix(grp).to_numpy(dtype=float)
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            continue
        i21 = icc_2_1(mat)
        i3k = icc_3_k(mat)
        alpha_val = cronbach_alpha(mat)
        flip = accuracy_flip_rate(mat)
        sd = per_case_run_sd(mat)
        row = {
            "model": keys[0], "dataset": keys[1],
            "stage": stage,
            "icc_2_1": i21["icc"], "icc_2_1_lo": i21["ci_lo"], "icc_2_1_hi": i21["ci_hi"],
            "icc_3_k": i3k["icc"], "icc_3_k_lo": i3k["ci_lo"], "icc_3_k_hi": i3k["ci_hi"],
            "cronbach_alpha": alpha_val,
            "accuracy_flip_rate": flip,
            "per_case_sd_mean": sd["mean"],
            "per_case_sd_p90": sd["p90"],
            "n_cases": sd["n_cases"], "n_runs": int(mat.shape[1]),
        }
        if field is not None:
            row["field"] = keys[2]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "chapter1_overall",
    "chapter1_confusion",
    "chapter2_overall",
    "chapter2_confusion_per_class",
    "chapter3_per_field_overall",
    "chapter3_per_field_by_organ",
    "chapter3_per_organ_overall",
    "chapter_multirun_reliability",
]
