"""Pure-data metric computations for the non-nested subcommand.

Every function here takes a DataFrame slice and returns either a row
dict or a small DataFrame. The orchestrator in ``run_non_nested.py``
groups the atomic table, calls these, and concatenates results.

All metrics that produce p-values get adjusted via
``stats_extra.adjust_pvalues`` AFTER concatenation by the orchestrator,
not here — so the multiple-comparisons correction applies to the
correct family.

References (full citations in ``docs/eval/methods_citations.md``):
    Wilson 1927; Cohen 1960, 1968; Matthews 1975; Pedregosa et al. 2011.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.ci import (
    BootstrapResult, bootstrap_ci, t_ci, wilson_ci,
)
from digital_registrar_research.benchmarks.eval.metrics import normalize as _normalize


def normalize(v):
    """Normalise to a hashable string-or-None for label/CM comparisons.

    The shared :func:`metrics.normalize` keeps bools as bools and ints
    as ints, which breaks label matching against the canonical
    ``["true", "false"]`` enum lists. For confusion-matrix and per-class
    metric work we always coerce to string so labels and observations
    are comparable.
    """
    n = _normalize(v)
    if n is None:
        return None
    if isinstance(n, bool):
        return "true" if n else "false"
    if isinstance(n, (int, float)):
        return str(n)
    return str(n)
from digital_registrar_research.benchmarks.eval.scope import (
    get_allowed_values,
)
from digital_registrar_research.benchmarks.eval.semantic_neighbors import (
    is_neighbor, neighbors_for_field,
)

from .._common.stats_extra import (
    adjust_pvalues, balanced_accuracy, confusion_matrix_long,
    matthews_corrcoef, per_class_prf1, rank_distance_distribution,
    top_k_ordinal_accuracy,
)

logger = logging.getLogger(__name__)


# --- Atomic-table reductions -------------------------------------------------


def per_field_summary(
    df: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Per-field accuracy (attempted + effective) with multiple CI flavors.

    Output columns:
        field, n_total, n_eligible, n_attempted, n_correct,
        attempted_accuracy + Wilson + bootstrap CI,
        effective_accuracy + Wilson CI,
        completeness_penalty (= attempted_acc − effective_acc),
        mean_per_run_accuracy + Student-t CI,
        attempted_rate, parse_error_rate, field_missing_rate
        (each + Wilson CI).
    """
    out_rows: list[dict] = []
    for field, sub in df.groupby("field"):
        out_rows.append(_per_field_row(sub, field=field, n_boot=n_boot,
                                       alpha=alpha, seed=seed))
    return pd.DataFrame(out_rows)


def per_field_by_organ_summary(
    df: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Same as :func:`per_field_summary` but stratified by ``organ``."""
    out_rows: list[dict] = []
    for (organ, field), sub in df.groupby(["organ", "field"], dropna=False):
        row = _per_field_row(sub, field=field, n_boot=n_boot,
                             alpha=alpha, seed=seed)
        row["organ"] = organ
        out_rows.append(row)
    cols = ["organ"] + [c for c in out_rows[0].keys() if c != "organ"] if out_rows else []
    return pd.DataFrame(out_rows, columns=cols)


def per_field_subgroup_summary(
    df: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Per (field, subgroup) — multi_primary vs single_primary breakdown."""
    out_rows: list[dict] = []
    for (subgroup, field), sub in df.groupby(["subgroup", "field"], dropna=False):
        row = _per_field_row(sub, field=field, n_boot=n_boot,
                             alpha=alpha, seed=seed)
        row["subgroup"] = subgroup
        out_rows.append(row)
    cols = ["subgroup"] + [c for c in out_rows[0].keys() if c != "subgroup"] if out_rows else []
    return pd.DataFrame(out_rows, columns=cols)


def per_organ_overall_summary(
    df: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Aggregate accuracy across all fields per organ + a cross-organ ALL row.

    Each row reports the overall (cross-field) attempted and effective
    accuracy for one organ. Bootstrap CI is computed on the atomic table
    by case-stratified resampling, so the CI reflects between-case
    variance.
    """
    out_rows: list[dict] = []
    for organ in sorted(df["organ"].dropna().unique().tolist()):
        sub = df[df["organ"] == organ]
        out_rows.append(_overall_row(sub, organ=organ, label=organ,
                                     n_boot=n_boot, alpha=alpha, seed=seed))
    # Cross-organ ALL row.
    out_rows.append(_overall_row(df, organ="ALL", label="ALL",
                                 n_boot=n_boot, alpha=alpha, seed=seed))
    return pd.DataFrame(out_rows)


def _overall_row(
    sub: pd.DataFrame, *, organ: str, label: str,
    n_boot: int, alpha: float, seed: int,
) -> dict:
    """Single aggregate row across all fields in ``sub``."""
    n_total = len(sub)
    n_attempted = int(sub["attempted"].sum())
    n_correct = int(sub["correct"].sum())
    n_wrong = int(sub["wrong"].sum())
    n_field_missing = int(sub["field_missing"].sum())
    n_parse_error = int(sub["parse_error"].sum())
    n_eligible = int(sub["gold_present"].sum())

    if n_attempted > 0:
        attempted_accuracy = n_correct / n_attempted
        att_lo, att_hi = wilson_ci(n_correct, n_attempted, alpha)
    else:
        attempted_accuracy = float("nan")
        att_lo = att_hi = float("nan")

    effective_accuracy = n_correct / n_total if n_total else float("nan")
    eff_lo, eff_hi = (wilson_ci(n_correct, n_total, alpha) if n_total
                      else (float("nan"), float("nan")))

    # Bootstrap CI on attempted accuracy, resampling case×field rows.
    if n_attempted > 0:
        records = sub[sub["attempted"]][["case_id", "correct"]].to_dict("records")
        boot = bootstrap_ci(
            records,
            lambda xs: float(np.mean([r["correct"] for r in xs])),
            n_boot=n_boot, alpha=alpha, random_state=seed,
        )
        boot_lo, boot_hi = boot.lo, boot.hi
    else:
        boot_lo = boot_hi = float("nan")

    n_unique_fields = int(sub["field"].nunique()) if not sub.empty else 0
    n_unique_cases = int(sub["case_id"].nunique()) if not sub.empty else 0
    n_unique_runs = int(sub["run_id"].nunique()) if not sub.empty else 0

    # Per-field mean of attempted accuracy (different summary — gives
    # equal weight to every field rather than weighting by sample count).
    per_field_acc = (
        sub[sub["attempted"]]
        .groupby("field")["correct"].mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    macro_field_acc = (float(per_field_acc.mean())
                       if per_field_acc.size else float("nan"))

    return {
        "organ": label,
        "n_unique_fields": n_unique_fields,
        "n_unique_cases": n_unique_cases,
        "n_unique_runs": n_unique_runs,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_attempted": n_attempted,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_field_missing": n_field_missing,
        "n_parse_error": n_parse_error,
        "attempted_accuracy_micro": attempted_accuracy,
        "attempted_acc_wilson_lo": att_lo,
        "attempted_acc_wilson_hi": att_hi,
        "attempted_acc_boot_lo": boot_lo,
        "attempted_acc_boot_hi": boot_hi,
        "effective_accuracy_micro": effective_accuracy,
        "effective_acc_wilson_lo": eff_lo,
        "effective_acc_wilson_hi": eff_hi,
        "macro_field_accuracy": macro_field_acc,
        "completeness_penalty": (
            attempted_accuracy - effective_accuracy
            if not (np.isnan(attempted_accuracy) or np.isnan(effective_accuracy))
            else float("nan")
        ),
        "attempted_rate": (n_attempted / n_total if n_total else float("nan")),
        "parse_error_rate": (n_parse_error / n_total if n_total else float("nan")),
        "field_missing_rate": (n_field_missing / n_total if n_total else float("nan")),
    }


def _per_field_row(
    sub: pd.DataFrame, *, field: str,
    n_boot: int, alpha: float, seed: int,
) -> dict:
    """One headline row for a (field, ...) slice of the atomic table."""
    n_total = len(sub)
    n_eligible = int(sub["gold_present"].sum())
    n_attempted = int(sub["attempted"].sum())
    n_correct = int(sub["correct"].sum())
    n_wrong = int(sub["wrong"].sum())
    n_field_missing = int(sub["field_missing"].sum())
    n_parse_error = int(sub["parse_error"].sum())

    if n_attempted > 0:
        attempted_accuracy = n_correct / n_attempted
        att_wilson_lo, att_wilson_hi = wilson_ci(n_correct, n_attempted, alpha)
    else:
        attempted_accuracy = float("nan")
        att_wilson_lo = att_wilson_hi = float("nan")

    effective_accuracy = n_correct / n_total if n_total else float("nan")
    eff_wilson_lo, eff_wilson_hi = (wilson_ci(n_correct, n_total, alpha)
                                     if n_total else (float("nan"), float("nan")))

    # Bootstrap CI on attempted accuracy via case-level resampling.
    case_ids = sub["case_id"].tolist()
    correct_vec = sub["correct"].astype(int).tolist()
    attempted_vec = sub["attempted"].astype(bool).tolist()
    if n_attempted > 0:
        records = list(zip(case_ids, correct_vec, attempted_vec))
        att_boot = bootstrap_ci(
            records,
            lambda xs: (
                float(np.sum([c for _, c, a in xs if a])) /
                max(1, sum(1 for _, _, a in xs if a))
            ),
            n_boot=n_boot, alpha=alpha, random_state=seed,
        )
    else:
        att_boot = BootstrapResult(float("nan"), float("nan"), float("nan"),
                                   np.array([]), "n/a")

    # Per-run mean accuracies → Student-t CI (only meaningful with ≥2 runs).
    per_run = sub.groupby("run_id").apply(
        lambda g: (g["correct"].sum() / max(1, g["attempted"].sum())),
        include_groups=False,
    ).to_numpy(dtype=float)
    if per_run.size >= 2:
        run_mean, run_lo, run_hi = t_ci(per_run.tolist(), alpha)
    else:
        run_mean = float(per_run[0]) if per_run.size == 1 else float("nan")
        run_lo = run_hi = float("nan")

    # Wilson CIs on missingness components.
    pe_lo, pe_hi = wilson_ci(n_parse_error, n_total, alpha) if n_total else (float("nan"),) * 2
    fm_lo, fm_hi = wilson_ci(n_field_missing, n_total, alpha) if n_total else (float("nan"),) * 2
    at_lo, at_hi = wilson_ci(n_attempted, n_total, alpha) if n_total else (float("nan"),) * 2

    return {
        "field": field,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_attempted": n_attempted,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_field_missing": n_field_missing,
        "n_parse_error": n_parse_error,
        "n_runs": int(sub["run_id"].nunique()),
        "n_cases": int(sub["case_id"].nunique()),
        "attempted_accuracy": attempted_accuracy,
        "attempted_acc_wilson_lo": att_wilson_lo,
        "attempted_acc_wilson_hi": att_wilson_hi,
        "attempted_acc_boot_lo": att_boot.lo,
        "attempted_acc_boot_hi": att_boot.hi,
        "effective_accuracy": effective_accuracy,
        "effective_acc_wilson_lo": eff_wilson_lo,
        "effective_acc_wilson_hi": eff_wilson_hi,
        "completeness_penalty": (
            attempted_accuracy - effective_accuracy
            if not (np.isnan(attempted_accuracy) or np.isnan(effective_accuracy))
            else float("nan")
        ),
        "mean_per_run_accuracy": run_mean,
        "mean_per_run_t_ci_lo": run_lo,
        "mean_per_run_t_ci_hi": run_hi,
        "attempted_rate": (n_attempted / n_total if n_total else float("nan")),
        "attempted_rate_lo": at_lo, "attempted_rate_hi": at_hi,
        "parse_error_rate": (n_parse_error / n_total if n_total else float("nan")),
        "parse_error_rate_lo": pe_lo, "parse_error_rate_hi": pe_hi,
        "field_missing_rate": (n_field_missing / n_total if n_total else float("nan")),
        "field_missing_rate_lo": fm_lo, "field_missing_rate_hi": fm_hi,
    }


# --- Confusion matrices + per-class P/R/F1 ----------------------------------


def confusion_for_field(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
) -> pd.DataFrame | None:
    """Confusion matrix for one (field, organ) slice as a long-form DF.

    Returns ``None`` for non-categorical fields (no confusion matrix
    defined for continuous values).
    """
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"]]
    if sub.empty:
        return None
    y_true = [normalize(v) for v in sub["gold_value"]]
    y_pred = [normalize(v) for v in sub["pred_value"]]
    labels = [normalize(v) for v in allowed]
    # Extend with any actually-observed values not in the canonical list
    # so the matrix is exhaustive (and skip if there's no overlap with
    # gold — sklearn raises otherwise).
    extra = sorted(set(y_true) | set(y_pred) - set(labels) - {None})
    full_labels = labels + extra
    if not (set(y_true) & set(full_labels)):
        return None
    try:
        rows = confusion_matrix_long(y_true, y_pred, labels=full_labels)
    except ValueError:
        return None
    cm_df = pd.DataFrame(rows)
    cm_df["field"] = field
    cm_df["organ"] = organ or "ALL"
    return cm_df


def per_class_for_field(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
) -> pd.DataFrame | None:
    """Per-class precision/recall/F1 for one (field, organ) slice."""
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"]]
    if sub.empty:
        return None
    labels = [normalize(v) for v in allowed]
    y_true = [normalize(v) for v in sub["gold_value"]]
    y_pred = [normalize(v) for v in sub["pred_value"]]
    extra = sorted(set(y_true) | set(y_pred) - set(labels) - {None})
    full_labels = labels + extra
    if not (set(y_true) & set(full_labels)):
        return None
    try:
        prf = per_class_prf1(y_true, y_pred, labels=full_labels)
    except ValueError:
        return None
    rows = []
    for label, stats in prf.items():
        if not isinstance(stats, dict):
            continue
        rows.append({
            "field": field, "organ": organ or "ALL",
            "class": label,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "support": stats.get("support", 0),
        })
    return pd.DataFrame(rows)


def headline_classification_metrics(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
) -> dict | None:
    """MCC + balanced accuracy + Cohen's κ for one (field, organ) slice.

    Implementation:
        - MCC: ``sklearn.metrics.matthews_corrcoef`` (Matthews 1975).
        - Balanced accuracy: ``sklearn.metrics.balanced_accuracy_score``.
        - Cohen's κ: ``sklearn.metrics.cohen_kappa_score`` (Cohen 1960).
    """
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"]]
    if len(sub) < 2:
        return None
    labels = [normalize(v) for v in allowed]
    y_true = [normalize(v) for v in sub["gold_value"]]
    y_pred = [normalize(v) for v in sub["pred_value"]]
    extra = sorted(set(y_true) | set(y_pred) - set(labels) - {None})
    full_labels = labels + extra
    from sklearn.metrics import cohen_kappa_score
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=full_labels))
    except Exception:
        kappa = float("nan")
    try:
        kappa_q = float(cohen_kappa_score(y_true, y_pred, labels=full_labels,
                                          weights="quadratic"))
    except Exception:
        kappa_q = float("nan")
    try:
        ba = balanced_accuracy(y_true, y_pred)
    except Exception:
        ba = float("nan")
    # MCC defined for binary; sklearn handles multi-class via the
    # generalisation. Returns NaN if degenerate.
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")
    return {
        "field": field, "organ": organ or "ALL",
        "n": int(len(sub)),
        "cohen_kappa": kappa,
        "cohen_kappa_quadratic": kappa_q,
        "balanced_accuracy": ba,
        "matthews_corrcoef": mcc,
    }


# --- Confusion-pair / semantic-neighbor analysis (item A) -------------------


def confusion_pairs_for_field(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame | None:
    """Top-N most frequent confusion pairs for a categorical field.

    Joins against the curated semantic-neighbor list and flags whether
    each pair is a known near-miss (``is_semantic_neighbor`` column).
    """
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    # Restrict to wrong predictions only — confusion pairs are defined
    # over off-diagonal cells.
    sub = sub[sub["attempted"] & sub["gold_present"] & sub["wrong"]]
    if sub.empty:
        return None
    pairs = sub.groupby(["gold_value", "pred_value"]).size().reset_index(name="count")
    pairs = pairs.sort_values("count", ascending=False).head(top_n)
    pairs["field"] = field
    pairs["organ"] = organ or "ALL"
    pairs["is_semantic_neighbor"] = [
        is_neighbor(field, g, p)
        for g, p in zip(pairs["gold_value"], pairs["pred_value"], strict=True)
    ]
    return pairs


def accuracy_collapsing_neighbors(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
) -> dict | None:
    """Alternative accuracy treating curated neighbor errors as correct.

    Returns ``None`` if no curated neighbors exist for this field.
    """
    if not neighbors_for_field(field):
        return None
    organ_arg = organ if organ != "ALL" else None
    if not get_allowed_values(field, organ_arg):
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"]]
    if sub.empty:
        return None
    n = len(sub)
    n_correct_strict = int(sub["correct"].sum())
    n_correct_collapsed = n_correct_strict + int(sub.apply(
        lambda r: (not r["correct"]) and is_neighbor(field, r["gold_value"], r["pred_value"]),
        axis=1,
    ).sum())
    return {
        "field": field, "organ": organ or "ALL",
        "n": int(n),
        "accuracy_strict": n_correct_strict / n if n else float("nan"),
        "accuracy_collapsing_neighbors": n_correct_collapsed / n if n else float("nan"),
        "n_neighbor_recovered": n_correct_collapsed - n_correct_strict,
    }


# --- Ordinal rank-distance --------------------------------------------------


def rank_distance_for_field(
    df: pd.DataFrame, *, field: str, organ: str | None = None,
) -> dict | None:
    """Ordinal-aware error severity. Returns mean rank distance and the
    full distribution of per-case rank differences.

    Skips if the field doesn't have an ordinal allowed-values list.
    """
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"] & sub["wrong"]]
    if sub.empty:
        return None
    labels = [normalize(v) for v in allowed]
    y_true = [normalize(v) for v in sub["gold_value"]]
    y_pred = [normalize(v) for v in sub["pred_value"]]
    dist = rank_distance_distribution(y_true, y_pred, ordinal_order=labels)
    if not dist:
        return None
    distances = [d for d, c in dist.items() for _ in range(c)]
    return {
        "field": field, "organ": organ or "ALL",
        "n_wrong": int(len(sub)),
        "mean_rank_distance": float(np.mean(distances)),
        "max_rank_distance": int(max(distances)),
        "top1_off_count": int(dist.get(1, 0)),
        "top2_off_count": int(dist.get(1, 0) + dist.get(2, 0)),
        "histogram": str({int(k): int(v) for k, v in sorted(dist.items())}),
    }


def top_k_for_ordinal_field(
    df: pd.DataFrame, *, field: str, organ: str | None = None, k: int = 1,
) -> dict | None:
    """Top-k ordinal accuracy: ``|rank(pred) - rank(gold)| ≤ k``."""
    organ_arg = organ if organ != "ALL" else None
    allowed = get_allowed_values(field, organ_arg)
    if not allowed:
        return None
    sub = df[df["field"] == field]
    if organ is not None and organ != "ALL":
        sub = sub[sub["organ"] == organ]
    sub = sub[sub["attempted"] & sub["gold_present"]]
    if sub.empty:
        return None
    labels = [normalize(v) for v in allowed]
    y_true = [normalize(v) for v in sub["gold_value"]]
    y_pred = [normalize(v) for v in sub["pred_value"]]
    acc = top_k_ordinal_accuracy(y_true, y_pred, ordinal_order=labels, k=k)
    return {
        "field": field, "organ": organ or "ALL",
        "k": k, "n": int(len(sub)),
        "accuracy_within_k": acc,
    }


# --- Run-to-run consistency (extends multirun.run_consistency) --------------


def run_consistency_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Per-field consistency: Fleiss κ on correctness, Fleiss κ on
    prediction values, missing-flip rate, stability accuracy.

    Output rows:
        field, n_cases, n_runs, fleiss_kappa_correctness,
        fleiss_kappa_values, flip_rate, missing_flip_rate,
        stability_accuracy, brittle_case_rate.
    """
    from digital_registrar_research.benchmarks.eval.multirun import fleiss_kappa
    rows: list[dict] = []
    for field, sub in df.groupby("field"):
        if sub["run_id"].nunique() < 2:
            continue
        # Fleiss κ on correctness (binary 0/1).
        pivot_corr = sub.pivot_table(
            index="case_id", columns="run_id", values="correct",
            aggfunc="first",
        ).astype(float)
        pivot_attempted = sub.pivot_table(
            index="case_id", columns="run_id", values="attempted",
            aggfunc="first",
        ).astype(float)

        # Fleiss κ on the value itself (categorical) — codes as integers.
        # For continuous values this is meaningless; skip categorical
        # κ if too many distinct values.
        pivot_val = sub.pivot_table(
            index="case_id", columns="run_id", values="pred_value",
            aggfunc="first",
        )
        try:
            value_codes = (pivot_val.fillna("__missing__")
                           .applymap(lambda v: normalize(v))
                           .astype("category"))
            codes = value_codes.apply(lambda col: col.cat.codes).to_numpy()
            n_distinct = pd.unique(value_codes.values.ravel()).size
            fk_values = (fleiss_kappa(codes.astype(float))
                         if 2 <= n_distinct <= 50 else float("nan"))
        except Exception:
            fk_values = float("nan")

        try:
            fk_corr = fleiss_kappa(pivot_corr.to_numpy(dtype=float))
        except Exception:
            fk_corr = float("nan")

        # Flip rate on correctness
        m = pivot_corr.to_numpy(dtype=float)
        if m.size:
            valid_rows = ~np.any(np.isnan(m), axis=1)
            flips = ~np.all(m[valid_rows] == m[valid_rows][:, [0]], axis=1)
            flip_rate = float(flips.mean()) if valid_rows.any() else float("nan")
        else:
            flip_rate = float("nan")

        # Missing-flip rate on attempted
        ma = pivot_attempted.to_numpy(dtype=float)
        if ma.size:
            valid_rows = ~np.any(np.isnan(ma), axis=1)
            missing_flips = (
                np.any(ma[valid_rows] == 0, axis=1) &
                np.any(ma[valid_rows] == 1, axis=1)
            )
            missing_flip_rate = (float(missing_flips.mean())
                                 if valid_rows.any() else float("nan"))
        else:
            missing_flip_rate = float("nan")

        # Stability accuracy: accuracy on cases where all runs agree.
        all_agree_idx = (
            np.all(m == m[:, [0]], axis=1) & ~np.any(np.isnan(m), axis=1)
            if m.size else np.array([])
        )
        stability_accuracy = (float(m[all_agree_idx, 0].mean())
                              if all_agree_idx.any() else float("nan"))
        # Brittle: ≥1 run wrong AND ≥1 run right
        brittle = (
            np.any(m == 0, axis=1) & np.any(m == 1, axis=1)
            & ~np.any(np.isnan(m), axis=1)
            if m.size else np.array([])
        )
        brittle_rate = float(brittle.mean()) if m.size else float("nan")

        rows.append({
            "field": field,
            "n_cases": int(pivot_corr.shape[0]),
            "n_runs": int(pivot_corr.shape[1]),
            "fleiss_kappa_correctness": fk_corr,
            "fleiss_kappa_values": fk_values,
            "flip_rate": flip_rate,
            "missing_flip_rate": missing_flip_rate,
            "stability_accuracy": stability_accuracy,
            "brittle_case_rate": brittle_rate,
        })
    return pd.DataFrame(rows)


# --- Schema-conformance / OOV (item E) --------------------------------------


def schema_conformance(df: pd.DataFrame) -> pd.DataFrame:
    """Per (field, organ) — out-of-vocabulary rate among attempted preds.

    For categorical fields only. Pairs with the modularity-advantage
    argument: schema-constrained pipelines should produce ~0% OOV.
    """
    from digital_registrar_research.benchmarks.eval.completeness import (
        out_of_vocab_rate as _oov,
    )

    rows: list[dict] = []
    fields = df["field"].dropna().unique()
    for field in fields:
        for organ in [*sorted(df["organ"].dropna().unique()), "ALL"]:
            organ_arg = organ if organ != "ALL" else None
            allowed = get_allowed_values(field, organ_arg)
            if not allowed:
                continue
            allowed_norm = {normalize(v) for v in allowed}
            sub = df[(df["field"] == field) & df["attempted"]]
            if organ != "ALL":
                sub = sub[sub["organ"] == organ]
            n_attempted = len(sub)
            n_oov = int(sub["pred_value"].apply(
                lambda v: normalize(v) not in allowed_norm and v is not None
            ).sum())
            if n_attempted == 0:
                continue
            lo, hi = wilson_ci(n_oov, n_attempted)
            rows.append({
                "field": field, "organ": organ,
                "n_attempted": n_attempted, "n_oov": n_oov,
                "oov_rate": n_oov / n_attempted,
                "oov_rate_ci_lo": lo, "oov_rate_ci_hi": hi,
            })
    return pd.DataFrame(rows)


# --- Section rollup ---------------------------------------------------------


def section_rollup(
    df: pd.DataFrame,
    *,
    section_of_field: dict[str, str],
    n_boot: int, alpha: float, seed: int,
) -> pd.DataFrame:
    """Mean attempted_accuracy across fields in each section, with
    bootstrap CI over fields.
    """
    typed = df.copy()
    typed["section"] = typed["field"].map(section_of_field).fillna("other")
    rows: list[dict] = []
    for section, sub in typed.groupby("section"):
        per_field = (
            sub.groupby("field").apply(
                lambda g: (g["correct"].sum() / max(1, g["attempted"].sum())),
                include_groups=False,
            ).to_numpy(dtype=float)
        )
        per_field = per_field[~np.isnan(per_field)]
        if per_field.size == 0:
            continue
        # Bootstrap over fields.
        rng = np.random.default_rng(seed)
        boot = np.array([
            float(rng.choice(per_field, size=per_field.size, replace=True).mean())
            for _ in range(n_boot)
        ])
        lo = float(np.quantile(boot, alpha / 2))
        hi = float(np.quantile(boot, 1 - alpha / 2))
        rows.append({
            "section": section,
            "n_fields": int(per_field.size),
            "mean_field_accuracy": float(per_field.mean()),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


__all__ = [
    "per_field_summary",
    "per_field_by_organ_summary",
    "per_field_subgroup_summary",
    "per_organ_overall_summary",
    "confusion_for_field",
    "per_class_for_field",
    "headline_classification_metrics",
    "confusion_pairs_for_field",
    "accuracy_collapsing_neighbors",
    "rank_distance_for_field",
    "top_k_for_ordinal_field",
    "run_consistency_extended",
    "schema_conformance",
    "section_rollup",
]
