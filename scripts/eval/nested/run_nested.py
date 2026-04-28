"""Nested subcommand orchestrator.

For each ``--field`` (one of ``regional_lymph_node``, ``margins``,
``biomarkers``) builds a per-case scoring table and emits:

    manifest.json
    nested_atomic.parquet           — per (run, case, organ) per-case scorer output
                                      + outcome flags + length metrics
    per_field_per_organ.csv         — micro F1, hallucination, miss rate, attempted
                                      F1 vs effective F1, count MAE
    per_attribute_per_organ.csv     — conditional accuracy on matched pairs
    case_level_per_organ.csv        — case-level summary (any-positive, totals, etc)
    nested_missingness.csv          — 4-level missingness decomposition
    multirun_consistency.csv        — F1 SD across runs, missing-flip
    support_distribution.csv        — histogram of |gold| and |pred| per case

Field handling:
    - regional_lymph_node → uses ``nested_metrics.score_lymph_nodes``.
    - margins             → uses ``nested_metrics.score_margins``.
    - biomarkers          → uses ``nested.biomarkers.score_biomarkers``.

Per-organ stratification is mandatory. Organs without the field in
their schema (e.g. biomarkers outside breast/colorectal) emit explicit
``n/a`` rows rather than silent drops.
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.ci import (
    bootstrap_ci, wilson_ci,
)
from digital_registrar_research.benchmarks.eval.metrics import normalize
from digital_registrar_research.benchmarks.eval.multi_primary import (
    subgroup_label,
)
from digital_registrar_research.benchmarks.eval.nested_metrics import (
    score_lymph_nodes, score_margins,
)
from digital_registrar_research.benchmarks.eval.scope import (
    get_field_value, get_nested_list_fields,
)

from .._common.args import (
    add_common_args, add_model_args, parse_cases, parse_organs,
    parse_run_ids, require_model,
)
from .._common.loaders import load_json, load_prediction, ParseError
from .._common.outcome import CaseLoad
from .._common.paths import Paths, from_args
from .._common.reporting import (
    setup_logging, write_csv, write_manifest, write_parquet,
)
from .._common.stratify import organ_name
from .biomarkers import score_biomarkers

logger = logging.getLogger("scripts.eval.nested")


SCORERS: dict[str, callable] = {
    "regional_lymph_node": score_lymph_nodes,
    "margins": score_margins,
    "biomarkers": score_biomarkers,
}

# Per-field item-level TP/FP/FN column names.
PRF_KEYS: dict[str, tuple[str, str, str]] = {
    "regional_lymph_node": ("ln_station_tp", "ln_station_fp", "ln_station_fn"),
    "margins": ("margin_tp", "margin_fp", "margin_fn"),
    "biomarkers": ("biomarker_tp", "biomarker_fp", "biomarker_fn"),
}

# Per-attribute "conditional accuracy on matched pairs" mappings.
PER_ATTRIBUTE_KEYS: dict[str, list[tuple[str, str]]] = {
    "regional_lymph_node": [
        ("station_involved_correct", "ln_station_involved_correct"),
        ("station_examined_correct", "ln_station_examined_correct"),
        ("station_category_correct", "ln_station_category_correct"),
        ("station_side_correct", "ln_station_side_correct"),
    ],
    "margins": [
        ("status_correct", "margin_status_correct"),
        ("distance_correct", "margin_distance_correct"),
        ("category_correct", "margin_category_correct"),
    ],
    "biomarkers": [
        ("expression_correct", "biomarker_expression_correct"),
        ("percentage_correct_tol", "biomarker_percentage_correct_tol"),
        ("score_correct", "biomarker_score_correct"),
    ],
}

# Per-field "matched-pair count" column name.
MATCHED_KEY: dict[str, str] = {
    "regional_lymph_node": "ln_station_matched",
    "margins": "margin_matched",
    "biomarkers": "biomarker_matched",
}


# --- Argparse registration ---------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "nested",
        help="Score nested-list fields (regional_lymph_node, margins, biomarkers).",
        description=__doc__,
    )
    add_common_args(parser, subcommand="nested")
    add_model_args(parser)
    parser.add_argument(
        "--field", required=True, choices=tuple(SCORERS.keys()),
        help="Which nested field to score.",
    )
    parser.set_defaults(_handler=_main)


# --- Main entry point --------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    require_model(args)

    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    run_ids = parse_run_ids(args) or _autodiscover_runs(paths, args)
    case_filter = parse_cases(args)

    if args.method == "llm" and not run_ids:
        raise SystemExit("no run_ids and none auto-discovered.")
    effective_runs = run_ids if args.method == "llm" else [""]

    field = args.field
    scorer = SCORERS[field]
    logger.info(
        "scoring nested field=%s method=%s model=%s with %d run(s)",
        field, args.method, args.model, len(effective_runs),
    )

    atomic, n_per_organ = _build_atomic(
        paths=paths, args=args, run_ids=effective_runs, organs=organs,
        case_filter=case_filter, field=field, scorer=scorer,
    )
    if atomic.empty:
        logger.error("atomic table is empty — check inputs.")
        return 1
    logger.info("atomic table: %d rows", len(atomic))

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(atomic, args.out / "nested_atomic.parquet")

    # Per (organ, ALL) headline metrics.
    write_csv(
        _per_field_per_organ(atomic, field=field,
                             n_boot=args.n_boot, alpha=args.alpha,
                             seed=args.seed),
        args.out / "per_field_per_organ.csv",
    )
    write_csv(
        _per_attribute_per_organ(atomic, field=field, alpha=args.alpha),
        args.out / "per_attribute_per_organ.csv",
    )
    write_csv(
        _case_level_per_organ(atomic, field=field, alpha=args.alpha),
        args.out / "case_level_per_organ.csv",
    )
    write_csv(
        _nested_missingness(atomic, field=field, alpha=args.alpha),
        args.out / "nested_missingness.csv",
    )
    write_csv(
        _support_distribution(atomic, field=field),
        args.out / "support_distribution.csv",
    )
    if atomic["run_id"].nunique() > 1:
        write_csv(
            _multirun_consistency(atomic, field=field),
            args.out / "multirun_consistency.csv",
        )

    write_manifest(
        args.out, args, subcommand="nested",
        n_cases_per_organ=n_per_organ,
        extra={
            "field": field,
            "n_runs": len(effective_runs),
            "run_ids": effective_runs,
            "n_atomic_rows": int(len(atomic)),
            "n_unique_cases": int(atomic["case_id"].nunique()),
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Atomic-table builder ----------------------------------------------------


def _build_atomic(
    *,
    paths: Paths,
    args: argparse.Namespace,
    run_ids: Iterable[str],
    organs: Iterable[int],
    case_filter: set[str] | None,
    field: str,
    scorer,
) -> tuple[pd.DataFrame, dict[int, int]]:
    n_per_organ: dict[int, int] = {}
    rows: list[dict] = []

    case_index: list[tuple[int, str]] = []
    for organ_idx, case_id in paths.case_ids(args.annotator, tuple(organs)):
        if case_filter and case_id not in case_filter:
            continue
        case_index.append((organ_idx, case_id))
    for oi, _ in case_index:
        n_per_organ[oi] = n_per_organ.get(oi, 0) + 1

    for run_id in run_ids:
        for organ_idx, case_id in case_index:
            gold_path = paths.annotation(args.annotator, organ_idx, case_id)
            try:
                gold = load_json(gold_path)
            except ParseError as e:
                logger.warning("skipping %s: %s", case_id, e)
                continue
            organ = normalize(gold.get("cancer_category")) or organ_name(args.dataset, organ_idx)
            subgroup = subgroup_label(gold)

            # Skip cases where the field is not in the organ schema.
            organ_supports_field = field in get_nested_list_fields(organ)
            # Determine gold list state.
            g_list = get_field_value(gold, field) if isinstance(get_field_value(gold, field), list) else None
            gold_field_key_present = (
                field in gold or field in (gold.get("cancer_data") or {})
            )

            pred_path = paths.prediction(
                method=args.method, model=args.model,
                run_id=run_id or None, organ_idx=organ_idx, case_id=case_id,
            )
            lo = load_prediction(pred_path)
            case_load = CaseLoad.from_load_outcome(lo)

            base = {
                "run_id": run_id or "",
                "method": args.method,
                "model": args.model or "",
                "annotator": args.annotator,
                "case_id": case_id,
                "organ_idx": int(organ_idx),
                "organ": organ,
                "subgroup": subgroup,
                "field": field,
                "organ_supports_field": organ_supports_field,
                "gold_field_present": gold_field_key_present,
                "n_gold_items": int(len(g_list)) if isinstance(g_list, list) else 0,
            }

            if not case_load.ok:
                rows.append({
                    **base, "parse_error": True,
                    "field_key_absent": False, "empty_list": False,
                    "attempted": False, "n_pred_items": 0,
                    "error_mode": case_load.error_mode,
                    "tp": 0, "fp": 0, "fn": 0, "f1": 0.0,
                    "matched": 0,
                })
                continue

            pred = case_load.pred or {}
            pred_field_present = (
                field in pred or field in (pred.get("cancer_data") or {})
            )
            p_list = get_field_value(pred, field)
            if not isinstance(p_list, list):
                p_list = None

            field_key_absent = not pred_field_present
            empty_list = (p_list is not None and len(p_list) == 0
                          and isinstance(g_list, list) and len(g_list) > 0)

            row = {**base, "parse_error": False,
                   "field_key_absent": field_key_absent,
                   "empty_list": empty_list,
                   "attempted": pred_field_present and p_list is not None,
                   "n_pred_items": int(len(p_list)) if isinstance(p_list, list) else 0,
                   "error_mode": None}

            if pred_field_present and p_list is not None:
                stats = scorer(gold, pred)
                tp_key, fp_key, fn_key = PRF_KEYS[field]
                tp = int(stats.get(tp_key, 0))
                fp = int(stats.get(fp_key, 0))
                fn = int(stats.get(fn_key, 0))
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                row["tp"] = tp
                row["fp"] = fp
                row["fn"] = fn
                row["f1"] = f1
                row["matched"] = int(stats.get(MATCHED_KEY[field], 0))
                # Stash all of the scorer's raw fields for downstream
                # per-attribute metrics.
                for k, v in stats.items():
                    row[f"raw__{k}"] = v
            else:
                row["tp"] = 0
                row["fp"] = 0
                row["fn"] = 0
                row["f1"] = 0.0
                row["matched"] = 0
            rows.append(row)

    df = pd.DataFrame(rows)
    return df, n_per_organ


# --- Reductions --------------------------------------------------------------


def _per_field_per_organ(df: pd.DataFrame, *, field: str, n_boot: int,
                         alpha: float, seed: int) -> pd.DataFrame:
    """Headline F1 + hallucination/miss + count MAE, per organ + ALL."""
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        if sub.empty:
            continue
        n_total = len(sub)
        n_attempted = int(sub["attempted"].sum())
        n_eligible = int((sub["n_gold_items"] > 0).sum())
        attempted = sub[sub["attempted"]]
        if attempted.empty:
            rows.append(_empty_organ_row(field=field, organ=organ,
                                         n_total=n_total,
                                         n_attempted=0, n_eligible=n_eligible))
            continue

        tp = int(attempted["tp"].sum())
        fp = int(attempted["fp"].sum())
        fn = int(attempted["fn"].sum())
        prec_micro = tp / (tp + fp) if (tp + fp) else float("nan")
        rec_micro = tp / (tp + fn) if (tp + fn) else float("nan")
        f1_micro = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)
                    if (prec_micro and rec_micro and not np.isnan(prec_micro)
                        and not np.isnan(rec_micro)) else float("nan"))

        # Macro F1 = mean per-case F1, attempted only.
        f1_macro_arr = attempted["f1"].astype(float).to_numpy()
        f1_macro = float(f1_macro_arr.mean()) if f1_macro_arr.size else float("nan")

        # Bootstrap CI on F1 macro (case-stratified).
        records = list(zip(attempted["case_id"].tolist(),
                           f1_macro_arr.tolist()))
        if records:
            res = bootstrap_ci(
                records,
                lambda xs: float(np.mean([f for _, f in xs])),
                n_boot=n_boot, alpha=alpha, random_state=seed,
            )
            f1_macro_lo, f1_macro_hi = res.lo, res.hi
        else:
            f1_macro_lo = f1_macro_hi = float("nan")

        # Hallucination / miss
        hallucination = fp / (tp + fp) if (tp + fp) else float("nan")
        miss = fn / (tp + fn) if (tp + fn) else float("nan")
        h_lo, h_hi = (wilson_ci(fp, tp + fp, alpha) if (tp + fp)
                      else (float("nan"), float("nan")))
        m_lo, m_hi = (wilson_ci(fn, tp + fn, alpha) if (tp + fn)
                      else (float("nan"), float("nan")))

        # Count MAE
        gold_counts = sub["n_gold_items"].to_numpy()
        pred_counts = sub["n_pred_items"].to_numpy()
        count_mae = float(np.abs(gold_counts - pred_counts).mean())

        # Effective F1 — macro F1 across ALL eligible cases (parse-error
        # and field-missing cases contribute 0 F1 if gold has items).
        eligible_idx = sub["n_gold_items"] > 0
        eligible_sub = sub[eligible_idx]
        if not eligible_sub.empty:
            f1_effective = float(eligible_sub["f1"].astype(float).mean())
        else:
            f1_effective = float("nan")

        attempted_rate = n_attempted / n_total if n_total else float("nan")
        ar_lo, ar_hi = (wilson_ci(n_attempted, n_total, alpha) if n_total
                        else (float("nan"), float("nan")))

        rows.append({
            "field": field, "organ": organ,
            "n_total": n_total, "n_eligible": n_eligible,
            "n_attempted": n_attempted,
            "tp": tp, "fp": fp, "fn": fn,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "f1_micro": f1_micro,
            "attempted_f1_macro": f1_macro,
            "attempted_f1_macro_ci_lo": f1_macro_lo,
            "attempted_f1_macro_ci_hi": f1_macro_hi,
            "effective_f1_macro": f1_effective,
            "completeness_penalty_f1": (
                f1_macro - f1_effective
                if not (np.isnan(f1_macro) or np.isnan(f1_effective))
                else float("nan")
            ),
            "hallucination_rate": hallucination,
            "hallucination_rate_lo": h_lo, "hallucination_rate_hi": h_hi,
            "miss_rate": miss,
            "miss_rate_lo": m_lo, "miss_rate_hi": m_hi,
            "count_mae": count_mae,
            "count_correlation": _safe_pearson(gold_counts, pred_counts),
            "attempted_rate": attempted_rate,
            "attempted_rate_lo": ar_lo, "attempted_rate_hi": ar_hi,
        })
    return pd.DataFrame(rows)


def _empty_organ_row(*, field, organ, n_total, n_attempted, n_eligible) -> dict:
    return {
        "field": field, "organ": organ, "n_total": n_total,
        "n_eligible": n_eligible, "n_attempted": n_attempted,
        "tp": 0, "fp": 0, "fn": 0,
        "precision_micro": float("nan"), "recall_micro": float("nan"),
        "f1_micro": float("nan"),
        "attempted_f1_macro": float("nan"),
        "attempted_f1_macro_ci_lo": float("nan"),
        "attempted_f1_macro_ci_hi": float("nan"),
        "effective_f1_macro": float("nan"),
        "completeness_penalty_f1": float("nan"),
        "hallucination_rate": float("nan"),
        "hallucination_rate_lo": float("nan"),
        "hallucination_rate_hi": float("nan"),
        "miss_rate": float("nan"),
        "miss_rate_lo": float("nan"), "miss_rate_hi": float("nan"),
        "count_mae": float("nan"), "count_correlation": float("nan"),
        "attempted_rate": float("nan"),
        "attempted_rate_lo": float("nan"),
        "attempted_rate_hi": float("nan"),
    }


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _per_attribute_per_organ(df: pd.DataFrame, *, field: str,
                              alpha: float) -> pd.DataFrame:
    """Per-attribute conditional accuracy on matched pairs, with CI."""
    attrs = PER_ATTRIBUTE_KEYS[field]
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        attempted = sub[sub["attempted"]]
        if attempted.empty:
            continue
        matched_total = int(attempted["matched"].sum())
        if matched_total == 0:
            continue
        for label, raw_col in attrs:
            col = f"raw__{raw_col}"
            if col not in attempted.columns:
                continue
            n_correct = int(attempted[col].fillna(0).sum())
            acc = n_correct / matched_total
            lo, hi = wilson_ci(n_correct, matched_total, alpha)
            rows.append({
                "field": field, "organ": organ, "attribute": label,
                "matched_total": matched_total,
                "n_correct": n_correct,
                "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _case_level_per_organ(df: pd.DataFrame, *, field: str,
                           alpha: float) -> pd.DataFrame:
    """Case-level summary metrics — the clinically actionable headlines."""
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        attempted = sub[sub["attempted"]]
        if attempted.empty:
            continue
        if field == "regional_lymph_node":
            keys = [
                ("any_positive_acc", "raw__ln_any_positive_correct"),
                ("examined_acc_tol1", "raw__ln_examined_total_correct_tol"),
                ("involved_acc_tol1", "raw__ln_involved_total_correct_tol"),
            ]
            mae_keys = [
                ("examined_mae", "raw__ln_examined_total_abs_err"),
                ("involved_mae", "raw__ln_involved_total_abs_err"),
            ]
        elif field == "margins":
            keys = [
                ("any_involved_acc", "raw__margin_any_involved_correct"),
            ]
            mae_keys = []
        else:  # biomarkers
            keys = []
            mae_keys = []

        for label, col in keys:
            if col not in attempted.columns:
                continue
            n = int(attempted[col].fillna(0).sum())
            total = int(attempted[col].notna().sum())
            if total == 0:
                continue
            acc = n / total
            lo, hi = wilson_ci(n, total, alpha)
            rows.append({
                "field": field, "organ": organ, "metric": label,
                "n_correct": n, "n_total": total, "value": acc,
                "ci_lo": lo, "ci_hi": hi,
            })

        for label, col in mae_keys:
            if col not in attempted.columns:
                continue
            v = attempted[col].dropna()
            if v.empty:
                continue
            rows.append({
                "field": field, "organ": organ, "metric": label,
                "n_correct": pd.NA, "n_total": int(len(v)),
                "value": float(v.mean()),
                "ci_lo": float(v.mean() - v.std()),
                "ci_hi": float(v.mean() + v.std()),
            })
    return pd.DataFrame(rows)


def _nested_missingness(df: pd.DataFrame, *, field: str,
                          alpha: float) -> pd.DataFrame:
    """4-level missingness decomposition per organ."""
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        if sub.empty:
            continue
        n_total = len(sub)
        n_pe = int(sub["parse_error"].sum())
        n_fka = int(sub["field_key_absent"].sum())
        n_empty = int(sub["empty_list"].sum())
        n_attempted = int(sub["attempted"].sum())
        rate_pe = n_pe / n_total if n_total else float("nan")
        rate_fka = n_fka / n_total if n_total else float("nan")
        rate_empty = n_empty / n_total if n_total else float("nan")
        rate_att = n_attempted / n_total if n_total else float("nan")
        pe_lo, pe_hi = wilson_ci(n_pe, n_total, alpha) if n_total else (float("nan"), float("nan"))
        fka_lo, fka_hi = wilson_ci(n_fka, n_total, alpha) if n_total else (float("nan"), float("nan"))
        empty_lo, empty_hi = wilson_ci(n_empty, n_total, alpha) if n_total else (float("nan"), float("nan"))
        rows.append({
            "field": field, "organ": organ,
            "n_total": n_total,
            "parse_error_rate": rate_pe,
            "parse_error_rate_lo": pe_lo, "parse_error_rate_hi": pe_hi,
            "field_key_absent_rate": rate_fka,
            "field_key_absent_rate_lo": fka_lo,
            "field_key_absent_rate_hi": fka_hi,
            "empty_list_rate": rate_empty,
            "empty_list_rate_lo": empty_lo,
            "empty_list_rate_hi": empty_hi,
            "attempted_rate": rate_att,
        })
    return pd.DataFrame(rows)


def _support_distribution(df: pd.DataFrame, *, field: str) -> pd.DataFrame:
    """Per-organ histogram-style summary of |gold| and |pred| per case."""
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        if sub.empty:
            continue
        for source, col in [("gold", "n_gold_items"), ("pred", "n_pred_items")]:
            v = sub[col].astype(float)
            rows.append({
                "field": field, "organ": organ, "source": source,
                "mean": float(v.mean()),
                "median": float(v.median()),
                "p90": float(np.percentile(v, 90)) if v.size else float("nan"),
                "max": float(v.max()) if v.size else 0.0,
                "n_zero": int((v == 0).sum()),
                "n_cases": int(len(v)),
            })
    return pd.DataFrame(rows)


def _multirun_consistency(df: pd.DataFrame, *, field: str) -> pd.DataFrame:
    """SD of F1 across runs per case, missing-flip rate, mean F1 ± SD."""
    rows: list[dict] = []
    organs = sorted(df["organ"].dropna().unique().tolist()) + ["ALL"]
    for organ in organs:
        sub = df if organ == "ALL" else df[df["organ"] == organ]
        if sub.empty:
            continue
        # Pivot per case.
        pivot_f1 = sub.pivot_table(
            index="case_id", columns="run_id", values="f1", aggfunc="first",
        )
        pivot_att = sub.pivot_table(
            index="case_id", columns="run_id", values="attempted",
            aggfunc="first",
        )
        if pivot_f1.shape[1] < 2:
            continue
        # Per-case SD of F1
        sd_per_case = pivot_f1.std(axis=1, ddof=1).dropna()
        mean_per_case = pivot_f1.mean(axis=1).dropna()
        # Missing-flip: ≥1 run attempted AND ≥1 run not attempted.
        att_mat = pivot_att.astype(float).to_numpy()
        if att_mat.size:
            valid = ~np.any(np.isnan(att_mat), axis=1)
            mflips = (np.any(att_mat[valid] == 0, axis=1) &
                      np.any(att_mat[valid] == 1, axis=1))
            missing_flip_rate = (float(mflips.mean())
                                 if valid.any() else float("nan"))
        else:
            missing_flip_rate = float("nan")
        rows.append({
            "field": field, "organ": organ,
            "n_cases": int(pivot_f1.shape[0]),
            "n_runs": int(pivot_f1.shape[1]),
            "mean_per_case_f1_mean": float(mean_per_case.mean())
                if not mean_per_case.empty else float("nan"),
            "per_case_f1_sd_mean": float(sd_per_case.mean())
                if not sd_per_case.empty else float("nan"),
            "per_case_f1_sd_max": float(sd_per_case.max())
                if not sd_per_case.empty else float("nan"),
            "missing_flip_rate": missing_flip_rate,
        })
    return pd.DataFrame(rows)


# --- Helpers -----------------------------------------------------------------


def _autodiscover_runs(paths: Paths, args: argparse.Namespace) -> list[str]:
    if args.method != "llm" or not args.model:
        return []
    return [rid for rid, _ in paths.discover_runs(args.model, method="llm")]


__all__ = ["register"]
