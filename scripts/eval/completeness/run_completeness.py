"""Completeness subcommand orchestrator.

Takes one or more (method, model, run_ids) triples plus a reference
annotator. Builds a unified atomic table across all methods, then
emits the cross-method missingness deltas, error-mode decomposition,
and the modularity-advantage table.

Output tree:
    manifest.json
    missingness_atomic.parquet
    per_method_per_field.csv
    per_method_per_organ.csv
    per_method_per_fieldtype.csv
    error_mode_decomposition.csv
    method_pairwise_deltas.csv      — paired Δ + McNemar between every method pair
    modularity_advantage.csv        — sorted ablation headline
    schema_conformance_deltas.csv
    refusal_calibration.csv
    position_in_schema_correlation.csv
    heatmap_attempted_rate.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

import pandas as pd

from digital_registrar_research.benchmarks.eval.completeness import (
    aggregate_missingness, method_pair_deltas,
    position_in_schema_correlation, refusal_calibration,
)
from digital_registrar_research.benchmarks.eval.iaa import classify_field
from digital_registrar_research.benchmarks.eval.metrics import normalize
from digital_registrar_research.benchmarks.eval.scope import (
    BREAST_BIOMARKERS, FAIR_SCOPE,
)

from .._common.args import (
    add_common_args, parse_cases, parse_organs, parse_run_ids,
)
from .._common.loaders import load_json, load_prediction, ParseError
from .._common.outcome import CaseLoad, classify_outcome
from .._common.paths import KNOWN_METHODS, Paths, from_args
from .._common.reporting import (
    setup_logging, write_csv, write_manifest, write_parquet,
)
from .._common.stratify import organ_name

logger = logging.getLogger("scripts.eval.completeness")


# --- Argparse registration --------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "completeness",
        help="Detailed missingness analysis across methods.",
        description=__doc__,
    )
    add_common_args(parser, subcommand="completeness")
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="One or more 'method:model[:run_ids_csv]' triples. "
             "Examples: 'llm:gpt_oss_20b:run01,run02' "
             "(omitting run_ids means autodiscover); "
             "'clinicalbert:v2_finetuned'; 'rule_based:'.",
    )
    parser.add_argument(
        "--annotator", default="gold",
        help="Annotator subdir to score against (default: %(default)s).",
    )
    parser.set_defaults(_handler=_main)


# --- Main entry --------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    case_filter = parse_cases(args)
    method_specs = [_parse_method_spec(s) for s in args.methods]

    # Build atomic across methods.
    frames: list[pd.DataFrame] = []
    n_per_organ: dict[int, int] = {}
    for method, model, run_ids in method_specs:
        if method == "llm" and not run_ids:
            run_ids = [rid for rid, _ in paths.discover_runs(model)]
        if method == "llm" and not run_ids:
            logger.warning("no runs for %s/%s, skipping", method, model)
            continue
        effective_runs = run_ids if method == "llm" else [""]
        df, npo = _build_atomic_for_method(
            paths=paths, method=method, model=model,
            run_ids=effective_runs, annotator=args.annotator,
            organs=organs, case_filter=case_filter,
        )
        if df.empty:
            continue
        frames.append(df)
        for k, v in npo.items():
            n_per_organ[k] = max(n_per_organ.get(k, 0), v)

    if not frames:
        logger.error("no atomic rows produced; check --methods.")
        return 1
    atomic = pd.concat(frames, ignore_index=True)
    logger.info("atomic table: %d rows across %d methods",
                len(atomic), atomic["method_label"].nunique())

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(atomic, args.out / "missingness_atomic.parquet")

    # --- Aggregations -----------------------------------------------------

    write_csv(
        aggregate_missingness(atomic, by=("method_label", "field")),
        args.out / "per_method_per_field.csv",
    )
    write_csv(
        aggregate_missingness(atomic, by=("method_label", "organ")),
        args.out / "per_method_per_organ.csv",
    )
    write_csv(
        aggregate_missingness(atomic, by=("method_label", "field_kind")),
        args.out / "per_method_per_fieldtype.csv",
    )

    # --- Error-mode decomposition -----------------------------------------
    em = (
        atomic[atomic["parse_error"]]
        .assign(error_mode=atomic["error_mode"].fillna("other"))
        .groupby(["method_label", "error_mode"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not em.empty:
        totals = em.groupby("method_label")["count"].transform("sum")
        em["share"] = em["count"] / totals
        write_csv(em, args.out / "error_mode_decomposition.csv")

    # --- Method-pair Δ ---------------------------------------------------
    deltas = _method_pair_deltas_using_label(atomic, by=("field", "organ"))
    if not deltas.empty:
        write_csv(deltas, args.out / "method_pairwise_deltas.csv")

    # --- Modularity advantage ---------------------------------------------
    if not deltas.empty:
        adv = (deltas[["method_a", "method_b", "field", "organ",
                       "delta_attempted_rate", "n_paired"]]
               .sort_values("delta_attempted_rate",
                            key=lambda s: s.abs(), ascending=False))
        write_csv(adv, args.out / "modularity_advantage.csv")

    # --- Heatmap (rows=fields, cols=methods) -----------------------------
    heatmap = (
        atomic.groupby(["method_label", "field"])["attempted"]
        .mean()
        .reset_index(name="attempted_rate")
        .pivot(index="field", columns="method_label", values="attempted_rate")
        .reset_index()
    )
    write_csv(heatmap, args.out / "heatmap_attempted_rate.csv", index=False)

    # --- Refusal calibration ---------------------------------------------
    write_csv(
        refusal_calibration(atomic, by=("method_label", "field", "organ")),
        args.out / "refusal_calibration.csv",
    )

    # --- Schema conformance Δ between methods ----------------------------
    # Skip when the atomic table doesn't carry pred_value (lightweight
    # build path); user can re-run non_nested for this dimension.
    if "pred_value" in atomic.columns:
        from scripts.eval.non_nested.metrics_non_nested import schema_conformance
        sc_rows = []
        for method, sub in atomic.groupby("method_label"):
            f = schema_conformance(sub)
            if not f.empty:
                f["method_label"] = method
                sc_rows.append(f)
        if sc_rows:
            write_csv(pd.concat(sc_rows, ignore_index=True),
                      args.out / "schema_conformance_per_method.csv")

    # --- Position-in-schema correlation ---------------------------------
    field_order = list(FAIR_SCOPE) + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    # Use method_label (cross-method axis) as the grouping key, so drop
    # the inner ``method`` column first to avoid duplicate names.
    pos_input = atomic.drop(columns=["method"]).rename(
        columns={"method_label": "method"}
    )
    write_csv(
        position_in_schema_correlation(
            pos_input,
            field_order=field_order, by="method",
        ),
        args.out / "position_in_schema_correlation.csv",
    )

    write_manifest(
        args.out, args, subcommand="completeness",
        n_cases_per_organ=n_per_organ,
        extra={
            "method_specs": [_describe_spec(s) for s in method_specs],
            "n_atomic_rows": int(len(atomic)),
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Method-spec parsing ----------------------------------------------------


def _parse_method_spec(s: str) -> tuple[str, str | None, list[str] | None]:
    parts = s.split(":")
    if len(parts) < 1 or parts[0] not in KNOWN_METHODS:
        raise SystemExit(
            f"--methods entry must start with {KNOWN_METHODS}: got {s!r}"
        )
    method = parts[0]
    model = parts[1] if len(parts) > 1 and parts[1] else None
    run_ids = (parts[2].split(",") if len(parts) > 2 and parts[2] else None)
    if method != "rule_based" and not model:
        raise SystemExit(f"--methods entry needs model: {s!r}")
    return method, model, run_ids


def _describe_spec(spec: tuple[str, str | None, list[str] | None]) -> str:
    method, model, runs = spec
    return f"{method}/{model or ''}/{','.join(runs or []) or 'auto'}"


# --- Atomic-table builder (per-method) --------------------------------------


def _build_atomic_for_method(
    *, paths: Paths, method: str, model: str | None,
    run_ids: Iterable[str], annotator: str,
    organs: Iterable[int], case_filter: set[str] | None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Build the atomic outcome table for one method.

    Adds a ``method_label`` column = ``"<method>/<model>"`` for
    cross-method comparison (so two LLM models can be compared).
    """
    fields = list(FAIR_SCOPE) + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    rows: list[dict] = []
    n_per_organ: dict[int, int] = {}
    method_label = f"{method}/{model or 'rule_based'}"

    case_index: list[tuple[int, str]] = []
    for organ_idx, case_id in paths.case_ids(annotator, tuple(organs)):
        if case_filter and case_id not in case_filter:
            continue
        case_index.append((organ_idx, case_id))
    for oi, _ in case_index:
        n_per_organ[oi] = n_per_organ.get(oi, 0) + 1

    for run_id in run_ids:
        for organ_idx, case_id in case_index:
            try:
                gold = load_json(paths.annotation(annotator, organ_idx, case_id))
            except ParseError as e:
                logger.warning("skipping %s: %s", case_id, e)
                continue
            organ = normalize(gold.get("cancer_category")) or organ_name(organ_idx)
            pred_path = paths.prediction(
                method=method, model=model, run_id=run_id or None,
                organ_idx=organ_idx, case_id=case_id,
            )
            lo = load_prediction(pred_path)
            case_load = CaseLoad.from_load_outcome(lo)
            for field in fields:
                if field.startswith("biomarker_") and organ != "breast":
                    continue
                # Use the same biomarker-aware classifier path as non_nested.
                from scripts.eval.non_nested.run_non_nested import (
                    _classify_biomarker,
                )
                if field.startswith("biomarker_"):
                    out = _classify_biomarker(gold, case_load, field)
                else:
                    out = classify_outcome(gold, case_load, field)

                if (not out.gold_present and not out.attempted
                        and not out.parse_error):
                    continue
                rows.append({
                    "run_id": run_id or "",
                    "method": method,
                    "model": model or "",
                    "method_label": method_label,
                    "annotator": annotator,
                    "case_id": case_id,
                    "organ_idx": int(organ_idx),
                    "organ": organ,
                    "field": field,
                    "field_kind": classify_field(field, organ),
                    "gold_present": out.gold_present,
                    "attempted": out.attempted,
                    "correct": out.correct,
                    "wrong": out.wrong,
                    "field_missing": out.field_missing,
                    "parse_error": out.parse_error,
                    "error_mode": out.error_mode,
                })
    return pd.DataFrame(rows), n_per_organ


def _method_pair_deltas_using_label(df: pd.DataFrame, *, by) -> pd.DataFrame:
    """Wrapper to call src/.../eval/completeness.method_pair_deltas with
    ``method_label`` instead of ``method`` as the grouping column."""
    df = df.rename(columns={"method": "_orig_method", "method_label": "method"})
    out = method_pair_deltas(df, by=by)
    if out.empty:
        return out
    return out


__all__ = ["register"]
