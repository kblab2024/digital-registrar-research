"""Cascade-gated evaluation orchestrator.

CLI entry point: ``python -m scripts.eval.cli cascade``.

Walks (run, case), invokes :func:`benchmarks.eval.metrics.score_case`
under cascade gating, builds ``cascade_atomic.parquet``, and writes
the three chapter folders. Multi-model runs (e.g. gpt-oss-20b,
Qwen3-30B-A3B, Gemma-3-27B) are handled by passing
``--multi-model-roots`` so paired-test CSVs can be emitted.

Output tree under ``--out``:

    manifest.json
    cascade_atomic.parquet
    chapter1_eligibility/
        overall.csv
        confusion.csv
        multirun_consistency.csv          (when n_runs > 1)
        agreement_kappas.csv
    chapter2_organ_classification/
        overall.csv
        confusion_per_class.csv
        multirun_consistency.csv          (when n_runs > 1)
        agreement_kappas.csv
        others/
            others_ledger.csv
            others_subtype_breakdown.csv
            others_confusion.csv
            others_eligibility_audit.csv
    chapter3_field_extraction/
        per_field_overall.csv
        per_field_by_organ.csv
        per_organ_overall.csv
        nested_per_field_per_organ.csv     (margins / LN / biomarkers)
        nested_per_attribute_per_organ.csv
        biomarker_per_category.csv
        multirun_consistency.csv          (when n_runs > 1)
        cascade_funnel.csv
        conditional_accuracy_grid.csv
    model_pair_tests/                     (multi-model runs only)
        chapter1_eligibility.csv
        chapter2_organ_classification.csv
        chapter3_scalar_fields.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from digital_registrar_research.benchmarks.eval.metrics import (
    match_nested_list_filtered,
    normalize,
    score_case,
)
from digital_registrar_research.benchmarks.eval.multi_primary import (
    subgroup_label,
)
from digital_registrar_research.benchmarks.eval.nested_metrics import (
    score_lymph_nodes,
    score_margins,
)
from digital_registrar_research.benchmarks.eval.scope import (
    BIOMARKER_WHITELIST,
    biomarkers_for_organ,
    get_field_value,
    get_organ_scoreable_fields,
)
from digital_registrar_research.benchmarks.eval.stats import (
    cascade_funnel,
    conditional_accuracy_grid,
)

from .._common.args import (
    add_common_args, add_model_args, parse_cases, parse_organs,
    parse_run_ids, require_model,
)
from .._common.loaders import load_json, load_prediction, ParseError
from .._common.outcome import CaseLoad, classify_outcome
from .._common.paths import Paths, from_args
from .._common.reporting import (
    setup_logging, write_csv, write_manifest, write_parquet,
)
from .._common.stratify import organ_name
from .biomarkers import score_biomarkers_filtered
from .others import (
    build_others_row, others_confusion, others_eligibility_audit,
    others_subtype_breakdown,
)
from .paired_tests import (
    cochran_q_per_field, pairwise_mcnemar_grid, stuart_maxwell_per_organ,
)
from .reductions import (
    chapter1_confusion, chapter1_overall,
    chapter2_confusion_per_class, chapter2_overall,
    chapter3_per_field_by_organ, chapter3_per_field_overall,
    chapter3_per_organ_overall,
    chapter_multirun_reliability,
)

logger = logging.getLogger("scripts.eval.cascade")


# --- CLI registration ------------------------------------------------------

def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cascade",
        help="Cascade-gated evaluation: triage → organ → field extraction.",
        description=__doc__,
    )
    add_common_args(parser, subcommand="cascade")
    add_model_args(parser)
    parser.add_argument(
        "--multi-model-roots", nargs="*", default=None,
        help="Optional list of MODEL=METHOD/<model_slug>[/run_ids...] entries "
             "to score additional models alongside --model. When supplied, "
             "model_pair_tests/ outputs are written.",
    )
    parser.set_defaults(_handler=_main)


# --- Main ------------------------------------------------------------------

def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    require_model(args)

    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    run_ids = parse_run_ids(args) or _autodiscover_runs(paths, args)
    case_filter = parse_cases(args)

    if args.method == "llm" and not run_ids:
        raise SystemExit(
            f"no runs found under {paths.predictions_dir / 'llm' / args.model} "
            f"and --run-ids not given."
        )
    effective_runs = run_ids if args.method == "llm" else [""]

    logger.info(
        "cascade scoring: method=%s model=%s runs=%d organs=%d",
        args.method, args.model, len(effective_runs), len(organs),
    )

    atomic, ledger, n_per_organ = _build_atomic_and_ledger(
        paths=paths, args=args, run_ids=effective_runs,
        organs=organs, case_filter=case_filter,
    )
    if atomic.empty:
        logger.error("cascade atomic table empty — check inputs.")
        return 1
    logger.info("cascade atomic: %d rows | others ledger: %d rows",
                len(atomic), len(ledger))

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(atomic, args.out / "cascade_atomic.parquet")

    # --- Chapter 1: eligibility -----------------------------------------
    ch1_dir = args.out / "chapter1_eligibility"
    ch1_dir.mkdir(parents=True, exist_ok=True)
    write_csv(chapter1_overall(atomic, alpha=args.alpha),
              ch1_dir / "overall.csv")
    write_csv(chapter1_confusion(atomic),
              ch1_dir / "confusion.csv")
    if atomic["run_id"].nunique() > 1:
        write_csv(chapter_multirun_reliability(atomic, stage="A"),
                  ch1_dir / "multirun_consistency.csv")

    # --- Chapter 2: organ classification --------------------------------
    ch2_dir = args.out / "chapter2_organ_classification"
    ch2_dir.mkdir(parents=True, exist_ok=True)
    write_csv(chapter2_overall(atomic, alpha=args.alpha),
              ch2_dir / "overall.csv")
    write_csv(chapter2_confusion_per_class(atomic),
              ch2_dir / "confusion_per_class.csv")
    if atomic["run_id"].nunique() > 1:
        write_csv(chapter_multirun_reliability(atomic, stage="B"),
                  ch2_dir / "multirun_consistency.csv")

    # Others ledger.
    others_dir = ch2_dir / "others"
    others_dir.mkdir(parents=True, exist_ok=True)
    if not ledger.empty:
        write_csv(ledger, others_dir / "others_ledger.csv")
        write_csv(others_subtype_breakdown(ledger),
                  others_dir / "others_subtype_breakdown.csv")
        write_csv(others_confusion(ledger),
                  others_dir / "others_confusion.csv")
        write_csv(others_eligibility_audit(ledger),
                  others_dir / "others_eligibility_audit.csv")

    # --- Chapter 3: field extraction ------------------------------------
    ch3_dir = args.out / "chapter3_field_extraction"
    ch3_dir.mkdir(parents=True, exist_ok=True)
    write_csv(chapter3_per_field_overall(atomic, alpha=args.alpha),
              ch3_dir / "per_field_overall.csv")
    write_csv(chapter3_per_field_by_organ(atomic, alpha=args.alpha),
              ch3_dir / "per_field_by_organ.csv")
    write_csv(chapter3_per_organ_overall(atomic, alpha=args.alpha),
              ch3_dir / "per_organ_overall.csv")

    # Cascade-specific diagnostics.
    write_csv(cascade_funnel(atomic), ch3_dir / "cascade_funnel.csv")
    write_csv(conditional_accuracy_grid(atomic),
              ch3_dir / "conditional_accuracy_grid.csv")

    # --- Manifest -------------------------------------------------------
    cascade_funnel_df = cascade_funnel(atomic)
    funnel_record = {
        row["stage"]: {
            "n_total": int(row["n_total"]),
            "n_passed": int(row["n_passed"]),
            "n_dropped": int(row["n_dropped"]),
        }
        for _, row in cascade_funnel_df.iterrows()
    }
    write_manifest(
        args.out, args, subcommand="cascade",
        n_cases_per_organ=n_per_organ,
        extra={
            "n_runs": len(effective_runs),
            "run_ids": effective_runs,
            "n_atomic_rows": int(len(atomic)),
            "n_unique_cases": int(atomic["case_id"].nunique()),
            "n_others_ledger_rows": int(len(ledger)),
            "cascade_funnel": funnel_record,
        },
    )

    logger.info("cascade done. outputs in %s", args.out)
    return 0


# --- Atomic + ledger builder ----------------------------------------------

def _build_atomic_and_ledger(
    *,
    paths: Paths,
    args: argparse.Namespace,
    run_ids: Iterable[str],
    organs: Iterable[int],
    case_filter: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Walk every (run, case) and emit cascade rows."""
    n_per_organ: dict[int, int] = {}
    rows: list[dict] = []
    others_rows: list[dict] = []

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

            organ = normalize(gold.get("cancer_category")) or organ_name(
                args.dataset, organ_idx,
            )
            subgroup = subgroup_label(gold)

            pred_path = paths.prediction(
                method=args.method, model=args.model,
                run_id=run_id or None, organ_idx=organ_idx, case_id=case_id,
            )
            lo = load_prediction(pred_path)
            case_load = CaseLoad.from_load_outcome(lo)
            pred = case_load.pred or {}

            # Score under cascade gating.
            score = score_case(gold, pred)

            base = {
                "run_id": run_id or "",
                "method": args.method,
                "model": args.model or "",
                "annotator": args.annotator,
                "case_id": case_id,
                "organ_idx": int(organ_idx),
                "organ": organ,
                "subgroup": subgroup,
                "dataset": args.dataset,
            }

            # Stage A row
            stage_a = score.get("stage_a", {})
            stage_a_correct = stage_a.get("correct")
            rows.append({
                **base,
                "cascade_stage": "A",
                "gate_pass": bool(stage_a_correct) if stage_a_correct is not None else False,
                "others_disposition": score.get("others_disposition", "none"),
                "field": "cancer_excision_report",
                "field_kind": "binary",
                "gold_present": stage_a.get("gold") is not None,
                "attempted": stage_a_correct is not None,
                "correct": (bool(stage_a_correct)
                            if stage_a_correct is not None else None),
                "wrong": (not bool(stage_a_correct)
                          if stage_a_correct is not None else None),
                "field_missing": stage_a_correct is None and case_load.ok,
                "parse_error": not case_load.ok,
                "error_mode": case_load.error_mode,
                "gold_value": stage_a.get("gold"),
                "pred_value": stage_a.get("pred"),
            })

            others_disposition = score.get("others_disposition", "none")

            # Record others ledger if applicable (regardless of stage).
            if others_disposition != "none":
                others_rows.append(build_others_row(
                    run_id=run_id or "", model=args.model or "",
                    dataset=args.dataset,
                    case_id=case_id, organ_idx=int(organ_idx),
                    gold=gold, pred=pred,
                    stage_a_correct=stage_a_correct,
                    others_disposition=others_disposition,
                ))

            # Stage B row (only if Stage A passed).
            if "stage_b" in score:
                stage_b = score["stage_b"]
                stage_b_correct = stage_b.get("correct")
                rows.append({
                    **base,
                    "cascade_stage": "B",
                    "gate_pass": (bool(stage_b_correct)
                                  if stage_b_correct is not None else False),
                    "others_disposition": others_disposition,
                    "field": "cancer_category",
                    "field_kind": "nominal",
                    "gold_present": stage_b.get("gold") is not None,
                    "attempted": stage_b_correct is not None,
                    "correct": (bool(stage_b_correct)
                                if stage_b_correct is not None else None),
                    "wrong": (not bool(stage_b_correct)
                              if stage_b_correct is not None else None),
                    "field_missing": stage_b_correct is None and case_load.ok,
                    "parse_error": not case_load.ok,
                    "error_mode": case_load.error_mode,
                    "gold_value": stage_b.get("gold"),
                    "pred_value": stage_b.get("pred"),
                })

            # Stage C rows (only if cascade let through).
            if score.get("stage_c_eligible"):
                rows.extend(_emit_stage_c_rows(
                    base=base, score=score, gold=gold, pred=pred,
                    case_load=case_load,
                    others_disposition=others_disposition,
                ))

    atomic = pd.DataFrame(rows)
    ledger = pd.DataFrame(others_rows)
    return atomic, ledger, n_per_organ


def _emit_stage_c_rows(
    *,
    base: dict,
    score: dict,
    gold: dict,
    pred: dict,
    case_load: CaseLoad,
    others_disposition: str,
) -> list[dict]:
    """Emit one Stage-C row per scored field for the case."""
    rows: list[dict] = []
    organ = base["organ"]

    # Scalar fields scored by score_case (skip the cascade keys).
    skip = {
        "stage_a", "stage_b", "stage_c_eligible", "_nested",
        "others_disposition", "cancer_excision_report", "cancer_category",
    }
    for field, correct in score.items():
        if field in skip:
            continue
        if isinstance(correct, dict):
            continue  # _nested scoring handled separately
        # Field kind from organ scope; default to nominal.
        kind = "nominal"
        try:
            kind = get_organ_scoreable_fields(organ).get(field, "nominal")
        except Exception:
            pass
        if field.startswith("biomarker_"):
            kind = "binary"
        g_val = get_field_value(gold, field)
        p_val = get_field_value(pred, field) if case_load.ok else None
        attempted = correct is not None
        rows.append({
            **base,
            "cascade_stage": "C",
            "gate_pass": True,  # by definition we're past A and B
            "others_disposition": others_disposition,
            "field": field,
            "field_kind": kind,
            "gold_present": g_val is not None,
            "attempted": attempted,
            "correct": (bool(correct) if correct is not None else None),
            "wrong": (not bool(correct) if correct is not None else None),
            "field_missing": (not attempted) and case_load.ok,
            "parse_error": not case_load.ok,
            "error_mode": case_load.error_mode,
            "gold_value": g_val,
            "pred_value": p_val,
        })
    return rows


# --- Helpers --------------------------------------------------------------

def _autodiscover_runs(paths: Paths, args: argparse.Namespace) -> list[str]:
    if args.method != "llm" or not args.model:
        return []
    return [rid for rid, _ in paths.discover_runs(args.model, method="llm")]


__all__ = ["register", "_main"]
