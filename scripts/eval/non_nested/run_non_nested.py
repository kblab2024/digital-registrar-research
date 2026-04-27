"""Non-nested subcommand orchestrator.

Builds the atomic outcome table from (method, model, run_ids,
annotator), then dispatches to ``metrics_non_nested.py`` to produce all
the per-field, per-organ, per-subgroup summaries.

Output tree under ``--out``:

    manifest.json
    correctness_table.parquet
    per_field_overall.csv
    per_field_by_organ.csv
    per_field_by_subgroup.csv
    headline_classification.csv
    confusion/<field>__<organ>.csv
    per_class_prf1.csv
    confusion_pairs.csv
    accuracy_collapsing_neighbors.csv
    rank_distance.csv
    top_k_ordinal.csv
    schema_conformance.csv
    refusal_calibration.csv
    run_consistency.csv
    section_rollup.csv
    missingness_summary.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

import pandas as pd

from digital_registrar_research.benchmarks.eval.completeness import (
    aggregate_missingness, refusal_calibration,
)
from digital_registrar_research.benchmarks.eval.iaa import classify_field
from digital_registrar_research.benchmarks.eval.metrics import (
    is_attempted, normalize,
)
from digital_registrar_research.benchmarks.eval.multi_primary import (
    subgroup_label,
)
from digital_registrar_research.benchmarks.eval.scope import (
    BREAST_BIOMARKERS, FAIR_SCOPE, get_field_value,
    get_organ_scoreable_fields,
)
from digital_registrar_research.benchmarks.eval.scope_organs import (
    ORGAN_LIST_OF_LITERALS,
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
from . import metrics_non_nested as M

logger = logging.getLogger("scripts.eval.non_nested")


# --- Argparse registration --------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "non_nested",
        help="Score scalar (non-nested) fields with three-way outcome model.",
        description=__doc__,
    )
    add_common_args(parser)
    add_model_args(parser)
    parser.set_defaults(_handler=_main)


# --- Main entry point -------------------------------------------------------


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

    # If method has no run dimension (clinicalbert, rule_based), treat as
    # a single-element run list with run_id="" so the rest of the
    # pipeline doesn't special-case it.
    effective_runs = run_ids if args.method == "llm" else [""]

    logger.info(
        "scoring method=%s model=%s annotator=%s with %d run(s) over %d organ(s)",
        args.method, args.model, args.annotator, len(effective_runs), len(organs),
    )

    atomic, n_per_organ = _build_atomic_table(
        paths=paths, args=args, run_ids=effective_runs,
        organs=organs, case_filter=case_filter,
    )
    if atomic.empty:
        logger.error("atomic table is empty — nothing to score. "
                     "Check --root / --dataset / --annotator / --model.")
        return 1
    logger.info("atomic table: %d rows", len(atomic))

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(atomic, args.out / "correctness_table.parquet")

    # --- Headline summaries -------------------------------------------------

    per_field_overall = M.per_field_summary(
        atomic, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
    )
    write_csv(per_field_overall, args.out / "per_field_overall.csv")

    per_field_by_organ = M.per_field_by_organ_summary(
        atomic, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
    )
    write_csv(per_field_by_organ, args.out / "per_field_by_organ.csv")

    per_field_by_subgroup = M.per_field_subgroup_summary(
        atomic, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
    )
    write_csv(per_field_by_subgroup, args.out / "per_field_by_subgroup.csv")

    # Per-organ aggregate (across all that organ's fields) + cross-organ ALL row.
    per_organ_overall = M.per_organ_overall_summary(
        atomic, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
    )
    write_csv(per_organ_overall, args.out / "per_organ_overall.csv")

    # --- Classification metrics --------------------------------------------

    organ_universe = sorted(atomic["organ"].dropna().unique().tolist()) + ["ALL"]
    fields = sorted(atomic["field"].dropna().unique().tolist())

    headline_rows = []
    perclass_frames = []
    confusion_pairs_frames = []
    collapsing_rows = []
    rank_dist_rows = []
    topk_rows = []
    confusion_dir = args.out / "confusion"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for field in fields:
        for organ in organ_universe:
            cm = M.confusion_for_field(atomic, field=field, organ=organ)
            if cm is not None and not cm.empty:
                # Only persist non-degenerate matrices to keep file count tidy.
                if cm["count"].sum() > 0:
                    write_csv(
                        cm,
                        confusion_dir / f"{_safe(field)}__{_safe(organ)}.csv",
                    )
            pc = M.per_class_for_field(atomic, field=field, organ=organ)
            if pc is not None and not pc.empty:
                perclass_frames.append(pc)
            head = M.headline_classification_metrics(
                atomic, field=field, organ=organ,
            )
            if head:
                headline_rows.append(head)
            cp = M.confusion_pairs_for_field(atomic, field=field, organ=organ)
            if cp is not None and not cp.empty:
                confusion_pairs_frames.append(cp)
            col = M.accuracy_collapsing_neighbors(
                atomic, field=field, organ=organ,
            )
            if col:
                collapsing_rows.append(col)
            rd = M.rank_distance_for_field(atomic, field=field, organ=organ)
            if rd:
                rank_dist_rows.append(rd)
            for k in (1, 2):
                tk = M.top_k_for_ordinal_field(
                    atomic, field=field, organ=organ, k=k,
                )
                if tk:
                    topk_rows.append(tk)

    if headline_rows:
        write_csv(
            pd.DataFrame(headline_rows),
            args.out / "headline_classification.csv",
        )
    if perclass_frames:
        write_csv(
            pd.concat(perclass_frames, ignore_index=True),
            args.out / "per_class_prf1.csv",
        )
    if confusion_pairs_frames:
        write_csv(
            pd.concat(confusion_pairs_frames, ignore_index=True),
            args.out / "confusion_pairs.csv",
        )
    if collapsing_rows:
        write_csv(
            pd.DataFrame(collapsing_rows),
            args.out / "accuracy_collapsing_neighbors.csv",
        )
    if rank_dist_rows:
        write_csv(
            pd.DataFrame(rank_dist_rows),
            args.out / "rank_distance.csv",
        )
    if topk_rows:
        write_csv(
            pd.DataFrame(topk_rows),
            args.out / "top_k_ordinal.csv",
        )

    # --- Schema conformance / refusal calibration --------------------------

    write_csv(M.schema_conformance(atomic), args.out / "schema_conformance.csv")

    # Refusal calibration via the src/.../eval helper.
    write_csv(
        refusal_calibration(atomic, by=("field", "organ")),
        args.out / "refusal_calibration.csv",
    )

    # --- Multi-run consistency ---------------------------------------------

    if atomic["run_id"].nunique() > 1:
        cons = M.run_consistency_extended(atomic)
        write_csv(cons, args.out / "run_consistency.csv")

    # --- Section / fieldtype rollup ----------------------------------------

    section_of_field = _section_map(fields, atomic)
    write_csv(
        M.section_rollup(
            atomic, section_of_field=section_of_field,
            n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
        ),
        args.out / "section_rollup.csv",
    )

    # --- Missingness summary (uses src/.../eval/completeness) --------------

    write_csv(
        aggregate_missingness(atomic, by=("field", "organ")),
        args.out / "missingness_summary.csv",
    )

    # --- Manifest ----------------------------------------------------------

    write_manifest(
        args.out, args, subcommand="non_nested",
        n_cases_per_organ=n_per_organ,
        extra={
            "n_runs": len(effective_runs),
            "run_ids": effective_runs,
            "n_atomic_rows": int(len(atomic)),
            "n_unique_cases": int(atomic["case_id"].nunique()),
            "n_unique_fields": int(atomic["field"].nunique()),
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Atomic-table builder ---------------------------------------------------


def _build_atomic_table(
    *,
    paths: Paths,
    args: argparse.Namespace,
    run_ids: Iterable[str],
    organs: Iterable[int],
    case_filter: set[str] | None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Score every (run, case, organ-scoreable field) tuple → long DataFrame.

    The field list is **per organ** — every categorical, boolean, span,
    and list-of-literals field declared for that organ in
    ``scope_organs``. This produces ~25–30 fields per organ rather than
    the 12 in ``FAIR_SCOPE``, giving full coverage of the per-organ
    schema.

    Returns ``(atomic_df, n_cases_per_organ)`` for the manifest.
    """
    n_per_organ: dict[int, int] = {}
    rows: list[dict] = []

    case_index: list[tuple[int, str]] = []
    for organ_idx, case_id in paths.case_ids(args.annotator, tuple(organs)):
        if case_filter and case_id not in case_filter:
            continue
        case_index.append((organ_idx, case_id))
    for oi, _ in case_index:
        n_per_organ[oi] = n_per_organ.get(oi, 0) + 1

    # Pre-resolve the per-organ field list so we don't re-derive it per case.
    fields_by_organ: dict[str, dict[str, str]] = {}
    biomarker_fields = [f"biomarker_{b}" for b in BREAST_BIOMARKERS]

    for run_id in run_ids:
        for organ_idx, case_id in case_index:
            gold_path = paths.annotation(args.annotator, organ_idx, case_id)
            try:
                gold = load_json(gold_path)
            except ParseError as e:
                logger.warning("skipping %s: %s", case_id, e)
                continue

            organ = organ_name(organ_idx)
            organ_from_gold = normalize(gold.get("cancer_category"))
            if organ_from_gold and organ_from_gold != "others":
                organ = organ_from_gold

            subgroup = subgroup_label(gold)

            pred_path = paths.prediction(
                method=args.method, model=args.model,
                run_id=run_id or None, organ_idx=organ_idx, case_id=case_id,
            )
            lo = load_prediction(pred_path)
            case_load = CaseLoad.from_load_outcome(lo)

            # Build the per-organ field list once per organ, cached.
            if organ not in fields_by_organ:
                organ_fields = get_organ_scoreable_fields(organ)
                # cancer_category and cancer_excision_report are top-level —
                # not declared per-organ in scope_organs; add them explicitly.
                organ_fields.setdefault("cancer_category", "nominal")
                organ_fields.setdefault("cancer_excision_report", "binary")
                fields_by_organ[organ] = organ_fields

            field_kinds = fields_by_organ[organ]
            # Score every per-organ field plus breast biomarkers (when applicable).
            field_iter = list(field_kinds.keys())
            if organ == "breast":
                field_iter += biomarker_fields

            for field in field_iter:
                if field.startswith("biomarker_"):
                    if organ != "breast":
                        continue
                    out = _classify_biomarker(gold, case_load, field)
                    field_kind = "binary"
                else:
                    out = classify_outcome(gold, case_load, field, organ=organ)
                    field_kind = field_kinds.get(field) or classify_field(field, organ)

                # Skip rows that are uninformative: gold absent AND pred
                # absent AND no parse_error. Keep rows where pred attempted
                # (for OOV / refusal calibration) and rows where gold has
                # an answer the model might have missed.
                if (not out.gold_present and not out.attempted
                        and not out.parse_error):
                    continue

                rows.append({
                    "run_id": run_id or "",
                    "method": args.method,
                    "model": args.model or "",
                    "annotator": args.annotator,
                    "case_id": case_id,
                    "organ_idx": int(organ_idx),
                    "organ": organ,
                    "subgroup": subgroup,
                    "field": field,
                    "field_kind": field_kind,
                    "gold_present": out.gold_present,
                    "attempted": out.attempted,
                    "correct": out.correct,
                    "wrong": out.wrong,
                    "field_missing": out.field_missing,
                    "parse_error": out.parse_error,
                    "error_mode": out.error_mode,
                    "gold_value": out.gold_value,
                    "pred_value": out.pred_value,
                })

    df = pd.DataFrame(rows)
    return df, n_per_organ


def _classify_biomarker(gold: dict, case_load: CaseLoad, field: str):
    """Specialised classifier for ``biomarker_<category>`` synthetic fields.

    The biomarker category lives nested in ``biomarkers`` list. We
    extract by ``biomarker_category`` key and route through
    ``classify_outcome`` using a synthetic flat representation.
    """
    from .._common.outcome import Outcome

    cat = field.removeprefix("biomarker_")
    g_list = get_field_value(gold, "biomarkers") or []
    g_entry = next(
        (b for b in g_list if normalize(b.get("biomarker_category")) == cat),
        None,
    )
    g_value = g_entry.get("expression") if isinstance(g_entry, dict) else None
    gold_present = g_value is not None

    if not case_load.ok:
        return Outcome(
            kind="parse_error",
            gold_present=gold_present, parse_error=True,
            field_missing=False, attempted=False,
            correct=False, wrong=False,
            error_mode=case_load.error_mode,
            pred_value=None, gold_value=g_value,
        )

    pred = case_load.pred or {}
    p_list = get_field_value(pred, "biomarkers") or []
    p_entry = next(
        (b for b in p_list if normalize(b.get("biomarker_category")) == cat),
        None,
    )
    if p_entry is None or not is_attempted(p_entry, "expression"):
        return Outcome(
            kind="field_missing",
            gold_present=gold_present, parse_error=False,
            field_missing=True, attempted=False,
            correct=False, wrong=False, error_mode=None,
            pred_value=None, gold_value=g_value,
        )
    p_value = p_entry.get("expression")
    is_correct = normalize(p_value) == normalize(g_value)
    return Outcome(
        kind="correct" if is_correct else "wrong",
        gold_present=gold_present, parse_error=False,
        field_missing=False, attempted=True,
        correct=is_correct, wrong=not is_correct,
        error_mode=None, pred_value=p_value, gold_value=g_value,
    )


# --- Helpers ----------------------------------------------------------------


def _autodiscover_runs(paths: Paths, args: argparse.Namespace) -> list[str]:
    if args.method != "llm" or not args.model:
        return []
    return [rid for rid, _ in paths.discover_runs(args.model, method="llm")]


def _section_map(fields: Iterable[str], atomic: pd.DataFrame) -> dict[str, str]:
    """Map each field to its high-level section.

    Sections: ``top_level`` (cancer_excision_report, cancer_category),
    ``staging`` (T/N/M, stage groups), ``grading`` (grade family),
    ``invasion`` (LVI, PNI), ``size`` (tumor_size etc), ``biomarker``,
    ``other``.
    """
    out: dict[str, str] = {}
    for f in fields:
        if f in {"cancer_excision_report", "cancer_category"}:
            out[f] = "top_level"
        elif f in {"pt_category", "pn_category", "pm_category",
                   "tnm_descriptor", "stage_group", "overall_stage",
                   "pathologic_stage_group", "anatomic_stage_group",
                   "ajcc_version"}:
            out[f] = "staging"
        elif "grade" in f or f in {"nuclear_grade", "tubule_formation",
                                    "mitotic_rate", "total_score",
                                    "dcis_grade"}:
            out[f] = "grading"
        elif f in {"lymphovascular_invasion", "perineural_invasion"}:
            out[f] = "invasion"
        elif "size" in f:
            out[f] = "size"
        elif f.startswith("biomarker_"):
            out[f] = "biomarker"
        else:
            out[f] = "other"
    return out


def _safe(s: str) -> str:
    """Make a string safe for use as a filename component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


__all__ = ["register"]
