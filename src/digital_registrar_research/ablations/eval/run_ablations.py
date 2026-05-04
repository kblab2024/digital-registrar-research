"""
Aggregate ablation cell predictions written under the canonical layout.

Canonical layout (see ``runners/_base.py``):

    {root}/results/ablations/{dataset}/{cell_id}/{model_slug}/{run_id}/{organ_n}/{case_id}.json

Per-cell × per-model × per-run grading reuses
:mod:`benchmarks.eval.metrics`. Per-run results are emitted then
aggregated across runs (mean per (cell, model, field)).

Output files, all under ``--results-root`` (default:
``{folder}/results/ablations/{dataset}/``):

    ablation_grid.csv         long-form: one row per (cell, model, run, case, field)
    ablation_summary.csv      per-(cell, model, field): accuracy + coverage
    ablation_table.csv        pivot: rows=field, cols=<cell>_<model>, cells=accuracy
    cell_deltas.csv           per-field deltas vs the configured baseline
    efficiency.csv            mean / median latency, schema-error rate, parse-error rate

Statistical CSVs (``ablation_paired_deltas.csv`` etc.) are written by
:mod:`stats` when ``--with-stats`` is on (default for non-smoke
results-roots).

Usage::

    python -m digital_registrar_research.ablations.eval.run_ablations \\
        --folder dummy --dataset tcga
    python -m digital_registrar_research.ablations.eval.run_ablations \\
        --results-root /custom/path/ablations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ...benchmarks.eval.metrics import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
    match_nested_list,
    score_case,
    summary_table,
)
from ...benchmarks.eval.scope import IMPLEMENTED_ORGANS

DEFAULT_BASELINE = "dspy_modular"


# ---------------------------------------------------------------------------
# Canonical path resolution
# ---------------------------------------------------------------------------

def _ablations_root(args: argparse.Namespace) -> Path:
    """Resolve the per-dataset ablations root from args.

    Three input shapes (in order of precedence):

    1. ``--results-root <path>`` — an absolute or relative path to a
       directory containing per-cell subdirs. Used as-is.
    2. ``--folder <root> --dataset <name>`` — canonical;
       resolves to ``{root}/results/ablations/{dataset}``.
    3. Neither — falls back to
       :data:`digital_registrar_research.paths.ABLATIONS_RESULTS`.
    """
    if args.results_root is not None:
        return Path(args.results_root)
    if args.experiment_root is not None and args.dataset:
        return (Path(args.experiment_root) / "results" / "ablations"
                / args.dataset)
    from ...paths import ABLATIONS_RESULTS
    return ABLATIONS_RESULTS


def _gold_root(args: argparse.Namespace) -> Path:
    """Locate the gold annotation tree.

    Canonical: ``{folder}/data/{dataset}/annotations/gold/``. Falls
    back to the legacy ``GOLD_ANNOTATIONS`` constant if --folder isn't
    given.
    """
    if args.experiment_root is not None and args.dataset:
        return (Path(args.experiment_root) / "data" / args.dataset
                / "annotations" / "gold")
    from ...paths import GOLD_ANNOTATIONS
    return GOLD_ANNOTATIONS


# ---------------------------------------------------------------------------
# Canonical-tree discovery
# ---------------------------------------------------------------------------

def _discover_runs(ablations_root: Path,
                   cells: list[str] | None = None,
                   models: list[str] | None = None,
                   ) -> list[tuple[str, str, str, Path]]:
    """Yield ``(cell_id, model_slug, run_id, run_dir)`` for every
    completed (with ``_summary.json``) run under the canonical tree."""
    if not ablations_root.is_dir():
        return []
    out: list[tuple[str, str, str, Path]] = []
    for cell_dir in sorted(ablations_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        if cells and cell_dir.name not in cells:
            continue
        for model_dir in sorted(cell_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            if models and model_dir.name not in models:
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                if (run_dir / "_summary.json").exists():
                    out.append((cell_dir.name, model_dir.name,
                                run_dir.name, run_dir))
    return out


def _gold_for(case_id: str, organ_n: str, gold_root: Path) -> dict | None:
    """Read the gold annotation for ``(organ_n, case_id)`` from the
    canonical layout. Returns None if missing."""
    gold_path = gold_root / organ_n / f"{case_id}.json"
    if not gold_path.exists():
        return None
    try:
        with gold_path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Case- and field-level status classification
# ---------------------------------------------------------------------------
#
# Mutually-exclusive case_status precedence. Earlier entries win when a
# case carries multiple sentinels.
CASE_STATUS_PRECEDENCE: tuple[str, ...] = (
    "gold_missing",
    "prediction_unreadable",
    "pipeline_error",
    "parse_error",
    "schema_error",
    "renest_error",
    "b2_parse_error",
    "section_error",
    "skipped_intentional",
    "ok",
)

# Case statuses under which a field can still be graded normally. The
# others render every field unscoreable_due_to_case_error.
_GRADABLE_CASE_STATUS: frozenset[str] = frozenset({
    "ok",
    "schema_error",
    "renest_error",
    "b2_parse_error",
    "section_error",
})


def _classify_case(pred: dict | None,
                   pred_unreadable_reason: str | None = None,
                   gold_missing: bool = False,
                   ) -> tuple[str, list[str]]:
    """Resolve a case-level status from runner sentinel keys.

    Reads the runner-emitted sentinels (``_pipeline_error``,
    ``_parse_error`` / ``_error``, ``_schema_errors``,
    ``_renest_errors``, ``_b2_parse_errors``, ``_per_section_errors``,
    ``_skip_reason``) and resolves to a single ``case_status`` plus the
    full list of truthy flags.

    Args:
        pred: parsed prediction dict, or None if unreadable.
        pred_unreadable_reason: short repr of the parse exception, if
            the prediction file failed to load. Forces
            ``prediction_unreadable``.
        gold_missing: whether the matching gold annotation is absent.
            Forces ``gold_missing``.

    Returns:
        ``(case_status, flags)`` — ``flags`` is the list of every
        truthy sentinel observed, in precedence order. Useful for
        diagnostic columns; the chosen ``case_status`` is the first
        entry of the precedence list that fired.
    """
    flags: list[str] = []
    if gold_missing:
        flags.append("gold_missing")
    if pred_unreadable_reason is not None:
        flags.append("prediction_unreadable")
    if isinstance(pred, dict):
        if pred.get("_pipeline_error"):
            flags.append("pipeline_error")
        if pred.get("_parse_error") or pred.get("_error"):
            flags.append("parse_error")
        if pred.get("_schema_errors"):
            flags.append("schema_error")
        if pred.get("_renest_errors"):
            flags.append("renest_error")
        if pred.get("_b2_parse_errors"):
            flags.append("b2_parse_error")
        if pred.get("_per_section_errors"):
            flags.append("section_error")
        if pred.get("_skip_reason") in ("not_cancer", "unknown_organ"):
            flags.append("skipped_intentional")
    if not flags:
        flags.append("ok")
    # Precedence resolution.
    for status in CASE_STATUS_PRECEDENCE:
        if status in flags:
            return status, flags
    return "ok", flags


def _classify_field(case_status: str,
                    gold: dict | None,
                    pred: dict | None,
                    field: str,
                    scored: object,
                    is_nested: bool = False,
                    schema_errors: list[str] | None = None,
                    ) -> tuple[str, str]:
    """Resolve a field-level status given the case-level outcome and
    the per-field scoring result from ``score_case`` / ``match_nested_list``.

    Returns ``(field_status, field_error_detail)``. ``field_error_detail``
    is ≤120 chars and is empty for ``correct`` rows.
    """
    if case_status not in _GRADABLE_CASE_STATUS:
        return "unscoreable_due_to_case_error", ""

    # Gold-presence check. Missing gold field is distinct from gold
    # being null (which still scores against null).
    g_value = None
    if isinstance(gold, dict):
        if field in gold:
            g_value = gold.get(field)
        else:
            cd = gold.get("cancer_data") or {}
            if field in cd:
                g_value = cd.get(field)
            else:
                # biomarker_<X> fields live under cancer_data.biomarkers
                if field.startswith("biomarker_"):
                    bm = (cd.get("biomarkers") or {})
                    if field[len("biomarker_"):] not in bm:
                        return "gold_missing", ""
                else:
                    return "gold_missing", ""

    # Nested fields (regional_lymph_node, margins): scored is a float
    # F1 in (0, 1]. f1==1 → correct; f1==0 with any non-empty side →
    # treat as wrong_value; in-between → misaligned_list.
    if is_nested and isinstance(scored, (int, float)):
        if scored >= 1.0:
            return "correct", ""
        if scored <= 0.0:
            return "wrong_value", ""
        return "misaligned_list", f"nested_f1={float(scored):.2f}"

    # Scalar / list-of-literals fields.
    if scored is True:
        return "correct", ""
    if scored is False:
        # Distinguish wrong_type from wrong_value using schema errors
        # that name this field.
        detail = ""
        if schema_errors:
            for err in schema_errors:
                if isinstance(err, str) and field in err:
                    detail = err[:120]
                    return "wrong_type", detail
        return "wrong_value", _short_pred_gold(g_value, _pred_value(pred, field))
    # scored is None: not attempted. Distinguish missing_key vs null_value.
    if isinstance(pred, dict):
        if _pred_has_key(pred, field):
            return "null_value", ""
        return "missing_key", ""
    return "missing_key", ""


def _pred_has_key(pred: dict, field: str) -> bool:
    if field in pred:
        return True
    cd = pred.get("cancer_data") or {}
    if field in cd:
        return True
    if field.startswith("biomarker_"):
        bm = (cd.get("biomarkers") or {})
        return field[len("biomarker_"):] in bm
    return False


def _pred_value(pred: dict | None, field: str):
    if not isinstance(pred, dict):
        return None
    if field in pred:
        return pred[field]
    cd = pred.get("cancer_data") or {}
    if field in cd:
        return cd[field]
    if field.startswith("biomarker_"):
        bm = (cd.get("biomarkers") or {})
        return bm.get(field[len("biomarker_"):])
    return None


def _short_pred_gold(gold_value, pred_value) -> str:
    """Format a short ``gold=… pred=…`` description, ≤120 chars."""
    g = repr(gold_value)
    p = repr(pred_value)
    if len(g) > 50:
        g = g[:47] + "..."
    if len(p) > 50:
        p = p[:47] + "..."
    out = f"gold={g} pred={p}"
    return out[:120]


# ---------------------------------------------------------------------------
# Long-form scoring
# ---------------------------------------------------------------------------

def _grade_run(run_dir: Path, gold_root: Path,
               dataset: str | None = None) -> list[dict]:
    """Score every per-case JSON under a single run dir.

    Yields long-form rows ready for the master DataFrame. For each case
    where both gold and prediction supply ``cancer_category``, the rows
    carry ``cancer_category_mismatch=True`` when the two strings
    disagree — an accuracy signal, not a runtime error. Folder numbers
    are treated as case-id keys only and are not compared against
    ``cancer_category``.
    """
    rows: list[dict] = []
    for organ_dir in sorted(run_dir.iterdir()):
        if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
            continue
        organ_n = organ_dir.name
        for pred_path in sorted(organ_dir.glob("*.json")):
            case_id = pred_path.stem
            gold = _gold_for(case_id, organ_n, gold_root)
            gold_missing = gold is None
            pred: dict | object = {}
            pred_unreadable_reason: str | None = None
            try:
                with pred_path.open(encoding="utf-8") as f:
                    pred = json.load(f)
            except Exception as exc:
                pred = {}
                pred_unreadable_reason = repr(exc)[:200]
            case_status, case_flags = _classify_case(
                pred if isinstance(pred, dict) else None,
                pred_unreadable_reason=pred_unreadable_reason,
                gold_missing=gold_missing,
            )
            cc_mismatch = (
                isinstance(pred, dict)
                and isinstance(gold, dict)
                and pred.get("cancer_category") is not None
                and gold.get("cancer_category") is not None
                and pred["cancer_category"] != gold["cancer_category"]
            )
            if cc_mismatch:
                print(f"[aggregate] cancer_category mismatch: case={case_id} "
                      f"folder={organ_n} gold={gold['cancer_category']!r} "
                      f"pred={pred['cancer_category']!r}")
            schema_errors = (pred.get("_schema_errors")
                             if isinstance(pred, dict) else None)
            flags_str = "|".join(case_flags)

            # When gold is missing OR the case is not gradable, emit
            # FAIR_SCOPE rows with correct=None / attempted=False and
            # the appropriate field_status.
            if gold_missing or case_status not in _GRADABLE_CASE_STATUS:
                for field in FAIR_SCOPE:
                    f_status, f_detail = _classify_field(
                        case_status, gold, pred if isinstance(pred, dict) else None,
                        field, scored=None, is_nested=False,
                        schema_errors=schema_errors,
                    )
                    rows.append({
                        "case_id": case_id, "organ": organ_n, "field": field,
                        "correct": None, "attempted": False,
                        "cancer_category_mismatch": cc_mismatch,
                        "case_status": case_status,
                        "case_flags": flags_str,
                        "field_status": f_status,
                        "field_error_detail": f_detail,
                    })
                continue

            result = score_case(gold, pred)
            for field in FAIR_SCOPE + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]:
                if field not in result:
                    continue
                correct = result[field]
                f_status, f_detail = _classify_field(
                    case_status, gold, pred, field, scored=correct,
                    is_nested=False, schema_errors=schema_errors,
                )
                rows.append({
                    "case_id": case_id, "organ": organ_n, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                    "cancer_category_mismatch": cc_mismatch,
                    "case_status": case_status,
                    "case_flags": flags_str,
                    "field_status": f_status,
                    "field_error_detail": f_detail,
                })
            for nested_field, f1d in result.get("_nested", {}).items():
                f1_val = f1d["f1"]
                f_status, f_detail = _classify_field(
                    case_status, gold, pred, nested_field,
                    scored=f1_val, is_nested=True,
                    schema_errors=schema_errors,
                )
                rows.append({
                    "case_id": case_id, "organ": organ_n,
                    "field": nested_field,
                    "correct": f1_val, "attempted": True,
                    "cancer_category_mismatch": cc_mismatch,
                    "case_status": case_status,
                    "case_flags": flags_str,
                    "field_status": f_status,
                    "field_error_detail": f_detail,
                })
    return rows


def build_grid_dataframe(runs: list[tuple[str, str, str, Path]],
                         gold_root: Path,
                         dataset: str | None = None) -> pd.DataFrame:
    """Build the ablation_grid.csv master long-form table.

    Each row carries ``cell, model, run, case_id, organ, field, correct,
    attempted, cancer_category_mismatch, method`` (where
    ``method = f"{cell}_{model}"`` for backward compatibility with
    downstream stats code).
    """
    all_rows: list[dict] = []
    for cell, model, run_id, run_dir in runs:
        rows = _grade_run(run_dir, gold_root, dataset=dataset)
        for r in rows:
            r["cell"] = cell
            r["model"] = model
            r["run"] = run_id
            r["method"] = f"{cell}_{model}"
            all_rows.append(r)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def compute_efficiency(runs: list[tuple[str, str, str, Path]]) -> pd.DataFrame:
    """Aggregate per-run timings + error counts from each ``_summary.json``."""
    rows = []
    for cell, model, run_id, run_dir in runs:
        try:
            with (run_dir / "_summary.json").open(encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        # Per-case latencies from _log.jsonl when present.
        latencies: list[float] = []
        log_path = run_dir / "_log.jsonl"
        if log_path.exists():
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    lat = rec.get("latency_s")
                    if isinstance(lat, (int, float)):
                        latencies.append(float(lat))

        # Per-case error counts from the on-disk JSONs.
        # Mutually-exclusive buckets so the rates can never sum past 1.0:
        #   - schema_only: case has _schema_errors AND no other failure
        #   - parse_only:  case has _parse_error / _error / _pipeline_error
        #                  AND no _schema_errors
        #   - both:        case has both flags
        # ``schema_errors`` and ``parse_errors`` are exposed separately so
        # downstream stats can still report each rate independently;
        # ``failed_total`` = schema_only + parse_only + both gives the
        # overall failure rate (always <= 1.0).
        schema_only = 0
        parse_only = 0
        both = 0
        for organ_dir in run_dir.iterdir():
            if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
                continue
            for pred_path in organ_dir.glob("*.json"):
                try:
                    with pred_path.open(encoding="utf-8") as f:
                        pred = json.load(f)
                except Exception:
                    parse_only += 1
                    continue
                if not isinstance(pred, dict):
                    continue
                has_schema = bool(pred.get("_schema_errors"))
                has_parse = bool(pred.get("_parse_error")
                                 or pred.get("_error")
                                 or pred.get("_pipeline_error"))
                if has_schema and has_parse:
                    both += 1
                elif has_schema:
                    schema_only += 1
                elif has_parse:
                    parse_only += 1

        schema_errors = schema_only + both
        parse_errors = parse_only + both
        failed_total = schema_only + parse_only + both

        rows.append({
            "cell": cell,
            "model": model,
            "run": run_id,
            "n_cases": int(summary.get("n_cases", 0)),
            "mean_latency_s": (sum(latencies) / len(latencies)
                               if latencies else None),
            "median_latency_s": (sorted(latencies)[len(latencies) // 2]
                                 if latencies else None),
            "schema_errors": schema_errors,
            "parse_errors": parse_errors,
            "failed_total": failed_total,
            "validation_retries": summary.get("validation_retries", 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cell deltas (legacy summary)
# ---------------------------------------------------------------------------

def compute_cell_deltas(long_df: pd.DataFrame,
                        baseline_method: str) -> pd.DataFrame:
    """For each (model, field): Δ accuracy of every other cell vs the
    configured baseline. Point-estimate only; the rich CIs live in
    ``ablation_paired_deltas.csv`` (written by :mod:`stats`)."""
    if long_df.empty or "method" not in long_df.columns:
        return pd.DataFrame()
    df = long_df[long_df["attempted"] == True].copy()  # noqa: E712
    df["accuracy"] = pd.to_numeric(df["correct"], errors="coerce")

    # Per-method × field mean accuracy.
    pivot = df.groupby(["method", "field"])["accuracy"].mean().unstack("method")
    if baseline_method not in pivot.columns:
        return pivot.reset_index()
    base = pivot[baseline_method]
    deltas = pivot.subtract(base, axis="index")
    deltas.columns = [f"delta_{c}_minus_{baseline_method}"
                      for c in pivot.columns]
    out = pd.concat([pivot, deltas], axis=1).reset_index()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", default=None,
                    type=Path,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy' or 'workspace' resolves against "
                         "the repo root.")
    ap.add_argument("--dataset", default=None, choices=("cmuh", "tcga"),
                    help="Dataset name under data/ (cmuh or tcga).")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Override path to scan (overrides --folder/--dataset).")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Restrict to these cell ids (default: all)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Restrict to these model slugs (default: all)")
    ap.add_argument("--baseline", default="dspy_modular_gpt_oss_20b",
                    help="<cell>_<model_slug> key to use as the Δ baseline")
    ap.add_argument("--with-stats", dest="with_stats",
                    action="store_true", default=None,
                    help="also call ablations.eval.stats.run_all "
                         "(default: ON for real results-root, OFF for _smoke_)")
    ap.add_argument("--no-stats", dest="with_stats", action="store_false")
    args = ap.parse_args(argv)

    # Resolve --folder via the same shortcut as the runners.
    if args.experiment_root is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                                   / "scripts"))
            from _config_loader import resolve_folder  # noqa
            args.experiment_root = resolve_folder(args.experiment_root)
        except Exception:
            args.experiment_root = Path(args.experiment_root).resolve()

    results_root = _ablations_root(args)
    gold_root = _gold_root(args)

    runs = _discover_runs(results_root, cells=args.cells, models=args.models)
    if not runs:
        # Soft-fail: emit empty CSVs with full headers so downstream
        # consumers (multirun, paper-table orchestrators) can still run
        # to completion.
        print(f"[aggregate][warn] No completed runs found under "
              f"{results_root}. Writing empty summary scaffolding.",
              file=sys.stderr)
        results_root.mkdir(parents=True, exist_ok=True)
        empty_grid_cols = [
            "cell", "model", "run", "case_id", "organ", "field",
            "correct", "attempted", "cancer_category_mismatch", "method",
            "case_status", "case_flags", "field_status", "field_error_detail",
        ]
        pd.DataFrame(columns=empty_grid_cols).to_csv(
            results_root / "ablation_grid.csv", index=False)
        pd.DataFrame(columns=empty_grid_cols).to_parquet(
            results_root / "atomic.parquet")
        pd.DataFrame(columns=[
            "method", "field", "attempted", "total",
            "coverage", "accuracy_attempted",
        ]).to_csv(results_root / "ablation_summary.csv", index=False)
        return 0

    print(f"[aggregate] results_root={results_root}")
    print(f"[aggregate] gold_root={gold_root}")
    print(f"[aggregate] discovered {len(runs)} runs across "
          f"{len({(c, m) for c, m, _, _ in runs})} (cell, model) pairs")

    grid_df = build_grid_dataframe(runs, gold_root, dataset=args.dataset)
    grid_csv = results_root / "ablation_grid.csv"
    grid_df.to_csv(grid_csv, index=False)
    print(f"Wrote {grid_csv}  ({len(grid_df)} rows)")

    # Atomic parquet — long-form, identical schema to the CSV but with
    # Arrow types and used as the canonical input by canonical_stats.
    atomic_path = results_root / "atomic.parquet"
    try:
        grid_df.to_parquet(atomic_path)
        print(f"Wrote {atomic_path}  ({len(grid_df)} rows)")
    except Exception as exc:
        print(f"[aggregate][warn] failed to write {atomic_path}: {exc!r}",
              file=sys.stderr)

    summary = summary_table(grid_df.rename(columns={}))  # method col present
    summary_path = results_root / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    pivot = summary.pivot_table(
        index="field", columns="method",
        values="accuracy_attempted", aggfunc="first")
    pivot.to_csv(results_root / "ablation_table.csv")
    print(f"Wrote {results_root / 'ablation_table.csv'}")

    deltas = compute_cell_deltas(grid_df, args.baseline)
    if not deltas.empty:
        deltas_path = results_root / "cell_deltas.csv"
        deltas.to_csv(deltas_path, index=False)
        print(f"Wrote {deltas_path}")

    eff = compute_efficiency(runs)
    if not eff.empty:
        eff.to_csv(results_root / "efficiency.csv", index=False)
        print(f"Wrote {results_root / 'efficiency.csv'}")

    # Canonical statistics suite — runs against the same long-form grid.
    # ``method`` column is built as f"{cell}_{model}"; the modular
    # baseline supplied via --baseline becomes the comparator.
    try:
        from . import canonical_stats
        canonical_stats.run_canonical_stats(
            grid_df, modular_method=args.baseline, out_dir=results_root)
    except Exception as exc:
        print(f"[aggregate][warn] canonical stats layer failed: {exc!r}",
              file=sys.stderr)

    is_smoke = results_root.name.startswith("_smoke")
    with_stats = args.with_stats if args.with_stats is not None else not is_smoke
    if with_stats:
        from . import stats as ablation_stats
        try:
            outputs = ablation_stats.run_all(
                results_root, baseline_method=args.baseline)
            for stage, path in outputs.items():
                print(f"Wrote {path}  (stats: {stage})")
        except Exception as exc:
            print(f"[warn] stats layer failed: {exc!r}")

    print("\nper-method mean accuracy:")
    print(pivot.mean().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Silence unused-import warnings for re-exports kept for downstream use.
_ = (FAIR_SCOPE, NESTED_LIST_FIELDS, IMPLEMENTED_ORGANS, match_nested_list)
