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

Reviewer-grade statistical CSVs (``ablation_paired_deltas.csv`` etc.)
are written by :mod:`stats` when ``--with-stats`` is on (default for
non-smoke results-roots).

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
# Long-form scoring
# ---------------------------------------------------------------------------

def _grade_run(run_dir: Path, gold_root: Path) -> list[dict]:
    """Score every per-case JSON under a single run dir.

    Yields long-form rows ready for the master DataFrame.
    """
    rows: list[dict] = []
    for organ_dir in sorted(run_dir.iterdir()):
        if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
            continue
        organ_n = organ_dir.name
        for pred_path in sorted(organ_dir.glob("*.json")):
            case_id = pred_path.stem
            gold = _gold_for(case_id, organ_n, gold_root)
            if gold is None:
                # No gold for this case — record as un-gradable.
                for field in FAIR_SCOPE:
                    rows.append({
                        "case_id": case_id, "organ": organ_n, "field": field,
                        "correct": None, "attempted": False,
                    })
                continue
            try:
                with pred_path.open(encoding="utf-8") as f:
                    pred = json.load(f)
            except Exception:
                pred = {}
            if isinstance(pred, dict) and pred.get("_pipeline_error"):
                for field in FAIR_SCOPE:
                    rows.append({
                        "case_id": case_id, "organ": organ_n, "field": field,
                        "correct": None, "attempted": False,
                    })
                continue
            result = score_case(gold, pred)
            for field in FAIR_SCOPE + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]:
                if field not in result:
                    continue
                correct = result[field]
                rows.append({
                    "case_id": case_id, "organ": organ_n, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                })
            for nested_field, f1d in result.get("_nested", {}).items():
                rows.append({
                    "case_id": case_id, "organ": organ_n,
                    "field": nested_field,
                    "correct": f1d["f1"], "attempted": True,
                })
    return rows


def build_grid_dataframe(runs: list[tuple[str, str, str, Path]],
                         gold_root: Path) -> pd.DataFrame:
    """Build the ablation_grid.csv master long-form table.

    Each row carries ``cell, model, run, case_id, organ, field, correct,
    attempted, method`` (where ``method = f"{cell}_{model}"`` for
    backward compatibility with downstream stats code).
    """
    all_rows: list[dict] = []
    for cell, model, run_id, run_dir in runs:
        rows = _grade_run(run_dir, gold_root)
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
        schema_errors = 0
        parse_errors = 0
        for organ_dir in run_dir.iterdir():
            if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
                continue
            for pred_path in organ_dir.glob("*.json"):
                try:
                    with pred_path.open(encoding="utf-8") as f:
                        pred = json.load(f)
                except Exception:
                    parse_errors += 1
                    continue
                if not isinstance(pred, dict):
                    continue
                if pred.get("_schema_errors"):
                    schema_errors += 1
                if (pred.get("_parse_error") or pred.get("_error")
                        or pred.get("_pipeline_error")):
                    parse_errors += 1

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
        sys.exit(f"No completed runs found under {results_root}.")

    print(f"[aggregate] results_root={results_root}")
    print(f"[aggregate] gold_root={gold_root}")
    print(f"[aggregate] discovered {len(runs)} runs across "
          f"{len({(c, m) for c, m, _, _ in runs})} (cell, model) pairs")

    grid_df = build_grid_dataframe(runs, gold_root)
    grid_csv = results_root / "ablation_grid.csv"
    grid_df.to_csv(grid_csv, index=False)
    print(f"Wrote {grid_csv}  ({len(grid_df)} rows)")

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
_ = (NESTED_LIST_FIELDS, IMPLEMENTED_ORGANS, match_nested_list)
