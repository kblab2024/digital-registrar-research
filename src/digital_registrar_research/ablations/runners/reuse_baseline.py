"""
Cell A — reuse the existing modular-DSPy outputs.

The modular DSPy pipeline IS the parent project; predictions for the
test split already live in the canonical pipeline tree:

    {root}/results/predictions/{dataset}/llm/{model_slug}/{run}/{organ}/{case_id}.json

Cell A copies those per-case JSONs into the canonical ablation tree:

    {root}/results/ablations/{dataset}/dspy_modular/{model_slug}/{run}/{organ}/{case_id}.json

so every cell has a consistent on-disk layout for the aggregator.

Single-source import (default): one pipeline run → one ablation `runNN`
slot. Pass ``--source-run runNN`` to pin a specific source run; otherwise
the most-recent completed pipeline run is used.

Multi-source import: pass ``--source-runs r1 r2 ...`` to import several
named runs (each into its own next-free `runNN` slot), or
``--all-source-runs`` to import every completed pipeline run found under
``predictions/{dataset}/llm/{slug}/``. This lets the aggregator pair
multiple modular seeds against multiple monolithic / raw_json seeds for
honest multi-run paired statistics.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss \\
        [--source-run runNN | --source-runs r1 r2 ... | --all-source-runs] \\
        [--run runNN]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from . import _base

CELL_ID = "dspy_modular"


def _pipeline_predictions_dir(experiment_root: Path, dataset: str,
                              model_slug_str: str) -> Path:
    return (experiment_root / "results" / "predictions" / dataset / "llm"
            / model_slug_str)


def _all_completed_pipeline_runs(pipeline_dir: Path) -> list[str]:
    if not pipeline_dir.is_dir():
        raise SystemExit(
            f"no pipeline predictions found at {pipeline_dir}. "
            f"Run scripts/pipeline/run_dspy_ollama_single.py first.")
    return sorted(
        p.name for p in pipeline_dir.iterdir()
        if p.is_dir() and (p / "_summary.json").exists()
    )


def _resolve_source_runs(pipeline_dir: Path,
                         source_run: str | None,
                         source_runs: list[str] | None,
                         all_source_runs: bool) -> list[str]:
    """Resolve the list of pipeline run-ids to import.

    Precedence (mutually exclusive — error if more than one is set):
      1. ``--all-source-runs`` → every completed pipeline run.
      2. ``--source-runs <r1> <r2> ...`` → the explicit list.
      3. ``--source-run <r>``         → length-1 list.
      4. (none)                       → most-recent completed pipeline run.
    """
    flags_set = sum(bool(x) for x in (all_source_runs, source_runs, source_run))
    if flags_set > 1:
        raise SystemExit(
            "--source-run, --source-runs, and --all-source-runs are mutually "
            "exclusive; pick at most one.")

    if all_source_runs:
        runs = _all_completed_pipeline_runs(pipeline_dir)
        if not runs:
            raise SystemExit(
                f"no completed runs (with _summary.json) under {pipeline_dir}")
        return runs

    if source_runs:
        missing = [r for r in source_runs
                   if not (pipeline_dir / r / "_summary.json").exists()]
        if missing:
            raise SystemExit(
                f"--source-runs entries not present (or not finalised) "
                f"under {pipeline_dir}: {missing}")
        return list(source_runs)

    if source_run:
        if not (pipeline_dir / source_run / "_summary.json").exists():
            raise SystemExit(
                f"--source-run {source_run!r} not present (or not finalised) "
                f"under {pipeline_dir}")
        return [source_run]

    runs = _all_completed_pipeline_runs(pipeline_dir)
    if not runs:
        raise SystemExit(
            f"no completed runs (with _summary.json) under {pipeline_dir}")
    return [runs[-1]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--source-run", default=None,
                   help="pipeline run-id to copy from (default: most recent "
                        "completed run under predictions/{dataset}/llm/{slug}/)")
    g.add_argument("--source-runs", nargs="+", default=None,
                   help="explicit list of pipeline run-ids to import; each "
                        "lands in its own next-free ablation runNN slot")
    g.add_argument("--all-source-runs", action="store_true",
                   help="import every completed pipeline run under "
                        "predictions/{dataset}/llm/{slug}/ — useful for "
                        "stacking multi-run modular paired stats")
    return ap.parse_args(argv)


def _copy_one_source(*, paths: _base.AblationPaths, run_name: str,
                     source_run: str, source_run_dir: Path,
                     organs: list[tuple[str, Path]],
                     args: argparse.Namespace,
                     logger) -> _base.RunSummary:
    """Copy one pipeline run into one ablation runNN slot. Side-effect:
    writes ``_log.jsonl``, ``_summary.json``, ``_run_meta.json``, and
    appends the run entry to the cell-level ``_manifest.yaml``."""
    summary = _base.RunSummary(
        run=run_name, cell=CELL_ID, model_slug=paths.model_slug,
        model_alias=args.model, dataset=args.dataset,
        seed=None,
    )

    log_path = paths.run_dir(run_name) / "_log.jsonl"
    t_run = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_fh:
        for organ_n, _organ_dir in organs:
            src_organ = source_run_dir / organ_n
            if not src_organ.is_dir():
                logger.warning("source has no organ %s — skipping", organ_n)
                continue
            sources = sorted(src_organ.glob("*.json"))
            if args.limit:
                sources = sources[:args.limit]
            for src_path in sources:
                case_id = src_path.stem
                dst_path = paths.case_path(run_name, organ_n, case_id)
                if not args.overwrite and dst_path.exists():
                    summary.record(organ_n, "cached")
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "cached",
                    }) + "\n")
                    continue
                try:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    with src_path.open(encoding="utf-8") as f:
                        payload = json.load(f)
                    is_cancer = bool(payload.get("cancer_excision_report"))
                    summary.record(organ_n, "ok", is_cancer=is_cancer)
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "ok",
                        "is_cancer": is_cancer,
                        "cancer_category": payload.get("cancer_category"),
                    }) + "\n")
                except Exception as exc:
                    summary.record(organ_n, "pipeline_error")
                    logger.error("[%s/%s] copy error: %s",
                                 organ_n, case_id, exc)
                    log_fh.write(json.dumps({
                        "case_id": case_id, "organ": organ_n,
                        "status": "pipeline_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }) + "\n")

    summary.wall_time_s = time.perf_counter() - t_run
    _base.finalize_run(
        paths, run_name, summary, model_alias=args.model,
        decoding={"copied_from": source_run},
        manifest_extra={"source_run": source_run,
                        "source_dir": str(source_run_dir)},
        extra_meta={"source_run": source_run,
                    "source_dir": str(source_run_dir)},
    )
    return summary


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    pipeline_dir = _pipeline_predictions_dir(
        args.experiment_root, args.dataset, paths.model_slug)
    source_runs = _resolve_source_runs(
        pipeline_dir,
        source_run=getattr(args, "source_run", None),
        source_runs=getattr(args, "source_runs", None),
        all_source_runs=getattr(args, "all_source_runs", False),
    )

    logger = _base.make_logger("reuse_baseline", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s n_source_runs=%d",
                CELL_ID, args.model, paths.model_slug, len(source_runs))

    # First source uses the pre-allocated slot from resolve_run_paths
    # (preserves --run runNN semantics for the single-source case).
    # Subsequent sources each call pick_next_run for a fresh slot.
    totals = {"ok": 0, "err": 0, "cached": 0, "n": 0}
    for i, source_run in enumerate(source_runs):
        if i == 0:
            slot = run_name
        else:
            slot = _base.pick_next_run(paths.cell_dir)
            paths.run_dir(slot).mkdir(parents=True, exist_ok=True)
        source_run_dir = pipeline_dir / source_run
        logger.info("[%d/%d] copying %s -> %s",
                    i + 1, len(source_runs), source_run_dir,
                    paths.run_dir(slot))
        summary = _copy_one_source(
            paths=paths, run_name=slot,
            source_run=source_run, source_run_dir=source_run_dir,
            organs=organs, args=args, logger=logger,
        )
        totals["ok"] += summary.n_ok
        totals["err"] += summary.n_pipeline_error
        totals["cached"] += summary.n_cached
        totals["n"] += summary.n_cases
        print(f"[{i + 1}/{len(source_runs)}] source={source_run} "
              f"slot={slot} OK={summary.n_ok} ERR={summary.n_pipeline_error} "
              f"CACHED={summary.n_cached} N={summary.n_cases}")

    print(f"OK={totals['ok']} ERR={totals['err']} CACHED={totals['cached']} "
          f"N={totals['n']} N_SOURCES={len(source_runs)}")
    print(f"cell dir: {paths.cell_dir}")

    if totals["err"] > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main())
