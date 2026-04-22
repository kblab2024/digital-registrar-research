#!/usr/bin/env python3
"""Smoke-test the DSPy + Ollama cancer-extraction stack on 5 random reports.

Purpose
-------
Before committing to a multi-hour full run, verify end-to-end that:
  * Ollama is reachable and the requested model key resolves to a pulled model
  * DSPy signatures still bind to the pipeline code
  * Reports under ``{experiment_root}/cmuh_dataset/*/`` are discoverable
  * JSON predictions are written where the full-run script would write them

Unlike the full driver, this script fails loudly: the process exits non-zero
if *any* sampled case raises a pipeline exception. A green smoke run is the
go/no-go for the real run.

Usage
-----
    python scripts/run_dspy_ollama_smoke.py \\
        --experiment-root /path/to/exp \\
        --model gpt \\
        [--n 5] [--seed 0]

Output
------
    {experiment_root}/output/{model}/_smoke_{date}/
        <case_id>.json        prediction (one per sampled case)
        _log.jsonl            one row per case
        _summary.json         aggregate
        _run.log              full-verbosity log
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from digital_registrar_research.models.common import model_list  # noqa: E402
from digital_registrar_research.pipeline import setup_pipeline  # noqa: E402
from digital_registrar_research.util.logger import setup_logger  # noqa: E402

# Reuse the low-level helpers from the full-run script so the two stay in
# lockstep w/r/t output format and atomic-write semantics.
from run_dspy_ollama_single import (  # noqa: E402
    PIPELINE_LOGGER_NAME,
    _atomic_write_json,
    process_case,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--experiment-root", type=Path, required=True,
                    help="Root directory containing cmuh_dataset/.")
    ap.add_argument("--model", required=True,
                    help=f"Model key (one of: {', '.join(model_list.keys())}).")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of random cases to run (default: 5).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed for reproducibility (default: 0).")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.model not in model_list:
        print(f"error: unknown --model {args.model!r}. "
              f"Valid keys: {', '.join(model_list.keys())}", file=sys.stderr)
        return 2

    dataset_root = args.experiment_root / "cmuh_dataset"
    if not dataset_root.is_dir():
        print(f"error: dataset not found at {dataset_root}", file=sys.stderr)
        return 2

    all_reports = sorted(dataset_root.glob("*/*.txt"))
    if len(all_reports) < args.n:
        print(f"error: only {len(all_reports)} reports under {dataset_root}, "
              f"cannot sample {args.n}", file=sys.stderr)
        return 2

    sampled = random.Random(args.seed).sample(all_reports, args.n)

    date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.experiment_root / "output" / args.model
               / f"_smoke_{date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(out_dir / "_run.log"),
        json_format=False,
    )
    logger.info("smoke dir: %s", out_dir)
    logger.info("model: %s (%s)", args.model, model_list[args.model])
    logger.info("sampled %d / %d reports (seed=%d)",
                len(sampled), len(all_reports), args.seed)
    for p in sampled:
        logger.info("  - %s (%s)", p.name, p.parent.name)

    setup_pipeline(args.model)

    summary = {
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0,
        "cancer_positive": 0, "sampled": [],
    }
    with (out_dir / "_log.jsonl").open("a", encoding="utf-8") as log_fh:
        for report_path in sampled:
            # Tag the log row with the source subset name so users can trace
            # the sample back to its subdir even though outputs are flat here.
            row = process_case(
                report_path,
                subset=report_path.parent.name,
                out_dir=out_dir,
                log_fh=log_fh,
                logger=logger,
                overwrite=True,  # smoke runs should never silently skip
            )
            summary["n_cases"] += 1
            summary["sampled"].append({
                "case_id": row["case_id"], "subset": row["subset"],
                "status": row["status"], "wall_ms": row["wall_ms"],
            })
            if row["status"] == "ok":
                summary["n_ok"] += 1
                if row.get("is_cancer"):
                    summary["cancer_positive"] += 1
            elif row["status"] == "pipeline_error":
                summary["n_pipeline_error"] += 1

    _atomic_write_json(out_dir / "_summary.json", summary)
    summary_line = (
        f"SMOKE OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
        f"N={summary['n_cases']}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"smoke dir: {out_dir}")

    return 0 if summary["n_pipeline_error"] == 0 and summary["n_ok"] == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
