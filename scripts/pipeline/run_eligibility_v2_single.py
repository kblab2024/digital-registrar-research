#!/usr/bin/env python3
"""Single-run eligibility-only v2 experiment (DSPy + Ollama / OpenAI).

Iterates every ``<case_id>.txt`` under
``{experiment_root}/data/{dataset}/reports/{organ_n}/`` and writes one
eligibility-only prediction JSON per report to
``{experiment_root}/results/eligibility_v2/{dataset}/{model_slug}/{run}/{organ_n}/<case_id>.json``.

Unlike ``run_dspy_ollama_single.py``, this runner:
  * invokes ONLY the v2 eligibility signature (``is_cancer_v2``) via
    ``EligibilityOnlyPipelineV2`` — no Jsonize, no organ-specific extractors.
  * emits the six v2 eligibility fields per case
    (``cancer_excision_report``, ``cancer_category``,
    ``cancer_category_others_description``, ``secondary_primary_present``,
    ``secondary_cancer_category``, ``secondary_cancer_category_others_description``)
    plus a ``_meta`` block with provenance / timing.
  * writes to a separate ``results/eligibility_v2/`` tree so it cannot
    collide with v1 full-pipeline predictions.

Use this to evaluate the v2 eligibility step in isolation — primary
category recall vs. v1, secondary-primary detection precision/recall —
before paying the cost of the full v2 pipeline.

Usage
-----
    python scripts/pipeline/run_eligibility_v2_single.py \\
        --model gptoss --folder dummy --dataset tcga \\
        [--run run01] [--organs 1 2] [--limit N] [--overwrite] \\
        [--tolerate-errors] [-v]

``--model`` must be one of the unified aliases (same set the v1 single
runner accepts). YAML decoding overrides under
``configs/dspy_ollama_{alias}.yaml`` are applied when present.

Output tree
-----------
    {experiment_root}/results/eligibility_v2/{dataset}/{model_slug}/
        {run}/                       e.g. run01
            _summary.json            run-level totals
            _log.jsonl               one row per case
            _run.log                 full-verbosity log
            _run_meta.json           model / env / argv provenance
            {organ_n}/
                <case_id>.json       eligibility prediction
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Make the in-tree package importable without `pip install -e .`.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))            # _config_loader, _run_id
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))  # v1 runner helpers

from _config_loader import (  # noqa: E402
    load_model_config,
    resolve_folder,
    split_decoding_overrides,
)
from _run_id import format_run_id, machine_slug  # noqa: E402

# Reuse v1 runner helpers verbatim so output layout / discovery / IDs stay
# byte-identical to the production single-run conventions.
from run_dspy_ollama_single import (  # noqa: E402
    DATASETS,
    MAX_RUN_SLOTS,
    PIPELINE_LOGGER_NAME,
    UNIFIED_MODELS,
    _atomic_write_json,
    _git_sha,
    _utc_now_iso,
    discover_cases,
    discover_organs,
    model_slug,
)

from digital_registrar_research.models.common import (  # noqa: E402
    load_model,
    localaddr,
    model_list,
)
from digital_registrar_research.pipeline import setup_pipeline  # noqa: E402
from digital_registrar_research.pipelines.eligibility_only import (  # noqa: E402
    EligibilityOnlyPipelineV2,
)
from digital_registrar_research.util.logger import setup_logger  # noqa: E402

# Constant — distinct from v1's "predictions" tree so the two cannot mix.
RESULTS_SUBDIR = "eligibility_v2"


# --- IO helpers --------------------------------------------------------------


def _valid_existing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("_pipeline_error")


def _split_report_rows(report_text: str) -> list[str]:
    rows = report_text.split("\n\n")
    return [row.strip() for row in rows if row.strip()]


# --- Discovery (mirrors v1 single runner) -----------------------------------


def pick_next_run(model_dir: Path) -> str:
    """Return the first run-id in 01..MAX_RUN_SLOTS without ``_summary.json``."""
    slug = machine_slug()
    for k in range(1, MAX_RUN_SLOTS + 1):
        name = format_run_id(k, padded=True)
        if not (model_dir / name / "_summary.json").exists():
            return name
    suffix_msg = f" (machine={slug!r})" if slug else ""
    raise RuntimeError(
        f"all {MAX_RUN_SLOTS} run slots are populated under {model_dir}{suffix_msg}; "
        f"pass --run runNN explicitly to re-run one (use --overwrite to clobber cases)."
    )


# --- Per-case work ----------------------------------------------------------


def process_case(
    pipeline: EligibilityOnlyPipelineV2,
    report_path: Path,
    organ: str,
    run_name: str,
    seed: Any,
    out_dir: Path,
    log_fh,
    logger: logging.Logger,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Run v2 eligibility-only for one case. Writes
    ``{out_dir}/{organ}/<case_id>.json`` and appends one log row."""
    case_id = report_path.stem
    organ_out_dir = out_dir / organ
    out_path = organ_out_dir / f"{case_id}.json"
    started_at = _utc_now_iso()

    if not overwrite and _valid_existing(out_path):
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "cached", "latency_s": 0.0, "parse_success": True,
            "is_cancer": None, "cancer_category": None,
            "secondary_primary_present": None, "secondary_cancer_category": None,
            "error": None, "started_at": started_at,
        }
        log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fh.flush()
        logger.info("[%s/%s/%s] cached — skipped", run_name, organ, case_id)
        return row

    report = report_path.read_text(encoding="utf-8")
    report_rows = _split_report_rows(report)
    t0 = time.perf_counter()
    try:
        result = pipeline(
            report=report_rows,
            logger=logging.getLogger(PIPELINE_LOGGER_NAME),
            fname=case_id,
        )
        latency_s = round(time.perf_counter() - t0, 3)
        payload = {
            "cancer_excision_report": result.get("cancer_excision_report"),
            "cancer_category": result.get("cancer_category"),
            "cancer_category_others_description": result.get("cancer_category_others_description"),
            "secondary_primary_present": result.get("secondary_primary_present"),
            "secondary_cancer_category": result.get("secondary_cancer_category"),
            "secondary_cancer_category_others_description":
                result.get("secondary_cancer_category_others_description"),
            "_meta": {
                "case_id": case_id,
                "organ": organ,
                "run": run_name,
                "latency_s": latency_s,
                "started_at": started_at,
                "pipeline": "EligibilityOnlyPipelineV2",
                "signature": "is_cancer_v2",
            },
        }
        _atomic_write_json(out_path, payload)
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "ok", "latency_s": latency_s, "parse_success": True,
            "is_cancer": bool(payload["cancer_excision_report"])
                if payload["cancer_excision_report"] is not None else None,
            "cancer_category": payload["cancer_category"],
            "secondary_primary_present": (
                bool(payload["secondary_primary_present"])
                if payload["secondary_primary_present"] is not None else None
            ),
            "secondary_cancer_category": payload["secondary_cancer_category"],
            "error": None, "started_at": started_at,
        }
        logger.info(
            "[%s/%s/%s] ok (%.2fs, cancer=%s, primary=%s, secondary=%s/%s)",
            run_name, organ, case_id, latency_s,
            row["is_cancer"], row["cancer_category"],
            row["secondary_primary_present"], row["secondary_cancer_category"],
        )
    except Exception as exc:
        latency_s = round(time.perf_counter() - t0, 3)
        sentinel = {
            "_pipeline_error": True,
            "reason": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        _atomic_write_json(out_path, sentinel)
        row = {
            "case_id": case_id, "organ": organ, "run": run_name, "seed": seed,
            "status": "pipeline_error", "latency_s": latency_s,
            "parse_success": False, "is_cancer": None, "cancer_category": None,
            "secondary_primary_present": None, "secondary_cancer_category": None,
            "error": f"{type(exc).__name__}: {exc}", "started_at": started_at,
        }
        logger.error("[%s/%s/%s] pipeline error: %s",
                     run_name, organ, case_id, exc)

    log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_fh.flush()
    return row


# --- Run driver -------------------------------------------------------------


def run_single(
    run_dir: Path, run_name: str, organs: list[tuple[str, Path]],
    seed: Any, logger: logging.Logger, args: argparse.Namespace,
) -> dict[str, Any]:
    """One full pass over all discovered cases. Returns the ``_summary.json`` payload."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline = EligibilityOnlyPipelineV2()

    summary: dict[str, Any] = {
        "run": run_name,
        "model": model_slug(args.model),
        "seed": seed,
        "dataset": args.dataset,
        "pipeline": "EligibilityOnlyPipelineV2",
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0, "n_cached": 0,
        "cancer_positive": 0,
        "secondary_primary_positive": 0,
        "per_organ": {},
        "wall_time_s": 0.0,
        "created_at": _utc_now_iso(),
    }
    t_run = time.perf_counter()
    log_path = run_dir / "_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log_fh:
        for organ_n, organ_dir in organs:
            cases = discover_cases(organ_dir, args.limit)
            logger.info("organ %s: %d cases", organ_n, len(cases))
            per = summary["per_organ"].setdefault(organ_n, {
                "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0,
                "n_cached": 0, "cancer_positive": 0,
                "secondary_primary_positive": 0,
            })
            for report_path in cases:
                row = process_case(
                    pipeline, report_path, organ_n, run_name, seed, run_dir,
                    log_fh, logger, overwrite=args.overwrite,
                )
                summary["n_cases"] += 1
                per["n_cases"] += 1
                status = row["status"]
                if status == "ok":
                    summary["n_ok"] += 1
                    per["n_ok"] += 1
                    if row.get("is_cancer"):
                        summary["cancer_positive"] += 1
                        per["cancer_positive"] += 1
                    if row.get("secondary_primary_present"):
                        summary["secondary_primary_positive"] += 1
                        per["secondary_primary_positive"] += 1
                elif status == "cached":
                    summary["n_cached"] += 1
                    per["n_cached"] += 1
                elif status == "pipeline_error":
                    summary["n_pipeline_error"] += 1
                    per["n_pipeline_error"] += 1

    summary["wall_time_s"] = round(time.perf_counter() - t_run, 1)
    summary["parse_error_rate"] = (summary["n_pipeline_error"]
                                   / max(summary["n_cases"], 1))
    _atomic_write_json(run_dir / "_summary.json", summary)
    return summary


# --- Entry point ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                    help="Model alias: one of " + ", ".join(UNIFIED_MODELS))
    ap.add_argument("--folder", dest="experiment_root", required=False, default=None,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthands 'dummy', 'workspace', 'obfustrated' "
                         "resolve against the repo root.")
    ap.add_argument("--obfustrated", action="store_true",
                    help="Shortcut for --folder obfustrated.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset under data/ (cmuh or tcga).")
    ap.add_argument("--run", default=None,
                    help="Run slot name, e.g. run01..run10 "
                         "(default: next free slot under the model dir).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Only run these numeric organ directories.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per organ (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Exit 0 even if some cases failed.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
    return ap.parse_args(argv)


def run_with_args(
    args: argparse.Namespace, overrides: dict | None = None,
) -> int:
    if args.model not in model_list:
        print(f"error: unknown --model {args.model!r}. "
              f"Valid keys: {', '.join(model_list.keys())}", file=sys.stderr)
        return 2

    if args.experiment_root is None:
        if getattr(args, "obfustrated", False):
            args.experiment_root = resolve_folder("obfustrated")
        else:
            print("error: --folder is required (use 'dummy', 'workspace', "
                  "'workspace_obfustrated', or an absolute path), or pass "
                  "--obfustrated.", file=sys.stderr)
            return 2

    reports_root = args.experiment_root / "data" / args.dataset / "reports"
    if not reports_root.is_dir():
        print(f"error: reports not found at {reports_root}", file=sys.stderr)
        return 2

    organs = discover_organs(reports_root, args.organs)
    if not organs:
        suffix = f" matching {args.organs}" if args.organs else ""
        print(f"error: no organ dirs with *.txt found under {reports_root}"
              f"{suffix}", file=sys.stderr)
        return 2

    slug = model_slug(args.model)
    model_dir = (args.experiment_root / "results" / RESULTS_SUBDIR
                 / args.dataset / slug)

    if args.run:
        if not re.fullmatch(r"run\d{2}(-[a-z0-9][a-z0-9-]*)?", args.run):
            print(f"error: --run must look like 'run01' or 'run01-<machine>' "
                  f"(got {args.run!r})", file=sys.stderr)
            return 2
        run_name = args.run
    else:
        try:
            run_name = pick_next_run(model_dir)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    run_dir = model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(run_dir / "_run.log"),
        json_format=False,
    )
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("dataset: %s", args.dataset)
    logger.info("model: %s (%s) → slug=%s",
                args.model, model_list[args.model], slug)
    logger.info("run: %s", run_name)
    logger.info("run dir: %s", run_dir)
    logger.info("pipeline: EligibilityOnlyPipelineV2")
    logger.info("organs: %s", [o[0] for o in organs])
    if overrides:
        logger.info("decoding overrides: %s", overrides)

    setup_pipeline(args.model, overrides=overrides)
    lm = load_model(args.model, overrides=overrides)
    lm_kwargs = {
        "temperature": getattr(lm, "temperature", None) or lm.kwargs.get("temperature"),
        "top_p": lm.kwargs.get("top_p"),
        "top_k": lm.kwargs.get("top_k"),
        "max_tokens": lm.kwargs.get("max_tokens"),
        "num_ctx": lm.kwargs.get("num_ctx"),
        "repeat_penalty": lm.kwargs.get("repeat_penalty"),
        "keep_alive": lm.kwargs.get("keep_alive"),
        "cache": lm.kwargs.get("cache"),
        "seed": lm.kwargs.get("seed"),
    }

    started_at = _utc_now_iso()
    t_run = time.perf_counter()
    try:
        summary = run_single(
            run_dir, run_name, organs, lm_kwargs.get("seed"), logger, args,
        )
    finally:
        finished_at = _utc_now_iso()

    _atomic_write_json(run_dir / "_run_meta.json", {
        "run": run_name,
        "pipeline": "EligibilityOnlyPipelineV2",
        "signature": "is_cancer_v2",
        "model_key": args.model,
        "model_id": model_list[args.model],
        "model_slug": slug,
        "dataset": args.dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "organs": [o[0] for o in organs],
        "ollama_endpoint": localaddr,
        "started_at": started_at,
        "finished_at": finished_at,
        "dspy_lm_kwargs": lm_kwargs,
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    })

    wall_s = int(time.perf_counter() - t_run)
    wall_fmt = (f"{wall_s // 3600:02d}:{(wall_s % 3600) // 60:02d}:"
                f"{wall_s % 60:02d}")
    summary_line = (
        f"OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
        f"CACHED={summary['n_cached']} N={summary['n_cases']} "
        f"CA+={summary['cancer_positive']} 2P+={summary['secondary_primary_positive']} "
        f"WALL={wall_fmt}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"run dir: {run_dir}")

    if summary["n_pipeline_error"] > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.model)
    overrides = split_decoding_overrides(cfg.get("decoding"))
    return run_with_args(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
