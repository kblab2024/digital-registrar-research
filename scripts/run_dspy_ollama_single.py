#!/usr/bin/env python3
"""Single-run DSPy + Ollama cancer-extraction experiment.

Iterates every ``<case_id>.txt`` report in
``{experiment_root}/cmuh_dataset/{n}/*`` and writes one prediction JSON per
report to ``{experiment_root}/output/{model_name}/{date}/{n}/<case_id>.json``,
preserving the subset-directory structure.

Unlike the multirun driver in ``run_gpt_oss_multirun.py``, this script:
  * talks to a local Ollama daemon via DSPy (no OpenAI-compatible endpoint)
  * performs one pass with a single seed (whatever ``models.common.load_model``
    configures), rather than K seeded repetitions
  * needs no frozen-protocol YAML — the arguments are just a path and a model

Usage
-----
    python scripts/run_dspy_ollama_single.py \\
        --experiment-root /path/to/exp \\
        --model gpt \\
        [--subsets 1 2 3] [--limit N] [--overwrite] \\
        [--date 20260422_141530] [--tolerate-errors] [-v]

Output tree
-----------
    {experiment_root}/output/{model}/{date}/
        _manifest.json        run-level metadata
        _run.log              rotating file log (full verbosity)
        {n}/
            <case_id>.json    prediction or {"_pipeline_error": true, ...}
            _log.jsonl        one row per case
            _summary.json     per-subset totals
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the in-tree package importable without requiring `pip install -e .`.
sys.path.insert(0, str(REPO_ROOT / "src"))

from digital_registrar_research.models.common import (  # noqa: E402
    load_model,
    localaddr,
    model_list,
)
from digital_registrar_research.pipeline import (  # noqa: E402
    run_cancer_pipeline,
    setup_pipeline,
)
from digital_registrar_research.util.logger import setup_logger  # noqa: E402

PIPELINE_LOGGER_NAME = "experiment_logger"  # fixed by pipeline.run_pipeline


# --- IO helpers --------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a tmp file + rename so a killed run never leaves a
    half-written ``<case_id>.json`` that would later be mistaken for a valid
    cached prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _valid_existing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("_pipeline_error")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


# --- Discovery ---------------------------------------------------------------


def discover_subsets(
    dataset_root: Path, subset_filter: list[str] | None,
) -> list[tuple[str, Path]]:
    """Return sorted ``(subset_name, subset_dir)`` pairs for every subdir
    under ``dataset_root`` that contains at least one ``*.txt``.
    ``subset_filter`` is matched against the directory *name*."""
    if not dataset_root.is_dir():
        return []
    picked: list[tuple[str, Path]] = []
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if subset_filter and child.name not in subset_filter:
            continue
        if not any(child.glob("*.txt")):
            continue
        picked.append((child.name, child))
    return picked


def discover_cases(subset_dir: Path, limit: int | None) -> list[Path]:
    cases = sorted(subset_dir.glob("*.txt"), key=lambda p: p.name)
    if limit is not None:
        cases = cases[:limit]
    return cases


# --- Per-case work -----------------------------------------------------------


def process_case(
    report_path: Path,
    subset: str,
    out_dir: Path,
    log_fh,
    logger: logging.Logger,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = report_path.stem
    out_path = out_dir / f"{case_id}.json"
    started_at = _utc_now_iso()

    if not overwrite and _valid_existing(out_path):
        row = {
            "case_id": case_id, "subset": subset, "status": "cached",
            "wall_ms": 0, "is_cancer": None, "cancer_category": None,
            "error": None, "started_at": started_at,
        }
        log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fh.flush()
        logger.info("[%s/%s] cached — skipped", subset, case_id)
        return row

    report = report_path.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    try:
        output, _elapsed_s = run_cancer_pipeline(report=report, fname=case_id)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        _atomic_write_json(out_path, output)
        row = {
            "case_id": case_id, "subset": subset, "status": "ok",
            "wall_ms": wall_ms,
            "is_cancer": bool(output.get("cancer_excision_report")),
            "cancer_category": output.get("cancer_category"),
            "error": None, "started_at": started_at,
        }
        logger.info(
            "[%s/%s] ok (%.1fs, cancer=%s, category=%s)",
            subset, case_id, wall_ms / 1000.0,
            row["is_cancer"], row["cancer_category"],
        )
    except Exception as exc:  # DSPy/Ollama errors surface here
        wall_ms = int((time.perf_counter() - t0) * 1000)
        sentinel = {
            "_pipeline_error": True,
            "reason": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        _atomic_write_json(out_path, sentinel)
        row = {
            "case_id": case_id, "subset": subset, "status": "pipeline_error",
            "wall_ms": wall_ms, "is_cancer": None, "cancer_category": None,
            "error": f"{type(exc).__name__}: {exc}", "started_at": started_at,
        }
        logger.error("[%s/%s] pipeline error: %s", subset, case_id, exc)

    log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_fh.flush()
    return row


# --- Subset driver -----------------------------------------------------------


def run_subset(
    subset: str, subset_dir: Path, run_root: Path,
    logger: logging.Logger, args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = run_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(subset_dir, args.limit)
    logger.info("subset %s: %d cases", subset, len(cases))

    summary = {
        "subset": subset, "n_cases": 0, "n_ok": 0,
        "n_pipeline_error": 0, "n_cached": 0, "cancer_positive": 0,
        "wall_s": 0,
    }
    t_subset = time.perf_counter()
    log_path = out_dir / "_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log_fh:
        for report_path in cases:
            row = process_case(
                report_path, subset, out_dir, log_fh, logger,
                overwrite=args.overwrite,
            )
            summary["n_cases"] += 1
            status = row["status"]
            if status == "ok":
                summary["n_ok"] += 1
                if row.get("is_cancer"):
                    summary["cancer_positive"] += 1
            elif status == "cached":
                summary["n_cached"] += 1
            elif status == "pipeline_error":
                summary["n_pipeline_error"] += 1
    summary["wall_s"] = int(time.perf_counter() - t_subset)
    _atomic_write_json(out_dir / "_summary.json", summary)
    return summary


# --- Manifest ----------------------------------------------------------------


def build_manifest(
    args: argparse.Namespace, date_str: str,
    subsets: list[str], per_subset: dict[str, dict],
    started_at: str, finished_at: str,
) -> dict[str, Any]:
    lm = load_model(args.model)  # same kwargs that were applied to dspy
    totals = {
        "n_cases": sum(s["n_cases"] for s in per_subset.values()),
        "n_ok": sum(s["n_ok"] for s in per_subset.values()),
        "n_pipeline_error":
            sum(s["n_pipeline_error"] for s in per_subset.values()),
        "n_cached": sum(s["n_cached"] for s in per_subset.values()),
        "cancer_positive":
            sum(s["cancer_positive"] for s in per_subset.values()),
        "wall_s": sum(s["wall_s"] for s in per_subset.values()),
    }
    return {
        "model_name": args.model,
        "model_id": model_list[args.model],
        "experiment_root": str(args.experiment_root.resolve()),
        "date": date_str,
        "dspy_lm_kwargs": {
            "temperature": getattr(lm, "temperature", None)
                           or lm.kwargs.get("temperature"),
            "top_p": lm.kwargs.get("top_p"),
            "max_tokens": lm.kwargs.get("max_tokens"),
            "num_ctx": lm.kwargs.get("num_ctx"),
            "seed": lm.kwargs.get("seed"),
        },
        "ollama_endpoint": localaddr,
        "started_at": started_at,
        "finished_at": finished_at,
        "subsets": subsets,
        "per_subset": per_subset,
        "totals": totals,
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    }


# --- Entry point -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--experiment-root", type=Path, required=True,
                    help="Root directory containing cmuh_dataset/ and where "
                         "output/ will be written.")
    ap.add_argument("--model", required=True,
                    help=f"Model key from models.common.model_list "
                         f"(one of: {', '.join(model_list.keys())}).")
    ap.add_argument("--date", default=None,
                    help="Override the YYYYMMDD_HHMMSS folder name "
                         "(default: current local time).")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Only run these subset directory names "
                         "(default: every subdir of cmuh_dataset/).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per subset (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Always exit 0 if the script completes, even when "
                         "some cases failed. Default: non-zero on any error.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set console log level to DEBUG.")
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

    subsets = discover_subsets(dataset_root, args.subsets)
    if not subsets:
        print(f"error: no subsets with *.txt found under {dataset_root}"
              + (f" matching {args.subsets}" if args.subsets else ""),
              file=sys.stderr)
        return 2

    date_str = args.date or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.experiment_root / "output" / args.model / date_str
    run_root.mkdir(parents=True, exist_ok=True)

    log_file = run_root / "_run.log"
    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(log_file),
        json_format=False,
    )
    logger.info("run root: %s", run_root)
    logger.info("model: %s (%s)", args.model, model_list[args.model])
    logger.info("subsets: %s", [s[0] for s in subsets])

    setup_pipeline(args.model)  # autoconf_dspy under the hood

    started_at = _utc_now_iso()
    per_subset: dict[str, dict] = {}
    t_run = time.perf_counter()
    try:
        for subset_name, subset_dir in subsets:
            per_subset[subset_name] = run_subset(
                subset_name, subset_dir, run_root, logger, args,
            )
    finally:
        finished_at = _utc_now_iso()
        manifest = build_manifest(
            args, date_str, [s[0] for s in subsets],
            per_subset, started_at, finished_at,
        )
        _atomic_write_json(run_root / "_manifest.json", manifest)

    totals = manifest["totals"]
    wall_s = int(time.perf_counter() - t_run)
    wall_fmt = f"{wall_s // 3600:02d}:{(wall_s % 3600) // 60:02d}:{wall_s % 60:02d}"
    summary_line = (
        f"OK={totals['n_ok']} ERR={totals['n_pipeline_error']} "
        f"CACHED={totals['n_cached']} N={totals['n_cases']} WALL={wall_fmt}"
    )
    logger.info(summary_line)
    print(summary_line)
    print(f"manifest: {run_root / '_manifest.json'}")

    if totals["n_pipeline_error"] > 0 and not args.tolerate_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
