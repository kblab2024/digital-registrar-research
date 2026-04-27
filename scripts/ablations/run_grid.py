#!/usr/bin/env python3
"""Full ablation grid driver — Grid 1 / Grid 2 / Grid 3.

Reads a YAML grid spec listing the (cell, model, args) tuples to run,
executes each in sequence, and aggregates at the end. Per the
suggestions doc §1.3:

  * Grid 1 — minimum-viable lesion study (~2-3 days; gpt-oss only)
  * Grid 2 — recommended factorial (~1 week; 3 models)
  * Grid 3 — stretch / supplementary

Initial implementation supports the **already-runnable** axes:
    Cell A (modular DSPy reuse), Cell B (monolithic ± skip-jsonize),
    Cell C (raw JSON, OpenAI / Ollama).

New axes from the suggestions doc §1.2 (constrained decoding,
few-shot, ChainOfThought, compiled DSPy) are NOT wired up here yet —
TODO comments mark where each one would slot in once their cells exist
under ``digital_registrar_research/ablations/runners/``.

YAML format
-----------
    name: grid_1
    description: "Minimum-viable lesion study"
    results_root: null  # null → digital_registrar_research.paths.ABLATIONS_RESULTS
    runs:
      - cell: dspy_monolithic
        model: gpt          # cell-specific (model_list key for Cell B)
        slug: gpt-oss        # display slug used in the output dir + aggregator
        args: {}            # per-cell extra kwargs (e.g. skip_jsonize: true)
      - cell: raw_json
        model: gpt-oss:20b
        slug: gpt-oss
        args: {}
      - cell: dspy_monolithic
        model: gpt
        slug: gpt-oss-nojsonize
        args:
          skip_jsonize: true

Usage
-----
    python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, cell_output_dir  # noqa: E402

import yaml  # noqa: E402

CELL_DISPATCH = {
    "dspy_modular":    "digital_registrar_research.ablations.runners.reuse_baseline",
    "dspy_monolithic": "digital_registrar_research.ablations.runners.dspy_monolithic",
    "raw_json":        "digital_registrar_research.ablations.runners.raw_json",
    # TODO(suggestions §1.2): when new cells land, register them here:
    #   "constrained_decoding": "...runners.constrained_decoding",
    #   "fewshot":              "...runners.fewshot",
    #   "chain_of_thought":     "...runners.cot",
    #   "compiled_dspy":        "...runners.compiled_dspy",
}


def _git_sha(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = repo_root / ".git" / ref.split(maxsplit=1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except Exception:
        return None


def _build_args_for_cell(cell: str, model: str, out_dir: Path,
                         extra: dict) -> argparse.Namespace:
    """Map the YAML ``args`` block to the runner's expected Namespace."""
    if cell == "dspy_modular":
        # Reuse-baseline expects a source dir (not a model name).
        ns = argparse.Namespace(
            modular_gpt_oss_dir=Path(extra["modular_gpt_oss_dir"])
                if "modular_gpt_oss_dir" in extra else None,
            modular_gpt4_dir=Path(extra["modular_gpt4_dir"])
                if "modular_gpt4_dir" in extra else None,
            out_root=out_dir.parent,  # runner will create cell_<model> within
            limit=extra.get("limit"),
        )
        return ns
    if cell == "dspy_monolithic":
        return argparse.Namespace(
            model=model, out=str(out_dir),
            limit=extra.get("limit"),
            skip_jsonize=extra.get("skip_jsonize", False),
        )
    if cell == "raw_json":
        return argparse.Namespace(
            model=model, api_base=extra.get("api_base"),
            out=str(out_dir), limit=extra.get("limit"),
        )
    raise ValueError(f"Unknown cell {cell!r}")


def _run_one(cell: str, model: str, slug: str, extra: dict,
             results_root: Path) -> dict:
    """Run a single (cell, model, slug) tuple and return a manifest row."""
    if cell not in CELL_DISPATCH:
        raise ValueError(f"Unknown cell {cell!r} — registered cells: "
                         f"{sorted(CELL_DISPATCH)}")
    out_dir = cell_output_dir(results_root, cell, slug)

    import importlib
    module = importlib.import_module(CELL_DISPATCH[cell])
    if not hasattr(module, "run"):
        raise RuntimeError(
            f"Cell module {module.__name__} has no run(args) function — "
            "did the runner refactor not land yet?")

    ns = _build_args_for_cell(cell, model, out_dir, extra)
    t0 = time.perf_counter()
    err: str | None = None
    try:
        module.run(ns)
    except Exception as e:
        err = repr(e)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        manifest = {
            "cell": cell,
            "model": model,
            "slug": slug,
            "out_dir": str(out_dir),
            "args": extra,
            "elapsed_s": elapsed,
            "git_sha": _git_sha(REPO_ROOT),
            "started_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "error": err,
        }
        meta_path = out_dir / "_run_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  → wrote {meta_path}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True,
                    help="YAML grid spec — see configs/ablations/grid_1.yaml")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="run the cells but don't run run_ablations afterwards")
    args = ap.parse_args()

    if not args.config.is_file():
        sys.exit(f"Grid config not found: {args.config}")

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not spec.get("runs"):
        sys.exit(f"No runs listed in {args.config}")

    from digital_registrar_research.paths import ABLATIONS_RESULTS

    results_root_arg = spec.get("results_root")
    results_root = Path(results_root_arg) if results_root_arg else ABLATIONS_RESULTS
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"[grid] config={args.config}")
    print(f"[grid] name={spec.get('name', '<unnamed>')}")
    print(f"[grid] runs={len(spec['runs'])}  results_root={results_root}")

    manifests: list[dict] = []
    cells_seen: set[str] = set()
    slugs_seen: set[str] = set()

    t0 = time.perf_counter()
    for i, run in enumerate(spec["runs"], 1):
        cell = run["cell"]
        model = run.get("model", "")
        slug = run.get("slug") or model
        extra = run.get("args") or {}
        print(f"\n[{i}/{len(spec['runs'])}] cell={cell} model={model} slug={slug}")
        manifest = _run_one(cell, model, slug, extra, results_root)
        manifests.append(manifest)
        cells_seen.add(cell)
        slugs_seen.add(slug)

    print(f"\n[grid] all runs complete ({time.perf_counter() - t0:.1f}s)")

    # Top-level grid manifest
    grid_meta = {
        "config_path": str(args.config),
        "spec": spec,
        "manifests": manifests,
        "git_sha": _git_sha(REPO_ROOT),
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (results_root / "_grid_meta.json").write_text(
        json.dumps(grid_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.skip_aggregate:
        print("[grid] --skip-aggregate set; not running run_ablations")
        return

    print("\n[grid] running aggregator…")
    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )
    eval_argv = [
        "--cells", *sorted(cells_seen),
        "--models", *sorted(slugs_seen),
        "--results-root", str(results_root),
    ]
    eval_main(eval_argv)
    print(f"\n[grid] OK — see {results_root}/ablation_summary.csv")


if __name__ == "__main__":
    main()
