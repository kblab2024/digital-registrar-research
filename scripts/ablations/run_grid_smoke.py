#!/usr/bin/env python3
"""Grid-wide smoke runner — pre-flight before a multi-day Grid-1/2 sweep.

Runs every requested ablation cell on a 2-3 case sample, then aggregates
across all cells in a single timestamped output root. Designed to catch
inter-cell layout regressions and aggregator-side bugs that a per-cell
smoke would miss.

Contract (from suggestions doc §1.6):
  * Single timestamped output root ``workspace/results/_smoke_<ts>/``
  * Each cell writes into ``<root>/<cell_id>_<model_slug>/``
  * Default cells: B and C (Cell A only if ``--modular-source-dir`` set)
  * Wall-time target: ≤ 10 minutes total
  * **Fail loud** — first cell exception aborts the rest

Usage
-----
    # Smoke Cells B and C on local gpt-oss
    python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b

    # Include Cell A by pointing at a source of modular DSPy predictions
    python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b \\
        --modular-source-dir E:/experiment/20260422/gpt-oss

    # Subset cells (skip B if Ollama bootstrap is slow on this machine)
    python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b \\
        --cells c
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELL_MAP,
    cell_output_dir,
    ensure_aggregator_round_trip,
    model_slug,
    smoke_root,
)


def _resolve_cell_b_model(arg: str) -> str:
    """Best-effort mapping from a literal Ollama model id back to a key
    in ``models.common.model_list`` so Cell B's runner accepts it.

    Cell B's ``_setup_model`` expects keys like ``gpt`` / ``gemma3`` /
    ``qwen3`` / ``medgemmasmall``. If the user passes ``gpt-oss:20b``
    we map it to ``gpt`` automatically; if they pass ``gpt`` we leave
    it alone.
    """
    from digital_registrar_research.models.common import model_list

    if arg in model_list or arg == "gpt4":
        return arg
    # Heuristic mapping; extend as needed.
    if arg.startswith("gpt-oss"):
        return "gpt"
    if arg.startswith("gemma3"):
        return "gemma3"
    if arg.startswith("gemma4"):
        return "gemma4"
    if arg.startswith("qwen"):
        return "qwen3_5"
    if arg.startswith("medgemma"):
        return "medgemmalarge" if "27" in arg else "medgemmasmall"
    raise ValueError(
        f"Cannot map model id {arg!r} to a Cell B model key. "
        f"Available keys: {sorted(model_list.keys()) + ['gpt4']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="model id (e.g. 'gpt-oss:20b' or 'gpt')")
    ap.add_argument("--cells", nargs="+", default=["b", "c"],
                    choices=list(CELL_MAP.keys()),
                    help="cells to smoke (default: b c)")
    ap.add_argument("--n", type=int, default=2,
                    help="cases per cell (default 2)")
    ap.add_argument("--api-base", default=None,
                    help="(Cell C) override API base")
    ap.add_argument("--modular-source-dir", default=None,
                    help="(Cell A) source directory of modular DSPy "
                         "predictions; only required if 'a' is in --cells")
    args = ap.parse_args()

    if "a" in args.cells and not args.modular_source_dir:
        ap.error("--modular-source-dir is required when --cells contains 'a'")

    root = smoke_root()
    slug = model_slug(args.model)
    cells_run: list[str] = []
    slugs_run: list[str] = []
    t0 = time.perf_counter()

    print(f"[grid-smoke] root={root} model={args.model} cells={args.cells} "
          f"n={args.n}")

    for short in args.cells:
        cell_id = CELL_MAP[short]
        print(f"\n=== smoking cell {short} ({cell_id}) ===")
        t_cell = time.perf_counter()

        if short == "a":
            from digital_registrar_research.ablations.runners.reuse_baseline import (
                run as run_cell_a,
            )
            run_cell_a(argparse.Namespace(
                modular_gpt_oss_dir=Path(args.modular_source_dir),
                modular_gpt4_dir=None,
                out_root=root,
                limit=args.n,
            ))
            cells_run.append(cell_id)
            slugs_run.append("gpt-oss")  # Cell A's runner hardcodes this label
        elif short == "b":
            from digital_registrar_research.ablations.runners.dspy_monolithic import (
                run as run_cell_b,
            )
            out_dir = cell_output_dir(root, cell_id, args.model)
            b_model = _resolve_cell_b_model(args.model)
            run_cell_b(argparse.Namespace(
                model=b_model, out=str(out_dir), limit=args.n,
                skip_jsonize=False,
            ))
            cells_run.append(cell_id)
            slugs_run.append(slug)
        elif short == "c":
            from digital_registrar_research.ablations.runners.raw_json import (
                run as run_cell_c,
            )
            out_dir = cell_output_dir(root, cell_id, args.model)
            run_cell_c(argparse.Namespace(
                model=args.model, api_base=args.api_base,
                out=str(out_dir), limit=args.n,
            ))
            cells_run.append(cell_id)
            slugs_run.append(slug)

        print(f"=== cell {short} done in {time.perf_counter() - t_cell:.1f}s")

    # De-duplicate slugs while preserving order so the aggregator only
    # iterates over each (cell, model) pair once.
    uniq_slugs = list(dict.fromkeys(slugs_run))
    uniq_cells = list(dict.fromkeys(cells_run))

    print(f"\n[grid-smoke] all cells finished — round-tripping aggregator…")
    ensure_aggregator_round_trip(root, uniq_cells, uniq_slugs)

    elapsed = time.perf_counter() - t0
    print(f"\n[grid-smoke] OK — total wall time {elapsed:.1f}s")
    print(f"[grid-smoke] summary at {root / 'ablation_summary.csv'}")
    if elapsed > 600:
        print(f"[grid-smoke] WARNING: wall time exceeded 10-minute target. "
              f"Consider reducing --n or --cells before running real grids.")


if __name__ == "__main__":
    main()
