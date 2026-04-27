#!/usr/bin/env python3
"""Per-cell smoke runner for ablations.

Runs **one** ablation cell on a small case sample and round-trips the
output through the eval aggregator. Designed to be the go/no-go before
launching a multi-day Grid-1/2 sweep.

Contract (from suggestions doc §1.6):
  * 1 model, 1 seed, 2-3 cases (default n=2)
  * Output dir prefixed with ``_smoke_<ts>/`` so the regular aggregator
    glob ignores it
  * **Fail loud** — any exception propagates, exit code != 0
  * Round-trips through ``run_ablations.main()`` so the eval reader is
    exercised, not just the cell
  * Asserts ``ablation_summary.csv`` is non-empty before returning

Usage
-----
    # Cell C × local gpt-oss:20b — fastest smoke (no DSPy)
    python scripts/ablations/run_cell_smoke.py --cell c --model gpt-oss:20b

    # Cell B × local gpt-oss (via models.common.model_list key)
    python scripts/ablations/run_cell_smoke.py --cell b --model gpt --n 2

    # Cell A — needs a source dir of pre-computed predictions
    python scripts/ablations/run_cell_smoke.py --cell a \\
        --modular-source-dir E:/experiment/20260422/gpt-oss \\
        --model gpt-oss
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELL_MAP,
    cell_output_dir,
    ensure_aggregator_round_trip,
    model_slug,
    smoke_root,
)


def _run_cell_b(model: str, n: int, out_dir: Path,
                skip_jsonize: bool = False) -> None:
    from digital_registrar_research.ablations.runners.dspy_monolithic import (
        run as run_cell_b,
    )
    args = argparse.Namespace(
        model=model, out=str(out_dir), limit=n, skip_jsonize=skip_jsonize,
    )
    run_cell_b(args)


def _run_cell_c(model: str, n: int, out_dir: Path,
                api_base: str | None = None) -> None:
    from digital_registrar_research.ablations.runners.raw_json import (
        run as run_cell_c,
    )
    args = argparse.Namespace(
        model=model, api_base=api_base, out=str(out_dir), limit=n,
    )
    run_cell_c(args)


def _run_cell_a(modular_source_dir: Path, n: int,
                results_root: Path, model: str) -> None:
    """For Cell A, the smoke is "copy <n> predictions and round-trip the
    aggregator". The runner handles --limit so we forward it."""
    from digital_registrar_research.ablations.runners.reuse_baseline import (
        run as run_cell_a,
    )
    # Cell A's runner picks the destination subdir name based on which
    # `--modular-*-dir` flag is set (gpt-oss → dspy_modular_gpt-oss,
    # gpt4 → dspy_modular_gpt4). For arbitrary smoke models, route via
    # the gpt-oss flag — the resulting dir is `dspy_modular_gpt-oss`,
    # which we re-tag for the aggregator below.
    args = argparse.Namespace(
        modular_gpt_oss_dir=modular_source_dir,
        modular_gpt4_dir=None,
        out_root=results_root,
        limit=n,
    )
    run_cell_a(args)
    # Cell A always names the dir 'dspy_modular_gpt-oss' regardless of
    # the actual model — caller is responsible for matching --models on
    # the aggregator call. Smoke uses `gpt-oss` as the canonical label.
    _ = model  # placeholder for future per-model dir naming


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True, choices=list(CELL_MAP.keys()),
                    help="ablation cell short-name: a (modular), "
                         "b (monolithic), c (raw JSON)")
    ap.add_argument("--model", required=True,
                    help="model id; for Cell B this is a key in "
                         "models.common.model_list (e.g. 'gpt'); for "
                         "Cells A/C it is the display label used to "
                         "match the aggregator's --models")
    ap.add_argument("--n", type=int, default=2,
                    help="number of cases to run (default 2)")
    ap.add_argument("--api-base", default=None,
                    help="(Cell C) override API base; default is local "
                         "Ollama for gpt-oss/gemma/qwen/phi/llama")
    ap.add_argument("--modular-source-dir", default=None,
                    help="(Cell A) source directory of modular DSPy "
                         "predictions to copy")
    args = ap.parse_args()

    cell_id = CELL_MAP[args.cell]
    slug = model_slug(args.model)
    root = smoke_root()
    print(f"[smoke] cell={args.cell} ({cell_id}) model={args.model} "
          f"slug={slug} n={args.n} root={root}")

    if args.cell == "a":
        if not args.modular_source_dir:
            ap.error("--modular-source-dir is required for --cell a")
        # Cell A always produces dspy_modular_gpt-oss; force model_slug
        # to match so the aggregator finds it.
        slug = "gpt-oss"
        _run_cell_a(Path(args.modular_source_dir), args.n, root, args.model)
    elif args.cell == "b":
        out_dir = cell_output_dir(root, cell_id, args.model)
        _run_cell_b(args.model, args.n, out_dir)
    elif args.cell == "c":
        out_dir = cell_output_dir(root, cell_id, args.model)
        _run_cell_c(args.model, args.n, out_dir, args.api_base)

    print(f"\n[smoke] cell finished — round-tripping aggregator…")
    ensure_aggregator_round_trip(root, [cell_id], [slug])
    print(f"[smoke] OK — summary at {root / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
