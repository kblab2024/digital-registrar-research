#!/usr/bin/env python3
"""Cell-A wrapper — copy modular-DSPy predictions into the ablation tree.

Cell A is the parent project's pipeline run end-to-end with the modular
per-organ DSPy signatures (``CancerPipeline`` in
``digital_registrar_research.pipeline``). Since those predictions come
from full pipeline sweeps that run elsewhere, this "runner" doesn't
generate predictions — it copies them into
``workspace/results/ablations/dspy_modular_<model>/`` so the aggregator
can score them alongside Cells B and C.

Usage
-----
    python scripts/ablations/run_cell_a.py \\
        --modular-gpt-oss-dir E:/experiment/20260422/gpt-oss \\
        --modular-gpt4-dir   ../digitalregistrar-benchmarks/results/gpt4_dspy

Optional ``--out-root`` overrides the destination, used by smoke runners
to redirect into ``workspace/results/_smoke_<ts>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `_common` importable as a sibling module before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402, F401  (side-effect: src/ on sys.path)

from digital_registrar_research.ablations.runners.reuse_baseline import (  # noqa: E402
    run as run_cell_a,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modular-gpt-oss-dir", default=None,
                    help="directory with Cell A × gpt-oss predictions")
    ap.add_argument("--modular-gpt4-dir", default=None,
                    help="directory with Cell A × gpt-4-turbo predictions")
    ap.add_argument("--out-root", default=None,
                    help="override the default ABLATIONS_RESULTS root")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap matches copied (smoke uses --limit 2)")
    args = ap.parse_args()

    if not (args.modular_gpt_oss_dir or args.modular_gpt4_dir):
        ap.error("at least one of --modular-gpt-oss-dir / "
                 "--modular-gpt4-dir is required")

    ns = argparse.Namespace(
        modular_gpt_oss_dir=Path(args.modular_gpt_oss_dir) if args.modular_gpt_oss_dir else None,
        modular_gpt4_dir=Path(args.modular_gpt4_dir) if args.modular_gpt4_dir else None,
        out_root=Path(args.out_root) if args.out_root else None,
        limit=args.limit,
    )
    run_cell_a(ns)


if __name__ == "__main__":
    main()
