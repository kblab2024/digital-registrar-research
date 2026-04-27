#!/usr/bin/env python3
"""Cell-B wrapper — DSPy + monolithic single-signature per organ.

Replaces the modular per-section DSPy chain with one monolithic
signature per organ. Same ``is_cancer`` router and (optionally) same
``ReportJsonize`` intermediate as Cell A; the difference is a single
``dspy.Predict`` over a merged organ signature instead of five-to-seven
chained predicts.

Backed by ``digital_registrar_research.ablations.runners.dspy_monolithic``.

Usage
-----
    # Cell B × gpt-oss:20b (local Ollama)
    python scripts/ablations/run_cell_b.py --model gpt \\
        --out workspace/results/ablations/dspy_monolithic_gpt-oss

    # Drop the ReportJsonize intermediate too (ablation-of-ablation)
    python scripts/ablations/run_cell_b.py --model gpt --skip-jsonize \\
        --out workspace/results/ablations/dspy_monolithic_gpt-oss_nojsonize

    # Cell B × gpt-4-turbo
    OPENAI_API_KEY=... python scripts/ablations/run_cell_b.py --model gpt4 \\
        --out workspace/results/ablations/dspy_monolithic_gpt4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402, F401

from digital_registrar_research.ablations.runners.dspy_monolithic import (  # noqa: E402
    run as run_cell_b,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="key in models.common.model_list (e.g. 'gpt') or 'gpt4'")
    ap.add_argument("--out", required=True,
                    help="output directory (per-case JSON)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-jsonize", action="store_true",
                    help="ablation-of-ablation: also drop ReportJsonize")
    args = ap.parse_args()

    run_cell_b(args)


if __name__ == "__main__":
    main()
