#!/usr/bin/env python3
"""Axis A3 wrapper — DSPy monolithic without ``ReportJsonize``.

Backed by
``digital_registrar_research.ablations.runners.dspy_monolithic_no_jsonize``.

Usage::

    python scripts/ablations/run_cell_b_no_jsonize.py \\
        --folder dummy --dataset tcga --model gptoss
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401

from digital_registrar_research.ablations.runners.dspy_monolithic_no_jsonize import (  # noqa: E402
    main as runner_main,
)

if __name__ == "__main__":
    sys.exit(runner_main())
