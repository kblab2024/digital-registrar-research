"""Shared helpers for ablation wrapper scripts.

The wrappers under ``scripts/ablations/`` are thin orchestrators on top
of the runners in ``src/digital_registrar_research/ablations/runners/``.
This module collects:

* the canonical {short-name → cell-id} mapping used everywhere downstream,
* a uniform smoke-output-directory format, and
* the ``sys.path`` bootstrap so the wrappers also work without a
  ``pip install -e .`` of the source tree.

Importing this module is enough to make ``digital_registrar_research``
importable by sibling scripts.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# scripts/ablations/_common.py → parents[2] = repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Make the package importable when running uninstalled. No-op if already
# resolvable (e.g. after `pip install -e .`).
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Short names exposed to users (--cell a/b/c) → canonical cell IDs that
# match the on-disk subdir naming convention used by
# ``digital_registrar_research.ablations.eval.run_ablations``.
CELL_MAP: dict[str, str] = {
    "a": "dspy_modular",
    "b": "dspy_monolithic",
    "c": "raw_json",
}
CELL_SHORT_TO_FULL = CELL_MAP
CELL_FULL_TO_SHORT = {v: k for k, v in CELL_MAP.items()}


def smoke_root(prefix: str = "_smoke", ts: str | None = None) -> Path:
    """Return ``workspace/results/_smoke_<ts>/`` (created if absent).

    ``ts`` defaults to ``YYYYMMDD-HHMM`` of *now*. The leading underscore
    keeps the directory out of ``run_ablations._discover``'s default scan
    and out of any manuscript-table glob that filters ``startswith('_')``.
    """
    from digital_registrar_research.paths import RESULTS_ROOT

    if ts is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    root = RESULTS_ROOT / f"{prefix}_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cell_output_dir(results_root: Path, cell_id: str, model: str) -> Path:
    """Return ``<results_root>/<cell_id>_<model>/``.

    Matches the convention ``run_ablations._discover`` expects, so the
    aggregator (called with ``--results-root <results_root>``) finds the
    cell automatically. ``model`` may contain colons (e.g. ``gpt-oss:20b``)
    — replaced with ``-`` so the directory name is filesystem-portable.
    """
    safe_model = model.replace(":", "-").replace("/", "-")
    d = results_root / f"{cell_id}_{safe_model}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_slug(model: str) -> str:
    """File-safe rendering of a model id (mirrors ``cell_output_dir``)."""
    return model.replace(":", "-").replace("/", "-")


def ensure_aggregator_round_trip(results_root: Path, cell_ids: list[str],
                                 model_slugs: list[str]) -> None:
    """Run the eval aggregator over ``results_root`` and assert non-empty.

    Smoke wrappers call this after each cell finishes so a green smoke
    means **both** the cell and the eval reader work end-to-end. Raises
    ``RuntimeError`` if the resulting summary CSV is empty or missing.
    """
    import pandas as pd

    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )

    eval_argv = [
        "--cells", *cell_ids,
        "--models", *model_slugs,
        "--results-root", str(results_root),
    ]
    eval_main(eval_argv)

    summary_path = results_root / "ablation_summary.csv"
    if not summary_path.exists():
        raise RuntimeError(
            f"Aggregator did not write {summary_path} — smoke run failed.")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(
            f"{summary_path} is empty — aggregator produced no rows. "
            "Check that the cell wrote per-case JSONs in the expected layout.")


__all__ = [
    "REPO_ROOT", "CELL_MAP", "CELL_SHORT_TO_FULL", "CELL_FULL_TO_SHORT",
    "smoke_root", "cell_output_dir", "model_slug",
    "ensure_aggregator_round_trip",
]
