"""Shared helpers for ablation wrapper scripts.

The wrappers under ``scripts/ablations/`` are thin orchestrators on top
of the runners in ``src/digital_registrar_research/ablations/runners/``.
This module collects:

* the canonical ``{short-name → cell-id}`` mapping,
* the per-(cell, model) output-dir layout (delegated to ``_base``), and
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

# Make the package importable when running uninstalled.
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = str(REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Short names (--cell a, b, c, ...) → canonical cell IDs that match the
# on-disk subdir naming convention used by the canonical
# ``results/ablations/{dataset}/{cell_id}/{model_slug}/...`` layout.
CELL_MAP: dict[str, str] = {
    # Existing baselines (Cells A/B/C)
    "a": "dspy_modular",
    "b": "dspy_monolithic",
    "c": "raw_json",
    # Axis 1 — Pipeline decomposition extensions
    "a4": "no_router",
    "a5": "per_section",
    # Axis 2 — Output structuring extensions
    "b2": "str_outputs",
    "b4": "constrained_decoding",
    "b6": "free_text_regex",
    # Axis 3 — Prompting strategy
    "c2": "fewshot_demos",   # n_shots=3
    "c3": "fewshot_demos",   # n_shots=5
    "c4": "chain_of_thought",
    "c5": "compiled_dspy",
    "c6": "minimal_prompt",
    # Axis 6 — Schema specificity
    "f2": "union_schema",
    "f3": "flat_schema",
}
CELL_SHORT_TO_FULL = CELL_MAP

# C2 and C3 share the runner; the first short code wins the inverse map.
CELL_FULL_TO_SHORT: dict[str, str] = {}
for _short, _full in CELL_MAP.items():
    CELL_FULL_TO_SHORT.setdefault(_full, _short)


def smoke_root(folder: Path, dataset: str, *,
               prefix: str = "_smoke",
               ts: str | None = None) -> Path:
    """Return ``{folder}/results/ablations/{dataset}/_smoke_<ts>/``.

    The leading underscore keeps the directory out of the aggregator's
    default scan and out of any manuscript-table glob that filters
    ``startswith('_')``. Smoke runs deliberately collide with the
    canonical layout so the aggregator can be invoked the same way (with
    ``--results-root`` pointing at the smoke dir).
    """
    if ts is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    root = (Path(folder) / "results" / "ablations" / dataset
            / f"{prefix}_{ts}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_aggregator_round_trip(experiment_root: Path, dataset: str,
                                 cell_ids: list[str],
                                 model_slugs: list[str] | None = None,
                                 results_root: Path | None = None,
                                 ) -> None:
    """Run the eval aggregator and assert non-empty summary.

    Smoke wrappers call this after each cell finishes so a green smoke
    means **both** the cell and the eval reader work end-to-end.

    Pass ``results_root`` to point the aggregator at a smoke subdirectory
    instead of the canonical ``{folder}/results/ablations/{dataset}/``.
    """
    import pandas as pd

    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )

    eval_argv: list[str] = [
        "--folder", str(experiment_root),
        "--dataset", dataset,
        "--cells", *cell_ids,
        "--no-stats",
    ]
    if results_root is not None:
        eval_argv += ["--results-root", str(results_root)]
    if model_slugs:
        eval_argv += ["--models", *model_slugs]

    eval_main(eval_argv)

    summary_path = (results_root if results_root is not None else
                    experiment_root / "results" / "ablations" / dataset
                    ) / "ablation_summary.csv"
    if not summary_path.exists():
        raise RuntimeError(
            f"Aggregator did not write {summary_path} — smoke run failed.")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(
            f"{summary_path} is empty — aggregator produced no rows.")

    # Organ-label consistency: spot-check that every per-case JSON's
    # ``cancer_category`` matches the organ implied by its folder name
    # via the dataset-aware mapping. Catches the silent misalignment
    # that the old ``IMPLEMENTED_ORGANS[idx-1]`` mapping produced.
    from digital_registrar_research.benchmarks.organs import organ_n_to_name
    import json
    out_root = (results_root if results_root is not None else
                experiment_root / "results" / "ablations" / dataset)
    mismatches: list[str] = []
    for cell_dir in out_root.iterdir():
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        for model_dir in cell_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                for organ_dir in run_dir.iterdir():
                    if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
                        continue
                    expected = organ_n_to_name(dataset, organ_dir.name)
                    if not expected:
                        continue
                    for pred_path in organ_dir.glob("*.json"):
                        try:
                            with pred_path.open(encoding="utf-8") as f:
                                pred = json.load(f)
                        except Exception:
                            continue
                        if not isinstance(pred, dict):
                            continue
                        # Skip not-cancer / unknown-organ skips — they
                        # legitimately have no cancer_category.
                        if pred.get("_skip_reason"):
                            continue
                        got = pred.get("cancer_category")
                        if got and got != expected:
                            mismatches.append(
                                f"{cell_dir.name}/{model_dir.name}/"
                                f"{run_dir.name}/{organ_dir.name}/"
                                f"{pred_path.name}: expected={expected!r} "
                                f"got={got!r}")
                        # One mismatch per cell is enough for the smoke
                        # contract — break out to keep this fast.
                        break
                    if mismatches and any(m.startswith(cell_dir.name)
                                          for m in mismatches):
                        break
    if mismatches:
        joined = "\n  ".join(mismatches[:10])
        raise RuntimeError(
            f"organ-folder mismatch detected in {len(mismatches)} case(s):"
            f"\n  {joined}")


__all__ = [
    "REPO_ROOT", "CELL_MAP", "CELL_SHORT_TO_FULL", "CELL_FULL_TO_SHORT",
    "smoke_root", "ensure_aggregator_round_trip",
]
