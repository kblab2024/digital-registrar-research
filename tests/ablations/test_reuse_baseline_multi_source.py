"""Tests for ``runners.reuse_baseline`` multi-source import.

Each source pipeline run should land in its own ablation ``runNN`` slot
with ``_run_meta.json["source_run"]`` referencing the origin. The
aggregator's _discover_runs then sees N runs and the cascade reduction
treats them as a multirun.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from digital_registrar_research.ablations.runners.reuse_baseline import (
    _resolve_source_runs,
)


def _make_pipeline_run(pipeline_dir: Path, run_id: str,
                       organ_n: str = "1", n_cases: int = 3) -> Path:
    """Synthesize a completed pipeline run with N per-case JSONs and an
    ``_summary.json`` to mark it complete."""
    run_dir = pipeline_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_summary.json").write_text(
        json.dumps({"n_cases": n_cases, "n_ok": n_cases,
                    "n_pipeline_error": 0}),
        encoding="utf-8")
    organ_dir = run_dir / organ_n
    organ_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cases):
        (organ_dir / f"case_{i:03d}.json").write_text(
            json.dumps({"cancer_excision_report": True,
                        "cancer_category": "breast"}),
            encoding="utf-8")
    return run_dir


# --- _resolve_source_runs ---------------------------------------------------

def test_resolve_source_runs_default_picks_most_recent(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    _make_pipeline_run(pipeline_dir, "run01")
    _make_pipeline_run(pipeline_dir, "run02")
    _make_pipeline_run(pipeline_dir, "run03")
    out = _resolve_source_runs(pipeline_dir, source_run=None,
                               source_runs=None, all_source_runs=False)
    assert out == ["run03"]


def test_resolve_source_runs_explicit_single(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    _make_pipeline_run(pipeline_dir, "run01")
    _make_pipeline_run(pipeline_dir, "run02")
    out = _resolve_source_runs(pipeline_dir, source_run="run01",
                               source_runs=None, all_source_runs=False)
    assert out == ["run01"]


def test_resolve_source_runs_explicit_list(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    for r in ("run01", "run02", "run03"):
        _make_pipeline_run(pipeline_dir, r)
    out = _resolve_source_runs(
        pipeline_dir, source_run=None,
        source_runs=["run01", "run03"], all_source_runs=False)
    assert out == ["run01", "run03"]


def test_resolve_source_runs_all(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    for r in ("run01", "run02", "run03"):
        _make_pipeline_run(pipeline_dir, r)
    out = _resolve_source_runs(pipeline_dir, source_run=None,
                               source_runs=None, all_source_runs=True)
    assert out == ["run01", "run02", "run03"]


def test_resolve_source_runs_mutually_exclusive(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    _make_pipeline_run(pipeline_dir, "run01")
    with pytest.raises(SystemExit):
        _resolve_source_runs(pipeline_dir, source_run="run01",
                             source_runs=["run01"], all_source_runs=False)


def test_resolve_source_runs_explicit_missing(tmp_path: Path):
    pipeline_dir = tmp_path / "pred"
    _make_pipeline_run(pipeline_dir, "run01")
    with pytest.raises(SystemExit):
        _resolve_source_runs(pipeline_dir, source_run=None,
                             source_runs=["run99"], all_source_runs=False)
