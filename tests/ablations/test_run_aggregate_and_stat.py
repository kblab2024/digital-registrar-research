"""Tests for the manual aggregator's per-cell manifest rebuild.

Two-machine workflow: machine alpha2 wrote run01-alpha2; machine beta2
wrote run01-beta2. After rsync, the second machine's cell-level
``_manifest.yaml`` clobbered the first's, so the manifest lists only
half the runs. The manual aggregator's manifest rebuild must rewrite
this from on-disk runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))


def _make_run(model_dir: Path, run_id: str, *, n_cases: int = 3,
              n_pipeline_error: int = 0,
              seed: int | None = None) -> Path:
    run_dir = model_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": run_id, "cell": model_dir.parent.name,
        "model_slug": model_dir.name,
        "n_cases": n_cases, "n_pipeline_error": n_pipeline_error,
        "n_ok": n_cases - n_pipeline_error, "n_cached": 0,
        "seed": seed,
        "per_organ": {}, "wall_time_s": 1.0,
        "started_at": "2026-05-08T10:00:00Z",
        "finished_at": "2026-05-08T11:00:00Z",
    }
    (run_dir / "_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    meta = {
        "run": run_id, "cell": model_dir.parent.name,
        "model_alias": "gptoss",
        "model_slug": model_dir.name,
        "decoding": {"seed": seed},
    }
    (run_dir / "_run_meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    return run_dir


def test_rebuild_one_manifest_includes_all_runs(tmp_path: Path):
    """Two-machine tree (slugs alpha2, beta2). Pre-rebuild, the manifest
    might list only one machine's runs. Post-rebuild, the manifest
    must list every run found on disk."""
    from run_aggregate_and_stat import _rebuild_one_manifest

    cell_dir = tmp_path / "cmuh" / "dspy_monolithic"
    model_dir = cell_dir / "gpt_oss_20b"
    model_dir.mkdir(parents=True)

    _make_run(model_dir, "run01-alpha2", seed=111)
    _make_run(model_dir, "run02-alpha2", seed=222)
    _make_run(model_dir, "run01-beta2", seed=333)
    _make_run(model_dir, "run02-beta2", seed=444)

    # Simulate a stale manifest from the alpha2 rsync that listed only
    # the alpha2 runs:
    (model_dir / "_manifest.yaml").write_text(
        yaml.safe_dump({
            "cell": "dspy_monolithic", "dataset": "cmuh",
            "model_slug": "gpt_oss_20b", "model_alias": "gptoss",
            "runs": [
                {"run": "run01-alpha2", "valid": True},
                {"run": "run02-alpha2", "valid": True},
            ],
            "k": 2,
        }),
        encoding="utf-8")

    n = _rebuild_one_manifest(cell_dir, model_dir)
    assert n == 4

    with (model_dir / "_manifest.yaml").open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    run_names = {r["run"] for r in manifest["runs"]}
    assert run_names == {"run01-alpha2", "run02-alpha2",
                         "run01-beta2", "run02-beta2"}
    assert manifest["k"] == 4


def test_per_machine_breakdown(tmp_path: Path):
    """The summary print should bucket runs by machine slug."""
    from run_aggregate_and_stat import _per_machine_breakdown

    results_root = tmp_path / "cmuh"
    cell_dir = results_root / "dspy_monolithic"
    model_dir = cell_dir / "gpt_oss_20b"
    model_dir.mkdir(parents=True)
    _make_run(model_dir, "run01-alpha2")
    _make_run(model_dir, "run02-alpha2")
    _make_run(model_dir, "run01-beta2")

    out = _per_machine_breakdown(results_root)
    assert out[("dspy_monolithic", "gpt_oss_20b")] == {
        "alpha2": 2, "beta2": 1,
    }


def test_combined_grid_meta(tmp_path: Path):
    """Per-machine _grid_meta.json files should produce a thin combined
    record without merging the spec blocks."""
    from run_aggregate_and_stat import _write_combined_grid_meta

    results_root = tmp_path / "cmuh"
    results_root.mkdir(parents=True)

    # Two source grid metas (one per contributing machine).
    (results_root / "_grid_meta.json").write_text(json.dumps({
        "config_path": "configs/local/grid_1.yaml",
        "git_sha": "abc123",
        "completed_utc": "2026-05-08T11:00:00",
        "manifests": [{"cell": "dspy_monolithic"}],
    }), encoding="utf-8")
    (results_root / "_grid_meta_alpha2.json").write_text(json.dumps({
        "config_path": "configs/local/grid_1.yaml",
        "git_sha": "abc123",
        "completed_utc": "2026-05-07T11:00:00",
        "manifests": [{"cell": "dspy_modular"}],
    }), encoding="utf-8")

    out_path = _write_combined_grid_meta(results_root)
    assert out_path is not None
    with out_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["n_source_grids"] == 2
    files = {g["file"] for g in doc["source_grids"]}
    assert "_grid_meta.json" in files
    assert "_grid_meta_alpha2.json" in files
