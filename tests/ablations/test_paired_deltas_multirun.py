"""Regression tests for ``paired_deltas_vs_baseline`` under multirun.

The historical bug: when `cascade_atomic` carried multiple rows per
(case_id, field) — one per run_id — `set_index("case_id")` produced a
duplicate-keyed index. ``common = sub.index.intersection(...)`` returned
the *unique* intersection (e.g. 75 cases) but ``sub.loc[common,
"attempted"]`` returned 75 × n_runs rows, so the boolean mask was longer
than ``common``. The fix collapses both sides via
``groupby(["case_id","field"]).agg(...)`` before computing ``common``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from digital_registrar_research.ablations.eval.stats import (
    _collapse_runs_per_case_field,
    paired_deltas_vs_baseline,
)


def _make_multirun_grid(n_cases: int = 10, n_runs: int = 3,
                        seed: int = 0) -> pd.DataFrame:
    """Synthesize a long-form ablation grid with ``n_runs`` runs per
    (case_id, field) for each of two methods. Only one binary field
    (so McNemar exercises). Method A is the baseline; method B differs
    on a few cases per run."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for run in range(n_runs):
        for case in range(n_cases):
            base_correct = bool(rng.integers(0, 2))
            target_correct = base_correct if rng.random() > 0.4 else (
                not base_correct)
            rows.append({
                "method": "modular_m",
                "cell": "modular",
                "model": "m",
                "run_id": f"run0{run + 1}-alpha2",
                "case_id": f"c{case:03d}",
                "field": "tumor_present",
                "correct": base_correct,
                "attempted": True,
            })
            rows.append({
                "method": "monolithic_m",
                "cell": "monolithic",
                "model": "m",
                "run_id": f"run0{run + 1}-alpha2",
                "case_id": f"c{case:03d}",
                "field": "tumor_present",
                "correct": target_correct,
                "attempted": True,
            })
    return pd.DataFrame(rows)


def test_collapse_runs_per_case_field_dedupes():
    df = _make_multirun_grid(n_cases=5, n_runs=3)
    sub = df[df["method"] == "modular_m"]
    collapsed = _collapse_runs_per_case_field(sub)
    # 5 cases × 1 field × 1 method = 5 rows
    assert len(collapsed) == 5
    # Keys are unique:
    assert collapsed.set_index(["case_id", "field"]).index.is_unique
    # correct_f is a probability in [0, 1]
    assert (collapsed["correct_f"] >= 0).all()
    assert (collapsed["correct_f"] <= 1).all()


def test_paired_deltas_vs_baseline_no_index_error_on_multirun():
    """The original IndexError 150 vs 75 reproducer: 75-case × 2-run grid.
    Pre-fix this raised ``IndexError: Boolean index has wrong length``;
    post-fix it returns a populated DataFrame.
    """
    df = _make_multirun_grid(n_cases=75, n_runs=2)
    out = paired_deltas_vs_baseline(
        df, baseline_method="modular_m",
        n_boot=50,  # tiny — we're testing shape, not bootstrap stability
        random_state=42,
    )
    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    # Exactly one row per (target_method, field):
    assert len(out) == 1
    assert out["n_paired"].iloc[0] == 75
    assert out["cell"].iloc[0] == "monolithic"


def test_paired_deltas_vs_baseline_handles_machine_slug_run_ids():
    """Multi-machine runs use slugged run_ids like run01-alpha2 / run01-beta2.
    The collapse keys on (case_id, field) so the slug suffix doesn't
    disturb the reduction."""
    df = _make_multirun_grid(n_cases=10, n_runs=2)
    df.loc[df["run_id"] == "run02-alpha2", "run_id"] = "run01-beta2"
    out = paired_deltas_vs_baseline(
        df, baseline_method="modular_m",
        n_boot=50, random_state=42,
    )
    assert not out.empty
    assert out["n_paired"].iloc[0] == 10


def test_paired_deltas_mcnemar_majority_vote():
    """Under multirun the McNemar discordant counts come from
    threshold-collapsing each case's mean correctness at 0.5
    (majority-vote-correct, ties counted as correct)."""
    # 4 cases × 3 runs. Build correctness vectors so the post-collapse
    # binary vectors are deterministic:
    #   case 0: baseline 1/1/1, target 1/1/1 → both 1 (concordant)
    #   case 1: baseline 1/1/1, target 0/0/0 → 1 vs 0  (discordant b)
    #   case 2: baseline 0/0/0, target 1/1/1 → 0 vs 1  (discordant c)
    #   case 3: baseline 0/0/0, target 0/0/0 → both 0 (concordant)
    rows: list[dict] = []
    plan = {
        "c000": (1, 1),
        "c001": (1, 0),
        "c002": (0, 1),
        "c003": (0, 0),
    }
    for run in range(3):
        for case_id, (a, b) in plan.items():
            rows.append({"method": "modular_m", "cell": "modular", "model": "m",
                         "run_id": f"run0{run + 1}-alpha2",
                         "case_id": case_id, "field": "tumor_present",
                         "correct": bool(a), "attempted": True})
            rows.append({"method": "monolithic_m", "cell": "monolithic", "model": "m",
                         "run_id": f"run0{run + 1}-alpha2",
                         "case_id": case_id, "field": "tumor_present",
                         "correct": bool(b), "attempted": True})
    df = pd.DataFrame(rows)
    out = paired_deltas_vs_baseline(
        df, baseline_method="modular_m",
        n_boot=50, random_state=0,
    )
    row = out.iloc[0]
    # baseline correct & target wrong = 1, baseline wrong & target correct = 1
    assert row["mcnemar_b"] == 1
    assert row["mcnemar_c"] == 1
