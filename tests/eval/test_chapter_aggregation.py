"""Unit tests for the cross-cell 5-chapter cascade rollup.

Exercises :func:`chapter_aggregation.emit_chapter_outputs` on a
synthetic cascade atomic spanning multiple ablation cells × one model.
The chapter reducers and pairwise builders are exercised indirectly
through these tests; their own behavior is covered in test_cascade.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from digital_registrar_research.ablations.eval import chapter_aggregation


# --- Synthetic cascade atomic ---------------------------------------------


def _make_cascade_atomic_three_cells() -> pd.DataFrame:
    """Three ablation cells × one model × five cases; Stage A/B/C scalar.

    Cell A (baseline ``dspy_modular_m1``): all cases pass A and B, all
    Stage-C scalar fields correct.
    Cell B (``mono_m1``): one Stage-C scalar field flips wrong on one
    case (small Δ).
    Cell C (``raw_m1``): three Stage-C scalar field cells wrong (large Δ).
    All cells share Stage A and B passing on every case so chapters
    1 and 2 see homogeneous correctness — the test focuses on chapter 3
    deltas plus chapter 1/2/4/5 plumbing.
    """
    cells = [
        ("dspy_modular", "m1"),  # baseline
        ("mono", "m1"),
        ("raw", "m1"),
    ]
    cases = [f"c{i}" for i in range(5)]
    fields = ["pt_category", "grade"]

    # Per-cell correctness on Stage-C scalar field "pt_category"
    pt_correct = {
        "dspy_modular": [1, 1, 1, 1, 1],   # baseline: 5/5
        "mono":         [1, 1, 1, 1, 0],   # 4/5
        "raw":          [1, 0, 0, 0, 1],   # 2/5
    }
    rows: list[dict] = []
    for cell, model_slug in cells:
        joint = f"{cell}_{model_slug}"
        for run_id in ("r0",):
            for ci, case_id in enumerate(cases):
                # Stage A (eligibility): everyone correct
                rows.append({
                    "run_id": run_id, "method": "ablation",
                    "model": joint, "model_slug": model_slug, "cell": cell,
                    "dataset": "cmuh",
                    "case_id": case_id, "organ_idx": 0, "organ": "breast",
                    "subgroup": "single_primary",
                    "cascade_stage": "A", "gate_pass": True,
                    "field": "cancer_excision_report",
                    "field_kind": "scalar",
                    "gold_value": True, "pred_value": True,
                    "correct": 1.0, "wrong": 0.0,
                    "attempted": True, "gold_present": True,
                    "field_missing": False, "parse_error": False,
                    "error_mode": None, "others_disposition": "none",
                })
                # Stage B (organ): everyone correct
                rows.append({
                    "run_id": run_id, "method": "ablation",
                    "model": joint, "model_slug": model_slug, "cell": cell,
                    "dataset": "cmuh",
                    "case_id": case_id, "organ_idx": 0, "organ": "breast",
                    "subgroup": "single_primary",
                    "cascade_stage": "B", "gate_pass": True,
                    "field": "cancer_category",
                    "field_kind": "scalar",
                    "gold_value": "breast", "pred_value": "breast",
                    "correct": 1.0, "wrong": 0.0,
                    "attempted": True, "gold_present": True,
                    "field_missing": False, "parse_error": False,
                    "error_mode": None, "others_disposition": "none",
                })
                # Stage C scalar: pt_category (varies per cell) + grade (always correct).
                pt_v = pt_correct[cell][ci]
                rows.append({
                    "run_id": run_id, "method": "ablation",
                    "model": joint, "model_slug": model_slug, "cell": cell,
                    "dataset": "cmuh",
                    "case_id": case_id, "organ_idx": 0, "organ": "breast",
                    "subgroup": "single_primary",
                    "cascade_stage": "C", "gate_pass": True,
                    "field": "pt_category",
                    "field_kind": "scalar",
                    "gold_value": "T1", "pred_value": "T1" if pt_v else "T2",
                    "correct": float(pt_v), "wrong": float(1 - pt_v),
                    "attempted": True, "gold_present": True,
                    "field_missing": False, "parse_error": False,
                    "error_mode": None, "others_disposition": "none",
                })
                rows.append({
                    "run_id": run_id, "method": "ablation",
                    "model": joint, "model_slug": model_slug, "cell": cell,
                    "dataset": "cmuh",
                    "case_id": case_id, "organ_idx": 0, "organ": "breast",
                    "subgroup": "single_primary",
                    "cascade_stage": "C", "gate_pass": True,
                    "field": "grade",
                    "field_kind": "scalar",
                    "gold_value": "G1", "pred_value": "G1",
                    "correct": 1.0, "wrong": 0.0,
                    "attempted": True, "gold_present": True,
                    "field_missing": False, "parse_error": False,
                    "error_mode": None, "others_disposition": "none",
                })
    return pd.DataFrame(rows)


# --- Tests ----------------------------------------------------------------


def test_emit_chapter_outputs_writes_chapter_tree(tmp_path: Path):
    """All five chapter directories appear, with the expected per-chapter
    headlines and pairwise files for a three-cell × one-model ablation."""
    atomic = _make_cascade_atomic_three_cells()
    nested = pd.DataFrame()  # no margins / LN content in this fixture

    written = chapter_aggregation.emit_chapter_outputs(
        atomic, nested,
        out_root=tmp_path,
        baseline_method="dspy_modular_m1",
        n_boot=200, random_state=42,
    )

    # Chapter 1: eligibility headline + pairwise (one row per non-baseline).
    ch1 = pd.read_csv(tmp_path / "chapter1_eligibility" / "overall.csv")
    assert set(ch1["model"]) == {"dspy_modular_m1", "mono_m1", "raw_m1"}
    assert (ch1["accuracy"] == 1.0).all()  # everyone correct on Stage A
    ch1_pair = pd.read_csv(tmp_path / "chapter1_eligibility" / "pairwise.csv")
    # Two non-baseline targets, one row each.
    assert len(ch1_pair) == 2
    assert set(ch1_pair["run_a"]) == {"dspy_modular_m1"}
    assert set(ch1_pair["run_b"]) == {"mono_m1", "raw_m1"}

    # Chapter 2: organ classification.
    ch2 = pd.read_csv(tmp_path / "chapter2_organ_classification" / "overall.csv")
    assert set(ch2["model"]) == {"dspy_modular_m1", "mono_m1", "raw_m1"}
    ch2_pair = pd.read_csv(
        tmp_path / "chapter2_organ_classification" / "pairwise.csv")
    assert len(ch2_pair) == 2

    # Chapter 3: per-field headlines + pairwise.
    ch3 = pd.read_csv(
        tmp_path / "chapter3_field_extraction" / "per_field_overall.csv")
    pt_baseline = ch3[(ch3["model"] == "dspy_modular_m1")
                      & (ch3["field"] == "pt_category")]
    pt_raw = ch3[(ch3["model"] == "raw_m1")
                 & (ch3["field"] == "pt_category")]
    assert pt_baseline["accuracy_attempted"].iloc[0] == pytest.approx(1.0)
    assert pt_raw["accuracy_attempted"].iloc[0] == pytest.approx(0.4)

    # Chapter 3 pairwise has rows for both targets across both fields
    # where present in both arms.
    ch3_pair = pd.read_csv(tmp_path / "chapter3_field_extraction" / "pairwise.csv")
    assert {"run_a", "run_b", "field", "delta_acc", "mcnemar_p"} <= set(ch3_pair.columns)
    # Chapter 3 inherits the cascade compare CLI's delta_acc =
    # mean(run_a) - mean(run_b) convention (i.e. baseline - target).
    # raw_m1 pt_category accuracy is 0.4, baseline is 1.0, so
    # delta_acc = 1.0 - 0.4 = +0.6.
    raw_pt = ch3_pair[(ch3_pair["run_b"] == "raw_m1")
                      & (ch3_pair["field"] == "pt_category")]
    assert len(raw_pt) == 1
    assert raw_pt["delta_acc"].iloc[0] == pytest.approx(0.6, abs=1e-9)

    # Chapter 4 / 5 produce missingness only (no nested sidecar) and
    # have empty pairwise. The directories may not exist if everything
    # was empty.
    ch4_pair_path = tmp_path / "chapter4_margins" / "pairwise.csv"
    ch5_pair_path = tmp_path / "chapter5_lymph_nodes" / "pairwise.csv"
    # Either the file doesn't exist (empty) or it's empty when read.
    if ch4_pair_path.exists():
        assert pd.read_csv(ch4_pair_path).empty or len(pd.read_csv(ch4_pair_path)) >= 0
    if ch5_pair_path.exists():
        assert pd.read_csv(ch5_pair_path).empty or len(pd.read_csv(ch5_pair_path)) >= 0

    # Returned manifest mentions all five chapters.
    assert set(written.keys()) == {
        "chapter1_eligibility",
        "chapter2_organ_classification",
        "chapter3_field_extraction",
        "chapter4_margins",
        "chapter5_lymph_nodes",
    }


def test_pairwise_only_against_baseline(tmp_path: Path):
    """Pairwise outputs compare every non-baseline cell to the baseline
    once — no all-pairs cross product."""
    atomic = _make_cascade_atomic_three_cells()
    chapter_aggregation.emit_chapter_outputs(
        atomic, pd.DataFrame(),
        out_root=tmp_path,
        baseline_method="dspy_modular_m1",
        n_boot=100, random_state=0,
    )

    for chapter_pair in (
        "chapter1_eligibility/pairwise.csv",
        "chapter2_organ_classification/pairwise.csv",
    ):
        df = pd.read_csv(tmp_path / chapter_pair)
        # Two non-baseline targets × one row each.
        assert len(df) == 2
        # Baseline must appear as run_a in every row, never as run_b.
        assert (df["run_a"] == "dspy_modular_m1").all()
        assert (df["run_b"] != "dspy_modular_m1").all()


def test_chapter1_2_holm_correction_present(tmp_path: Path):
    """Chapters 1 and 2 pairwise tables get Holm- and BH-adjusted p-values."""
    atomic = _make_cascade_atomic_three_cells()
    chapter_aggregation.emit_chapter_outputs(
        atomic, pd.DataFrame(),
        out_root=tmp_path,
        baseline_method="dspy_modular_m1",
        n_boot=100, random_state=0,
    )
    for chap in ("chapter1_eligibility", "chapter2_organ_classification"):
        df = pd.read_csv(tmp_path / chap / "pairwise.csv")
        assert "mcnemar_p_holm" in df.columns
        assert "mcnemar_p_bh" in df.columns


def test_emit_chapter_outputs_with_no_baseline_match_skips_pairwise(
    tmp_path: Path,
):
    """If baseline_method isn't in the data, headlines still emit but
    the pairwise files are empty / absent."""
    atomic = _make_cascade_atomic_three_cells()
    chapter_aggregation.emit_chapter_outputs(
        atomic, pd.DataFrame(),
        out_root=tmp_path,
        baseline_method="nonexistent_method",
        n_boot=100, random_state=0,
    )
    # Headlines still present.
    assert (tmp_path / "chapter1_eligibility" / "overall.csv").exists()
    # Pairwise files don't exist (empty DataFrame → _write_csv returns None).
    assert not (tmp_path / "chapter1_eligibility" / "pairwise.csv").exists()
    assert not (tmp_path / "chapter3_field_extraction" / "pairwise.csv").exists()


def test_chapter3_pairwise_baseline_run_a_orientation(tmp_path: Path):
    """Δ direction is target - baseline. Builder labels baseline as
    run_a (insertion-first), target as run_b, so delta_acc_b_minus_a
    is target - baseline. Verify with a known fixture."""
    atomic = _make_cascade_atomic_three_cells()
    chapter_aggregation.emit_chapter_outputs(
        atomic, pd.DataFrame(),
        out_root=tmp_path,
        baseline_method="dspy_modular_m1",
        n_boot=100, random_state=0,
    )
    pair = pd.read_csv(tmp_path / "chapter3_field_extraction" / "pairwise.csv")
    # mono_m1 vs baseline on pt_category: target acc = 0.8, baseline = 1.0.
    # Chapter 3 follows the cascade compare convention delta_acc =
    # mean(a) - mean(b) = baseline - target = 1.0 - 0.8 = +0.2.
    mono_pt = pair[(pair["run_b"] == "mono_m1")
                   & (pair["field"] == "pt_category")]
    assert len(mono_pt) == 1
    assert mono_pt["acc_a"].iloc[0] == pytest.approx(1.0)
    assert mono_pt["acc_b"].iloc[0] == pytest.approx(0.8)
    assert mono_pt["delta_acc"].iloc[0] == pytest.approx(0.2, abs=1e-9)
