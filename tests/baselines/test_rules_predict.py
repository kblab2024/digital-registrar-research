"""End-to-end test for the rule_based predict pipeline.

Builds a tiny synthetic data root (mimicking dummy/data/<dataset>/), runs
``rules.predict()`` against it, and asserts the prediction layout +
content match the contract that ``benchmarks/eval/run_all.py`` expects.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from digital_registrar_research.benchmarks.baselines import rules
from digital_registrar_research.benchmarks.eval.metrics import (
    aggregate_cases_to_df,
)


# Per-organ-index schema follows the project convention used in
# gen_dummy_skeleton.py: 1=breast, 2=colorectal, 3=esophagus, 4=liver, 5=stomach.
ORGAN_BY_IDX = {
    "1": "breast",
    "2": "colorectal",
    "3": "esophagus",
    "4": "liver",
    "5": "stomach",
}


def _build_synthetic_data_root(root: Path) -> None:
    """Create dummy/data/<ds>/{reports,annotations/gold}/<idx>/<id>.{txt,json}."""
    fixtures: dict[tuple[str, str], tuple[str, dict]] = {
        ("cmuh", "1"): (
            "cmuh1_1",
            (
                "Modified radical mastectomy. Infiltrating ductal carcinoma. "
                "Nottingham grade G2. Tumor size 2.5 cm. "
                "Stage pT2 N1mi MX. Lymphovascular invasion present."
            ),
        ),
        ("tcga", "1"): (
            "tcga1_1",
            (
                "Wide local excision. Invasive lobular carcinoma. "
                "Grade 1. Tumor size 8 mm. pT1c N0 MX."
            ),
        ),
        ("tcga", "3"): (
            "tcga3_1",
            (
                "Esophagectomy. Squamous cell carcinoma. Grade 2. "
                "pT3 N2 MX. Stage IIIB. Lymphovascular invasion present."
            ),
        ),
    }

    for (ds, organ_idx), (case_id, report) in fixtures.items():
        ds_root = root / "data" / ds
        rep_dir = ds_root / "reports" / organ_idx
        ann_dir = ds_root / "annotations" / "gold" / organ_idx
        rep_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"{case_id}.txt").write_text(report, encoding="utf-8")
        gold = {
            "cancer_excision_report": True,
            "cancer_category": ORGAN_BY_IDX[organ_idx],
            "cancer_category_others_description": None,
            "cancer_data": {},
        }
        (ann_dir / f"{case_id}.json").write_text(
            json.dumps(gold), encoding="utf-8"
        )

    # Splits: every fixture goes into "test" (no train phase for rules).
    for ds in ("cmuh", "tcga"):
        ids = [cid for (d, _), (cid, _) in fixtures.items() if d == ds]
        if not ids:
            continue
        splits = {"train": [], "test": ids}
        (root / "data" / ds / "splits.json").write_text(
            json.dumps(splits), encoding="utf-8"
        )


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    _build_synthetic_data_root(tmp_path)
    return tmp_path


def test_predict_writes_per_dataset_jsons(
    synthetic_data_root: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        data_root=str(synthetic_data_root),
        dataset="both",
        datasets="cmuh,tcga",
        organs=",".join(["breast", "colorectal", "esophagus",
                          "liver", "stomach"]),
        out=str(out_dir),
    )
    rules.predict(args)

    # cmuh1_1 (breast)
    cmuh = out_dir / "cmuh" / "cmuh1_1.json"
    assert cmuh.exists()
    pred = json.loads(cmuh.read_text(encoding="utf-8"))
    assert pred["cancer_category"] == "breast"
    assert pred["cancer_excision_report"] is True
    assert pred["cancer_data"]["pt_category"] == "t2"
    assert pred["cancer_data"]["lymphovascular_invasion"] is True

    # tcga1_1 (breast)
    tcga = out_dir / "tcga" / "tcga1_1.json"
    assert tcga.exists()

    # tcga3_1 (esophagus)
    tcga3 = out_dir / "tcga" / "tcga3_1.json"
    assert tcga3.exists()
    pred3 = json.loads(tcga3.read_text(encoding="utf-8"))
    assert pred3["cancer_category"] == "esophagus"
    assert pred3["cancer_data"]["pt_category"] == "t3"


def test_predict_outputs_score_with_eval_harness(
    synthetic_data_root: Path, tmp_path: Path,
) -> None:
    """End-to-end: rule predictions feed cleanly into aggregate_cases_to_df."""
    from digital_registrar_research.benchmarks.baselines._data import load_cases
    from digital_registrar_research.benchmarks.eval.bert_scope import (
        bert_scope_for_organ,
    )

    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        data_root=str(synthetic_data_root),
        dataset="both",
        datasets="cmuh,tcga",
        organs=",".join(["breast", "colorectal", "esophagus",
                          "liver", "stomach"]),
        out=str(out_dir),
    )
    rules.predict(args)

    cases = load_cases(
        datasets=["cmuh", "tcga"], split="test",
        root=synthetic_data_root,
        organs={"breast", "colorectal", "esophagus", "liver", "stomach"},
    )
    assert len(cases) == 3
    df = aggregate_cases_to_df(
        cases, {"rule_based": out_dir}, scope=bert_scope_for_organ,
    )
    assert not df.empty
    assert set(df["method"].unique()) == {"rule_based"}
    assert set(df["dataset"].unique()) == {"cmuh", "tcga"}
