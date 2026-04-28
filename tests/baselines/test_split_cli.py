"""Tests for the refactored registrar-split CLI (canonical, multi-dataset)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from digital_registrar_research.benchmarks import organs as _organs  # noqa: E402
from digital_registrar_research.benchmarks.data.split import (  # noqa: E402
    load_gold_cases, main as split_main, stratified_split,
)


def _seed_canonical_gold(folder: Path, dataset: str, organs: dict[int, str],
                         cases_per_organ: int = 5) -> None:
    """Write canonical gold annotations: {folder}/data/{dataset}/annotations/gold/<organ>/<id>.json."""
    for organ_idx, organ_name in organs.items():
        d = folder / "data" / dataset / "annotations" / "gold" / str(organ_idx)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, cases_per_organ + 1):
            case_id = f"{dataset}{organ_idx}_{i}"
            (d / f"{case_id}.json").write_text(
                json.dumps({
                    "cancer_excision_report": True,
                    "cancer_category": organ_name,
                    "cancer_category_others_description": None,
                    "cancer_data": {},
                }), encoding="utf-8",
            )


def _yaml_organs(dataset: str, names: list[str]) -> dict[int, str]:
    """Build a {organ_n: organ_name} dict using the dataset's canonical
    yaml mapping — keeps the seeded fixture aligned with reality."""
    full = _organs.organs_for(dataset)
    rev = {n.lower(): i for i, n in full.items()}
    return {rev[name.lower()]: name for name in names}


def test_load_gold_cases_walks_canonical(tmp_path: Path) -> None:
    # CMUH organ_n: 2=breast, 5=esophagus (per configs/organ_code.yaml).
    organs = _yaml_organs("cmuh", ["breast", "esophagus"])
    _seed_canonical_gold(tmp_path, "cmuh", organs, cases_per_organ=2)
    cases = load_gold_cases(tmp_path, "cmuh")
    assert len(cases) == 4
    cats = sorted(c["cancer_category"] for c in cases)
    assert cats == ["breast", "breast", "esophagus", "esophagus"]


def test_load_gold_cases_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_gold_cases(tmp_path, "tcga") == []


def test_stratified_split_default_fraction(tmp_path: Path) -> None:
    # TCGA organ_n: 1=breast, 2=colorectal, 5=liver (per organ_code.yaml).
    organs = _yaml_organs("tcga", ["breast", "colorectal", "liver"])
    _seed_canonical_gold(tmp_path, "tcga", organs, cases_per_organ=10)
    cases = load_gold_cases(tmp_path, "tcga")
    sp = stratified_split(cases, test_fraction=0.34, seed=42)
    # 30 cases × 0.34 = 10.2 → rounds to 10 test, 20 train.
    assert len(sp["train"]) + len(sp["test"]) == 30
    assert len(sp["test"]) == 10
    # Every category appears in both folds.
    train_cats = {c["cancer_category"] for c in sp["train"]}
    test_cats = {c["cancer_category"] for c in sp["test"]}
    assert train_cats == {"breast", "colorectal", "liver"}
    assert test_cats == {"breast", "colorectal", "liver"}


def test_stratified_split_invalid_fraction(tmp_path: Path) -> None:
    cases = [{"id": "a", "cancer_category": "breast"}]
    with pytest.raises(ValueError, match="test_fraction"):
        stratified_split(cases, test_fraction=0.0, seed=0)
    with pytest.raises(ValueError, match="test_fraction"):
        stratified_split(cases, test_fraction=1.0, seed=0)


def test_stratified_split_singleton_category_goes_to_train(tmp_path: Path) -> None:
    """A category with only 1 case can't appear in both folds — it goes to train."""
    cases = [
        {"id": "cmuh1_1", "cancer_category": "breast"},
        {"id": "cmuh1_2", "cancer_category": "breast"},
        {"id": "cmuh1_3", "cancer_category": "breast"},
        {"id": "cmuh1_4", "cancer_category": "breast"},
        {"id": "cmuh1_5", "cancer_category": "rare_organ"},  # singleton
    ]
    sp = stratified_split(cases, test_fraction=0.34, seed=0)
    test_ids = {c["id"] for c in sp["test"]}
    assert "cmuh1_5" not in test_ids


def test_split_main_writes_canonical_path(tmp_path: Path) -> None:
    """End-to-end: registrar-split --folder <tmp> --datasets tcga writes splits.json
    in bare-string format (not full case dicts)."""
    # TCGA's full 5-organ set: 1=breast, 2=colorectal, 3=thyroid, 4=stomach, 5=liver.
    organs = _organs.organs_for("tcga")
    _seed_canonical_gold(tmp_path, "tcga", organs, cases_per_organ=5)
    rc = split_main([
        "--folder", str(tmp_path),
        "--datasets", "tcga",
        "--test-fraction", "0.34",
        "--seed", "0",
    ])
    assert rc == 0
    out = tmp_path / "data" / "tcga" / "splits.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["test_fraction"] == 0.34
    assert data["seed"] == 0
    assert data["total"] == 25
    assert len(data["train"]) + len(data["test"]) == 25
    # 25 × 0.34 = 8.5 → 8 or 9 test (depending on rounding).
    assert len(data["test"]) in (8, 9)
    # Bare-string serialization contract — see split.py:write_split.
    assert all(isinstance(x, str) for x in data["train"])
    assert all(isinstance(x, str) for x in data["test"])


def test_split_main_skips_missing_dataset(tmp_path: Path, capsys) -> None:
    """If --datasets includes a dataset with no gold, it's skipped (warned), not fatal."""
    # CMUH organ_n: 2=breast, 4=colorectal (per yaml).
    organs = _yaml_organs("cmuh", ["breast", "colorectal"])
    _seed_canonical_gold(tmp_path, "cmuh", organs, cases_per_organ=3)
    # tcga has no gold — should be skipped.
    rc = split_main([
        "--folder", str(tmp_path),
        "--datasets", "cmuh", "tcga",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[skip] tcga" in captured.err
    assert (tmp_path / "data" / "cmuh" / "splits.json").is_file()
    assert not (tmp_path / "data" / "tcga" / "splits.json").exists()


def test_split_main_no_datasets_have_gold_returns_error(tmp_path: Path) -> None:
    rc = split_main([
        "--folder", str(tmp_path),
        "--datasets", "cmuh",
    ])
    assert rc == 1


def test_split_seed_is_deterministic(tmp_path: Path) -> None:
    """Two runs with the same seed produce identical splits."""
    # CMUH organ_n: 2=breast, 4=colorectal, 5=esophagus (per yaml).
    organs = _yaml_organs("cmuh", ["breast", "colorectal", "esophagus"])
    _seed_canonical_gold(tmp_path, "cmuh", organs, cases_per_organ=10)
    cases = load_gold_cases(tmp_path, "cmuh")
    sp1 = stratified_split(cases, test_fraction=0.34, seed=12345)
    sp2 = stratified_split(cases, test_fraction=0.34, seed=12345)
    assert [c["id"] for c in sp1["train"]] == [c["id"] for c in sp2["train"]]
    assert [c["id"] for c in sp1["test"]] == [c["id"] for c in sp2["test"]]
