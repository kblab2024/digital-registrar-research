"""Regression tests for the BERT baseline data loader (_data.py).

Pinpoint coverage for the bizarre `_organ_n` / `_resolve_dummy` /
`_load_one_dataset` interaction:

- Two splits.json formats exist in the wild (bare strings from
  gen_dummy_skeleton, full case dicts from registrar-split).
- Loader must normalize entries before path construction.
- _organ_n must operate on strings, not dicts.

Without these tests the silent "BERT discovers no cases" failure
(broken paths -> cancer_category=None -> filter drops everything)
will recur.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from digital_registrar_research.benchmarks.baselines._data import (
    _load_one_dataset,
    _normalize_entry,
    _organ_n,
    _resolve_dummy,
)


def _seed_dummy_case(
    root: Path, dataset: str, organ_n: str, case_id: str,
    cancer_category: str,
) -> None:
    ann_dir = root / "data" / dataset / "annotations" / "gold" / organ_n
    rep_dir = root / "data" / dataset / "reports" / organ_n
    ann_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{case_id}.json").write_text(
        json.dumps({"cancer_category": cancer_category}), encoding="utf-8",
    )
    (rep_dir / f"{case_id}.txt").write_text("dummy report", encoding="utf-8")


def _write_splits(root: Path, dataset: str, train, test) -> None:
    p = root / "data" / dataset / "splits.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"train": train, "test": test}), encoding="utf-8")


# --- _organ_n ----------------------------------------------------------------


def test_organ_n_parses_string() -> None:
    assert _organ_n("cmuh1_42", "cmuh") == "1"
    assert _organ_n("cmuh10_5", "cmuh") == "10"
    assert _organ_n("tcga3_99", "tcga") == "3"


def test_organ_n_dataset_prefix_required() -> None:
    with pytest.raises(ValueError, match="could not parse organ_n"):
        _organ_n("tcga1_42", "cmuh")


def test_organ_n_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        _organ_n("not_a_case_id", "cmuh")


def test_organ_n_no_longer_indexes_dict() -> None:
    """The hand-fix `case_id["id"]` was a workaround masking the splits-format
    bug; reverting it means strings are required and dicts must be normalized
    upstream. This pins that behavior so the workaround can't sneak back."""
    with pytest.raises(TypeError):
        _organ_n({"id": "cmuh1_42"}, "cmuh")  # type: ignore[arg-type]


# --- _normalize_entry --------------------------------------------------------


def test_normalize_entry_passes_strings_through() -> None:
    assert _normalize_entry("cmuh1_1") == "cmuh1_1"


def test_normalize_entry_extracts_id_from_dict() -> None:
    assert _normalize_entry({"id": "cmuh1_1", "report_path": "/x"}) == "cmuh1_1"


def test_normalize_entry_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="unrecognized splits.json entry"):
        _normalize_entry(42)
    with pytest.raises(ValueError, match="unrecognized splits.json entry"):
        _normalize_entry({"no_id": "x"})


# --- _resolve_dummy ----------------------------------------------------------


def test_resolve_dummy_with_string_id_builds_paths(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "1", "cmuh1_1", "breast")
    case = _resolve_dummy(tmp_path, "cmuh", "cmuh1_1")
    assert case["id"] == "cmuh1_1"
    assert case["dataset"] == "cmuh"
    assert case["organ_n"] == "1"
    assert case["cancer_category"] == "breast"
    assert Path(case["annotation_path"]).exists()
    assert Path(case["report_path"]).exists()


# --- _load_one_dataset (the integration that was silently broken) ------------


def test_load_one_dataset_bare_strings(tmp_path: Path) -> None:
    """Dummy-format splits.json (bare strings) must load and resolve correctly."""
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "cmuh", "9", "cmuh9_1", "stomach")
    _write_splits(tmp_path, "cmuh", ["cmuh2_1"], ["cmuh9_1"])

    train = _load_one_dataset("cmuh", "train", tmp_path)
    test = _load_one_dataset("cmuh", "test", tmp_path)

    assert [c["id"] for c in train] == ["cmuh2_1"]
    assert [c["id"] for c in test] == ["cmuh9_1"]
    assert train[0]["cancer_category"] == "breast"
    assert test[0]["cancer_category"] == "stomach"


def test_load_one_dataset_legacy_dict_entries(tmp_path: Path, capsys) -> None:
    """Legacy registrar-split splits.json (full case dicts) must also load.

    This is the bug that caused 'BERT discovers no cases': dict entries
    were silently corrupting path construction. Normalizer must rescue.
    """
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _write_splits(
        tmp_path, "cmuh",
        train=[{"id": "cmuh2_1", "annotation_path": "/legacy/path",
                "report_path": "/legacy/path", "cancer_category": "breast"}],
        test=[],
    )

    cases = _load_one_dataset("cmuh", "train", tmp_path)

    assert [c["id"] for c in cases] == ["cmuh2_1"]
    assert cases[0]["cancer_category"] == "breast"
    assert Path(cases[0]["annotation_path"]).exists()
    captured = capsys.readouterr()
    assert "legacy dict entries" in captured.out


def test_load_one_dataset_all_concatenates(tmp_path: Path) -> None:
    _seed_dummy_case(tmp_path, "cmuh", "2", "cmuh2_1", "breast")
    _seed_dummy_case(tmp_path, "cmuh", "9", "cmuh9_1", "stomach")
    _write_splits(tmp_path, "cmuh", ["cmuh2_1"], ["cmuh9_1"])

    cases = _load_one_dataset("cmuh", "all", tmp_path)
    assert sorted(c["id"] for c in cases) == ["cmuh2_1", "cmuh9_1"]
