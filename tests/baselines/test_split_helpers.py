"""Tests for the train/test split helpers used by train_bert + eval wrappers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baselines._split_helpers import (  # noqa: E402
    load_split, per_organ_counts, resolve_case_allowlist, write_allowlist_file,
)


def _write_splits(folder: Path, dataset: str, train: list[str], test: list[str]) -> None:
    p = folder / "data" / dataset / "splits.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"train": train, "test": test}), encoding="utf-8")


def test_load_split_string_ids(tmp_path: Path) -> None:
    _write_splits(tmp_path, "cmuh", ["cmuh1_1", "cmuh1_2"], ["cmuh1_3"])
    sp = load_split(tmp_path, "cmuh")
    assert sp == {"train": ["cmuh1_1", "cmuh1_2"], "test": ["cmuh1_3"]}


def test_load_split_dict_ids(tmp_path: Path) -> None:
    """splits.json may carry full case dicts (production TCGA layout) — extract id."""
    p = tmp_path / "data" / "tcga" / "splits.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "train": [{"id": "tcga1_1", "report_path": "/some/path"}],
        "test": [{"id": "tcga1_2"}],
    }), encoding="utf-8")
    sp = load_split(tmp_path, "tcga")
    assert sp == {"train": ["tcga1_1"], "test": ["tcga1_2"]}


def test_load_split_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="splits.json"):
        load_split(tmp_path, "cmuh")


def test_resolve_allowlist_test(tmp_path: Path) -> None:
    _write_splits(tmp_path, "cmuh", ["cmuh1_1"], ["cmuh1_2", "cmuh1_3"])
    ids = resolve_case_allowlist(tmp_path, "cmuh", "test")
    assert ids == ["cmuh1_2", "cmuh1_3"]


def test_resolve_allowlist_train(tmp_path: Path) -> None:
    _write_splits(tmp_path, "cmuh", ["cmuh1_1"], ["cmuh1_2"])
    ids = resolve_case_allowlist(tmp_path, "cmuh", "train")
    assert ids == ["cmuh1_1"]


def test_resolve_allowlist_all_returns_none(tmp_path: Path) -> None:
    """split='all' is the sentinel — no filter, no file read."""
    # No splits.json needed for split='all'.
    assert resolve_case_allowlist(tmp_path, "cmuh", "all") is None


def test_write_allowlist_file_format(tmp_path: Path) -> None:
    out = tmp_path / "ids.txt"
    write_allowlist_file(["cmuh1_1", "cmuh1_2", "tcga3_5"], out)
    content = out.read_text(encoding="utf-8").splitlines()
    assert content == ["cmuh1_1", "cmuh1_2", "tcga3_5"]


def test_per_organ_counts() -> None:
    counts = per_organ_counts(["cmuh1_1", "cmuh1_2", "cmuh3_5", "tcga2_1"])
    assert counts == {1: 2, 3: 1, 2: 1}


def test_per_organ_counts_unknown_format() -> None:
    counts = per_organ_counts(["weirdid", "cmuh1_5"])
    assert counts == {0: 1, 1: 1}


def test_train_test_disjoint(tmp_path: Path) -> None:
    """The helper doesn't enforce disjointness — that's the caller's job
    (train_bert.py) and the helpers should faithfully report what's in the file.
    Verify behavior on overlapping splits matches expectations: both lists
    contain the overlapping id."""
    _write_splits(tmp_path, "cmuh",
                   ["cmuh1_1", "cmuh1_2"],
                   ["cmuh1_2", "cmuh1_3"])
    sp = load_split(tmp_path, "cmuh")
    overlap = set(sp["train"]) & set(sp["test"])
    assert overlap == {"cmuh1_2"}
