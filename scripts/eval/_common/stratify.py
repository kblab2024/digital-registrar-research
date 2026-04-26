"""Organ index ↔ name mapping and field-classification helpers.

The dataset folder layout uses numeric organ indices (``1``..``10``).
The eval pipeline uses string organ names. This module is the single
source of truth for the conversion.

The numeric mapping matches ``scripts/gen_dummy_skeleton.py`` and the
canonical schemas under ``src/digital_registrar_research/schemas/data``.
"""
from __future__ import annotations

from typing import Iterable

# Single source of truth — keep aligned with gen_dummy_skeleton.py.
ORGAN_INDEX_TO_NAME: dict[int, str] = {
    1: "breast",
    2: "cervix",
    3: "colorectal",
    4: "esophagus",
    5: "liver",
    6: "lung",
    7: "pancreas",
    8: "prostate",
    9: "stomach",
    10: "thyroid",
}

ORGAN_NAME_TO_INDEX: dict[str, int] = {v: k for k, v in ORGAN_INDEX_TO_NAME.items()}

ALL_ORGAN_INDICES: tuple[int, ...] = tuple(sorted(ORGAN_INDEX_TO_NAME.keys()))
ALL_ORGAN_NAMES: tuple[str, ...] = tuple(sorted(ORGAN_NAME_TO_INDEX.keys()))


def organ_name(idx: int) -> str:
    """Return the organ name for a numeric index.

    Raises ``KeyError`` on unknown indices — callers should validate
    arguments before calling.
    """
    return ORGAN_INDEX_TO_NAME[idx]


def organ_index(name: str) -> int:
    """Return the numeric index for an organ name.

    Raises ``KeyError`` on unknown names.
    """
    return ORGAN_NAME_TO_INDEX[name]


def parse_organ_arg(values: Iterable[str | int]) -> list[int]:
    """Convert mixed list of organ names / indices to indices.

    Accepts: ``["1", "2", "breast"]`` or ``[1, "lung"]``. Used by the
    ``--organs`` CLI argument so users can write either form.
    """
    out: list[int] = []
    for v in values:
        if isinstance(v, int):
            if v not in ORGAN_INDEX_TO_NAME:
                raise ValueError(f"unknown organ index: {v}")
            out.append(v)
            continue
        s = str(v).strip().lower()
        if s.isdigit():
            i = int(s)
            if i not in ORGAN_INDEX_TO_NAME:
                raise ValueError(f"unknown organ index: {i}")
            out.append(i)
        elif s in ORGAN_NAME_TO_INDEX:
            out.append(ORGAN_NAME_TO_INDEX[s])
        else:
            raise ValueError(f"unknown organ: {v!r}")
    return sorted(set(out))


def parse_case_id(case_id: str) -> tuple[str, int, int]:
    """Decode a case id of the form ``{dataset}{organ_idx}_{case_num}``.

    Examples:
        ``"cmuh1_42"`` → ``("cmuh", 1, 42)``
        ``"tcga6_50"`` → ``("tcga", 6, 50)``

    The dataset prefix is greedy — letters until the first digit. Returns
    ``(dataset, organ_idx, case_num)``. Raises ``ValueError`` on malformed
    case ids.
    """
    i = 0
    while i < len(case_id) and case_id[i].isalpha():
        i += 1
    if i == 0 or "_" not in case_id[i:]:
        raise ValueError(f"malformed case id: {case_id!r}")
    dataset = case_id[:i]
    rest = case_id[i:]
    organ_str, _, num_str = rest.partition("_")
    if not organ_str.isdigit() or not num_str.isdigit():
        raise ValueError(f"malformed case id: {case_id!r}")
    organ_idx = int(organ_str)
    if organ_idx not in ORGAN_INDEX_TO_NAME:
        raise ValueError(f"unknown organ index in case id: {case_id!r}")
    return dataset, organ_idx, int(num_str)


__all__ = [
    "ORGAN_INDEX_TO_NAME",
    "ORGAN_NAME_TO_INDEX",
    "ALL_ORGAN_INDICES",
    "ALL_ORGAN_NAMES",
    "organ_name",
    "organ_index",
    "parse_organ_arg",
    "parse_case_id",
]
