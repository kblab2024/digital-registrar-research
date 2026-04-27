"""Resolve the train/test split for a given (folder, dataset).

The split is data-tree-specific:

- Dummy / workspace: ``{folder}/data/{dataset}/splits.json`` is the
  authoritative source. Either a list of case-id strings, or a list of
  dicts with an ``"id"`` key.
- Packaged TCGA fallback: ``digital_registrar_research.paths.SPLITS_JSON``
  carries the production TCGA split when no per-folder splits.json
  exists (Mac-absolute paths, but the ``id`` field is what we read).

This module is the single source of truth for "which cases should the
benchmark be evaluated on" — both the eval wrappers and train_bert.py
use it to keep train/test separation honest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

SplitName = Literal["train", "test", "all"]


def _extract_ids(entries: Iterable) -> list[str]:
    """Accepts a list of strings or list of dicts; returns the case ids."""
    out: list[str] = []
    for item in entries:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "id" in item:
            out.append(item["id"])
    return out


def load_split(folder: Path, dataset: str) -> dict[str, list[str]]:
    """Return ``{"train": [...], "test": [...]}`` for ``(folder, dataset)``.

    Looks first under ``{folder}/data/{dataset}/splits.json``; if that
    doesn't exist, falls back to the packaged TCGA splits when
    ``dataset == 'tcga'``. Raises ``FileNotFoundError`` otherwise.
    """
    local = folder / "data" / dataset / "splits.json"
    if local.is_file():
        with local.open(encoding="utf-8") as f:
            data = json.load(f)
        return {
            "train": _extract_ids(data.get("train") or []),
            "test": _extract_ids(data.get("test") or []),
        }

    if dataset == "tcga":
        try:
            from digital_registrar_research.paths import SPLITS_JSON
        except ImportError:  # pragma: no cover
            raise FileNotFoundError(
                f"splits.json not found at {local} and packaged TCGA "
                f"splits import failed."
            )
        if SPLITS_JSON.is_file():
            with SPLITS_JSON.open(encoding="utf-8") as f:
                data = json.load(f)
            return {
                "train": _extract_ids(data.get("train") or []),
                "test": _extract_ids(data.get("test") or []),
            }

    raise FileNotFoundError(
        f"no splits.json found at {local}; for dataset={dataset!r} the "
        f"packaged fallback is only available for 'tcga'. Run "
        f"`python scripts/data/gen_dummy_skeleton.py` (dummy) or "
        f"`registrar-split` (workspace) to generate splits.json first."
    )


def resolve_case_allowlist(
    folder: Path, dataset: str, split: SplitName,
) -> list[str] | None:
    """Return the case-id allowlist for ``split``, or ``None`` for ``'all'``.

    ``None`` is the sentinel that means "do not pass --cases to non_nested".
    """
    if split == "all":
        return None
    parts = load_split(folder, dataset)
    if split not in parts:
        raise SystemExit(f"split={split!r} not present in splits.json")
    return parts[split]


def write_allowlist_file(case_ids: list[str], out_path: Path) -> Path:
    """Write a one-id-per-line file suitable for ``--cases @<path>``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for cid in case_ids:
            f.write(cid + "\n")
    return out_path


def per_organ_counts(case_ids: Iterable[str]) -> dict[int, int]:
    """Count cases per organ index by parsing the case-id prefix.

    Case IDs follow ``{dataset}{organ_n}_{idx}`` (e.g. ``cmuh3_17``).
    Cases that don't match are bucketed under organ 0.
    """
    import re
    out: dict[int, int] = {}
    pat = re.compile(r"^[a-z]+(\d+)_")
    for cid in case_ids:
        m = pat.match(cid)
        organ = int(m.group(1)) if m else 0
        out[organ] = out.get(organ, 0) + 1
    return out


__all__ = [
    "SplitName",
    "load_split",
    "resolve_case_allowlist",
    "write_allowlist_file",
    "per_organ_counts",
]
