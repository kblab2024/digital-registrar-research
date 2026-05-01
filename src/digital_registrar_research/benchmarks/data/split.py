"""Deterministic stratified train/test split for any dataset under the canonical
data layout.

Reads gold annotations from ``{folder}/data/{dataset}/annotations/gold/<organ>/<case>.json``
and writes a stratified train/test split to ``{folder}/data/{dataset}/splits.json``.

Defaults
--------
- ``--folder``: ``workspace`` (use ``--folder dummy`` for the synthetic fixture).
- ``--datasets``: ``cmuh tcga`` — pooled training is the contract, so both
  splits get refreshed in the same call by default.
- ``--test-fraction``: ``0.34`` (≈ the ratio used in the original 100/51 TCGA
  reference split, but no longer baked in as a count).
- ``--seed``: ``20251117`` — keep stable across reruns for reproducibility.
- ``--stratify-by``: ``cancer_category`` so every organ appears in both folds.

Usage
-----
    # Refresh both splits at the canonical default location.
    registrar-split

    # Override location / focus on a single dataset.
    registrar-split --folder dummy --datasets cmuh
    registrar-split --folder workspace --datasets tcga --test-fraction 0.30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3].parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from _config_loader import resolve_folder
except ImportError:  # pragma: no cover — only triggers when scripts/ is missing
    def resolve_folder(raw):  # type: ignore[misc]
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p.resolve()


DEFAULT_DATASETS = ("cmuh", "tcga")
DEFAULT_TEST_FRACTION = 0.34
DEFAULT_SEED = 20251117
DEFAULT_STRATIFY_BY = "cancer_category"


def load_gold_cases(folder: Path, dataset: str) -> list[dict]:
    """Walk ``{folder}/data/{dataset}/annotations/gold/<organ>/<case>.json``.

    Returns ``[{"id": ..., "annotation_path": ..., "report_path": ...,
    "cancer_category": ...}]`` with ``cancer_category`` defaulting to
    ``"null"`` when the annotation has no category set, so stratification
    has a bucket for non-cancer / unclassified cases.
    """
    gold_root = folder / "data" / dataset / "annotations" / "gold"
    reports_root = folder / "data" / dataset / "reports"
    if not gold_root.is_dir():
        return []
    cases: list[dict] = []
    for ann_path in sorted(gold_root.rglob("*.json")):
        organ_dir = ann_path.parent.name
        case_id = ann_path.stem
        report_path = reports_root / organ_dir / f"{case_id}.txt"
        try:
            with ann_path.open(encoding="utf-8") as f:
                ann = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[warn] skipping malformed annotation {ann_path}: {e}",
                  file=sys.stderr)
            continue
        cases.append({
            "id": case_id,
            "annotation_path": str(ann_path),
            "report_path": str(report_path),
            "cancer_category": ann.get("cancer_category") or "null",
        })
    return cases


def stratified_split(
    cases: list[dict], *, test_fraction: float, seed: int,
    stratify_by: str = DEFAULT_STRATIFY_BY,
) -> dict:
    """Return ``{"train": [...], "test": [...], "seed": ..., "test_fraction": ...}``.

    Stratifies by ``cases[i][stratify_by]``. Each stratum allocates
    ``round(len(stratum) * test_fraction)`` to test, with a per-stratum
    floor of 1 when the stratum has at least 2 cases (so every category
    appears in both folds). Strata with only 1 case go entirely to
    train. The total test count is then trimmed/padded from the largest
    stratum to hit the global target ``round(N * test_fraction)``.

    ``test_fraction == 0.0`` is a valid edge case: all cases go to
    train, test is empty. Useful for cross-corpus training where one
    dataset is fully held out (e.g. CMUH-train / TCGA-test for BERT).
    """
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in [0, 1); got {test_fraction!r}"
        )
    if test_fraction == 0.0:
        return {
            "train": list(cases),
            "test": [],
            "seed": seed,
            "test_fraction": test_fraction,
            "total": len(cases),
            "stratify_by": stratify_by,
        }
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c.get(stratify_by) or "null"].append(c)

    total = len(cases)
    target_test = round(total * test_fraction)
    train: list[dict] = []
    test: list[dict] = []
    for items in by_cat.values():
        rng.shuffle(items)
        if len(items) < 2:
            train.extend(items)
            continue
        n_test_cat = max(1, round(len(items) * test_fraction))
        test.extend(items[:n_test_cat])
        train.extend(items[n_test_cat:])

    # Adjust to hit the global target test count exactly.
    while len(test) > target_test and len(test) > 1:
        largest_cat = _largest_cat(test, stratify_by)
        for i, c in enumerate(test):
            if c.get(stratify_by) == largest_cat:
                train.append(test.pop(i))
                break
    while len(test) < target_test:
        largest_cat = _largest_cat(train, stratify_by)
        moved = False
        for i, c in enumerate(train):
            if c.get(stratify_by) == largest_cat:
                test.append(train.pop(i))
                moved = True
                break
        if not moved:
            break

    # Returns the in-memory split as full case dicts so the caller can
    # print per-stratum distributions. Serialization (`write_split`)
    # collapses the dicts to bare ID strings — see that function's
    # docstring for the format contract.
    return {
        "train": train,
        "test": test,
        "seed": seed,
        "test_fraction": test_fraction,
        "total": total,
        "stratify_by": stratify_by,
    }


def _largest_cat(items: list[dict], stratify_by: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for c in items:
        counts[c.get(stratify_by) or "null"] += 1
    return max(counts, key=lambda k: counts[k])


def write_split(folder: Path, dataset: str, split: dict) -> Path:
    """Serialize a split to ``{folder}/data/{dataset}/splits.json``.

    Format contract: ``train`` and ``test`` are lists of bare case-id
    strings (not full case dicts). This matches what
    ``gen_dummy_skeleton.py`` writes and what
    ``benchmarks/baselines/_data.py`` expects. Other fields
    (``seed``, ``test_fraction``, ``total``, ``stratify_by``) are
    preserved as serialized metadata.

    Embedding full case dicts (with paths and cancer_category) was a
    historical workaround that silently desynced when files moved and
    broke the BERT loader's path construction; bare strings are the
    canonical shape now.
    """
    payload = dict(split)
    for fold in ("train", "test"):
        items = payload.get(fold) or []
        payload[fold] = [c["id"] if isinstance(c, dict) else c for c in items]
    out_path = folder / "data" / dataset / "splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _print_distribution(label: str, fold: list[dict], stratify_by: str) -> None:
    counts: dict[str, int] = defaultdict(int)
    for c in fold:
        counts[c.get(stratify_by) or "null"] += 1
    pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {label:<5}: n={len(fold)}  {pretty}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--folder", default="workspace", type=resolve_folder,
        help="Experiment root: 'workspace' (default), 'dummy', or abs path. "
             "Reads {folder}/data/{dataset}/annotations/gold/, writes "
             "{folder}/data/{dataset}/splits.json.",
    )
    ap.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS),
        help=f"Datasets to refresh (default: {' '.join(DEFAULT_DATASETS)}). "
             f"Pass a single name to limit to one.",
    )
    ap.add_argument(
        "--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
        help=f"Fraction of cases to allocate to the test fold "
             f"(default: {DEFAULT_TEST_FRACTION}).",
    )
    ap.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    ap.add_argument(
        "--stratify-by", default=DEFAULT_STRATIFY_BY,
        help=f"Annotation field to stratify on (default: {DEFAULT_STRATIFY_BY}).",
    )
    args = ap.parse_args(argv)

    if not 0.0 <= args.test_fraction < 1.0:
        ap.error(f"--test-fraction must be in [0, 1); got {args.test_fraction!r}")

    n_written = 0
    for dataset in args.datasets:
        cases = load_gold_cases(args.folder, dataset)
        if not cases:
            print(
                f"[skip] {dataset}: no gold annotations under "
                f"{args.folder / 'data' / dataset / 'annotations' / 'gold'}.",
                file=sys.stderr,
            )
            continue
        split = stratified_split(
            cases, test_fraction=args.test_fraction, seed=args.seed,
            stratify_by=args.stratify_by,
        )
        out_path = write_split(args.folder, dataset, split)
        n_written += 1
        print(f"[{dataset}] wrote {out_path}")
        print(f"  total: {split['total']}  "
              f"target test: {round(split['total'] * args.test_fraction)} "
              f"(fraction={args.test_fraction})")
        print(f"  actual: train={len(split['train'])}  test={len(split['test'])}")
        _print_distribution("train", split["train"], args.stratify_by)
        _print_distribution("test", split["test"], args.stratify_by)

    if n_written == 0:
        print("error: no datasets had gold annotations to split.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
