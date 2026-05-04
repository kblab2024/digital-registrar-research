#!/usr/bin/env python3
"""Build the canonical statistics suite over a unified atomic table.

Reads the existing non-ablation ``correctness_table.parquet`` (modular
pipeline + baselines, produced by ``scripts/eval/non_nested/run_non_nested.py``)
and the ablation ``atomic.parquet`` (produced by
``digital_registrar_research.ablations.eval.run_ablations``), unions
them under one canonical schema, and runs
``ablations.eval.canonical_stats.run_canonical_stats`` to emit eight
canonical CSVs plus a markdown run report.

Both inputs are optional: pass either or both. When only one is
present, the suite still runs against that subset.

Output layout::

    {folder}/results/canonical/{dataset}/
        master_atomic.parquet
        headline.csv
        failure_modes.csv
        per_field.csv
        per_organ.csv
        seed_consistency.csv          (only when multi-run)
        modularity_advantage.csv
        low_performer_diagnostics.csv
        canonical_stats_report.md

Usage::

    python scripts/eval/canonical/make_paper_tables.py \\
        --folder workspace --dataset cmuh \\
        --modular-method dspy_modular_gpt_oss_20b \\
        [--nonablation-parquet PATH] \\
        [--ablation-parquet PATH] \\
        [--out-dir PATH]

The ``--modular-method`` value is the row in ``method`` column to use as
the comparator for delta / McNemar / OR computations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from digital_registrar_research.ablations.eval import canonical_stats  # noqa: E402


# ---------------------------------------------------------------------------
# Schema remap: non-ablation 5-status → canonical 8-status
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    "case_id", "organ", "method", "run", "field",
    "correct", "attempted", "gold_present",
    "case_status", "case_flags", "field_status", "field_error_detail",
]


def _remap_nonablation(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a non-ablation correctness table into the canonical schema.

    The non-ablation atomic carries Boolean flags ``parse_error,
    field_missing, attempted, correct, wrong``. We map each row to a
    single ``case_status`` and ``field_status`` consistent with the
    ablation atomic.
    """
    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    out = df.copy()
    if "run_id" in out.columns and "run" not in out.columns:
        out = out.rename(columns={"run_id": "run"})
    if "run" not in out.columns:
        out["run"] = ""
    if "method" not in out.columns:
        out["method"] = "unknown"
    if "gold_present" not in out.columns:
        out["gold_present"] = True

    def _row_status(r: pd.Series) -> tuple[str, str, str]:
        """(case_status, field_status, field_error_detail)"""
        if bool(r.get("parse_error")):
            return ("parse_error", "unscoreable_due_to_case_error", "")
        if not bool(r.get("gold_present", True)):
            return ("ok", "gold_missing", "")
        if bool(r.get("field_missing")):
            # No pred dict here — can't distinguish missing_key vs
            # null_value. Default to missing_key.
            return ("ok", "missing_key", "")
        if bool(r.get("correct")):
            return ("ok", "correct", "")
        if bool(r.get("wrong")):
            return ("ok", "wrong_value",
                    f"gold={r.get('gold_value')!r} "
                    f"pred={r.get('pred_value')!r}"[:120])
        # Fallback: row carries no informative flag.
        return ("ok", "wrong_value", "")

    statuses = out.apply(_row_status, axis=1)
    out["case_status"] = [s[0] for s in statuses]
    out["field_status"] = [s[1] for s in statuses]
    out["field_error_detail"] = [s[2] for s in statuses]
    out["case_flags"] = out["case_status"]

    # Keep only the canonical columns that exist.
    keep = [c for c in _CANONICAL_COLUMNS if c in out.columns]
    return out[keep].copy()


def _remap_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Pass through the ablation grid, ensuring canonical column names.

    The ablation aggregator already emits ``case_status / case_flags /
    field_status / field_error_detail``; we just normalise column
    names. ``method`` is built as ``f"{cell}_{model}"`` upstream.
    """
    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    out = df.copy()
    if "method" not in out.columns:
        if "cell" in out.columns and "model" in out.columns:
            out["method"] = (out["cell"].astype(str) + "_"
                             + out["model"].astype(str))
        else:
            out["method"] = "ablation"
    for col in ("case_status", "case_flags", "field_status",
                "field_error_detail"):
        if col not in out.columns:
            out[col] = ""
    if "gold_present" not in out.columns:
        out["gold_present"] = out.get("attempted", False)
    keep = [c for c in _CANONICAL_COLUMNS if c in out.columns]
    return out[keep].copy()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_folder(folder: str | Path) -> Path:
    """Resolve the folder shorthand (workspace / dummy / abs path)."""
    if folder is None:
        return None
    try:
        from _config_loader import resolve_folder  # noqa
        return resolve_folder(folder)
    except Exception:
        return Path(folder).resolve()


def _default_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    """Resolve default input / output paths from --folder/--dataset.

    Looked-for inputs:
        {folder}/results/non_nested/{method}_{model}/{dataset}/correctness_table.parquet
        {folder}/results/ablations/{dataset}/atomic.parquet

    Output: {folder}/results/canonical/{dataset}/
    """
    folder = _resolve_folder(args.folder) if args.folder else None
    out_dir = (args.out_dir if args.out_dir is not None else
               (folder / "results" / "canonical" / args.dataset
                if folder and args.dataset else None))
    if out_dir is None:
        raise SystemExit(
            "Must supply --out-dir or both --folder and --dataset.")
    abl = (args.ablation_parquet if args.ablation_parquet is not None else
           (folder / "results" / "ablations" / args.dataset
            / "atomic.parquet"
            if folder and args.dataset else None))
    nonabl = args.nonablation_parquet
    return {"out": Path(out_dir), "ablation": abl, "nonablation": nonabl}


# ---------------------------------------------------------------------------
# IAA republish
# ---------------------------------------------------------------------------

def _publish_iaa_summary(folder: Path | None, dataset: str | None,
                         out_dir: Path) -> Path | None:
    """Locate an existing per-field IAA CSV under
    ``{folder}/results/iaa/{dataset}/`` and copy it to
    ``out_dir/iaa_summary.csv`` with canonical column names.

    Returns the destination path, or None if no source IAA CSV was
    found.
    """
    if folder is None or dataset is None:
        return None
    iaa_root = folder / "results" / "iaa" / dataset
    if not iaa_root.is_dir():
        return None
    # The IAA module's canonical per-field output. Best-effort lookup.
    candidates = [
        iaa_root / "iaa_per_field.csv",
        iaa_root / "per_field.csv",
        iaa_root / "iaa_summary.csv",
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        return None
    df = pd.read_csv(src)
    dest = out_dir / "iaa_summary.csv"
    df.to_csv(dest, index=False)
    print(f"Wrote {dest} (republished from {src})")
    return dest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", default=None,
                    help="Experiment root shortcut (workspace / dummy / "
                         "absolute path). Used to locate default inputs.")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name under data/ (e.g. cmuh, tcga). "
                         "Used to locate default inputs.")
    ap.add_argument("--modular-method", required=True,
                    help="The 'method' value to use as the comparator "
                         "for Δ / McNemar / OR computations "
                         "(e.g. 'dspy_modular_gpt_oss_20b').")
    ap.add_argument("--nonablation-parquet", type=Path, default=None,
                    help="Override path to non-ablation correctness "
                         "table. Default: search under "
                         "{folder}/results/non_nested/.")
    ap.add_argument("--ablation-parquet", type=Path, default=None,
                    help="Override path to ablation atomic.parquet. "
                         "Default: {folder}/results/ablations/{dataset}/"
                         "atomic.parquet.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: "
                         "{folder}/results/canonical/{dataset}/.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _default_paths(args)
    out_dir: Path = paths["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    if paths["nonablation"] is not None and Path(paths["nonablation"]).is_file():
        nonabl = pd.read_parquet(paths["nonablation"])
        frames.append(_remap_nonablation(nonabl))
        print(f"Loaded non-ablation atomic: {paths['nonablation']} "
              f"({len(nonabl)} rows)")
    else:
        print(f"[info] non-ablation atomic not found at "
              f"{paths['nonablation']}; proceeding without it.")

    if paths["ablation"] is not None and Path(paths["ablation"]).is_file():
        abl = pd.read_parquet(paths["ablation"])
        frames.append(_remap_ablation(abl))
        print(f"Loaded ablation atomic: {paths['ablation']} "
              f"({len(abl)} rows)")
    else:
        print(f"[info] ablation atomic not found at "
              f"{paths['ablation']}; proceeding without it.")

    if not frames:
        print("[warn] No input atomics found. Writing empty scaffolding.",
              file=sys.stderr)
        master = pd.DataFrame(columns=_CANONICAL_COLUMNS)
    else:
        master = pd.concat(frames, ignore_index=True)
    master_path = out_dir / "master_atomic.parquet"
    try:
        master.to_parquet(master_path)
    except Exception as exc:
        print(f"[warn] could not write {master_path}: {exc!r}",
              file=sys.stderr)
    print(f"Master atomic: {len(master)} rows. Wrote {master_path}")

    folder = _resolve_folder(args.folder) if args.folder else None
    canonical_stats.run_canonical_stats(
        master, modular_method=args.modular_method, out_dir=out_dir,
        command_line=" ".join(sys.argv))
    _publish_iaa_summary(folder, args.dataset, out_dir)
    print(f"\nCanonical paper tables: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
