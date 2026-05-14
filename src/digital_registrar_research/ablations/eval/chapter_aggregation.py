"""Cross-cell 5-chapter cascade rollup for ablation runs.

The ablation aggregator (:mod:`run_ablations`) shells out to
``scripts.eval.cli cascade`` per (cell, model) pair, so each pair has
its own production-shape ``chapter1_eligibility/`` ... ``chapter5_lymph_nodes/``
tree at ``{results}/{cell}/{model}/_cascade_eval/``. This module
produces the **cross-cell** roll-up: one set of chapter folders directly
under ``results_root`` where each row corresponds to an ablation cell ×
model pair, and pairwise tests compare every non-baseline pair against
the configured ``baseline_method``.

Output layout under ``out_root`` (mirrors the canonical cascade tree
documented in :mod:`scripts.eval.cascade.run_cascade`):

    chapter1_eligibility/
        overall.csv
        confusion.csv
        pairwise.csv                    (cell vs baseline)
        multirun_consistency.csv        (only when n_runs >= 2)
    chapter2_organ_classification/
        overall.csv
        confusion_per_class.csv
        confusion_per_class_compare.csv
        pairwise.csv
        multirun_consistency.csv
    chapter3_field_extraction/
        per_field_overall.csv
        per_field_by_organ.csv
        per_organ_overall.csv
        nested_per_field_per_organ.csv
        nested_per_attribute_per_organ.csv
        biomarker_per_category.csv
        pairwise.csv
        multirun_consistency.csv
    chapter4_margins/
        overall.csv
        per_attribute.csv
        per_category.csv
        confusion_matrices.csv
        missingness.csv
        pairwise.csv
        multirun_consistency.csv
    chapter5_lymph_nodes/
        overall.csv
        per_attribute.csv
        per_category.csv
        per_station.csv
        confusion_matrices.csv
        missingness.csv
        pairwise.csv
        multirun_consistency.csv

The headlines come from the canonical chapter reducers in
:mod:`scripts.eval.cascade.reductions` and
:mod:`scripts.eval.cascade.nested_reductions` — same code that
production cascade runs use, applied to the master atomic / nested
tables after re-keying ``model`` to ``<cell>_<model_slug>`` so each
ablation cell × model becomes a distinct group.

The pairwise rows come from the cascade compare builders re-exported
in :mod:`scripts.eval.cascade.compare_components`. We invoke them
once per non-baseline target with a two-method dict
``{baseline: ..., target: ...}`` so the resulting ``run_a`` is always
the baseline and ``run_b`` is the target. The per-target results are
concatenated and Holm- / BH-adjusted family-wise.

Delta-sign conventions (inherited from the cascade compare CLI):
    * Chapters 1, 2 (stage pairwise): ``delta_acc_b_minus_a`` =
      ``mean(run_b) - mean(run_a)`` = target - baseline. **Negative**
      when the target is worse than the baseline.
    * Chapter 3 (Stage-C scalar pairwise): ``delta_acc`` =
      ``mean(run_a) - mean(run_b)`` = baseline - target. **Positive**
      when the target is worse than the baseline. (This sign is
      inherited from ``compare_runs._build_pairwise``; we preserve it
      for cascade conformance rather than silently flipping it.)
    * Chapters 4, 5 (nested pairwise): ``delta_f1_b_minus_a`` =
      ``mean_f1(run_b) - mean_f1(run_a)`` = target - baseline.
      **Negative** when the target is worse than the baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The cascade chapter reducers and compare builders live under
# ``scripts/`` (not ``src/``), so the repo's ``scripts/`` directory
# must be on ``sys.path`` for these imports to resolve. ``run_ablations``
# already inserts it before importing this module, but make the
# insertion idempotent here so direct callers (tests, ad-hoc scripts)
# work without needing to know about the path setup.
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval.cascade import nested_reductions, reductions  # noqa: E402
from eval.cascade import compare_components as cmp  # noqa: E402

try:
    from statsmodels.stats.multitest import multipletests  # noqa: E402
except Exception:  # pragma: no cover - statsmodels is a hard runtime dep
    multipletests = None  # type: ignore[assignment]


def _rekey_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``model`` set to ``<cell>_<model_slug>``.

    The chapter reducers and compare builders group by ``model`` to
    produce one row per system. For ablation rollups we want one row
    per ablation cell × model pair, so we rewrite ``model`` to the
    composite key on a copy. The on-disk parquet is unchanged.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "model_slug" not in out.columns:
        out["model_slug"] = out.get("model")
    out["cell"] = out["cell"].astype(str)
    out["model_slug"] = out["model_slug"].astype(str)
    out["model"] = out["cell"] + "_" + out["model_slug"]
    return out


def _split_per_method(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice a re-keyed atomic / nested frame into ``{method_label: df}``."""
    if df is None or df.empty or "model" not in df.columns:
        return {}
    return {
        str(m): sub for m, sub in df.groupby("model", dropna=False)
    }


def _adjust_family_pvalues(
    df: pd.DataFrame, p_col: str = "mcnemar_p",
) -> pd.DataFrame:
    """Append Holm and Benjamini-Hochberg adjusted p-values across rows.

    Used for chapters 1/2/4/5 where the pairwise output has one row per
    target (no within-row family). Applied to the entire DataFrame as a
    single family — i.e. correction across non-baseline targets.
    """
    if df.empty or p_col not in df.columns or multipletests is None:
        return df
    pvals = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(pvals)
    if not mask.any():
        df[f"{p_col}_holm"] = float("nan")
        df[f"{p_col}_bh"] = float("nan")
        return df
    holm_full = np.full(len(pvals), float("nan"))
    bh_full = np.full(len(pvals), float("nan"))
    _, holm_adj, _, _ = multipletests(pvals[mask], method="holm")
    _, bh_adj, _, _ = multipletests(pvals[mask], method="fdr_bh")
    holm_full[mask] = holm_adj
    bh_full[mask] = bh_adj
    df[f"{p_col}_holm"] = holm_full
    df[f"{p_col}_bh"] = bh_full
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> Path | None:
    """Write a DataFrame to ``path`` if non-empty, returning the path or None."""
    if df is None or df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _pairwise_target_vs_baseline(
    atomic_per_method: dict[str, pd.DataFrame],
    *,
    baseline_method: str,
    builder,
    builder_kwargs: dict,
    target_methods: list[str] | None = None,
) -> pd.DataFrame:
    """Run a compare-builder once per target with ``{baseline, target}`` dict.

    The pairwise builders in ``compare_components`` walk every i<j label
    pair and emit one row per pair (or per (pair, field)). Calling them
    with a two-method dict produces only the baseline-vs-target rows we
    want, with ``run_a == baseline`` (because dicts preserve insertion
    order and the baseline is inserted first).
    """
    if baseline_method not in atomic_per_method:
        return pd.DataFrame()
    if target_methods is None:
        target_methods = [m for m in atomic_per_method.keys()
                          if m != baseline_method]
    pieces: list[pd.DataFrame] = []
    base_df = atomic_per_method[baseline_method]
    for target in target_methods:
        if target not in atomic_per_method:
            continue
        # Insertion order matters: baseline first → run_a; target second
        # → run_b; delta_*_b_minus_a == target - baseline.
        pair = {baseline_method: base_df, target: atomic_per_method[target]}
        try:
            df = builder(pair, **builder_kwargs)
        except Exception as exc:
            print(f"[chapter_aggregation][warn] {builder.__name__} failed for "
                  f"target={target!r}: {exc!r}", file=sys.stderr)
            continue
        if df is not None and not df.empty:
            pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def emit_chapter_outputs(
    cascade_atomic: pd.DataFrame,
    cascade_nested: pd.DataFrame,
    *,
    out_root: Path,
    baseline_method: str,
    alpha: float = 0.05,
    n_boot: int = 2000,
    random_state: int = 0,
) -> dict[str, list[Path]]:
    """Emit cross-cell 5-chapter outputs under ``out_root``.

    Parameters
    ----------
    cascade_atomic, cascade_nested
        Master cascade atomic table and nested sidecar produced by the
        ablation aggregator. Must carry ``cell`` and ``model_slug``
        columns (cascade walker stamps these for ablation method runs).
        Pass an empty DataFrame for ``cascade_nested`` if no nested
        rows were produced — chapters 4 and 5 will be skipped.
    out_root
        Directory under which the ``chapter1_eligibility/`` ...
        ``chapter5_lymph_nodes/`` subdirectories will be written.
    baseline_method
        The ``f"{cell}_{model_slug}"`` key to compare every other cell
        × model pair against. Must exist in the data; otherwise the
        pairwise outputs are empty (headlines still emit).
    alpha, n_boot, random_state
        Standard CI / bootstrap controls. Defaults match
        ``docs/eval/ci_methods.md``.

    Returns
    -------
    dict
        Mapping ``chapter_dir_name -> list[Path]`` of all CSV files
        written. Empty list when a chapter produced no data.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    atomic_keyed = _rekey_model(cascade_atomic)
    nested_keyed = _rekey_model(cascade_nested)
    atomic_per_method = _split_per_method(atomic_keyed)

    written: dict[str, list[Path]] = {}

    # --- Chapter 1: eligibility ------------------------------------------
    ch1_dir = out_root / "chapter1_eligibility"
    ch1_paths: list[Path] = []
    ch1_paths.append(_write_csv(
        reductions.chapter1_overall(atomic_keyed, alpha=alpha),
        ch1_dir / "overall.csv"))
    ch1_paths.append(_write_csv(
        reductions.chapter1_confusion(atomic_keyed),
        ch1_dir / "confusion.csv"))
    if "run_id" in atomic_keyed.columns and atomic_keyed["run_id"].nunique() > 1:
        ch1_paths.append(_write_csv(
            reductions.chapter_multirun_reliability(atomic_keyed, stage="A"),
            ch1_dir / "multirun_consistency.csv"))
    ch1_pair = _pairwise_target_vs_baseline(
        atomic_per_method,
        baseline_method=baseline_method,
        builder=cmp.build_stage_pairwise,
        builder_kwargs=dict(stage="A", n_boot=n_boot, alpha=alpha,
                            random_state=random_state),
    )
    ch1_pair = _adjust_family_pvalues(ch1_pair)
    ch1_paths.append(_write_csv(ch1_pair, ch1_dir / "pairwise.csv"))
    written["chapter1_eligibility"] = [p for p in ch1_paths if p is not None]

    # --- Chapter 2: organ classification ---------------------------------
    ch2_dir = out_root / "chapter2_organ_classification"
    ch2_paths: list[Path] = []
    ch2_paths.append(_write_csv(
        reductions.chapter2_overall(atomic_keyed, alpha=alpha),
        ch2_dir / "overall.csv"))
    ch2_paths.append(_write_csv(
        reductions.chapter2_confusion_per_class(atomic_keyed),
        ch2_dir / "confusion_per_class.csv"))
    ch2_paths.append(_write_csv(
        cmp.build_stage_b_confusion_compare(atomic_per_method),
        ch2_dir / "confusion_per_class_compare.csv"))
    if "run_id" in atomic_keyed.columns and atomic_keyed["run_id"].nunique() > 1:
        ch2_paths.append(_write_csv(
            reductions.chapter_multirun_reliability(atomic_keyed, stage="B"),
            ch2_dir / "multirun_consistency.csv"))
    ch2_pair = _pairwise_target_vs_baseline(
        atomic_per_method,
        baseline_method=baseline_method,
        builder=cmp.build_stage_pairwise,
        builder_kwargs=dict(stage="B", n_boot=n_boot, alpha=alpha,
                            random_state=random_state),
    )
    ch2_pair = _adjust_family_pvalues(ch2_pair)
    ch2_paths.append(_write_csv(ch2_pair, ch2_dir / "pairwise.csv"))
    written["chapter2_organ_classification"] = [p for p in ch2_paths if p is not None]

    # --- Chapter 3: field extraction (Stage-C scalar + nested-summary) ---
    ch3_dir = out_root / "chapter3_field_extraction"
    ch3_paths: list[Path] = []
    ch3_paths.append(_write_csv(
        reductions.chapter3_per_field_overall(atomic_keyed, alpha=alpha),
        ch3_dir / "per_field_overall.csv"))
    ch3_paths.append(_write_csv(
        reductions.chapter3_per_field_by_organ(atomic_keyed, alpha=alpha),
        ch3_dir / "per_field_by_organ.csv"))
    ch3_paths.append(_write_csv(
        reductions.chapter3_per_organ_overall(atomic_keyed, alpha=alpha),
        ch3_dir / "per_organ_overall.csv"))
    if not nested_keyed.empty:
        ch3_paths.append(_write_csv(
            reductions.chapter3_nested_per_field_per_organ(
                nested_keyed, alpha=alpha),
            ch3_dir / "nested_per_field_per_organ.csv"))
        ch3_paths.append(_write_csv(
            reductions.chapter3_nested_per_attribute_per_organ(
                nested_keyed, alpha=alpha),
            ch3_dir / "nested_per_attribute_per_organ.csv"))
    ch3_paths.append(_write_csv(
        reductions.chapter3_biomarker_per_category(atomic_keyed, alpha=alpha),
        ch3_dir / "biomarker_per_category.csv"))
    # Stage-C pairwise: builder applies Holm / BH per (run_a, run_b)
    # family across fields. With our two-method calls each family is
    # the baseline-vs-target pair, so that correction is exactly what
    # we want — within-target across fields.
    ch3_pair = _pairwise_target_vs_baseline(
        atomic_per_method,
        baseline_method=baseline_method,
        builder=cmp.build_pairwise,
        builder_kwargs=dict(n_boot=n_boot, alpha=alpha,
                            random_state=random_state),
    )
    ch3_paths.append(_write_csv(ch3_pair, ch3_dir / "pairwise.csv"))
    if "run_id" in atomic_keyed.columns and atomic_keyed["run_id"].nunique() > 1:
        # chapter_multirun_reliability expects a single field for Stage C;
        # iterate over fields present and concat.
        c_rows = atomic_keyed[atomic_keyed.get("cascade_stage") == "C"]
        if "field_kind" in c_rows.columns:
            c_rows = c_rows[c_rows["field_kind"] != "nested_list"]
        c_pieces: list[pd.DataFrame] = []
        for field in sorted(c_rows["field"].dropna().unique()):
            try:
                piece = reductions.chapter_multirun_reliability(
                    atomic_keyed, stage="C", field=str(field))
            except Exception:
                continue
            if not piece.empty:
                c_pieces.append(piece)
        if c_pieces:
            ch3_paths.append(_write_csv(
                pd.concat(c_pieces, ignore_index=True),
                ch3_dir / "multirun_consistency.csv"))
    written["chapter3_field_extraction"] = [p for p in ch3_paths if p is not None]

    # --- Chapter 4: margins ----------------------------------------------
    ch4_dir = out_root / "chapter4_margins"
    ch4_paths: list[Path] = []
    if not nested_keyed.empty:
        ch4_paths.append(_write_csv(
            nested_reductions.chapter4_margins_overall(
                nested_keyed, atomic_keyed, alpha=alpha),
            ch4_dir / "overall.csv"))
        ch4_paths.append(_write_csv(
            nested_reductions.chapter4_margins_per_attribute(
                nested_keyed, alpha=alpha),
            ch4_dir / "per_attribute.csv"))
        ch4_paths.append(_write_csv(
            nested_reductions.chapter4_margins_per_category(
                nested_keyed, alpha=alpha),
            ch4_dir / "per_category.csv"))
        ch4_paths.append(_write_csv(
            nested_reductions.chapter4_margins_confusion(nested_keyed),
            ch4_dir / "confusion_matrices.csv"))
    ch4_paths.append(_write_csv(
        nested_reductions.chapter4_margins_missingness(atomic_keyed, alpha=alpha),
        ch4_dir / "missingness.csv"))
    if "run_id" in atomic_keyed.columns and atomic_keyed["run_id"].nunique() > 1:
        ch4_paths.append(_write_csv(
            nested_reductions.chapter4_margins_multirun(atomic_keyed),
            ch4_dir / "multirun_consistency.csv"))
    ch4_pair = _pairwise_target_vs_baseline(
        atomic_per_method,
        baseline_method=baseline_method,
        builder=cmp.build_nested_pairwise,
        builder_kwargs=dict(field="margins", n_boot=n_boot, alpha=alpha,
                            random_state=random_state),
    )
    # _build_nested_pairwise has no McNemar (continuous F1), but family-
    # wise CI inspection still benefits from a delta_ci_lo>0 / hi<0 sign
    # check downstream — no p-value adjustment needed.
    ch4_paths.append(_write_csv(ch4_pair, ch4_dir / "pairwise.csv"))
    written["chapter4_margins"] = [p for p in ch4_paths if p is not None]

    # --- Chapter 5: lymph nodes ------------------------------------------
    ch5_dir = out_root / "chapter5_lymph_nodes"
    ch5_paths: list[Path] = []
    if not nested_keyed.empty:
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_overall(
                nested_keyed, atomic_keyed, alpha=alpha),
            ch5_dir / "overall.csv"))
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_per_attribute(
                nested_keyed, alpha=alpha),
            ch5_dir / "per_attribute.csv"))
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_per_category(
                nested_keyed, alpha=alpha),
            ch5_dir / "per_category.csv"))
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_per_station(
                nested_keyed, alpha=alpha),
            ch5_dir / "per_station.csv"))
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_confusion(nested_keyed),
            ch5_dir / "confusion_matrices.csv"))
    ch5_paths.append(_write_csv(
        nested_reductions.chapter5_lymph_nodes_missingness(
            atomic_keyed, alpha=alpha),
        ch5_dir / "missingness.csv"))
    if "run_id" in atomic_keyed.columns and atomic_keyed["run_id"].nunique() > 1:
        ch5_paths.append(_write_csv(
            nested_reductions.chapter5_lymph_nodes_multirun(atomic_keyed),
            ch5_dir / "multirun_consistency.csv"))
    ch5_pair = _pairwise_target_vs_baseline(
        atomic_per_method,
        baseline_method=baseline_method,
        builder=cmp.build_nested_pairwise,
        builder_kwargs=dict(field="regional_lymph_node",
                            n_boot=n_boot, alpha=alpha,
                            random_state=random_state),
    )
    ch5_paths.append(_write_csv(ch5_pair, ch5_dir / "pairwise.csv"))
    written["chapter5_lymph_nodes"] = [p for p in ch5_paths if p is not None]

    return written


__all__ = ["emit_chapter_outputs"]
