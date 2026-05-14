"""Public re-exports of the cross-method cascade comparison builders.

These are the data-shaping helpers used by the cascade ``compare`` CLI
subcommand ([`compare_runs`](compare_runs.py)). They take a
``dict[label, cascade_atomic_df]`` and return a tidy DataFrame ready
for ``write_csv`` — pure pandas in / pandas out, no I/O.

We re-export them under public names (``build_*`` rather than
``_build_*``) so the ablation aggregator's chapter rollup
(``digital_registrar_research.ablations.eval.chapter_aggregation``) can
import them without reaching into private symbols.

The originals stay in ``compare_runs.py`` to avoid a churny lift; this
module is a stable public surface that callers should import from. If
``compare_runs`` is ever restructured, only this re-export needs to
move with the bodies.
"""
from __future__ import annotations

from .compare_runs import (
    _bootstrap_mean_ci as bootstrap_mean_ci,
    _build_funnel_compare as build_funnel_compare,
    _build_headline as build_headline,
    _build_nested_comparison as build_nested_comparison,
    _build_nested_pairwise as build_nested_pairwise,
    _build_others_compare as build_others_compare,
    _build_pairwise as build_pairwise,
    _build_per_field_wide as build_per_field_wide,
    _build_per_organ_wide as build_per_organ_wide,
    _build_stage_b_confusion_compare as build_stage_b_confusion_compare,
    _build_stage_comparison as build_stage_comparison,
    _build_stage_pairwise as build_stage_pairwise,
    _build_verdict as build_verdict,
    _safe_acc as safe_acc,
    _stage_c_scalar as stage_c_scalar,
)

__all__ = [
    "build_funnel_compare",
    "build_headline",
    "build_nested_comparison",
    "build_nested_pairwise",
    "build_others_compare",
    "build_pairwise",
    "build_per_field_wide",
    "build_per_organ_wide",
    "build_stage_b_confusion_compare",
    "build_stage_comparison",
    "build_stage_pairwise",
    "build_verdict",
    "bootstrap_mean_ci",
    "safe_acc",
    "stage_c_scalar",
]
