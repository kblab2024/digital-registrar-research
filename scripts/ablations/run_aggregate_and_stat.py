#!/usr/bin/env python3
"""Manual aggregator + stats for an ablation results tree.

The user runs this **after** rsync-ing per-machine results into a single
canonical ablation tree::

    {root}/results/ablations/{dataset}/{cell_id}/{model_slug}/runNN-<slug>/

Each contributing machine sets a different ``DRR_MACHINE_ID`` (or
``configs/local/runtime.yaml`` ``machine_id:`` field), so the ``-<slug>``
suffix on every ``runNN`` directory keeps the slot spaces disjoint at
rsync time. After the trees are merged, the per-cell ``_manifest.yaml``
files are stale (each was written by one machine and lists only that
machine's runs; whichever rsync wrote last clobbered the others).

This script:

1. Walks ``{cell}/{model}/runNN*/`` under the results root, reads each
   ``_summary.json`` + ``_run_meta.json``, and rewrites every per-cell
   ``_manifest.yaml`` from the union of discovered runs.

2. Discovers any top-level ``_grid_meta*.json`` files and writes a
   thin ``_grid_meta_combined.json`` that lists which grids
   contributed (per-machine provenance), without trying to merge the
   ``spec`` blocks.

3. Delegates to
   ``digital_registrar_research.ablations.eval.run_ablations.main()``
   to produce ``cascade_atomic.parquet``, the legacy
   ``ablation_*.csv`` outputs, the ``cell_deltas.csv``,
   ``efficiency.csv``, and the full stats pack.

4. Prints a per-(cell, model) summary at the end: how many runs total,
   broken down by machine slug.

Usage::

    python scripts/ablations/run_aggregate_and_stat.py \\
        --folder workspace --dataset cmuh

    python scripts/ablations/run_aggregate_and_stat.py \\
        --results-root /path/to/merged/cmuh
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config_loader import resolve_folder  # noqa: E402
from _run_id import parse_run_id  # noqa: E402

from digital_registrar_research.ablations.runners import _base as _ab_base  # noqa: E402

_RUN_DIR_RE = re.compile(r"^run\d+(?:-[a-z0-9][a-z0-9-]*)?$")


def _resolve_results_root(args: argparse.Namespace) -> Path:
    if args.results_root is not None:
        return Path(args.results_root).resolve()
    if args.experiment_root is None or not args.dataset:
        raise SystemExit(
            "either --results-root, or both --folder and --dataset, must be set.")
    root = (Path(args.experiment_root) / "results" / "ablations"
            / args.dataset)
    return root.resolve()


def _slug_of(run_id: str) -> str:
    """``run01-alpha2`` → ``alpha2``; ``run01`` → ``""``."""
    try:
        _, slug = parse_run_id(run_id)
    except ValueError:
        return ""
    return slug


class _ManifestPaths:
    """Duck-typed adapter for :func:`_base.update_cell_manifest`.

    The function only reads ``paths.cell_id``, ``paths.model_slug``,
    ``paths.dataset``, and ``paths.cell_dir`` — not the full
    :class:`_base.AblationPaths` shape. Building the real dataclass
    requires an ``experiment_root`` we don't have when scanning a
    merged tree by results-root alone, so we adapt.
    """

    def __init__(self, cell_id: str, model_slug: str, dataset: str,
                 cell_dir: Path):
        self.cell_id = cell_id
        self.model_slug = model_slug
        self.dataset = dataset
        self.cell_dir = cell_dir


def _rebuild_one_manifest(cell_dir: Path, model_dir: Path) -> int:
    """Rebuild ``_manifest.yaml`` under ``<cell>/<model>/`` from on-disk runs.

    Reuses :func:`_base.update_cell_manifest` so the entry shape stays
    canonical. Returns the number of runs found.
    """
    # Wipe the stale manifest so update_cell_manifest starts fresh.
    manifest_path = model_dir / "_manifest.yaml"
    if manifest_path.exists():
        manifest_path.unlink()

    paths = _ManifestPaths(
        cell_id=cell_dir.name,
        model_slug=model_dir.name,
        dataset=cell_dir.parent.name,
        cell_dir=model_dir,
    )

    runs_found = 0
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if not _RUN_DIR_RE.fullmatch(run_dir.name):
            continue
        summary_path = run_dir / "_summary.json"
        meta_path = run_dir / "_run_meta.json"
        if not summary_path.exists():
            continue
        try:
            with summary_path.open(encoding="utf-8") as f:
                summary_json = json.load(f)
        except Exception:
            continue
        meta_json: dict = {}
        if meta_path.exists():
            try:
                with meta_path.open(encoding="utf-8") as f:
                    meta_json = json.load(f)
            except Exception:
                pass

        per_run_extra: dict = {}
        for key in ("source_run", "source_dir", "skip_jsonize"):
            if key in meta_json:
                per_run_extra[key] = meta_json[key]

        class _Summary:
            run = run_dir.name
            seed = ((meta_json.get("decoding") or {}).get("seed")
                    or summary_json.get("seed"))
            n_pipeline_error = int(summary_json.get("n_pipeline_error", 0))
            n_cases = int(summary_json.get("n_cases", 0))

        _ab_base.update_cell_manifest(
            paths, run_dir.name, _Summary(),
            model_alias=meta_json.get("model_alias", ""),
            decoding=meta_json.get("decoding"),
            extra=per_run_extra,
        )
        runs_found += 1

    return runs_found


def _rebuild_all_manifests(results_root: Path) -> dict[tuple[str, str], int]:
    """Rebuild every per-cell ``_manifest.yaml`` under the results tree.

    Returns ``{(cell, model_slug): n_runs}`` for the summary print.
    """
    counts: dict[tuple[str, str], int] = {}
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    for cell_dir in sorted(results_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        for model_dir in sorted(cell_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            n = _rebuild_one_manifest(cell_dir, model_dir)
            counts[(cell_dir.name, model_dir.name)] = n
    return counts


def _per_machine_breakdown(results_root: Path
                           ) -> dict[tuple[str, str], dict[str, int]]:
    """Return ``{(cell, model): {slug: n_runs}}`` for the summary print."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for cell_dir in sorted(results_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        for model_dir in sorted(cell_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            buckets: dict[str, int] = {}
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                if not _RUN_DIR_RE.fullmatch(run_dir.name):
                    continue
                if not (run_dir / "_summary.json").exists():
                    continue
                slug = _slug_of(run_dir.name) or "(no-slug)"
                buckets[slug] = buckets.get(slug, 0) + 1
            if buckets:
                out[(cell_dir.name, model_dir.name)] = buckets
    return out


def _write_combined_grid_meta(results_root: Path) -> Path | None:
    """Collect every ``_grid_meta*.json`` and write a thin combined record.

    Doesn't merge the ``spec`` blocks — keeps per-machine provenance.
    """
    grids: list[dict] = []
    for path in sorted(results_root.glob("_grid_meta*.json")):
        if path.name == "_grid_meta_combined.json":
            continue
        try:
            with path.open(encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        grids.append({
            "file": path.name,
            "config_path": doc.get("config_path"),
            "git_sha": doc.get("git_sha"),
            "completed_utc": doc.get("completed_utc"),
            "n_manifests": len(doc.get("manifests") or []),
        })
    if not grids:
        return None
    out_path = results_root / "_grid_meta_combined.json"
    out_path.write_text(json.dumps({
        "results_root": str(results_root),
        "n_source_grids": len(grids),
        "source_grids": grids,
        "combined_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _delegate_to_aggregator(args: argparse.Namespace,
                            results_root: Path) -> int:
    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )

    eval_argv: list[str] = [
        "--results-root", str(results_root),
        "--device", args.device,
    ]
    if args.experiment_root is not None:
        eval_argv += ["--folder", str(args.experiment_root)]
    if args.dataset:
        eval_argv += ["--dataset", args.dataset]
    if args.cells:
        eval_argv += ["--cells", *args.cells]
    if args.models:
        eval_argv += ["--models", *args.models]
    if args.baseline:
        eval_argv += ["--baseline", args.baseline]
    if args.no_stats:
        eval_argv += ["--no-stats"]

    return eval_main(eval_argv) or 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", default=None,
                    type=resolve_folder,
                    help="experiment root containing data/ and results/. "
                         "shorthand 'dummy' / 'workspace' resolved against repo root.")
    ap.add_argument("--dataset", default=None, choices=("cmuh", "tcga"),
                    help="dataset name under data/ (cmuh or tcga).")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="override path to scan; takes precedence over "
                         "--folder/--dataset.")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="restrict to these cell ids (default: all).")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model slugs (default: all).")
    ap.add_argument("--baseline", default="dspy_modular_gpt_oss_20b",
                    help="<cell>_<model_slug> key to use as the Δ baseline.")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip the stats pack (cascade only).")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="cpu",
                    help="device for the bootstrap / GLMM layers.")
    ap.add_argument("--skip-manifest-rebuild", action="store_true",
                    help="don't rewrite per-cell _manifest.yaml (the "
                         "default is to rebuild from on-disk runs).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = _resolve_results_root(args)
    print(f"[run_aggregate_and_stat] results_root={results_root}")

    if not args.skip_manifest_rebuild:
        print("[run_aggregate_and_stat] rebuilding per-cell _manifest.yaml…")
        counts = _rebuild_all_manifests(results_root)
        for (cell, model), n in counts.items():
            print(f"  {cell}/{model}: {n} runs")
    else:
        print("[run_aggregate_and_stat] --skip-manifest-rebuild set; "
              "leaving _manifest.yaml files untouched.")

    combined = _write_combined_grid_meta(results_root)
    if combined is not None:
        print(f"[run_aggregate_and_stat] wrote {combined}")

    rc = _delegate_to_aggregator(args, results_root)
    if rc != 0:
        print(f"[run_aggregate_and_stat] aggregator exited rc={rc}",
              file=sys.stderr)
        return rc

    # Per-machine summary.
    breakdown = _per_machine_breakdown(results_root)
    if breakdown:
        print("\n[run_aggregate_and_stat] per-(cell, model) machine breakdown:")
        for (cell, model), buckets in breakdown.items():
            total = sum(buckets.values())
            byslug = ", ".join(f"{s}={n}" for s, n in sorted(buckets.items()))
            print(f"  {cell}/{model}: total={total} ({byslug})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
