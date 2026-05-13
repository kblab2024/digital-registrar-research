"""Audit case-id stem agreement between gold annotations and LLM run outputs.

For each ``run*`` subdirectory under
``<base>/results/predictions/<dataset>/llm/<model>/`` the script compares the
run's stem set against gold annotation stems
(``<base>/data/<dataset>/annotations/gold/``):

  extras  = stems in run but NOT in gold  (deletion candidates)
  missing = stems in gold but NOT in run  (informational only)

For every run with non-zero extras the script asks ``[y/N]`` before
deleting; ``--dry-run`` blocks deletion outright and ``--yes`` skips the
prompt.

Usage:
    python scripts/data/audit_prediction_stems.py --dummy --dry-run
    python scripts/data/audit_prediction_stems.py --base workspace
    python scripts/data/audit_prediction_stems.py --base workspace \\
        --run run01 --run run02 --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval._common.paths import Paths  # noqa: E402

DEFAULT_MODEL = "gpt_oss_20b"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--base", type=Path, default=None,
        help="Base dir containing data/<dataset>/... and results/predictions/... "
             "(default: ./workspace, or ./dummy if --dummy)",
    )
    ap.add_argument(
        "--dummy", action="store_true",
        help="Shortcut for --base ./dummy.",
    )
    ap.add_argument(
        "--dataset", default="cmuh",
        help="Dataset name under <base>/data/ (default: cmuh)",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model dir under predictions/<dataset>/llm/ (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--run", dest="runs", action="append", default=None,
        help="Restrict to a specific run id (repeatable). "
             "Default: every run* dir under the model.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print extras but do not delete anything.",
    )
    ap.add_argument(
        "--yes", action="store_true",
        help="Auto-confirm every delete prompt (non-interactive).",
    )
    ap.add_argument(
        "--max-show", type=int, default=20,
        help="Cap on stems printed inline per run (default: 20).",
    )
    args = ap.parse_args()

    if args.dummy:
        if args.base is not None and args.base != REPO_ROOT / "dummy":
            ap.error("--dummy and --base are mutually exclusive")
        args.base = REPO_ROOT / "dummy"
    elif args.base is None:
        args.base = REPO_ROOT / "workspace"

    if args.max_show < 0:
        ap.error("--max-show must be >= 0")
    return args


def collect_stems(folder: Path, suffix: str = ".json") -> dict[str, Path]:
    """Return ``{stem: file_path}`` for every ``<organ_idx>/*<suffix>`` file."""
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for organ_dir in sorted(folder.iterdir()):
        if not organ_dir.is_dir():
            continue
        for p in sorted(organ_dir.glob(f"*{suffix}")):
            if p.stem in out:
                print(
                    f"warning: duplicate stem {p.stem!r} in {folder} "
                    f"(keeping {out[p.stem]}, ignoring {p})",
                    file=sys.stderr,
                )
                continue
            out[p.stem] = p
    return out


def format_stem_list(stems: list[str], max_show: int) -> str:
    if not stems:
        return "[]"
    head = stems[:max_show]
    tail = len(stems) - len(head)
    body = ", ".join(head)
    return f"[{body}]" + (f" (+{tail} more)" if tail > 0 else "")


def prompt_delete(run_id: str, extras: list[Path], yes: bool) -> bool:
    if yes:
        return True
    try:
        ans = input(
            f"Delete {len(extras)} extra prediction file(s) from {run_id}? [y/N]: "
        ).strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def delete_files(paths: list[Path]) -> tuple[int, int]:
    deleted = errors = 0
    for p in paths:
        try:
            p.unlink()
            deleted += 1
        except OSError as e:
            print(f"  error: failed to delete {p}: {e}", file=sys.stderr)
            errors += 1
    return deleted, errors


def main() -> int:
    args = parse_args()
    paths = Paths(root=args.base.resolve(), dataset=args.dataset)
    if not paths.data_dir.is_dir():
        sys.exit(f"error: data directory missing: {paths.data_dir}")

    gold_dir = paths.annotations_dir / "gold"
    gold_stems = collect_stems(gold_dir, ".json")
    if not gold_stems:
        sys.exit(f"error: no gold annotations found under {gold_dir}")
    gold_set = set(gold_stems)

    runs = paths.discover_runs(args.model, method="llm")
    if not runs:
        sys.exit(
            f"error: no run* dirs found under "
            f"{paths.predictions_dir / 'llm' / args.model}"
        )
    if args.runs:
        wanted = set(args.runs)
        runs = [(rid, rdir) for rid, rdir in runs if rid in wanted]
        if not runs:
            sys.exit(f"error: none of --run {sorted(wanted)} match discovered runs")
        missing_filter = wanted - {rid for rid, _ in runs}
        if missing_filter:
            print(
                f"warning: requested runs not found: {sorted(missing_filter)}",
                file=sys.stderr,
            )

    print()
    print(f"Base:      {args.base}")
    print(f"Dataset:   {args.dataset}")
    print(f"Model:     {args.model}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print(f"Reference: gold ({len(gold_set)} stems)")
    print()

    unresolved_extras = False
    for run_id, run_dir in runs:
        run_stems = collect_stems(run_dir, ".json")
        run_set = set(run_stems)
        extras = sorted(run_set - gold_set)
        missing = sorted(gold_set - run_set)
        print(f"{run_id}: {len(run_set)} stems")
        print(f"  extras   (in run, not in gold): {len(extras):>5}  "
              f"{format_stem_list(extras, args.max_show)}")
        print(f"  missing  (in gold, not in run): {len(missing):>5}  "
              f"{format_stem_list(missing, args.max_show)}")
        if not extras:
            print()
            continue
        extra_paths = [run_stems[s] for s in extras]
        print(f"  extra file paths:")
        for p in extra_paths[:args.max_show]:
            print(f"    {p}")
        if len(extra_paths) > args.max_show:
            print(f"    (+{len(extra_paths) - args.max_show} more)")
        if args.dry_run:
            print(f"  [dry-run] would prompt to delete {len(extras)} file(s)")
            unresolved_extras = True
            print()
            continue
        if prompt_delete(run_id, extra_paths, args.yes):
            deleted, errors = delete_files(extra_paths)
            print(f"  deleted {deleted}/{len(extra_paths)} file(s)"
                  f"{f', {errors} error(s)' if errors else ''}")
            if errors:
                unresolved_extras = True
        else:
            print(f"  skipped: {len(extras)} extra file(s) remain in {run_id}")
            unresolved_extras = True
        print()

    return 1 if unresolved_extras else 0


if __name__ == "__main__":
    sys.exit(main())
