"""Audit case-id stem agreement across cmuh annotation folders + reports.

Walks the six trees that should share the same case-id stem set
(``gold``, ``kpc_with_preann``, ``nhc_with_preann``, ``kpc_without_preann``,
``nhc_without_preann``, and ``reports``) and prints a per-folder summary of
extras (stems present here but missing from the *full-set reference*) and
missing stems (informational). The full-set reference is the union of stems
across ``gold``, ``kpc_with_preann``, ``nhc_with_preann``, and ``reports`` —
the four folders that are meant to track the full dataset. The two
``*_without_preann`` folders are curated subsets, so their "missing" counts
are expected and reported but never prompt for deletion.

For every folder with non-zero extras the script asks ``[y/N]`` before
deleting; ``--dry-run`` blocks deletion outright and ``--yes`` skips the
prompt.

Usage:
    python scripts/data/audit_annotation_stems.py --dummy --dry-run
    python scripts/data/audit_annotation_stems.py --base workspace
    python scripts/data/audit_annotation_stems.py --base workspace --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval._common.paths import Paths  # noqa: E402

ANNOTATOR_FOLDERS: tuple[str, ...] = (
    "gold",
    "kpc_with_preann",
    "nhc_with_preann",
    "kpc_without_preann",
    "nhc_without_preann",
)
FULL_SET_ANNOTATORS: frozenset[str] = frozenset(
    {"gold", "kpc_with_preann", "nhc_with_preann"}
)
REPORTS_KEY = "reports"
FULL_SET_KEYS: frozenset[str] = FULL_SET_ANNOTATORS | {REPORTS_KEY}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--base", type=Path, default=None,
        help="Base dir containing data/<dataset>/... "
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
        "--dry-run", action="store_true",
        help="Print extras but do not delete anything.",
    )
    ap.add_argument(
        "--yes", action="store_true",
        help="Auto-confirm every delete prompt (non-interactive).",
    )
    ap.add_argument(
        "--max-show", type=int, default=20,
        help="Cap on stems printed inline per folder (default: 20).",
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


def collect_stems(folder: Path, suffix: str) -> dict[str, Path]:
    """Return ``{stem: file_path}`` for every ``<organ_idx>/*<suffix>`` file.

    Empty dict if the folder doesn't exist (the caller logs that).
    Duplicate stems across organ subdirs keep the first path encountered in
    sorted order — duplicates would themselves be a layout bug and are
    flagged in the warnings list.
    """
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


def prompt_delete(folder_label: str, extras: list[Path], yes: bool) -> bool:
    if yes:
        return True
    try:
        ans = input(
            f"Delete {len(extras)} extra file(s) from {folder_label}/? [y/N]: "
        ).strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def delete_files(paths: list[Path]) -> tuple[int, int]:
    """Delete each path; return ``(deleted_count, error_count)``."""
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

    folders: dict[str, Path] = {
        ann: paths.annotations_dir / ann for ann in ANNOTATOR_FOLDERS
    }
    folders[REPORTS_KEY] = paths.reports_dir

    stems_by_folder: dict[str, dict[str, Path]] = {}
    for key, folder in folders.items():
        suffix = ".txt" if key == REPORTS_KEY else ".json"
        if not folder.is_dir():
            print(f"warning: folder does not exist: {folder}", file=sys.stderr)
        stems_by_folder[key] = collect_stems(folder, suffix)

    reference: set[str] = set()
    for key in FULL_SET_KEYS:
        reference.update(stems_by_folder[key].keys())

    label_width = max(len(k) for k in folders)
    print()
    print(f"Base:      {args.base}")
    print(f"Dataset:   {args.dataset}")
    print(f"Dry-run:   {args.dry_run}")
    print()
    print("Folder counts:")
    for key in folders:
        n = len(stems_by_folder[key])
        print(f"  {key:<{label_width}} : {n:>6} stems")
    print()
    print(f"Reference (union of {sorted(FULL_SET_KEYS)}): {len(reference)} stems")
    print()

    print("Extras (stems NOT in reference) — deletion candidates:")
    extras_per_folder: dict[str, list[str]] = {}
    for key in folders:
        extras = sorted(set(stems_by_folder[key]) - reference)
        extras_per_folder[key] = extras
        print(f"  {key:<{label_width}} : {len(extras):>6}  "
              f"{format_stem_list(extras, args.max_show)}")
    print()

    print("Missing (in reference but not in folder) — informational:")
    for key in folders:
        missing = sorted(reference - set(stems_by_folder[key]))
        note = ""
        if key not in FULL_SET_KEYS:
            note = "  (curated subset; expected)"
        if key in FULL_SET_KEYS:
            shown = format_stem_list(missing, args.max_show)
        else:
            shown = ""
        print(f"  {key:<{label_width}} : {len(missing):>6}{note}  {shown}")
    print()

    unresolved_extras = False
    for key in folders:
        extras = extras_per_folder[key]
        if not extras:
            continue
        extra_paths = [stems_by_folder[key][s] for s in extras]
        print(f"Folder {key}: {len(extras)} extra file(s):")
        for p in extra_paths[:args.max_show]:
            print(f"  {p}")
        if len(extra_paths) > args.max_show:
            print(f"  (+{len(extra_paths) - args.max_show} more)")
        if args.dry_run:
            print(f"  [dry-run] would prompt to delete {len(extras)} file(s)")
            unresolved_extras = True
            continue
        if prompt_delete(key, extra_paths, args.yes):
            deleted, errors = delete_files(extra_paths)
            print(f"  deleted {deleted}/{len(extra_paths)} file(s)"
                  f"{f', {errors} error(s)' if errors else ''}")
            if errors:
                unresolved_extras = True
        else:
            print(f"  skipped: {len(extras)} extra file(s) remain in {key}/")
            unresolved_extras = True
        print()

    return 1 if unresolved_extras else 0


if __name__ == "__main__":
    sys.exit(main())
