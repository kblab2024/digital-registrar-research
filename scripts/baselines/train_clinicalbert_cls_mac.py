#!/usr/bin/env python3
"""Train the ClinicalBERT multi-head classifier baseline on a MacBook M3.

Picks up the Apple-silicon GPU via Metal Performance Shaders (MPS) when
available, rebuilds `splits.json` against the local annotation/dataset
folders (so POSIX paths resolve on macOS), and then hands off to the
packaged `clinicalbert_cls.train` routine.

Usage (from the repo root, with `pip install -e '.[benchmarks]'`):

    python scripts/train_clinicalbert_cls_mac.py \
        --annotations data/tcga_annotation_20251117 \
        --dataset     data/tcga_dataset_20251117 \
        --epochs 5 \
        --ckpt ckpts/clinicalbert_cls.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from digital_registrar_research import paths
from digital_registrar_research.benchmarks.baselines import clinicalbert_cls
from digital_registrar_research.benchmarks.data import split as splits_builder


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rebuild_splits(annotations: Path, dataset: Path) -> None:
    paths.GOLD_ANNOTATIONS = annotations
    paths.RAW_REPORTS = dataset
    splits_builder.GOLD_ROOT = annotations
    splits_builder.REPORT_ROOT = dataset
    splits_builder.main()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train clinicalbert_cls on Mac (MPS).")
    ap.add_argument("--annotations", type=Path, default=paths.GOLD_ANNOTATIONS,
                    help="Gold annotation folder (default: %(default)s).")
    ap.add_argument("--dataset", type=Path, default=paths.RAW_REPORTS,
                    help="Raw reports folder (default: %(default)s).")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_cls.pt")
    ap.add_argument("--included-only", action="store_true",
                    help="Restrict training to in-scope cancer_excision_report cases.")
    ap.add_argument("--skip-rebuild-splits", action="store_true",
                    help="Reuse the existing splits.json instead of rebuilding.")
    args = ap.parse_args()

    device = pick_device()
    print(f"[mac-train] torch={torch.__version__}  device={device}")
    clinicalbert_cls.DEVICE = device

    if not args.skip_rebuild_splits:
        print(f"[mac-train] rebuilding splits.json from "
              f"annotations={args.annotations} dataset={args.dataset}")
        rebuild_splits(args.annotations.resolve(), args.dataset.resolve())

    train_args = SimpleNamespace(
        epochs=args.epochs,
        ckpt=args.ckpt,
        included_only=args.included_only,
    )
    clinicalbert_cls.train(train_args)


if __name__ == "__main__":
    main()
