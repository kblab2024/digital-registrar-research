#!/usr/bin/env python3
"""Run predictions with a trained ClinicalBERT multi-head classifier on Mac M3.

Mirrors `train_clinicalbert_cls_mac.py`: uses MPS when available, rebuilds
`splits.json` against the local annotation/dataset folders, then calls into
the packaged `clinicalbert_cls.predict` routine to emit one JSON prediction
per test case.

Usage (from the repo root, with `pip install -e '.[benchmarks]'`):

    python scripts/predict_clinicalbert_cls_mac.py \
        --annotations data/tcga_annotation_20251117 \
        --dataset     data/tcga_dataset_20251117 \
        --ckpt        ckpts/clinicalbert_cls.pt \
        --out         results/benchmarks/clinicalbert_cls
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
    ap = argparse.ArgumentParser(description="Predict with clinicalbert_cls on Mac (MPS).")
    ap.add_argument("--annotations", type=Path, default=paths.GOLD_ANNOTATIONS,
                    help="Gold annotation folder (default: %(default)s).")
    ap.add_argument("--dataset", type=Path, default=paths.RAW_REPORTS,
                    help="Raw reports folder (default: %(default)s).")
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_cls.pt",
                    help="Checkpoint produced by the training script.")
    ap.add_argument("--out", default="results/benchmarks/clinicalbert_cls",
                    help="Output folder for per-case prediction JSONs.")
    ap.add_argument("--skip-rebuild-splits", action="store_true",
                    help="Reuse the existing splits.json instead of rebuilding.")
    args = ap.parse_args()

    device = pick_device()
    print(f"[mac-predict] torch={torch.__version__}  device={device}")
    clinicalbert_cls.DEVICE = device

    if not args.skip_rebuild_splits:
        print(f"[mac-predict] rebuilding splits.json from "
              f"annotations={args.annotations} dataset={args.dataset}")
        rebuild_splits(args.annotations.resolve(), args.dataset.resolve())

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}. "
                         f"Run train_clinicalbert_cls_mac.py first.")

    predict_args = SimpleNamespace(ckpt=str(ckpt_path), out=args.out)
    clinicalbert_cls.predict(predict_args)
    print(f"[mac-predict] wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
