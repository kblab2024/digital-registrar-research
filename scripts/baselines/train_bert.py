#!/usr/bin/env python3
"""ClinicalBERT — canonical-layout trainer for CLS and QA heads.

Trains the CLS and/or QA head on the **pooled** train split across both
configured datasets (cmuh + tcga). The pooled-training contract is
load-bearing: the annotated pool is too small to train each head on a
single dataset, so we pool train and evaluate per-dataset later.

Output: checkpoints under ``ckpts/`` (default), or wherever ``--ckpt-*``
points. The predict step (``run_bert.py``) loads from these paths.

Usage
-----
    # Train both heads on dummy data (smoke test):
    python scripts/baselines/train_bert.py \\
        --folder dummy --datasets cmuh tcga \\
        --heads cls qa \\
        [--epochs-cls 5] [--epochs-qa 3] [-v]

    # Train CLS only on real workspace data:
    python scripts/baselines/train_bert.py \\
        --folder workspace --datasets cmuh tcga \\
        --heads cls --epochs-cls 5

Mac note: MPS (Apple Silicon) is auto-detected; otherwise CUDA → CPU.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# MPS fallback — silenced on non-Apple platforms.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from _config_loader import resolve_folder  # noqa: E402

DATASETS = ("cmuh", "tcga")
DEFAULT_HEADS = ("cls", "qa")
DEFAULT_ORGANS = ("breast", "colorectal", "esophagus", "liver", "stomach")
LOGGER_NAME = "scripts.baselines.train_bert"


def _device_pick() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_cls(args: argparse.Namespace, logger: logging.Logger) -> None:
    from digital_registrar_research.benchmarks.baselines import clinicalbert_cls

    device = _device_pick()
    clinicalbert_cls.DEVICE = device
    logger.info("[cls] device=%s ckpt=%s", device, args.ckpt_cls)

    Path(args.ckpt_cls).parent.mkdir(parents=True, exist_ok=True)
    train_args = SimpleNamespace(
        ckpt=str(args.ckpt_cls),
        epochs=args.epochs_cls,
        organs=",".join(args.organs),
        datasets=",".join(args.datasets),
        data_root=str(args.experiment_root),
        included_only=args.included_only,
    )
    clinicalbert_cls.train(train_args)


def train_qa(args: argparse.Namespace, logger: logging.Logger) -> None:
    from digital_registrar_research.benchmarks.baselines import clinicalbert_qa

    device = _device_pick()
    clinicalbert_qa.DEVICE = device
    logger.info("[qa] device=%s ckpt=%s", device, args.ckpt_qa)

    Path(args.ckpt_qa).mkdir(parents=True, exist_ok=True)
    train_args = SimpleNamespace(
        ckpt=str(args.ckpt_qa),
        epochs=args.epochs_qa,
        organs=",".join(args.organs),
        datasets=",".join(args.datasets),
        data_root=str(args.experiment_root),
    )
    clinicalbert_qa.train(train_args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root (dummy/workspace/abs path).")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS,
                    help="Datasets to pool for training (default: both).")
    ap.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS),
                    choices=("cls", "qa"),
                    help="Heads to train (default: both).")
    ap.add_argument("--organs", nargs="*", default=list(DEFAULT_ORGANS),
                    help="Cancer-category names to keep.")
    ap.add_argument("--ckpt-cls", default="ckpts/clinicalbert_cls.pt",
                    help="Output path for the CLS checkpoint.")
    ap.add_argument("--ckpt-qa", default="ckpts/clinicalbert_qa",
                    help="Output dir for the QA checkpoint.")
    ap.add_argument("--epochs-cls", type=int, default=5)
    ap.add_argument("--epochs-qa", type=int, default=3)
    ap.add_argument("--included-only", action="store_true",
                    help="CLS only: drop cases where cancer_excision_report=False.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("datasets: %s", args.datasets)
    logger.info("heads: %s", args.heads)

    if "cls" in args.heads:
        train_cls(args, logger)
    if "qa" in args.heads:
        train_qa(args, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
