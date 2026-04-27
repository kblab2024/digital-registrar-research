"""Centralised filesystem path resolution for the research package.

All hardcoded references to example-data folders / results / split files
funnel through this module so the repo can be moved or renamed without
breaking downstream code.
"""
from pathlib import Path

# src/digital_registrar_research/paths.py → parents[2] is the repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Example data (TCGA gold set) — shipped in-repo. The flat three-folder
# layout (`{prefix}_dataset_{date}/`, `{prefix}_result_{date}/`,
# `{prefix}_annotation_{date}/`) is the contract the doctor-facing annotation
# UI expects (see annotation.io.discover_folders).
DATA_ROOT: Path = REPO_ROOT / "data"
RAW_REPORTS: Path = DATA_ROOT / "tcga_dataset_20251117"
PREANNOTATIONS: Path = DATA_ROOT / "tcga_result_20251117"
GOLD_ANNOTATIONS: Path = DATA_ROOT / "tcga_annotation_20251117"

# Runtime artifacts (gitignored). The workspace tree mirrors the dummy/
# skeleton: workspace/results/{predictions,eval,ablations,benchmarks}/...
RESULTS_ROOT: Path = REPO_ROOT / "workspace" / "results"
BENCHMARKS_RESULTS: Path = RESULTS_ROOT / "benchmarks"
ABLATIONS_RESULTS: Path = RESULTS_ROOT / "ablations"

# Packaged train/test split
SPLITS_JSON: Path = Path(__file__).resolve().parent / "benchmarks" / "data" / "splits.json"

# Packaged JSON schemas (generated from canonical Pydantic models)
SCHEMAS_DATA: Path = Path(__file__).resolve().parent / "schemas" / "data"

__all__ = [
    "REPO_ROOT", "DATA_ROOT", "RAW_REPORTS", "PREANNOTATIONS", "GOLD_ANNOTATIONS",
    "RESULTS_ROOT", "BENCHMARKS_RESULTS", "ABLATIONS_RESULTS",
    "SPLITS_JSON", "SCHEMAS_DATA",
]
