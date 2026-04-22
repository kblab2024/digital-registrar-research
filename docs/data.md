# Example data — TCGA gold set

Ships at `data/`:

```
data/
├── tcga_dataset_20251117/      # raw pathology .txt reports
│   ├── extract_tcga_ids.py     # provenance / case-id extraction script
│   ├── tcga_case_ids.csv
│   └── tcga{1..5}/             # report folders
├── tcga_result_20251117/       # GPT-OSS pre-annotations (one *_output.json per case)
│   └── tcga{1..5}/
└── tcga_annotation_20251117/   # 151 doctor-validated final annotations
    └── 1/                       (the gold set)
```

| Folder | Files | Size | Role |
|---|--:|--:|---|
| `tcga_dataset_20251117/`   | ~600 | ~3.2 MB | source TCGA pathology reports |
| `tcga_result_20251117/`    | ~600 | ~3.4 MB | GPT-OSS:20b pre-annotations (run via `registrar-pipeline`) |
| `tcga_annotation_20251117/` | 151 | ~628 KB | the **gold** set used by benchmarks + ablations |

## Provenance

The TCGA (The Cancer Genome Atlas) cases were sampled to cover all ten currently-supported cancer organs. Pre-annotations were produced by the modular DSPy pipeline running gpt-oss:20b on a local Ollama instance. Doctors then reviewed and corrected each pre-annotation in the [annotation UI](annotation.md), saving the result to `tcga_annotation_20251117/`. Those 151 doctor-validated cases are the gold standard.

## Three-folder layout (load-bearing)

The folder naming `{prefix}_{kind}_{date}/` is the contract the annotation UI's `discover_folders` expects (see [annotation.md](annotation.md)). When adding new datasets, mirror this convention so the app picks them up automatically.

## Splits

`registrar-split` produces a deterministic 100/51 stratified split (stratified by `cancer_category`) at `src/digital_registrar_research/benchmarks/data/splits.json`. The split is checked into the repo so benchmark and ablation numbers are reproducible.
