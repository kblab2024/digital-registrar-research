# Digital Registrar — Research

> Research stack for **The Digital Registrar**: the pipeline, the annotation UI, the comparison benchmarks, and the ablation study — one pyproject, one import root.

For the slim, production-facing extraction package (what non-academic users typically install), see the standalone [`digitalregistrar`](../digitalregistrar) repo. This package **vendors** that pipeline and adds the research apparatus around it.

## Install

```bash
git clone <this-repo> digital-registrar-research
cd digital-registrar-research
pip install -e .[all]
```

Extras are split by concern — install only what you need:

| Extra | What you get |
|---|---|
| `[annotation]` | Streamlit UI (`streamlit`) |
| `[benchmarks]` | GPT-4 / ClinicalBERT / rule-based baselines (`torch`, `transformers`, `openai`, `scikit-learn`, …) |
| `[ablations]` | Raw-JSON baseline (`jsonschema`) |
| `[dev]` | `pytest`, `ruff`, `mypy` |
| `[all]` | all of the above |

The core install (no extras) gives you the DSPy extraction pipeline plus the canonical Pydantic schemas.

## What's in the box

```
src/digital_registrar_research/
├── pipeline.py, experiment.py    # DSPy extraction pipeline (vendored from digitalregistrar)
├── models/                        # per-organ DSPy signatures + nested Pydantic types
├── util/                          # logging, prediction dump
├── schemas/                       # canonical Pydantic case-models → generated JSON schemas
│   ├── pydantic/                  # ← the source of truth
│   └── data/*.json                # ← generated artifacts used by downstream tools
├── annotation/                    # Streamlit UI for doctors to review pre-annotations
├── benchmarks/                    # baselines (GPT-4, ClinicalBERT, rules) + eval harness
└── ablations/                     # modular-vs-monolithic DSPy + raw-JSON grid
```

Data (TCGA gold set, ~7 MB) ships in `data/tcga_20251117/` so benchmarks and ablations run out-of-the-box.

## Console scripts

```bash
registrar-pipeline   --input data/tcga_20251117/dataset/tcga1     # batch extraction
registrar-annotate                                                # launches Streamlit UI
registrar-benchmark                                               # aggregates baseline comparisons
registrar-ablate                                                  # runs ablation grid
registrar-split                                                   # regenerates train/test split
registrar-schemas                                                 # regenerates JSON from Pydantic (use --check in CI)
```

## Documentation

- [docs/index.md](docs/index.md) — overview
- [docs/pipeline.md](docs/pipeline.md) — how the extraction pipeline works
- [docs/annotation.md](docs/annotation.md) — using the Streamlit app
- [docs/schemas.md](docs/schemas.md) — canonical Pydantic pattern; regenerating JSON; concordance test
- [docs/benchmarks.md](docs/benchmarks.md) — comparison methodology + literature review
- [docs/ablations.md](docs/ablations.md) — ablation design rationale
- [docs/data.md](docs/data.md) — TCGA gold-set provenance

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
