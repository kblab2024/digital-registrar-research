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

Data and results use a flat, convention-driven layout covering
`data/{dataset}/reports|preannotation|annotations/` and
`results/predictions/{dataset}/{llm,clinicalbert,rule_based}/...` —
see [docs/data.md](docs/data.md) for the full tree, and
[scripts/gen_dummy_skeleton.py](scripts/gen_dummy_skeleton.py) to
generate a runnable dummy under `dummy/`.

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
- [docs/data.md](docs/data.md) — datasets, layout, and naming conventions
- [docs/experiment_protocol.md](docs/experiment_protocol.md) — the 2026-04 experiment cross-product, evaluation questions, and invariants
- [docs/branching_strategy.md](docs/branching_strategy.md) — the 12-branch working model (testing / refactor / experiment state)

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
