# Ablation study

The Digital Registrar pipeline ships with per-organ subsection signatures (5–7 DSPy signatures per organ), originally chosen because of context-window limits when chaining JSON-serialised DSPy responses through small LMs. The ablation study answers two questions:

1. **Does modular help?** Modular (5–7 signatures per organ) vs monolithic (one big signature per organ).
2. **Does DSPy itself help?** DSPy vs raw JSON-mode LLM calls against the per-organ Pydantic schemas.

## The 2×2 grid

|                       | gpt-oss:20b (local) | gpt-4-turbo (cloud) |
|---|---|---|
| **Cell A — DSPy modular** (baseline)   | reuse existing pipeline runs |
| **Cell B — DSPy monolithic**           | one merged DSPy signature per organ; auto-built from parent's signatures |
| **Cell C — Raw JSON**                  | no DSPy; OpenAI-compatible chat API + `response_format=json_object` against the canonical schemas |

## Layout

```
src/digital_registrar_research/ablations/
├── runners/
│   ├── reuse_baseline.py       # Cell A: copy modular outputs into workspace/results/ablations/
│   ├── dspy_monolithic.py      # Cell B: one DSPy signature per organ
│   └── raw_json.py             # Cell C: raw OpenAI-compatible chat API
├── signatures/
│   └── monolithic.py           # Dynamically merges per-subsection signatures into one
└── eval/
    └── run_ablations.py        # aggregator → workspace/results/ablations/tables/
```

## Why monolithic.py is dynamic

`signatures/monolithic.py` introspects `models.modellist.organmodels[organ]` and the per-subsection signature classes, building a single `dspy.Signature` subclass that contains the union of every OutputField. **It does not hand-fork signatures**, so when the parent pipeline gains a field the monolithic baseline picks it up automatically.

## Why Cell C uses the canonical Pydantic schemas

Cell C bypasses DSPy entirely. To keep the comparison apples-to-apples (same target field set, same Literal vocabularies), the raw-JSON runner reads its target schema via `digital_registrar_research.schemas.load_json_schema(organ)` — the **same** JSON the annotation UI consumes, generated from the **same** Pydantic case-models the DSPy pipeline targets. Agreement between Cell A/B/C reflects only model and framework behaviour, not schema mismatch.

## Running

```bash
registrar-ablate                                    # runs the eval aggregator
# Per-cell runners output to workspace/results/ablations/<cell>_<model>/
python -m digital_registrar_research.ablations.runners.reuse_baseline \
    --modular-gpt-oss-dir <existing modular gpt-oss runs>
python -m digital_registrar_research.ablations.runners.dspy_monolithic --model gpt --out workspace/results/ablations/dspy_monolithic_gpt-oss
OPENAI_API_KEY=... python -m digital_registrar_research.ablations.runners.raw_json --model gpt-4-turbo --out workspace/results/ablations/raw_json_gpt4
```

## Output tables

`run_ablations.py` writes:

| File | Contents |
|---|---|
| `ablation_grid.csv`     | Long-form: one row per (cell, model, case, field) |
| `ablation_summary.csv`  | Per-(cell, model, field): accuracy + coverage |
| `ablation_table.csv`    | Pivot: rows=field, cols=`<cell>_<model>`, cells=accuracy |
| `cell_deltas.csv`       | A-vs-B and B-vs-C per-field deltas per model |
| `efficiency.csv`        | Mean latency + schema-error rate per cell/model |

Scoring is reused verbatim from `benchmarks.eval.{scope, metrics}` so ablation cell numbers slot directly into the benchmark comparison tables.
