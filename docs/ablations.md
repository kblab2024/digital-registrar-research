# Ablation studies

The Digital Registrar pipeline makes joint design choices — DSPy as the
LM-calling framework, schema constraints realised through DSPy `Literal`
type hints, and a per-organ modular decomposition into 5–7 sub-signatures
chained through an intermediate `ReportJsonize` step. The ablation suite
attributes the headline accuracy back to those choices, one knob at a time.

## Layout

```
src/digital_registrar_research/ablations/
├── runners/
│   ├── reuse_baseline.py     # Cell A: copy modular outputs into ablations tree
│   ├── dspy_monolithic.py    # Cell B: one DSPy signature per organ
│   └── raw_json.py           # Cell C: raw OpenAI-compatible chat API + JSON mode
├── signatures/
│   └── monolithic.py         # Dynamically merges per-subsection signatures
└── eval/
    └── run_ablations.py      # aggregator → ablation_grid / summary / table CSVs

scripts/ablations/
├── _common.py                # CELL_MAP, smoke-root layout, aggregator round-trip
├── run_cell_a.py             # thin wrapper around reuse_baseline.run
├── run_cell_b.py             # thin wrapper around dspy_monolithic.run
├── run_cell_c.py             # thin wrapper around raw_json.run
├── run_cell_smoke.py         # per-cell smoke (1 model × 2 cases, fail-loud)
├── run_grid_smoke.py         # full grid-wide smoke (≤ 10 min)
└── run_grid.py               # YAML-driven full grid driver

configs/ablations/
├── smoke.yaml                # smoke defaults (n, cells, model)
└── grid_1.yaml               # Grid 1 minimum-viable lesion study
```

The runners expose a `run(args: argparse.Namespace)` entry point separate
from `main()`, so the wrappers under `scripts/ablations/` can construct
an argparse Namespace directly without sys.argv mutation. The CLI
surface (`python -m digital_registrar_research.ablations.runners.X`) is
preserved for ad-hoc invocations.

## Ablation axes

The full menu of ablation axes is enumerated in
[`workspace/reviewer-response-suggestions.md` §1.2](../workspace/reviewer-response-suggestions.md)
(gitignored). This document tracks **what's currently implemented** vs
what's wired up as a TODO in [run_grid.py](../scripts/ablations/run_grid.py).

### Axis 1 — Pipeline decomposition

| Level | Description | Status |
|---|---|---|
| A1 | Full modular (5–7 signatures per organ) | Cell A — implemented |
| A2 | Monolithic single signature per organ | Cell B — implemented |
| A3 | Monolithic, no `ReportJsonize` | Cell B `--skip-jsonize` |
| A4 | Monolithic, no `is_cancer` router | TODO |
| A5 | Per-section decomposition (header / dx / comments) | TODO |

### Axis 2 — Output structuring discipline

| Level | Description | Status |
|---|---|---|
| B1 | DSPy + Literal enums + Pydantic | Cells A, B — current default |
| B2 | DSPy with `str` outputs + post-hoc parser | TODO |
| B3 | Raw JSON-mode (`response_format=json_object`) | Cell C — implemented |
| B4 | Constrained decoding (outlines / lm-format-enforcer) | TODO |
| B5 | GBNF grammar (llama.cpp) | TODO |
| B6 | Free-text + regex post-extractor | TODO |

### Axis 3 — Prompting strategy

| Level | Description | Status |
|---|---|---|
| C1 | Zero-shot signature docstring | current default |
| C2–C3 | + 3 / 5 in-context examples | TODO |
| C4 | `dspy.ChainOfThought` wrapper | TODO |
| C5 | Compiled DSPy program (`BootstrapFewShotWithRandomSearch`) | TODO |
| C6 | Minimal raw prompt (degenerate baseline) | TODO |

### Axes 4–6 — Decoding, model identity, schema specificity

Decoding (temperature / num_ctx / self-consistency), model identity
(`gemma3:4b/27b`, `gpt-oss:20b`, `qwen3:30b`, `medgemma`, `llama3-med42`),
and schema-specificity (organ-specific vs union vs flat) are documented
in the suggestions doc. Decoding is config-driven via
[`configs/dspy_ollama_<model>.yaml`](../configs/) (no new code needed for
sweeps); model identity is handled by passing different `--model` keys;
schema-specificity needs new runners — TODO.

## The Grid 1 lesion study

The minimum-viable lesion study lives at
[`configs/ablations/grid_1.yaml`](../configs/ablations/grid_1.yaml).

Conditions (suggestions doc §1.3):

1. **Full pipeline** — modular DSPy + ReportJsonize + Literal enums
2. **Monolithic DSPy** — drops the modular per-section chain
3. **Monolithic DSPy without ReportJsonize** — also drops the
   intermediate JSON structuring step
4. **No DSPy** — raw OpenAI-compatible JSON-mode against local Ollama
5. *(TODO)* **No schema** — free-text generation + regex post-extractor

Single backbone (`gpt-oss:20b`), single seed for the first pass; for
multi-seed reproducibility, copy the YAML, change `slug` per seed, and
bump `decoding.seed` in the model config between invocations.

Wall-clock estimate: ~2–3 days on a single 48 GB GPU.

## Smoke runners — pre-flight before a multi-day sweep

Catching a typo or schema-binding regression eight hours into a sweep is
catastrophic. Every ablation kickoff goes through smoke first.

### Smoke contract

- **1 model**, **1 seed**, **2–3 cases** (default `--n 2`)
- Output dir prefixed with `_smoke_<YYYYMMDD-HHMM>/` so the regular
  aggregator's directory glob ignores it. Smoke never contaminates the
  real grid.
- **Fail loud** — any cell exception propagates; exit code ≠ 0 on
  partial failure.
- **Round-trip the aggregator** — smoke calls
  `digital_registrar_research.ablations.eval.run_ablations.main()` with
  `--results-root <smoke dir>` and asserts `ablation_summary.csv` is
  non-empty before returning success. A green smoke means **both** the
  cell and the eval reader work end-to-end.
- **Wall-time target**: ≤ 10 minutes for grid-wide smoke.

### Per-cell smoke

```bash
# Cell C × local gpt-oss:20b — fastest smoke (no DSPy bootstrap)
python scripts/ablations/run_cell_smoke.py --cell c --model gpt-oss:20b

# Cell B × local gpt-oss (uses models.common.model_list keys)
python scripts/ablations/run_cell_smoke.py --cell b --model gpt --n 2

# Cell A — needs a source dir of pre-computed predictions
python scripts/ablations/run_cell_smoke.py --cell a \
    --modular-source-dir E:/experiment/20260422/gpt-oss \
    --model gpt-oss
```

### Grid-wide smoke

```bash
# Smoke Cells B + C (default) on local gpt-oss
python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b

# Include Cell A too
python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b \
    --modular-source-dir E:/experiment/20260422/gpt-oss

# Subset cells (skip B if Ollama bootstrap is slow on this machine)
python scripts/ablations/run_grid_smoke.py --model gpt-oss:20b --cells c
```

### Recommended pre-push hook

Extend [`scripts/repo/install_git_hooks.sh`](../scripts/repo/install_git_hooks.sh)
to run the grid-wide smoke when files under
`src/digital_registrar_research/ablations/` are touched. Ten minutes of
pre-push insurance against pushing a broken cell that wastes a multi-day
sweep.

## Running a real grid

After a green smoke:

```bash
# Edit configs/ablations/grid_1.yaml first — set modular_gpt_oss_dir
# to point at the parent project's full-pipeline run output.

python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml
```

Each cell writes:
- `<results_root>/<cell_id>_<slug>/<case_id>.json` — per-case predictions
- `<results_root>/<cell_id>_<slug>/_ledger.json` — per-case timings
- `<results_root>/<cell_id>_<slug>/_run_meta.json` — git SHA, UTC, args

Top-level: `<results_root>/_grid_meta.json` — full grid manifest.

For ad-hoc per-cell runs without the grid driver, the wrappers all take
the same args as the underlying runners:

```bash
python scripts/ablations/run_cell_b.py --model gpt \
    --out workspace/results/ablations/dspy_monolithic_gpt-oss
python scripts/ablations/run_cell_c.py --model gpt-oss:20b \
    --out workspace/results/ablations/raw_json_gpt-oss
```

## Aggregator outputs

`run_ablations.main()` writes the following under `--results-root`
(default: `workspace/results/ablations/`):

| File | Contents |
|---|---|
| `ablation_grid.csv` | Long-form: one row per (cell, model, case, field) |
| `ablation_summary.csv` | Per-(cell, model, field): accuracy + coverage + nested F1 |
| `ablation_table.csv` | Pivot: rows=field, cols=`<cell>_<model>`, cells=accuracy |
| `cell_deltas.csv` | A→B and B→C per-field deltas per model |
| `efficiency.csv` | Mean / median latency, schema-error rate, parse-error rate |

Scoring is reused verbatim from
[`benchmarks.eval.{scope, metrics, completeness}`](eval/index.md) so
ablation cell numbers slot directly into the benchmark comparison
tables. Statistical analysis (paired-bootstrap CIs on per-field deltas,
McNemar at the case level, Fleiss κ across seeds) goes through the
shared toolkit in [`eval/ci_methods.md`](eval/ci_methods.md) and
[`eval/multirun.md`](eval/multirun.md).

## Reading the ablation result for the paper

The headline figure is the **lesion table**: full → −Decomposition →
−ReportJsonize → −DSPy → −Schema. Each column reports per-field macro
accuracy (FAIR_SCOPE) with paired-bootstrap 95 % CI vs. the full
pipeline. The completeness columns show *where* each lesion bleeds
quality:

- A → B (modularity off) typically loses on **breast-biomarker** and
  **regional_lymph_node** because monolithic context fills up first on
  multi-list organs.
- B → C (DSPy off) typically loses on **schema conformance** —
  Cell C's parse-error rate climbs on local LMs.
- B `--skip-jsonize` typically loses on **fields buried in narrative
  prose** (anything in "Comments" or "Final Diagnosis" sections), which
  the intermediate JSON structuring step normally surfaces.

The OFAT factorial in Grid 2 is reported as supplementary depth — see
the suggestions doc §1.3 for the recipe.

## Pre-registration discipline

Before kicking off a real grid:

1. Lock the test split (verify the hash of the packaged
   [`benchmarks/data/splits.json`](../src/digital_registrar_research/benchmarks/data/splits.json)).
2. Pre-register endpoints in
   [`configs/eval_endpoints.yaml`](../configs/eval_endpoints.yaml) —
   primary endpoint = per-field macro accuracy on FAIR_SCOPE; secondary
   = nested F1, completeness, latency.
3. Multiple-comparisons correction within each axis: Holm-Bonferroni
   ([`scripts/eval/_common/stats_extra.py`](../scripts/eval/_common/stats_extra.py)).
4. Reference the locked endpoint config (with git SHA) in the response
   letter / paper Methods section.

## Related documentation

- [ablations_design_rationale.md](ablations_design_rationale.md) — the
  why-these-cells reasoning
- [eval/index.md](eval/index.md) — the metric explanations every
  ablation number traces back to
- [eval/ci_methods.md](eval/ci_methods.md) — Wilson / BCa / paired
  bootstrap / GLMM CIs
- [eval/multirun.md](eval/multirun.md) — Fleiss κ and multi-run
  consistency
- [eval/multiple_comparisons.md](eval/multiple_comparisons.md) —
  Holm-Bonferroni vs BH-FDR
- [`workspace/reviewer-response-suggestions.md`](../workspace/reviewer-response-suggestions.md)
  *(gitignored)* — full menu of ablation axes and the rebuttal-letter
  ordering.
