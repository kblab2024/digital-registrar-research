# Dummy skeleton — redesigned layout

This tree mirrors the canonical layout for the experiment:

- `data/{cmuh,tcga}/` — reports, pre-annotation, human annotations (4 modes + gold)
- `results/predictions/{dataset}/{llm,clinicalbert,rule_based}/...` — per-method outputs
- `results/evaluation/{dataset}/...` — (populated by eval scripts)
- `configs/{datasets,models,annotators}/` — machine-readable experiment metadata
- `models/clinicalbert/{v1_baseline,v2_finetuned}/` — checkpoint placeholders

## Naming

| Thing | Pattern |
|---|---|
| Case ID | `{dataset}{N}_{idx}` (`tcga1_7`, `cmuh2_3`) |
| Organ dir | numeric — `1/` = breast, `2/` = colorectal |
| Run dir | `run01`..`run10` |
| Annotator-mode dir | `{annotator}_{with,without}_preann` |
| Sidecar files | leading underscore (`_summary.json`, `_manifest.yaml`, `_log.jsonl`) |
| Case files | `{case_id}.json` — annotator/run/model is encoded by the folder |

## Regenerate

```
python scripts/gen_dummy_skeleton.py --out dummy --clean
```
