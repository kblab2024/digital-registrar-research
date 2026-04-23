"""Generate a dummy skeleton of the redesigned dataset + results layout.

Produces a minimal-but-schema-valid tree under `--out` (default: ./dummy)
so the new directory convention can be exercised end-to-end before real
data or the TCGA migration land.

Layout produced mirrors the final design:

    {out}/data/{cmuh,tcga}/
        reports/{organ}/{case_id}.txt
        preannotation/gpt_oss_20b/{organ}/{case_id}.json
        annotations/{nhc,kpc}_{with,without}_preann/{organ}/{case_id}.json
        annotations/gold/{organ}/{case_id}.json
        splits.json
        dataset_manifest.yaml
    {out}/results/predictions/{dataset}/
        llm/{model}/run{01,02}/{organ}/{case_id}.json
        llm/{model}/run{01,02}/_summary.json  _log.jsonl
        llm/{model}/_manifest.yaml
        clinicalbert/{v1_baseline,v2_finetuned}/{organ}/{case_id}.json (+ _summary.json)
        rule_based/{organ}/{case_id}.json  _summary.json
    {out}/configs/
        datasets/{cmuh,tcga}.yaml
        models/{...}.yaml
        annotators/annotators.yaml
    {out}/models/clinicalbert/{v1_baseline,v2_finetuned}/{config.yaml,checkpoint.pt.placeholder}
    {out}/README.md

Values are canonical (schema-valid) but trivial; the three cases per
organ vary slightly so scoring and agreement aren't degenerate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ---- Constants --------------------------------------------------------------

DATASETS = ["cmuh", "tcga"]
ORGANS = {"1": "breast", "2": "colorectal"}           # numeric dir -> organ
CASES_PER_ORGAN = 3
LLM_MODELS = ["gpt_oss_20b", "gemma4_30b", "qwen3_30b", "gemma4_e2b"]
LLM_RUNS_IN_DUMMY = 2                                 # 2 of the 10 real runs
BERT_MODELS = ["v1_baseline", "v2_finetuned"]
ANNOTATORS = ["nhc", "kpc"]
MODES = ["with_preann", "without_preann"]
PREANN_MODEL = "gpt_oss_20b"

# ---- Schema-valid payload factories -----------------------------------------

def breast_case(tumor_size: int, grade: int, lvi: bool, margin_involved: bool, er: bool) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "wide_excision",
            "cancer_quadrant": "upper_outer_quadrant",
            "cancer_clock": None,
            "cancer_laterality": "right",
            "histology": "invasive_carcinoma_no_special_type",
            "tumor_size": tumor_size,
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "dcis_present": False,
            "dcis_size": None,
            "dcis_comedo_necrosis": None,
            "dcis_grade": None,
            "nuclear_grade": grade,
            "tubule_formation": 2,
            "mitotic_rate": 2,
            "total_score": grade + 4,
            "grade": grade,
            "tnm_descriptor": None,
            "pt_category": "t1c" if tumor_size <= 20 else "t2",
            "pn_category": "n0",
            "pm_category": "mx",
            "pathologic_stage_group": None,
            "anatomic_stage_group": None,
            "ajcc_version": None,
            "margins": [
                {"margin_category": "superficial", "margin_involved": margin_involved,
                 "distance": None if margin_involved else 5, "description": None},
                {"margin_category": "base", "margin_involved": False,
                 "distance": 8, "description": "negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_side": "right", "lymph_node_category": "sentinel",
                 "involved": 0, "examined": 2, "station_name": "sentinel #1"},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
            "biomarkers": [
                {"biomarker_category": "er", "expression": er, "percentage": None,
                 "score": None, "biomarker_name": "estrogen receptor"},
                {"biomarker_category": "pr", "expression": er, "percentage": None,
                 "score": None, "biomarker_name": "progesterone receptor"},
                {"biomarker_category": "her2", "expression": None, "percentage": None,
                 "score": 0, "biomarker_name": "human epidermal growth factor receptor 2"},
                {"biomarker_category": "ki67", "expression": None, "percentage": None,
                 "score": None, "biomarker_name": "ki67 proliferation index"},
            ],
        },
    }


def colorectal_case(grade: int, pt: str, lvi: bool, involved_nodes: int) -> dict:
    return {
        "cancer_excision_report": True,
        "cancer_category": "colorectal",
        "cancer_category_others_description": None,
        "cancer_data": {
            "procedure": "segmental_colectomy",
            "surgical_technique": None,
            "cancer_primary_site": "sigmoid_colon",
            "histology": "adenocarcinoma",
            "grade": grade,
            "tumor_invasion": "pericolorectal_tissue",
            "lymphovascular_invasion": lvi,
            "perineural_invasion": None,
            "extracellular_mucin": None,
            "signet_ring": None,
            "tumor_budding": None,
            "type_of_polyp": None,
            "distant_metastasis": None,
            "treatment_effect": None,
            "tnm_descriptor": None,
            "pt_category": pt,
            "pn_category": "n0" if involved_nodes == 0 else "n1",
            "pm_category": "mx",
            "stage_group": "iia" if involved_nodes == 0 else "iii",
            "ajcc_version": None,
            "margins": [
                {"margin_category": "proximal", "margin_involved": False,
                 "distance": None, "description": "Negative"},
                {"margin_category": "distal", "margin_involved": False,
                 "distance": None, "description": "Negative"},
                {"margin_category": "radial_or_circumferencial", "margin_involved": False,
                 "distance": None, "description": "Negative"},
            ],
            "regional_lymph_node": [
                {"lymph_node_category": "mesenteric", "involved": involved_nodes,
                 "examined": 22, "station_name": None},
            ],
            "extranodal_extension": None,
            "maximal_ln_size": None,
            "biomarkers": None,
        },
    }


def gold_for(dataset: str, organ_n: str, idx: int) -> dict:
    """Deterministic schema-valid gold annotation for (dataset, organ, idx)."""
    if ORGANS[organ_n] == "breast":
        return breast_case(
            tumor_size=15 + idx * 3,
            grade=((idx - 1) % 3) + 1,
            lvi=(idx == 3),
            margin_involved=(idx == 2),
            er=(idx != 3),
        )
    else:  # colorectal
        return colorectal_case(
            grade=((idx - 1) % 3) + 1,
            pt={1: "t2", 2: "t3", 3: "t4a"}[idx],
            lvi=(idx == 2),
            involved_nodes={1: 0, 2: 0, 3: 2}[idx],
        )


def noisify(gold: dict, error_rate: float, rng: random.Random) -> dict:
    """Produce a prediction by flipping a small fraction of leaf values."""
    import copy
    out = copy.deepcopy(gold)
    data = out["cancer_data"]

    # Optionally flip a top-level boolean / categorical
    if rng.random() < error_rate and "lymphovascular_invasion" in data:
        v = data["lymphovascular_invasion"]
        if isinstance(v, bool):
            data["lymphovascular_invasion"] = not v

    # Optionally perturb a numeric by ±1
    for num_field in ("tumor_size", "grade", "nuclear_grade"):
        if num_field in data and isinstance(data[num_field], int) and rng.random() < error_rate:
            data[num_field] = max(1, data[num_field] + rng.choice([-1, 1]))

    # Optionally flip ER expression in biomarkers
    if data.get("biomarkers") and rng.random() < error_rate:
        for b in data["biomarkers"]:
            if b["biomarker_category"] == "er" and isinstance(b["expression"], bool):
                b["expression"] = not b["expression"]
                break

    return out


def report_text(dataset: str, case_id: str, organ: str, gold: dict) -> str:
    """Short synthetic pathology-report-ish text."""
    cd = gold["cancer_data"]
    bits = [
        f"SYNTHETIC PATHOLOGY REPORT (dummy) — dataset={dataset}, id={case_id}, organ={organ}",
        f"FINAL DIAGNOSIS:",
        f"  Procedure: {cd.get('procedure')}",
        f"  Histology: {cd.get('histology')}",
    ]
    if organ == "breast":
        bits += [
            f"  Tumor size: {cd.get('tumor_size')} mm, laterality={cd.get('cancer_laterality')}",
            f"  Grade {cd.get('grade')}, LVI={cd.get('lymphovascular_invasion')}",
            f"  Margins: {'involved' if any(m['margin_involved'] for m in cd['margins']) else 'negative'}",
            f"  ER: {cd['biomarkers'][0]['expression']}, PR: {cd['biomarkers'][1]['expression']}",
            f"  pTNM: {cd.get('pt_category')} {cd.get('pn_category')} {cd.get('pm_category')}",
        ]
    else:
        ln = cd["regional_lymph_node"][0]
        bits += [
            f"  Grade {cd.get('grade')}, LVI={cd.get('lymphovascular_invasion')}",
            f"  pTNM: {cd.get('pt_category')} {cd.get('pn_category')} {cd.get('pm_category')}, stage {cd.get('stage_group')}",
            f"  Lymph nodes: {ln['involved']}/{ln['examined']} involved ({ln['lymph_node_category']})",
        ]
    return "\n".join(bits) + "\n"


# ---- Writers ----------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    """Minimal YAML writer (avoid pyyaml dep). Only handles dicts/lists/scalars."""
    def _dump(obj, indent=0):
        pad = "  " * indent
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.extend(_dump(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {_scalar(v)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    sub = _dump(item, indent + 1)
                    if sub:
                        sub[0] = pad + "- " + sub[0].lstrip()
                        lines.extend(sub)
                else:
                    lines.append(f"{pad}- {_scalar(item)}")
        return lines

    def _scalar(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        if any(c in s for c in ":#\n") or s in ("true", "false", "null", ""):
            return json.dumps(s)
        return s

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_dump(payload)) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---- Top-level builders -----------------------------------------------------

def case_ids(dataset: str, organ_n: str) -> list[str]:
    return [f"{dataset}{organ_n}_{i}" for i in range(1, CASES_PER_ORGAN + 1)]


def build_dataset(root: Path, dataset: str, rng: random.Random) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ds_root = root / "data" / dataset
    all_ids: list[str] = []

    for organ_n, organ in ORGANS.items():
        for idx in range(1, CASES_PER_ORGAN + 1):
            case_id = f"{dataset}{organ_n}_{idx}"
            all_ids.append(case_id)
            gold = gold_for(dataset, organ_n, idx)

            # Raw report
            write_text(ds_root / "reports" / organ_n / f"{case_id}.txt",
                       report_text(dataset, case_id, organ, gold))

            # Gold
            write_json(ds_root / "annotations" / "gold" / organ_n / f"{case_id}.json", gold)

            # Pre-annotation (gpt-oss:20b, slightly noisy)
            preann = noisify(gold, error_rate=0.15, rng=rng)
            write_json(ds_root / "preannotation" / PREANN_MODEL / organ_n / f"{case_id}.json",
                       preann)

            # Human annotations: 4 modes, each slightly different
            for annotator in ANNOTATORS:
                for mode in MODES:
                    # Annotators working WITH preann drift slightly less from gold
                    err = 0.05 if mode == "with_preann" else 0.10
                    ann = noisify(gold, error_rate=err, rng=rng)
                    ann["_meta"] = {
                        "filename": f"{case_id}.json",
                        "annotator": annotator,
                        "mode": mode,
                        "annotated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    write_json(
                        ds_root / "annotations" / f"{annotator}_{mode}" / organ_n / f"{case_id}.json",
                        ann,
                    )

    # splits.json — simple train/test
    split_idx = len(all_ids) // 2
    write_json(ds_root / "splits.json", {
        "seed": 20251117,
        "train": all_ids[:split_idx],
        "test": all_ids[split_idx:],
    })

    # dataset_manifest.yaml
    write_yaml(ds_root / "dataset_manifest.yaml", {
        "dataset": dataset,
        "created_at": now,
        "n_cases": len(all_ids),
        "organs": [{"n": n, "name": ORGANS[n]} for n in ORGANS],
        "cases_per_organ": CASES_PER_ORGAN,
        "source": "synthetic_dummy_skeleton",
        "schema_version": "fair_plus_nested_plus_biomarkers_v1",
    })


def build_llm_predictions(root: Path, dataset: str, rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "llm"
    for model in LLM_MODELS:
        # Manifest for the model
        run_entries = []
        for run_i in range(1, LLM_RUNS_IN_DUMMY + 1):
            run_name = f"run{run_i:02d}"
            run_dir = preds_root / model / run_name

            # Model-specific error rate so methods differ in quality
            base_err = {
                "gpt_oss_20b": 0.05, "gemma4_30b": 0.08,
                "qwen3_30b": 0.09, "gemma4_e2b": 0.15,
            }[model]
            seed = 42 + run_i - 1

            log_rows: list[dict] = []
            total_tokens = 0
            parse_errors = 0

            for organ_n in ORGANS:
                for idx in range(1, CASES_PER_ORGAN + 1):
                    case_id = f"{dataset}{organ_n}_{idx}"
                    gold = gold_for(dataset, organ_n, idx)
                    pred = noisify(gold, error_rate=base_err, rng=random.Random(seed * 1000 + idx))
                    write_json(run_dir / organ_n / f"{case_id}.json", pred)

                    tok = rng.randint(400, 1200)
                    total_tokens += tok
                    log_rows.append({
                        "case_id": case_id,
                        "organ": organ_n,
                        "run": run_name,
                        "seed": seed,
                        "tokens": tok,
                        "parse_success": True,
                        "latency_s": round(rng.uniform(0.4, 2.5), 3),
                    })

            # _summary.json
            write_json(run_dir / "_summary.json", {
                "run": run_name,
                "model": model,
                "seed": seed,
                "dataset": dataset,
                "n_cases": sum(CASES_PER_ORGAN for _ in ORGANS),
                "total_tokens": total_tokens,
                "parse_errors": parse_errors,
                "parse_error_rate": 0.0,
                "wall_time_s": round(rng.uniform(30, 120), 1),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            # _log.jsonl
            (run_dir / "_log.jsonl").write_text(
                "\n".join(json.dumps(r) for r in log_rows) + "\n", encoding="utf-8"
            )

            run_entries.append({
                "run": run_name,
                "seed": seed,
                "valid": True,
                "parse_error_rate": 0.0,
            })

        # _manifest.yaml
        write_yaml(preds_root / model / "_manifest.yaml", {
            "experiment_id": f"multirun_{model}_{dataset}_v1",
            "dataset": dataset,
            "model": model,
            "k": LLM_RUNS_IN_DUMMY,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "config_hash": hashlib.sha256(f"{model}-{dataset}-dummy".encode()).hexdigest()[:12],
            "runs": run_entries,
            "note": "dummy skeleton — real sweeps use k=10 per configs/experiments/",
        })


def build_bert_predictions(root: Path, dataset: str, rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "clinicalbert"
    for model_id in BERT_MODELS:
        err = 0.07 if model_id == "v1_baseline" else 0.05
        for organ_n in ORGANS:
            for idx in range(1, CASES_PER_ORGAN + 1):
                case_id = f"{dataset}{organ_n}_{idx}"
                gold = gold_for(dataset, organ_n, idx)
                pred = noisify(gold, error_rate=err, rng=rng)
                write_json(preds_root / model_id / organ_n / f"{case_id}.json", pred)
        write_json(preds_root / model_id / "_summary.json", {
            "model": f"clinicalbert_{model_id}",
            "dataset": dataset,
            "n_cases": sum(CASES_PER_ORGAN for _ in ORGANS),
            "checkpoint": f"models/clinicalbert/{model_id}/checkpoint.pt",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


def build_rule_based(root: Path, dataset: str, rng: random.Random) -> None:
    preds_root = root / "results" / "predictions" / dataset / "rule_based"
    for organ_n in ORGANS:
        for idx in range(1, CASES_PER_ORGAN + 1):
            case_id = f"{dataset}{organ_n}_{idx}"
            gold = gold_for(dataset, organ_n, idx)
            pred = noisify(gold, error_rate=0.25, rng=rng)
            write_json(preds_root / organ_n / f"{case_id}.json", pred)
    write_json(preds_root / "_summary.json", {
        "method": "rule_based",
        "dataset": dataset,
        "n_cases": sum(CASES_PER_ORGAN for _ in ORGANS),
        "implementation": "src/digital_registrar_research/benchmarks/baselines/rules.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def build_configs(root: Path) -> None:
    cfg = root / "configs"

    # datasets/
    for ds in DATASETS:
        write_yaml(cfg / "datasets" / f"{ds}.yaml", {
            "name": ds,
            "reports_dir": f"data/{ds}/reports",
            "annotations_dir": f"data/{ds}/annotations",
            "preannotation_dir": f"data/{ds}/preannotation",
            "splits_path": f"data/{ds}/splits.json",
            "manifest_path": f"data/{ds}/dataset_manifest.yaml",
            "organs": [{"n": n, "name": ORGANS[n]} for n in ORGANS],
        })

    # models/
    for m in LLM_MODELS:
        write_yaml(cfg / "models" / f"{m}.yaml", {
            "name": m,
            "family": "llm",
            "backend": "ollama" if m.startswith(("gemma", "qwen")) else "vllm",
            "runs": {"k": 10, "seeds": list(range(42, 52))},
            "decoding": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 2048},
        })
    for m in BERT_MODELS:
        write_yaml(cfg / "models" / f"clinicalbert_{m}.yaml", {
            "name": f"clinicalbert_{m}",
            "family": "clinicalbert",
            "checkpoint": f"models/clinicalbert/{m}/checkpoint.pt",
        })
    write_yaml(cfg / "models" / "rule_based.yaml", {
        "name": "rule_based",
        "family": "rule_based",
        "module": "digital_registrar_research.benchmarks.baselines.rules",
    })

    # annotators/
    write_yaml(cfg / "annotators" / "annotators.yaml", {
        "annotators": [
            {"id": "nhc", "display": "NHC", "modes": MODES, "enabled": True},
            {"id": "kpc", "display": "KPC", "modes": MODES, "enabled": True},
        ],
        "modes": {
            "with_preann": "Annotator sees gpt-oss:20b pre-annotation first",
            "without_preann": "Annotator starts from blank template",
        },
        "preannotation_model": PREANN_MODEL,
    })


def build_model_dirs(root: Path) -> None:
    """Placeholders for trained ClinicalBERT checkpoints."""
    for m in BERT_MODELS:
        mdir = root / "models" / "clinicalbert" / m
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "checkpoint.pt.placeholder").write_text(
            "Replace with real .pt checkpoint.\n", encoding="utf-8")
        write_yaml(mdir / "config.yaml", {
            "model_id": f"clinicalbert_{m}",
            "base_model": "emilyalsentzer/Bio_ClinicalBERT",
            "trained_on": "cmuh_train" if m == "v2_finetuned" else "generic_discharge",
        })


def write_readme(root: Path) -> None:
    (root / "README.md").write_text("""# Dummy skeleton — redesigned layout

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
""", encoding="utf-8")


# ---- Entry point ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dummy"),
                    help="Root directory for the generated skeleton (default: ./dummy)")
    ap.add_argument("--clean", action="store_true",
                    help="Delete --out before generating")
    ap.add_argument("--seed", type=int, default=20251117)
    args = ap.parse_args()

    out = args.out.resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for ds in DATASETS:
        build_dataset(out, ds, rng)
        build_llm_predictions(out, ds, rng)
        build_bert_predictions(out, ds, rng)
        build_rule_based(out, ds, rng)

    build_configs(out)
    build_model_dirs(out)
    write_readme(out)

    # Summary
    counts: dict[str, int] = {}
    for p in out.rglob("*"):
        if p.is_file():
            counts[p.suffix or "(no-ext)"] = counts.get(p.suffix or "(no-ext)", 0) + 1
    print(f"Generated skeleton at: {out}")
    for ext, n in sorted(counts.items()):
        print(f"  {ext:>10}: {n}")


if __name__ == "__main__":
    main()
