# Canonical CLI invocations (recipes)

Copy-paste examples for every common evaluation task. All examples assume you're at the repo root.

## Score one model on /dummy

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/cmuh/gpt_oss_20b
```

Auto-discovers all `run*` subdirectories under the model. To restrict to specific runs:

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --run-ids run01 run02 \
    --out workspace/results/eval/non_nested/cmuh/gpt_oss_20b__r01_r02
```

## Single-run mode

Multi-run is the default; for a single-run analysis just pass one ID:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --run-ids run01 \
    --annotator gold --out workspace/results/eval/non_nested/single_run
```

## Score nested fields

```
python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field regional_lymph_node \
    --out workspace/results/eval/nested/lymph_nodes

python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field margins \
    --out workspace/results/eval/nested/margins

python -m scripts.eval.cli nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --field biomarkers \
    --out workspace/results/eval/nested/biomarkers
```

## IAA across all annotators

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators gold nhc_with_preann nhc_without_preann \
                 kpc_with_preann kpc_without_preann \
    --preann-model gpt_oss_20b \
    --out workspace/results/eval/iaa
```

To run only a specific subset of pairs:

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators gold nhc_with_preann kpc_with_preann \
    --pairs gold:nhc_with_preann gold:kpc_with_preann \
            nhc_with_preann:kpc_with_preann \
    --out workspace/results/eval/iaa
```

## Compare humans with vs without preann (preann-effect headline)

This is built into the `iaa` subcommand — the `preann/` subdirectory of the output captures Δκ, anchoring index, convergence-to-preann, disagreement reduction, and edit distance:

```
python -m scripts.eval.cli iaa \
    --root workspace --dataset cmuh \
    --annotators nhc_with_preann nhc_without_preann \
                 kpc_with_preann kpc_without_preann gold \
    --out workspace/results/eval/iaa
ls workspace/results/eval/iaa/preann/
```

## Modularity-advantage table (ablation headline)

Compare multiple methods on the same case set. The `completeness` subcommand emits the modularity-advantage CSV.

```
python -m scripts.eval.cli completeness \
    --root workspace --dataset cmuh \
    --methods llm:gpt_oss_20b llm:qwen3_30b \
              clinicalbert:v2_finetuned rule_based: \
    --annotator gold \
    --out workspace/results/eval/completeness
cat workspace/results/eval/completeness/modularity_advantage.csv | head
```

## Reviewer-rebuttal diagnostics

After running `non_nested` and `iaa`:

```
python -m scripts.eval.cli diagnostics \
    --non-nested-out workspace/results/eval/non_nested/cmuh/gpt_oss_20b \
    --iaa-out workspace/results/eval/iaa \
    --out workspace/results/eval/diagnostics
```

Outputs `error_source_decomposition.csv` (model_error vs report_ambiguity vs report_silent buckets) and `accuracy_by_difficulty_tier.csv` (accuracy stratified by IAA-derived field difficulty).

## Cross-dataset generalisation

Score the same model on both datasets, then compare:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/cmuh

python -m scripts.eval.cli non_nested \
    --root workspace --dataset tcga \
    --model gpt_oss_20b --annotator gold \
    --out workspace/results/eval/non_nested/tcga

python -m scripts.eval.cli cross_dataset \
    --left workspace/results/eval/non_nested/cmuh \
    --right workspace/results/eval/non_nested/tcga \
    --out workspace/results/eval/cross_dataset
```

## Joint headline forest plot

Combines non_nested + IAA outputs into a single long-form CSV for plotting:

```
python -m scripts.eval.cli headline \
    --non-nested-out workspace/results/eval/non_nested/cmuh/gpt_oss_20b \
    --iaa-out workspace/results/eval/iaa \
    --out workspace/results/eval/headline
```

## Subset to specific organs

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --organs breast lung colorectal \
    --out workspace/results/eval/non_nested/breast_lung_colon
```

Accepts either organ names or numeric indices: `--organs 1 6 3` (breast=1, lung=6, colorectal=3).

## Subset to specific cases

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --cases cmuh1_42 cmuh1_43 cmuh1_44 \
    --out workspace/results/eval/non_nested/spot_check

# Or from a file:
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --cases @/tmp/case_list.txt \
    --out workspace/results/eval/non_nested/spot_check
```

## Faster smoke-tests with fewer bootstrap reps

Default `n_boot = 2000` — for smoke tests use `--n-boot 100` for ~20× speedup:

```
python -m scripts.eval.cli non_nested \
    --root dummy --dataset cmuh \
    --model gpt_oss_20b --annotator gold \
    --n-boot 100 \
    --out /tmp/smoke
```

## Score against a non-gold annotator

The `--annotator` arg accepts any annotator subdirectory:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --model gpt_oss_20b --annotator nhc_with_preann \
    --out workspace/results/eval/non_nested/vs_nhc_with_preann
```

Useful for measuring "how close to NHC's annotations" rather than to gold.

## Score ClinicalBERT or rule_based

These methods don't have run IDs:

```
python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method clinicalbert --model v2_finetuned \
    --annotator gold \
    --out workspace/results/eval/non_nested/clinicalbert

python -m scripts.eval.cli non_nested \
    --root workspace --dataset cmuh \
    --method rule_based \
    --annotator gold \
    --out workspace/results/eval/non_nested/rule_based
```

## Inspecting outputs

Every output dir contains a `manifest.json` with full provenance:

```
cat workspace/results/eval/non_nested/cmuh/gpt_oss_20b/manifest.json | jq .
```

Use `git_sha` from the manifest when citing numbers in the paper — it's the canonical reproducibility key.
