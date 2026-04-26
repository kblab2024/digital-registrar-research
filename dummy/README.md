# Dummy fixture (synthetic, regenerable)

This tree is produced by `scripts/gen_dummy_skeleton.py`. The toolkit is
checked in; the output is not. Regenerate any time:

```
python scripts/gen_dummy_skeleton.py --out dummy --clean
```

Defaults: cmuh = 100 × 10 organs, tcga = 50 × 10 organs, 80% cancer / 20% non-cancer, 3 LLM runs.
For a 10-run sweep matching the real experiments: `--llm-runs 10`.

CMUH reports are clean key-value text. TCGA reports are chaotic
(dictation-style, abbreviations, shuffled sections) so the two datasets
exercise different parser robustness.
