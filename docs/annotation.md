# Annotation UI

A Streamlit app for doctors to review GPT-OSS pre-annotations and save corrected annotations.

## Workflow

The flow is **GPT-OSS pre-annotates → doctor corrects in app → save as final annotation**. Doctors need to see at a glance what the model pre-filled vs. what they changed; the UI highlights diffs with a `✎` marker plus a "pre-annotated" caption.

## Three-folder data convention

The app expects a base folder with three timestamped sibling subfolders:

```
<base_dir>/
├── tcga_dataset_20251117/      # raw .txt reports
├── tcga_result_20251117/       # GPT-OSS pre-annotations (one *_output.json per case)
└── tcga_annotation_20251117/   # doctor-saved final annotations (auto-created on first save)
```

Folder discovery is by regex on `{prefix}_{kind}_{date}` — see `annotation.io.discover_folders`. The packaged TCGA example data under [`data/`](../data) follows this convention so the app works out of the box.

## JSON contract

Saved annotations match the canonical Pydantic schema for the organ — top-level routing fields plus a flat `cancer_data` object (no section-based intermediate layer). For example:

```json
{
  "cancer_excision_report": true,
  "cancer_category": "lung",
  "cancer_category_others_description": null,
  "cancer_data": {
    "procedure": "lobectomy",
    "histology": "adenocarcinoma",
    "grade": 2,
    "margins": [{"margin_category": "bronchial", "margin_involved": false, "distance": 12}],
    "regional_lymph_node": [...]
  }
}
```

This shape is identical to what `CancerPipeline` emits at runtime — the annotation tool just hand-corrects it.

## Launching

```bash
registrar-annotate                              # spawns `streamlit run` on the app
registrar-annotate --server.port 8080           # forward args to streamlit
```

Or directly:

```bash
streamlit run src/digital_registrar_research/annotation/app.py
```

## UI principles

The user is a busy clinician. Defaults that should not change without a strong reason:

- Use **native Streamlit widgets** — no over-customisation.
- Folder selection uses a **native GUI dialog** (Windows + macOS); both must work.
- Pre-annotation diffs are marked with `✎` and a caption — never hide the model's input.
