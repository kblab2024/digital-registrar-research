"""
ClinicalBERT multi-head classifier baseline.

One shared Bio_ClinicalBERT encoder + one linear head per categorical
(`Literal[...]`) field. Trained on the 100-case train split, evaluated
on the 51-case test split.

Why this design
---------------
ClinicalBERT (Alsentzer et al., 2019) is encoder-only and cannot generate
structured outputs. We therefore reduce every categorical field to a
classification task over the `[CLS]` pooled embedding. Nullable fields
use an extra "null" class so the model can say "absent / not mentioned".

Scope covered
-------------
- All fields in the DSPy signatures declared with `Literal[...]` type.
- Boolean fields (True / False / null → 3-way classification).
- The top-level `cancer_category` field.

Scope NOT covered (by design — see docs/literature_review.md for the
architectural-scope argument):
- Nested list fields (`margins`, `biomarkers`, `regional_lymph_node`).
- Continuous numeric fields (tumor_size in mm, distance in mm) — see
  `clinicalbert_qa.py` instead.
- Free-text fields (description, station_name, etc.).

Usage:
    python baselines/clinicalbert_cls.py --phase train
    python baselines/clinicalbert_cls.py --phase predict \\
        --ckpt ckpts/clinicalbert_cls.pt --out ../results/clinicalbert_cls
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse scope definitions from the eval module.
from ..eval.scope import (
    CATEGORICAL_FIELDS,     # dict[field_name, list[str]] — option lists
    CANCER_CATEGORIES,      # list[str] — top-level cancer_category options
    get_field_value,        # helper to read a field out of a gold annotation
)
from ...paths import SPLITS_JSON

MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS_PATH = SPLITS_JSON

# Cases counted as "included" in the study scope: true cancer excision
# reports whose top-level cancer_category is one of the implemented organs.
# Everything else ("others", null, non-excision) has empty cancer_data and
# provides no signal for per-field classification heads.
INCLUDED_CATEGORIES = {
    "breast", "lung", "colorectal", "prostate", "esophagus",
    "pancreas", "thyroid", "cervix", "liver", "stomach",
}


def is_included(annotation: dict) -> bool:
    return (
        bool(annotation.get("cancer_excision_report"))
        and annotation.get("cancer_category") in INCLUDED_CATEGORIES
    )


# --- Dataset ------------------------------------------------------------------

class PathologyCases(Dataset):
    def __init__(self, cases: list[dict], tokenizer,
                 field_to_idx: dict[str, dict[str, int]]):
        self.cases = cases
        self.tok = tokenizer
        self.field_to_idx = field_to_idx

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> dict:
        c = self.cases[i]
        report = Path(c["report_path"]).read_text(encoding="utf-8")
        with open(c["annotation_path"], encoding="utf-8") as f:
            gold = json.load(f)

        enc = self.tok(
            report, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        labels: dict[str, int] = {}
        for field, idx_map in self.field_to_idx.items():
            val = get_field_value(gold, field)
            # 'null' bucket captures absent fields consistently.
            key = "null" if val is None else str(val).lower()
            labels[field] = idx_map.get(key, idx_map.get("null", 0))

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate(batch: list[dict]) -> dict:
    keys_labels = batch[0]["labels"].keys()
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": {
            k: torch.tensor([b["labels"][k] for b in batch], dtype=torch.long)
            for k in keys_labels
        },
    }


# --- Model --------------------------------------------------------------------

class MultiHeadClassifier(nn.Module):
    def __init__(self, field_cardinalities: dict[str, int]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            f: nn.Linear(self.encoder.config.hidden_size, k)
            for f, k in field_cardinalities.items()
        })

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return {f: head(h) for f, head in self.heads.items()}


# --- Train / predict ----------------------------------------------------------

def build_field_vocab() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """field_to_idx[field][value] = class_index,  field_cardinalities[field] = N"""
    field_to_idx: dict[str, dict[str, int]] = {}
    for field, options in {**CATEGORICAL_FIELDS,
                            "cancer_category": CANCER_CATEGORIES}.items():
        vocab = {str(v).lower(): i for i, v in enumerate(options)}
        vocab.setdefault("null", len(vocab))
        field_to_idx[field] = vocab
    card = {f: len(v) for f, v in field_to_idx.items()}
    return field_to_idx, card


def load_cases(split_name: str, included_only: bool = False) -> list[dict]:
    with SPLITS_PATH.open(encoding="utf-8") as f:
        split = json.load(f)
    cases = split[split_name]
    if not included_only:
        return cases
    kept = []
    for c in cases:
        with open(c["annotation_path"], encoding="utf-8") as f:
            ann = json.load(f)
        if is_included(ann):
            kept.append(c)
    return kept


def train(args) -> None:
    field_to_idx, card = build_field_vocab()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MultiHeadClassifier(card).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    cases = load_cases("train", included_only=args.included_only)
    print(f"Training on {len(cases)} cases"
          f"{' (included-only)' if args.included_only else ''}")
    train_loader = DataLoader(
        PathologyCases(cases, tok, field_to_idx),
        batch_size=4, shuffle=True, collate_fn=collate,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(ids, mask)
            loss = torch.stack([
                loss_fn(logits[f], batch["labels"][f].to(DEVICE))
                for f in logits
            ]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(train_loader):.4f}")

    ckpt_dir = Path(args.ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "field_to_idx": field_to_idx,
                "card": card,
                "included_only": args.included_only,
                "n_train_cases": len(cases)}, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")


def predict(args) -> None:
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    field_to_idx, card = ckpt["field_to_idx"], ckpt["card"]
    idx_to_field = {f: {i: v for v, i in vocab.items()}
                    for f, vocab in field_to_idx.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MultiHeadClassifier(card).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in load_cases("test"):
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        enc = tok(report, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        preds = {f: idx_to_field[f][l.argmax(dim=-1).item()]
                 for f, l in logits.items()}

        # Shape output like the gold schema.
        cancer_category = preds.pop("cancer_category", None)
        if cancer_category == "null":
            cancer_category = None
        cancer_data = {
            f: (None if v == "null" else v) for f, v in preds.items()
        }
        result = {
            "cancer_excision_report": cancer_category is not None,
            "cancer_category": cancer_category,
            "cancer_category_others_description": None,
            "cancer_data": cancer_data,
        }
        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_cls.pt")
    ap.add_argument("--out", default="../results/clinicalbert_cls")
    ap.add_argument(
        "--included-only", action="store_true",
        help="Train only on cases where cancer_excision_report is True "
             "and cancer_category is one of the implemented organs "
             "(drops 'others' / null / non-excision cases that have "
             "empty cancer_data).",
    )
    args = ap.parse_args()

    if args.phase == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
