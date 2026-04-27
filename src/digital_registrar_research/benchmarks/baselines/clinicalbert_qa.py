"""
ClinicalBERT extractive QA baseline.

For numeric / span-style fields (tumor size, margin distance, grade
scores, etc.) we frame extraction as SQuAD-style extractive QA:
    question = "What is the tumor size in mm?"
    context  = <report>
and train a span-prediction head over Bio_ClinicalBERT.

Design notes
------------
* Span supervision is **weak / silver**: at train time we auto-locate
  each gold numeric value inside the report with a best-effort regex
  search around the field-specific phrase. Cases where the value is not
  findable in the text are dropped (logged).
* At inference we query every `(question × test_report)` pair and keep
  the highest-confidence span per field.
* This module complements `clinicalbert_cls.py` (which handles
  categorical fields). Together they populate the scoped-comparison
  fields for ClinicalBERT.

Scope NOT covered: nested lists (`margins: [...]`, `biomarkers: [...]`,
`regional_lymph_node: [...]`). These are reported as N/A in the results
table — encoder-only architectures cannot emit variable-length nested
objects without an additional generative decoder.

Usage:
    python baselines/clinicalbert_qa.py --phase train
    python baselines/clinicalbert_qa.py --phase predict \\
        --ckpt ckpts/clinicalbert_qa/ --out workspace/results/benchmarks/clinicalbert_qa
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ...paths import BENCHMARKS_RESULTS, SPLITS_JSON
from ..eval.scope import SPAN_FIELDS, get_field_value

MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DOC_STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS_PATH = SPLITS_JSON

QUESTIONS = {
    "tumor_size":        "What is the tumor size in millimeters?",
    "dcis_size":         "What is the size of ductal carcinoma in situ in millimeters?",
    "maximal_ln_size":   "What is the maximal size of the metastatic lymph node in millimeters?",
    # Grade scores (numeric within a bounded range — QA keeps gradient
    # signal stronger than a 4-way classifier would).
    "grade":             "What is the histologic grade?",
    "nuclear_grade":     "What is the nuclear grade?",
    "tubule_formation":  "What is the tubule formation score?",
    "mitotic_rate":      "What is the mitotic rate score?",
    "total_score":       "What is the Nottingham total score?",
}


# --- Silver span alignment ----------------------------------------------------

def find_span(report: str, value: str | int | float) -> tuple[int, int] | None:
    """Best-effort locate the gold value as a literal substring."""
    if value is None:
        return None
    s = str(value)
    for variant in [s, f"{s} mm", f"{s}mm", f"{s} cm", f"{s}cm"]:
        idx = report.lower().find(variant.lower())
        if idx != -1:
            return idx, idx + len(variant)
    # For numeric grades (1, 2, 3) fall back to regex around the question topic.
    m = re.search(rf"\b{re.escape(s)}\b", report)
    if m:
        return m.start(), m.end()
    return None


class QADataset(Dataset):
    """Builds (question, context, answer_span) triples from gold annotations."""

    def __init__(self, cases: list[dict], tokenizer):
        self.examples: list[dict] = []
        self.tok = tokenizer
        for c in cases:
            report = Path(c["report_path"]).read_text(encoding="utf-8")
            with open(c["annotation_path"], encoding="utf-8") as f:
                gold = json.load(f)
            for field, question in QUESTIONS.items():
                if field not in SPAN_FIELDS:
                    continue
                val = get_field_value(gold, field)
                span = find_span(report, val) if val is not None else None
                if span is None:
                    continue
                self.examples.append({
                    "question": question,
                    "context": report,
                    "answer_start": span[0],
                    "answer_text": report[span[0]:span[1]],
                })
        print(f"Silver-aligned training examples: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        enc = self.tok(
            ex["question"], ex["context"],
            max_length=MAX_LEN, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=False,
            return_offsets_mapping=True, padding="max_length",
        )
        offsets = enc.pop("offset_mapping")
        start_char = ex["answer_start"]
        end_char = start_char + len(ex["answer_text"])
        start_idx = end_idx = 0
        for i_tok, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_idx = i_tok
            if s < end_char <= e:
                end_idx = i_tok
                break
        enc["start_positions"] = start_idx
        enc["end_positions"] = end_idx
        return {k: torch.tensor(v) for k, v in enc.items()}


def load_cases(split_name: str) -> list[dict]:
    with SPLITS_PATH.open(encoding="utf-8") as f:
        return json.load(f)[split_name]


def train(args) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID).to(DEVICE)

    ds = QADataset(load_cases("train"), tok)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=len(loader) * args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            scheduler.step()
            total += out.loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(loader):.4f}")

    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.ckpt)
    tok.save_pretrained(args.ckpt)
    print(f"Saved to {args.ckpt}")


def predict(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt).to(DEVICE)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in load_cases("test"):
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        cancer_data: dict = {}
        for field, question in QUESTIONS.items():
            enc = tok(question, report, max_length=MAX_LEN,
                      truncation="only_second", padding="max_length",
                      return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0]
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            s = out.start_logits.argmax(dim=-1).item()
            e = out.end_logits.argmax(dim=-1).item()
            if e < s:
                continue
            start_char = int(offsets[s][0])
            end_char = int(offsets[e][1])
            answer = report[start_char:end_char].strip()
            # Coerce numerics.
            m = re.search(r"\d+(?:\.\d+)?", answer)
            if m:
                val = float(m.group())
                cancer_data[field] = int(val) if val.is_integer() else val

        # This module intentionally only populates span fields — the
        # categorical/cancer_category side is handled in clinicalbert_cls.py.
        result = {
            "cancer_excision_report": None,
            "cancer_category": None,
            "cancer_category_others_description": None,
            "cancer_data": cancer_data,
        }
        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_qa")
    ap.add_argument("--out", default=str(BENCHMARKS_RESULTS / "clinicalbert_qa"),
                    help="Output dir (default: %(default)s).")
    args = ap.parse_args()

    if args.phase == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
