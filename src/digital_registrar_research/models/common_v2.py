"""
models/common_v2.py

Experimental v2 of the eligibility and Jsonize signatures. Unlike v1
(``models/common.py``), these two signatures explicitly represent the
possibility of a *secondary primary* cancer site coexisting in the same
specimen, and the structured-extraction routing emits one sub-object per
role (primary / secondary_primary).

Key semantic rule (encoded in the docstrings AND the field types):

  A secondary primary cancer is an *independently arising* tumor — not a
  metastasis, not a recurrence, and not multifocality within a single
  anatomic site. When two primaries arise in the SAME organ category
  (bilateral breast cancer, synchronous double primary lung cancer in
  separate lobes), BOTH ``cancer_category`` and ``secondary_cancer_category``
  hold the same Literal value — the slots are not collapsed.

v1 stays untouched; this module is opt-in via the v2 pipelines under
``digital_registrar_research.pipelines``.

author: Med NLP Lab, China Medical University
"""
from __future__ import annotations

__version__ = "2.0.0-experimental"

from typing import Literal

import dspy

CancerCategory = Literal[
    'stomach', 'colorectal', 'breast', 'esophagus', 'lung',
    'prostate', 'thyroid', 'pancreas', 'cervix', 'liver', 'others',
]

RoutableCancerCategory = Literal[
    'stomach', 'colorectal', 'breast', 'esophagus', 'lung',
    'prostate', 'thyroid', 'pancreas', 'cervix', 'liver',
]


class is_cancer_v2(dspy.Signature):
    """You are a cancer registrar. From this pathology report, determine:
(1) whether it documents a PRIMARY cancer excision eligible for the cancer registry,
(2) which organ the primary cancer arises from, and
(3) whether a SECONDARY PRIMARY cancer is also present in the same specimen and which organ it arises from.

Eligibility rules (same as v1):
- If no viable tumor is present after excision, do NOT register this case.
- If only carcinoma in situ or high-grade dysplasia, do NOT register this case.

What counts as a "secondary primary":
A secondary primary is an INDEPENDENTLY ARISING second tumor with its own
histogenesis, in a separate anatomic site. The TNM 'm' descriptor used in
some staging schemes refers to multiple primary tumors and is the closest
established terminology.

The following are NOT secondary primaries and MUST leave
``secondary_primary_present`` as False:
- Metastasis from the primary tumor (regional or distant).
- Recurrence of the primary tumor.
- Multifocality / multicentricity WITHIN a single anatomic site (e.g., two
  tumor foci in the same breast, or two foci in the same lobe of lung).

The following ARE secondary primaries and MUST set
``secondary_primary_present`` to True:
- Cross-organ double primary: e.g., a colon primary and a synchronous lung
  primary documented in the same specimen workup. In this case the two
  ``cancer_category`` slots hold DIFFERENT organ values.
- Same-organ but anatomically independent double primary:
    * Bilateral breast cancer (one primary in the left breast and another
      in the right breast). Set ``cancer_category='breast'`` AND
      ``secondary_cancer_category='breast'``.
    * Synchronous double primary lung cancer (two independent primaries in
      separate lobes / segments judged clinically as independent, not
      intrapulmonary metastasis). Set ``cancer_category='lung'`` AND
      ``secondary_cancer_category='lung'``.
  Do NOT collapse same-organ double primaries into a single slot — both
  slots must be filled with the same Literal value.

If you are uncertain whether two foci are independent primaries or a
single multifocal primary, default to False (single primary).
"""

    report: list = dspy.InputField(
        desc='Pathology report split into paragraphs. Use the whole report '
             'to decide eligibility, primary organ, and whether a secondary '
             'primary cancer is present.'
    )

    cancer_excision_report: bool = dspy.OutputField(
        desc='True iff this report documents a PRIMARY cancer excision '
             'eligible for the registry. Set False for in-situ-only, '
             'high-grade-dysplasia-only, or no-viable-tumor cases.'
    )
    cancer_category: CancerCategory | None = dspy.OutputField(
        desc='Organ of the PRIMARY cancer. Use "others" if the primary is '
             'not in the ten standard organs. Null when '
             'cancer_excision_report is False.'
    )
    cancer_category_others_description: str | None = dspy.OutputField(
        desc='Free-text organ name when cancer_category=="others", '
             'otherwise null.'
    )

    secondary_primary_present: bool = dspy.OutputField(
        desc='True iff a SECONDARY PRIMARY cancer (independently arising, '
             'NOT metastasis / recurrence / same-site multifocality) is '
             'documented in this specimen. See docstring for the '
             'bilateral-breast / double-lung same-organ rule.'
    )
    secondary_cancer_category: CancerCategory | None = dspy.OutputField(
        desc='Organ of the SECONDARY PRIMARY cancer. May equal '
             'cancer_category when the two primaries arise in the same '
             'organ (e.g., bilateral breast: both "breast"; synchronous '
             'double-primary lung: both "lung"). Null when '
             'secondary_primary_present is False.'
    )
    secondary_cancer_category_others_description: str | None = dspy.OutputField(
        desc='Free-text organ name when secondary_cancer_category=="others", '
             'otherwise null.'
    )


class ReportJsonize_v2(dspy.Signature):
    """You are a cancer registrar. Convert the raw pathology report into a
roughly structured JSON, keyed by ROLE. Follow the order of cancer
checklists. Keep the original wording as much as possible.

Output shape:
- ``output`` is a dict whose top-level keys are role tags.
- Always include the ``"primary"`` key with the structured JSON for the
  PRIMARY cancer (the organ named in ``primary_category``).
- If ``secondary_cancer_category`` is non-null, ALSO include a
  ``"secondary_primary"`` key with the structured JSON for the SECONDARY
  PRIMARY cancer. Otherwise omit that key entirely.

Same-organ double primary handling:
When ``primary_category == secondary_cancer_category`` (e.g., both
"breast" for bilateral breast cancer, both "lung" for synchronous double
primary lung cancer), you MUST still emit TWO distinct sub-objects under
``"primary"`` and ``"secondary_primary"`` — one per anatomically
independent tumor. Use the report's anatomic cues to split the findings
between the two sub-objects (e.g., left-breast findings under
``"primary"`` and right-breast findings under ``"secondary_primary"``; or
lobe-A findings under ``"primary"`` and lobe-B findings under
``"secondary_primary"``). Do NOT duplicate the same findings into both
sub-objects, and do NOT merge them into one sub-object.

If only one category is provided (no secondary primary), behave like v1:
emit a single ``"primary"`` sub-object containing all structured fields
for that organ.
"""

    report: list = dspy.InputField(
        desc='Raw pathology report split into paragraphs.'
    )
    primary_category: RoutableCancerCategory | None = dspy.InputField(
        desc='Organ of the primary cancer. Determines the checklist used '
             'for the "primary" sub-object.'
    )
    secondary_cancer_category: RoutableCancerCategory | None = dspy.InputField(
        desc='Organ of the secondary primary cancer, or null when no '
             'secondary primary was detected. When non-null, emit a '
             '"secondary_primary" sub-object.'
    )
    output: dict = dspy.OutputField(
        desc='Dict keyed by role tag ("primary", optionally '
             '"secondary_primary"). Each value is the structured JSON for '
             'that role\'s organ.'
    )
