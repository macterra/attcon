# Reproduce the bound-content study

Run from the repository root using Python 3.12 and the dependencies in
`pyproject.toml`. The committed checkpoints and API records support offline
verification; an API key is unnecessary for that verification.

```bash
python -m unittest discover -s tests
python scripts/verify_bound_mechanism.py
python scripts/assess_bound_prose_v4.py --study language_confirmation_v1
python scripts/bound_confirmation_gates.py
python scripts/assess_bound_prose_v6.py --study language_confirmation_v2
python scripts/bound_confirmation_gates_v2.py
python scripts/verify_bound_study.py
python scripts/build_bound_explorer.py --study language_confirmation_v2
```

The final two commands require the completed archive and its verification
manifest. While confirmation is running, reserved attempts are unfinished records,
not successful reports. No command above generates language or changes a model.
The assessor rewrites deterministic summaries; the explorer rewrites the HTML page.

## Archive and provenance

`audits/bound_content` contains the pilot, two visual confirmations, three prose
pilots, failed prose confirmation v1, and fresh prose confirmation v2. The visual
checkpoints are paired with the frozen predictive attention models 901, 911, 921
from `audits/predictive_attention`. Visual seed 1221's failed joint-accuracy gate
is retained; visual seeds 1301, 1311, 1321 are the fresh confirmed models.

Each prose study archives exact prompts, complete API responses, the generating
source and config, checkpoint hashes, original scene patches/labels, model tensors,
and every intervention. Source replay uses the archived generator rather than
assuming the latest generator produced earlier records. The replay checks every
request and tensor exactly. The archive manifest hashes every experimental file.

Extractor folders contain the separate model's complete responses and parsed
claims. It sees only the prose, never the source state, condition, theory, or
expected answer. Fixture answers are scored locally and never sent to the API.
Version 4 uses quoted evidence; version 6 uses original sentence IDs and distinguishes
dominant identities from explicit possibilities. Earlier failures are retained.
Automated extraction and character judgments can be wrong; they are not human ratings.

## New experiments

The training scripts `train_bound_content.py` and `train_bound_content_v2.py`
produce the visual models. Use new output directories for new runs. A new language
study requires a new registered config and fresh output name, then:

```bash
python scripts/bound_reports.py --config configs/bound_content/NEW_STUDY.json
python scripts/extract_bound_prose_v6.py --study NEW_STUDY --limit 240
```

These two commands require an OpenAI API key and incur costs. Each attempt is
reserved before the request and has no automatic retry. Existing paths are not
silently replaced. Do not remove failed or incomplete attempts to rerun them.
The manifests freeze model IDs, prompts, request/token limits, and provenance.

## Interpretation

Perception, attention forecasting, binding-mediated control, prose factuality,
paired intervention following, and report character are separate checks. Read
[confirmation v2](BOUND_PROSE_CONFIRMATION_V2.md) for the unchanged acceptance
thresholds. Passing a report-character rubric would support a limited engineered
analogue, not establish subjective experience or the source-of-qualia theory.
