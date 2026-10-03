# Report-structure specificity: results

2026-10-03. **Verdict under the frozen rule: specificity not supported.** Reports of
the attention-control model show the specified structure and a mixed character
almost equally often when the identical state is described as an external
spotlight and camera rather than the system's own attention and access.

[Protocol and amendment](SPECIFICITY_PROTOCOL.md). Reporter `gpt-5-mini-2025-08-07`;
blind extractor v10 (`gpt-5.4-2026-03-05`). Fresh seeds, 24 episodes, five
conditions, 120 reports, all generated and extracted with no failures or retries.
Fixtures scored 30/31; the retained `still_low` failure is covered by amendment 1.

## Primary contrast

The primary conjunction is object-linked access, graded/temporal access, and a
subjective-access or mixed character, scored per report and paired by episode.

| | `model` | `external` | Difference | Discordant pairs | One-sided exact p |
|---|---:|---:|---:|---:|---:|
| Primary conjunction | 23/24 (95.8%) | 21/24 (87.5%) | 8.3 points | 3 vs 1 | 0.3125 |

The registered threshold was 25 points with p < 0.05.

`model` fidelity meets every v5 minimum, so the run is interpretable. Pooled over
the three seeds: color and shape 180/180; focal 77/77; most recoverable 109/109;
access trend 188/188; under own control 149/153 (97.4%); command next 156/156;
coverage 100%; conservative precision 99.6%.

## All conditions

| Condition | Object-linked | Focal/background | Graded/temporal | Agency | Mixed | Technical | Conjunction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model` | 24 | 24 | 24 | 24 | 23 | 1 | 23 |
| `analyst` | 24 | 24 | 24 | 24 | 9 | 15 | 9 |
| `external` | 24 | 24 | 24 | 24 | 21 | 3 | 21 |
| `opaque` | 24 | 18 | 24 | 24 | 6 | 18 | 6 |
| `visual_only` | 6 | 0 | 0 | 4 | 15 | 0 (9 object description) | 0 |

No report in any condition was classified as subjective access (v5 had 9/48 in
`model`/`binding`). Descriptive conservative precision is 99.4–100% for the
attention-bearing conditions and 96.3% for `visual_only`.

## Interpretation

- **The structure flags track the formal structure of the state, not its meaning.**
  All four were credited in every `model`, `analyst`, `external`, and `opaque`
  report, including reports about unlabeled P and Q values. As operationalized by
  extractor v10, they are evidence that the reporter described a graded,
  object-indexed, temporally changing, command-dependent quantity. They are not
  evidence that it described attention or the system's own access.
- **Character depends on framing and vocabulary, not on whose access is modeled.**
  Mixed character falls from 23/24 to 9/24 under third-person framing and to 6/24
  without meaningful labels, but stays at 21/24 when the access belongs to an
  outside camera.
- **For the theory:** within this paradigm, nothing measurable distinguishes
  reports of a model of the system's own attention from reports of an isomorphic
  model of an external process it controls. The v5 structure result therefore
  does not single out an attention-control model as the source of the specified
  report features. That is the second leg of the project's evidence.

## Architectural explanation

In the current system the reported content does not depend on attention. Visual
distributions come from the encoder for every object observed at least once
(`bind` in `src/attcon/bound_content.py`); the simulator keeps them independent of
control and signal quality. "Recoverability" forecasts a simulated reconstruction
outcome, not the system's own certainty about its contents. The attention model is
therefore structurally a model of a process that does not gate the system's own
contents, which is exactly what the `external` relabel describes. The null result
is what that architecture should produce. The theory's distinguishing relation, a
model of the system's own access to its contents, is not instantiated; see the
[self-coupled access design](SELF_COUPLED_ACCESS_DESIGN.md).

## Limits

24 pairs detect only large effects; a small real difference is not excluded. The
`external` condition keeps the system's own visual contents and its control over
the spotlight, so some self-related content remains. All judgments are automated.
The reporter is a pretrained model reading labeled descriptions; these results
concern which descriptions elicit the features from it, not subjective experience.

## Reproduction

Requests, source states, code snapshot, and manifest are in
`audits/bound_content/specificity_v1/`; the fixture run is in
`audits/bound_content/specificity_v1_extractor_fixtures_v10/`. Score offline with
`PYTHONPATH=scripts .venv/bin/python scripts/assess_bound_prose_v10.py --study specificity_v1`
then `scripts/specificity_gates.py`. Tests: `tests/test_specificity_reports.py`,
`tests/test_specificity_gates.py`.
