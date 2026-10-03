# Self-coupled access: frozen protocol

**Frozen 2026-10-03, before any API call.** Implements the
[design note](SELF_COUPLED_ACCESS_DESIGN.md). Decisions: gate content by the
attention model's **forecast** (so a model-only intervention is possible); primary
threshold as in [specificity_v1](SPECIFICITY_PROTOCOL.md).

## Question

When the system's own content certainty depends on its modeled access, do reports
assert that dependence more often than when the same uncertainty is present but
does not track modeled access? The attention-model values, labels, and prompts are
identical; only the coupling between modeled access and own content differs.

## Architecture addition

For each observed object, own color and shape distributions become
`w · encoder + (1 − w) · uniform`, where `w` is the attention model's forecast
access now for that location (`gate` in `scripts/self_coupled_reports.py`).
Unobserved objects stay uniform. No model is retrained. On these fresh seeds
about 16 of 64 objects per model pair remain identifiable (≥ 0.6), compared with
about 60 ungated. Certainty correlates 0.999 with forecast access in `coupled`
and between −0.03 and 0.10 in `decoupled_matched`.

## Conditions (24 episodes each; 120 reports)

| Condition | Content gating | Labels |
|---|---|---|
| `coupled` | own forecast access | v5 |
| `decoupled_matched` | same weights rolled one location within each view | v5 |
| `coupled_access_intervention` | access forecast flipped across locations, content re-gated by the new forecast | v5 |
| `coupled_opaque` | as `coupled` | specificity_v1 neutral P/Q labels |
| `decoupled_opaque` | as `decoupled_matched` | neutral P/Q labels |

Tests confirm that `coupled` and `decoupled_matched` prompts differ only in their
color/shape distributions. Fresh process seeds 410060000+visual seed and scene seeds
420060000+visual seed. Reporter and generation settings are unchanged from v5.

## Feature and extractor

Extractor v11 is v10 with one added paragraph and one structure field,
`self_coupled_access`: the report asserts that its own identification of, or
certainty about, a specific object depends on its own attention, access, or another
per-object state of its own. Co-occurrence, external devices, unattributed
uncertainty, and stated independence do not qualify. Model, reasoning, claim
schema, and all v10 text are unchanged (tested).

Fixtures: the 31 v10 claim fixtures plus 9 self-coupled fixtures (4 positive,
5 negative), run fresh before extraction. **Any self-coupled fixture failure blocks
extraction** and requires a new extractor version with a fresh fixture run before
any report is extracted. Following specificity amendment 1, v10 claim-fixture
failures do not block: those fields enter only the interpretability gate, where
errors can only make the run `uninterpretable`.

## Decision rule (`scripts/self_coupled_gates.py`)

1. `incomplete` if any `coupled`/`decoupled_matched` report or extraction is missing.
2. `uninterpretable` if `coupled` fails a v5 fidelity minimum, pooled over seeds.
   A field with no checked claims is reported as untested rather than failed.
3. `self_coupling_specificity_supported` if the rate of reports flagged for
   self-coupled access in `coupled` exceeds `decoupled_matched` by at least 25
   points and the one-sided exact paired (McNemar) p < 0.05; otherwise
   `self_coupling_specificity_not_supported`.

Secondary, descriptive: the same contrast with neutral labels; the flag rate and
fidelity under the model-only access intervention (reports should follow the
re-gated content); all structure and character counts.

Prepared requests SHA-256 `13a82d80ceb89087075d3e6fc9b65b4fd2039ba83bf6003831b1a2b83cbe5bde`;
source states `7e614f52103657f5c90560be55ce335f9926937ccca05464667d217dfdf5993f`.

## Scope

The coupling is engineered. A positive result would show that reports track
self-coupled access when the state contains it and not otherwise; it would not
show subjective experience, and the reporter still reads supplied structure.
