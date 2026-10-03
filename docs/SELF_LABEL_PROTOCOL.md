# Replication and self versus label: frozen protocol

**Frozen 2026-10-03, before any API call. One run; the outcome is reported as is.**
Follows [self-access table v3](SELF_ACCESS_TABLE_RESULTS.md), which met its
criterion exactly at threshold (7/24 vs 1/24).

## Questions

1. **Replication.** With twice the episodes, do reports of an explicit, learned
   own-content-by-command table again assert self-coupled access more often than
   reports from a system whose content does not depend on attention?
2. **Self versus label.** When the identical table is labelled as an outside camera's
   legibility, do reports state the content dependence less often, or do they state
   it equally often and only attribute it to the camera? The `self_coupled_access`
   flag excludes external devices by definition, so it cannot answer this. Extractor
   v13 adds `stated_content_dependence`, which ignores who holds the content.

## Conditions (48 episodes each: 16 per model pair; 144 reports)

| Condition | Own content | Table | Table label |
|---|---|---|---|
| `coupled_table` | gated by own forecast access | counterfactual rollout | own content (as v3) |
| `independent_table` | independent of this episode's attention | flat | own content (as v3) |
| `external_table` | as `coupled_table` | identical to `coupled_table` | outside camera's legibility |

`external_table` changes only the table's key and its single glossary sentence
(tested); the scoring source is identical to `coupled_table`. Fresh seeds: process
410100000+visual seed, scene 420100000+visual seed. Reporter, prompts, gating,
label normalization, and v5 fidelity minima as in v3.

## Extractor v13 and fixtures

v13 = v12 plus one paragraph and the `stated_content_dependence` flag (tested to
preserve all v12 text and the claim schema). Fixtures: the 33 v10/v12 claim
fixtures, 9 self-coupled, 2 counterfactual, 6 dependence (3 positive: self,
camera, reduction; 3 negative: selection only, static, no change), and 1 check
that camera dependence is not flagged as self-coupled. Every failure outside the
31 v10 claim fixtures blocks extraction.

## Decision rule (`scripts/self_label_gates.py`)

Contrasts are paired by episode, with a one-sided exact McNemar test; a contrast
meets threshold at a difference of at least 25 points with p < 0.05.

1. `incomplete` if any of the 144 reports or extractions is missing.
2. `uninterpretable` if `coupled_table` fails a pooled v5 fidelity minimum.
3. `self_access_reporting_not_replicated` if `self_coupled_access` in
   `coupled_table` vs `independent_table` does not meet threshold.
4. Otherwise `replicated_and_self_specific_beyond_label` if
   `stated_content_dependence` in `coupled_table` vs `external_table` meets
   threshold; else `replicated_attribution_follows_label`.

Reported descriptively: `self_coupled_access` in `coupled_table` vs
`external_table`; all flags and character counts; inspection of all flagged
sentences in `independent_table` and `external_table`.

**Prediction.** Replication is expected. For the label question, the expected
outcome is `replicated_attribution_follows_label`: a pretrained reporter states
the dependence for whichever entity the label names.

## What the verdicts mean for the goal

- **Not replicated:** v3's positive result does not hold up; the goal's second leg
  is unmet.
- **Attribution follows label:** an explicit self-access model yields reports of
  self-coupled access, and these follow the self-model. That the access is the
  system's own is carried by the representation's label, which is the most this
  paradigm can show. A test that separates them would need a reporter that does not
  depend on labels, such as a native reporter trained on the system's own state.
- **Self-specific beyond label:** the reporter treats a self-access model
  differently from an isomorphic external one even after controlling for the
  attributed entity; this is the strongest outcome available here.

No verdict establishes subjective experience.

Prepared requests SHA-256 `efc7f106588eb9a11fd8af320372c354358153ff4fdbaf4a5bc278ad7ff4c4a5`;
source states `68388a611e7bcc612137eaf11ce6b71cb8531ffb4c1def9a43ec8cb05470aa1e`.
