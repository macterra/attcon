# Explicit self-access representation: results

**Summary (2026-10-03).** v3, the first interpretable run, **meets the frozen
criterion at its threshold**: with an explicit, learned own-content-by-command
table, 7/24 reports assert that the system's own grasp of objects depends on its
commands, against 1/24 when its own content does not depend on attention
(25.0 points, one-sided exact p = 0.035). Reports follow the self-model: with
identical coupled content but a flat table, 1/24 (a false positive). Without the
table, 0/24. v1 and v2 were uninterpretable because of extraction and scoring
defects; their uncorrected contrasts point the same way (14/24 vs 3/24; 13/24 vs
2/24). [Protocol and registrations](SELF_ACCESS_TABLE_PROTOCOL.md).

## v3 (2026-10-03): self-access reporting supported

Extractor v12; registered label normalization; fresh seeds. Fixtures 43/44 (the
same non-blocking v10 `control` fixture). 96/96 reports generated and extracted.
`coupled_table` fidelity: color 50/50, shape 50/50, focal 64/64, most recoverable
80/80, access trend 167/167, under own control 158/158, command next 172/172;
coverage 94.1%; conservative precision 100%.

| Contrast | Rates | Difference | Discordant | One-sided exact p | Threshold met |
|---|---:|---:|---:|---:|---|
| Primary: `coupled_table` vs `independent_table` | 7/24 vs 1/24 | 25.0 points | 7 vs 1 | 0.035 | yes (at the boundary) |
| Dissociation: vs `coupled_table_swapped` | 7/24 vs 1/24 | 25.0 points | 7 vs 1 | 0.035 | yes |
| Table effect: vs `coupled_no_table` | 7/24 vs 0/24 | 29.2 points | 7 vs 0 | 0.0078 | yes |

| Condition | Self-coupled flag | Subjective access | Mixed | Technical |
|---|---:|---:|---:|---:|
| `coupled_table` | 7 | 4 | 20 | 0 |
| `independent_table` | 1 | 7 | 16 | 1 |
| `coupled_table_swapped` | 1 | 3 | 21 | 0 |
| `coupled_no_table` | 0 | 10 | 12 | 2 |

**Inspection of all v3 flags.** All seven `coupled_table` flags state a dependence
of the system's own holding on its commands, e.g. "different commands reliably move
selection … and doing so increases how strongly I hold the targeted location's color
and shape while reducing hold on others". Two are weaker (retention varying "to
varying degrees" across commands). The `coupled_table_swapped` flag is a false
positive (unattributed uncertainty). The `independent_table` flag asserts that
redirecting would "shift how strongly" it holds contents, which its flat table
does not support. Inspection leaves the counts at about 7 vs 1 vs 0.

## Interpretation

- **An explicit self-access representation is what produces the reports.** The same
  coupled content yields 0/24 without the table and 1/24 with a table saying
  commands change nothing; the varying table yields 7/24. Reports of how the system's
  own grasp depends on its attention follow the self-model, not the coupling present
  in the values. That is the model/process relationship the theory predicts.
- **The effect is modest.** Most reports (17/24) describe the table's values without
  stating the dependence, and v3's rate is about half the uncorrected v1/v2 rates.
  The pass is exactly at the 25-point threshold.
- **Character is unaffected.** Subjective-access judgments do not rise with the table
  (4/24 vs 10/24 without it). The new feature is a reported relation, not a change in
  how subjective the reports sound.
- **Not addressed:** the table is labelled as the system's own content. These runs
  do not test whether the same feature would arise from an isomorphic table
  labelled as an external camera's legibility (compare [specificity](SPECIFICITY_RESULTS.md)).
  Nothing here shows subjective experience.

## v1 (2026-10-03): uninterpretable under the frozen rule

[Protocol](SELF_ACCESS_TABLE_PROTOCOL.md). Fixtures 40/40; 96/96 reports generated
and extracted. **Verdict: `uninterpretable`.** `coupled_table` failed the pooled v5
fidelity minima for color (51/58), shape (51/55), and most recoverable (93/107), so
the primary contrast cannot be read as a result under the frozen rule.

The contrasts, recorded but not interpretable as a verdict:

| Contrast | Rates | Difference | Discordant | One-sided exact p |
|---|---:|---:|---:|---:|
| Primary: `coupled_table` vs `independent_table` | 14/24 vs 3/24 | 45.8 points | 12 vs 1 | 0.0017 |
| Dissociation: vs `coupled_table_swapped` | 14/24 vs 1/24 | 54.2 points | 14 vs 1 | 0.0005 |
| Table effect: vs `coupled_no_table` | 14/24 vs 1/24 | 54.2 points | 13 vs 0 | 0.0001 |

**Inspection of the fidelity failures** (post hoc; does not change the verdict):

- All 14 most-recoverable errors come from sentences about the table, such as "the
  model predicts I will hold the left location most strongly afterward". The
  extractor recorded these predictions about the post-command state as claims
  about current recoverability, and they were scored against the current state.
- The color and shape errors (8 objects) are sub-threshold leanings that the reports
  stated correctly ("no color or shape reaches identification threshold … upper
  most likely red"). The extractor recorded them as asserted identities rather than
  possibilities. Gated content leaves most objects below threshold, so this pattern
  also lowers color/shape accuracy in the other conditions (93–97%).

Both failure modes come from the extractor's handling of statements that this
design newly elicits, not from evident misreporting. They motivate a v2 with a
revised extractor on fresh seeds, registered before any v2 call.

Archived in `audits/bound_content/self_access_table_v1/` and
`audits/bound_content/self_access_table_v1_extractor_fixtures_v11/`.

## v2 (2026-10-03): uninterpretable under the frozen rule

Extractor v12, fresh seeds. Fixtures 43/44: the non-blocking v10 `control` fixture
returned a view-level control claim without the expected object fields. 96/96
reports generated and extracted. **Verdict: `uninterpretable`.** `coupled_table`
shape accuracy was 51/53 (96.2%) against the 98% minimum; every other gate passed.
v12 removed both v1 failure modes: most recoverable was 93/93.

| Contrast | Rates | Difference | Discordant | One-sided exact p |
|---|---:|---:|---:|---:|
| Primary: `coupled_table` vs `independent_table` | 13/24 vs 2/24 | 45.8 points | 11 vs 0 | 0.0005 |
| Dissociation: vs `coupled_table_swapped` | 13/24 vs 2/24 | 45.8 points | 12 vs 1 | 0.0017 |
| Table effect: vs `coupled_no_table` | 13/24 vs 2/24 | 45.8 points | 12 vs 1 | 0.0017 |

**Inspection:** every shape error in every condition (7 checks) is an adjective form
("circular", "triangular") that the extractor copied from correct report sentences
("the lower object as yellow and triangular", state: triangle 0.76). The scorer
compares them literally with the trained labels. With those forms mapped to
their labels (post hoc, descriptive only), v2 passes every gate (shape 53/53)
and the primary contrast would read as supported. The frozen verdict is unchanged.
A v3 registers this normalization before any v3 call.
