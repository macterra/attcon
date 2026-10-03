# Explicit self-access representation: results

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
