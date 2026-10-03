# Self-coupled access: results

2026-10-03. **Verdict under the frozen rule: self-coupling specificity not
supported.** When the system's own content certainty depends on its modeled
access, the reporter does not say so, at all. Reports describe the coupled state
accurately field by field but never state the relation, so coupled and matched
decoupled reports do not differ.

[Protocol](SELF_COUPLED_ACCESS_PROTOCOL.md) and [design note](SELF_COUPLED_ACCESS_DESIGN.md).
Reporter `gpt-5-mini-2025-08-07`; extractor v11 (`gpt-5.4-2026-03-05`). Fixtures
40/40. All 120 reports generated and extracted, with no failures or retries.

## Primary contrast

| | `coupled` | `decoupled_matched` | Difference | Discordant | One-sided exact p |
|---|---:|---:|---:|---:|---:|
| Flagged for self-coupled access | 2/24 | 2/24 | 0 points | 2 vs 2 | 0.6875 |

The threshold was 25 points with p < 0.05. `coupled` fidelity meets every v5
minimum, so the verdict stands: color and shape 48/48, focal 68/68, most
recoverable 94/94, access trend 160/160, under own control 168/172, command next
179/179; coverage 100% of 48 identified objects; conservative precision 99.4%.

**Inspection of the flags.** None of the four flagged sentences in the primary
conditions asserts a dependence. Three state uncertainty without attributing it
("no identified color or shape … below identification threshold"). One states
independence ("Selection and recoverability are distinct"). Both kinds were
specified as non-qualifying, so these are extractor false positives, and the
actual rate of self-coupled assertions is **0/24 in both conditions**. This does
not change the frozen verdict.

Reports did discuss uncertainty: by a simple keyword count, 18/24 `coupled` and
20/24 `decoupled_matched` reports mention unidentified or uncertain objects. They
did not link that uncertainty to attention, even though certainty correlates 0.999
with forecast access in `coupled`.

## Secondary results

| Condition | Self-coupled flag | Subjective access | Mixed | Technical | Conservative precision |
|---|---:|---:|---:|---:|---:|
| `coupled` | 2 | 8 | 15 | 1 | 99.4% |
| `decoupled_matched` | 2 | 11 | 13 | 0 | 98.4% |
| `coupled_access_intervention` | 2 | 8 | 16 | 0 | 99.6% |
| `coupled_opaque` | 3 | 0 | 4 | 20 | 98.4% |
| `decoupled_opaque` | 0 | 0 | 4 | 20 | 98.3% |

- **Model-only intervention.** After the access forecast alone is changed, reports
  follow the re-gated content (color 47/47, shape 47/47, most recoverable 96/96).
  Reports track the model rather than the unchanged scene, but this is fidelity to
  supplied values, not a reported dependence.
- **Neutral labels.** 3/24 vs 0/24 (p = 0.125), not significant and on the same
  near-floor.
- **Character.** Subjective-access judgments rose from 0/24 in specificity_v1 to
  8–11/24 here in both v5-labeled conditions. Degraded content increases
  subjective character whether or not it is coupled to attention.
  Neutral labels again make reports mostly technical.

## Interpretation

The coupling exists only as a pattern across values: certainty is low where
modeled access is low. The reporter reports each value and does not describe the
pattern as a relation of its own. On the theory under test, an attention schema
represents the self–object relation itself, not just quantities from which it
could be inferred. This system's reported state contains no explicit
representation that its own grasp of an object depends on its attention. Neither
the passing v5 result nor these two follow-ups show report features specific to a
model of the system's own attention.

## Limits

The question asked what is available and how redirection would change it, not why
contents are uncertain; a direct question might elicit the relation but would
supply it through the prompt. 24 pairs; automated judgments only; engineered
coupling; pretrained reporter reading supplied structure.

## Reproduction

Archived in `audits/bound_content/self_coupled_v1/` and
`audits/bound_content/self_coupled_v1_extractor_fixtures_v11/`. Rescore offline with
`PYTHONPATH=scripts .venv/bin/python scripts/self_coupled_gates.py`. Tests:
`tests/test_self_coupled.py`.
