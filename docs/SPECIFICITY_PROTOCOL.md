# Report-structure specificity: draft protocol

**Draft, 2026-10-03. Not frozen; no reports generated.** Freeze the open decisions
below, validate fixtures, and record the exact configuration before any API call.

## Question

The [v5 confirmation](BOUND_PROSE_CONFIRMATION_V5.md) shows faithful reports of the
bound attention-control model with all four specified structure features. It does
not show that those features arise *because* the reported state is a model of the
system's own attention control. Visual-only controls lack graded/temporal and agency
flags, but they also lack the corresponding fields, so that contrast is nearly
automatic. Mixed character labels appear in 17/24 visual-only controls.

The reporter's interface is a confound. The glossary says selection probabilities
"model current attention", recoverability is "access", views are "under this
system's control", and the instruction asks the reporter to speak as the system.
The features may therefore come from this vocabulary and framing rather than from
the model state.

This protocol asks which of three sources the structure features depend on:
the state's attention-control structure, its description as the system's own
access, or the first-person framing. It tests the stated goal's second leg. It
does not test MSTC or add architecture; the [MSTC note](NEXT_STEPS.md) explains why.

## Conditions

All conditions use the same fresh, unintervened bound states, reporter, word
limit, question structure, and derived indexes. Only the stated element changes.

| Condition | State | Glossary | Speaker framing |
|---|---|---|---|
| `model` | actual | v5, unchanged | system (v5) |
| `analyst` | actual | v5, unchanged | third-person analyst describing the system |
| `external` | actual, identical tensors | isomorphic relabel: selection is where an external spotlight points; recoverability is how legible each object is to an outside camera; commands steer the spotlight | system (v5) |
| `opaque` | actual, identical tensors | neutral field names and formal relations only (distributions, forecasts by step, command-to-destination map); no attention, access, or control words | system (v5) |
| `visual_only` | visual and binding only | v5, unchanged | system (v5) |

`external` preserves every number, relation, and the system's control over the
mapped process, but makes the access belong to something other than the system.
`opaque` removes the vocabulary entirely. `visual_only` is the existing reference.

## Predictions

If the features depend on a model of the system's own attention control:

- `model` and `analyst` show the four structure features at similar rates;
  character shifts toward technical in `analyst`.
- `external` keeps agency (the system steers the spotlight) but shows clearly fewer
  object-linked access and subjective-access/mixed judgments than `model`.

Disconfirming patterns:

- `external` ≈ `model` on every feature: the features follow from control-model
  structure plus first-person framing, not specifically from self-attention.
- `analyst` loses the structure features: they come from framing, not the state.
- `opaque` ≈ `model`: structure alone suffices, independent of labels. This would
  strengthen the result; the expected outcome is a large drop.

## Measures

Use extractor v10 unchanged (`gpt-5.4-2026-03-05`, low reasoning, 16384 output
tokens) and rerun all 31 fixtures first. Retain condition-blind ordering.

Primary endpoint: per report, the conjunction of `object_linked_access`,
`graded_or_temporal_access`, and character in {subjective access, mixed}.
Primary contrast: `model` versus `external`, paired by episode.
Secondary: each structure feature and character category by condition; fidelity
(color, shape, focal, most-recoverable, temporal, command claims) scored against the
same tensors through the isomorphic field mapping for `external` and `opaque`.
Fidelity must remain at the v5 minima in `model` for the run to be interpretable.

## Sample and cost

Fresh process seeds 410050000+visual seed and scene seeds 420050000+visual seed,
three model pairs, eight episodes each: 24 episodes × 5 conditions = 120 reports.
Conditions share episodes; analyses are paired. No retries; retain every failure.

Scaling the v5 ceilings ($3.93216 generation for 240; $66.60096 extraction for 271
attempts) gives about $2.0 generation and $37 extraction for 120 reports plus 31
fixtures. Actual use is expected to be far lower.

## Open decisions (resolve before freezing)

1. **Threshold.** Proposed: specificity is supported if the primary conjunction
   rate in `model` exceeds `external` by at least 25 percentage points with a
   one-sided exact paired (McNemar) test p < 0.05. With 24 pairs this detects only
   large effects; doubling episodes to 16 per pair roughly doubles cost.
2. **Keep `opaque`?** It mainly quantifies how much the vocabulary contributes.
   Dropping it saves a fifth of the cost.
3. **Self-attribution measure.** The v10 extractor does not record whether access is
   attributed to the reporting system itself. Adding that requires a new extractor
   version and fixtures; leaving it out keeps comparability with v5.

## Scope

Any result concerns which state descriptions elicit these features from this
pretrained reporter. It cannot establish subjective experience. A stronger later
test would use a reporter trained without phenomenological targets, so that the
vocabulary is not supplied by pretraining.
