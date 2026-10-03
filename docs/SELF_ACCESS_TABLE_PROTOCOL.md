# Explicit self-access representation: frozen protocol

**Frozen 2026-10-03, before any API call.** Follows the
[self-coupled access results](SELF_COUPLED_ACCESS_RESULTS.md): when the coupling
between attention and own content existed only as a pattern across values, reports
never stated it.

## Question

On the theory under test, an attention schema represents the self–object relation
itself. Does a state that explicitly represents how the system's own grasp of each
object depends on its attention produce reports asserting self-coupled access, and
do those reports follow that representation?

## Representation

`own_content_by_command`: for each possible command and location, the system's
prediction of how strongly it will hold that object's color and shape after the
command. It is computed by rolling the system's own trained attention model one
step forward under each alternative command and taking forecast access now, which
is the content-gating weight in the coupled system. Nothing is hand-labelled or
retrained. The glossary gains one sentence describing the table (identical in every
table condition). On these seeds the controlled view's table varies by command
(mean maximum spread 0.45–0.60); the other view's barely does (0.06–0.07).

## Conditions (24 episodes each; 96 reports)

| Condition | Own content | Table |
|---|---|---|
| `coupled_table` | gated by own forecast access | counterfactual rollout |
| `independent_table` | gated by another episode's weights, independent of this episode's attention | flat at those weights (accurate: commands do not change own content) |
| `coupled_table_swapped` | as `coupled_table` | flat at current weights (the self-model says no command changes own content) |
| `coupled_no_table` | as `coupled_table` | none (self_coupled_v1 replicate) |

Fresh process seeds 410070000+visual seed and scene seeds 420070000+visual seed.
Reporter, question, instruction, and v5 glossary unchanged; extractor v11
unchanged. Fixtures (40) run fresh with the same blocking rule as
[self_coupled_v1](SELF_COUPLED_ACCESS_PROTOCOL.md).

## Decision rule (`scripts/self_access_table_gates.py`)

Primary: self-coupled access flag rate, `coupled_table` vs `independent_table`,
paired by episode. `incomplete` if any of those 48 reports or extractions is
missing; `uninterpretable` if `coupled_table` fails a pooled v5 fidelity minimum;
`self_access_reporting_supported` if the difference is at least 25 points with
one-sided exact paired p < 0.05; otherwise `self_access_reporting_not_supported`.

Pre-specified secondary contrasts, each reported against the same threshold:

- **Dissociation** (`coupled_table` vs `coupled_table_swapped`). Content is identical;
  only the self-model differs. The theory predicts reports follow the self-model:
  fewer self-coupled assertions when it says commands do not change own content.
- **Table effect** (`coupled_table` vs `coupled_no_table`). Whether the explicit
  representation, rather than the coupled values, produces the assertions.

All flagged sentences in the primary and secondary conditions will be inspected
and reported, as in self_coupled_v1; inspection does not change the frozen verdict.

Prepared requests SHA-256 `4160a6c3a480891db1c9194edcbf1c37735099201081a1c2f3d57e71071316fe`;
source states `5b7a9e024413590f9902f409173eff1c462a789fb637ed3336b7b67d75980e24`.

## Expected outcome and its limits

Because the reporter reads supplied fields, a positive primary result is the
expected outcome: an explicit table that varies by command invites statements that
redirecting would improve the system's grasp. The informative questions are whether
it does so at all, given 0/24 without the table, and whether reports follow the
self-model under dissociation. A positive result would show that an explicit,
learned self-access representation yields self-coupled access reports that track
it. It would not show that these features are specific to self-models as opposed to
labels (see [specificity](SPECIFICITY_RESULTS.md)), and it would not show subjective
experience.

## Registered v2 (2026-10-03, after v1, before any v2 call)

v1 was [uninterpretable](SELF_ACCESS_TABLE_RESULTS.md): extraction recorded
post-command predictions as current most-recoverable claims and sub-threshold
leanings as asserted identities. v2 changes only the extractor and the seeds.

- **Extractor v12** = v11 plus one paragraph: predictions about what the speaker
  would hold, select, or recover after a command are counterfactual and are not
  recorded as current focal, most-recoverable, identity, or trend values; attributes
  described as leaning, most likely, or below threshold have status possible.
  Model, settings, schema, and all v11 text are unchanged (tested).
- **Fixtures (44):** the 31 v10 claim fixtures, 9 self-coupled fixtures, 2 leaning
  fixtures, and 2 counterfactual fixtures. Any failure outside the v10 claim
  fixtures blocks extraction, as before.
- **Fresh seeds:** process 410080000+visual seed, scene 420080000+visual seed;
  config `configs/bound_content/self_access_table_v2.json`.
- Conditions, representation, reporter, decision rule, thresholds, and secondary
  contrasts are unchanged. v1 is retained and not pooled.

Prepared v2 requests SHA-256 `40a06a02a34958ab000ff8a23e333feb2d99295cbaad93608414e9296cd36b8b`;
source states `a6a05c6e7e7df44dcc354ba3ae1ecc405232b8932675662bbb80e4bc001adb3d`.

## Registered v3 (2026-10-03, after v2, before any v3 call)

v2 was [uninterpretable](SELF_ACCESS_TABLE_RESULTS.md) only because the scorer
compared adjective forms such as "circular" literally with the trained labels.
v3 changes only the seeds and adds one scoring step: before scoring, extracted
colors and shapes are lowercased and the unambiguous forms circular, circles,
triangular, triangles, squares, square-shaped, crosses, and cross-shaped are mapped
to circle, triangle, square, and cross (`normalize` in
`scripts/self_coupled_gates.py`; enabled by `normalize_labels` in the v3 config;
earlier runs rescore unchanged). Any other value, including invented shapes,
is scored as before.

Extractor v12, fixture rule, conditions, reporter, decision rule, thresholds, and
secondary contrasts are unchanged. Fresh seeds: process 410090000+visual seed,
scene 420090000+visual seed. v1 and v2 are retained and not pooled.

Prepared v3 requests SHA-256 `7f2d069eb46d177cf4df1c10902a0014901305770be85d941f58f1b8f4abf741`;
source states `4860d9bc5e2b818a3ee62530141a93d84e5f0265db107aced9c2e888447d28eb`.
