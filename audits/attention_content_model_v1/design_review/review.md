# Attention-content model v1: independent prospective design review

2026-10-11. `/root/spatial_content_manner_critical_review` read only
[the proposed contract](../../../docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md) and
its [design record](../design/design.md)/[metadata](../design/metadata.json).
This is static design review: no checkpoints or archives were loaded, no worlds,
inference, fits, Reporter calls, Git mutations or external messages were performed.
No new results, performance thresholds or semantic gate values are introduced.

**Verdict: the proposed trained architecture is a useful bounded
attention-control content/state-estimation development. Approve that direction;
resolve the contract clarifications below before implementation is frozen or
registered. This review authorizes no execution or Reporter release.** The design
does not qualify a presentation counterpart. The existing unresolved native
numerical reconstruction is an execution dependency, as the design already says.

## Current versus prospective observations

The main visibility contract is coherent. At decision t≥1, the last revealed
snapshot is W(t−1); the command that constructed W(t) is known, while W(t) is
concealed. Per-address h contains completed transition feedback only through
W(t−1). A_current replaces the most recently commanded address in that snapshot
with an estimate; it does not receive its scored W(t) target. A_next replaces the
candidate address in A_current with its command-conditioned prediction. Other
addresses can inherit current uncertainty rather than being silently repaired
with actual W(t). At t=0, current V is exact, correctly excluded from the learned
current loss. Current predictions at t=1..14 and next predictions at t=0..14 are
separate scores. Terminal W15 is revealed without another scored decision.

Revealing W(t) only after all decisions, scoring and fork comparisons is essential.
The recurrent update for that reveal must use the preceding command's address,
symbol and completed map; subsequent common-spine command execution must not
update h until its delayed reveal. Training losses may propagate gradients through
later permitted feedback updates, but targets must never enter an earlier forward
call. Targets, evaluator mappings and selected-policy fork outcomes must remain
outside the inference API. The proposed independent reconstruction should compare
per-call observation histories, not infer nonleakage from accurate predictions.

**C1 — make donor replay indices and restoration exact.** “Replay the same fixed
common command prefix ... and delayed feedback” is reasonable but leaves a
high-impact off-by-one choice to the implementation. At a donor intervention at
decision t, define explicitly: execute common commands 0..t−1 to construct the
donor W(t), but update donor h only from feedback W1..W(t−1), corresponding to
commands 0..t−2. Its just-completed W(t) remains concealed. Archive the exact
update list and pre/post-reveal masks. Define each restoration pair as restoration
of the recipient's original h and corresponding estimates/policy outputs before
any feedback update, with exact equality to its unpatched baseline. The design's
restoration mechanics for actual W should remain a separate ledger.

The donor h is reached under a different counterfactual mapping/history with the
same V. Patching it into an unchanged recipient observation history is a selective
synthetic intervention, not an ordinary observation-compatible recipient state.
The document states this correctly. “Nonselected unchanged” must refer to the
patched address and decoder source dependence: a current-estimate patch can also
change an inherited, nonqueried address in a full A_next assembly. The runner must
not mistake that intended inheritance for cross-address decoder leakage.

## Unknown mappings and native policy use

Assignment shuffles are per address and split, with balanced six-permutation
marginals. Address/world/seed names and true assignments are evaluator-only.
The disclosed operator family is equally available to all comparators, while A
uses a genuinely trained GRU/decoder rather than an analytic T in its forward
path. Analytic elimination from revealed maps is a legitimate calibrated
alternative; retaining all tolerance-consistent mappings avoids falsely declaring
an uninformative transition identified. A zero-survivor posterior is correctly
an invariant failure. Posterior-mean maps and expectations of H over hypotheses
have different meanings, appropriately separated in the design.

There is an important limit on the final policy test. The first two symbol
commands at every address have been revealed by decision t=9. When the three
T(V) outcomes are distinguishable, two different-symbol outcomes determine a
permutation of three bandwidths. At t=12, the first final query, l0–l2 have three
revealed outcomes and l3 has two. Thus every final query can occur after full
analytic mapping calibration. This is not leakage or a design failure: early
current/next predictions still test unknown-assignment learning. It means final
command choices should not be described as resolving an unknown mapping online.
Tabulate posterior survivor count/entropy and revealed-symbol count at each
decision; retain ambiguous maps, and report early versus calibrated errors
descriptively without choosing a favorable calibration subset as primary.

The native Controller consumes A_next through H to choose a symbol, and actual
H(W) then issues the grid action. Those are distinct consumers and outputs. This
is a real causal control role for the predicted content map, rather than a
Reporter selecting an operation. But the command choice depends only on three
role maxima; the confidence-only policy must reproduce it exactly. A_current has
no separate native policy role except its contribution to assembled A_next and
its scored current estimates. The design admits these limits. No task-necessity
claim or special manner inference should follow from policy performance or a
donor-induced command change.

## Fairness, arithmetic and split claims

The ordinary predictor has identical architecture, input permissions, histories,
losses, initialization policy, optimizer budget and Controller use. Its independent
initialization tests another fit, not a theory-specific architecture. The exact
same-weight alias check properly demonstrates that renaming the computation
cannot change its outputs. Analytic T knowledge is declared for everyone; the
mapping estimator has the same feedback visibility but additional hardcoded
numerical structure rather than matched trainable capacity. This difference is
explicit, not concealed as an oracle. Sensory-only and extra-W controls are
correctly labelled information alternatives.

The proposed parameter arithmetic checks out:

| Module | Calculation | Parameters |
|---|---|---:|
| GRUCell | 144×771 + 144×48 + 2×144 | 118,224 |
| Decoder first layer | 435×96 + 96 | 41,856 |
| Decoder second layer | 96×384 + 384 | 37,248 |
| Total | GRU + both decoder layers | 197,328 |

The proposed envelope is 864 worlds, 288 in each split across three encoders.
Heldout 288×15=4,320 transitions, 288×14=4,032 selected-current estimates,
288×45=12,960 selected-next alternatives and 288×180=51,840 addressed-next item
records are correct. Each query has one scheduled role, so 288×3×3=2,592 queried
operator/role actions is correct; it is not a both-role action denominator.
Current and next loss terms average separately before their equal weighting.

Splits contain whole worlds, including histories and donors. Numeric namespaces
are separated across split and encoder within the proposed world range. Repeated
layouts and mapping permutations are openly acknowledged. Independent nuisance
realizations do not establish held-out layouts or a model-family sample. Balanced
mapping assignments impose a fixed design with dependence across assignment rows;
the world-bootstrap intervals should retain their conditional descriptive scope,
not acquire an iid model-population or significance interpretation.

**C2 — make comparator and replay ledgers explicit.** The primary table is correct
per prediction source, but “heldout all encoders” does not say which ledgers are
shared actual forks and which repeat for A, ordinary and analytic predictors.
Freeze a source-indexed ledger. Selected-policy execution, estimates and donor
outputs need per-source denominators; confidence/alias duplicate outputs must be
marked, and shared actual operator targets must not be counted as independent
observations. Two donor schedules at t=12,13,14 entail 2×(12+13+14)=78 donor
executor transitions per world, or 22,464 heldout donor-prefix transitions if both
scheduled prefixes are replayed without deduplication. Their hidden-state updates
are fewer because the last completion is concealed. Declare those workload and
provenance counts separately from the 1,728 scored donor pairs. This does not add
new statistical evidence or change any existing primary denominator.

## Implementation freeze and semantic boundary

**C3 — specify the sealing/artifact sequence.** The document calls for a committed
registration before the one new development run, permits a fixed-fit dev inspection,
and says heldout sources stay sealed until checkpoint hashes are registered.
Clarify whether these hashes mean the existing frozen encoder/H inputs or the
new fitted A/ordinary outputs, and define the exact fit→dev→heldout release phases
within the single exclusive execution. Bind input H weights and its native
normalizer, the extraction site, source closure and data permissions explicitly.
If fitted checkpoints must be preserved before heldout release, provide that
phase's immutable receipt without allowing a second fit or a new initialization.
This is bookkeeping precision, not an extra performance-dependent approval loop.

**C4 — independently validate estimated content before future reporting.** Native
H utility validates its role on actual native maps; the decoder can generate
maps outside that support, and even posterior means need not be an executed T
state. Low map MSE alone does not ensure H(A) retains literal constituent content.
Before any future source/report gate, inspect the already proposed field L1,
role-specific cell/coarse disagreement, wrong/ambiguous current estimates,
content/address crossings and W-derived alternatives on the declared independent
world split. Distinguish H(A) evidence, H(W) evidence and evaluator truth. Freeze
those measurements before execution; do not train H again or choose report
queries after inspecting failures. A faithfully exported erroneous estimate can
be a reportable model content, but accurate export wording cannot supply missing
semantic qualification. No new numeric manner gate is recommended here.

This source development is more than unchanged raw-W filter prediction in one
limited sense: it produces its own history-dependent, command-conditioned current
and future maps before completed feedback, and those estimates affect an actual
operation choice. Its numerical target is still the existing V→T(V) filter family.
The whole scheme has an equally informed ordinary-state-estimation explanation,
and its spatial manner remains the previously unqualified fine-content/access
candidate. Training A, observing map errors or patching h does not upgrade that
candidate automatically. Technical prediction/access and null causal effects are
valid endpoints, even with excellent task utility.

The existing negative native presentation verdict does not prohibit this bounded
A development. It does prohibit treating its eventual technical success as
permission for phenomenal correspondence. The proposed separately registered
neutral report study has the right information parity, timestamps, A/W crossings,
source-aware fidelity and separate blinded structure assessment. Its release
must retain the source-semantic/bookkeeping gates. The user's intended restricted
evidence is accurate model-content reports independently resembling the relevant
consciousness-report structure, not proof of inaccessible private experience or
A necessity for targeting. Ordinary parity bounds that evidence rather than
making it logically impossible. Modeler refinement is a stronger optional native
role; its absence is correctly not a mandatory gate for this narrower goal.

This is separate task allocation within the same AI model family with inherited
hypothesis context, not independent-family validation, human endorsement or
empirical replication. No scientific goal is marked achieved and no implementation
or execution result is presumed.
