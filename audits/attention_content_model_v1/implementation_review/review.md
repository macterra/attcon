# Learned attention-content core: static pretest review

2026-10-11. Reviewer `/root/spatial_native_numerical_execution_review`; separate
allocation within the same AI model team with inherited hypothesis/project
context. No independent-family or human endorsement is supplied.

**Approve the unchanged bounded handcrafted toy suite after preservation of this
first source/review. No blocking core defect was found statically.** This does
not approve a collector, runner, fit, artifact loading, actual A execution or
Reporter call. The author first toy tests have not been executed by this reviewer.
Runtime outcomes remain pending and must be preserved before any repair.

## Reviewed bytes and scope

The review reads the amended approved contract, learned-core implementation,
proposed tests and implementation record. Exact hashes are:

- `scripts/attention_content_model_v1.py`: SHA256 `a950a4228e5358a6896c73b729ac0e85e630cec8ae05b3944e03a6019221c0b8`.
- `tests/test_attention_content_model_v1.py`: SHA256 `5a22e976865791028a04640203cd0d6ddd48727d79caf23bdb7da96c12543045`.
- `docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md`: SHA256 `b64062f3c07aa1e29b8fd5d3f50bcc04b2188edfe2a8d6b26cafb4a9d04ff2af`.
- `audits/attention_content_model_v1/implementation/design.md`: SHA256 `172003e4108d68676fb77af4c452972f4db8064797baebdda4ba43d74d09fa0b`.

The supplied amended contract hash matches the independently approved C1–C4
clarifications. The native corrected numerical result is reported by the
coordinator as passed/preserved at `c84d3da`; this static review did not reopen
its archives. No source imports/execution, model initialization, production
calls, archive/checkpoint loading, renderer calls, gradient execution, optimizer,
fit, Reporter, Git operation or network action occurred here. Only static reads,
AST parsing, byte hashes and integer arithmetic were used. The only owned writes
are this review and `metadata.json`.

## Architecture and equations

The learned class has the specified CPU float32 shared GRUCell(771,48) and residual
435→96→384 decoder. Static parameter arithmetic is:

| Component | Arithmetic | Parameters |
|---|---|---:|
| GRU | 3×48×771 + 3×48×48 + 2×3×48 | 118,224 |
| Decoder | 435×96 + 96 + 96×384 + 384 | 79,104 |
| Total | GRU + decoder | 197,328 |

The recurrent input is normalized selected-address V[384], revealed completed
W[384] and symbol one-hot[3]. The decoder input is normalized selected V[384],
that address's h[48] and symbol[3]. It adds its output to normalized V and
explicitly denormalizes before returning raw-unit maps. Fit-only population
statistics use scales below 1e-6 replaced with one. There is no address embedding,
world seed, bandwidth classifier, analytic smoothing or block-average call in
the learned forward path. Analytic operator-family code is confined to the
separate declared comparator.

`source_key` is an audit label used in state/provenance validation and exported
attribution. It is not concatenated into network inputs or used to select a
numeric transformation. Ordinary has the identical learned architecture,
normalizer and API, with seed 320000000+encoder_seed+1000000 versus A's
seed 320000000+encoder_seed. ExactAlias uses the same A instance and computation;
it does not create an independent source or new evidence sample. Actual matched
training history, loss and budget remain future runner obligations.

## Current/next timing and permitted inputs

Observation validation requires last-revealed timestamp max(0,t−1); the state
must agree with that timestamp and contain exactly update command indices
0..revealed_through−1. At t=0 the observed current state is exactly V and no
previous command is allowed. At t>=1 current assembly clones the last-revealed
snapshot and predicts only the most recent command's addressed item. Each of
three prospective assemblies clones that current estimate and replaces only its
candidate address. Uncertain current estimates inherited at other addresses are
preserved, matching the amended contract.

At decision 12, for example, source state has feedback through W11 and command
updates 0..10, while the issued command 11 supplies the current prediction of W12.
W12 is not an inference argument. A next candidate predicts W13 without receiving
W12. Feedback validation then advances one completion with its command index and
post-scoring tag; duplicate, skipped or inconsistent timestamps fail validation.

No pixel, target cell, mapping/bandwidth truth, driver label, renderer seed or
world identifier enters the learned API. An extra current-target argument is
absent and is explicitly rejected in the proposed tests. The API cannot prove
that a caller supplied truthful timing tags, immutable V or the proper historical
snapshot; the implementation record states this limit. Collector/runner input
manifests and phase receipts must establish actual permissions and scoring order.
The terminal t=15 reveal is an archive event without an additional scored decision;
the later runner must not invent a scored tag merely to call an unnecessary core
update. Episode limits and terminal archival are outside this core module.

## Gradients, locality and selective interventions

Only the selected address enters a GRU update. The recipient hidden tensor is
cloned and that addressed result assigned, leaving all other addresses unchanged.
There is no detach on latent-state updates or snapshots, so the computation graph
can propagate through recurrent feedback and subsequent decoding. Fit normalizer
coordinates are deliberately detached and frozen; native H diagnostics run under
no_grad. Static reasoning cannot certify actual gradient values.

The proposed single gradient fixture calls backward from decoded content through
one feedback update and checks a nonzero GRU input-weight gradient. It has no
optimizer, fit loop or parameter update. This is bounded software validation,
not the later fifteen-transition fit/BPTT test or an actual study fit.

Snapshot/restore helpers clone tensor storage, including both analytic masks,
without detaching permitted latent graphs. Address patches require matching
source, timestamp and exact update-index history. They replace only addressed h
or posterior masks; they do not mutate recipient observation, V, decoder weights
or actual W. Current content changes may intentionally propagate to an assembled
next state queried at another address; tests preserve this inheritance while
checking nonpatched decoder content.

The toy suite checks exact restored current/next predictions and storage
isolation. Full restored policy equality, actual-W restoration and physically
reached donor prefixes require the later runner/verifier. Core patch helpers
cannot certify a donor's command/completion hash history. The amended requirement
remains: execute donor commands 0..t−1 but update only from W1..W(t−1)/commands 0..t−2,
with W(t) concealed at decision t. A patch is an explicit selective intervention;
its donor history need not match the recipient observation history.

## Analytic ambiguity and native use

The analytic source enumerates the six symbol-to-bandwidth permutations. Each
revealed transition compares the declared T(V) family with the revealed selected
map at the unchanged atol 1e-6, rtol 1e-5. It intersects survivors for that address
only, records revealed-symbol masks and rejects zero survivors. Uninformative
V retains multiple/all mappings; no arbitrary assignment is forced. Decoding
uses an equal-weight float64 mean over the surviving maps and then returns
float32. It receives no evaluator mapping or bandwidth truth. Primary H is
applied to that predicted mean; secondary H-integrated-over-mappings records
remain a later collector/runner requirement, enabled by the exposed family and
survivor masks.

NativePredictionReadout copies/freezes the retained H and its native coordinates.
Raw estimated maps are passed itemwise through that H before role fields are
computed. It does not substitute H(actual W) for H(A). Choice consumes the three
prospective queried-address fields, selects the queried role's maximum across
all 16 sites, and chooses the greatest per-symbol maximum with first-symbol tie
breaking. Both the content choice and confidence-only duplicate call the same
choose_from_maxima function; numerical choice parity is structural and explicitly
asserted by the proposed toy tests. Actual grid-cell actions remain the separate
native actual-W Controller's responsibility.

## Toy disposition and remaining boundaries

There are 13 proposed tests, using handcrafted tensors and freshly initialized
toy weights. Static inspection covers architecture/RNG handling, normalization,
current/next assembly, gradients/locality, timestamp rejection, bad inputs,
address equivariance, donor patch/restore, analytic ambiguity/elimination, native
itemwise readout and scalar-symbol parity. The suite invokes no renderer,
archive/checkpoint loader, optimizer or Reporter. Its first execution may be
approved within this bounded scope once the coordinator preserves the exact
source and this first review; all runtime failures must be retained before repair.

Collector, training/loss/unroll implementation, common/donor history manifests,
split sealing, registration, source-indexed ledgers, immutable fitted artifacts,
actual policy forks and independent numerical reconstruction remain separate
unreviewed components. This core approval releases none of them.

Controller is not Reporter. A learned delayed map predictor, an equally informed
ordinary process predictor, analytic calibration and scalar choice parity are
legitimate technical/access explanations. Architecture, H(A) and accurate state
bookkeeping cannot by themselves qualify literal predicted content, presentation
manner or attention model A as a source of qualia. No fitted content or presentation
claim is made by this static milestone. All prior negative semantic endpoints
remain, and the full project goal is unachieved.
