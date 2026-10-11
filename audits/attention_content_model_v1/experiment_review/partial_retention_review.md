# E1r correction and eight engine fixtures: bounded static approval

2026-10-11. Separate same-AI-family review task with inherited context; no human
or independent-family endorsement. Source/text/hash inspection only. No imports,
compilation, fixtures, model calls, generation, production artifact loads,
fitting, Reporter or Git operations occurred in this review.

Exact reviewed bytes:

| File | SHA-256 |
|---|---|
| [Engine](../../../scripts/attention_content_model_v1_experiment.py) | `4f537c112c3660f0dbd60d72f751a806f83f00b05224d42e6e62699452db8b45` |
| [Eight fixtures](../../../tests/test_attention_content_model_v1_experiment.py) | `c338aa4513952e1f84c2e557d8b715c46e47ebaaac6e076affb34d7b27da50fc` |
| [Correction record](../experiment_software/partial_retention_correction.md) | `a975e51428c57d1bc09c2dbd53f92d416a6d838d203ea831a8ac43dbe530207a` |
| [Correction metadata](../experiment_software/partial_retention_metadata.json) | `05ef89ad8ccd127e1b9e09abf65018b3c5a396f850d5772ac898c822803b223b` |

**E1r is resolved statically. Approve one first execution of only these eight
bounded fixtures after preserving/pushing the exact source, records and review.**
Preserve the first exclusive log/results and failures before any repair or
separately labeled repetition. Actual registered fit/dev/heldout execution,
production registration and scientific qualification remain unapproved. This
review asserts no fixture result.

The engine adds references to the completed `source_update_visibility` and
`source_decoder_visibility` dictionaries, plus `actual_actions` and
`analytic_secondary`, immediately when those mutable collections are created.
Existing active list references remain. Thus completed sources' actual-call
records and accumulated shared/secondary outputs stay reachable from caller-owned
partial state when a subsequent source fails. Removing precisely the three added
lines, which retain these four collections, reproduces previous engine SHA-256
`09a5cbd3e6c7f3c37c65d2bdde35e1db40d65d35904c832d24d7c1533d887878`.
This independently confirms the engine change contains only the proposed retained
references: no model calls, equations, budgets, gates, tolerances or denominators
changed. Original source-design and both prior reviews/metadata retain their hashes.

The fixtures import torch and the reviewed modules and perform **toy numerical
forward evaluation**, unlike the earlier stdlib-only boundary suites. They use
two explicitly handcrafted maps, a manual fifteen-command schedule and manual
mapping/truth assignments. They patch only engine `WORLDS` to two; no reserved
`schedule`, collector, renderer, encoder, native archive or checkpoint loader is
called. Sources are freshly initialized and untrained; H is a fresh manually
set numerical head and the original-path control extractor is a stub. Four CPU
threads and deterministic algorithms are set before fixture construction.
Normalization uses only these handcrafted V maps and is not production fit data.

Six tests share one full fifteen-decision toy evaluation. They check named hard
gate sets/shapes and unchanged source weights; exact successful compaction plus a
corrupted fork retaining raw first evidence; candidate/latent/feedback hashes and
fixed donor update schedules; analytic masks/donor restoration/terminal exports;
saved paired bootstrap draws and independently enumerated inherited errors; and
full predetermined donor field/scalar/factual records. These are useful component
fixtures, not independent reconstruction of native or trained A evidence.

The seventh test performs a separate toy evaluation and injects an analytic
prediction failure at t=1 after its t=0 output. It requires both completed learned
sources'86 update records and27 decoder records, all9 shared action batches,
the existing analytic secondary output and the incomplete failing prediction
record in partial state. This directly exercises the E1r handoff defect without
retrying an actual execution.

The eighth test runs the fixed300-iteration fit **control flow** with both learned
architectures replaced by scalar fake sources and Adam replaced by a no-op step
counter. Real autograd backward and gradient clipping run on a zero-derivative
scalar; backward itself is not mocked. No registered GRU/decoder is trained and
the no-op optimizer modifies no parameter. The second optimizer construction is
forced to fail; completed A weights/optimizer count and ordinary initial state
must remain. Afterward the test deliberately changes the fake scalar by1 to
check that the retained completed snapshot is independent storage. Report that
toy mutation accurately; it is not an optimized parameter update or evidence
that real300-update fitting passed.

Pending complete executable review must still bind runner, engine, permission
schema and finalized independent verifier, validate actual native/input/phase
bindings and reconstruct the complete registered evidence. The fixtures do not
prove fit trajectory, production observation isolation, literal content, manner,
attention specificity or conscious-report correspondence. Controller remains
distinct from Reporter; no reporting release follows. Negative and null outcomes
remain legitimate, with the scientific goal unachieved.
