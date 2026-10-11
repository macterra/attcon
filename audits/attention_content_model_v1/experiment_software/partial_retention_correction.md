# E1r retention correction and first fixture source

2026-10-11. Prospective source-only change after the parent reported preservation
and publication of the [E1–E5 follow-up review](../experiment_review/after_corrections.md)
at `c994095`. The earlier engine hash
`09a5cbd3e6c7f3c37c65d2bdde35e1db40d65d35904c832d24d7c1533d887878`,
[correction record](after_corrections.md), its metadata, original design and both
original review files remain historical records and are unchanged.

The corrected [engine](../../../scripts/attention_content_model_v1_experiment.py)
SHA-256 is `4f537c112c3660f0dbd60d72f751a806f83f00b05224d42e6e62699452db8b45`.
E1r attaches four existing mutable collections to caller-owned evaluation
progress immediately upon creation: `source_update_visibility`,
`source_decoder_visibility`, `actual_actions` and `analytic_secondary`.
Completed sources' update/decoder calls and accumulated shared action/analytic
secondary outputs therefore remain reachable if a later source fails. Existing
active lists remain attached. There are no new calls, gates, opportunities,
thresholds, weights, optimizer changes, budgets or semantic claims.

The new [fixture source](../../../tests/test_attention_content_model_v1_experiment.py)
SHA-256 is `c338aa4513952e1f84c2e557d8b715c46e47ebaaac6e076affb34d7b27da50fc`.
Its eight tests use two handcrafted numerical maps, a manual fifteen-command
schedule, manual evaluator assignments, fresh untrained learned sources and a
fresh numerical local H. The control extractor is explicitly a numerical stub.
Only the engine's `WORLDS` constant is temporarily patched to two; no reserved
schedule, renderer, encoder artifact, native archive or checkpoint is used.
The fixture module setup explicitly sets four CPU threads and deterministic
algorithms before constructing any fresh source or evaluating it.

Six fixtures share one complete fifteen-decision evaluation. They cover exact
named gate inventories/shapes and weight nonmutation; failed compaction retaining
original raw arrays; actual source candidates/latent/update manifests; analytic
seen masks, donor exports and restoration; saved paired bootstrap draws and
independently reduced inherited error; and unconditional donor field/scalar/
factual diagnostics. Negative compaction deliberately corrupts one fork value
and demands a false gate plus raw evidence preservation.

The E1r fixture injects an analytic prediction exception after its first primary
decision and demands retained completed A/ordinary update/decoder collections,
all nine shared actual-action batches, a nonempty analytic-secondary collection,
and the incomplete failing-call record. It makes no retry. The fit handoff
fixture replaces both registered learned architectures with scalar fake sources
whose derivative is zero and replaces Adam with a no-op counter. It runs the
unchanged 300-iteration control flow, then injects the second optimizer constructor
failure; it demands a completed immutable A snapshot and retained ordinary initial
state. No real optimizer is constructed and no parameter is optimized. The fake
source intentionally provides no evidence about GRU fitting or task performance.

Only Python 3 syntax compilation was performed, successfully, without importing
the engine or executing any fixture. An initial `python` syntax command failed
because that executable is absent; the subsequent explicit `python3` compilation
passed. No actual software test outcome exists yet. Parent preservation and
independent static review precede the first fixture execution and its exclusive
unchanged output record. Production execution still requires complete integrated
review, registration and phase release; source content/manner remain unqualified.
