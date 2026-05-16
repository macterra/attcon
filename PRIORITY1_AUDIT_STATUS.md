# Priority 1 Audit Status

This note records the disposition of the bounded claims after adding the Priority 1 audit machinery. It should be read as a working audit note, not as a replacement for the roadmap.

Smoke-check target:

- checkpoint: `outputs/tune_prob_035/experiment.pt`
- config base: `configs/tune_prob_035.yaml`
- audit mode: reduced probe batches and epochs for a fast local sanity check

## Survives Current Smoke Audit

- [x] Stage 4B hidden self-model probes survive the matched-capacity previous-observation audit at smoke-test scale.
  - Hidden cell BCE advantage over matched baseline: positive.
  - Hidden target BCE advantage over matched baseline: positive.
- [x] Stage 6A structured report probes survive the matched-capacity observation audit at smoke-test scale.
- [x] Stage 6B uncertainty/allocation-error probes survive the matched-capacity previous-observation audit at smoke-test scale.
  - Current wrong-candidate, wrong-candidate history, revisit-under-unresolved-search, and allocation-error positive-recall advantages are nonnegative against the matched baseline.
- [x] Explicit negative controls are now first-class evaluator outputs.
  - Feedforward-summary, shuffled-feedback, and high-capacity observation-only controls are reported under `negative_controls`.

## Requires Downgrade Or Caution

- [ ] Stage 7 local opaque-token reportability should not be promoted beyond bounded support on the basis of the new audit alone.
  - The smoke audit preserved a memory-content advantage over the matched-budget observation-only baseline.
  - The stricter joint matched-budget audit did not pass in the smoke run because joint accuracy advantage was not positive.
  - Current disposition: keep the narrower Stage 7 claim focused on faithful current/remembered content, and require a full regenerated report before claiming the whole structured report schema survives matched-budget auditing.

## Newly Instrumented But Not Yet Full Robust Support

- [x] Complete zero target-attention shaping is now included in the reduced-shaping sweep as weight `0.0`.
- [x] First-class comparator-system runs are now available under `comparator_systems`.
  - Static/feedforward comparator.
  - Recurrent feedforward-summary comparator.
  - Matched transformer-style feedforward comparator.
  - Large-LM-without-loop proxy via observation-only Stage 7 reporting.
  - Trivial uniform regulator.

## Current Claim Boundary

The Priority 1 work strengthens the evaluation harness, but it does not create robust Stage 8-relevant support by itself. Robust support still requires full regenerated reports, multiple seeds/checkpoint families, comparator failures under the full suite, and cross-architecture/cross-benchmark replication.
