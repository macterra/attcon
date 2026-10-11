# Independent mechanical review after saved-restoration correction

2026-10-10 local date. Follow-up by `/root/attention_source_bridge_critical_review`, limited to the root's mechanical correction and one execution of the exact unchanged independent toy fixture.

**Verdict: the reported accepted-input exact-restoration defect is fixed.** All nine unchanged independent tests pass. The original first pass remains eight passes and one failure; its fixture, log, review and metadata are byte-identical to saved commit `c3699e2`.

The corrected source has SHA-256 `659a90bf1c52b8ae82a28bf47255068ad81447e9dba8953add1bac97d9d8a13e`, matching the root's specified source. `restore_saved` now copies each validated immutable `SpatialBuffer` using `dataclasses.replace` instead of round-tripping its tuple values through a dtype-constrained tensor. For the formerly failing valid direct float32 buffer containing Python `0.1`, the restored complete state now equals the saved state. Tensor values and exports remain equal. Factory-created saved states with completed history also continue to restore exactly.

The same unchanged fixtures retain their successful ownership, fixed V/logit/identity preservation, selected-address isolation, unseen refusal, current/prospective time separation, analytic mapping-oracle disclosure and matched endogenous/exogenous output checks. These are toy mechanical results, not a general floating-point theorem or expanded historical-provenance validation.

One fixture execution was performed after correction; output is preserved in `run_after_correction.log`. No fixture, author source/test, original first-pass artifact, prior review or frozen protocol was edited. The saved commit was read through the root-provided milestone checkout `/tmp/attcon-review-milestones-vvjs166i`; the original workspace `.git` does not contain this milestone commit. Byte comparison used the actual saved Git blobs, not only the correction receipt's assertion.

The mechanical defect is resolved within the tested accepted-input scope. The prototype still implements no learned A, unknown-mapping learner, encoder extraction, production evaluation or Reporter. This correction supplies no semantic/presentation approval and does not revise the prior source-of-qualia verdict. No checkpoints/worlds, training, learned-model inference, reports or additional fixture runs occurred.
