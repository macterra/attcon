# First-pass prefix contract-QA review

Separate-context read-only review by `/root/goal_evidence_audit`, 2026-10-19.
No live resource-v2 outcome, quote, verdict, contract result or reference label was
read. This source was not executed.

The operational design is defensible: explicitly post-launch and pre-contract
analysis, fixed first 12 scheduled positions, at most three valid annotations as
a budget guard, no representative-sample or theory-pass claim, and no automatic
stop. Source recomputes the original full audit request inventory, uses strict raw
JSON decoding plus the frozen response parser and validator, and emits no quote,
verdict, fidelity or reference endpoint labels. The fixed prefix is narrow and
correlated; it supports only this operational decision. A stop still requires
its separately preserved/reviewed addendum and full missingness preservation.

Two bounded changes are required before execution approval:

1. Validate the live output root's first_batch_claim.json against the exact released
manifest SHA, publication commit, stage and complete request inventory, with
retry_performed false. Request hashes alone do not establish the resource/settings
lineage of files in the live root.
2. Freeze the result path in the plan and enforce it in source. Create an exclusive
once-only QA claim after all 12 required outcome paths exist and before reading
any of their raw contents. An arbitrary fresh --output path plus exists-check
allows duplicate operational assessment of the same live data.

These are provenance and once-only execution fixes, not changes to the fixed
prefix, threshold, contract, scientific endpoints or original audit outputs.
Source may handle secondary outcomes separately if desired, but must never repair,
select favorable storage or treat missing primary files as successful outcomes.

Disposition: approve the bounded operational design in principle; exact source
execution approval withheld until these two source/plan bindings are corrected.
