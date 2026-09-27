# Reporting confirmation v2: explicit unknown precedence

Frozen before new requests, 2026-09-27. V1 remains a failed full confirmation:
96/96 primary reports, 72/72 intervention pairs, and 24/24 restorations were
correct, but only 11/24 constant-state reports used null focality. The others
reported false on tied allocation. This is a prompt-contract failure, not evidence
for or against qualia. No v1 output or gate is changed.

Keep every v1 model, condition, criterion, scoring rule, style manipulation, and
request limit. Replace the prose rule for focality with explicit branching:
if there is more than one maximum, focal must be null; otherwise compare the
target allocation with the channel maximum to return true or false. Likewise,
equal command variation explicitly requires responsive_channel null.

Fresh contexts use seed+970010000 (v1 used seed+970000000), four episodes for each
confirmed model 901/911/921. The 240-attempt limit, 4096 output ceiling, model
snapshot, concurrency eight, and no retries remain unchanged. Maximum output
cost $1.96608; expected total below $2.25, typically much lower as seen in v1.
This consumes new report contexts without retraining or choosing A checkpoints.
The existing blinded rubric and unresolved report-character criterion are retained.
