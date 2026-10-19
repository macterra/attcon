# First-review retention contract

This describes code awaiting root's independent source review and first fictional
fixture execution. No actual reference labels, narrator reports, source-review
packets or first judgments were loaded by the implementation author.

Launch JSON has exactly `schema`, `manifest_sha256`, `protocol_sha256`,
`output_schema_sha256`, and `contexts`. Its schema value is
`attention_relational_narrator_source_closure_collection_v1_launch`.
Each context has exactly:

- `reviewer_id`: `reviewer_0` or `reviewer_1`.
- `independent_context_id`: unique nonempty string across both streams.
- `chunk_number`: integer naming a prepared candidate chunk.
- `chunk_sha256`: exact prepared chunk hash.
- `first_artifact_path`: globally unique reserved first path.
- `fresh_context`: true; `first_pass`: true; `peer_judgments_supplied`: false.

Both streams must cover every candidate chunk once. A chunk contains at most four
narrator packets; candidate bindings remain inside their original narrator record.
With no candidates, the two streams both have zero launch contexts and no reviewer
call is manufactured.

Outcome JSON has exactly `schema`, `launch_receipt_sha256`, `manifest_sha256`, and
`contexts`. Its schema value is
`attention_relational_narrator_source_closure_collection_v1_outcomes`.
Each outcome repeats the launch's `reviewer_id`, `independent_context_id`,
`chunk_number`, and `chunk_sha256`, and adds exactly:

- `first_artifact`: null or `{path, sha256}`, naming the reserved first path.
- `terminal_attestation`: null or an object with `independent_context_id`,
  `chunk_sha256`, `observed_terminal: true`, `observation_kind` equal to
  `completed`, `failed`, or `interrupted`, and `observation_artifact: {path, sha256}`.

An observation artifact must retain a nonempty actual root observation. A live
handle, timeout, intention, or unverified lock file is not terminal evidence.
These bindings cannot make this collector observe liveness or context freshness.

A matching retained first artifact establishes readiness, including a malformed
first artifact that will fail the whole chunk. Without matching first bytes, an
actual bound terminal observation is required. A readable mismatched first file
is retained with observed bytes/hash, original declared descriptor and errors;
an unreadable first file retains its descriptor/error. Both invalidate their
whole chunks. No hash is repaired, no individual row is salvaged, and no first
judgment is replaced or gap-filled.

Production independently re-runs the source-closure preparer's original V3
terminal/all-47-files checks, complete reference/preparation/source/factual
recomputation, and exact prepared-source bindings before opening prepared chunks
or first review files. Root still has to make and retain the actual observation.

Structural retention checks exact source addresses/JSON value types, original
binding/status/feature/evidence inventories, individual metadata paths and valid
operand indices. They do not establish that operand inventories are sufficient,
that source-category judgments are correct, or that asserted calculations were
independently and correctly performed on those actual operands. Separate retained
calculation artifacts need independent source review. Supporting a justified
recorded absence or ambiguity does not become an explicit positive witness.

The complete 552-position source inventory and 684 pending pair dispositions
survive collection. All narrator fidelity remains unestablished. This collector
does not compute endpoints, perform property tracking, establish appearance or
qualia, launch reviewers, or make API/model calls.
