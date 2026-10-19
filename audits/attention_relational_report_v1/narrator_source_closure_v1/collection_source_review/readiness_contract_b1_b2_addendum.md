# Preserved first-review corrections B1 and B2

The original readiness contract and blocking review remain unchanged. No tests
were run by the implementation author before or after these corrections.

B1 adds required CLI arguments `--root-terminal-attestation` and
`--root-terminal-attestation-sha256`, independently supplied before the prepared
manifest is read. That manifest contains full narrator text and reference labels;
opening it is data exposure, not a metadata-only prerequisite.

The metadata-only precheck binds the pinned original runtime, root's actual
terminal observation for session 5286, all 47 current runtime-file hashes, the
actual first execution receipt and log, and normal completion or interrupted
terminal metadata. A missing, false, mismatched or nonterminal prerequisite stops
before the prepared manifest can open. The interrupted terminal must bind the
same original observed tool session. These checks do not themselves poll a tool.

After that barrier, the manifest's root-receipt descriptor must equal the
separately supplied descriptor exactly. Its first-execution hash and resolved
path must match the independently verified actual execution. The frozen
preparer's complete actual-input/source recomputation still runs unchanged
before chunks or first judgments are opened. The collection records the separate
root receipt among its input dependencies. Launch/outcome context contracts and
zero-candidate/whole-chunk retention policies remain unchanged.

B2 constrains `represented_time_paths` to individually resolved strict integer
fields named `represented_time`, `record_time`, `feedback_through`, `shown_at`,
`archived_origin_time`, or `decision_time`. Boolean, floating-point, wrong-named
scalar, missing and container paths fail. Each retained timestamp records its
exact path, value and original `field_kind`, with syntax verified and its semantic
link to the operand explicitly unverified. A record/feedback/archive/display
time is never silently renamed represented time. Source meaning, ownership,
operand sufficiency and arithmetic truth remain independent reviewer questions.
