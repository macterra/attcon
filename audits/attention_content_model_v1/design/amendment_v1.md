# Attention-content model v1: prospective amendment v1

2026-10-11. Static proposal clarification before any A implementation or outputs.
The amended exact contract is
[ATTENTION_CONTENT_MODEL_V1_DESIGN.md](../../../docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md).
Its SHA-256 is
`b64062f3c07aa1e29b8fd5d3f50bcc04b2188edfe2a8d6b26cafb4a9d04ff2af`.
The original design was preserved/published at `1af48e9` and the independent
review at `52a28aa`, as supplied by the parent task; this amendment performs no
Git inspection or mutation. Original design/design.md and metadata.json and both
design_review files remain untouched. This new record does not rewrite them.

The [preserved review](../design_review/review.md) approved the bounded direction
and requested C1–C4 before executable freeze. Each is addressed prospectively:

| Issue | Contract disposition |
|---|---|
| C1: donor indices/restoration | At decision t=12/13/14 execute donor commands 0..t−1, update source state only from W1..W(t−1)/commands 0..t−2, conceal W(t), preserve exact masks/update hashes; restore original recipient state/estimates/policy before feedback with exact equality. Address-local decoder isolation is distinguished from intentional inheritance into assembled A_next. Actual-W restoration remains separate. |
| C2: comparator/replay ledgers | Each primary prediction and patch ledger has an explicit A/ordinary/analytic source index; shared actual targets, derived confidence/analytic records and same-weight aliases are not new samples. Donor replay is 78 executor transitions and 72 permitted feedback events per world: 22,464 shared heldout transitions, 20,736 events per source, 62,208 total source-update operations, and 1,728 scored/restored pairs per source. |
| C3: sealing and hashes | Pre-run registration binds frozen input weights/normalizers/extraction/source closure/permissions; fit-only execution produces immutable fitted hashes before dev release; fixed dev produces an immutable receipt before heldout release. Both other splits remain sealed during fit. No extra fit, initialization, result-dependent permission loop or source-threshold change is permitted. |
| C4: predicted literal content/export | Fixed dev diagnostics independently assess H(A), H(W), truth, wrong/ambiguous estimates, crossings, fields/scalars and continuous distance to actual operator-family maps. Literal-source-content, exact export bookkeeping and presentation manner have separate dispositions; accurate wording or H(W) utility cannot qualify H(A). No new numeric support/manner threshold or readout training is added. |

The unchanged trainable architecture has 197,328 parameters, 96 fit/96 dev/96
heldout worlds per encoder, three existing encoders, fifteen transitions per
world and one 300-update Adam fit for A and ordinary separately. No new worlds,
training losses or post-fit performance thresholds are introduced. The final
policy queries can be fully analytically calibrated; early/calibrated errors and
mapping ambiguity are now explicitly tabulated rather than recast as online
mapping discovery.

The supplied native numerical diagnosis explains all 106 original failures as
source/reference-kernel arithmetic, but corrected full revalidation has not yet
occurred. This record has not inspected that numerical evidence. Corrected full
native validation and independent reconstruction remain prerequisites for actual
A execution; diagnosis alone does not release it. Broad user authorization is
retained, with no invented user-reapproval requirement.

Disposition: C1–C4 have proposed contract resolutions, **pending independent
amendment review**. No implementation is frozen, registration completed, execution
released, literal content qualified, manner qualified or goal achieved by this
record. The original negative semantic endpoint and native study gates remain.
Modeler refinement remains an optional stronger extension, not a narrow-report
gate. No checkpoints/results loads, world generation, fits, inference, Reporter
calls or Git operations occurred. Only the exact source document and these two
new amendment files were changed.
