# Reconstruction source corrections after first static rejection

2026-10-11. Same model team, separate review allocation and inherited hypothesis
context; no independent family/human endorsement. Source-only corrections, never
imported or executed. AST/syntax compilation passed for verifier and13 fixtures.
Original records remain unchanged: verifier/design milestone31d57ef, original
fixture e6102693 preserved16aa6a4, and first V1/V2/F1 rejection preserved/pushed
`91038f07036d228b1b65faf2ca008ad7ddb5c764` before main corrections were authorized.

Corrected verifier SHA-256:
`11b821bea207f4ef8250fa589148777df9d600bb3dffd01ef600cf4f78a423ab`.
Corrected fixture SHA-256:
`365103a3973571f88f5058bb205ee6eeae7f7be0498aeb8d9ee386b3f1afa198`.
Prior verifier SHA `0332633eaef01755b458ec6c7b21b9373f1e949f79b295a3d2471c4141737902`
and prior fixture SHA `e610269349758b19352ba010fd3b0bf7dc6fe48103f08c5ff9cd7381a6cd0c3a`
retain their original first-review status. No executed failure or runtime pass is
being relabelled. Preparation metadata.json is not an actual phase report; future
fit/dev/heldout reports must use separate new exclusive paths.

V1: Serialize before writing. If strict JSON rejects primitive audit evidence,
retain all partial checks/errors, the original verification traceback and a
separate serialization traceback. Recursively tag nonfinite floats with explicit
type/value/finite fields; keep finite primitives unchanged. Bytes/tensors/unsupported
values have labelled encodings rather than being silently removed. A serialization
fault forces failed status and adds its error without deleting any original check.
Large integers that cannot be represented by the configured decimal JSON conversion
receive an explicit hexadecimal integer tag. No retained archive is changed.

Primary report creation remains exclusive. Write/flush/close faults retain a separate
exclusive PATH.writer_failure.json with the original writer error and complete first
report evidence. The partial primary remains untouched; no retry/overwrite occurs.
If even secondary retention fails, full evidence plus both errors is sent to external
stderr capture. Disk/log failures can make durable retention unavailable; this is
recorded failure, never a passing audit. An existing primary path is rejected before
execution exactly as before.

V2: The existing named Adam step checks use optimizer_step_is_exact_300. Numeric
count must equal300 exactly; a fractional deviation that falls inside ordinary
float tolerance is rejected. No new float dtype, threshold or scientific gate is
introduced. All other ATOL1e-6/RTOL1e-5 comparisons are unchanged. Existing step-check
names/counts remain; evidence now marks zero-tolerance exact numeric count.

F1: Remove the invalid .tensor() call from role_fields(item), whose return is already
a Tensor. The ten original fixture intents remain. Three prospective regressions
now cover typed nonfinite evidence with original failure/all checks retained,
fractional Adam-step rejection while generic numerical tolerance stays unchanged,
and partial writer-fault retention with complete exclusive sidecar evidence.

All13 fixtures remain handcrafted tensors/fresh weights or mocked CLI/Git streams.
Declared configuration: four CPU threads, deterministic algorithms. No native
archive/checkpoint, reserved world, renderer, optimizer/backward, registered fit,
actual Git subprocess or Reporter occurs in their planned execution. Fixture
FitNormalizer.fit computes numerical moments only. Independent equations share
PyTorch CPU primitives and the same model team; these are arithmetic/software
tests, not independent-family validation, fit-trajectory proof or semantic evidence.

Remaining: separate scoped review/publication of these exact corrected bytes and
explicit authorization before one preserved first fixture execution. Full reviewed
source closure/registration and preserved actual phases are later prerequisites.
Controller is not Reporter; engineering validation cannot establish qualia or
presentation. The full scientific goal remains unachieved.
