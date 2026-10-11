# Independent numerical reconstruction design

2026-10-10. Prepared by `/root/spatial_native_task_numerical_auditor` before any
development results. This task allocation is separate from the source/runner
authors, within the same AI model team and with inherited project/theory context.
It is not an independent model family, human endorsement or empirical replication.

The owned verifier is `scripts/verify_spatial_native_task_v1.py`. This design and
the program are preparation only. No development archive/checkpoint is loaded,
world generated, fit performed or model inference run during preparation. Only
static source/protocol/review reads and syntax compilation are permitted here.
The coordinator must notify this reviewer that first actual results are committed
before the numerical pass. The first pass will preserve stdout/stderr unchanged
in `first_run.log` and its exclusive JSON result in `metadata.json`, followed by
`review.md`. Any subsequent verifier correction must preserve that first finding
and receive a separately labelled output. There is no automatic fit repair,
new source packet, fallback, retry or production-source edit in this review.

Preparation command preservation: the first syntax-check shell invocation used
`python -m py_compile scripts/verify_spatial_native_task_v1.py` and returned exit
127 (`python: command not found`) before any Python code executed. The subsequent
`python3 -m py_compile` succeeded. This is an environment command correction in
pre-output preparation, not a development or first archive verification failure.

## Independent reconstruction

The reference implementation imports torch but neither the production runner nor
its numerical helpers. It independently reconstructs the encoder convolution,
ReLU/pool and retained original category/orientation pathways from archived frozen
state dicts; compares original maps, raw grids and controls; indexes disjoint T2/T4
blocks and explicitly sums in float64 before casting back to float32; checks exact
T1 copies and repeated block values; and applies H with archived trained weights
and the archived fit-only coordinates. Fit channel means/population variance are
independently recomputed from the 128 archived fit worlds. No optimizer is created.

Occupancy CE uses a stable float64 log partition and selected class logit, with
equal weight on every site. Role fields use float64 role/background subtraction,
explicit exponentiation and normalization, then return float32. Exact targets,
all tied cells and first-row-major actions are independently reconstructed from
the emitted fields after checking them against reference H outputs. This keeps
roundoff tolerance from concealing a tie or redefining the emitted action. Coarse
answers use expected columns and the fixed 1e-6 unknown band, separately from
evaluator order. Full occupancy logits, fields, CE and per-role actions remain
inspectable, not merely aggregate pass flags.

Every record in all 4,608 primary role actions and scheduled duplicate controls
is checked, including unchanged, wrong, tied and erased cases. Sham/driver/restored
evidence and actions compare exactly; operator history preserves endogenous versus
exogenous provenance. Outer-address checks carry the same original item to the
declared new address. Site rolls independently reconstruct W, logits, fields,
map statistics and original pathways. They preserve entropy/max/margin within
tolerance; action equivariance is tested only for unique maxima. T4 ineffective
rolls and all tied cases remain in denominators. Exact tie counts, zero T2 margins
and T4 uniform forced localization are measured directly.

The production Controller is reenacted separately from reference math, solely to
test the declared input boundary using archived H and inputs. Every scheduled
opportunity compares actual logits, fields and actions when retained V is zeroed,
identity replaced, retained raw logits changed and nonqueried W zeroed. Archived
isolation outputs are compared when present. An evaluator-truth argument must be
refused. This explicitly labelled reenactment is not a source-independent consumer
implementation and provides no evidence for attention prediction or qualia.

## Aggregation, tolerances and provenance

Reference summaries expose red/blue and pooled targeting, coarse accuracy,
occupancy CE, ties, entropy/max/margin and expected columns. Whole-world means and
256 bootstrap resamples use seed `230000000 + encoder_seed`. Independent sorted
linear interpolation reconstructs .025/.975 quantiles. The T2 change/retained
conjunction uses the fixed >1e-6 field-change threshold and requires both T1 and
T2 answers correct, across all 768 items. The same fields also give an exact-change
frequency. The verifier independently recomputes the per-encoder .50/.75/.95
engineering gates and the all-model verdict without choosing a favorable subset.

Float32 CPU tensors/four threads reproduce the specified H/source arithmetic;
field/CE/statistics/uncertainty reference calculations use float64 where declared.
Numerical comparisons report actual shapes/dtypes, maximum absolute differences,
unequal elements and tolerance violations, with atol=1e-6 and rtol=1e-5. Copies,
repeated blocks, identities of controls, target classes, query keys, tie lists and
discrete actions are exact comparisons. CE accumulated in float64 is a numerical
reference checked within tolerance against archived float32 CE.

The verifier binds current source/checkpoint/review hashes, review scope/status,
registration and exclusive execution paths, completion/archive/results/log hashes
and explicitly provided preservation commits using read-only local Git. It loads
registered checkpoint values only in the authorized post-preservation numerical
pass to compare them to both archived before/after source states. It checks full
128/64 whole-world inventories, namespaces, layout targets and per-role balance,
the six recorded 500-update traces, initial/final CE and final H outputs. It cannot
prove the optimizer trajectory without replaying fits, cannot rule out transient
source-weight mutations between equal snapshots, and cannot rule out unrecorded
external executions. The receipt supports the one recorded registered attempt
within the reviewed program, with these limits stated explicitly.

Conditional bootstrap intervals describe worlds under these fixed models and
shared deterministic layout support. They do not establish uncertainty over people,
model families or unseen position-layout combinations. All outcomes are DEVELOPMENT
task/content semantics; no passing result qualifies presentation manner, identifies
attention model A as the source, or establishes qualia. The full project goal remains
unachieved. No Git mutation, GitHub/network operation or Reporter call is authorized
or performed by this numerical reviewer.
