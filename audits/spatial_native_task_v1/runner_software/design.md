# Spatial native task v1 runner: software-only first pass

2026-10-10. Development task/content semantics only. This runner contains no
attention model A or Reporter, and cannot qualify presentation or qualia.

`scripts/run_spatial_native_task_v1.py` has separate `--register` and `--run`
entrypoints. `--help` and in-memory compilation import only the standard library;
registration reads and hashes source, review and checkpoint bytes without loading
models or generating worlds. This software first pass performs help/compilation
only, once. It does not test numerical execution or independently approve source.

Registration requires an independent JSON review receipt with exactly:

```json
{
  "source_review_status": "approved_for_development_only",
  "status": "approved",
  "receipt_scope": "spatial_native_task_v1_preexecution_source_review",
  "independent": true,
  "reviewer": "independent reviewer identity",
  "approved_sha256": {"repository-relative/path": "sha256"}
}
```

The complete approved mapping must equal the runner's required source inventory
plus every optional `--source` binding. Required sources include this runner,
the task module, primitives, FrozenSensor, checkpoint-path helper and transitive
local import dependencies, protocol and all three runner software records. The
receipt cannot bind itself; registration separately hashes its exact bytes.
Additional review text, software tests and preserved failed reviews can be bound
through `--source`. The source and receipt must exist byte-for-byte in the supplied
full source commit of the read-only preservation checkout.

`--register` also declares fixed `--execution` and `--output-dir` paths and creates
the registration exclusively. `--run` requires those same paths and verifies the
registration is committed in a full commit descending from the reviewed source
commit, with unchanged source/receipt/registration bytes. The execution claim is
created exclusively before runtime imports, checkpoint loading or generation.
The output directory, archive files, log, results and completion/failure records
also use exclusive creation. Changing output paths cannot reopen one registration.
A failed attempt retains its claim, log, partial seed archive and failure receipt.

The exact envelope is three seeds, 128 fit and 64 diagnostic worlds per seed,
four items per world, frozen 24×4×4 maps, 899-parameter native head and the
227-parameter raw-grid alternative. Native and raw initializations use
`220000000+seed` and `221000000+seed`. Each head receives 500 full-batch CPU float32
Adam updates with four threads and the protocol optimizer. Standardizers use
fit-only channel means and population deviations; their immutable tuple values
are retained along with initial/final weights, fit maps, targets, logits and every
loss. No diagnostic selection, stopping, retry, layer search or fallback exists.

Each primary action consumes a completed `AddressedState` through
`NativeController.consume`; red and blue target calls use only its resulting
snapshot for the selected address. Initial states have all four observed items,
V=W, literal `role_pair`, preserved original orientation raw logit and delivery/
completion/time zero. Every bandwidth starts from that saved state with issue
zero/completion one. Actual logits/fields are checked against batch computations
at atol 1e-6/rtol 1e-5, but archived actions come from actual native snapshots.

The primary ledger has 2,304 item/operator opportunities and 4,608 role queries.
Saved-initial T1 sham, matched exogenous T2 and restoration after T2 each have
768 items/1,536 role queries. Address roll +1 and flattened-site roll +1 are
executed on every primary opportunity, each adding 2,304 items/4,608 role queries.
All action/tie/coarse outputs and input maps are retained or exactly reconstructible
from preserved V/W and schedules. Address remapping carries event history as well
as retained item metadata. Synthetic site controls retain scalar invariance checks
and field/logit equivariance; action equivariance is asserted only at unique maxima.
Uniform/unchanged/tied cases remain in the ledger. Isolation comparisons alter
retained identity, raw logit, source V, nonqueried W and evaluator-only role truth;
their actual logits/fields are archived without emitting additional action queries.

Trace archives preserve pixels, V, actual W, truth, fit targets/coordinates/weights,
full native logits/role fields/actions/ties, original and transformed category and
orientation outputs, standardized hidden controls and uncalibrated orientation
sigmoid coordinates, and frozen encoder/orientation weights before/after. A partial
archive retains completed records if execution fails. Raw-grid feasibility outputs
are retained separately. One bounded gzip torch archive is created per completed
seed; a failed seed instead receives a partial archive. SHA256 receipts accompany
archives/results/logs. The first log records every fit loss and runtime versions;
completion/failure receipts record elapsed time.

Unconditional metrics preserve pooled and separate red/blue targeting, loss,
entropy/max/margin, expected columns, unknown coarse answers and ties. Bootstrap
indices are 256 resamples of all 64 worlds using `230000000+seed`; .025/.975
quantiles describe conditional world uncertainty only. Field change means maximum
absolute T1/T2 role-field difference greater than 1e-6. Correctly retained coarse
order requires both native T1 and T2 answers to match evaluator truth. T2's .25
fine ceiling and T4's .0625 fine accuracy/unknown coarse order are imposed policy
effects, never experienced blur. Engineering gates are exactly the protocol's
per-seed T1 fine .50/coarse .75 and raw fine .95, with no semantic acceptance gate.

The accompanying first-run log and validation JSON report only compilation and
help parsing. Independent preexecution source review and numerical reconstruction
are still required. No checkpoint load, world generation, fit or inference was
performed in this first pass.
