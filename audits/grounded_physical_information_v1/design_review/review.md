# Grounded physical-information diagnostic: pre-output protocol/code review

2026-10-10. Reviewer `/root/theory_endpoint_gap_review`. Independent source review within the commissioned AI workflow; continuing the earlier endpoint/design-review context. No production checkpoints loaded, production registration/attempt reserved, production inference executed or Reporter outputs generated/rated by this reviewer.

## First-pass disposition: material numerical gate issue

The physical-only objective, source boundaries, fitting/evaluation design and archival machinery are appropriate for this bounded diagnostic. **The original reviewed implementation requires correction before production:** its bitwise row-permutation assertions fail a separately authored calibrated toy fixture using nonidentity shared normalization. The 11 supplied synthetic fixtures pass but use identity normalization in their model fixture. Exact inverse restoration passes in the independent fixture; the failing operation is row-equivariance under reordered floating-point inference.

Original source hashes are fully preserved, with source text, in `metadata.json`. The script reviewed at this first pass is `e509d487d4fe3c51ec795e9038cad775da946e7cc0aad76714620d224bfaf8e2`; tests are `2a2ad443613ae0fab9f50d27b2af5c7663579367200f8fe4b69abb0a38816414`. The protocol is `b1b91bd2e78b18172bd40afa549c3e6403b16be879555a7203a1dd1c83be4629`. The wrapper is `be74ce53852a7e297ef9877932f123266a6aeba31211059690e7527a7287aa32`.

The independently authored [additional_fixture.py](additional_fixture.py) constructs only random toy sensory tensors and random fixture networks, applies handmade outcome calibration and shared nonidentity normalization, fits toy external readouts, then tests the declared intervention. It generates no worlds and loads no checkpoints. The unchanged [first-pass terminal output](additional_fixture_first_pass_output.txt) records the complete failure and numerical measurements.

| Permuted source | Maximum absolute gap | Maximum relative gap, denominator floored at 1e-12 | Bitwise inverse restoration |
|---|---:|---:|---|
| risk_hidden | 4.76837e-7 | 6.68147e-6 | Pass |
| current logits | 4.47035e-8 | 3.23845e-7 | Pass |
| refresh logits | 1.78814e-7 | 5.25191e-7 | Pass |
| calibrated current | 7.24447e-9 | 2.02870e-8 | Pass |
| calibrated refresh | 2.81574e-8 | 8.38207e-8 | Pass |

These gaps are consistent with float32 batched linear arithmetic, not a location-specific normalizer: the normalizer is a shared 66-vector. The source is algebraically row-equivariant, but numerical execution need not be bitwise row-equivariant. This finding is pre-output fixture evidence, not a production failure or a post-result relaxation.

The parent proposed a fixed source permutation tolerance of `rtol=1e-5, atol=1e-6`, retaining exact input/inverse restoration. I approve this concept, subject to reviewing its implementation. It is a numerical source-integrity rule; it is not an appearance, performance or consciousness success margin. Each actual local feature view must separately satisfy that source tolerance.

For the shared linear physical decoder, avoid inventing an unrelated large output tolerance. With actual feature difference `dX` between changed and permuted-original source, feature scale `sX`, physical weights `W`, and target scale `sY`, an elementwise propagation envelope is `abs(dX)/sX @ abs(W) * sY`, plus a preregistered float64 rounding floor of `1e-12`. Compare each decoded difference with its own envelope and retain discrepancies/bounds, not merely a pooled maximum. This assumes the identical shared fitted transformation in both paths. In the toy fixture, every element meets this envelope plus the floor:

| Local physical readout | Maximum actual difference | Maximum propagated envelope |
|---|---:|---:|
| risk_hidden | 3.56040e-7 | 7.44459e-7 |
| outcomes_only | 2.78301e-7 | 3.78914e-7 |
| rawlogit_outcomes | 2.77504e-7 | 3.78263e-7 |
| sensory_hidden / observed_only | 0 | 0 |

The envelope is an instrumentation check under measured source roundoff. It must not excuse a source discrepancy that fails the independent source tolerance. Exact restoration and fixed-source forecast-patch invariance remain bitwise. Any revised source hash must be reviewed and pinned before registration; the original failure remains archived.

## Protocol/code correspondence

**Targets and actual delivery.** `physical_targets` computes population SD over all RGB/spatial values and an equal average of horizontal/vertical valid-neighbor squared-difference means before square root. It does not compute channel edges, padding differences, pure luminance/spatial SD, nominal contrast, blur or clarity. `collect` renders the actually delivered location on each of eight independently sampled commands, sends only pixels to the sensor, and overwrites only that location's memory and latest-image/time record. Last delivered image, hidden state, raw logit and target remain linked. Unseen placeholders are explicit and masked by observed status. No future branch is sampled or used.

**Evaluator boundary and frozen source.** World orientation/contrast/mask/phase/mapping determine rendering and remain separate archived truth for stratification. Structured/joint prediction receives only stored hidden64/raw-logit/observed rows and latest route-history rows. External probe targets are objective delivered-image statistics and the stored raw-logit sign. Auxiliary agreement means consistency with the stored decision, not correctness against world orientation, calibrated identity confidence, or perception certification. `load_frozen` selects only the calibrated instrument object; `FrozenSensor` uses the existing raw classifier and its normalization, excluding failed confidence calibrators. No new attention/sensory model is trained.

**Fresh RNG and splits.** World namespace 1.3e9, command namespace 1.4e9, noise namespace 1.5e9 and bootstrap namespace 1.6e9 are separate. The visual-seed multiplier leaves 100,000 between the three checkpoint blocks; split offsets are 1,000 and the maximum world count is 512. Arithmetic enumeration confirms disjoint world seeds and the world primitive's nuisance/label offsets across these blocks. Earlier outcome/source experiments use lower namespaces. Commands, noise and bootstrap use private generators; synthetic replay verifies global RNG preservation. No held-out target, correctness, extreme forecast or reference similarity selects worlds/rows.

**Probe layout and fitting.** The local views use one shared pooled-seen-row decoder and normalizer; global joint/route hidden vectors use four distinct location-specific decoders. This preserves the intended local address operation without demanding global-joint equivariance. Dimensions, decoder counts and parameters are reported; budgets differ and cannot establish architectural necessity. Feature and target means/scales use training seen rows only, with near-zero scales replaced by one. Centering supplies an unpenalized intercept. Ridge alpha uses only calibration standardized physical MSE, with first-index tie resolution. Auxiliary 0/1 sign labels share that alpha and produce a continuous uncalibrated score thresholded at .5; they do not influence physical alpha selection. Evaluation does not fit normalization, coefficients or alpha.

**Denominators, strata and uncertainty.** Every seen evaluation row enters the all-seen summary. Location, true-contrast, true-orientation and five fixed predicted-current-success bins are declared; true labels enter assessment only. The estimand pools seen rows, so worlds with more seen locations contribute more rows; uncertainty still resamples whole worlds, retaining repeated/within-world rows. Empty groups have explicit zero counts and missing scores. `r2_vs_train_mean` is `1-SSE/SSE(training-mean baseline)`, not conventional evaluation-mean R². Zero baseline energy makes that score undefined; it is serialized as null with valid-bootstrap counts. MAE/RMSE and macro-location results are separate. Undefined location/bootstrap values are excluded only from their explicitly named finite-value summaries, never converted to a pass. Intervals condition on frozen checkpoints/fitted probes and do not infer a model population. Paired error differences share bootstrap worlds and remain descriptive; the protocol correctly limits the question to fixed-readout recoverability.

**Interventions and restoration.** Memory rows are rolled at fixed route history; observed flag and sensory coordinates move together. This may contradict past actual delivery history. The code now marks the state synthetic, gives patched memory precedence only for current contents, and preserves historical deliveries. Delivery-command rows are patched after inference and post-success is recomposed; no hidden joint intervention is claimed. Source/memory restoration uses inverse rotations and predictor re-execution, not merely assigning saved tensors. The forthcoming numerical change must alter only permutation checks and leave those inverse/source restoration checks exact.

**Freeze, attempt and artifacts.** Registration exclusively creates a manifest/source snapshot and requires review files; source/input/review hashes, torch version and thread count are checked before execution. The wrapper is included among frozen sources. Production execution exclusively reserves `attempt.json` before collection; errors retain partial archives and terminal failure metadata. An existing reservation prevents retry. Each raw archive is exclusive and checked below 100 MB. The intended budget creates seven split archives per split, plus decoders/evaluation predictions/bootstrap/interventions: 25 per model, 75 total. Fits include chosen coefficients, candidate losses, normalization, row counts and auxiliary score semantics. Raw features, delivered pixels/noise, evaluator truth, predictions and intervention traces are retained.

Exact verification re-collects, re-infers, re-fits and compares archive tensors/containers, numerical summaries and hashes. It does not replace a production attempt or create another scientific result. Wrapper success/failure logs and terminal receipts are separate from scientific outputs; static wrapper review is complete, but I did not launch it against production. Source review files themselves must remain fixed after registration because their hashes are part of the manifest.

## Test receipts and their limits

At the initial reviewed code, the supplied 11-fixture suite passed in 2.211 seconds. It covers target definitions, latest-delivery alignment, RNG replay, unseen poisoning, training/calibration boundaries, physical-only penalty selection, shared/global layout, cluster metrics, synthetic intervention/source restoration, exclusive archives, fake registration/tamper detection, terminal reservation failure and a complete toy-sensor/random-model pipeline/replay. Fake registration/attempt tests run only in temporary roots with dummy inputs or mocked collection; no real study was registered or executed.

Four independent arithmetic fixtures also passed: scalar-loop SD/edge computation, unequal seen-count world/row weighting, undefined zero-baseline serialization and arithmetic seed-set nonoverlap. An initial version of the scalar fixture used a default float32 expected tensor against the float64 target and raised a reviewer-fixture dtype error; this was corrected to float64 without changing study code. That incidental checker failure is retained in metadata and is distinct from the substantive calibrated-model failure.

Passing synthetic fixtures establishes implementation checks, not physical retention in the frozen production models, presentation semantics or consciousness. The independently calibrated nonidentity-normalization fixture found the material issue that supplied tests missed; it must pass the reviewed pre-output numerical correction before final approval.

## Goal and exposure

The protocol and returned flags correctly keep appearance, participant data, new reports, source retraining, qualia and goal achievement false. A positive physical decoder result would qualify only bounded objective information recoverability; a negative linear result limits the specified probe/source, not every representation or experience. Source-aware fidelity, independently grounded qualitative presentation correspondence and selective attention-source evidence remain future work. Neither technical access prose nor this engineering diagnostic achieves the user's goal, and task necessity is not required.

This agent's original spawn used `fork_turns:none`. I have the supplied task/follow-up messages and prior endpoint/design-review context, not the full parent history. I read the requested implementation/tests/protocol, their relevant source dependencies, the wrapper and implementation notes. Prior historical results and sibling AI reviews are known; synthetic test output is exposed. I did not inspect production tensors/results, load production checkpoints, collect production outcomes, invoke a real registration/run stage, generate reports, contact anyone or use GitHub. Parent steering proposed the numerical fix after my failing fixture; that influence is disclosed. Shared AI model/training and commissioning limit methodological independence, and no human/author validation is claimed.

## Pre-output correction review: numerical fix passes, archival details pending

The revised script `0f3f396ff15904f9ad6f72fb3811a5be441258a07cd3b7b1f150924f9ebc25ac` and protocol `31126c555c443c8632b34c3d7049147a1a54573d878918ef8e272a98c0406a93` implement fixed source tolerance and an elementwise linear decoder propagation envelope, keeping inverse/forecast-patch restoration exact. The preserved independent calibrated fixture now passes; its [corrected terminal output](additional_fixture_corrected_output.txt) is retained. The supplied 11 fixtures pass in 2.131 seconds.

Further toy arithmetic isolates the issue: standardized inputs remain bitwise permuted, while the first float32 linear layer differs by at most 7.15256e-7 under row reordering; computing that layer in float64 gives 6.66134e-16. This supports a numerical matrix-arithmetic diagnosis rather than a shared-normalizer semantic defect. No production observation determined the tolerance or envelope.

Two details still need correction before final freeze. The revised function constructs elementwise feature/output discrepancies and bounds but omits them from the returned trace, retaining only maxima; add those tensors to the archived intervention trace. The manifest also retains obsolete zero `intervention_permutation_atol/rtol` fields beside the new nonzero tolerance; remove or rename/update the conflicting fields. The parent was notified before production. All first-pass findings and original source snapshots remain preserved.

## Final pre-output disposition: approved at pinned sources

**Approve registration and the single bounded physical-information diagnostic at the final hashes below. No material issue remains from this review.** Approval is for this physical/source-information component, not appearance correspondence, qualitative-source qualification, qualia or goal completion. This reviewer has not registered or executed production.

| Final artifact | SHA256 |
|---|---|
| script | `c37d9eb88a132170181cf4b3b7316e9a431d8c801159ed6a82847fe1cf3632e6` |
| tests | `62fa5c62c620f7ebb82f626436d0d19bf69f4a3381fb9b0164e6be8e15c664be` |
| protocol | `31126c555c443c8632b34c3d7049147a1a54573d878918ef8e272a98c0406a93` |
| execution wrapper | `be74ce53852a7e297ef9877932f123266a6aeba31211059690e7527a7287aa32` |

The final revision returns all five local views' elementwise feature/output discrepancies and propagated envelopes in `decode_permutation_error_bounds`, updates both manifest tolerance fields consistently, and hashes every regular file in the review directory while retaining the required review/metadata gate. Review output, original failure and preserved fixture are therefore included in the registration hash record.

Only checks relevant to those new issues were repeated after the last revision, as requested. The preserved calibrated fixture passes the final implementation; independently checked trace keys, tensor shapes and every elementwise bound pass. A temporary-root registration with dummy inputs confirms all tolerance fields agree and the extra preserved-review file is hashed. The [final check output](final_issue_checks_output.txt) retains this receipt. The final test changes add matching archive/manifest assertions and were inspected; a full-suite run after this last additive revision is not claimed. The last full-suite receipt remains 11 passing fixtures in 2.131 seconds at the first numerical correction.

All numerical corrections were made before production registration/inference using toy fixture evidence. Exact inverse restoration and fixed-source forecast-patch invariance remain bitwise; only row-permutation/source checks use the fixed tolerance and measured propagation envelope. If any pinned source changes before registration, the parent must have that change reviewed rather than treating this disposition as approval of unseen code. Historical failures, the independent failing first pass and all exposure limits remain preserved.
