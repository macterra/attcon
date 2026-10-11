# First fit reconstruction failure: read-only diagnosis and proposal

2026-10-11. Separate task allocation within the same model team, with inherited
hypothesis context; no independent model family or human endorsement. This is a
source/report diagnosis, not an executed numerical correction or study rerun.

The original first fit reconstruction remains failed. Its preserved report contains
27,616 checks, all passed, plus one verification_exception. It stops during source A,
encoder seed 1301, production-core replay at decision time 0. No complete fit
reconstruction pass is established for seed 1301 or the remaining two seeds. The
six completed 300-update learned models are unchanged; their archived bytes were
not loaded for this diagnosis.

First fit publication: 28406d63c053b810c0fd19348cad84b147f03690.
First failed reconstruction publication: 393eb124449634c0b72597231d6f3cd54ab3645b.
Registration publication: 304a052e537115e04b641e4fca3b3c3136fbd739.
[Original full report gzip](../fit_reconstruction/first_report.json.gz),
[transport manifest](../fit_reconstruction/first_report_manifest.json), and
[unchanged first log](../fit_reconstruction/first_cli.log) retain the first failure.
The gzip was decompressed as JSON only; its raw and compressed hashes and sizes
match the published manifest. No tensor archive, checkpoint, model, renderer,
fixture, optimizer, verifier or registered phase was executed.

## Cause and scope

The traceback identifies registered verifier line 1020:
`model.predict(replace(obs,v=decoy_v),decoy_state,address)`.
The isolation probe replaces V at nonrelevant addresses but leaves the initial
last_revealed_w unchanged. At time 0, the original observation has W0 = V exactly;
its decoy changes that required equality. Core ObservationBatch.validate, lines
129–131, correctly rejects the inconsistent observation with
`initial state is exactly observed V without a command`. That contract predates
the run and is part of the reviewed source. This is an invalid reference replay
probe, not evidence of a production-core arithmetic or permission defect.

The last retained check is
`1301/fit/A/production_core_replay/inherited_next/0`: float32 shape
[96,3,4,24,4,4], exact, zero maximum difference, zero unequal elements and zero
tolerance violations. The failing probe has no actual output to compare, so an
actual-versus-expected numerical difference does not exist for that call. Later
checks and seeds remain unexecuted by this first verification. Passing partial
checks cannot establish the unexecuted invariants or a complete fitting trajectory.

## Narrow separately labelled reference correction, proposed only

Prepare a new script `scripts/verify_attention_content_model_v1_corrected.py` after
independent critical review. Leave the registered verifier and registration bytes
unchanged. Preserve the existing relevance mask, state perturbation and all
original reference comparisons, discrete policies, check names and tolerances.
Only the decoy observation construction changes:

```python
decoy_obs = replace(
    obs, v=decoy_v,
    last_revealed_w=decoy_v if time == 0 else obs.last_revealed_w,
)
decoy = model.predict(decoy_obs, decoy_state, address)
```

At time 0, candidate-address V and latent state remain untouched, and there is no
previous command/current decoder. Updating the noncandidate initial snapshot
keeps W0 = V coherent without adding hidden or future information. Selected next
maps must still match exactly. Full copied noncandidate maps can differ in this
probe; no existing check asks them to equal the original baseline. At later times,
retain the original last revealed snapshot and previous/candidate relevance mask.
No production observation, fitted weight, retained output or posterior is edited.
A future bounded fixture should verify coherent time-0 decoys for learned and
analytic sources, exact selected-next parity, original inconsistency rejection,
and unchanged later-time snapshots. This fixture has not been written or executed.

The corrected script must bind its new exact bytes and approval to a committed
reference amendment while continuing to validate the original registered closure.
Keep the existing registration/exact_verifier_bytes check against the unchanged
registered file; add separate checks for the executing corrected program and
amendment. Its report should use an explicitly corrected-reference schema/version,
actual corrected verifier hash, original registered verifier hash, original first
failure commit/raw hash, amendment commit/hash and original phase/fitted bindings.
The original first report is never relabelled or replaced. A future authorized run
would be the first run of this separately preserved corrected reference, not a new
fit or regenerated study phase. No correction runtime is authorized by this record.

## Minimum versioned phase-boundary amendment, proposed only

The current runner validates exact registration/source/input bytes before every
phase. A changed runner in its registered path would fail those bindings. The
current heldout runner additionally compares the dev report verifier_sha256 to the
original registered verifier hash (line 446); the registered verifier does the same
at line 1503. Merely writing a corrected report or changing its verifier_sha256 to
pretend it came from the original script is unacceptable.

Use a separately versioned boundary entry point for future dev/heldout release and
an immutable, independently approved reference-amendment JSON. The original runner,
registration, engine, collector, scientific modules, native inputs and fit artifacts
remain byte-identical and continue to pass their existing committed bindings.
The amendment must bind the registration SHA/commit, original complete source
closure, immutable fit receipt/artifact hashes and fit publication, original failed
report transport/raw identity and publication, exact corrected verifier and boundary
entry-point bytes, critical approval bytes, and a full amendment preservation commit.
Require committed approval and ancestry before any future phase claim or generation.
The boundary entry point must reject fit/register dispatch and authorize only the
already designed dev/heldout phases; the retained original execution logic stays
unchanged except amended report acceptance and amendment provenance in new claims/
receipts (or an explicitly bound companion boundary receipt).

Both amended heldout acceptance implementations must accept only the explicitly
labelled corrected-reference report whose actual executing hash and amendment
identity match that one reviewed amendment. Retain every existing full-check-pass,
error/exception absence, raw report commitment/SHA, dev receipt, fitted artifact,
registration command, ancestry, explicit source disposition and Reporter prohibition
condition. Keep original-mode acceptance valid only for the original schema/hash;
never replace the registered source hash or broadly allow arbitrary verifier hashes.
No gzip-only dev transport change is needed: the full corrected dev JSON still has
to be committed at the canonical bound raw path. New amendment checks are provenance
checks, not weaker or additional scientific thresholds.

Prepare and publish the exact corrected source/boundary/amendment review before a
separate authorized correction execution or dev release. The corrected full fit
report must first be preserved and assessed; this proposal grants no release.
No new fit, data, parameter budget, tolerance, study gate, Reporter or scientific
threshold is proposed. Controller is not Reporter; even a complete numerical pass
cannot establish qualia or presentation. The scientific goal remains unachieved.
