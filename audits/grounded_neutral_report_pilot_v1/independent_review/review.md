# Independent exporter first-pass review

2026-10-10. **Changes required before source-only registration.** The immutable
draft has one reproduced output-binding defect. This review does not authorize
selected real-source loading, production export, Reporter dispatch, qualification
responses or target prose.

The author explicitly marked the exporter and tests READY. Root then saved draft
commit `20db548faca904fcedd415cfb5005ebbbed930b2` in the recovery checkout
`/tmp/attcon-review-milestones-vvjs166i`. Review execution began only afterward,
using matching source bytes in `/home/david/dev/attcon`. No author code, tests,
protocol, source inventory or frozen interface was edited. No real archive or
checkpoint was opened, network/model inference performed, target record generated,
Reporter dispatched, or factual/character qualification run. Temporary fake Git
commits were authorized solely to exercise the freeze boundary.

The reviewer was a separate-context AI agent commissioned by the same project.
Shared training, shared tools/repository and project commissioning limit its
independence. It received the review scope and immutable protocol/interface, then
read the final exporter/tests. It was not the exporter author. Its fixture imports
the pre-existing generated interface toy-source helper; its assertions and
adversarial disk fixture were independently written after the draft was saved.

## Preserved first pass

Command: `PYTHONPATH=src:scripts:tests .venv/bin/python audits/grounded_neutral_report_pilot_v1/independent_review/fixture.py > audits/grounded_neutral_report_pilot_v1/independent_review/run.log 2>&1`.

The first execution ran **12 tests in 11.714 seconds: 11 passed, 1 failed; exit
status 1**. The fixture and complete log are unchanged. There was no correction
or retry before this review was written. Generated packet-like test objects and
temporary disk exports used toy sources only and were deleted with their temporary
directories; they were not prospective production records or Reporter outputs.

## Finding requiring correction

**R1 — `verify_export` does not enforce the registered output root.**
`export_once` checks that its resolved output equals
`repo_root / registration['output_root']`; `verify_export` omits that check.
Independent test 11 supplied a complete internally consistent generated export
under `unregistered_copy`, while the generated registration declared `targets`.
It supplied the expected freeze-return and generated bundles via explicit mocks
to isolate this directory invariant. Actual packet hashes, audit serialization,
saved verification and deterministic full-source replay were checked by the real
implementation. Verification returned successfully instead of rejecting the
unregistered location. The log preserves `ValueError not raised`.

The packet information remains faithful in this reproduction. The defect is the
sole-output-location contract: a copied export can be certified outside the
registered location. Require the same resolved-path comparison in verification
before accepting any output artifacts, and retain this first failure. A correction
must receive a fresh saved-code receipt and independent rerun before source-only
registration. The universally closed `reporter_release` prevents this finding from
currently authorizing prose.

## Checks that passed

- Exactly 13 arms and 624 distinct slots; all 16 worlds for each of three models;
  all four locations in every arm supplying location content; indexed location
  `world % 4` and command `world // 4`; distinct 128-bit opaque packet suffixes.
- Full-information inventories contain 21 units, including all four delivery and
  all four post-command scalar cells, the three source-meaning units, and explicit
  unseen null/default and unknown native-history distinctions. Every saved fact
  pointer resolves to its full-precision packet value, including table packets.
- Natural, sham and both restorations have exactly equal instruction, query and
  source bytes after excluding the transport packet ID. Apparatus changes only
  the specified referent sentence. Packet IDs remain audit/transport identifiers;
  the future dispatch adapter must preserve this substantive input boundary.
- Crossed memory/route conditions retain moved native content and readouts plus
  unchanged native history. Independently undoing the command-row rotation and
  recomposing returns the archived memory view exactly; archived inverse memory
  restoration returns natural exactly. This is source arithmetic/restoration,
  without asserting new joint network inference or prose restoration.
- Address, evidence and source identities have exact inverse mappings. The second
  ID mapping differs from the ordinary mapping. Independently reversing table
  row order, joining by address and restoring canonical order recovers natural.
- Missing thematic groups are typed `not_supplied` and the three ablations are
  marked unmatched. Counts remain sensory-only 9 supplied + 3 absence units,
  route-only 6 + 3, and no-external 18 + 1; route-only has no content denominator.
- Canonical transform boundaries reject direct mapping-truth injection; packet
  schemas exclude truth, pixels, chronology, provenance and scoring pointers;
  negative objective readout estimates remain unclipped.
- Missing manifest-declared upstream dependencies stop before any loader call.
  Static inspection additionally confirms registered source hashes are checked
  before and after loading and loader provenance must agree. This is fake-only
  validation, not a receipt for the real inventory or archives.
- Generated failed export consumes its directory, preserves attempt/failure
  receipts, prohibits retry in that directory and rejects alternate export roots.
  A generated real Git commit binds registration, actual imported code bytes,
  protocol and review receipts; each edited fixture file is rejected.
- `reporter_release` rejects even a purported perfect qualification and performs
  no loading. It remains unavailable pending a separately reviewed reporting
  registration and source-linked qualification verifier.

## Remaining factual and scientific boundaries

This first pass checks the exporter as factual source instrumentation. Source-only
registration is still pending R1 correction/re-review and a concrete frozen
registration binding the actual source inventory and final reviewed bytes. The
review has not inspected selected archives or established their exported values.
No target adapter, factual challenge/key/scorer qualification receipt, independent
key approval, Reporter configuration, full pre-prose reporting registration,
factual auditor qualification or target character-rating run is qualified by this
exporter review. Completion of those stages must preserve their own first passes
and the fail-closed release boundary.

Faithful factual export does not qualify a native appearance counterpart.
`Faithful_P_presentation` remains `not_established`; felt truth remains
unauditable. Whether the attention-control model is the source of qualia remains
the open main theory goal. The synthetic memory/route arithmetic, objective
readouts and exact controls establish no positive theory-facing result.

## Reviewed byte identities

| File | SHA-256 |
|---|---|
| `scripts/grounded_neutral_report_pilot_v1.py` | `4a4ba2ff97a522b6070347644f8590256e2deb515852dd3a24ee29aa5d44aefe` |
| `tests/test_grounded_neutral_report_pilot_v1.py` | `85658452a33341e82427c253097f4d0db50c72c492cff7815572f70f0b6d1974` |
| `scripts/grounded_report_interface_v1.py` | `b1d27f6131cf9c49c4ec24aa06b2ebdf9456de4c8062486e712594046558cc64` |
| `docs/GROUNDED_REPORT_SOURCE_INVENTORY_V1.md` | `264ca6c06b8da36729e7cfefb450e28a4ff80a7ba2109c23317727d910852d97` |
| `docs/GROUNDED_NEUTRAL_REPORT_PILOT_V1.md` | `db8f77141762a8776f789a0f7c688d8b32524392ee7a81165daa2dba1c29c0d6` |
| `independent_review/fixture.py` | `fe48d6bf3e75b3adae6e9e79b8fa5beec747fe990b5608dc293aed9502404e7e` |
| `independent_review/run.log` | `2e578e1599f3e63ccd01e873080161eeaa5f72eba45e372ba4e84ef34399b715` |
