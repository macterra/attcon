# Independent mechanical review: content-form primitives v1

2026-10-10 local date. First-pass static inspection and independent toy fixtures by `/root/attention_source_bridge_critical_review`. Scope is mechanical correctness only. No learned model, encoder extraction, checkpoint/world loading, training, production inference, Reporter, report rating or presentation qualification was attempted. Other agents' artifacts and the prior bridge reviews were unchanged.

**Verdict: the declared toy mechanics mostly hold, with one bounded saved-restoration defect for a valid public constructor input.** The preserved independent first run executed nine tests: eight passed and one failed. The author's preserved twelve-test passing run remains a separate first pass. No repair or rerun preceded this review. This finding does not introduce an additional semantic gate.

## Confirmed mechanics

Independent fixtures check rectangular noncontiguous maps with two channels across float16, bfloat16, float32 and float64. Registered bandwidths 1/2/4 agree exactly with an independent loop reference on these finite integer-valued toy inputs. These fixtures do not establish exact conservation for every floating-point map; the design appropriately allows reduction/cast roundoff.

Tuple-backed buffers own copied source values. Mutating constructor tensors, accessor tensors and nested current/prospective export lists did not mutate saved state. Frozen fields and tuple entries refused normal assignment. Executed operations preserved retained V, raw logits, identity and delivery records; only the addressed W/completion time changed. Unselected and unseen items were unchanged. A later bandwidth-1 operation reread V rather than attempting to invert previously lost W arrangement.

Unknown/unseen addresses, stale issue times and nonforward completion times were refused. Unseen exports have no identity, actual map or event times. Current actual W has snapshot/completion times distinct from the prospective operator alternative. Prospective export leaves state/history unchanged, explicitly says `analytic_operator_result_not_observed_or_learned`, and declares `learned_prediction=false`.

Endogenous/exogenous execution produced identical addressed items, snapshot times and native current exports for both observed addresses across all registered bandwidths. Only audit-history driver labels differed. This is a numerically matched mechanism control; the labels alone do not identify psychological attention or distinguish causal source theories. The known-operator comparator expressly marks its supplied command mapping `oracle_not_empirically_learned`; no unknown-mapping learner is covertly implemented.

Saved snapshots created through tensor factories restored equal to the entire saved state, including times and history. Restoration did not operate on the altered state or invert averaging. Static inspection found no learned, phenomenal or semantic-positive claim in the prototype/design; the toy identity strings are supplied literal records, not inferred content semantics.

## First-pass defect: accepted direct buffers do not restore equal

The public `SpatialBuffer` constructor accepts this finite input:

```python
SpatialBuffer((1, 4, 4), (0.1,) * 16, torch.float32)
```

Its tuple stores Python `0.1`. `restore_saved` reconstructs each buffer through `tensor()` and `from_tensor()`, changing each tuple value to float32's representable `0.10000000149011612`. Consequently the valid restored `AddressedState` compares unequal to the saved state. This violates the documented complete/exact saved-state restoration claim over accepted public inputs. The failing fixture first verified that tensor values and current exports remain identical: this is an immutable-storage representation defect, not corruption of the exported spatial pattern.

Minimal correction: preserve the already immutable buffer tuples/dtype/shape directly when copying a saved state. Alternatively canonicalize all accepted constructor values on entry, with that contract made explicit; this changes construction semantics and should receive a separate regression check. Keep the present fixture and failed log unchanged, then record any author repair and its validation separately. No author code/test repair was made by this reviewer.

## Boundaries and disposition

The state constructors validate shapes, dtype, finiteness, address uniqueness and chronological completed history. They are not a replay/provenance verifier for arbitrary user-supplied W/history combinations; a supplied toy state is a starting snapshot. No broader historical replay guarantee is inferred here. The copied-state ownership guarantee concerns normal public use, not deliberate Python frozen-dataclass bypasses.

Fix the one exact-restoration edge contract before calling the entire supported-input mechanical suite clean. The fixture success supports only the declared prototype mechanics and the limits above. It neither qualifies a native presentation counterpart nor revises the prior negative phenomenal-source verdict. No additional experiment/report approval is supplied by this review.

Preserved first-pass artifacts: `fixture.py`, `run.log`, this review and `metadata.json`. Metadata records the inspected author-file hashes, independent fixture result, exposure and zero experimental/report receipts.
