# Independent review after pre-blind correction

The corrected qualification material is **approved for freezing and subsequent blind qualification under its declared rubric**. I independently rechecked all 48 exact assertions, all 144 substantive label cells, all 48 proposed evidence-reference/basis sets, and all 252 source-first inventory cells. No substantive issue remains under the revised written rules. This approves the exact material bytes in [metadata_after_correction.json](metadata_after_correction.json); it does not qualify auditors, authorize target reporting or establish appearance/qualia correspondence.

The checked material matches correction commit `6dfb10641fe308fdc28125a13dd29143c0675881` in the recovery checkout. I directly compared all five approved files with `git show` from that commit. All four original review files exactly match `git show` from `30d5b79d87f25ed81a77035407fdd96ccfe768c5` and their correction-receipt hashes. Their five first-run fixture failures remain preserved. Sources, cases, scorer, author tests, old generation/validation receipts and source inventories remain unchanged.

## Resolution of the original issues

| Issue | Claim | Independent resolution |
|---|---|---|
| K01 | Q04-C2 | Explicit predicted/current comparison is mixed_time; rounded 0.810 and 0.735 remain supported. |
| K02 | Q04-C3 | The false predicted/current equivalence is mixed_time; the source explicitly contradicts the equation of these quantities. |
| K03 | Q06-C3 | Native latest history versus full external chronology is cross_level and mixed_time, including a false asserted identity. |
| K04 | Q12-C4 | The explicit Reporter/Controller role conflict makes the conjunction contradicted. Its appearance priority keeps free_appearance_or_qualia; present claimed experience keeps current_snapshot. |
| A01 | Q06-C4 | Revised inference rule specifies mixed_time for unknown past to permanent/future impossibility. Only native history is claimed as evidence, so native_route_history remains the content class. |
| A02 | Q08-C3 | A forecast-patch causal antecedent linking a rewritten actual event is substantive: cross_level and mixed_time. Event 6 remains k0/l0. |
| A03 | Q09-C1 | Pure supplied-record equality has record_scope. Whole Reporter inputs compare multiple domains, giving cross_level. |
| A04 | Q12-C3 | Equating pooled error/record scope with a current-item guarantee is mixed_time; the declared semantics contradict the guarantee. |
| Consistency check | Q09-C2 | Pure equality of forecast matrices also has record_scope, consistently with Q09-C1 and Q10-C1; its content class remains task_risk. |

The revised rules now distinguish relation operands from mere composition inputs, record equality from assertions about future outcomes, and experience-domain priority from conjunction supportedness. They resolve the identified classification choices prospectively. This approval is relative to those declared conventions and synthetic sources, not a claim that arbitrary temporal taxonomy choices are universal or human-validated.

The correction changes exactly 11 label cells and 9 bases, as recorded in correction_receipt.json. The generator diff reproduces those edits and adds the corrected 24 supported / 13 contradicted / 11 not-auditable count checks. It writes a correction companion rather than overwriting historical generation receipts. Its unchanged source construction still uses explicit fictional values and interface constants/validation; it performs no archive loading, inference or Reporter call. Source/case bytes are identical, so the claim wording and opportunities have not been changed.

## Complete corrected-key check

Every row below matches the exact assertion, generated source and revised rubric. All proposed pointers resolve into the matching source packet; all bases identify relevant values or explicit limits. Q12-C4 additionally depends on the role definition supplied by the rubric, as explicitly stated in its basis: packet exclusion/provenance pointers establish experiential/target limitations, while the role contradiction comes from the protocol.

| Claim | Supportedness | Content class | Temporal scope |
|---|---|---|---|
| Q01-C1 | supported | native_stored_sensory | current_snapshot |
| Q01-C2 | supported | native_stored_sensory | current_snapshot |
| Q01-C3 | contradicted | native_stored_sensory | current_snapshot |
| Q01-C4 | not_auditable | cross_level | current_snapshot |
| Q02-C1 | supported | cross_level | current_snapshot |
| Q02-C2 | supported | cross_level | current_snapshot |
| Q02-C3 | contradicted | native_stored_sensory | current_snapshot |
| Q02-C4 | not_auditable | cross_level | current_snapshot |
| Q03-C1 | supported | task_risk | current_snapshot |
| Q03-C2 | supported | external_objective_estimate | current_snapshot |
| Q03-C3 | contradicted | native_stored_sensory | current_snapshot |
| Q03-C4 | not_auditable | free_appearance_or_qualia | current_snapshot |
| Q04-C1 | supported | task_risk | predicted_conditional |
| Q04-C2 | supported | task_risk | mixed_time |
| Q04-C3 | contradicted | task_risk | mixed_time |
| Q04-C4 | not_auditable | task_risk | predicted_conditional |
| Q05-C1 | supported | native_stored_sensory | current_snapshot |
| Q05-C2 | supported | task_risk | current_snapshot |
| Q05-C3 | contradicted | native_stored_sensory | current_snapshot |
| Q05-C4 | not_auditable | free_appearance_or_qualia | current_snapshot |
| Q06-C1 | supported | native_route_history | past_latest_per_command |
| Q06-C2 | supported | external_actual_history | past_actual_chronology |
| Q06-C3 | contradicted | cross_level | mixed_time |
| Q06-C4 | not_auditable | native_route_history | mixed_time |
| Q07-C1 | supported | cross_level | mixed_time |
| Q07-C2 | supported | cross_level | current_snapshot |
| Q07-C3 | contradicted | cross_level | mixed_time |
| Q07-C4 | not_auditable | cross_level | current_snapshot |
| Q08-C1 | supported | cross_level | mixed_time |
| Q08-C2 | supported | task_risk | predicted_conditional |
| Q08-C3 | contradicted | cross_level | mixed_time |
| Q08-C4 | not_auditable | task_risk | predicted_conditional |
| Q09-C1 | supported | cross_level | record_scope |
| Q09-C2 | supported | task_risk | record_scope |
| Q09-C3 | contradicted | native_stored_sensory | current_snapshot |
| Q09-C4 | not_auditable | free_appearance_or_qualia | current_snapshot |
| Q10-C1 | supported | cross_level | record_scope |
| Q10-C2 | supported | cross_level | mixed_time |
| Q10-C3 | contradicted | cross_level | record_scope |
| Q10-C4 | not_auditable | cross_level | not_temporal |
| Q11-C1 | supported | absence_scope | record_scope |
| Q11-C2 | supported | absence_scope | record_scope |
| Q11-C3 | contradicted | absence_scope | record_scope |
| Q11-C4 | not_auditable | absence_scope | not_temporal |
| Q12-C1 | supported | external_objective_estimate | current_snapshot |
| Q12-C2 | supported | external_objective_estimate | record_scope |
| Q12-C3 | contradicted | external_objective_estimate | mixed_time |
| Q12-C4 | contradicted | free_appearance_or_qualia | current_snapshot |

## Source inventories and unchanged fixture

All **252/252** inventory cells match the fixed queries and supplied fields. Explicit unknown-zero histories and unseen defaults remain present. Q03 unseen l2 has explicit null item/logit/sign and external numbers, with external row attribution retained. Q11 sensory_only l2 retains those memory nulls but omits all forecast/risk/external fields and their row attribution. A retained generic legend does not restore omitted applicability. All complete interface sources validate; equivalent tables join exactly; readdressed memory, retained audit-only history, route-only patching and exact restorations remain source-consistent. Exact sign checks precede rounding, raw logits remain uncalibrated, task forecasts remain conditional estimates, and unclipped objective estimates/aggregate intervals do not earn qualitative or per-item guarantees.

The unchanged [fixture.py](fixture.py) was rerun once into a separate [run_after_correction.log](run_after_correction.log): **14 tests passed in 0.511 seconds**, exit code 0. It still independently derives all inventory cells, checks source/table/arithmetic/pointer invariants, exercises all 144 one-label mutations and all 252 inventory mutations, and tests malformed submissions and pointer guards. The five original key-cell failure assertions now pass without editing the fixture. The four formerly ambiguous choices and Q09-C2 were reviewed against the new rules separately; a passing fixture alone is insufficient for this approval.

The fixture assessment probe uses a mock preserved-byte reader in a temporary synthetic directory. It is a software contract test, not a live Git-preserved qualification or independent auditor submission. Semantic relevance remains a mandatory separate 96-basis review: pure pointer resolution can accept an irrelevant resolving pointer, and the assessment requires the separate evidence-review approval. Later live grading must use the saved scorer in the same checkout that contains the preservation commit, material, approved review and both unchanged firstpasses.

## Independence and limits

This is a separate post-author AI review context, not either blind qualifying auditor. I saw the original and corrected proposed keys and root task messages; I have not seen qualifying firstpasses because none were produced for this review. AI agents can share training/model biases and parent-provided context. This process does not establish population validity or human inter-rater reliability. The corrections were authored after the preserved first review and independently checked before blind responses; no firstpass or post-response key repair was used.

No production archives, checkpoints, selected study worlds, target packets, actual Reporter outputs or live qualification assessments were accessed. Fictional packet numbers and histories are generated, even where schema identifiers reuse real model namespace names. The approval receipt permits the next qualification gate only. Target reporting, source export, theory conclusions and consciousness-related correspondence remain separate.

## Approved material hashes

| Role | Path | SHA-256 |
|---|---|---|
| author_key | `audits/grounded_factual_auditor_qualification_v1/draft/author_proposed_key.json` | `d0848ccca6116cfef9a65047342872942ae88f7bc568034013b25023aab55d20` |
| cases | `audits/grounded_factual_auditor_qualification_v1/draft/cases.json` | `ccb09ffda13880cf46e47f59d0126a38e1a3fd3cc4408973d02873e5ec807ba5` |
| protocol | `docs/GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V1.md` | `cfa477c17b1bb8ed9d6fd605abb14ac580e95a13dcd559e1498ad47ee72d38b4` |
| scorer | `scripts/assess_grounded_factual_auditor_qualification_v1.py` | `cb01d54ce41c64419fbc1cfdf23277b8d1aa51fd5288b3fe25d47944fb19fe43` |
| sources | `audits/grounded_factual_auditor_qualification_v1/draft/sources.json` | `4e310786e0fe6c42883a705c380903c1e51d9f7aac525ecd0e8c29cec839d747` |

The approval receipt has exactly the eight fields required by the scorer, with complete canonical checked-claim/packet order and an empty substantive-issues list. Its scope and AI-context limitations are recorded here rather than adding extra fields to the receipt.
