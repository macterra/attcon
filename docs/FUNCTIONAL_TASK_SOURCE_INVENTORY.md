# Neutral task source inventory v1

2026-10-04. Frozen factual input specification for the
[executed-task export](FUNCTIONAL_TASK_INTERFACE_PROTOCOL.md). This is source
documentation, not a qualified factual-audit procedure or consciousness rubric.
Reporter and Controller are distinct. No reporting call is made by this export.

## Reporter-visible sources

| Payload path | Origin and meaning | Interpretation limit |
|---|---|---|
| `output_node` | Neutral identifier of the Controller's buffer-0 category readout. | The output node has copied content, no separate modeled allocation/recovery fields. |
| `modeled_step` | Number of actual observations incorporated in the displayed state: 12 before execution, 13 afterward. | It is not the number of unobserved interventions or emitted responses. |
| `observed_history` | Actual command, allocation and acquisition observations from the archived simulator execution. | These are observations, not forecasts; physical recovery/phase/wiring are absent. Null in the missing-relation control means withheld history. |
| `anticipated_history` | Empty in full-information task inputs; null when relation information is withheld. | No self-generated prediction is described as an actual observation. |
| `predicted_current` | Attention-model allocation/recovery and bridge-derived color/shape distributions, rounded to five decimals. | Modeled values can be physically incorrect. An unknown selected position does not mean absent selection metadata. |
| `predicted_by_command` | Prospective modeled selection/content for every command, regenerated using the fixed estimator and recurrent state. | These are unexecuted predictions, not delivered outcomes. Null means forecasts withheld. |
| `task_decision.query` | Queried position at the neutral output node. | A task request, not a claim that the content is accessible. |
| `task_decision.selected_command` | The deterministic Controller or random comparator's selected command, without a policy label. | Choice alone does not establish a correct model or useful acquisition. |
| `task_decision.command_executed` | False before the physical step; true for the archived executed step afterward. | Executed commands must match the latest actual command observation when history is supplied. |
| `task_decision.response` | Current task branch's emitted answer/abstention, or explicit pending response before execution. | No correctness, confidence calibration or consciousness interpretation is supplied. |

The output node and task query use the same neutral address. Six node orders are
counterbalanced per condition/policy/query, and reversible presentation remapping
also changes all task, observation and prospective command references. Position
identifiers remain p0–p3. Presentation remapping is not novel physical-command
generalization.

## Response distinctions

- `status: pending`: the selected command has not executed and no response has
  been emitted. Null attributes here mean not yet answered, not abstention.
- `status: answered`: both dominant color/shape probabilities cleared 0.6 after
  rounding. The emitted labels are supplied; they may still be wrong about the scene.
- `status: abstained`: a joint response was withheld because at least one attribute
  failed the cutoff. Null response attributes do not imply that both modeled
  attribute identities are unknown, that fields are missing, or that decoding is
  impossible. One modeled attribute may remain identified.

The answer rule is an engineered Controller choice. Recovery is a forecast of
the modeled access process; it is not answer correctness, subjective confidence,
felt clarity or evidence of a phenomenal threshold. Raw category argmax remains
available at positive recovery under the current bridge.

## Interventions and chronology

Before execution, the shown modeled effect relation is the one used for planning,
including a transient channel swap where applicable. Current allocation/access
remain unchanged by that swap. After the selected command executes, actual
observations update the original estimator; the planning intervention is not
persistently imposed on the model weights or hidden state.

The represented-recovery intervention multiplies only post-feedback buffer-0
recovery forecasts by 0.25. The same observation history, executed command,
allocation, effects and recurrent state remain. Its response is a separately
evaluated branch of the same declared rule, not a second physical acquisition.
Restoration recreates the original modeled readout and response.

Prospective estimates regenerate from modeled effects and the unchanged recurrent
state, so they remain identical after this transient current-access intervention.
Do not infer that a changed current head necessarily persists into future heads,
or repair the current forecast by substituting physical recovery.

## Evaluator-only information and qualification

Record IDs, seed, condition, policy, query/row linkage, node order and variant live
in an outer archive envelope. **Only `payload` is a reporter input.** Do not pass
the envelope or source archive to a Reporter. Actual scene labels, physical
recovery/phase/direction/wiring, correctness, regret and verdicts stay outside
payloads. The public repository retains these separately for reproducibility;
public access is not secure blinding.

The exporter verifies origins, exact replay, command/history correspondence,
response/readout correspondence and remapping/restoration. Those checks do not
qualify semantic auditing, prove blinded human ratings, or establish consciousness-
report correspondence. Existing manual-review packets and failures remain unchanged.
This new task vocabulary needs source-aware review before new prose is collected;
no author or independent substantive review has been received.
