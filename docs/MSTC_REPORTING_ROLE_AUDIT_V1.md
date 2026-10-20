# MSTC role audit and the reporting-source gap

2026-10-10. This is a code-and-theory audit, not a new experiment or a claim that
the system is conscious. It separates our reporting hypothesis from the stronger
architectural identification in [Heile's MSTC v5](https://arxiv.org/abs/2512.01073v5),
checked against the [full paper](https://arxiv.org/pdf/2512.01073v5), especially
pp. 2–3, 6–7 and 13–16.

## Constraint from the referenced theory

MSTC distinguishes world construction by the Modeler, action selection by the
Controller and focal allocation by the Targeter. Its proposed phenomenal locus
is the Modeler-schema, which converts world contents and focal information into
a distinct comparison representation and uses discrepancies to refine the
Modeler. Regulation alone is insufficient under that proposal. The theory also
distinguishes full-field from focal-only cross-saccadic comparison; a negative
test of the full-field extension does not refute its broader identification.
Diffuse content need not be directly reportable before focal selection.

These are claims of the theory, not established identifications of our code.
Our external Reporter is instrumentation. It must not be conflated with the
Controller or treated as the proposed experiencing locus. Likewise, the names
“attention model” and “Modeler-schema” do not establish the same functional role.

## What the existing implementation supplies

| Implemented component | Verified code role | Remaining identification gap |
|---|---|---|
| Visual encoder and addressed memory | Represent sensory categories/evidence and retain delivered samples | Neither implementation labels its contents as experienced presentation |
| Older recurrent attention predictor | Learns allocation, recovery and command-effect forecasts from observations/history | Training-time errors and recurrence do not demonstrate use of reported bound contents as a Modeler-refinement medium |
| Current learned content model | Uses addressed recurrent h, sensory V and a command to estimate current/conditional maps; queried candidate readouts select an operation | Faithful learned-content readout and operation selection do not by themselves identify the Modeler-schema's comparison/refinement representation |
| BoundState | Joins addressed content to attention forecasts; its content policy selects commands | Selection using a binding is not temporal comparison and correction of the content constructor |
| Grounded structured outcome model | Local sensory-to-success network and separate history-to-delivery network | Changing route history cannot transform its local sensory/risk representation through this forward path |
| External physical decoder | Reads image-property information from hidden state | Recoverable stimulus statistics do not establish a native model of presentation |
| External language Reporter | Describes exported state | Accurate narration alone does not identify a quale-generating computation |

The relevant implementation is
[bound_content.py](../src/attcon/bound_content.py),
[predictive_attention.py](../src/attcon/predictive_attention.py) and
[grounded_outcome_models_v1.py](../scripts/grounded_outcome_models_v1.py).
The structured network computes `risk_hidden` from sensory memory and
`route_hidden` from latest-command history separately. Its after-command success
is composed from delivery, current and refresh forecasts. The joint comparator
has a different input path, but joint input alone does not establish a temporal
comparison/refinement role.

2026-10-19 addendum: the [current source-path audit](ATTENTION_RELATIONAL_NATIVE_READOUT_AUDIT_V1.md)
and its independent technical review cover
[attention_content_model_v1.py](../scripts/attention_content_model_v1.py).
`LearnedContentModel.update` changes addressed h from V, completed-W feedback and
the executed command. A learned decoder supplies constituents of hybrid maps;
the exporter separately supplies origin labels. This permits genuine A-content
reporting, while leaving implicit comparison/status representations unestablished.
Static inspection does not establish their absence. Operation selection,
`NativeController` cell targeting and the external Reporter remain distinct stages.

The primary paper assigns narrative access to focal information and appraisal
signals, while reserving quale generation and comparison for the Modeler-schema
([v5, pp. 10–11 and 13–16](https://arxiv.org/pdf/2512.01073v5)). That theoretical
language role does not identify our external Reporter with our native Controller.
The current reporting analysis tests its narrower source hypothesis; the stronger
comparison/refinement architecture remains an optional extension. This addendum
changes no frozen report criterion, first judgment, scope or binding.

This audit relies on inspected code and preserved studies, not fresh model
execution. The [consistency diagnostic](CONSISTENCY_DIAGNOSTIC_RESULTS.md) already
distinguishes training-time prediction loss, recurrent feedback and actual use of
bound contents. Its unchanged downstream forecasts after decoded-head overrides
do not exclude implicit hidden comparisons. The [grounded physical study](GROUNDED_PHYSICAL_INFORMATION_V1_RESULTS.md)
establishes decodable objective content, with stronger sensory readouts and
retained control limitations. The [independent counterpart review](../audits/qualitative_source_counterpart_review_v1/review.md)
does not qualify a strict presentation counterpart.

## Consequence for the next experiment

The real gap is not solved by more fluent reports or another successful factual
gate. A useful new source experiment must specify a native content-specific
representational property, independently validate what it denotes, and selectively
vary its relationship to attentional processing while preserving sensory input
and decision/uncertainty controls. Source changes followed by reports are an
instrumentation result unless the source's own role and competing explanations
are also tested.

Two hypotheses must remain distinct. The narrower reporting-source hypothesis
asks whether a learned attention-control representation tracks content-specific
changes in the system's internal representational form, and whether neutral reports
faithfully describe that form with independently assessed presentation structure.
A stronger MSTC mechanism hypothesis additionally asks whether the same bound
comparison representation detects discrepancy and refines the underlying
Modeler/World Model. Correcting only the Controller's next command tests attention
control, not that stronger Modeler-refinement role.

For a new architecture, record which pathways and training objectives were
deliberately introduced. Do not call an engineered representation discovered
experience. Compare against equally informed ordinary state estimation, direct
sensory readouts and generic metacognition. A conventional predictor may reproduce
the same behavior; such parity limits discriminative evidence. Lesion and exact
restoration should establish the claimed representation's role, rather than merely
that an arbitrary necessary wire can be disconnected.

The full goal still requires accurate state reporting, independently specified
report correspondence, selective source effects and matched alternatives, then
powered confirmation and replication. Felt truth remains unauditable. An absent
current-source counterpart is retained as a result; it neither proves absence of
experience nor authorizes redefining access/correctness as qualia.
