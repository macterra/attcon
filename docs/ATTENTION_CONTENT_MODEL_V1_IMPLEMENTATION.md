# Learned attention-content model v1: software implementation

2026-10-11. The approved learned model and evaluator collector are implemented.
Their bounded software fixtures pass; **no registered A training, development or
heldout run has occurred, and no new reports were generated**.

The [design](ATTENTION_CONTENT_MODEL_V1_DESIGN.md) fixes a shared GRUCell(771,48)
and residual decoder 435→96→384: 197,328 parameters per learned source. The
[module](../scripts/attention_content_model_v1.py) keeps addressed recurrent h,
estimated current contents and prospective contents distinct. It consumes delayed
completion feedback, predicts raw maps, and supplies frozen H(A) fields to the
operation-selection rule. Completed W and H(W) remain independently attributed
actual-state measurements. Controller and external Reporter remain separate.

After [independent static review](../audits/attention_content_model_v1/implementation_review/review.md)
and publication of exact source at `dbaa593`, the first core suite passed **13/13**
tests. The [unchanged first log](../audits/attention_content_model_v1/implementation/first_test_run.log)
and [receipt](../audits/attention_content_model_v1/implementation/first_test_metadata.json)
record the command, source hashes and result. Fixtures cover parameter arithmetic,
normalization, delayed prediction/update ordering, address locality, gradient
propagation, source interventions/restoration, ordinary alias parity, analytic
mapping ambiguity and itemwise native readout/policy behavior. The gradient check
uses backward only, with no optimizer or parameter update.

The [evaluator collector](../scripts/attention_content_model_v1_data.py) constructs
fixed command histories, same-V actual forks and donor prefixes. Its first five
fixtures passed, but [static review](../audits/attention_content_model_v1/collector_review/review.md)
found two malformed-input rejection defects. That source, first tests and review
were preserved at `036ac46` before correction. Upfront command-range checks,
integer permutation checks and donor-index validation now resolve both defects.
The [separate corrected suite](../audits/attention_content_model_v1/collector_software/corrected_first_run.log)
passes **7/7** tests; [review](../audits/attention_content_model_v1/collector_review/after_corrections.md)
approves bounded evaluator-helper integration. Original records remain unchanged.

Every executed fixture uses handcrafted tensors or freshly initialized toy
weights. None generates reserved task worlds or loads native artifacts. Software
passing does not establish learned predictive accuracy, constituent semantics,
presentation manner or consciousness-like reporting.

The native prerequisite is now satisfied numerically: the separately labelled
[revalidation](SPATIAL_NATIVE_TASK_V1_RESULTS.md) passes all 206,070 checks of the
retained experiment. The corrected phase runner's first
[native prerequisite preflight](../audits/attention_content_model_v1/runner_software/first_native_acceptance_preflight.json)
also passes against the current accepted native review and archive hashes, using
JSON and byte checks only, without loading tensor archives or releasing task data.
The first
[boundary suite](../audits/attention_content_model_v1/runner_software/first_boundary_test_run.log)
passes **7/7** after exact source and scoped static approval were published at
`d1e3d36`. These fixtures use temporary files and mocked bindings/import failures;
they check phase inventory rejection, commit pinning and preservation of an
original error when partial-output retention also fails. They do not import torch,
load scientific inputs or certify the engine's tensor checks.

The experiment engine and phase runner still need complete integration review,
independent reconstruction and committed input registration. The engine's first
review identified E1–E5 provenance, state-isolation and diagnostic omissions;
prospective corrections retain complete analytic masks, actual prediction/update
manifests, named assembly gates, completed-source failure snapshots and full paired
diagnostics. Their runtime and independent reconstruction remain pending. The
heldout boundary must also bind an actual passing independent dev reconstruction
and source semantic assessment before release. The first
[dev-release boundary suite](../audits/attention_content_model_v1/runner_software/first_dev_release_test_run.log)
passes **5/5**, after exact source and review were published at `f2f78eb`.
Temporary synthetic reports check rejection of an unbound assertion, a failed
report, another verifier and another fit commit. A passing bound report with
negative source semantics permits the fixed technical heldout run while reporting
remains unreleased. Mocked fixtures provide no actual dev reconstruction evidence.
Fit, dev and heldout releases must follow the design's immutable receipt sequence.
Their first outputs, errors, ordinary/analytic parity and negative semantic
endpoints will be retained. Literal source contents and later neutral report
correspondence remain empirical questions; the qualia-source goal is unachieved.
