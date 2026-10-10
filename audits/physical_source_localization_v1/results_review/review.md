# Independent subagent review of the physical source diagnostic

Date: 2026-10-10. Reviewer: `/root/physical_results_audit`.
This review examined the protocol, frozen implementation, production model code,
registration, retained raw tensors and results. It was commissioned after outputs
existed and is not blinded. No participant judgments, human validation, new model
training, Controller execution or Reporter output were obtained in this review.

## Finding

The recorded computations support a limited physical-information localization
result. All seven registered source hashes, six checkpoint hashes, source snapshot
hashes and raw/result completion hashes match the files inspected. No integrity
failure or critical implementation flaw was found that invalidates this stated
diagnostic. The result does not establish the consciousness-related project goal.

The raw trace contains three pairs, 128 development and 128 evaluation base scenes
per pair, six transformations, two views and four objects. Thus the evaluation
budget is 384 base scenes and 18,432 dependent object-variant rows. Development
and evaluation seeds differ; all variants remain grouped within their base scene.
The scene is the bootstrap unit. This is not 18,432 independent observations or a
three-model population inference.

I recomputed pixel means and edge energies directly from retained base images,
using separate horizontal and vertical Gaussian convolutions instead of the
experiment's two-dimensional convolution. Target agreement is within 5.96e-8
throughout. Independently recomputed development standardization, ridge
coefficients, held-out predictions, all published R² values and bootstrap
percentile intervals exactly match at the registered two-thread setting. An
initial default-thread exact comparison differed through numerical rounding;
setting the registered thread count resolved it. No experimental artifact changed.

Every recorded attention hidden and forecast feature tensor is exactly equal
across the six pixel variants within each base scene/view/object. This checks
actual retained tensors, rather than trusting an invariance boolean. Production
code supplies the recurrent estimator with allocation, acquisition and command
history and no pixel input. The independent zero-acquisition control changes its
state. The exact invariance is therefore a result about this fixed input path,
not evidence against all bound or relational representations.

Upstream 64-dimensional visual hidden features predict the within-object pixel
mean with R² 0.9891–0.9942 and edge energy with R² 0.7449–0.8728. The existing
binding operation discards this hidden vector. These readouts do not demonstrate
that the diagnostic Reporter was previously given those features.

## Alternative explanations and limits

Category/bound/buffer readouts retain small physical signals. These are compatible
with changes in upstream classification probabilities, not an attention-specific
appearance representation. Category confidence is a nonlinear function of those
same probabilities; its larger linear-probe R² does not introduce new information.
Across evaluation variants, joint category argmax differs from the unblurred
scale-one baseline in 9, 3 and 4 object-variant rows for the three pairs. Maximum
within-object classifier max-probability drift is 0.3897, 0.4797 and 0.5051.
High aggregate category accuracy does not establish matched category certainty.
No task-answer or response-confidence matching experiment occurs here.

The retained category/q/interaction features exactly reconstruct the engineered
categorical buffer. Halving q changes that buffer, restores it exactly, and
preserves the attention allocation/effects supplied to that intervention. A
binding roll changes content-location relations and restores them exactly with
fixed visual content. These checks demonstrate defined dependencies. They do
not make q perceived clarity, and probe differences between buffer and its
reconstructing baseline can follow feature basis, standardization and ridge
regularization. Memory masking and q mixing are distinct paths and remain so in
the retained tensors.

Within-object centering restricts the estimand to responses to this six-variant
transformation family. It needs the group structure and other variants at
prediction time. It is not an isolated-image prediction result. Development-only
scaling and fitting avoid fitting coefficients to evaluation targets. The
conditional bootstrap covers base-scene variation at fixed checkpoints and
readouts, not participant, model-population or training uncertainty. Linear-probe
performance does not measure all information available to a nonlinear decoder.
Different source dimensions and feature bases further limit probe-rank comparisons.

Intensity scaling also scales edge energy, and blur affects both targets. These
objective targets are not independently manipulated subjective dimensions.
Synthetic 16×16 stimuli and transformations outside training distribution do not
qualify published participant stimuli or natural-scene transfer. No subjective
appearance labels, prose reference distribution or consciousness acceptance gate
were tested. The manifest's `new_training: false` refers to no retraining of the
frozen architecture; development ridge readouts are fitted as explicitly registered.

## Consequence

A separately registered participant-data and source-sufficiency assay is still
needed. It must distinguish independently observed appearance judgments from
physical features, category uncertainty, access history and reporter wording.
Subagent review is authorized and removes a human-only workflow dependency, but
these agents share training and commissioning context and do not constitute
independent human or biological validation. No human requirement is reinstated
by this review. The current physical result can guide that work; it cannot replace it.
