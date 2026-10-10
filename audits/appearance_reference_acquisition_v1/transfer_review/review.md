# Independent appearance-reference transfer feasibility review

**Accept the article as a candidate behavioral reference; reject direct frozen-model transfer as an attention-origin appearance test.** This judgment concerns identification, not the truth of the theory or absence of consciousness.

## Primary evidence

[Rausch and Zehetleitner (2016)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00591/full), DOI 10.3389/fpsyg.2016.00591. Twenty observers discriminated horizontal/vertical binary gratings in small circular/square targets. Six contrast levels and 200 ms presentations were used. Each trial collected continuous visibility and correctness-confidence judgments; rating order was balanced across observers. Visibility retained a contrast association on error trials more than confidence did. The article links archived data and analysis. Archive bytes, coding, exclusions and stimuli were not inspected here. Contrast, not attention, is experimentally manipulated. The article supplies no measured allocation/acquisition trajectory. Numerical judgments are not a free-prose consciousness-report corpus. These facts qualify a candidate distinction, not an artificial system's experience.

## Architecture and task mismatch

The code identifies `Forecast` as the attention-control model state. Its GRU receives 20 numbers: allocation, acquisition and command history. Heads predict allocation, delayed recovery and command effects. It receives no RGB patches or object identity. With history and initialization fixed, changing image contrast cannot change this deterministic recurrent state. This does not imply that the larger bound representation lacks appearance-relevant information, or that attention-related qualia must mean brightness.

The separate RGB 16-by-16 encoder has a 64-dimensional hidden layer and eight categorical outputs for four colors and four shapes. Grayscale grating conversion is numerically possible, but native color/shape identity is not the human orientation task. Orientation may be decodable in frozen hidden features despite no explicit head. Test it; do not infer architectural impossibility from head names. Conversion has no established equivalent for physical luminance, spatial frequency, visual angle or temporal exposure.

`BoundState.contents` explicitly binds visual categories to locations; this is distinct from GRU state. `buffer_contents` mixes categorical distributions with uniform probability proportional to recovery. An appearance prediction from a buffer could therefore originate in upstream features, generic certainty or that engineered bridge. Renaming the buffer cannot localize the source.

## Rejected shortcuts

- Recoloring or replacing gratings with plain shapes while retaining original human orientation errors/visibility labels changes the content and counterfactual task.
- Putting human visibility/confidence into acquisition quality, or selecting contrast-to-quality conversion to maximize visibility fit, injects the target into the purported source.
- Attention and upstream readouts must have comparable information boundaries and bounded capacity. Participant labels, human answers or human confidence cannot be exclusive attention inputs.
- Artificial attention interventions are not interventions on these human observers. The source experiment contains no corresponding attention manipulation.
- Predicting visibility from contrast, hidden visual features, categorical entropy or a monotonic recovery function is not itself attention-origin evidence. Failure under severe task mismatch is not a general refutation.

## Concrete valid route

1. **Qualify the archive.** Preserve primary deposit bytes and provenance. Independently verify participant/trial identifiers, stimulus/response orientation, correctness, contrast, visibility, confidence, order and training/exclusion rows. Check whether exact stimuli or generation parameters exist. Reproduce the published dissociation with participant-level uncertainty, retaining discrepancies rather than tuning exclusions. This supplies a numeric behavioral distinction, not an artificial system result.
2. **Separate sensory transfer.** Freeze an orientation-preserving input conversion before inspecting human-appearance targets. Preserve available contrast/aperture/presentation information and explicitly mark unknown parameters. Test frozen visual hidden-feature orientation decoding against generated stimulus labels only, holding out rendering and contrast conditions. Failure pauses same-task transfer; success permits an ancillary sensory comparison with an out-of-distribution limitation. Neither localizes attention.
3. **If needed, register a new apparatus.** Ground acquisition/recovery in independently measured synthetic orientation outcomes, not human visibility/confidence. Retraining an encoder or adding a sensory-quality bridge changes the model and requires new registration, seeds and retained old failures. Keep the Controller's action selection separate from diagnostic reporting. Declare whether the candidate locus is recurrent attention state or bound content/attention state; do not silently alternate between them.
4. **Add attention leverage.** Seek an archived experiment independently varying attention and sensory strength with appearance judgments, task answers and confidence where available. Matching task content and a declared sensory conversion are prerequisites. Human attention assignment supplies a behavioral counterfactual, not direct observation of the internal attention model. In the artificial apparatus, selectively intervene on and restore the identified attention-model state with upstream representation and actual task history fixed; retain upstream-only and generic uncertainty/memory alternatives. Register independent participant/model units, uncertainty, capacity, negatives and permissible conditioning. Avoid matching post hoc on task answers that are intervention descendants.
5. **Return to diagnostic reporting after source sufficiency.** A validated attention-specific appearance correspondence motivates neutral reporting and blind characterization. Numeric scale fitting alone does not establish that accurate reports resemble consciousness reports.

The contrast archive can establish that visibility must not simply be renamed confidence, and can test ancillary predictive sufficiency. It cannot alone provide the attention-origin counterfactual. Brightness/visibility is one operational axis, not the definition of qualia or of the full research goal.

## Independence and limits

Only this primary article and the three source files listed in metadata were inspected. No model output, checkpoint, archive, training run, report batch or other review was inspected. The parent supplied the hypothesis, task mismatch and absence-of-traces concern; this is output-unexposed analysis within that framing, not blind hypothesis discovery. Shared subagent training limits independence; no independent human review or human contact occurred. No transfer performance was measured. The original theory-evidence objective remains unachieved.
