# Physical-information source localization: results

Date: 2026-10-10. All reserved scenes completed; no retraining, reports, retries,
replacement models, category-based exclusions or participant appearance judgments.
The [frozen protocol](PHYSICAL_SOURCE_LOCALIZATION_V1_PROTOCOL.md) was reviewed and
committed before the single attempt. Human review was not a prerequisite.

## Finding

**Strong physical transformation signal is upstream in visual hidden features.**
The fixed diagnostic probes explain 98.91–99.42% of within-object pixel-mean
variation and 74.49–87.28% of neighboring-pixel squared-difference variation there.
The existing category, memory-masked bound and recovery-buffer representations give
much weaker prediction under these probes. The recurrent attention hidden vector
and forecast are **exactly invariant** across changed images with fixed history.
They do respond to a separate acquisition-history change.

This result locates objective stimulus information for this assay. It does not
establish participant appearance, absence of experience, or the source of qualia.
The broader bound model includes visual content and its access relation; the core
estimator's input boundary is not a test of all properties of that model.

## Reserved budget and measurement

Three existing visual/attention checkpoint pairs, each with 128 development and
128 held-out base scenes. Six paired pixel transformations per scene: scales
0.70/0.85/1.00, crossed with no blur or Gaussian sigma 0.7. Each scene has eight
objects. Thus the evaluation has **384 independent base scenes conditional on
these checkpoints**, 2,304 scene variants and 18,432 object-variant rows. Development
has the same budget. Related objects and transformed rows are not independent
replicates, and the three trained pairs do not justify a model-population claim.

Targets are pixel mean and neighboring-pixel squared difference. Features and
targets are centered within each object's six variants. The fixed development-only
ridge readout and scaling predict **transformation response**, rather than an
isolated image or an experiential judgment. No meaningful-appearance or theory
success gate is defined in this diagnostic. Some transformed images are outside
the original training distribution; no participant-stimulus conversion is tested.

## Held-out readouts

Each cell gives within-object R² as **pixel mean / edge-difference energy**.
These are fixed-probe scores, not percentages of all information retained.

| Source | 1301 / 2011 | 1311 / 2021 | 1321 / 2031 |
|---|---:|---:|---:|
| Upstream visual hidden (64) | 0.9942 / 0.8728 | 0.9935 / 0.8581 | 0.9891 / 0.7449 |
| Visual category probabilities (8) | 0.0109 / 0.0219 | 0.0195 / 0.0140 | 0.0117 / 0.0186 |
| Category max probabilities/entropies (4) | 0.0822 / 0.0518 | 0.1097 / 0.0566 | 0.0738 / 0.0565 |
| Memory-masked bound contents (8) | 0.0107 / 0.0217 | 0.0199 / 0.0136 | 0.0116 / 0.0182 |
| Recovery-mixed categorical buffer (8) | 0.0075 / 0.0150 | 0.0126 / 0.0109 | 0.0086 / 0.0126 |
| Category/q/interaction baseline (17) | 0.0080 / 0.0082 | 0.0191 / 0.0142 | -0.0011 / 0.0129 |
| Recurrent attention hidden (64) | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Attention forecast (64) | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

The upstream hidden vector is extra diagnostic instrumentation: production binding
retains category probabilities, not this vector. No existing Reporter received
these hidden features or newly decoded physical measurements. Category-confidence
features explain 7.38–10.97% of pixel-mean and 5.18–5.66% of edge-target variation;
this is consistent with a confidence-associated route, without establishing its
complete mechanism. The nonzero category/bound scores are not erased.

All per-source intervals are in [the raw results](../audits/physical_source_localization_v1/results.json).
The visual-hidden conditional 95% base-scene intervals, respectively for mean and
edge, are 0.9939–0.9946 / 0.8662–0.8792 for pair 1301/2011,
0.9930–0.9940 / 0.8499–0.8659 for 1311/2021, and
0.9882–0.9900 / 0.7278–0.7609 for 1321/2031. They condition on the fitted readout
and checkpoint, exclude readout-training and model-population uncertainty, and
are not corrected across sources. No significance or superiority threshold is
applied, and readout dimensionality/basis differs across sources.

## Controls and retained confounds

- Every recurrent hidden/forecast tensor is exactly equal across the six pixel
  variants at identical process inputs. Centered attention-only probe response is
  zero, so its R² is zero. Zeroing history acquisition separately changes the
  hidden state by a maximum 1.27–1.59 and the forecast by 0.828–0.838, excluding a
  constant or broken extractor. This invariance follows from the actual input
  boundary; it is not a newly observed consciousness prediction.
- Category identity is almost preserved, with evaluation joint accuracy
  99.707–100%. Confidence is **not matched**: within-object max-probability changes
  reach 0.390–0.505. No subset was selected to hide these changes. Identity retention
  alone cannot supply an appearance-versus-certainty dissociation.
- Current-access attenuation at fixed pixels changes the engineered categorical
  bridge (maximum 0.337–0.342). Access restoration is exact; allocation and effects
  stay fixed. The category/q/interaction baseline reconstructs the bridge exactly.
  Probe differences between those features and the bridge reflect feature basis,
  scaling and regularization; they do not identify new information generated by q.
- Binding rotation changes content at the location (maximum 1.0) with visual and
  attention marginals fixed. Binding restoration is exact. The known-object
  fraction is 98.14–98.54%; memory-masked bound contents and the unmasked functional
  categorical bridge are explicitly separate paths.

## Review and verification

A separate AI reviewer inspected the protocol/code before outputs, identified
missing raw-source retention and nonzero-intervention assertions, and reread their
corrections before registration. First-pass findings, corrected-source hashes and
an exposure clarification are retained. A separate data reviewer investigated
primary sources. These reviews share model training and commissioning context;
they are not human review or fully independent empirical validation.

Four meaningful diagnostic tests pass: scene-confound removal, detectable paired
signal, development-only scaling, and physical blur behavior. Exact replay verifies
all raw features, objective targets, category decisions, source inputs, intervention
tensors, coefficients, bootstrap draws and numerical summaries, plus source and
checkpoint hashes. The lossless compressed archive is
[states.pt.gz](../audits/physical_source_localization_v1/states.pt.gz); execution and
replay receipts are retained alongside it. Historical reporting and qualification
verdicts remain unchanged.

A [separate post-output AI audit](../audits/physical_source_localization_v1/results_review/review.md)
checks retained tensors directly and independently reproduces coefficients, scores
and intervals at the registered thread count. An alternative separable blur
calculation reproduces physical targets within 5.96e-8. It confirms 16 evaluation
object-variant rows change category argmax; those rows remain included. The audit
is unblinded and shares training/commissioning context; it supplies a computation
check rather than independent empirical appearance validation.

## Remaining theory-facing work

The [reference discovery review](../audits/physical_source_localization_v1/reference_review/review.md)
identifies [Rausch and Zehetleitner's same-trial clarity/confidence study](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00591/full)
and complementary attention/contrast controls. Author archives are documented,
but actual trial data, stimulus code, license and file schema are unverified here.
No dataset was downloaded or qualified. These numerical ratings are not a validated
free-prose consciousness-report reference distribution.

The next step remains a separately registered **reference-qualified appearance
assay**, with actual archived stimuli and judgments, participant-group separation,
explicit category/confidence/physical-metric controls, and valid input conversion.
Do not silently add the encoder's physical decoder to the reporting source and
call that an attention-origin appearance measurement. A future representation
revision would need to make a predicted presentation/access relation explicit and
validate it on held-out reference data, preserving this frozen result. Neither
strong hidden physical decoding nor weak categorical decoding completes that work.

The research goal—accurate attention-model reporting with credible independently
grounded consciousness-related correspondence—**remains unachieved**. Subagent
reviews remove the human-only dependency; they do not replace missing empirical
reference data or certify subjective experience.
