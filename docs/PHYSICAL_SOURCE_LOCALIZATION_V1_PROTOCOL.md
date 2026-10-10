# Frozen physical-information source diagnostic

Date: 2026-10-10. Development diagnostic, registered before its new model outputs.
This is the first, physical calibration component of the proposed source-sufficiency
work. It is **not** the participant-appearance or consciousness-correspondence assay.
Independent appearance data must still be obtained and qualified separately.

## Question and scope

Where does information about controlled pixel intensity and edge degradation
survive in the existing frozen architecture? Distinguish the recurrent attention
estimator and its forecast from the integrated bound state and the upstream visual
encoder. Both the bound content relation and recovery-dependent categorical buffer
remain relevant model components; restricting a probe to the GRU would not test the
whole model. The Controller and Reporter are separate; neither is run here.

Pixel mean and neighboring-pixel squared differences are objective calibration
targets, not perceived brightness, felt clarity, qualia, or participant judgments.
Accurate physical decoding cannot achieve the research goal. Exact invariance of
an attention estimator to pixels does not falsify a theory about relational access
states or establish absence of experience. It only tests this input path.

## Fixed design

- Use all three existing pairs: visual checkpoints 1301/1311/1321 and functional
  attention checkpoints 2011/2021/2031. No selection, retraining, replacement or
  reporter generation.
- Per pair, generate 128 development scenes and 128 disjoint evaluation scenes,
  using scene seeds `860000000 + visual_seed + 10000 * split`, with split 0/1.
  History seeds are `870000000 + attention_seed + 10000 * split`, 12 steps,
  condition 0 of the existing physical simulator. History and scenes are independent.
- Each scene has two views and four objects. Preserve the same underlying image,
  object categories, geometry and process history across six pixel transformations:
  multiply by 0.70/0.85/1.00, crossed with no blur or fixed Gaussian blur sigma 0.7.
  Blur uses a normalized five-pixel kernel and replicate padding. These transformations
  include stimuli outside the original training distribution; no generalization to
  natural scenes or published participant stimuli is claimed.
- Keep all scene variants together across the development/evaluation boundary.
  The independent evaluation unit is the base scene, not its 48 object variants.
  There are three frozen model pairs, 384 evaluation scenes, 18,432 evaluation
  object-variant rows, and the same development budget. No pooled significance test.
- Measure all category argmax errors/retention and confidence drift. There is no
  claim that category certainty, task answer or human appearance is matched.
  No exclusion of category-changing scenes or post hoc selection.

## Sources and readout

Extract upstream visual hidden features (64; added diagnostic instrumentation),
visual category probabilities (8), category max probabilities and entropies (4),
actual memory-masked identity-bound contents (8), recovery-mixed categorical
contents (8), a category/q/interaction baseline (17), recurrent attention hidden
(64), and its flattened forecast (64). The hidden visual vector is discarded by
the production binding operation; probing it is not evidence that existing reports
had access to it. The functional buffer uses unmasked visual probabilities, while
the bound-content state masks objects never sampled. Report these as distinct paths.

For each base-scene/view/object, subtract the mean across its six variants from
both features and targets. This defines **within-object transformation response**,
not deployment prediction from an isolated image. Centering uses the fixed group
structure, not appearance labels or evaluation target values to fit coefficients.
Standardize feature columns using development RMS only; use scale 1 for zero
columns. Fit one fixed ridge readout per source and pair, minimizing development
mean squared error plus `0.01 * sum(weight**2)`, with no intercept on centered data.
No tuning or cross-validation. Evaluate both objective targets on every reserved
evaluation scene. Report within-group R² relative to zero centered response, and
95% percentile intervals from 256 base-scene bootstrap draws, seed
`880000000 + attention_seed`. These intervals condition on each frozen checkpoint
and fitted readout; they do not estimate model-population or human uncertainty.

Category+q+q*category contains an exact reconstruction of the engineered buffer.
Differences between their linear probes are differences of feature basis, scaling
and regularization, not new information created by the attention model.

## Causal and integrity checks

Recompute the recurrent model independently on the identical process history for
all six variants. Require exact equality of hidden and every forecast tensor.
As a non-degeneracy control, separately replace history acquisition by zero and
measure state differences; never claim that this control changes appearance.
Require nonzero physical-target variation and disjoint scene seeds.

At fixed pixels, halve current access. Measure the change in the categorical
bridge, preserve allocation/effects/visual inputs, and restore access exactly.
Reconstruct the bridge from category/q inputs exactly. Roll bound location-object
bindings with fixed visual content and attention, measure the relational change,
and restore exactly. These checks locate an engineered causal relation. They do
not turn q into an appearance variable or establish that a changed probe estimate
reflects a changed experience.

Freeze source/checkpoint hashes and this protocol before execution, reserve the
one attempt, retain every scene, coefficients and raw feature/target tensor.
Replay exact tensors and summaries. An integrity failure is retained as failure;
no revisions to the frozen attempt. All results are descriptive: **there is no
appearance or theory acceptance threshold in this diagnostic**.

## Interpretation and continuation

An upstream hidden-vector advantage locates physical information in the visual
encoder. Category/bound/buffer decoding can inherit that information. Attention
hidden/forecast invariance means these states cannot distinguish the controlled
pixel variants at fixed history; it says nothing about other representational
properties or the broader bound-model theory. Recovery and binding interventions
show defined model dependencies, already compatible with ordinary computation.

Next qualify archived participant-level appearance, task and confidence data and
its actual stimuli; predeclare a valid conversion into the frozen model's domain
or retain the transfer failure. Published scales alone are not an empirical prose
reference. Only a separate reference-qualified assay can test appearance judgments
and then source-faithful report correspondence. Subagent reviews replace the former
human-only gate, with common training and commissioning limitations disclosed.
