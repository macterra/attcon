# Independent post-output orientation diagnostic audit

The frozen synthetic engineering diagnostic passes its registered gates. This is
an upstream sensory-information result: the attention-control model, Controller,
Reporter and participant appearance judgments were not tested. It provides no
source-of-qualia or empirical human-stimulus-transfer result.

I inspected the frozen protocol, implementation, manifest, completion, numerical
results, raw tensors and visual architecture. A separate audit script, without
importing the diagnostic or calling its build/verify functions, regenerated the
renderer from the declared random seeds, recomputed frozen-checkpoint features,
fit the declared ridge systems, recomputed predictions and conditional bootstrap
intervals, and compared them with the retained outputs. Every comparison passed
exactly. Source/checkpoint/result/state hashes matched. The audit source and
per-model numerical checks are included in metadata.json for reproduction.

## Counts and registered gates

Each of the three checkpoints has 64 development base units and 64 evaluation
base units. Six contrasts and two counterfactual orientations produce 768 images
per split per checkpoint. The combined evaluation is 192 base units and 2,304
dependent images; there are not 2,304 independent orientation trials. All five
feature sets have exactly equal zero-contrast orientation features in both
splits, all zero-contrast images are equal, and every probe scores exactly 50%
there. With zero scores, the fixed tie rule chooses horizontal for both paired
labels, which yields one correct member per unit. That is the intended
zero-information control, not learned above-chance orientation recognition.

The physical directional-edge control and visual-hidden ridge readout each
score 100% at the highest evaluation contrast in all three checkpoints. This
passes the registered **point-accuracy >=90%** gate for all models. The protocol
did not gate on confidence-interval bounds or human-task accuracy. The bootstrap
intervals at this condition are [1,1] because all retained evaluation units are
correct; empirical resampling does not establish a population accuracy of one,
nor account for model-training/readout-estimation uncertainty. The conditional
256-draw intervals and all lower-contrast/probe descriptive results reproduce.
Maximum checked linear-system residual is below 6e-16.

Native color argmax is blue for all 768 evaluation images in each checkpoint.
Category-probability readouts retain some orientation information (highest-
contrast accuracies 79.69%, 95.31%, 75%), but that does not make blue an orientation
report or establish a production orientation interface. Hidden features are
extra instrumentation; the bound representation ordinarily keeps the eight
category probabilities, not the 64 hidden features.

## Scope and confounds

Centering uses both test orientations of each base unit and contrast. This is
legitimate under the registered paired-transformation estimand, but the resulting
accuracy does not establish isolated-image orientation classification. Fitting
uses only highest-contrast development pairs; evaluation geometry/noise/phase
are independently seeded, and all lower contrasts are excluded from fitting.
The development/evaluation data are not independent texture-class samples:
16-pixel four-cycle binary textures contain only four discrete phase-pattern
classes, all four of which occur in both splits for every checkpoint. Quantized
aperture masks also repeat between splits (9, 7 and 10 distinct overlapping masks
for checkpoints 1301, 1311 and 1321 respectively). The code/protocol already
acknowledge discrete pattern and mask repetition. Shared phase, aperture and
noise within an orientation pair do not give one class a systematically different
nuisance assignment; both labels occur once per base unit. These controls reduce
that nuisance route, while retaining a narrow fixed-frequency, paired-domain
inference. They do not establish generalization over unseen spatial frequencies,
textures, native stimuli or human viewing conditions.

The luminosity scaling, 16-pixel rendering, frequency, noise and exposure remain
synthetic choices. No photometric/native-stimulus equivalence, human appearance
correspondence, attention manipulation, attention-history mapping or recurrent
attention-model source information is established. I found no numerical or gate
failure that requires changing the frozen result. Further work must preserve
these limits rather than treating this ancillary pass as the scientific goal.
