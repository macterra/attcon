# Spatial native task software contract v1

2026-10-10. New DEVELOPMENT primitives, reviewed before any production checkpoint
load, dataset generation or fit. This software artifact does not establish task
utility, presentation semantics, attention provenance or qualia. Historical frozen
sources and archives are unchanged. The root author supplies the separate protocol
and runner; this author owns only the task module, its tests and this software audit.

## Rendering and evaluator truth

`layout_for_world(index)` returns immutable evaluator-only role cells. The code
uses seven bits to enumerate all 128 ordered opposite-half cell pairs. For outer
address a, its layout code is `(index % 128) XOR ((32*a) OR (a % 2))`. The lowest
bit gives red's half; the remaining bits give red row/inner column and crossed
blue row/inner column. The crossing is invertible over the complete 128-code
orbit. Every world has two red-left and two blue-left items. In aligned 32-world
blocks, each role visits each of 16 cells twice at every address. Each address has
12 red-above, 12 red-below and 8 equal-row cases. Across 128 worlds each address
visits every ordered opposite-half pair once. `balanced_layouts` refuses counts
other than positive multiples of 32. Seed-independent balancing means the two
future splits share layout support; there is no withheld-layout generalization
claim. Future namespace seeds supply independent nuisance realizations.

`render_world(seed,index)` renders exactly four addressed RGB patches. Every patch
is 16x16 pixels, partitioned into 4x4 pixel stimulus cells. Neutral background is
RGB (.08,.08,.08). A square occupies offsets (1,1),(1,2),(2,1),(2,2); a cross
occupies (0,1),(1,0),(1,1),(1,2),(2,1). Tokens remain inside their designated cell.
Role RGB is red (1,.08,.08) or blue (.08,.08,1), multiplied by independent
U(.65,1) intensities. Each token independently chooses square/cross with equal
probability. Noise sigma is independent U(.005,.02) per world; additive Gaussian
noise draws are independent by pixel conditional on sigma. RGB values are clipped
to [0,1]. A SHA256-derived local torch generator isolates seed/index streams and
preserves global RNG state. Shapes, intensities and sigma are recorded separately
from truth. They are not consumer inputs.

`EvaluatorGridTruth.occupancy()` produces long [4,4,4] labels: background=0,
red=1, blue=2, with exactly 14/1/1 sites per item. This is a stimulus task grid,
not a claim that convolutional receptive fields equal those pixel cells. Truth
enters only explicit occupancy supervision/evaluation; the native API accepts no
truth, pixels or renderer object. Known-one-of-each task construction does not
test role presence: spatial softmax always assigns each role total mass one.

## Extraction and controls

`FrozenSpatialExtractor` accepts caller-supplied encoder/readout modules and
orientation mean64/scale64. It performs no checkpoint lookup/loading. It deep
copies the modules, sets eval mode and disables all parameter gradients. It also
copies standardizer vectors. Extraction is exactly `encoder.net[:6](flat_pixels)`
and preserves float32 [*,24,4,4] units. Controls applied to actual W retain native
`net[6:]` category logits and two independent four-class softmax components,
plus `net[6:9]` hidden64 and the frozen standardized 64-to-32-to-1 orientation
readout. Orientation is an off-task pathway. No historical module is edited.

## Head, normalizer and actions

`LocalOccupancyHead(24)` is one shared Linear(24,32), ReLU, Linear(32,3) network
applied at every site, yielding [*,3,4,4] occupancy logits. Its parameter count is
899; the explicitly separate raw-grid `LocalOccupancyHead(3)` has 227 parameters.
`occupancy_loss(logits,evaluator_target)` is unweighted mean site CE. There is no
fit loop or optimizer here. Tests perform only a gradient calculation on
handcrafted tensors and a fresh random toy encoder, with no parameter update.

`ChannelStandardizer.fit(fit_w)` computes float32 channel means/population standard
deviations over every leading/spatial dimension. Scales below 1e-6 become one.
Mean/scale are immutable tuples. The caller must enforce fit-only whole-world
inputs. `apply(W)` uses those fixed coordinates; operators act on native units
before standardization. Linear channel normalization preserves repeated T2/T4 sites.

`role_fields(logits)` computes red-minus-background and blue-minus-background
scores in float64, applies a sixteen-site softmax, then returns the original
dtype. Identical scores explicitly produce exactly 1/16 at every site. Occupancy
logits remain separately retained. There is no presence threshold or sensory
truth override. Full fields allow all pairwise site contrasts to be reconstructed.

`NativeController.consume(AddressedState)` uses only observed items' actual W,
the fixed standardizer and H. It produces an immutable numerical `NativeSnapshot`
containing addressed local logits and role fields. It does not read source V,
identity, raw orientation logit, delivered pixels or evaluator truth. Its snapshot
time is the completed state time. Unseen items are omitted and cannot supply a
default targeting answer. Tensor access to stored fields returns copied buffers.

`target(snapshot,address,role)` executes the argmax row/column, selecting the first
row-major cell on exact ties and retaining every tied cell. `coarse_order` sums
each role field against columns 0,1,2,3, returning red-left-of-blue, its reverse,
or unknown when absolute expected-column difference is <=1e-6. This is the
protocol's expected-column coarse definition; it differs from the initial review's
halves query. Unknown must count as incorrect in future scoring. These actions
are current native task outputs, not A_current/A_next estimates or report prose.

## Operators and software evidence

The module reuses existing `AddressedState`, `PendingOperator`, `block_average`,
endogenous/exogenous execution and saved restoration. T1/T2/T4 always start from
retained V. A pending future operation is a separate object and rejected by
`consume`; it is never recorded as completed W or interpreted as a prediction.
Nonselected addresses and retained source/identity/raw-logit metadata remain
preserved by the existing mechanics. This task introduces no A or Reporter.

The first run was saved with exclusive file creation in `first_run.log` before
any subsequent work. All 14 tests passed; there was no fix or retry. Tests cover
balancing metadata, a repeated four-patch renderer fixture, exact occupancy,
head parameter counts, target supervision isolation, role/background contrasts,
uniform/tie policy, W dependence and decoy metadata independence, address and
site permutation behavior, T2 block repetition, T4 homogeneity, saved-V recovery,
driver parity, unseen rejection, standardization, malformed inputs and source
gradient/weight preservation. The encoder test uses newly initialized random
weights and handcrafted pixels. It verifies category and orientation pathways
against direct forward calculations, without any production checkpoint or task fit.

No development worlds, checkpoint loads, learned task inference, optimizer updates,
report outputs, Git operations or network actions were performed by this author.
These software invariants do not substitute for independent static review or the
future per-encoder engineering measurements.
