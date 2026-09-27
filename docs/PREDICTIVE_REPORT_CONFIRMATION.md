# Reporting confirmation v1: explicit target interface

Frozen before requests on fresh confirmation contexts, 2026-09-27. This follows
retained language pilots v1 and v2. V1 demonstrated position/index mistakes and
three token-limit failures plus one timeout. V2 changes style only and remains a
pilot regardless of outcome. This protocol changes the input interface, not A.

## Interface and sample

Use confirmed attention models 901/911/921. For each, simulator seed+970000000,
first four episodes, boundary 7. Channel alternates, slot cycles through four.
No episode selection on predictions. Each model sees its episode history; the
reporter receives target-specific continuous A values: allocation probability,
maximum allocation in that channel, current and delayed recoverability, and each
channel's command-effect variation. These are deterministic reductions of A, not
experience labels. Include target identity from V, held fixed in interventions.
Archive the full A tensors as well as the report interface, preserving identity
and the derivation. Explicit labels avoid array-index ambiguity.

Ten conditions: ordinary A; allocation-only rotation; access-only slot reversal;
command-effect channel swap; exact restored A; physical-process forecasts;
independently fitted history predictor; cyclically shuffled A; constant A; V only.
The matched history predictor is the corresponding development model 811/821/831
and receives the same episode observations, with no access to the target A.
It has the same capacity/training budget and predicts the same quantities but is
not used by this controller. This is not a control without an attention model.
Its success is compatible with accurate A reporting; its limitations are explicit.

Two interfaces have identical facts and question. Neutral: report only supported
facts in up to three prose sentences. Styled: add ordinary first-person language
and qualitative prose, keeping numbers in commitments. No experience examples or
claims, theory name, condition name, or desired conclusion is supplied. Style is
an experimental manipulation, never evidence by itself.

3 seeds × 4 episodes × 10 conditions × 2 styles = **240 attempts maximum**.
Fixed reporter `gpt-5-mini-2025-08-07`, low reasoning, JSON schema, 4096 maximum
output tokens, 120-second request timeout, no retries, at most eight concurrent
requests. Output cost ceiling $1.96608 at $2/M, expected total below $2.25 with
short target records. Retain all attempts and usage. Errors count as failures.

## Frozen mechanical criteria

For each seed and style, across ordinary and the three single-field interventions:
at least .95 complete commitment accuracy (focality, two access probabilities,
responsive channel; numeric tolerance .01). Restoration must exactly restore all
four commitments. For each paired single-field intervention, all changed factual
commitments must follow their supplied state, and all unchanged commitments must
remain correct. Aggregate paired correctness >=.95, with missing outputs failing.
Constant/missing state must produce no unsupported non-null commitments.
Report all controls' fidelity to their own supplied state and to original A,
plus correspondence to the altered A. Do not count difference alone as fidelity.
Report uncertainty and small sample size; four episodes per seed is a limited
confirmation of an interface, not broad generalization.

The model's worst reconstruction errors are retained, not excluded. Correct
reporting is evaluated against A even where A disagrees with the actual process.

## Report character and goal

Use the previously frozen blinded rubric. Independent assessment must evaluate
prose fidelity as well as manner of access, graded presentation, and agent-object
relation. First-person style, JSON fidelity, control utility, and the implementing
agent's impressions cannot fulfill this requirement. Raters' agreement and
condition differences must be reported. The pilot rubric has no preregistered
aggregate phenomenology threshold, so any threshold chosen after ratings must
be exploratory and tested on new reports. This confirmation cannot by itself
establish that final theoretical criterion. The overall goal remains active until
the report-character evidence and competing explanations have been addressed.
