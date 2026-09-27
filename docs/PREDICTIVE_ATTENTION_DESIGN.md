# Predictive attention model: implementation and pilot design

2026-09-27. This is a new version following the completed inspection-model study.
The goal is positive, controlled evidence about consciousness-report structure;
finishing a matrix alone does not achieve it. Pilot results may inform a new
frozen confirmation protocol, with fresh contexts and retained failures.

## Mechanism and authoritative state

Use two processing channels, each with four object slots. One channel follows
the agent's allocation commands; the other follows an independent replay
sequence. Which channel is controlled varies across contexts. Object identities
and properties V are separate from the attention model and remain unchanged in
allocation/access interventions.

A recurrent model A receives allocation observations, signal strength at sampled
slots, and preceding allocation commands. It predicts current recoverability,
recoverability after unattended delays, and next allocation under each possible
command. It never receives a consciousness, phenomenology, self, presence, or
ownership label. Training targets are simulator allocation and reconstruction
outcomes. A channel's controllability is computed from variation in A's predicted
allocation under alternative commands, not supplied as an owner flag.

The controller selects commands using A's predicted effects on the queried
object's processing. Thus A is a model of a process used in control, not an
arbitrary latent renamed after the result. Supervised prediction is permitted;
it is an engineered model, not a claim of spontaneous emergence.

The initial implementation uses a shared GRU and distinct heads for allocation
and recoverability predictions. Archive A's continuous output state, h, V,
actual processing state, and command consequences separately. Direct access to
A provides the report interface's authoritative state; it is not an extra
phenomenology label. A factorized interface avoids the previous global MLP's
fragile reconstruction of unrelated fields.

## Independently motivated report distinctions

The attention-schema account motivates agent/object/attention relations and
model/process disagreement. [Webb and Graziano (2015)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00500/full).
Human perceptual-awareness work distinguishes reported perceptual clarity from
mere objective correctness. [Early Local Activity in Temporal Areas Reflects
Graded Content of Visual Perception (2016)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00572/full).
These sources motivate distinctions, not a mapping from reconstruction probability
to felt experience or a claim of a human-equivalent scale.

| Report relation | A's independently validated property | Counterfactual test |
| --- | --- | --- |
| Focal vs accessible but nonfocal vs unavailable | Allocation and reconstruction forecasts are separate | Change allocation while object identity and current access are fixed. |
| Increasing/decreasing access quality | Graded reconstruction forecasts and retention | Change modeled access, leaving physical input and V fixed. |
| Own actionable relation vs replayed view | Effects of commands on predicted allocation | Swap the modeled command-effect tables while observations remain fixed. |
| Continuity/change across attention shifts | State forecasts at consecutive decision boundaries | Keep V fixed and test predicted report changes, including mistaken states. |

Controls: V only; physical process trace; a matched history predictor; shuffled,
constant, or missing A; restored A. Reporters and question variants are identical
across conditions. A control containing A is labeled as such. Failure of a
poorly informed or weakly trained control is not unique-source evidence.

## Language reporter and reviewer separation

A fixed dated language model receives a neutral input glossary, continuous model
forecasts, object descriptors, and one of two neutral prompts. It receives no
reference consciousness reports, theory name, condition name, desired conclusion,
or demonstration of experiential prose. It returns both free text and scored
factual commitments. This contract supplies output syntax, not desired answers.
No language output is used to update A.

Use the existing GPT-5 mini snapshot only if available; verify availability before
committing an API experiment. It supports the existing Responses/structured-output
interface. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).
The next API stage will pin a model, prompt, request count, token ceiling, and
budget estimate before requests. Pilot requests are not confirmation. Log all
requests/responses and usage, never credentials. No spending on failed internal
prediction models merely to obtain attractive prose.

Mechanical evaluation measures semantic commitments against A, complete report
coverage, conditional uncertainty, changed-field following, and unchanged-field
invariance. Independent condition-blind review assesses whether the free text
expresses a subjective manner of access rather than only a log or object fact.
The primary agent and the reporting model cannot supply that independent review.
A review form has been requested from the user; published reference work supplies
the initial rubric while implementation proceeds. If actual independent ratings
remain unavailable, mark that evidential requirement pending, not satisfied.

## Iteration and success

First validate predictive semantics and causal consumption locally. Use pilot
contexts to identify implementation failures. Freeze independent seeds, fresh
confirmation contexts, report prompts, metrics, and acceptance criteria only
after the pipeline is defined. Preserve every failed version and require new
confirmation after substantive changes. Do not weaken criteria on test outcomes.

Goal achieved requires reliable model predictions, faithful reports including
counterfactuals, the selected consciousness-report correspondences, and controls
that reject explanations based solely on language style or outside information.
A successful same-state simpler decoder is compatible with the goal. Whether
these findings prove subjective experience is not a success requirement.
