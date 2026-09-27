# Predictive attention language pilot v1

Development only; frozen before first request, 2026-09-27.
Model: `gpt-5-mini-2025-08-07`, account availability verified by model retrieval.
Responses API, low reasoning effort, structured JSON, maximum 2200 output tokens.
No consciousness examples, target conclusions, condition names, or theory name
are supplied. Two fixed neutral questions are applied to all conditions.
The [review rubric](PREDICTIVE_REPORT_RUBRIC.md) predates all reports.

Twelve fresh episodes from simulator seed 910000811, first twelve episodes,
decision boundary 7; target channel alternates, target slot cycles through four.
No selection by model accuracy or report attractiveness. Six conditions:

- factual A from the seed 811 pilot checkpoint;
- intervened A: allocation rotates one slot, access reverses across slots,
  command-effect channels swap (a compound synthetic intervention);
- physical process forecasts, holding the same world fixed;
- next episode's A, assigned cyclically;
- constant uniform allocation/effects and access 0.5;
- V only, no model forecasts.

Two prompts × 12 episodes × 6 conditions = **144 requests maximum**, each at most
2200 output tokens. At published $0.25/M input and $2/M output, the output ceiling
is $0.6336; expected total below $1 for these short inputs. Set an enforced
maximum of 144 attempts, no automatic SDK retries, archive errors as well as
successes. A failed request is not silently replaced. A subsequent revision must
use a new named pilot and retain this one.

Reporter sees a neutral glossary and target descriptor plus the two-channel
forecast. It emits free text and commitments: focality, target recoverability,
recoverability after two unattended steps, and more command-responsive channel.
Unavailable facts may be null. Mechanical fidelity compares commitments with
its *supplied* state, not with desired philosophical language. Differences from
original A are separately reported for outside/shuffled/constant controls.

This pilot checks implementation, prompt stability, and report diversity. It
cannot establish independent phenomenological correspondence or confirm the
hypothesis. The matched history-predictor control and separate allocation/access/
command-effect interventions are required in confirmation; this pilot's joint
intervention cannot establish field-specific independence.
