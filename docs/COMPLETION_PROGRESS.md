# Final evaluation progress

Protocol: [COMPLETION_PROTOCOL.md](COMPLETION_PROTOCOL.md).

- [x] Freeze completion criteria, experiment matrix, and scientific stopping rule.
- [x] Validate prospective environments, analytic policies, and equal-capacity controls.
- [x] Run serial fitted controllers across architectures and seeds.
- [x] Run routed fitted controllers across architectures and seeds.
- [x] Run reward-only exploration and native answer/decline evaluation.
- [ ] Fit independent reports and run controlled report/policy interventions.
- [ ] Evaluate reserved stress contexts.
- [ ] Validate artifact manifest, reproduction entry point, and full test suite.
- [ ] Publish final evidence map and completed-project status, preserving failures.

Completion concerns the bounded prototype and final evaluation. A positive
conscious-access or Stage 8 result is not presumed.

Environment validation: seven tests pass; GRU48 has13,704 parameters and RNN87
has13,683 (21 fewer). All three head families have identical initial weights
within architecture/seed. Six task/seed partitions are disjoint.

## Serial GRU fitted replication

All three state controllers pass task viability and beat every fixed policy at
every cost. None reaches the required0.02 advantage over the fair cue comparator
at any cost (one gain is0.019965 and remains a failure). Source/configuration,
initialization, update-count, and gate-consistency checks pass.
Source: [serial GRU summary](../audits/prospective_serial_gru_summary.json).

## Reward-only serial exploration

All three from-scratch state controllers learn useful sequential acquisition and
pass fresh-answer/forced-verification viability. At cost0.25, two seeds exceed
the fair comparator by0.02; this does not replicate at all costs/seeds. At cost0.4,
answer coverage ranges46.1–100%, showing that native decline is used in some runs.
The complete support gate remains unmet. The learner trains only on experienced
chosen-action correctness/costs, with no full answer vector or report labels.
Source: [serial exploration](../audits/exploration_serial_summary.json).

## Reward-only routing replication

From-scratch routing policies pass fresh-answer and forced-observation viability
in all seeds/costs. State return ranges0.7324–0.7528 at cost0.1,0.6250–0.6528 at
0.25,and0.5250–0.5471 at0.4. No state policy clears the fair-comparator gain at
any cost. At cost0.4 none beats the native never-inspect/answer-or-decline policy
by0.02. Some policies decline; this is a learned payoff decision, not evidence
of introspective source reporting. Both registered exploration tasks are now
complete, including all negative comparisons.
Source: [routing exploration](../audits/exploration_routing_summary.json).

## Serial RNN replication

The nearly parameter-matched RNN passes fresh-answer and forced-verification
viability in all seeds/costs, but policy returns are lower and seed-sensitive.
State returns range0.5629–0.7592,0.4643–0.6161,and0.4443–0.4877 at ascending costs.
Only one seed at cost0.4 passes the fair gain; no complete support gate replicates.
This is a viable second architecture with weaker acquisition under this fixed
recipe, not evidence that gating is necessary. Serial cross-architecture work
is complete. Source: [serial RNN summary](../audits/prospective_serial_rnn_summary.json).

## Serial GRU reports and coupling, with comparator correction

Registered report gates pass, but the omitted-identity flaw prevents interpreting
that as fair superiority. The corrected full-answer-logit-plus-cue reporter
matches or exceeds the state reporter in every seed (state gains-0.0150 to0).
See [correction](COMPLETION_CORRECTIONS.md); original results are retained.

Answer-preserving quality transplants change94.5–99.2% of quality reports and
36.1–58.6% of later inspection trajectories, with zero initial inspect/answer
switches. Random controls change0–4.7% of reports and3.5–27.3% of trajectories.
Initial answer logits and restoration checks pass. The effect is a prospective
quality representation shared by readout and control, not a uniquely
introspective mechanism or a gain over fair policies.
Source: [serial GRU reporting](../audits/prospective_reporting_serial_gru_summary.json).

## Serial RNN reports and coupling

The original report gates also pass in the RNN, but the corrected identity-cue
reporter again exceeds state accuracy in all seeds (state gains-0.0414 to-0.0257).
All answer-invariance and restoration checks pass. Quality transplants change
86.5–93.0% of quality reports and6.1–16.7% of inspection trajectories. Random
controls sometimes change more trajectories and improve return, so the RNN
policy effect is not selective evidence of beneficial access monitoring.
The cross-architecture result is a decodable prospective cue, not a replicated
fair-report or fair-control advantage.
Source: [serial RNN reporting](../audits/prospective_reporting_serial_rnn_summary.json).

## Routing GRU fitted replication

All three GRUs learn effective sensor selection. State returns range0.8509–0.8643,
0.7509–0.7643,and0.6509–0.6643 at ascending costs. Task viability and advantages
over never inspecting and either fixed sensor pass in all seeds/costs. No state
controller reaches the0.02 fair-cue advantage at any cost. This confirms the
need to represent prospective information quality, but not a benefit beyond a
comparator given that same quality information.
Source: [routing GRU summary](../audits/prospective_routing_gru_summary.json).

## Routing RNN and full fitted matrix

All36 fitted controllers have completed with no failed jobs. Routing RNN returns
are lower and variable; one seed fails forced-observation viability. Only one
seed passes the fair gain at each cost, while other seeds lose substantially to
the fair comparator. No task/architecture group clears all registered costs and
seeds. The protocol's prerequisite for promoting independent-content convergence
is therefore not met; the earlier unforced-convergence evidence stays unchanged.
Source: [routing RNN summary](../audits/prospective_routing_rnn_summary.json).
