# Final evaluation progress

Protocol: [COMPLETION_PROTOCOL.md](COMPLETION_PROTOCOL.md).

- [x] Freeze completion criteria, experiment matrix, and scientific stopping rule.
- [x] Validate prospective environments, analytic policies, and equal-capacity controls.
- [ ] Run serial fitted controllers across architectures and seeds.
- [ ] Run routed fitted controllers across architectures and seeds.
- [ ] Run reward-only exploration and native answer/decline evaluation.
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
