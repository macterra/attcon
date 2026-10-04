# Executed Controller tasks now have neutral reporting inputs

2026-10-04. The [frozen export protocol](FUNCTIONAL_TASK_INTERFACE_PROTOCOL.md)
connects retained Controller decisions to modeled-state reporting inputs, with a
versioned [source inventory](FUNCTIONAL_TASK_SOURCE_INVENTORY.md). Reporter and
Controller remain separate. No language reports, task trials, training, human
judgments or scientific success gate are added. All earlier failures remain.

## What is available

There are 5,184 archived input records: three models, three physical conditions,
four Controller policies, six source rows, four queries and six input variants.
Each condition/policy/query uses all six neutral node orders. Variants cover
before/after execution, represented-recovery attenuation, exact restoration,
reversible presentation remapping and missing relation information.

Every payload contains modeled allocation/recovery, bridge-derived object
distributions and prospective command consequences; full-information views also
contain the actual observation history. New task fields identify the query,
selected command, whether it executed, and the pending/answered/abstained response.
Executed commands match the latest actual observation; emitted responses match
the supplied current modeled readout under the declared joint cutoff.

Scene truth, physical recovery/phase/wiring, policy/condition labels and evaluator
outcomes are excluded from payloads. They remain in separate source archives or
outer record envelopes. Only `payload` is a reporter input. The interface signature
accepts no physical recovery, owner flag or correctness score. These sourcing and
replay checks do not establish semantic auditing or secure blinding in a public repo.

## Six source examples

[Download the example input views](assets/functional_task_interface_v1/examples.json).
These are supplied machine states and task actions; no Reporter has narrated them.
All examples use model 2011, source row 0, query p0 and the model-guided policy.
The output readout is n2, with no invented separate allocation/recovery fields.

| Example | Evaluator context | Modeled step | Selected command executed? | Task response |
|---|---|---|---|---|
| x1 | Before own-buffer execution | 12 | k0, no | Pending |
| x2 | After own-buffer execution | 13 | k0, yes | Yellow triangle |
| x3 | After represented-recovery attenuation | 13 | Same k0 execution | Abstained |
| x4 | After recovery restoration | 13 | Same k0 execution | Yellow triangle |
| x5 | After external-buffer execution | 13 | k3, yes | Yellow triangle |
| x6 | After disconnected execution | 13 | k2, yes | Yellow triangle |

The attenuation/restoration examples are separate readout branches of the same
executed acquisition. They do not add new world events or advance the observation
clock. Current recovery changes while prospective predictions regenerate from
unchanged effects/recurrent state and stay identical. This transient-head behavior
is retained rather than replaced by an intuitive persistent-loss narrative.

The same object answer in all three after-execution conditions illustrates why
object naming alone is insufficient for the proposed consciousness-report contrast.
Likewise, an engineered abstention is not a phenomenal threshold. Unknown identified
attributes, absent fields, unexecuted commands and withheld responses remain distinct.

## Checks and retained limits

Twenty relevant tests pass. Exact replay reproduces all 5,184 payloads and the
compressed archive/example bytes. Modeled effect restoration reproduces every
variant of the original policy's payloads; readout restoration also matches exactly.
Remapping covers task, history and forecast references and reverses without loss.
Missing-relation views retain the current modeled state and task decision unchanged.

The 4,272,503-byte record archive reuses existing episodes and checkpoints. Input
rendering does not establish fresh language generalization, factual fidelity of
prose, independent character ratings, a powered causal confirmation or replication.
The original review packet and language-development row remain unchanged; no
substantive author review, independent review or annotations have arrived.

The next measurement dependency is review of the rubric/theoretical contrast and
qualification of source-aware factual auditing, including this task vocabulary.
The project objective remains unachieved; this export closes a source-integration
component rather than replacing the consciousness-related evidence requirement.

## Archive and reproduction

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest tests.test_functional_task_interface tests.test_functional_decisions tests.test_functional_interface tests.test_functional_controls tests.test_functional_model tests.test_predictive_attention
.venv/bin/python scripts/export_functional_task_interface.py --stage verify
```

The [manifest](https://github.com/macterra/attcon/blob/main/audits/functional_task_interface_v1/manifest.json),
[summary](https://github.com/macterra/attcon/blob/main/audits/functional_task_interface_v1/summary.json)
and [complete record archive](https://github.com/macterra/attcon/blob/main/audits/functional_task_interface_v1/records.json.gz)
are retained with exact source snapshots and examples. Archive SHA-256:
`345205007704244a2cdcc1be34b42f1b00fad35d9be3e178f6e425056e5a1fca`.
The Pages example asset is byte-identical to the archived examples. Preparation
was published at `16e0b6c`; the completed export was published at `314c02c`.
