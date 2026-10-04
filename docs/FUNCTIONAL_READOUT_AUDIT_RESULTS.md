# Functional readout audit: identity persists while declared availability changes

2026-10-04. The [frozen protocol](FUNCTIONAL_READOUT_AUDIT_PROTOCOL.md) audits
the existing visual/neutral interface. It introduces no language reports, task
trials, training or success gate. All prior failed verdicts remain unchanged.

## What the bridge changes

For a visual category distribution `v` and modeled recovery `q`, the bridge is
`b = q*v + (1-q)/4`. For two categories, `b_i - b_j = q*(v_i - v_j)`.
Whenever `q > 0`, the bridge preserves their ordering and dominant category.
It changes confidence and the interface's thresholded identification/abstention.
Raw argmax is an audited candidate decoder, not an existing native Controller
decision rule.

The retained audit covers output buffer 0: 512 episodes, four positions, three
model seeds, three physical conditions and neutral/model-effect-swap variants
(18 contexts). Current and all-command future color/shape argmax match the
visual representation in every context, before and after the reporter's
five-decimal rounding. No position-episode changes dominant identity across
commands. These are correlated reuse of existing states, not independent trials.

All modeled recovery values are positive: the current minimum is 0.0193376 and
the future minimum is 0.0076628. Joint category accuracy against scene truth is
the inherited visual accuracy (99.9023%, 99.9512%, 100% for seeds 2011/2021/2031).
Physical current recovery can be exactly zero: 59, 66 and 76 of 2,048 positions
in the respective buffer-0-controlled cases. Physical and modeled recovery are
separate quantities.

## Threshold-dependent command effects

The interface identifies an attribute only at probability at least 0.6; otherwise
it returns null. The following counts are position-episodes where either future
attribute's identified/null result differs across the four commands, out of
2,048 per seed. Counts list seeds 2011, 2021 and 2031 in order, for neutral variants.

| Physical condition | Modeled threshold variation | Physical threshold variation |
|---|---|---|
| Commands control buffer 0 | 1,764 / 1,785 / 1,772 (86.133–87.158%) | 1,759 / 1,772 / 1,744 (85.156–86.523%) |
| Commands control buffer 1 | 80 / 80 / 87 (3.906–4.248%) | 0 / 0 / 0 |
| Commands control neither | 48 / 70 / 73 (2.344–3.564%) | 0 / 0 / 0 |

Nonzero modeled effects in physically independent conditions are forecast
differences crossing the cutoff; they do not demonstrate physical access changes.
After swapping modeled effect channels, threshold dependence moves mainly to
the external-control condition, while dominant categories remain invariant.
A faithful report of an incorrect forecast still demonstrates internal-state
fidelity; it does not validate the forecast against world truth.

## Functional and theoretical limits

The current integration builds forecasts, histories and reporting inputs. It
does not execute native category-response actions or retain category-task action
traces. Earlier `BoundState.content_policy` experiments do execute content-directed
control; this finding concerns the newer visual/neutral interface alone.

The next functional assay must specify and execute an actual Controller decision
rule and its information boundary. A confidence-sensitive abstaining rule would
be a new engineered choice, whose behavior must be measured. The current readout
boundary is an experimental definition, not an established phenomenal boundary.
Preserved identity does not invalidate faithful internal-state reporting, require
category changes for qualia, or introduce task necessity as a project criterion.

Consciousness-related interpretation still requires substantive definition review,
independently justified report features, qualified factual auditing, powered
confirmation and replication. Author review is coordinated, with no substantive
review or annotations received. This audit supplies no human judgments.

## Archive and reproduction

The [summary](https://github.com/macterra/attcon/blob/main/audits/functional_readout_audit_v1/summary.json)
and [all 18 metric rows](https://github.com/macterra/attcon/blob/main/audits/functional_readout_audit_v1/metrics.csv)
are archived with source hashes and a lossless state archive. Exact replay of
every retained array, metric and archive digest passed:

```bash
.venv/bin/python scripts/audit_functional_readout.py --stage verify
```

The state archive SHA-256 is
`3a1048b5a7e325fc412bf03c2af2ef8c2f575ea6e007ab00b553879a05002f13`.

## Subsequent implementation

The [explicit Controller decision assay](FUNCTIONAL_DECISIONS_RESULTS.md) now
executes a declared inspect-then-answer rule, with actual-world feedback and
separate model-only command/readout interventions. It supplies the missing action
traces while retaining this audit's category-invariance finding and the engineered
cutoff limitation. No new language or consciousness-related human ratings exist.
