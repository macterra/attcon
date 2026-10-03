# Replication and self versus label: results

2026-10-03. **Verdict under the frozen rule: `replicated_attribution_follows_label`**,
the predicted outcome. [Protocol and amendment](SELF_LABEL_PROTOCOL.md).

Extractor v14 (after a blocking v13 fixture failure, amended before any report);
fixtures 51/51. All 144 reports generated and extracted; no failures or retries.
`coupled_table` fidelity: color 108/108, shape 108/108, focal 129/129, most
recoverable 203/203, access trend 324/324, under own control 255/268 (95.1%),
command next 317/317; coverage 98.2%; conservative precision 99.2%.

## Registered contrasts (48 paired episodes)

| Contrast | Flag | Counts | Difference | Discordant | One-sided exact p | Threshold |
|---|---|---:|---:|---:|---:|---|
| Replication: `coupled_table` vs `independent_table` | self-coupled access | 26 vs 3 | 47.9 points | 25 vs 2 | 2.8 × 10⁻⁶ | met |
| Self vs label: `coupled_table` vs `external_table` | stated content dependence | 41 vs 39 | 4.2 points | 5 vs 3 | 0.36 | not met |
| Attribution (descriptive): same pair | self-coupled access | 26 vs 2 | 50.0 points | 24 vs 0 | 6 × 10⁻⁸ | — |

| Condition | Self-coupled | Stated dependence | Subjective access | Mixed | Technical |
|---|---:|---:|---:|---:|---:|
| `coupled_table` | 26 | 41 | 11 | 37 | 0 |
| `external_table` | 2 | 39 | 0 | 48 | 0 |
| `independent_table` | 3 | 6 | 9 | 38 | 1 |

**Inspection.** The `self_coupled_access` flags in the two controls are false
positives (uncertainty without attribution) or misreadings of a flat table as
varying. The six `independent_table` dependence flags are mostly the same
misreading ("after any command I expect to hold upper's contents strongly").

## Findings

- **Replicated.** With twice the sample, an explicit, learned self-access table
  yields reports that the system's own grasp of objects depends on its commands
  (26/48), against 3/48 when its content does not depend on attention. v3's
  threshold-level result holds up and is stronger here.
- **The dependence is reported regardless of label; the self-attribution follows
  the label.** Relabelled as an outside camera's legibility, the identical table
  yields the same rate of stated dependence (39/48 vs 41/48), now attributed to the
  camera (self-attribution 2/48).
- **Descriptive, not registered:** subjective-access character appears in 11/48
  self-labelled reports and 0/48 camera-labelled ones (paired 11 vs 0 discordant,
  p = 0.0005). With an explicit self-access representation, the self label shifts
  some reports from mixed to subjective-access character.

## What this means for the goal

The project's evidence plan requires accurate reports of the attention-control
model with independently specified consciousness-report features. Across the
studies:

1. Accurate, intervention-following reports of the attention-control model: met (v5).
2. The originally specified structure features: met, but not specific; they follow
   the state's format ([specificity](SPECIFICITY_RESULTS.md)).
3. A theory-derived feature, self-coupled access: absent when the coupling is only
   implicit ([self-coupled](SELF_COUPLED_ACCESS_RESULTS.md)); present, following the
   self-model, and replicated when the self-access relation is explicitly
   represented ([table](SELF_ACCESS_TABLE_RESULTS.md); here).
4. Self-specificity beyond labelling: not shown. The reporter states the same
   dependence for whichever entity the representation names.

This is the furthest this paradigm can go. A pretrained language model reporting
from a described state cannot separate "a model of the system's own access" from
"a model labelled as the system's own access". Separating them needs a reporter
whose reports do not depend on supplied labels, such as a native reporter trained
on the system's own state without phenomenological targets. That would be a new
research program, not another run of this design. None of these results establishes
subjective experience.

Archived in `audits/bound_content/self_label_v1/` and its fixture folders. Rescore
offline with `PYTHONPATH=scripts .venv/bin/python scripts/self_label_gates.py`.
Tests: `tests/test_self_label.py`.
