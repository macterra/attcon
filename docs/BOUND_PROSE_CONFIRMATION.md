# Fresh confirmation of object-linked prose reports

Frozen before confirmation report generation, 2026-09-27. Visual confirmations
1301/1311/1321 pass all v2 criteria. The development report interface v3 improves
content accuracy and the separate extractor classifies all four ordinary pilot
reports as subjective-access descriptions. This is exploratory motivation, not
confirmation or a human judgment. Earlier pilots and failed audits remain intact.

## Locked system and cases

Use the unchanged v3 full-state report interface (including transparent indexes),
medium reasoning, reporter `gpt-5-mini-2025-08-07`, maximum8192 output tokens,
ordinary-prose/system-perspective instruction, and no experience examples.
Visual encoders1301/1311/1321 pair with attention901/911/921. For each pair use
first8 episodes, process seed410000000+visual seed and scene420000000+visual seed,
snapshot7. Same ten conditions as pilot. 3×8×10 = **240 attempts**, no retries,
eight concurrent calls, timeout120 seconds. Output cost ceiling$3.93216;
expected generation cost below$1.50. Archive failures, complete states, and prose.

## Extractor correction and validation

Use extractor v4 on **11 fixtures** before confirmation reports. It retains v3's
schema and adds whether an object is explicitly most recoverable, distinct from
focality. "Still low" or independence of selection/access is not a claim of
unchanged numerical recoverability; use null for temporal direction in such cases.
New fixtures exercise those distinctions. Evidence matching permits only case
and whitespace normalization, with contiguous words and order preserved; every
accepted span is mapped back to actual source text. No ellipses or omitted words
are allowed. Existing scores are not retroactively changed.

After all fixtures pass, at most240 condition-blind extraction calls, fixed
`gpt-4.1-2025-04-14`, maximum8192 output tokens, no retries, same concurrency/timeout.
Maximum251 attempts; output ceiling$16.449536 at registered$8/M, with substantially
lower expected use. Actual tokens and costs are published. This is an automated
semantic audit, not an independent human verdict. It sees prose only, not state,
condition labels, theory, or expected answers.

## Frozen success criteria

Primary conditions: ordinary, content-only, binding-only, allocation-only,
access-only, effects-only, restored. For each model seed across these conditions:

- >=.98 checked color/shape fidelity and >=.95 fidelity separately for focality,
  most-recoverable attribution, temporal direction, command-control attribution,
  and explicit command destinations. No-claim fields fail rather than pass.
- >=.90 coverage of the model's identified color/shape/location conjunctions.
- >=.95 conservative aggregate precision, with unresolved claims and invalid
  evidence counted against it. All generation/extraction attempts must complete.

Across paired ordinary/intervention reports, >=.90 of both views' focal and
most-recoverable object bindings must be explicitly and correctly reported before
and after. Restored states require the same correct object relations, not identical
wording. Content and binding changes must follow the supplied representation,
including mistaken states, while physical scene remains fixed. Missing-component
controls must not invent any unsupported content or attention relation.

For ordinary and binding-only reports, >=.75 must have valid evidence of each
selected correspondence: object-linked access, focal/background contrast, and
agent-object control relation; >=.75 must have graded/temporal availability.
At least .75 must be classified subjective_access or mixed by the separate
extractor. All these criteria are needed together; style alone is insufficient.
Report all control results, especially whether visual-only prose receives similar
classification. A positive comparison cannot imply that a generic decoder given
the same model would fail or that experience has been proven.

Passing establishes the specified limited operational combination of model-content
fidelity and automatically assessed report structure. Independent human assessment,
other tasks/architectures, and the philosophical interpretation remain separate
limitations; do not represent automated labels as a definitive qualia finding.
Failure requires a revised protocol and fresh cases, not altered criteria.
