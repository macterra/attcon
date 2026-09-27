# Object-linked attention-model reporting

Started 2026-09-27 after the user inspected the technical reports and requested
reports of the attention model's contents. Previous assays and failures remain
unchanged. Goal: accurate reports of represented objects and their modeled
attention relations, with a separately assessed consciousness-report structure.

## Mechanism

Render colored shapes as small RGB patches. Train a visual encoder to recognize
color and shape, using ordinary perceptual targets. Its remembered distributions
V are separate from the previously validated predictive attention model A. A
binding matrix links A's spatial entries to V's object representations. The full
bound model contains object-content distributions, allocation/access predictions,
and predicted changes under each attention command. No phenomenological label
or sample report is used to train perception or attention.

The binding is an explicit architectural relation, not claimed to emerge. A
content-addressed controller uses the bound model to choose a command for a
requested color/shape. Validate that changing bindings with V and A's marginal
predictions fixed changes both selected objects and subsequent reports. This
makes the binding part of the used attention model, not a report-only join.

The visual vocabulary is deliberately small. These are simulated visual contents,
not natural vision or a probe of the language reporter's own internal attention.
Perception and prediction mistakes remain possible and must be archived.

## Cycles and criteria

1. Implement perceptual scenes, learned visual encoder, remembered object state,
   bound model, and content-directed control. Test selective interventions.
2. Pilot perception/control accuracy; retain all outcomes. Freeze fresh seeds
   and cases for confirmation after the mechanism works.
3. Provide the complete bound state to a frozen language reporter. Ask open
   questions about currently available contents and attention shifts. Generate
   free prose before any factual extraction; do not force experiential vocabulary
   or a list of report fields in the generation schema.
4. Test content-only, allocation-only, access-only, binding-only, restoration,
   and missing-component controls. Keep object identity/physical scene fixed
   except where explicitly intervened. Report must follow the model, including
   synthetic mistaken content, not external ground truth.
5. Assess actual prose, not only a parallel JSON answer. Archive full responses,
   extract claims with evidence spans, and independently review whether they
   exhibit the proposed manner-of-access distinctions. Generic first-person
   wording or spontaneous-sounding sentences are not sufficient.
6. Replicate successful criteria on fresh contexts, archive reproducible results,
   and update the preprint with actual reports and remaining limitations.

A pilot success does not complete the theory-facing goal. If the full criterion
cannot be established, preserve that fact rather than relaxing it to finish a
cycle count. Engineering the binding/report interface is permitted; necessity
for task performance and superiority over same-state decoders are not gates.

## Reporter

Retain `gpt-5-mini-2025-08-07` for continuity, verify account access, and use
Responses API. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).
Freeze exact prompts and bounded request/token limits before each API experiment.
Do not infer the state from first-person language. The reporter receives full
bound state rather than a single target's selected summaries. Independent
assessment remains distinct from the generating model and implementing agent.
