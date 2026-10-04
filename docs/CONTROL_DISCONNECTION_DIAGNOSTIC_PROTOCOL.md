# Fixed-model disconnection probes v1

2026-10-04. **Post-hoc engineering diagnostic**, chosen after observing the retained
[three-way study failure](FUNCTIONAL_CONTROLS_RESULTS.md). Freeze/publish this source
and design before executing the new probes. It cannot change the parent verdict,
qualify reporting, supply human review or establish consciousness-related features.

## Question and starting states

Do informative new command/observation pairs help the fixed model revise a retained
control relation after disconnection? How do observationally indistinguishable
command histories limit physical-wiring identification?

Use existing final models 2011/2021/2031 and all 512 episodes in each prior-channel
0→neither and 1→neither route. Start from the original 16-observation feedback state,
whose full recurrent/current/predicted values are archived in the
[visual/interface integration](FUNCTIONAL_INTERFACE_RESULTS.md). Obtain actual
automatic phase, direction and recovery from the original physical trace. Keep
modeled quantities and physical truth distinct. No training, replacement checkpoint,
new model seed, reporter call or best-window selection is permitted.

The initial misclassified subset is determined by the unchanged original control
mask criterion (command-effect TV >0.75 per buffer), before any new probe. Describe
its original command/allocation coincidence count alongside all episodes; this
post-hoc association alone is not a mechanism or significance claim.

## Three examiner policies and crossed physical continuations

For eight additional observations, acquisition quality 0.8:

| Examiner policy | Command sequence |
|---|---|
| Random | Uniform commands k0–k3, seed `770000000 + model_seed`. |
| Agree | At each step, select the upcoming automatic destination at the formerly controlled buffer. |
| Contradict | At each step, select that destination plus an independently uniform nonzero offset 1–3 modulo 4, seed `780000000 + model_seed`. |

Each policy is run in two physical worlds: continue with neither buffer controlled,
or restore the prior controlled buffer at the beginning of the continuation.
Both branches begin with identical physical recovery/phase and modeled state.
Reuse the same commands, automatic directions and qualities across those branches.
Policies agree/contradict deliberately use simulator phase; they are examiner
diagnostics, **not native Controller performance**. The learned model receives only
the resulting ordinary command/allocation/acquisition observations, never truth labels.
Reporter and Controller remain separate; this study has no Reporter calls.

Agreement commands make executed allocations and acquisitions exactly identical
in the two worlds, although their all-command counterfactual effect tables differ.
Require exact equality of observed histories, physical recovery, recurrent/model
states and prospective predictions. This is an identifiability/invariance control;
it is not a demand to distinguish worlds with identical available evidence.

Contradictory offsets vary, avoiding a fixed deterministic offset/command schedule.
In the disconnected world every new allocation contradicts the simple claim that
commands select their named position at the formerly controlled buffer. In the
restored world that command relation holds. These provide observable evidence
without changing model weights. Results in both worlds must be retained.

## Assessment and archival

At 1,2,4,8 additional observations, score control-mask accuracy, all-command effect
accuracy, and prospective recovery MAE against the common physical future of each
world. Track the original misclassified episode IDs and their subsequent correctness.
Report all 36 model/prior-channel/policy/world contexts and all windows, including
negative outcomes. Windows are descriptive; **there is no new success gate**.
Eight more observations do not move the original primary 16-observation endpoint.

Archive every observation, command, initial state, modeled allocation/access/effects,
recurrent state, physical target, prospective prediction and per-episode correctness
at each window. Record source/dependency/trace hashes, and replay every value without
training or API calls. Refuse overwrite/retry of an attempt. Save/publish preparation
and completed diagnostic results at their milestones.

The theory-facing goal remains credible evidence for the attention-control model as
source of qualia. Better physical prediction is an engineering component, not the
goal itself. Faithfully reporting a wrong internal prediction can still be model-
report fidelity. Independent rubric review, factual-audit qualification, powered
causal confirmation and replication remain outstanding. The author-review packet
and existing reporting failures are unaffected by this diagnostic.
