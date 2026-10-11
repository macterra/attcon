# Reconstruction fixtures: first source-only draft

2026-10-11. Separate task allocation within the same model team, inherited project
and hypothesis context; no independent family or human endorsement. Ten fixture
definitions were written and AST/syntax compiled, never imported or executed.
The main verifier remains unchanged at its preserved source-only milestone
`31d57ef` while the first static review is being published.

Draft [fixture source](../../../tests/test_verify_attention_content_model_v1.py)
SHA-256 `e610269349758b19352ba010fd3b0bf7dc6fe48103f08c5ff9cd7381a6cd0c3a`.
Verifier SHA-256 `0332633eaef01755b458ec6c7b21b9373f1e949f79b295a3d2471c4141737902`.
Verifier design SHA-256 `d9125a40fb23b05f94d356e0886b178937cf345f7e6d813b41cbb294e51c0b26`.
Engine E1r source SHA-256 `4f537c112c3660f0dbd60d72f751a806f83f00b05224d42e6e62699452db8b45`.
These are source bindings, not passing fixture/runtime evidence.

## Intended bounded checks

The declared configuration is four CPU threads and deterministic algorithms.
Handcrafted numerical maps have two samples, 24 channels and4×4 sites; these are
neither reserved task worlds nor source-encoder outputs. Fresh weights come from
explicit fixture seeds71/72/73, isolated with torch.random.fork_rng. There is no
optimizer, backward, registered source fitting, renderer, Reporter, native checkpoint
or archive. FitNormalizer.fit computes fixture moments only.

1. Independently expanded GRU gates versus a fresh torch.nn.GRUCell forward.
2. Independent residual-decoder linear equations versus fresh Linear/ReLU modules.
3. Population moments versus sums/squared sums, constant-channel fallback, normalized
   MSE zero/one identities and rejection of non-fit normalizer context.
4. Block operators versus literal 2×2 and4×4 means, family order and unchanged input.
5. Native H equations in itemwise shape versus fresh local-head modules, independent
   exp/sum fields versus the declared role_fields helper, and exact uniform fields.
6. Literal content discrepancies/accuracy, coarse order, entropy and exact uniform
   ties versus explicitly placed point/uniform fields.
7. Paired sorted/interpolated quantiles versus torch.quantile on256×4 handcrafted
   resample indices, including A-minus-ordinary differences.
8. False/missing committed-byte bindings and short-commit rejection with mocked Git
   streams; no Git subprocess actually runs in a fixture.
9. Exact discrete/dtype comparison failures and retention of a named false gate.
10. Missing registration retains a failed complete fixture report; exclusive report
    creation rejects overwrite, with native acceptance and tensor loads forbidden.

Independent reference here means separately written equations/checks within the same
Python/PyTorch environment and model team. Fresh native module comparisons rely on
the same CPU primitive library. Literal arithmetic and explicit point/tie fixtures
provide distinct expected values; they do not establish production archive correctness,
phase sealing, full optimizer trajectories, empirical success or semantic qualification.
Mocked byte streams test fail-closed control flow rather than authentic Git behavior.

## Defects identified before execution

The coordinator statically identified a fixture API mistake at line121:
role_fields(item) returns a Tensor, so the draft's .tensor() call would raise
AttributeError. The author confirms the inspected function returns a Tensor. The
draft remains unchanged for preservation. A narrow correction removing .tensor()
must follow published first source/review evidence and a separately reviewed release.
This is a source-review defect, not an executed fixture failure.

The coordinator also flagged potential main-verifier failure preservation when
json.dump(allow_nan=False), outside its runtime exception guard, encounters a
nonfinite primitive copied into audit evidence. That concern awaits the first
critical review. No main fix or test rerun has occurred. If confirmed, prepare a
separately preserved fix with explicit nonfinite evidence representation and a
bounded serialization-failure fixture; keep gates/tolerances/check inventory intact.

No fixture execution is currently approved. First execution requires preservation,
separate static approval, exact corrected hashes and explicit authorization. Capture
one unchanged exclusive first log/result, preserving every failure before any repair
or retry. These fixtures cannot produce a registered fit or a production numerical
pass. Controller remains distinct from Reporter; engineering checks cannot establish
qualia or presentation, and the scientific goal remains unachieved.
