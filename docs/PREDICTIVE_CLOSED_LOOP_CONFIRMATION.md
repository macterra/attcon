# Closed-loop confirmation on the reported models

Frozen before execution, 2026-09-27. The original internal confirmation tests
forecast accuracy and decisions at a fixed boundary; the earlier full closed-loop
extension used development checkpoints. This additional check executes the exact
three models used in report confirmation (901/911/921) in the environment.

Use `predictive_closed_loop.rollout`, seed+980000000, 1024 episodes, 16 steps,
four-step burn-in. Conditions: ordinary, command-effect rotation, restoration,
and shuffled effect models. Hold all exogenous random draws fixed within pairs.
Each seed must select controlled queried slots >=.99 ordinarily, <=.01 under
rotation, and recover every trajectory tensor exactly after restoration. Report
access MAE and reconstruction outcomes without a new threshold. This confirms
causal control by the reported models; task necessity is not a qualia criterion.
