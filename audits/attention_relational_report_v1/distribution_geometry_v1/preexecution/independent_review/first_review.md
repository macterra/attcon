# First preexecution method and software review

Decision: changes required before target execution. Separate reviewing agent; not an external scientific or semantic audit.

Source SHA-256: `70f11b5ddb3a76f7eb65fedd1698282543f4c5b9490146a9276b3b8dbe6c09a3`
Contract SHA-256: `bb507ffcee8c43711373ae38faa01239f0400c8d5b4d5bc79a53835ab2bda3dd`

No target numerical inputs/results, GitHub, model calls, or semantic ratings were inspected. No production or test files were edited.

- **R1 (blocking, source line 191):** Pair panel opportunities are the union of available query groups; panels unavailable on both sides disappear instead of receiving unavailable status. Derive panel opportunities from complete retained panel inventories and emit explicit unavailable entries.
- **R2 (blocking, source line 180):** score_rule is reconstructed unconditionally even when command identities differ in actual source or other non-command context. Later pair comparisons use only k0 common_identity, so this can also contaminate source-matched preference comparisons. Require exact common identity across all commands for a native score rule, otherwise retain explicit unmatched status without a preference; gate pair-score comparisons on that status.
- **R3 (medium, source line 211):** Final inventory assertions cover positions, original pairs, and groups but omit panel count, unavailable panels, command comparison count and full node-role-command coverage. Check the full fixed inventory and uniqueness, including 804 panels, 12 unavailable panels, 19008 command comparisons and complete both-role/three-command groups.
- **R4 (medium, source line 136):** Signed centroid displacement is a vector and therefore excluded from all summary quantiles by the scalar-only type filter. Only its L1 magnitude is summarized. Publish row and column displacement as separate scalar summary metrics or narrow the contract promise explicitly.

Geometry formulae, raw/normalized separation, units, linear-interpolation quantiles, and exact first-index ties are correct in reviewed code and bounded fixtures. Contract preserves narrator full-distribution access and limits phenomenal/semantic claims.

Bounded synthetic checks: opposite-corner distances and entropy, uniform-grid centroid/variance/peak ties, and first-index selector ties passed. Both-unavailable omission and cross-source command scoring were reproduced.
