# Audit Open Issues Checklist

This tracks the remaining items from the Priority 1 audit review in `FEEDBACK.md`.

## Immediate Fixes

- [x] Fix `PRIORITY1_AUDIT_STATUS.md` so Stage 6A matches the smoke artifact.
- [x] Fix `PRIORITY1_AUDIT_STATUS.md` so negative-control status matches the smoke artifact.
- [x] Split Stage 6B smoke output into gated audit signals and informational signals.

The smoke artifact is now generated with deterministic metric seeding, and the current
artifact/doc agree on Stage 6A and negative-control status.

## Follow-Up Work

- [ ] Run a full evaluation on a checkpoint trained after the Stage 4B feedback objective.
- [ ] Commit or otherwise preserve the resulting full evaluation report and quote concrete JSON paths from it.
- [ ] Add empirical permuted-label noise-floor calibration for audit thresholds.
