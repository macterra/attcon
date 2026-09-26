# Reporting comparator correction

Identified during the final matrix, after the three serial-GRU report artifacts
were inspected and before the remaining report groups were analyzed.

The registered confidence-plus-cue report head sees maximum probability, entropy,
and cued quality. It lacks answer identity, yet the joint target asks it to name a
verified value as well as quality/source. Its comparison with a full-state head
therefore confounds source monitoring with a missing content input. The registered
results, thresholds, and source files are retained unchanged, but a positive
registered report gate cannot establish a fair reporting advantage.

Add the following diagnostic for ALL12 fitted state controllers:

- Full six answer logits plus the same observed quality cue, padded to128.
- The same128→32 tanh→14 readout, fit-only normalization, two L2 candidates,
  200 Adam steps, seed offset, fit/validation/test partitions, and selection rule.
- Record component metrics and state-minus-comparator mean quality/source gain;
  apply the existing0.02 margin descriptively without overwriting old gates.
- Preserve original artifact hashes and record new comparator provenance.

This is an explicitly post-registration correction prompted by a design flaw,
not independent confirmation on untouched tests. It repairs an information
asymmetry; it does not certify a positive result from reused data. The control
policy comparator already receives all answer logits for choosing an answer, so
its acquisition-return comparison does not have this omitted-identity flaw.
The answer-preserving interventions still test sensitivity to a prospective cue;
they do not themselves establish superiority or introspective access.
