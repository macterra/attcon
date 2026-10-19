# Calculator execution provenance

The first semantic review is first_review.md and remains unchanged. No semantic blocker was found.

The first calculator execution failed before producing a receipt, because the review calculator initially assumed every legal public context had packet_path. Actual relation contexts intentionally carry both operand IDs and do not carry packet_path. The exact initial calculator source is preserved as recompute_first_attempt.py, and its exact stderr is preserved as calculator_first_attempt_stderr.txt. This was a review-calculator bookkeeping error, not an author/source/key defect.

Only the review calculator was corrected: the bypath comprehension now includes contexts with packet_path. This preserves all actual role/origin/metadata operands and permits actual relation contexts to coexist in the public inventory. No author source, key, public input, or target file was changed. The corrected calculator produced the preserved recomputation_receipt.json and successful stdout (calculator_first_success_stdout.json). The original failed attempt had no semantic verdict output, no receipt, and no first_review.md. An earlier shell attempt to invoke python rather than python3 returned exactly /bin/bash: line 1: python: command not found and was retried with python3.

The public-interface checks use actual role/origin/metadata contexts; review prose separately explains that equivalent relation contexts containing both operands are valid semantic bindings. The original first review and all materialization inputs remain preserved.
