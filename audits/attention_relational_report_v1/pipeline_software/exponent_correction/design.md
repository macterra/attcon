# Finite JSON exponent correction: author source record

2026-10-11. Separate narrow pre-output correction after original18-source
preservation b9ae58d, parent-reported pushed. No scientific/API output or fixture
execution was observed. Original malformed_verdict_correction records remain.

Python JSON decoding turns finite-looking exponent strings such as1e309 and
-1e309 into infinity even when literalNaN/Infinity parse_constant is rejected.
The [pipeline](../../../../scripts/attention_relational_reporting_pipeline_v1.py)
now supplies parse_float that converts the token and requires math.isfinite.
Overflow is retained as parse_error with the original raw output_text unchanged;
it cannot reach a positive judgment or break finite-only serialization.
Ordinary finite float decoding, scientific criteria, source/reference/protocol
and request inventories are unchanged. Root separately owns the corresponding
API-runner correction; this author did not edit that runner.

[Fixture19](../../../../tests/test_attention_relational_reporting_pipeline_v1.py)
covers both exponent signs, direct decoder rejection, first-outcome raw retention
and ordinary finite number acceptance. Original18 methods are unchanged by AST
hash equality in original_18_method_hashes.json. Syntax compilation exit0 only;
no pytest, archive, model, rendering, fit, reporting API or Git execution.

Disposition: source ready for preservation and independent review before the
first19-fixture execution. This author record is not independent approval.
