# Typed API schema amendment v2 — source-only candidate

The first v1 narrator batch retained 552 HTTP 400 failures and no provider error
bodies. Those failures remain the first v1 outcomes. The separately preserved
[first handcrafted diagnostic](../api_transport_diagnostic/run/first_outcome.json)
sent the same frozen schema and received `invalid_json_schema` with
`param: text.format.schema`: its `kind` leaf must have a `type` key. This directly
observes the provider's schema rejection on that diagnostic; the original 552
missing bodies are not reconstructed. This candidate does not supersede, rerun,
or relabel that batch.

The frozen exported response schema contains 22 `enum` or `const` leaves without
explicit `type`: 19 string domains and three nullable string command domains.
The [official OpenAI structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
uses typed enums and a `['string', 'null']` union for nullable string values.
The diagnostic directly identifies the missing type, and the official examples
support the minimal representation used here. The root coordinator must obtain
and retain a separately reviewed, versioned handcrafted acceptance diagnostic
for the amended schema before releasing another scientific batch. This source
amendment alone does not establish provider acceptance.

The new [schema helper](../../../scripts/attention_relational_api_schema_v2.py)
accepts only the exact canonical frozen v1 schema digest. It deep-copies that
schema and adds exactly 22 `type` entries. It leaves all enums, constants,
required arrays, numeric and array bounds, ordering, input prose, report
contract, deterministic auditor, tolerances, and scientific inputs unchanged.
It has no API client, file writes, model loading, or import-time execution.
The original v1 files and exported schema stay frozen.

The single new [handcrafted fixture](../../../tests/test_attention_relational_api_schema_v2.py)
checks that the old exported schema still matches its preserved raw digest and
the frozen source constant; the input is not mutated; exactly 19 string and
three nullable string additions occur; removing only those entries reconstructs
the old schema; and a changed contract is rejected. It also checks that the
existing deterministic auditor still rejects invalid literals and types,
duplicate peak cells, duplicate peak claims, and duplicate optional claims.
It does not load reports, archives, model weights, or fresh worlds. It is not a
general JSON-schema validator, an API acceptance test, a prose-fidelity audit,
or evidence for qualia. No fixture has been executed by this source author.

The source-only metadata binds exact helper, fixture and unchanged source
hashes. Root owns publication, independent review, the one first fixture run,
any retained provider diagnostic and any distinctly versioned subsequent batch.
No favorable output selection, threshold change, reference change, or v1 gate
promotion is authorized by this amendment.
