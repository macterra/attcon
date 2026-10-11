# First19 fixture failure: diagnosis and narrow source correction

2026-10-11. The parent executed the first19 suite once:18passed/1failed, preserved
and published at6d62377. That failure stays a failure. This source author did not
run any suite, source model, archive/export, API or Git command.

I used an obsolete guessed packet-schema name in the handcrafted summary fixture.
The actual exporter PACK_SCHEMA is attention_relational_report_v1_packet; the
fixture instead used attention_relational_report_v1_neutral_pack. The failure
occurred when the real inert required_propositions helper rejected that mismatch.
The correction imports PACK_SCHEMA directly from the immutable exporter and uses
it in this fixture. No production validator or semantic criterion was loosened.

Static tracing identified a second fixture-only defect before another execution:
summary evaluates its report/audit/rater request-builder arguments before invoking
the mocked lineage validator. The fixture supplied only exports.provenance, so a
schema-only repair would next fail on missing exports.requests, then packets or
templates. The same19th-file summary test now supplies complete one-row handcrafted
requests/packets/templates/provenance and decision_time12 case metadata. It remains
an explicitly mocked one-row join test, not a production552-case qualification.

The real helper's derived spelling is node.derived[red|blue].peaks, which already
matches this fixture. Nodes n0..n3, roles red/blue, exact row-major peak-cell arrays,
current view and null candidates match the sensor-required eight atoms. Report
schema/text/propositions have canonical keys. Both scientific inventory and
lineage validation remain deliberately mocked for the tiny join fixture only.
The production input boundary/validator, source/report/API contracts and all18
other passing fixture methods are unchanged, documented by AST-function hashes.

Remaining static path trace: required_propositions produces8current atoms;
audit_report compares those8same atoms and has no optional relation/scalar claim
to require omitted derived numeric values. Prose audit deliberately labels its
synthetic claim not_auditable. Two hypothetical source-blind annotations are
schema-valid but are not empirical reference judgments. Their raw explicit match
must not pass the fidelity conjunction. The second call uses an incomplete actor,
empty blind text and a valid ambiguous rating; it must stay unestablished even
for an upper match. Fixed helper files/instructions/reference are read only by the
future handcrafted suite; no narrator, model or scientific value is involved.

Source syntax compilation exit0; corrected19suite UNEXECUTED here. Independent
review/root release is required before one corrected-suite execution. Any later
failure must be preserved before a further version. First failure logs/results
and prior source records were not edited. This is diagnosis, not a passing result.
