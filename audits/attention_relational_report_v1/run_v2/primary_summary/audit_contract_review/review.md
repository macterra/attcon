# Original prose-auditor contract diagnosis

Independent bounded review by `/root/goal_evidence_audit`, 2026-10-19.
This review inspected original audit/report objects and source for contract QA.
It did not inspect reference feature/composite labels or quotations, and did not
read any live resource-v2 auditor outputs. No original annotation was repaired,
rescored, replaced, or normalized for acceptance. No API call or Git mutation.

## Diagnosis

The primary summary correctly leaves all 552 narrator fidelity positions
unestablished. That is not a finding that all reports are false. Only one of
453 provider-completed audits meets the structural contract. Its coverage flags
pass, but two of its 14 claims are not_auditable, so its fidelity gate also fails.
The 99 schema-invalid positions include 84 incomplete audits and 15 narrator
positions without eligible actor text; they are not 99 completed schema failures.

Of 453 completed audits, 426 contain a nonliteral quote and 438 contain binding
shape errors; 412 contain both. There are 3,278 nonliteral quote occurrences.
Only 11 match after a whitespace-only diagnostic, four after casefolding, and
3,263 fail these simple comparisons. These checks diagnose mismatch only; they
perform no repair or semantic equivalence judgment. The quote failure is thus
not predominantly a small whitespace/Unicode representation defect.

The 2,155 unsupplied-context bindings include 1,439 null nodes, 1,168 candidate
bindings with no command and 418 null/absent panels (categories overlap). There
are also 975 peaks/scalar bindings with no role, 162 record_scope bindings with
specific identifiers and 109 numeric bases assigned record_scope. These expose
systematic broad/all-context summaries emitted where the contract requires
explicit valid operands. No null node or command can be silently expanded into
all nodes/commands: that would add unasserted attribution and conceal errors.

The source validator behaves as specified: exact quote checking at pipeline
lines 455–459; record-scope restrictions at 478–482; supplied-context lookup at
483–484; role-specific basis at 485–486. The API schema permits nullable fields
individually and cannot itself express every supplied-context constraint
(lines 62–69). The prose instruction asks for exact spans and every relational
operand but does not enumerate valid context combinations or the all-or-nothing
validation consequences. The diagnosed failure concerns instrument emission and
coverage, not established contradiction of the underlying prose.

## Recommendation

Do not launch another full auditor with unchanged quoting/binding mechanics.
Prepare a distinct prospective interface that makes literal evidence and valid
source contexts easier to express, with a per-packet valid-context inventory and
explicit rules for record-level versus specific claims and all operands. A
context-ID interface mapped deterministically to exact supplied bindings is one
candidate; it needs its own honest schema/lineage and substantive audit before
release. It must not supply verdicts, reference labels, desired consciousness
vocabulary, source identity, hidden truth, or completion-selected examples.
Malformed/ambiguous evidence remains unestablished; all original gates and
failures remain intact. A mechanical fixture suite and fictional unscored
end-to-end check must precede another full target batch. Semantic audit remains
fallible even when the contract validates.

The already live resource-v2 auditor changes headroom and timeout, so original
failure does not prove that it will fail: more reasoning budget could improve
contract adherence. Do not announce it a failure or interrupt it solely as if
its own outcome were already known. Prefer a prospectively declared bounded
contract-only QA of its earliest fixed-order completed positions, reporting only
structural errors/validity and no support verdicts, reference labels or quotes.
Use that evidence for the operational decision. If it shows the same systematic
contract failure, stop only under an explicit published operational addendum,
retaining every raw outcome, unknown in-flight reservation, full 537-position
inventory and original denominators; do not resume or selectively complete it.
If QA shows useful contract adherence, finish its registered batch and assess it
as supplementary. The QA scope and any operational rule must be fixed before
looking at those outputs; no new scientific success threshold is justified.

This is a recommendation, not stop authority or a new gate on the user. The
currently live job was not touched. Completing a failing instrument for its own
sake is not the objective, but neither is obtaining positive annotations. The
objective remains a credible assessment of actual source fidelity and report
correspondence, followed by prospective confirmation and replication if warranted.

Chronology: the earlier quote-grounded reference-V3 design recommendation was
based on reference offset QA before this prose-auditor contract review. This
new diagnosis must not be backdated into that decision. Any further auditor
amendment is now informed by the preserved original audit-contract failures.
