# Independent exported-input review: preserved first pass

2026-10-10. **Review incomplete because of a checker defect.** The first pass
retains 48 inventory-check exceptions, one for each table packet. It does not
establish an exporter/source defect, and it does not yet pass the full bounded
exported-input review. No production artifact or export is to be repaired or
retried on the basis of these exceptions.

This review began after source-only registration freeze
`9cb69dfd86f5a654bddd822fe9ddcec1b721b546` and completed-export save
`ec65d39537e1b2612c823b65497183464b120ab6`. It read only the authorized exported
JSON, registration and execution receipts. The independent checker uses Python's
standard library and imports neither production source module. It performed no
binary archive/checkpoint reads, inference, decoder fitting, export, Reporter call,
factual qualification or character rating. No target reports exist in the inspected
packet export. The reviewer is a separate-context AI commissioned by the same
project; shared training, shared repository/tools and commissioning limit its
independence. The factual-auditor contexts were not contacted or exposed to targets.

Command: `.venv/bin/python audits/grounded_neutral_report_pilot_v1/export_review/check.py > audits/grounded_neutral_report_pilot_v1/export_review/run.log 2>&1`.
The single first execution completed **96,461 individual comparisons, with
48 exceptions and exit status 1**. These are engineering comparisons, not
statistical observations or independent scientific tests. `checks.json` and
`run.log` retain every packet ID, exception type and error message. No checker
repair, rerun, artifact overwrite or production change preceded this receipt.

## Exception diagnosis

All exceptions are `KeyError: 'native_route_history'` in the independent inventory
checker for `tables`. Its `row_path` helper routes every collection through
`source['tables']` whenever the packet has a `tables` member. The actual packet
schema keeps native history at the top level while separating evidence, risk,
external-readout, delivery and post-command tables.

After the first pass, the reviewer explicitly inspected **all 48 actual table
packets and their saved inventory units**. Each packet contains top-level
`native_route_history`, contains no history table, and has a
`latest_observed_delivery` unit whose `/native_route_history/<index>` pointer
resolves to exactly its saved value. Example packet
`P4b578b7eee0f27fc2cfdef8969698856` uses `/native_route_history/0`.
The source history is present and its inventory unit is correct. The checker must
be corrected in a separately preserved pass to address this top-level collection.
The remaining table inventory units were not fully checked in the interrupted
inventory routine and cannot be credited automatically.

## Factual findings within the completed checks

- There are **624 unique opaque packet IDs**, 48 model/world opportunities, exactly
  13 arms with 48 packets per arm, and 208 packets per model. The shuffled schedule,
  indexed location `world % 4`, indexed command `(world // 4) % 4`, and all 16
  location/command pairs per model agree with registration.
- All 624 packet file hashes match completion and source-aware association hashes.
  Audit and verification hashes, registration receipts and combined prepared-JSON
  hash agree. The registration matches its freeze Git blob. All 630 authorized
  input JSON files match the completed-export Git blobs exactly.
- Independently inverted aliases recover each saved canonical source view and
  query. Independent address-aware joining recovers the natural packet from every
  table arm. Shams and both restores have exactly equal substantive input after
  excluding transport packet IDs. Apparatus changes only the registered referent
  sentence; ID remapping preserves the normalized natural input.
- Every tested post-command cell satisfies exact
  `delivery*fresh + (1-delivery)*current`. Route and memory/route roll command
  delivery rows by +1; the crossed arm uses moved current/fresh values and keeps
  moved content plus unchanged native history. Donor evidence moves by +1 while
  actual history remains fixed. Observed current/fresh and external donor values
  happen to agree exactly in these saved JSON packets; this is no neural
  reconstruction or independently recomputed decoder-envelope check.
- The 48 natural sources contain all **192 location rows: 172 observed and 20
  unseen**, with 20 unknown native-history rows. Unseen rows retain null represented
  content and external estimates plus explicitly labelled horizontal defaults.
  Across arms supplying location rows there are 240 unseen rows; history-supplying
  arms retain 240 unknown rows.
- Completed inventory checks cover all non-table packets, including exact semantic
  constituents, four-address scalar vectors and full-precision JSON pointers.
  Fixed ablation masks agree: sensory-only 9 supplied + 3 absence units, route-only
  6 + 3, no-external 18 + 1. Missing units are typed `not_supplied`; ablations are
  unmatched and route-only has no content denominator. The 10,656 supplied-unit
  count in this first pass excludes the 48 interrupted table inventories;
  it is a checker-coverage count, not the final export denominator. All 336 required
  source-absence units were checked.
- Reporter packets exclude the checked evaluator-truth, pixel, chronology,
  intervention/provenance and scoring namespaces. Audit history pointers,
  source variants, frozen manifest/archive/model hash associations and interpretation
  flags agree with registration. The attempt/execution/completion receipts declare
  one completed source-only attempt, no retry, no new inference/fits and zero
  Reporter calls. The independent review checks these declarations and associations,
  not the underlying binary files or every possible hidden semantic disclosure.

The 48 checker exceptions are the only recorded first-pass anomalies. A complete
bounded JSON verdict awaits the separately saved checker correction and rerun.
Binary source hash verification, source tensor validation, decoder propagation
bounds and registered disk replay belong to the separate parent/source checks;
this reviewer has not independently reproduced them. Full pre-prose registration,
factual key/scorer approval and auditor qualification remain separate gates. This
input review cannot authorize target prose or establish appearance correspondence.
`Faithful_P_presentation` remains `not_established`, felt truth is unauditable,
and the source-of-qualia goal remains open.

First-pass artifact hashes:

- `check.py`: `8ef8b8d97bd7ab7c3a5cdf6b424e71f7bb34885e9f936d4fe4fe99f9effd8859`
- `checks.json`: `b9f2622b09730509a953b3a062fc2e6200bee6d54eb8816fd579bbf67159163d`
- `run.log`: `f52bd30c1cfedb3648c4f23f34b82fdb77ddb0139d0c90698122b054bc4b2328`
