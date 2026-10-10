# Qualitative report annotation protocol v2

2026-10-10. Development qualification interface, before new target prose. The 24 synthetic texts, all 13 author-proposed labels per case, proposed scope key and genuine-source anchors are reused unchanged from v1. This is not new participant data, a new reference corpus or evidence of semantic validity. The author key is withheld from raters.

The parent reports that corrected v1 grading retained 24/24 agreement on the 13-label tuple for each rater but only 11/24 full-case agreement including the frozen scope-set requirement. Both failed. Initial grading omitted that requirement; rater guidance supplied hyphenated presentation names and unresolved, and omitted the model-access scope, while the author key used canonical underscores and unclear. V2 makes the interface and full-case grading explicit. It does not retroactively qualify v1, assume every discrepancy was representational, or weaken any threshold. All v1 files and first passes remain intact.

Reporter remains external instrumentation, Controller selects/executes actions, and the attention-control model is the proposed source. Access correspondence remains separate from stricter presentation correspondence. Represented-access P alone does not achieve the source-of-qualia goal.

## Frozen linguistic definitions

Read text before assigning labels. Every field takes `explicit`, `ambiguous` or `absent`. A hedged assertion can be explicit; ambiguous concerns competing interpretations of the wording. Record exact text evidence with content/location, attribute, time and polarity; do not infer wording from source states.

| Field | Criterion |
|---|---|
| `C` | Identified perceptual/represented content with a named attribute, asserted, denied or hedged. Explicit location is an attribute; unspecified something, first-person grammar and consciousness words alone fail. |
| `M_access` | Qualitative current representational availability/access bound to content/attribute. Current available/unavailable counts. Future recovery, numerical success, category probability or confidence alone fails. |
| `M_presentation` | How identified content is currently presented: apparent character/detail or qualitative presentation. Access wording alone fails. An objective physical adjective or contrast measurement alone fails. Scope is recorded separately. |
| `M` | Union of `M_access` and `M_presentation`: explicit if either is explicit; ambiguous if neither is explicit and either is ambiguous; otherwise absent. |
| `S` | Identification/guess distinguished from presentation/access status, or statuses separately described across attributes/contents. Physical adjectives or decision-confidence clauses alone fail. |
| `P` | Compatible `C AND M AND S`; auxiliary union structure. |
| `P_access` | P where M describes only current access, without explicit presentation. If presentation's presence is ambiguous, this subtype can be ambiguous. |
| `P_presentation` | Compatible `C AND M_presentation AND S`; stricter linguistic presentation outcome. |
| `appearance_scope` | Explicit claimed perceptual appearance/experience scope of M_presentation; absent for display, sensor or model-presentation wording without that claim; ambiguous when scope remains unresolved. |
| `experiential_wording_subset` | P_presentation with explicit appearance_scope. This is a linguistic subset, not felt truth. |
| `A` | Identified content plus attention condition/change explicitly related to its presentation, emergence or access. Code direction, timing and association/forecast/causal language. Mere priority or movement with no content relation fails. |
| `identity_uncertainty` | Explicit attribute uncertainty, hedge or alternatives; bare categorical probabilities alone need not express it. |
| `response_confidence` | Stated confidence that a specified answer is correct. Willingness thresholds, abstention and estimated correctness alone do not supply it. |

Explicit conjunctions require explicit constituents and compatible referents/times. A necessary absent constituent or explicit incompatibility makes a conjunction absent; otherwise it is ambiguous. Multiple claim groups are retained; one compatible explicit group can supply an episode match without deleting other claims. Appearance negation remains explicit linguistic presentation, with its polarity retained. Declining to make an appearance claim supplies no appearance claim. Not-blurred does not imply clear. Unscoped clear must not be resolved from unstated information.

## Canonical scope interface

`scope_labels` is a unique **lexicographically sorted set encoded as a JSON array**, using exactly:

| Value | Scope of a manner claim |
|---|---|
| `display` | Qualitative presentation on a named display. |
| `sensor` | Qualitative presentation in a sensor/camera image. |
| `model_presentation` | Qualitative presentation within a model representation, beyond mere availability. |
| `model_representation_access` | Current representational availability/access described by M_access. |
| `claimed_experiential` | Explicit claimed perceptual appearance/experience. |
| `unclear` | An ambiguous M claim or unresolved scope of a manner claim. |
| `[]` | Empty array when no M claim supplies a scope. This is not a literal label string. |

Scope labels describe **M claims only**. Explicit current M_access supplies `model_representation_access`. An ambiguous manner/scope supplies `unclear`; do not list guessed alternative scopes for that claim. Separate explicit M claims can supply multiple scopes; deduplicate and sort their union. No scope is assigned merely because C mentions a display, sensor, model, physical attribute, answer or confidence. Additional source/device/role locations may be described in `role_scope_notes`, outside the graded scope set.

Names alone do not determine the semantic criterion: representation can name content without access or presentation; sensor can describe an objective measurement without presentation. First establish the M claim and its exact span, then its scope. Preserve physical-property versus presentation and current-access versus presentation boundaries. Do not use hyphens, spaces, unresolved, aliases, a literal empty-set string, or inferred scopes in the serialized set. An invalid scope value is a schema error, not a successful semantic match.

## Machine-readable rater output

Use this JSON Schema. Each synthetic case must appear once with its original id. The evidence offsets are zero-based Unicode code-point offsets into the exact blind-fixture text; `quote` must equal `text[start:end]`. Supply evidence for explicit/ambiguous assignments and concise rationale for absence in notes. Evidence for a conjunction can reference the spans supplying its constituents. Do not fabricate a quote for absent wording. Authentic anchors are recorded separately by original source location and paraphrase, without duplicating the prior review's quotations.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "rater_id", "exposure", "cases"],
  "properties": {
    "schema_version": {"const": "qualitative-report-annotations-v2"},
    "rater_id": {"type": "string"},
    "exposure": {"type": "object"},
    "cases": {"type": "array", "minItems": 24, "maxItems": 24, "items": {"$ref": "#/$defs/case"}}
  },
  "$defs": {
    "label": {"enum": ["explicit", "ambiguous", "absent"]},
    "labels": {
      "type": "object",
      "additionalProperties": false,
      "required": ["C", "M_access", "M_presentation", "M", "S", "P", "P_access", "P_presentation", "appearance_scope", "experiential_wording_subset", "A", "identity_uncertainty", "response_confidence"],
      "properties": {
        "C": {"$ref": "#/$defs/label"},
        "M_access": {"$ref": "#/$defs/label"},
        "M_presentation": {"$ref": "#/$defs/label"},
        "M": {"$ref": "#/$defs/label"},
        "S": {"$ref": "#/$defs/label"},
        "P": {"$ref": "#/$defs/label"},
        "P_access": {"$ref": "#/$defs/label"},
        "P_presentation": {"$ref": "#/$defs/label"},
        "appearance_scope": {"$ref": "#/$defs/label"},
        "experiential_wording_subset": {"$ref": "#/$defs/label"},
        "A": {"$ref": "#/$defs/label"},
        "identity_uncertainty": {"$ref": "#/$defs/label"},
        "response_confidence": {"$ref": "#/$defs/label"}
      }
    },
    "scope_set": {
      "type": "array",
      "uniqueItems": true,
      "items": {"enum": ["display", "sensor", "model_presentation", "model_representation_access", "claimed_experiential", "unclear"]}
    },
    "evidence": {
      "type": "object",
      "additionalProperties": false,
      "required": ["fields", "start", "end", "quote", "referent", "attribute", "time", "polarity"],
      "properties": {
        "fields": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"enum": ["C", "M_access", "M_presentation", "M", "S", "P", "P_access", "P_presentation", "appearance_scope", "experiential_wording_subset", "A", "identity_uncertainty", "response_confidence", "scope_labels"]}},
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 0},
        "quote": {"type": "string"},
        "referent": {"type": "string"},
        "attribute": {"type": "string"},
        "time": {"type": "string"},
        "polarity": {"enum": ["affirmed", "negated", "mixed", "unclear", "not_applicable"]}
      }
    },
    "case": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "labels", "scope_labels", "evidence", "notes", "role_scope_notes"],
      "properties": {
        "id": {"type": "string", "pattern": "^Q(0[1-9]|1[0-9]|2[0-4])$"},
        "labels": {"$ref": "#/$defs/labels"},
        "scope_labels": {"$ref": "#/$defs/scope_set"},
        "evidence": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
        "notes": {"type": "string"},
        "role_scope_notes": {"type": "string"}
      }
    }
  }
}
```

JSON Schema cannot check exact span equality, lexical set ordering or case-id uniqueness across objects; the grader must enforce those additional rules before semantic grading. Exposure records must state model/tool version, supplied file hashes, primary-source access and whether the key, peer ratings, v1 ratings, target reports or model outputs were seen. Fresh contexts must be blinded to those prohibited materials.

## Separate fidelity and genuine anchors

Source-blind raters do not score Faithful_P_presentation. It requires strict linguistic presentation, all inspectable required and unprompted claims passing independent source audit, and a predeclared independently qualified counterpart for the presentation property. Physical decoder accuracy does not alone validate represented clarity. Without a qualified counterpart, faithful presentation is not_established. Felt truth remains unauditable. Explicit unsupported felt wording retains linguistic credit; probability or physical measurements alone cannot certify it. Synthetic faithful scores remain not_evaluated.

Primary pointers are unchanged: Ramsøy and Overgaard (2004), original PAS paper p.12 typical example and Table 2, report elicitation pp.11–12, [original author-uploaded article](https://www.researchgate.net/publication/216733480_Introspection_and_Subliminal_Perception); Lutz et al. (2002), original PNAS 99:1586–1591, p.1587 S4 training quotation and p.1588 cluster descriptions, [journal PDF mirror](https://portablelaboratory.wordpress.com/wp-content/uploads/2014/04/lutzetal02.pdf); Chuyin et al., [version 3, 23 August 2024](https://f1000research.com/articles/11-69/v3), free-report/confidence Methods and Figure 4 descriptions. Inspect original passages, not an AI paraphrase masquerading as a participant report. A typical example without trial/subject id, a named training quotation and aggregate descriptions are different provenance classes. Authentic-anchor annotations remain provisional: no representative corpus validity, prevalence, sensitivity or specificity is claimed.

## Full-case gate and preservation

After the parent freezes v2 in a commit, two fresh AI contexts independently receive only this protocol, [blind_fixtures.json](../audits/qualitative_report_annotation_v2/blind_fixtures.json) and primary pointers. They must not receive fixtures.json, fixture_provenance.json, v1 raters or the author key. These are reused synthetic materials, not fresh empirical observations. They never train or prompt the Reporter. Retain both unedited first passes, exposure and hashes before revealing the key.

For each rater separately, a case is correct only when **all 13 labels AND the canonical scope set** exactly match the withheld author-proposed key. Qualification requires **at least 22/24 full cases** (at least 0.90), plus **zero explicit inferred M/core/presentation false credit** on the unchanged five-case risk/confidence-only or physical-adjective-only guard subset registered in provenance. Report tuple-only agreement separately; it cannot replace full-case agreement. Invalid/missing/duplicate records fail case correctness and are reported, never silently dropped. No averaging of rater scores, automatic alias repair after grading, case removal or lowered gates is allowed.

Preserve author-key and inter-rater disagreements separately. Adjudicate against exact text and frozen rules: explicit needs stated relations, absent follows missing necessary relations, and ambiguous preserves unresolved competing readings. Disagreement alone does not imply ambiguity. Adjudication cannot retroactively qualify a failed first pass. Substantive corrections require a new version and fresh independent qualification before target prose, preserving prior failures. The proposed key remains a development reference, not an independently validated gold standard.

Successful v2 interface qualification would establish bounded application of this rule to reused synthetic material. It would not establish source fidelity, general semantic validity, human-report correspondence, felt experience, or source-of-qualia goal achievement. A separately frozen fresh-report experiment remains necessary.
