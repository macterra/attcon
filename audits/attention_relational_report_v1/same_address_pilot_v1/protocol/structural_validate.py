#!/usr/bin/env python3
"""Non-mutating format diagnostics. Never certifies support, completeness or reliability."""
import argparse
import json
from pathlib import Path


def inspect(kind, report_text, annotation, schema):
    errors = []
    try:
        import jsonschema
        jsonschema.Draft202012Validator(schema).validate(annotation)
    except ImportError:
        errors.append('jsonschema unavailable: schema validation not performed')
    except Exception as exc:
        errors.append('schema: ' + str(exc))

    def span(entry, label):
        start, end, quote = entry.get('start'), entry.get('end'), entry.get('quote')
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(report_text):
            errors.append(label + ': invalid codepoint span')
        elif report_text[start:end] != quote:
            errors.append(label + ': quote differs from literal report substring')

    if kind == 'factual':
        claims = annotation.get('claims', [])
        ids = [c.get('claim_id') for c in claims]
        if len(ids) != len(set(ids)):
            errors.append('duplicate claim IDs')
        for c in claims:
            span(c, 'claim ' + str(c.get('claim_id')))
        cursor = 0
        for i, segment in enumerate(annotation.get('coverage_segments', [])):
            start, end = segment.get('start'), segment.get('end')
            if start != cursor or not isinstance(end, int) or not isinstance(start, int) or not 0 <= start <= end <= len(report_text):
                errors.append(f'coverage segment {i}: noncontiguous or invalid')
            if isinstance(end, int):
                cursor = end
            for cid in segment.get('claim_ids', []):
                if cid not in ids:
                    errors.append(f'coverage segment {i}: unknown claim {cid}')
            if segment.get('kind') == 'claim-bearing' and not segment.get('claim_ids'):
                errors.append(f'coverage segment {i}: claim-bearing without claim IDs')
        if cursor != len(report_text):
            errors.append('coverage partition does not reach report end')
        # Deliberately do not infer inventory_complete or whole_fidelity from syntax.
    elif kind == 'semantic':
        for category in ['features', 'auxiliary']:
            for key, entry in annotation.get(category, {}).items():
                for e in entry.get('evidence', []):
                    span(e, category + '.' + key)
                if entry.get('status') == 'explicit' and not entry.get('evidence'):
                    errors.append(category + '.' + key + ': explicit without literal evidence')
        for key, entry in annotation.get('composites', {}).items():
            bindings = entry.get('bindings', [])
            if entry.get('status') == 'explicit' and not bindings:
                errors.append('composites.' + key + ': explicit without binding')
            for i, b in enumerate(bindings):
                for e in b.get('evidence', []):
                    span(e, f'composites.{key}.bindings[{i}]')
                if entry.get('status') == 'explicit' and not b.get('evidence'):
                    errors.append(f'composites.{key}.bindings[{i}]: explicit without evidence')
                # Scope/time compatibility must be read, never inferred by a format check.
    return {'kind': kind, 'format_diagnostics': errors,
            'semantic_correctness': None, 'claim_inventory_completeness': None,
            'factual_support': None, 'auditor_qualification': None,
            'warning': 'No format diagnostic establishes whole-prose fidelity or measurement reliability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['factual', 'semantic'])
    parser.add_argument('report_text_path', type=Path)
    parser.add_argument('annotation_path', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    schema = json.loads((here / f'{args.kind}_first.schema.json').read_text())
    # Explicit invocation only; never searches target data or overwrites input artifacts.
    print(json.dumps(inspect(args.kind, args.report_text_path.read_text(),
                             json.loads(args.annotation_path.read_text()), schema), indent=2))


if __name__ == '__main__':
    main()
