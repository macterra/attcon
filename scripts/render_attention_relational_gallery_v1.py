"""Render every fixed narrator opportunity and literal control to Markdown/HTML.

Documentation export only: no API/model calls, narrator/reference scoring, favorable selection, or
reads of prose-auditor/reference-rater outcomes. Existing output is never replaced.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline


EXPORT_SHA256 = '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'
REPORTS_SHA256 = '80e07b465a4ac95832585be9594973ead12ba1b830f3f517cb250d0891ca6940'
STYLE = ('body{font:16px/1.55 system-ui,sans-serif;max-width:76rem;margin:2rem auto;padding:0 1rem}'
         'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem;border-radius:.4rem}'
         'p,h2,code{overflow-wrap:anywhere}section{border-top:1px solid #ccd2d8;margin-top:2rem;padding-top:1rem}'
         'details{margin:1rem 0}summary{cursor:pointer;font-weight:600}a{color:#1759a6}')
INTRO = (
    'This gallery preserves all 552 scheduled first narrator positions and their '
    '552 literal-template controls in the fixed export order across 12 archive cases. '
    'No favorable examples were selected. These are repeated reporting opportunities, '
    'not 552 independent worlds.\n\n'
    '`A_only` is a hybrid record: copied W11 fields with the previous node replaced '
    'by a source current estimate. Conditional candidates inherit that record and '
    'replace the query node with a source forecast. Ordinary and analytic sources '
    'also model attention effects. Joined records additionally supply native W.\n\n'
    'Narrator text is raw first narration, not certified faithful or conscious. '
    'Literal templates are deterministic controls. Incomplete, missing, invalid, '
    'or blank text remains explicitly unavailable or blank according to the unchanged '
    '`pipeline._report_text` helper; no partial response is promoted into a complete text. '
    'The exact returned text is HTML-escaped inside `<pre>` blocks. No prose-auditor '
    'or reference-rater outcomes are used.\n'
)


def html_page(title, body):
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + escape(title) + '</title><style>' + STYLE + '</style></head>'
            '<body><main>' + body + '</main></body></html>\n').encode('utf-8')


def render(exports, reports):
    inventory = pipeline.validate_export(exports)
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    received = pipeline._records(reports)
    ids = inventory['request_ids']
    pipeline.require(set(received) <= set(ids), 'foreign narrator collection identifier')
    groups, positions = {}, []
    for identifier in ids:
        case = inventory['provenance'][identifier]['case']
        key = (case['encoder_seed'], case['world_index'])
        record = received.get(identifier)
        text = pipeline._report_text(record)
        template = inventory['templates'][identifier]['report']['text']
        complete = bool(record and record.get('transport_complete'))
        row = {'id': identifier, 'template_id': identifier + '_template', 'case': case,
               'transport_complete': complete,
               'availability': record.get('availability', 'unspecified') if record else 'missing',
               'api_status': record.get('api_status') if record else None,
               'narrator_text_available': bool(text.strip()),
               'narrator_text_sha256': pipeline.digest(text.encode('utf-8')),
               'template_text_sha256': pipeline.digest(template.encode('utf-8'))}
        positions.append(row)
        groups.setdefault(key, []).append((row, text, template))
    expected_cases = [(seed, world) for seed in (1301, 1311, 1321) for world in range(4)]
    pipeline.require(list(groups) == expected_cases and all(len(rows) == 46 for rows in groups.values()),
                     'fixed12 cases /46 positions per case differ')
    completed = sum(p['transport_complete'] for p in positions)
    pipeline.require(len(positions) == 552 and completed == 537 and len(positions) - completed == 15,
                     'fixed552 positions /537 completed /15 incomplete inventory differs')
    pages, links, html_links = {}, [], []
    intro_html = '\n'.join('<p>' + escape(p.replace('`', '')) + '</p>'
                           for p in INTRO.strip().split('\n\n'))
    for (seed, world), rows in groups.items():
        name = f'seed-{seed}-world-{world:02d}.md'
        html_name = name.removesuffix('.md') + '.html'
        title = f'Encoder {seed}, world {world}: decision 12'
        links.append(f'- [Encoder {seed}, world {world} — 46 narrator positions and 46 controls]({name})')
        html_links.append(f'<li><a href="{html_name}">Encoder {seed}, world {world} — '
                          '46 narrator positions and 46 controls</a></li>')
        lines = [f'# {title}\n', '[Gallery index](index.md)\n', INTRO]
        html_lines = [f'<h1>{escape(title)}</h1>', '<p><a href="index.html">Gallery index</a></p>', intro_html]
        for number, (row, text, template) in enumerate(rows, 1):
            case = row['case']
            narrator_pre = '<pre>' + escape(text, quote=True) + '</pre>\n'
            template_pre = '<pre>' + escape(template, quote=True) + '</pre>\n'
            text_label = ('Narrator text:' if row['narrator_text_available'] else
                          'Narrator text unavailable or blank; this scheduled position is retained.')
            lines.extend([
                f"## {number:02d}. {row['id']}\n",
                f"Source: `{case['source']}`; view: `{case['view']}`; variant: `{case['variant']}`.\n",
                f"Narrator transport complete: `{str(row['transport_complete']).lower()}`; "
                f"availability: `{row['availability']}`; API status: `{row['api_status']}`.\n",
                f"Narrator text SHA-256: `{row['narrator_text_sha256']}`.\n",
                text_label + '\n', narrator_pre,
                '<details>\n<summary>Literal-template control</summary>\n',
                f"Template ID: `{row['template_id']}`.\n",
                f"Template text SHA-256: `{row['template_text_sha256']}`.\n",
                template_pre, '</details>\n'])
            html_lines.extend([
                f'<section id="{escape(row["id"], quote=True)}"><h2>{number:02d}. {escape(row["id"])}</h2>',
                '<p>' + escape(f"Source: {case['source']}; view: {case['view']}; variant: {case['variant']}.") + '</p>',
                '<p>' + escape(f"Narrator transport complete: {str(row['transport_complete']).lower()}; "
                               f"availability: {row['availability']}; API status: {row['api_status']}.") + '</p>',
                f"<p>Narrator text SHA-256: <code>{row['narrator_text_sha256']}</code>.</p>",
                '<p>' + escape(text_label) + '</p>', narrator_pre,
                '<details><summary>Literal-template control</summary>',
                '<p>Template ID: <code>' + escape(row['template_id']) + '</code>.</p>',
                f"<p>Template text SHA-256: <code>{row['template_text_sha256']}</code>.</p>",
                template_pre, '</details></section>'])
        pages[name] = '\n'.join(lines).encode('utf-8')
        pages[html_name] = html_page(title, '\n'.join(html_lines))
    pages['index.md'] = ('# Attention relational report v1: complete first-text gallery\n\n' + INTRO
        + '\n537 narrator transports completed; 15 were incomplete or unavailable. '
        'Every scheduled position and all 552 literal controls are retained.\n\n'
        + '\n'.join(links) + '\n\n[Standalone HTML gallery](index.html)\n\n'
        '[Machine-readable manifest](manifest.json)\n').encode('utf-8')
    pages['index.html'] = html_page('Attention relational report v1: complete first-text gallery',
        '<h1>Attention relational report v1: complete first-text gallery</h1>' + intro_html
        + '<p>537 narrator transports completed; 15 were incomplete or unavailable. '
        'Every scheduled position and all 552 literal controls are retained.</p><ul>'
        + '\n'.join(html_links) + '</ul><p><a href="index.md">Markdown index</a> · '
        '<a href="manifest.json">Machine-readable manifest</a></p>')
    manifest = {'schema': 'attention_relational_report_v1_complete_first_text_gallery',
        'ordered_ids': ids, 'positions': positions,
        'counts': {'archive_cases': 12, 'narrator_positions': 552, 'literal_templates': 552,
                   'transport_complete': completed, 'transport_incomplete_or_missing': 552 - completed,
                   'narrator_text_available': sum(p['narrator_text_available'] for p in positions)},
        'case_order': [dict(encoder_seed=s, world_index=w) for s, w in groups],
        'availability_counts': dict(Counter(p['availability'] for p in positions)),
        'pages_sha256': {name: pipeline.digest(raw) for name, raw in pages.items()},
        'protocol_manifest_sha256': pipeline.PROTOCOL_HASH,
        'narrator_instruction_sha256': pipeline.INSTRUCTION_HASHES['narrator'][1],
        'no_api_or_model_calls': True, 'no_new_narrator_or_reference_scoring': True,
        'export_validation_includes_existing_literal_template_checks': True,
        'no_favorable_selection': True,
        'no_prose_auditor_or_reference_rater_outputs_read': True,
        'formats': ['Markdown', 'standalone static HTML'], 'external_dependencies_or_scripts': False,
        'text_source': 'Unchanged pipeline._report_text; export literal templates',
        'text_escaping': 'html.escape(quote=True) inside pre; no text normalization'}
    return pages, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output-root', type=Path,
                        default=Path('docs/assets/attention_relational_report_v1_gallery'))
    args = parser.parse_args()
    pipeline.require(not args.output_root.exists(), 'gallery output root exists; refusing replacement')
    exports = pipeline.bound_json(args.exports, EXPORT_SHA256)
    reports = pipeline.bound_json(args.reports, REPORTS_SHA256)
    pages, manifest = render(exports, reports)
    manifest['input_dependencies'] = {
        'exports': {'path': str(args.exports), 'sha256': EXPORT_SHA256},
        'reports': {'path': str(args.reports), 'sha256': REPORTS_SHA256}}
    manifest['source_sha256'] = {str(Path(m.__file__)): pipeline.digest(Path(m.__file__).read_bytes())
                                 for m in (pipeline, exporter)}
    manifest['source_sha256'][str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name, raw in pages.items():
        with (args.output_root / name).open('xb') as stream:
            stream.write(raw)
    pipeline.exclusive_json(args.output_root / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
