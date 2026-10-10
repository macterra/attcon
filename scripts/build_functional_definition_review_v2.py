"""Prepare/replay draft-2 definition materials; no completed reviews or outputs included."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path('audits/functional_definition_review_v2')
ASSET = Path('docs/assets/functional_review_v2/definition_review.zip')
DOCS = ('docs/FUNCTIONAL_REPORT_RUBRIC_V2.md', 'docs/FUNCTIONAL_RUBRIC_REVIEW_FORM_V2.md',
        'docs/FUNCTIONAL_REPORT_RATING_FORM_V2.csv')
META = 'reviewer_id,reviewer_type,role,date,rubric_version,prior_exposure,relationship_to_project,prior_design_involvement,conflicts,notes\n'
README = '''# Definition review: draft 2

Read the rubric and complete the blank definition form before viewing new
experimental outputs or other reviewers' suggested answers. Record actual reviewer
type, role, relationship and exposure in reviewer_metadata.csv. An author review or
AI contributor review is not automatically independent human validation.

This draft separates recovery forecasts, category certainty, presentation manner
and answer confidence. Assess their definitions and theoretical relevance. All
current controls contain attention models; determine what feature/contrast could
distinguish consciousness-report correspondence from ordinary causal description.
No primary consciousness endpoint, agreement threshold or powered protocol is frozen.

The blank rating CSV illustrates candidate dimensions only. It is not a request
to score reports yet. The package contains no reports, source states, condition
outcomes, completed R-C1 review, disposition answers, factual gold or human ratings.
Its illustrations are authored examples, excluded from reporter prompts/training.
Prior review remains preserved separately with its exposure record.

Return original comments and proposed changes; preserve revisions separately.
Missing judgments are not approval. Preparing these materials contacts nobody.
Relative links are rendered as plain labels; primary-source web links are retained.
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def standalone(text):
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
        lambda m: m.group(0) if '://' in m[2] else m[1], text)


def build():
    sources = {p: Path(p).read_bytes() for p in DOCS}
    sources['scripts/build_functional_definition_review_v2.py'] = Path(__file__).read_bytes()
    files = {'README.md': README.encode(), 'reviewer_metadata.csv': META.encode()}
    for p in DOCS:
        files[Path(p).name] = standalone(sources[p].decode()).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16; info.create_system = 3
            archive.writestr(info, data)
    manifest = {'kind': 'draft_definition_review_materials', 'version': 2,
        'rubric_frozen_for_confirmation': False, 'independence_verified': False,
        'human_definition_reviews_received': 0, 'human_ratings_received': 0,
        'exposed_AI_contributor_review_received': 'R-C1; preserved outside this bundle',
        'no_new_reports': True, 'no_outreach': True,
        'source_sha256': {p: digest(data) for p, data in sources.items()},
        'bundle_sha256': {str(ASSET): digest(out.getvalue())},
        'bundle_contents': {p: digest(data) for p, data in files.items()},
        'link_adaptation': 'relative links rendered as plain labels; external links retained'}
    for name in ('reviewer_metadata.csv', 'FUNCTIONAL_REPORT_RATING_FORM_V2.csv'):
        assert len(list(csv.reader(io.StringIO(files[name].decode())))) == 1
    assert set(files) == {'README.md', 'reviewer_metadata.csv', *(Path(p).name for p in DOCS)}
    return out.getvalue(), manifest, {p: data.decode() for p, data in sources.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    archive, manifest, sources = build()
    products = {ASSET: archive,
        ROOT/'manifest.json': (json.dumps(manifest, indent=2)+'\n').encode(),
        ROOT/'source_code.json': (json.dumps(sources, indent=2)+'\n').encode()}
    if args.verify:
        for path, data in products.items(): assert path.read_bytes() == data, path
    else:
        if ROOT.exists() or ASSET.exists(): raise SystemExit('refusing overwrite of draft-2 materials')
        ROOT.mkdir(parents=True); ASSET.parent.mkdir(parents=True, exist_ok=True)
        for path, data in products.items(): path.write_bytes(data)
    print(('Exactly verified' if args.verify else 'Prepared')+
          ' draft-2 definition bundle; blank forms, no reports, no independent validation')


if __name__ == '__main__':
    main()
