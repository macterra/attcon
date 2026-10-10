"""Replay unchanged v1 bundles after the live blank form became received review R-C1."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('audits/manual_factual_review_v1')
REVISION = Path('audits/functional_rubric_revision_v2')
FORM = 'docs/FUNCTIONAL_RUBRIC_REVIEW_FORM.md'


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    received = json.loads((REVISION/'received_review.json').read_text())
    snapshot = (REVISION/'original_blank_review_form.md').read_bytes()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert sha(snapshot) == manifest['dependency_sha256'][FORM]
    live = Path(FORM).read_bytes()
    assert live == (REVISION/'review_R-C1.md').read_bytes()
    assert sha(live) == received['review_sha256']
    with tempfile.TemporaryDirectory(prefix='attcon-review-v1-') as folder:
        isolated = Path(folder)
        for name, expected in manifest['dependency_sha256'].items():
            data = snapshot if name == FORM else Path(name).read_bytes()
            assert sha(data) == expected, name
            path = isolated/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # Preserve the actual archived products too; the original generator must
        # reproduce them byte for byte using only its frozen input sources.
        for name in manifest['bundle_sha256']:
            source = Path(name)
            path = isolated/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
        for source in (ROOT/'manifest.json', ROOT/'author_proposed_gold.json', ROOT/'prose_linkage.json'):
            path = isolated/source; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
        subprocess.run([sys.executable, 'scripts/build_functional_review_packet.py', '--verify'],
                       cwd=isolated, check=True)
    print('Historical v1 replay uses archived blank form; live R-C1 is retained unchanged, not repackaged.')


if __name__ == '__main__':
    main()
