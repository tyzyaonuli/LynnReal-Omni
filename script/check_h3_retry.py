"""Inject and verify one missing preview in an isolated resumed evaluation copy."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(root, case, action):
    root = root.resolve()
    source = (root.parent / 'resumed-from-job.txt').read_text().strip()
    assert source.startswith('dlc') and root.parent.name != source
    assert case in json.loads((root / 'run.json').read_text())['case_ids']
    receipt = root.parent / 'retry-acceptance.json'
    target = root / case / 'light-web.mp4'
    assert target.resolve().is_relative_to(root)
    if action == 'prepare':
        assert not receipt.exists(), 'Do not inject twice'
        files = sorted(p for p in root.glob('*/*') if p.is_file()
                       and p.suffix in ('.pt', '.mp4'))
        hashes = {str(p.relative_to(root)): digest(p) for p in files}
        assert str(target.relative_to(root)) in hashes
        record = {'source_job': source, 'case': case, 'status': 'injected',
                  'fault': 'missing light browser preview in copied results',
                  'before_sha256': hashes,
                  'unchanged_records': {str(p.relative_to(root)): digest(p)
                                        for p in root.glob('*/*.json')
                                        if p != root / case / 'light.json'}}
        # Preserve the missing artifact as evidence outside the evaluation directory.
        target.rename(root.parent / 'injected-light-web.mp4')
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    else:
        record = json.loads(receipt.read_text())
        assert record['case'] == case and record['source_job'] == source
        for name, expected in record['before_sha256'].items():
            assert digest(root / name) == expected, name
        for name, expected in record['unchanged_records'].items():
            assert digest(root / name) == expected, name
        result = json.loads((root / case / 'light.json').read_text())
        assert result['status'] == 'complete'
        assert result['job_id'] == root.parent.name
        record['status'] = 'passed'
        record['all_artifact_hashes_restored'] = True
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print('H3_RETRY_ACCEPTANCE_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify'))
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    check(args.results, args.case, args.action)
