"""Read-only CPU rescue of small result metadata when the OSS mount fails."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import re
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'dlc[a-z0-9]+', args.job)
    root = Path('/cpfs/world-model/lynnreal-omni/results') / args.job
    assert (root / 'retry-acceptance.json').is_file()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.txt'):
                assert path.stat().st_size < 4 * 1024 * 1024
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    payload = buffer.getvalue()
    assert len(payload) < 2 * 1024 * 1024
    print('H3_METADATA_SHA256=' + hashlib.sha256(payload).hexdigest(), flush=True)
    encoded = base64.b64encode(payload).decode()
    for index in range(0, len(encoded), 4096):
        print('H3_METADATA_CHUNK=' + encoded[index:index + 4096], flush=True)
    print('H3_METADATA_END', flush=True)


if __name__ == '__main__':
    main()
