"""CPU-only controlled cold/warm application-cache preparation; never clear shared caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

PERSIST = Path('/cpfs/world-model/lynnreal-omni')
ARCHIVE = PERSIST / 'cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--mode', choices=['cold', 'warm'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.experiment):
        raise ValueError('Invalid experiment namespace')
    root = PERSIST / 'cache/startup-ab' / args.experiment
    started = time.perf_counter()
    receipt = {'mode': args.mode, 'experiment': args.experiment, 'root': str(root),
               'gpu_used': False, 'job_id': os.environ.get('DLC_JOB_ID'),
               'code_sha': os.environ.get('LYNNREAL_RELEASE_SHA'), 'started_unix': time.time(),
               'scope': 'application environment/weight cache; source CPFS, host page cache and image cache uncontrolled'}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'cold':
        root.mkdir(parents=True, exist_ok=False)
    else:
        marker = json.loads((root / 'READY.json').read_text())
        assert marker['complete'] and marker['experiment'] == args.experiment
        assert marker['code_sha'] == receipt['code_sha'], 'Use identical code for both runs'
    env_started = time.perf_counter()
    env_marker = json.loads((ARCHIVE.parent / 'h3-vae-ab-env-ready.json').read_text())
    if args.mode == 'cold':
        assert digest(ARCHIVE) == env_marker['archive_sha256']
        (root / 'environment').mkdir()
        subprocess.run(['tar', '-xzf', str(ARCHIVE), '-C', str(root / 'environment')], check=True)
        environment_files = {str(f.relative_to(root / 'environment')): f.stat().st_size
                             for f in (root / 'environment').rglob('*') if f.is_file()}
    else:
        environment_files = marker['environment_files']
        for name, size in environment_files.items():
            assert (root / 'environment' / name).stat().st_size == size
        assert marker['archive_sha256'] == env_marker['archive_sha256']
    receipt['environment_prepare_seconds'] = time.perf_counter() - env_started
    weights_started = time.perf_counter()
    copied = 0
    for name, dirname in [('h3', 'h3-bfc8ed0353f5a9733be73e6b2c98ec0948195b86'),
                          ('light', 'light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a')]:
        source = PERSIST / 'models' / dirname
        target = root / name
        source_marker = source / 'H3_VAE_AB_READY.json'
        model = json.loads(source_marker.read_text())
        assert model['complete']
        if args.mode == 'cold':
            target.mkdir()
        else:
            assert digest(target / source_marker.name) == digest(source_marker)
        for entry in model['files']:
            dst = target / entry['path']
            assert entry['verified']
            if args.mode == 'cold':
                dst.parent.mkdir(parents=True, exist_ok=True)
                sha = hashlib.sha256() if entry['sha256'] else hashlib.sha1()
                if not entry['sha256']:
                    sha.update(f"blob {entry['size']}\0".encode())
                with (source / entry['path']).open('rb') as src, dst.open('xb') as out:
                    for block in iter(lambda: src.read(16 << 20), b''):
                        sha.update(block)
                        out.write(block)
                assert sha.hexdigest() == (entry['sha256'] or entry['git_blob_oid']), entry['path']
                copied += entry['size']
            assert dst.stat().st_size == entry['size']
        if args.mode == 'cold':
            shutil.copyfile(source_marker, target / source_marker.name)
        receipt[name] = str(target)
        print(name, args.mode, 'verified', flush=True)
    receipt.update(weights_prepare_seconds=time.perf_counter() - weights_started,
                   copied_bytes=copied, elapsed_seconds=time.perf_counter() - started,
                   finished_unix=time.time(), complete=True,
                   archive_sha256=env_marker['archive_sha256'], environment_files=environment_files)
    # Publish readiness only after all files and the output directory were checked.
    if args.mode == 'cold':
        write(root / 'READY.json', receipt)
    write(args.output / 'startup-cache.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'environment_files'}), flush=True)


if __name__ == '__main__':
    main()
