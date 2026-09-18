"""Bounded retry of small TAE evidence/previews; lossless masters remain on CPFS."""
from pathlib import Path
import subprocess
import sys
import time

source, target = map(Path, sys.argv[1:])
files = [p for p in source.rglob('*') if p.is_file()
         and (p.suffix in ('.json', '.txt', '.log', '.html') or p.name.endswith('-web.mp4'))]
files.sort(key=lambda p: (p.suffix == '.mp4', str(p)))
for path in files:
    destination = target/path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            subprocess.run(['cp',str(path),str(destination)],check=True,timeout=90)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2: raise
            time.sleep(2)
    print('uploaded',path.relative_to(source),flush=True)
if (source/'READY').exists():
    subprocess.run(['cp',str(source/'READY'),str(target/'READY')],check=True,timeout=30)
