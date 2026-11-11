#!/usr/bin/env python3
"""Run each Alpha probe with bounded time and record reproducible evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

here = Path(__file__).resolve().parent
project = here.parents[3]
work = project / 'work/ecosystem-20261111'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['cross', 'native'], default='cross')
parser.add_argument('--profile', choices=['debug', 'release'], default='debug')
args = parser.parse_args()
target = work / (args.mode + '-target')
if args.mode == 'cross':
    target /= 'alphaev6-unknown-linux-gnu'
target /= args.profile
sdk = project / 'work/sdk/release/alpha-sdk'
env = os.environ.copy()
env['TMPDIR'] = str(work / 'tmp')
env['QEMU_LD_PREFIX'] = str(sdk / 'sysroot')
results = []
for name in ['foundations', 'concurrency', 'network', 'database']:
    binary = target / name
    label = f'{args.mode}-{args.profile}-{name}'
    log = work / 'logs' / (label + '.log')
    header = binary.read_bytes()[:20]
    assert header[:6] == b'\x7fELF\x02\x01' and int.from_bytes(header[18:20], 'little') == 0x9026
    start = time.monotonic()
    with log.open('w') as output:
        try:
            result = subprocess.run([str(sdk / 'bin/qemu-alpha-static'), '-cpu', 'ev6', '-L', str(sdk / 'sysroot'), str(binary)], env=env, stdout=output, stderr=subprocess.STDOUT, timeout=90)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            output.write('\nTIMEOUT after 90 seconds\n')
    results.append(dict(probe=name, exit_code=code, seconds=round(time.monotonic()-start, 2), binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), log=str(log)))
    print(f'{label}: {"PASS" if code == 0 else "FAIL"}', flush=True)
    print(log.read_text(), end='', flush=True)
(work / f'{args.mode}-{args.profile}-results.json').write_text(json.dumps(results, indent=2)+'\n')
raise SystemExit(any(r['exit_code'] != 0 for r in results))
