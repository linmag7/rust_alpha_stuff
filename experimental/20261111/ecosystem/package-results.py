#!/usr/bin/env python3
"""Export verified results, portable source and ready-to-run Alpha probes."""
import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

here = Path(__file__).resolve().parent
project = here.parents[3]
work = project / 'work/ecosystem-20261111'
metadata = json.loads((work / 'offline-metadata.json').read_text())
manifest = tomllib.loads((here / 'Cargo.toml').read_text())
versions = {p['name']: p['version'] for p in metadata['packages']}
validation = dict(scope='private experimental Alpha Linux integration smoke tests under QEMU EV6', direct_dependencies={name: versions[name] for name in manifest['dependencies']}, resolved_packages=len(metadata['packages']), vendored_registry_packages=len(list((work / 'offline/alpha-ecosystem/vendor').iterdir())), existing_patches=['libc 0.2.191 Alpha', 'rustix 1.1.5 libc backend/Alpha ioctl'], new_crate_patches=[], es40_tested=False, tls_tested=False, results={})
for label in ['cross-debug', 'cross-release', 'native-debug']:
    results = json.loads((work / (label + '-results.json')).read_text())
    assert len(results) == 4 and all(r['exit_code'] == 0 for r in results), label
    validation['results'][label] = results
validation['native_toolchain_archive_sha256'] = json.loads((project / 'work/native-20261111/status.json').read_text())['archive_sha256']
validation['cross_toolchain_id'] = (project / 'work/experimental-20261111/toolchain-id').read_text().strip()
validation['lockfile_sha256'] = hashlib.sha256((here / 'Cargo.lock').read_bytes()).hexdigest()
(here / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
(here / 'evidence').mkdir(exist_ok=True)
for log in (work / 'logs').glob('*.log'):
    if log.name != 'vendor.log': shutil.copy2(log, here / 'evidence' / log.name)
bundle = work / 'offline/alpha-ecosystem'
for name in ['README.txt', 'CRATES.txt', 'validation.json', 'Cargo.lock']:
    shutil.copy2(here / name, bundle / name)
shutil.copytree(here / 'evidence', bundle / 'evidence', dirs_exist_ok=True)
for label, source in [('cross-release', work/'cross-target/alphaev6-unknown-linux-gnu/release'), ('native-debug',work/'native-target/debug')]:
    dest = bundle / 'prebuilt' / label
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['foundations', 'concurrency', 'network', 'database']:
        shutil.copy2(source/name, dest/name)
files = sorted(p for p in bundle.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
(bundle/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle)}\n' for p in files))
dist = work / 'dist'
dist.mkdir(exist_ok=True)
archive = dist / 'alpha-ecosystem-offline-20261111.tar.xz'
with archive.open('wb') as out:
    tar = subprocess.Popen(['tar', '-C', str(bundle.parent), '-c', bundle.name], stdout=subprocess.PIPE)
    result = subprocess.run(['xz', '-T4', '-3'], stdin=tar.stdout, stdout=out)
    tar.stdout.close()
    assert tar.wait() == 0 and result.returncode == 0
subprocess.run(['xz', '-t', str(archive)], check=True)
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive)+'.sha256').write_text(f'{checksum}  {archive.name}\n')
print(archive)
print(checksum)
