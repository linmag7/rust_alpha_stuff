#!/usr/bin/env python3
"""Check the native normal/build graph for known Alpha SDK blockers."""
import json
import sys
from pathlib import Path


def check(metadata, workdir):
    packages = {p['id']: p for p in metadata['packages']}
    resolve = metadata['resolve']
    nodes = {n['id']: n for n in resolve['nodes']}
    pending = [resolve['root']] if resolve.get('root') else list(metadata['workspace_members'])
    seen = set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        for dep in nodes[key].get('deps', []):
            if any(k.get('kind') != 'dev' for k in dep['dep_kinds']):
                pending.append(dep['pkg'])
    errors = []
    known = {'lfs-core': {'0.20.0'}, 'crossterm': {'0.28.1'}, 'which': {'4.4.2'}, 'rustix': {'1.1.3'}, 'sysinfo': {'0.27.8', '0.29.11', '0.33.1', '0.39.5'}, 'terminal_size': {'0.2.6', '0.4.1'}}
    for key in sorted(seen):
        p = packages[key]
        name, version = p['name'], p['version']
        features = set(nodes[key].get('features', []))
        if name == 'rustix' and version == '0.38.44':
            errors.append('rustix 0.38.44: libc backend still requires raw bindings; adapt which 4.4.2 to SDK rustix 1.1.5 instead')
        if name == 'linux-raw-sys':
            errors.append(f'{name} {version}: active raw Linux bindings lack validated Alpha support; trace with cargo tree -i {name}@{version} and use the libc backend')
        if name == 'crossterm' and version == '0.28.1' and 'libc' not in features:
            errors.append('crossterm 0.28.1: enable libc feature for configure and build')
        needs_patch = version in known.get(name, set())
        if name == 'nix':
            affected = {'fs', 'signal', 'process', 'aio'} if version == '0.27.1' else {'fs'} if version == '0.30.1' else {'fs', 'resource', 'mman'}
            needs_patch = version in {'0.27.1', '0.30.1', '0.31.3'} and bool(features & affected)
        if needs_patch:
            expected = Path(workdir) / f'alpha-{name}-{version}' / 'Cargo.toml'
            if Path(p['manifest_path']).resolve() != expected.resolve():
                alias = {'0.27.1': 'nix_027', '0.30.1': 'nix_030', '0.31.3': 'nix_031'}.get(version, name) if name == 'nix' else name
                if name == 'rustix' and version == '1.1.3':
                    alias = 'rustix_113 (also set package=rustix)'
                errors.append(f'{name} {version}: enable alpha-rust_patch_crate {name} {version} SOURCE {alias} in src_prepare')
    return errors


if __name__ == '__main__':
    problems = check(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
    if problems:
        print('Known Alpha dependency blockers:', file=sys.stderr)
        for problem in problems:
            print('  ' + problem, file=sys.stderr)
        sys.exit(1)
    print('Known Alpha dependency preflight: PASS (not a full ABI validation)')
