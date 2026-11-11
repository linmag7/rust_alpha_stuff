#!/usr/bin/env python3
"""Prepare a self-contained crate source tree; toolchain installed separately."""
from pathlib import Path
import os
import shutil
import subprocess

here = Path(__file__).resolve().parent
project = here.parents[3]
work = project / 'work/ecosystem-20261111'
bundle = work / 'offline/alpha-ecosystem'
bundle.mkdir(parents=True, exist_ok=True)
for name in ['Cargo.toml', 'Cargo.lock', 'build-on-alpha.sh', 'README.txt']:
    shutil.copy2(here / name, bundle / name)
shutil.copytree(here / 'src', bundle / 'src', dirs_exist_ok=True)
(bundle / '.cargo').mkdir(exist_ok=True)
config = bundle / '.cargo/config.toml'
native = project / 'work/native-20261111'
patches = {'libc': 'libc', 'rustix': 'rustix-compiler', 'rustix_114': 'rustix', 'nix_030': 'nix-0.30.1', 'nix_031': 'nix-0.31.3', 'openssl-src': 'openssl-src'}
cmd = [str(project / 'work/experimental-20261111/rust-build/x86_64-unknown-linux-gnu/stage0/bin/cargo')]
for key, directory in patches.items():
    if key in ['rustix_114', 'nix_030', 'nix_031']:
        cmd += ['--config', f'patch.crates-io.{key}.package="{key.split("_")[0]}"']
    cmd += ['--config', f'patch.crates-io.{key}.path="{native / "src" / directory}"']
cmd += ['vendor', '--locked', '--manifest-path', str(bundle / 'Cargo.toml'), str(bundle / 'vendor')]
env = os.environ.copy()
env.update(CARGO_HOME=str(project / 'work/experimental-20261111/cargo-home'), TMPDIR=str(work / 'tmp'), RUSTC=str(native / 'host-rust-build/x86_64-unknown-linux-gnu/stage2/bin/rustc'))
with (work / 'logs/vendor.log').open('w') as log:
    result = subprocess.run(cmd, env=env, cwd=project, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
# Make Cargo's generated source replacement portable after extraction.
config.write_text(result.stdout.replace(str(bundle / 'vendor'), 'vendor'))
print(bundle)
