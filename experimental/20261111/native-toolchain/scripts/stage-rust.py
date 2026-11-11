from pathlib import Path
import os,shutil
w=Path(__file__).resolve().parent
prefix=w/'stage/opt/rust-alpha-experimental'
rust=w/'rust-build/alphaev6-unknown-linux-gnu/stage1'
triple='alphaev6-unknown-linux-gnu'
shutil.copy2(rust/'bin/rustc',prefix/'bin/rustc')
for src in (rust/'lib').glob('librustc_driver-*.so'):
    shutil.copy2(src,prefix/'lib'/src.name)
shutil.copytree(rust/'lib/rustlib'/triple,prefix/'lib/rustlib'/triple,dirs_exist_ok=True,symlinks=True)
jit=prefix/'lib/rustlib'/triple/'codegen-backends/lib'/triple/'libgccjit.so'
if jit.exists() or jit.is_symlink():jit.unlink()
jit.symlink_to(os.path.relpath(prefix/'lib/libgccjit.so',jit.parent))
support=prefix/'share/rust-alpha-experimental/support'
for name in ['libc','rustix','rustix-compiler','nix-0.30.1','nix-0.31.3','openssl-src']:
    shutil.copytree(w/'src'/name,support/name,dirs_exist_ok=True,symlinks=True,
        ignore=shutil.ignore_patterns('.git','target','__pycache__'))
for filename,compiler in [('alpha-cc','gcc'),('alpha-c++','g++')]:
    p=prefix/'bin'/filename
    p.write_text('''#!/usr/bin/env bash
set -euo pipefail
prefix=$(cd -- "$(dirname -- "$0")/.." && pwd)
extra=()
if [[ -n ${ALPHA_SYSROOT:-} ]]; then extra+=("--sysroot=$ALPHA_SYSROOT"); fi
exec "$prefix/bin/'''+compiler+'''" "${extra[@]}" "$@"
''');p.chmod(0o755)
p=prefix/'bin/cargo-alpha'
shutil.copy2(w/'cargo-alpha.in',p)
p.chmod(0o755)
licenses=prefix/'share/rust-alpha-experimental/licenses';licenses.mkdir(parents=True,exist_ok=True)
for name in ['LICENSE-APACHE','LICENSE-MIT','COPYRIGHT']:
    src=w/'src/rust'/name
    if src.exists():shutil.copy2(src,licenses/name)
print(prefix)
