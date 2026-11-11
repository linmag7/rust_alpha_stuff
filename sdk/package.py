#!/usr/bin/env python3
"""Assemble the validated input artifacts into a relocatable SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parent
PROJECT = REPO.parent
HOST = 'x86_64-unknown-linux-gnu'
TARGET = 'alphaev6-unknown-linux-gnu'
VERSION = '0.1.0-alpha1'


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def tree(source, destination):
    shutil.copytree(source, destination, symlinks=False,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', 'target'))
    for directory in [destination, *(p for p in destination.rglob('*') if p.is_dir())]:
        directory.chmod(directory.stat().st_mode | 0o700)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'work/sdk/alpha-sdk')
    parser.add_argument('--cross', type=Path, default=Path('/home/lindholm/x-tools')/TARGET)
    parser.add_argument('--stage0', type=Path, default=Path('/home/lindholm/toolchains/rust-git-dev'))
    parser.add_argument('--sysroot-archive', type=Path, default=PROJECT/'work/sdk/inputs/es40-glibc-2.41.tar.gz')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('Output already exists; choose a new directory to preserve installed SDKs')
    stage = PROJECT/'work/rust-build'/HOST/'stage1'
    rebased = PROJECT/'work/rebase-20261111'
    inputs = [stage/'bin/rustc', PROJECT/'work/sdk/backend-target/release/librustc_codegen_gcc.so',
              args.cross/'lib/libgccjit.so', args.sysroot_archive]
    inputs.extend(sorted((stage/'lib').glob('librustc_driver*.so')))
    inputs.extend(sorted((stage/'lib').glob('libLLVM.so*')))
    for source in inputs:
        if not source.is_file():
            parser.error('Missing input: ' + str(source))
    out.mkdir(parents=True)
    print('Assembling', out, flush=True)
    for name in ('bin', 'lib', 'libexec'):
        tree(args.cross/name, out/'gcc'/name)
    for name, target in (('libgccjit.so', 'libgccjit.so.0'), ('libgccjit.so.0', 'libgccjit.so.0.0.1')):
        (out/'gcc/lib'/name).unlink()
        (out/'gcc/lib'/name).symlink_to(target)
    tree(args.cross/TARGET/'bin', out/'gcc'/TARGET/'bin')
    # A single sysroot serves the C compiler, Rust linker and emulator.
    (out/'gcc'/TARGET/'sysroot').symlink_to('../../sysroot')
    (out/'sysroot').mkdir()
    with tarfile.open(args.sysroot_archive) as archive:
        archive.extractall(out/'sysroot', filter='data')
    (out/'sysroot/lib').symlink_to('usr/lib')
    for source in (args.cross/TARGET/'sysroot/lib').glob('libatomic*'):
        copy(source, out/'sysroot/usr/lib'/source.name)
    for source in (args.cross/TARGET/'sysroot/lib').glob('libgcc_s*'):
        copy(source, out/'sysroot/usr/lib'/source.name)
    for name in ('cargo', 'rustdoc'):
        copy(args.stage0/'bin'/name, out/'toolchain/bin'/name)
    copy(stage/'bin/rustc', out/'toolchain/bin/rustc')
    for pattern in ('librustc_driver*.so', 'libLLVM.so*'):
        for source in (stage/'lib').glob(pattern):
            copy(source, out/'toolchain/lib'/source.name)
    manifest = args.stage0/'lib/rustlib'/('manifest-rust-std-' + HOST)
    for record in manifest.read_text().splitlines():
        if record.startswith('file:'):
            source = Path(record[5:])
            copy(source, out/'toolchain'/source.relative_to(args.stage0))
    copy(PROJECT/'work/sdk/backend-target/release/librustc_codegen_gcc.so', out/'lib/librustc_codegen_gcc.so')
    jit = out/'toolchain/lib/rustlib'/HOST/'codegen-backends/lib'/TARGET/'libgccjit.so'
    jit.parent.mkdir(parents=True)
    jit.symlink_to(os.path.relpath(out/'gcc/lib/libgccjit.so', jit.parent))
    tree(rebased/'src/validated-std/library', out/'src/rust/library')
    src_link = out/'toolchain/lib/rustlib/src/rust'
    src_link.parent.mkdir(parents=True)
    src_link.symlink_to(os.path.relpath(out/'src/rust', src_link.parent))
    tree(rebased/'src/libc', out/'libc')
    target = json.loads((REPO/'target_specs'/ (TARGET + '-std.json')).read_text())
    copy(REPO/'target_specs'/ (TARGET + '-std.json'), out/'targets'/ (TARGET + '.json'))
    copy(Path('/usr/bin/qemu-alpha-static'), out/'bin/qemu-alpha-static')
    copy(SOURCE/'USAGE.txt', out/'README.txt')
    copy(SOURCE/'cli.py', out/'libexec/alpha-sdk.py')
    (out/'libexec/alpha-sdk.py').chmod(0o755)
    for name in ('cargo-alpha', 'alpha-runner', 'alpha-rustc', 'alpha-rustdoc', 'alpha-cc', 'alpha-ar'):
        (out/'bin'/name).symlink_to('../libexec/alpha-sdk.py')
    tree(REPO/'patches', out/'provenance/baseline-patches')
    tree(REPO/'rebased/20261111/patches', out/'provenance/rebased-patches')
    copy(REPO/'rebased/20261111/upstream.json', out/'provenance/upstream.json')
    for name in ('LICENSE-APACHE', 'LICENSE-MIT', 'COPYRIGHT'):
        source = PROJECT/'work/rust-src'/name
        if source.exists():
            copy(source, out/'licenses/rust'/name)
    for name in ('COPYING', 'COPYING3', 'COPYING.LIB', 'COPYING3.LIB', 'COPYING.RUNTIME'):
        source = rebased/'src/gcc'/name
        if source.exists():
            copy(source, out/'licenses/gcc'/name)
    copy(SOURCE/'cli.py', out/'provenance/sdk/cli.py')
    copy(Path(__file__), out/'provenance/sdk/package.py')
    copy(SOURCE/'export-es40-sysroot.py', out/'provenance/sdk/export-es40-sysroot.py')
    copy(SOURCE/'validate.py', out/'provenance/sdk/validate.py')
    copy(SOURCE/'build-backend.sh', out/'provenance/sdk/build-backend.sh')
    copy(SOURCE/'backend-relocatable-lto.patch', out/'provenance/sdk/backend-relocatable-lto.patch')
    tree(SOURCE/'tests', out/'provenance/sdk/tests')
    # Debug sections are unnecessary at runtime; never strip the input artifacts.
    for source in [out/'gcc/lib/libgccjit.so.0.0.1']:
        source.chmod(source.stat().st_mode | 0o200)
        subprocess.run(['strip', '--strip-debug', str(source)], check=True)
    (out/'config.example.json').write_text(json.dumps({'es40':{'target':'root@192.168.1.82','local_user':'codex'}}, indent=2)+'\n')
    checksums = {str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*'))
                 if path.is_file() and not path.is_symlink()}
    build_id = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()[:16]
    (out/'build-id').write_text(build_id+'\n')
    checksums['build-id'] = digest(out/'build-id')
    metadata = {
        'build_id':build_id,
        'version': VERSION, 'host': HOST, 'target': TARGET, 'cpu_minimum':'ev6',
        'glibc_baseline':'2.41', 'panic':'abort',
        'rust':'1.94.0-dev + Alpha ABI/std patches', 'gcc':'16.0.0 20251008',
        'libc':'0.2.191 + Alpha bindings',
        'source_revisions':{'rust':'7249ed419949185359935372378d4fcca937e31f',
                            'backend':'02f889aec5dc9b4e51d4cfd86c67ce5e4a0bbf72',
                            'libc':'ccc2ff4aaab962f7fde7b8edda7379c045b1ca12'},
        'std_compatibility':'POSIX_SPAWN_SETSID cast for libc 0.2.191',
        'inputs':{str(path):digest(path) for path in inputs},
        'sha256':checksums,
    }
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.archive:
        archive = out.parent/(out.name+'-'+VERSION+'.tar.gz')
        subprocess.run(['tar','--sort=name','--mtime=@1794355200','--owner=0','--group=0','--numeric-owner','-czf',str(archive),'-C',str(out.parent),out.name],check=True)
        print(archive)
    print('SDK ready:', out)


if __name__ == '__main__':
    main()
