from pathlib import Path
import difflib,hashlib,json,shutil
w=Path(__file__).resolve().parent;p=w.parent.parent
out=p/'rust_alpha_stuff/experimental/20261111/native-toolchain'
registry=p/'work/experimental-20261111/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f'
(out/'patches').mkdir(parents=True,exist_ok=True)
groups={
 'native-rust-extra.patch':(p/'work/rebase-20261111/src/rust',w/'src/rust',[
  'compiler/rustc_target/src/spec/mod.rs','compiler/rustc_target/src/spec/targets/alphaev6_unknown_linux_gnu.rs',
  'src/bootstrap/src/core/builder/cargo.rs','src/bootstrap/src/core/build_steps/compile.rs']),
 'native-libc-extra.patch':(p/'work/experimental-20261111/libc',w/'src/libc',['src/unix/linux_like/linux/gnu/alpha/constants.rs']),
 'openssl-src-alpha.patch':(registry/'openssl-src-300.6.1+3.6.3',w/'src/openssl-src',['src/lib.rs']),
 'rustix-1.1.4-alpha.patch':(registry/'rustix-1.1.4',w/'src/rustix',['Cargo.toml','src/ioctl/linux.rs']),
 'rustix-1.1.5-alpha.patch':(registry/'rustix-1.1.5',w/'src/rustix-compiler',['Cargo.toml','src/ioctl/linux.rs']),
}
for v in ['0.30.1','0.31.3']:
 groups[f'nix-{v}-alpha.patch']=(registry/f'nix-{v}',w/'src'/f'nix-{v}',['src/sys/signal.rs'])
for name,(a,b,files) in groups.items():
 text=''.join(''.join(difflib.unified_diff((a/f).read_text().splitlines(True),(b/f).read_text().splitlines(True),fromfile='a/'+f,tofile='b/'+f)) for f in files)
 (out/'patches'/name).write_text(text)
(out/'scripts').mkdir(exist_ok=True)
for pattern in ['*.sh','*.py','*-config.toml','*.in']:
 for src in w.glob(pattern):shutil.copy2(src,out/'scripts'/src.name)
for name in ['alpha-link','native-cc']:
 shutil.copy2(w/name,out/'scripts'/name)
shutil.copy2(w/'cargo-home/config.toml',out/'scripts/cargo-home-config.toml')
(out/'locks').mkdir(exist_ok=True)
for name,src in [('rust',w/'src/rust/Cargo.lock'),('library',w/'src/rust/library/Cargo.lock'),('backend',w/'src/rust/compiler/rustc_codegen_gcc/Cargo.lock'),('cargo',w/'src/rust/src/tools/cargo/Cargo.lock')]:
 shutil.copy2(src,out/'locks'/f'{name}.Cargo.lock')
(out/'evidence').mkdir(exist_ok=True)
for name in ['native-gcc-qemu.log','native-gcc-version.log','native-as-version.log','native-ld-version.log','math-qemu.log','metadata-before.log','metadata-after.log','native-cargo-qemu.log','native-cargo-https.log','native-cargo-git.log','rustix-nix-probe.log','native-rust-qemu.log','native-abi-qemu-final.log','package-qemu.log','rust-build-final.log','native-rust-format.log']:
 src=w/'logs'/name
 if src.exists():shutil.copy2(src,out/'evidence'/name)
for name in ['ld.sum','ld.log']:
 shutil.copy2(w/'native-binutils-tests'/name,out/'evidence'/('native-'+name))
for name in ['status.json','native-std-manifest.json']:
 shutil.copy2(w/name,out/name)
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'patches').glob('*.patch')}
(out/'patch-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out)
