from pathlib import Path
import shutil,subprocess,json
root=Path.cwd();w=root/'work/release-20261112';src=root/'rust_alpha_stuff/experimental/20261112';old=root/'rust_alpha_stuff/experimental/20261111/portage'
shutil.copytree(old/'overlay',src/'overlay',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for f in ['RUST-BUILD-RECIPE.txt']:shutil.copyfile(old/f,src/f)
stage=w/'stage/opt/rust-alpha-experimental';support=stage/'share/rust-alpha-experimental/support'
subprocess.run(['tar','-xf',str(w/'inputs/crt.tar'),'-C',str(support)],check=True)
shutil.copytree(src/'overlay/eclass/alpha-rust',support/'compatibility-patches',dirs_exist_ok=True)
p=stage/'bin/cargo-alpha';s=p.read_text().replace('-Cpanic=abort -Clto=off','-Cpanic=abort -Clto=off -Cllvm-args=-mieee');p.write_text(s)
for name in ['alpha-cc','alpha-c++']:
 p=stage/'bin'/name;s=p.read_text().replace('extra=()','extra=("-B$prefix/share/rust-alpha-experimental/support/crt/")');p.write_text(s)
(stage/'RELEASE-20261112.txt').write_text('Native SDK maintenance release. Compiler/backend/std binaries unchanged from 20261111. Cargo defaults to -mieee. C/C++ linker drivers select the five bundled Gentoo GCC startup objects fixing __dso_handle. Support includes versioned compatibility patches used by the overlay. No claim of rebuilt IEEE std or full ABI validation.\n')
(src/'wrappers').mkdir(exist_ok=True)
for name in ['cargo-alpha','alpha-cc','alpha-c++']:shutil.copyfile(stage/'bin'/name,src/'wrappers'/name)
# New compiler revision, keep original package for provenance/rollback.
p=src/'overlay/dev-lang/rust-alpha-bin';oldfile=p/'rust-alpha-bin-1.102.0_pre20261111.ebuild'
s=oldfile.read_text().replace('alpha-native-toolchain-20261111-experimental.tar.xz','alpha-native-toolchain-20261112-experimental.tar.xz');(p/'rust-alpha-bin-1.102.0_pre20261111-r1.ebuild').write_text(s)
