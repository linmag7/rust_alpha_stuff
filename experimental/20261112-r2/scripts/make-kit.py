from pathlib import Path
import shutil,json,hashlib,subprocess
root=Path.cwd();w=root/'work/release-20261112';s=root/'rust_alpha_stuff/experimental/20261112';b=w/'dist/rust-alpha-portage-20261112'
b.mkdir(parents=True,exist_ok=True)
packages=['sys-apps/ripgrep-15.2.0-r1','sys-apps/fd-10.5.0-r1','app-shells/zoxide-0.10.0-r1','app-shells/fish-4.9.3-r2','app-editors/helix-25.07.1-r1','sys-process/bottom-0.14.6-r1','dev-util/git-delta-0.19.2-r1','app-benchmarks/hyperfine-1.21.0-r1','sys-block/dust-1.2.3-r1','sys-apps/bat-0.26.1-r2','dev-vcs/git-2.56.1-r1']
(s/'config').mkdir(exist_ok=True)
(s/'config/package.accept_keywords').write_text('# Exact experimental versions only\n=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r1::rust-alpha ~alpha\n'+''.join('='+p+'::rust-alpha ~alpha\n' for p in packages)+'=dev-libs/libgit2-1.9.7::gentoo **\n')
(s/'config/package.unmask').write_text(''.join('='+p+'::rust-alpha\n' for p in packages))
(s/'config/package.use').write_text('=app-shells/fish-4.9.3-r2::rust-alpha -doc -nls -test\n=app-editors/helix-25.07.1-r1::rust-alpha grammar -debug\n=dev-vcs/git-2.56.1-r1::rust-alpha rust keyring -test\n')
(s/'config/package.use.mask').write_text('=dev-vcs/git-2.56.1-r1::rust-alpha -rust\n')
shutil.copyfile(root/'rust_alpha_stuff/experimental/20261111/portage/config/rust-alpha.conf',s/'config/rust-alpha.conf')
(s/'packages.json').write_text(json.dumps({'applications':packages,'validation':'Alpha builds and ES40 install/application smoke checks; recent Fish-through-Bat built natively; initial ripgrep/fd/zoxide packaging via QEMU; not full suites','compiler':'dev-lang/rust-alpha-bin-1.102.0_pre20261111-r1','compiler_binaries':'unchanged 20261111; maintenance wrappers/compatibility data 20261112'},indent=2)+'\n')
for name in ['overlay','config']:shutil.copytree(s/name,b/name,dirs_exist_ok=True)
(b/'distfiles').mkdir(exist_ok=True)
for f in (root/'work/portage-20261111/distfiles').glob('*experimental*.tar.xz'):shutil.copy2(f,b/'distfiles'/f.name)
for f in s.iterdir():
 if f.is_file():shutil.copy2(f,b/f.name)
