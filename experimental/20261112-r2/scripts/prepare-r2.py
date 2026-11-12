from pathlib import Path
import shutil,json
p=Path.cwd();s=p/'rust_alpha_stuff/experimental/20261112-r2';w=p/'work/release-20261112-r2'
old=s/'overlay/dev-lang/rust-alpha-bin/rust-alpha-bin-1.102.0_pre20261111-r1.ebuild'
(old.parent/'rust-alpha-bin-1.102.0_pre20261111-r2.ebuild').write_text(old.read_text().replace('alpha-native-toolchain-20261112-experimental.tar.xz','alpha-native-toolchain-20261112-r2-experimental.tar.xz'))
for name in ['setup-es40.sh','config/package.accept_keywords']:
 f=s/name;f.write_text(f.read_text().replace('pre20261111-r1','pre20261111-r2'))
for name in ['package.accept_keywords','package.unmask']:
 f=s/'config'/name
 with f.open('a') as out:
  for atom in ['dev-build/just-1.59.0-r1','sys-apps/eza-0.23.5-r1']:
   out.write('='+atom+'::rust-alpha'+(' ~alpha' if name=='package.accept_keywords' else '')+'\n')
f=s/'overlay/sys-apps/eza/eza-0.23.5-r1.ebuild';t=f.read_text();a=t.index('\t# Explicit opt-in');b=t.index('\talpha-rust_finish_prepare',a);t=t[:a]+t[b:];t=t.replace('inherit cargo shell-completion alpha-rust','inherit cargo shell-completion alpha-rust\nBDEPEND+=" >=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2:0"');f.write_text(t)
d=json.loads((s/'packages.json').read_text());d['applications']+=['dev-build/just-1.59.0-r1','sys-apps/eza-0.23.5-r1'];d['compiler']='dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2';d['compiler_binaries']='Native GCC backend rebuilt with commit 7cb1ce39972c1fce610e00dbd49768caed92c0a4; compiler driver, std, Cargo, GCC and binutils retained';(s/'packages.json').write_text(json.dumps(d,indent=2)+'\n')
f=s/'README.txt';t=f.read_text().replace('maintenance kit, 2026-11-12','maintenance kit, 2026-11-12 revision 2').replace('Compiler/backend/std binary payloads retain\n  the November11 build; this is a wrapper/packaging/compatibility update.','The GCC backend is rebuilt with the compare_bytes size_t fix (7cb1ce3).\n  The compiler driver/std and other toolchain binaries retain the November11 build.').replace('pre20261111-r1','pre20261111-r2');f.write_text(t)
(s/'MAINTENANCE-R2.txt').write_text('''Revision 2: validated native compare_bytes/memcmp size_t fix.
Backend commit 7cb1ce39972c1fce610e00dbd49768caed92c0a4, local only; not pushed.
Backend SHA256 c7885247dcbef9843d6ec02836bd165719f927e059909e11c951a121898f5602.
New native application coverage: Just1.59.0-r1 and Eza0.23.5-r1 (+git).
Eza built fully optimized with isolated candidate backend and passed file listing,
permissions, symlink rendering, traversal and Git modified-status smoke checks.
Just passed dependencies/arguments/environment/child-exit smoke checks.
No upstream full test-suite claims. Existing November12 release is preserved.
Installed SDK remains unchanged until the user installs this revision.
Use setup-es40.sh --check, then --update for an existing overlay (backs up config),
then emerge --ask =dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2::rust-alpha.
Existing ES40 candidate package.env is experimental-only; remove that exact Eza
candidate override after installing r2 to use the installed backend by default.
''')
e=s/'evidence';e.mkdir(exist_ok=True)
for name in ['minimal.rs','README.txt','minimal-old.log','minimal-fixed.log','runtime-minimal.log']:
 shutil.copy2(p/'work/compare-bytes-20261112/reduction'/name,e/('compare-bytes-'+name))
shutil.copy2(p/'work/compare-bytes-20261112/compare-bytes-size-t.patch',e/'compare-bytes-size-t.patch')
support=w/'stage/opt/rust-alpha-experimental/share/rust-alpha-experimental/support/backend-fixes';support.mkdir(parents=True,exist_ok=True)
shutil.copy2(e/'compare-bytes-size-t.patch',support/'compare-bytes-size-t.patch');shutil.copy2(s/'MAINTENANCE-R2.txt',support/'PROVENANCE.txt')
b=w/'dist/rust-alpha-portage-20261112-r2';shutil.copytree(p/'work/release-20261112/dist/rust-alpha-portage-20261112',b,dirs_exist_ok=True)
for name in ['overlay','config','evidence']:shutil.copytree(s/name,b/name,dirs_exist_ok=True)
for f in s.iterdir():
 if f.is_file():shutil.copy2(f,b/f.name)
print(b)
