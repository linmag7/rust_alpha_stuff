#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path.cwd();src=root/'rust_alpha_stuff/experimental/20261112';w=root/'work/release-20261112';b=w/'dist/rust-alpha-portage-20261112'
archive=b/'distfiles/alpha-native-toolchain-20261112-experimental.tar.xz'
data=archive.read_bytes();line=f'DIST {archive.name} {len(data)} BLAKE2B {hashlib.blake2b(data).hexdigest()} SHA512 {hashlib.sha512(data).hexdigest()}\n'
p=src/'overlay/dev-lang/rust-alpha-bin/Manifest';lines=[l for l in p.read_text().splitlines(True) if archive.name not in l];p.write_text(''.join(lines)+line)
inputs={}
for f in [w/'inputs/crt.tar',root/'work/portage-20261111/distfiles/alpha-native-toolchain-20261111-experimental.tar.xz']:
 inputs[str(f.relative_to(root))]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
(src/'release-inputs.json').write_text(json.dumps({'date':'2026-11-12','inputs':inputs,'compiler_rebuild':False,'installed_sdk_replaced':False,'published':False},indent=2)+'\n')
for f in src.iterdir():
 if f.is_file():shutil.copy2(f,b/f.name)
for name in ['overlay','config','wrappers','scripts']:shutil.copytree(src/name,b/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(w/'evidence',b/'evidence',dirs_exist_ok=True)
subprocess.run(['tar','-xf',str(w/'evidence/app-smokes.tar'),'-C',str(b/'evidence')],check=True)
for p in sorted((b/'distfiles').iterdir()):
 if p.suffix=='.sha256':continue
 # Check every bundled distfile against an overlay Manifest.
 matches=[line for f in (b/'overlay').rglob('Manifest') for line in f.read_text().splitlines() if line.startswith('DIST '+p.name+' ')]
 assert matches,p.name
 fields=matches[0].split();assert int(fields[2])==p.stat().st_size
 d=p.read_bytes()
 for i in range(3,len(fields),2):
  algorithm={'BLAKE2B':'blake2b','SHA512':'sha512','SHA256':'sha256'}[fields[i]]
  assert hashlib.new(algorithm,d).hexdigest()==fields[i+1],p.name
for p in (b/'overlay').rglob('*.ebuild'):subprocess.run(['bash','-n',str(p)],check=True)
subprocess.run(['bash','-n',str(b/'setup-es40.sh')],check=True)
(b/'evidence/release-verification.log').write_text('Bundled DIST hashes, ebuild/installer bash syntax: PASS\nArchive xz integrity: PASS (repack-sdk.py)\nNative setup --check: PASS; see installer-check.log (payload/installer unchanged after check)\n')
files=sorted(p for p in b.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(b/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(b))+'\n' for p in files))
subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=b,check=True)
out=w/'dist/rust-alpha-native-portage-20261112.tar'
subprocess.run(['tar','--sort=name','--owner=0','--group=0','--numeric-owner','-C',str(b.parent),'-cf',str(out),b.name],check=True)
out.with_suffix('.tar.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
print(out, out.stat().st_size)
