from pathlib import Path
import shutil,subprocess,hashlib,json
p=Path.cwd();w=p/'work/release-20261112-r2';s=p/'rust_alpha_stuff/experimental/20261112-r2';b=w/'dist/rust-alpha-portage-20261112-r2';a=b/'distfiles/alpha-native-toolchain-20261112-r2-experimental.tar.xz'
d=a.read_bytes();m=s/'overlay/dev-lang/rust-alpha-bin/Manifest';m.write_text(''.join(l for l in m.read_text().splitlines(True) if a.name not in l)+f'DIST {a.name} {len(d)} BLAKE2B {hashlib.blake2b(d).hexdigest()} SHA512 {hashlib.sha512(d).hexdigest()}\n')
for name in ['overlay','config','evidence','scripts','wrappers']:shutil.copytree(s/name,b/name,dirs_exist_ok=True)
for f in s.iterdir():
 if f.is_file():shutil.copy2(f,b/f.name)
for f in (b/'overlay').rglob('*.ebuild'):subprocess.run(['bash','-n',str(f)],check=True)
subprocess.run(['bash','-n',str(b/'setup-es40.sh')],check=True)
for f in (b/'distfiles').iterdir():
 if f.name.endswith('.sha256'):continue
 lines=[l for m in (b/'overlay').rglob('Manifest') for l in m.read_text().splitlines() if l.startswith('DIST '+f.name+' ')]
 assert lines,f.name
 fields=lines[0].split();data=f.read_bytes();assert int(fields[2])==len(data)
 for i in range(3,len(fields),2):assert hashlib.new(fields[i].lower(),data).hexdigest()==fields[i+1],f.name
(b/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(b))+'\n' for f in sorted(b.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=b,check=True)
out=w/'dist/rust-alpha-native-portage-20261112-r2.tar';subprocess.run(['tar','--sort=name','--owner=0','--group=0','--numeric-owner','-C',str(b.parent),'-cf',str(out),b.name],check=True)
out.with_suffix('.tar.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
shutil.copytree(s,p/'work/publish-20261111/rust_alpha_stuff/experimental/20261112-r2',dirs_exist_ok=True)
print(out, out.stat().st_size)
