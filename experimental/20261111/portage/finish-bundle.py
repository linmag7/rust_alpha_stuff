#!/usr/bin/env python3
from pathlib import Path
import os,sys,tarfile,hashlib,json,shutil,subprocess,bz2
here=Path(__file__).resolve().parent;p=here.parents[3];w=p/'work/portage-20261111'
os.environ.update(PORTAGE_OVERRIDE_EPREFIX='',PORTAGE_CONFIGROOT=str(w/'config'),ROOT=str(w/'root'),TMPDIR=str(w/'tmp'))
import portage
from portage.gpkg import gpkg
bundle=w/'release/rust-alpha-portage'
bundle.mkdir(parents=True,exist_ok=True)
for d in ['overlay','config']:shutil.copytree(here/d,bundle/d,dirs_exist_ok=True)
for f in ['README.txt','setup-es40.sh']:shutil.copy2(here/f,bundle/f)
for d in ['distfiles','binpkgs','evidence','source-provenance']:(bundle/d).mkdir(exist_ok=True)
results=[]
recipes=[('dev-lang','rust-alpha-bin','1.102.0_pre20261111',None),('sys-apps','ripgrep','15.2.0-r1','rg'),('sys-apps','fd','10.5.0-r1','fd'),('app-shells','zoxide','0.10.0-r1','zoxide')]
for cat,name,version,binary in recipes:
 pf=f'{name}-{version}'
 files=list((w/'binpkgs'/cat/name).glob('*.gpkg.tar'))
 package=max(files,key=lambda x:x.stat().st_mtime)
 obj=gpkg(portage.settings,gpkg_file=str(package),verify_signature=False)
 obj._verify_binpkg()
 metadata=obj.get_metadata()
 assert metadata['CHOST'].strip()==b'alphaev67-unknown-linux-gnu'
 with tarfile.open(package) as outer:
  member=next(m for m in outer if '/image.tar.' in m.name)
  with tarfile.open(fileobj=outer.extractfile(member),mode='r|*') as image:
   count=0
   for m in image:
    assert m.uid==m.gid==0,(package,m.name,m.uid,m.gid)
    count+=1
 if binary:
  assert 'PASS' in (w/'logs'/f'{name}-portage.log').read_text()
  environment=bz2.decompress(metadata['environment.bz2']).decode()
  assert '[[ ${MERGE_TYPE} == binary ]]' in environment
  extracted=w/'verify-extracted'/name
  extracted.mkdir(parents=True,exist_ok=True)
  obj.decompress(str(extracted))
  executable=extracted/'usr/bin'/binary
  assert executable.read_bytes()[:6]==b'\x7fELF\x02\x01'
  env=os.environ.copy();env['QEMU_LD_PREFIX']=str(p/'work/sdk/release/alpha-sdk/sysroot')
  with (w/'logs'/f'{name}-gpkg-smoke.log').open('w') as log:
   subprocess.run(['bash',str(here/'overlay'/cat/name/'files/smoke.sh'),str(executable),str(w/'fixtures'/f'{name}-gpkg')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
 rel=package.relative_to(w/'binpkgs');dest=bundle/'binpkgs'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(package,dest)
 results.append(dict(package=f'{cat}/{pf}',path=str(rel),sha256=hashlib.sha256(package.read_bytes()).hexdigest(),files=count,root_owned=True))
# Keep only the selected, verified build instances in the distributable index.
index=(w/'binpkgs/Packages').read_text().split('\n\n')
header=index[0].splitlines();header=[('PACKAGES: 4' if x.startswith('PACKAGES:') else x) for x in header]
paths={r['path'] for r in results}
entries=[entry for entry in index[1:] if any('PATH: '+path+'\n' in entry+'\n' for path in paths)]
assert len(entries)==4
(bundle/'binpkgs/Packages').write_text('\n'.join(header)+'\n\n'+'\n\n'.join(entries)+'\n\n')
for f in (w/'distfiles').glob('*experimental*.tar.xz'):shutil.copy2(f,bundle/'distfiles'/f.name)
for pattern in ['*-cross-smoke.log','*-gpkg-smoke.log','*-portage.log','*-normalize.log','emerge-pretend.log']:
 for f in (w/'logs').glob(pattern):shutil.copy2(f,bundle/'evidence'/f.name)
for f in (w/'reference').iterdir():
 if f.is_file():shutil.copy2(f,bundle/'source-provenance'/f.name)
for name,version in [('ripgrep','15.2.0'),('fd','10.5.0'),('zoxide','0.10.0')]:shutil.copy2(w/'src'/f'{name}-{version}'/'Cargo.lock',bundle/'source-provenance'/f'{name}-prepared.Cargo.lock')
report=dict(scope='private experimental Portage integration',es40_installed=False,host_sandbox_tested=False,applications_native_built_and_smoke_tested=True,packages=results)
for dest in [here/'validation.json',bundle/'validation.json']:dest.write_text(json.dumps(report,indent=2)+'\n')
files=sorted(f for f in bundle.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
(bundle/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(bundle)}\n' for f in files))
archive=w/'release/rust-alpha-portage-20261111.tar'
subprocess.run(['tar','--owner=0','--group=0','-C',str(bundle.parent),'-cf',str(archive),bundle.name],check=True)
Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print(archive,flush=True)
