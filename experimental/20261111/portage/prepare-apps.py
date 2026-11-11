#!/usr/bin/env python3
"""Resolve experimental patches, vendor, and cross-build pinned application sources."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
from urllib.request import urlopen

here=Path(__file__).resolve().parent
project=here.parents[3]
work=project/'work/portage-20261111'
native=project/'work/native-20261111'
experimental=project/'work/experimental-20261111'
apps={'ripgrep':('15.2.0','BurntSushi/ripgrep','15.2.0','rg'),'fd':('10.5.0','sharkdp/fd','v10.5.0','fd'),'zoxide':('0.10.0','ajeetdsouza/zoxide','v0.10.0','zoxide')}
patches={'libc':'libc','rustix':'rustix-compiler','rustix_114':'rustix','nix_030':'nix-0.30.1','nix_031':'nix-0.31.3','openssl-src':'openssl-src'}
options=[]
for key,directory in patches.items():
 if '_' in key:options+=['--config',f'patch.crates-io.{key}.package="{key.split("_")[0]}"']
 options+=['--config',f'patch.crates-io.{key}.path="{native/"src"/directory}"']
env=os.environ.copy()
env.update(TMPDIR=str(work/'tmp'),CARGO_HOME=str(experimental/'cargo-home'),RUSTC=str(native/'host-rust-build/x86_64-unknown-linux-gnu/stage2/bin/rustc'),CARGO_PROFILE_RELEASE_LTO='false',CARGO_PROFILE_RELEASE_STRIP='false',CARGO_PROFILE_RELEASE_DEBUG='0',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='16')
cargo=str(experimental/'rust-build/x86_64-unknown-linux-gnu/stage0/bin/cargo')
name=sys.argv[1]
version,repo,tag,binary=apps[name]
source=work/'src'/f'{name}-{version}'
archive=work/'distfiles'/f'{name}-{version}-upstream.tar.gz'
for d in ['tmp','logs','distfiles','src']: (work/d).mkdir(parents=True,exist_ok=True)
if not archive.exists():archive.write_bytes(urlopen(f'https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}',timeout=60).read())
if not source.exists():
 with tarfile.open(archive) as tar:tar.extractall(work/'src',filter='data')
original_lock=work/'reference'/f'{name}-{version}.Cargo.lock'
if not original_lock.exists():original_lock.write_bytes((source/'Cargo.lock').read_bytes())
with (work/'logs'/f'{name}-prepare.log').open('w') as log:
 def run(cmd,**kwargs):return subprocess.run(cmd,cwd=source,env=env,stderr=log,stdout=log,check=True,**kwargs)
 run([cargo,*options,'update','-p','libc','--precise','0.2.191'])
 # Upstream ripgrep sets a host-oriented static-link default; our launcher owns linking.
 config=source/'.cargo/config.toml'
 if config.exists():
  backup=work/'reference'/f'{name}-upstream-config.toml'
  if not backup.exists():backup.write_bytes(config.read_bytes())
  config.unlink()
 result=subprocess.run([cargo,*options,'vendor','--locked',str(source/'vendor')],cwd=source,env=env,stderr=log,stdout=subprocess.PIPE,text=True,check=True)
 config.parent.mkdir(exist_ok=True)
 config.write_text(result.stdout.replace(str(source/'vendor'),'vendor'))
 (work/'reference'/f'{name}-source.json').write_text(json.dumps(dict(upstream=f'https://github.com/{repo}',tag=tag,version=version,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2)+'\n')
with (work/'logs'/f'{name}-cross-build.log').open('w') as log:
 env['RUSTFLAGS']=f'-Clinker={native}/alpha-link --cfg=rustix_use_libc --cfg=rustix_no_linux_raw'
 cmd=[str(experimental/'cargo-alpha'),'build','--manifest-path',str(source/'Cargo.toml'),'--target-dir',str(work/'cross-target'/name),'--release','--locked','--offline','--bin',binary,'-j8',*options]
 subprocess.run(cmd,cwd=project,env=env,stdout=log,stderr=log,check=True)
print(name,'cross-build PASS',flush=True)
