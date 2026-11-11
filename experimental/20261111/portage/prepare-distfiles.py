#!/usr/bin/env python3
from pathlib import Path
import concurrent.futures,hashlib,shutil,subprocess
here=Path(__file__).resolve().parent
project=here.parents[3]
w=project/'work/portage-20261111'
dist=w/'distfiles'
recipes=[('sys-apps/ripgrep','15.2.0'),('sys-apps/fd','10.5.0'),('app-shells/zoxide','0.10.0')]
def pack(item):
 atom,version=item;name=atom.split('/')[1]
 output=dist/f'{name}-{version}-alpha-experimental-20261111.tar.xz'
 with output.open('wb') as out:
  tar=subprocess.Popen(['tar','-C',str(w/'src'),'-c',f'{name}-{version}'],stdout=subprocess.PIPE)
  result=subprocess.run(['xz','-T2','-3'],stdin=tar.stdout,stdout=out)
  tar.stdout.close()
  assert tar.wait()==0 and result.returncode==0
 return atom,output
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: files=list(pool.map(pack,recipes))
toolchain=project/'work/native-20261111/dist/alpha-native-toolchain-20261111-experimental.tar.xz'
local=dist/toolchain.name
if not local.exists():local.hardlink_to(toolchain)
files.append(('dev-lang/rust-alpha-bin',local))
for atom,path in files:
 data=path.read_bytes()
 line=f'DIST {path.name} {len(data)} BLAKE2B {hashlib.blake2b(data).hexdigest()} SHA512 {hashlib.sha512(data).hexdigest()}\n'
 (here/'overlay'/atom/'Manifest').write_text(line)
 print(atom,path.stat().st_size,flush=True)
