from pathlib import Path
import os,subprocess,json,resource
w=Path(__file__).resolve().parent;p=w.parent.parent
exp=p/'work/experimental-20261111';sdk=p/'work/sdk/release/alpha-sdk'
build_id=(exp/'toolchain-id').read_text().strip()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
results=[]
restore=os.environ.get('ALPHA_RESTORE_PROBES')=='1'
env=os.environ.copy();env.update(ALPHA_NO_STD='1',ALPHA_PROBE_PIE='1',ALPHA_LIBC_PROBE_PIE='1',ALPHA_RUNNER='qemu',ALPHA_CACHE_DIR=str(exp/'sdk-cache'))
if restore:
 env.pop('ALPHA_PROBE_PIE',None);env.pop('ALPHA_LIBC_PROBE_PIE',None)
for profile in ['debug','release']:
 for probe,binaries in [('abi_probe',['abi_scalars','abi_pair_arg','abi_pair_ret','abi_triple_arg','abi_triple_callback','abi_extended']),('libc_probe',['constants','layouts','runtime'])]:
  prefix='restore' if restore else 'pie'
  label=f'{prefix}-{probe}-{profile}'
  with (w/'logs'/f'{label}-build.log').open('w') as log:
   r=subprocess.run([str(exp/'cargo-alpha'),'build','--offline','--manifest-path',str(exp/'probes'/probe/'Cargo.toml'),*(['--release'] if profile=='release' else [])],env=env,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,label
  for binary in binaries:
   path=exp/'target'/build_id/'alphaev6-unknown-linux-gnu'/profile/binary
   elf=subprocess.check_output([str(w/'install/bin/alphaev6-unknown-linux-gnu-readelf'),'-hd',str(path)],text=True)
   (w/'logs'/f'{prefix}-{binary}-{profile}-elf.log').write_text(elf)
   assert ('EXEC (Executable file)' if restore else 'DYN (Position-Independent Executable file)') in elf
   with (w/'logs'/f'{prefix}-{binary}-{profile}-qemu.log').open('w') as log:
    r=subprocess.run([str(sdk/'bin/alpha-runner'),str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
   result={'binary':binary,'profile':profile,'exit':r.returncode,'dynamic_tags':[x.strip() for x in elf.splitlines() if 'JMPREL' in x or 'PLTRELSZ' in x]}
   results.append(result);print(result,flush=True)
   (w/f'{prefix}-results.json').write_text(json.dumps(results,indent=2)+'\n')

if restore:
 assert all(r['exit']==0 for r in results)
