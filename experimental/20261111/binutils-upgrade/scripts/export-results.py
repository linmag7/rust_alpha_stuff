from pathlib import Path
from collections import Counter
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;p=w.parent.parent;e=p/'work/experimental-20261111'
a=p/'rust_alpha_stuff/experimental/20261111';out=a/'binutils-upgrade'
out.mkdir(exist_ok=True);(out/'evidence').mkdir(exist_ok=True);(out/'scripts').mkdir(exist_ok=True)
suites={}
for name,path in [('ld',w/'build/ld/ld.sum'),('gas',w/'build/gas/testsuite/gas.sum'),('binutils',w/'build/binutils/binutils.sum')]:
 lines=path.read_text().splitlines()
 suites[name]={'counts':dict(Counter(x.split(':')[0] for x in lines if x.startswith(('PASS:','FAIL:','XPASS:','XFAIL:','UNSUPPORTED:','UNRESOLVED:','ERROR:')))), 'failures':[x for x in lines if x.startswith(('FAIL:','XPASS:','UNRESOLVED:','ERROR:'))]}
 shutil.copy2(path,out/'evidence'/f'{name}.sum')
checks={}
for tool in ['as','ld','ar','ranlib','nm','objdump','objcopy','readelf','strip']:
 path=w/'install/bin'/('alphaev6-unknown-linux-gnu-'+tool)
 checks[str(path)]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
validation=json.loads((e/'validation.json').read_text());assert len(validation['results'])==34 and all(x['exit']==0 for x in validation['results'])
restore=json.loads((w/'restore-results.json').read_text());assert len(restore)==18 and all(x['exit']==0 for x in restore)
assert not subprocess.check_output(['git','-C',str(w/'src'),'status','--porcelain']).strip()
metadata={
 'date':'2026-11-11','scope':'private experimental Alpha Rust cross toolchain',
 'binutils':{'version':'2.47.50.20261111','origin':'https://sourceware.org/git/binutils-gdb.git','revision':(w/'revision').read_text().strip(),'source_modified':False,'target':'alphaev6-unknown-linux-gnu','install':str(w/'install'),'sha256':checks},
 'integration':{'cargo_wrapper':str(e/'cargo-alpha'),'c_wrapper':str(w/'alpha-cc'),'toolchain_id':(e/'toolchain-id').read_text().strip(),'gcc_jit_reconfigured':True,'gcc_source_patch_unchanged':True,'packaged_sdk_replaced':False,'glibc':'2.41','compiler_and_jit_tool_selection_verified':'strace execve evidence'},
 'upstream_test_suites':suites,
 'ld_baseline_comparison':json.loads((w/'comparisons/ld-baseline.json').read_text()),
 'gas_baseline_comparison':json.loads((w/'comparisons/gas-baseline.json').read_text()),
 'matt_turner_regressions':json.loads((w/'comparisons/results.json').read_text()),
 'rust_validation':{'commands':34,'failures':0,'debug_release_std_abi_libc_inline_asm':'pass','constants_per_profile':2367,'cargo_compat_and_tests':'pass'},
 'small_pie':{'results':json.loads((w/'pie-results.json').read_text()),'workaround':'Keep existing -no-pie in ABI and libc probes; all 18 normal builds restored and pass','remaining_failure':'Nine release PIEs have DT_JMPREL=0 and loader exit 127; nine debug PIEs pass'},
 'native_es40':{'status':'not tested','ssh_connect':(w/'logs/es40-ssh-connect.log').read_text().strip()},
 'limitations':['Not a clean upstream suite: 16 ld FAIL, 1 ld XPASS, 5 gas FAIL; all reproduced with old tools.','Old-tools comparison was stopped after matching all current failures; see baseline_stop_reason.','No IFUNC runtime claim: retained glibc 2.41 sysroot.','Existing panic=abort, no LTO/incremental, unreliable f128 limitations remain.'],
 'work_directory':str(w),
}
(w/'status.json').write_text(json.dumps(metadata,indent=2)+'\n')
shutil.copy2(w/'status.json',out/'status.json')
for path in list((w/'logs').glob('*.log'))+list((w/'logs').glob('*.trace')):
 if path.name in ['build.log','install.log','gcc-rebuild.log','check-baseline.log']:continue
 shutil.copy2(path,out/'evidence'/path.name)
for path in (w/'comparisons').glob('*.json'):shutil.copy2(path,out/'evidence'/path.name)
for path in [w/'pie-results.json',w/'restore-results.json',e/'validation.json',e/'toolchain-sha256.json']:
 shutil.copy2(path,out/'evidence'/path.name)
for name in ['build.sh','alpha-cc','rebuild-jit.py','check.sh','check-baseline.sh','compare-linkers.py','validate-pie.py','export-results.py','jit-tools.c','hello.c','revision']:
 shutil.copy2(w/name,out/'scripts'/name)
for name in ['cargo-alpha','refresh-toolchain-id.py']:
 shutil.copy2(e/name,out/'scripts'/name)
state=json.loads((e/'status.json').read_text())
state['components']['binutils']=metadata['binutils']['revision']
state['binutils_upgrade']={'version':metadata['binutils']['version'],'status':str(out/'status.json'),'source_modified':False,'rust_validation':'34 commands passed; 18 restored probes passed','known_issue':'small release PIE DT_JMPREL=0; probe workaround retained'}
state['toolchain_id']=metadata['integration']['toolchain_id']
state['native_testing']['last_check']='SSH TCP connection timed out'
state['native_testing']['log']=str(w/'logs/es40-ssh-connect.log')
(e/'status.json').write_text(json.dumps(state,indent=2)+'\n')
for name in ['status.json','validation.json','toolchain-sha256.json']:shutil.copy2(e/name,a/name)
for path in (e/'logs').glob('*.log'):
 if path.name in {Path(x['log']).name for x in validation['results']}:shutil.copy2(path,a/'evidence'/path.name)
print('Saved',out)
