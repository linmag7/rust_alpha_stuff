from pathlib import Path
import json,subprocess,resource
w=Path(__file__).resolve().parent;p=w.parent.parent
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
source=w/'src/ld/testsuite/ld-alpha';out=w/'comparisons';out.mkdir(exist_ok=True)
results={}
for kind,tools in [('old',p/'work/sdk/release/alpha-sdk/gcc/bin'),('new',w/'install/bin')]:
 def run(label,tool,args):
  command=[str(tools/('alphaev6-unknown-linux-gnu-'+tool)),*map(str,args)]
  r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (out/f'{kind}-{label}.log').write_text(r.stdout)
  results[f'{kind}-{label}']={'exit':r.returncode,'command':command}
  return r
 for case in ['got-realloc','got-merge-gprel']:
  objects=[]
  for suffix in ['a','b']:
   obj=out/f'{kind}-{case}-{suffix}.o';objects.append(obj)
   assert run(case+'-'+suffix,'as',[source/f'{case}-{suffix}.s','-o',obj]).returncode==0
  extra=['-shared'] if case=='got-realloc' else []
  run(case,'ld',['-relax','-melf64alpha',*extra,*objects,'-o',out/f'{kind}-{case}.elf'])
 obj=out/f'{kind}-gp.o';binary=out/f'{kind}-gp.elf'
 assert run('gp-as','as',[source/'relax-gpdisp.s','-o',obj]).returncode==0
 assert run('gp-link','ld',['-relax',obj,'-o',binary]).returncode==0
 disasm=run('gp-disasm','objdump',['-dw','--no-show-raw-insn',binary]).stdout
 results[f'{kind}-gp-reload-preserved']='ldah\tgp,' in disasm.split('<_start>:',1)[1].split('<foo>:',1)[0].split('bsr',1)[1]
 # Reproduce two classes of generic test failures with both toolchains.
 elf=w/'src/ld/testsuite/ld-elf'
 run('generic-set-alias','as',['--defsym','NO_SET=1','--defsym','ALIGN=3',elf/'pr17550-3.s','-o',out/f'{kind}-alias.o'])
 obj=out/f'{kind}-retain.o'
 assert run('retain-as','as',['--defsym','NO_SET=1','--defsym','ALIGN=3',elf/'retain3.s','-o',obj]).returncode==0
 run('generic-retain','ld',['-e','_start','--gc-sections',obj,'-o',out/f'{kind}-retain.elf'])
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
for name,result in results.items():
 if not name.endswith(('-a','-b','-as','-disasm','-link')):print(name,result.get('exit') if isinstance(result,dict) else result)
assert results['new-got-realloc']['exit']==0
assert results['new-got-merge-gprel']['exit']==0
assert results['new-gp-reload-preserved']
