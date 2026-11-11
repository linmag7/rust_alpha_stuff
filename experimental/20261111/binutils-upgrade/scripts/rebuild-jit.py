from pathlib import Path
import re,shlex,subprocess,os
w=Path(__file__).resolve().parent
p=w.parent.parent
build=p/'work/rebase-20261111/build/gcc'
old=(w/'baseline/gcc-config.status').read_text()
args=shlex.split(re.search(r'^ac_cs_config="(.*)"$',old,re.M).group(1))
args=[a for a in args if not a.startswith(('--with-as=','--with-ld='))]
args += ['--with-as='+str(w/'install/bin/alphaev6-unknown-linux-gnu-as'),
         '--with-ld='+str(w/'install/bin/alphaev6-unknown-linux-gnu-ld')]
env=os.environ.copy();env['TMPDIR']=str(w/'tmp')
for label,command in [('gcc-reconfigure',[str(p/'work/rebase-20261111/src/gcc/configure'),*args]),('gcc-rebuild',['make','-j24','all-gcc'])]:
 with (w/'logs'/f'{label}.log').open('w') as log:
  subprocess.run(command,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
