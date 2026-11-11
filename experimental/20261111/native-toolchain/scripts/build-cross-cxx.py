from pathlib import Path
import re,shlex,subprocess,os
w=Path(__file__).resolve().parent;p=w.parent.parent
old=p/'work/rebase-20261111/build/gcc';build=w/'build/cross-gcc'
if not build.exists():
 subprocess.run(['cp','-a','--reflink=auto',str(old),str(build)],check=True)
 for cache in build.rglob('config.cache'):
  cache.rename(cache.with_name('config.cache.before-relocation'))
args=shlex.split(re.search(r'^ac_cs_config="(.*)"$',(old/'config.status').read_text(),re.M).group(1))
args=[a for a in args if not a.startswith(('--prefix=','--enable-languages=','--disable-libstdcxx','--with-sysroot='))]
args+=['--prefix='+str(w/'cross'),'--enable-languages=c,c++,jit,lto','--with-sysroot='+str(p/'work/sdk/release/alpha-sdk/sysroot')]
env=os.environ.copy();env['TMPDIR']=str(w/'tmp');env['PATH']=str(p/'work/binutils-20261111/install/bin')+os.pathsep+env['PATH']
for label,cmd in [('configure',[str(p/'work/rebase-20261111/src/gcc/configure'),*args]),('build',['make','-j24','all-gcc','all-target-libgcc','all-target-libatomic','all-target-libstdc++-v3']),('install',['make','install-gcc','install-target-libgcc','install-target-libatomic','install-target-libstdc++-v3'])]:
 with (w/'logs'/f'cross-cxx-{label}.log').open('w') as log:subprocess.run(cmd,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
