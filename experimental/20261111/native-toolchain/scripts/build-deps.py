from pathlib import Path
import hashlib,subprocess,os,urllib.request,concurrent.futures
w=Path(__file__).resolve().parent;p=w.parent.parent
sums=dict((name,digest) for digest,name in (line.split() for line in (p/'work/rebase-20261111/src/gcc/contrib/prerequisites.sha512').read_text().splitlines()))
archives=['gmp-6.3.0.tar.bz2','mpfr-4.2.2.tar.bz2','mpc-1.3.1.tar.gz']
def fetch(name):
 path=w/'downloads'/name
 if not path.exists():
  with urllib.request.urlopen('https://gcc.gnu.org/pub/gcc/infrastructure/'+name) as r,path.open('wb') as f:
   import shutil;shutil.copyfileobj(r,f)
 assert hashlib.file_digest(path.open('rb'),'sha512').hexdigest()==sums[name],name
 directory=w/'src'/name.split('.tar')[0]
 if not directory.exists():subprocess.run(['tar','-xf',str(path),'-C',str(w/'src')],check=True)
 return directory
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:sources=list(pool.map(fetch,archives))
env=os.environ.copy();env.update(TMPDIR=str(w/'tmp'),CC=str(p/'work/binutils-20261111/alpha-cc'),AR=str(p/'work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ar'),RANLIB=str(p/'work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ranlib'),CFLAGS='-O2 -g0 -fPIC -mcpu=ev6 -mieee -std=gnu17')
for src in sources:
 name=src.name.split('-')[0];build=w/'build'/name;build.mkdir(exist_ok=True)
 args=['--build=x86_64-pc-linux-gnu','--host=alphaev6-unknown-linux-gnu','--prefix='+str(w/'deps'),'--disable-shared','--enable-static','--with-pic']
 if name!='gmp':args+=['--with-gmp='+str(w/'deps')]
 if name=='mpc':args+=['--with-mpfr='+str(w/'deps')]
 for step,cmd in [('configure',[str(src/'configure'),*args]),('build',['make','-j12']),('install',['make','install'])]:
  with (w/'logs'/f'{name}-{step}.log').open('w') as log:subprocess.run(cmd,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'installed',flush=True)
