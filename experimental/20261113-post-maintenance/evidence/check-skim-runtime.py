#!/usr/bin/env python3
import os,pty,select,subprocess,tempfile,time,fcntl,struct,termios,signal,shlex
from pathlib import Path
b=Path('/root/rust-alpha-overnight-20261111')
assert subprocess.check_output(['hostname'],text=True).strip()=='ansgar'
assert subprocess.check_output(['uname','-m'],text=True).strip()=='alpha'
assert (b/'skim-retry2-exit-status').read_text().strip()=='0'
lock=(b/'candidates.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.run(['pgrep','-x','emerge'],stdout=subprocess.DEVNULL).returncode==1
w=Path(tempfile.mkdtemp(prefix='skim-smoke-',dir=b));env=dict(os.environ,TERM='xterm-256color',SKIM_DEFAULT_OPTIONS='',SKIM_DEFAULT_COMMAND='')
print(subprocess.check_output(['/usr/bin/sk','--version'],text=True).strip(),flush=True)
def test(label,args,data,expected,status=0):
 p=subprocess.run(['/usr/bin/sk',*args],input=data,text=True,capture_output=True,env=env,timeout=60)
 assert p.returncode==status,(label,p.returncode,p.stderr)
 assert p.stdout.splitlines()==expected,(label,p.stdout,expected)
 print(label+': PASS',flush=True)
test('exact filtering',['--filter','alpha','--exact'],'beta\nalpha\ngamma\n',['alpha'])
test('fuzzy ranking',['--filter','abc'],'a---b---c\nabc\nzzz\n',['abc','a---b---c'])
test('Unicode filtering',['--filter','日本','--exact'],'alpha\n日本語\nÅngström\n',['日本語'])
test('streamed records',['--filter','item','--no-sort'],'item1\nother\nitem2\n',['item1','item2'])
test('no-match status',['--filter','missing'],'alpha\nbeta\n',[],1)
(w/'input').write_text('beta\nalpha\ngamma\n')
def terminal(label,accept):
 out=w/(label+'.selection');capture=bytearray();pid,fd=pty.fork()
 if pid==0:
  os.chdir(w);os.environ.update(env)
  command='exec /usr/bin/sk --query alpha --preview '+shlex.quote('printf "SKIM_PREVIEW_%s\\n" {}')+' < '+shlex.quote(str(w/'input'))+' > '+shlex.quote(str(out))
  os.execl('/bin/bash','bash','-c',command)
 fcntl.ioctl(fd,termios.TIOCSWINSZ,struct.pack('HHHH',30,120,0,0))
 deadline=time.monotonic()+60;sent=False;status=None
 try:
  while time.monotonic()<deadline:
   if select.select([fd],[],[],0.2)[0]:
    try:chunk=os.read(fd,65536)
    except OSError:chunk=b''
    capture.extend(chunk)
    for query,reply in [(b'\x1b[6n',b'\x1b[1;1R'),(b'\x1b[5n',b'\x1b[0n'),(b'\x1b[c',b'\x1b[?1;2c')]:
     if query in chunk:os.write(fd,reply)
   if not sent and b'SKIM_PREVIEW_alpha' in capture:
    os.write(fd,b'\r' if accept else b'\x1b');sent=True
   result,status0=os.waitpid(pid,os.WNOHANG)
   if result:status=os.waitstatus_to_exitcode(status0);break
  assert sent,(label,'preview never appeared')
  assert status==(0 if accept else 130),(label,status)
  assert out.read_text()==('alpha\n' if accept else ''),(label,out.read_text())
  print(label+': PASS',flush=True)
 finally:
  (w/(label+'.ansi')).write_bytes(capture)
  if status is None:
   os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
  os.close(fd)
terminal('terminal preview and selection',True)
terminal('terminal cancellation',False)
print('Skim runtime smoke: PASS; fixture '+str(w),flush=True)
print('CLI only; image/frizbee/listen disabled. No full-suite claim.')
