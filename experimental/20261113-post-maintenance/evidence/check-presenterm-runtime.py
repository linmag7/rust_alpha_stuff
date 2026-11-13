#!/usr/bin/env python3
import os,pty,select,subprocess,tempfile,time,fcntl,struct,termios,signal,re
from pathlib import Path
b=Path('/root/rust-alpha-overnight-20261111')
assert subprocess.check_output(['hostname'],text=True).strip()=='ansgar'
assert subprocess.check_output(['uname','-m'],text=True).strip()=='alpha'
assert (b/'presenterm-retry1-exit-status').read_text().strip()=='0'
lock=(b/'candidates.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.run(['pgrep','-x','emerge'],stdout=subprocess.DEVNULL).returncode==1
w=Path(tempfile.mkdtemp(prefix='presenterm-smoke-',dir=b));(w/'config').mkdir();(w/'cache').mkdir()
(w/'slides.md').write_text('AlphaFirstSlide\n---\n\nÅngström 日本語\n\n* Native Markdown rendering\n\n<!-- end_slide -->\n\nAlphaSecondSlide\n---\n\n```rust\nfn alpha_probe() {\n    let answer = 42;\n}\n```\n')
print(subprocess.check_output(['/usr/bin/presenterm','--version'],text=True).strip(),flush=True)
pid,fd=pty.fork()
if pid==0:
 os.chdir(w);os.environ.update(TERM='xterm-256color',XDG_CONFIG_HOME=str(w/'config'),XDG_CACHE_HOME=str(w/'cache'))
 os.execl('/usr/bin/presenterm','presenterm','--image-protocol','ascii-blocks',str(w/'slides.md'))
fcntl.ioctl(fd,termios.TIOCSWINSZ,struct.pack('HHHH',40,120,0,0));capture=bytearray();stagebuf=bytearray();stage=0;status=None;deadline=time.monotonic()+90
try:
 while time.monotonic()<deadline:
  if select.select([fd],[],[],0.2)[0]:
   try:chunk=os.read(fd,65536)
   except OSError:chunk=b''
   capture.extend(chunk);stagebuf.extend(chunk)
   for q,r in [(b'\x1b[6n',b'\x1b[1;1R'),(b'\x1b[5n',b'\x1b[0n'),(b'\x1b[c',b'\x1b[?1;2c')]:
    if q in chunk:os.write(fd,r)
  text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',stagebuf.decode(errors='replace'))
  if stage==0 and 'AlphaFirstSlide' in text and '日本語' in text:
   print('Markdown and Unicode rendering: PASS',flush=True);stage=1;stagebuf.clear();os.write(fd,b'\x1b[C')
  elif stage==1 and 'AlphaSecondSlide' in text and 'alpha_probe' in text and '42' in text:
   assert b'\x1b[38;' in stagebuf or b'\x1b[3' in stagebuf,'no colored syntax output'
   print('Next slide and highlighted Rust snippet: PASS',flush=True);stage=2;stagebuf.clear();os.write(fd,b'\x1b[D')
  elif stage==2 and 'AlphaFirstSlide' in text:
   print('Previous slide: PASS',flush=True);stage=3;os.write(fd,b'q')
  result,raw=os.waitpid(pid,os.WNOHANG)
  if result:status=os.waitstatus_to_exitcode(raw);break
 assert stage==3 and status==0,(stage,status)
 print('Clean quit: PASS',flush=True)
 print('Presenterm runtime smoke: PASS; fixture '+str(w),flush=True)
 print('No snippet execution, external renderers, image protocol or full-suite validation.')
finally:
 (w/'terminal.ansi').write_bytes(capture)
 if status is None:os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
 os.close(fd)
