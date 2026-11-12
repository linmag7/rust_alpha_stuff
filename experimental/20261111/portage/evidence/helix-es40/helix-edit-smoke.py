import os,pty,fcntl,termios,struct,select,time,signal
from pathlib import Path
assert os.uname().nodename=='ansgar' and os.uname().machine=='alpha'
base=Path('/root/rust-alpha-overnight-20261111/helix-smoke-work');base.mkdir(exist_ok=True)
target=base/'edit.txt';assert not target.exists()
(base/'config.toml').write_text('[editor]\nauto-save = false\n')
pid,fd=pty.fork()
if pid==0:
 os.environ['TERM']='xterm-256color'
 os.environ['XDG_CONFIG_HOME']=str(base/'config')
 os.execv('/usr/bin/hx',['hx','--config',str(base/'config.toml'),'--log',str(base/'helix.log'),str(target)])
fcntl.ioctl(fd,termios.TIOCSWINSZ,struct.pack('HHHH',24,80,0,0))
output=bytearray();start=time.monotonic();stage=0;status=None
while time.monotonic()-start<30:
 if select.select([fd],[],[],0.1)[0]:
  try: data=os.read(fd,65536)
  except OSError: break
  output.extend(data)
  if b'\x1b[6n' in data: os.write(fd,b'\x1b[1;1R')
  if b'\x1b[c' in data: os.write(fd,b'\x1b[?1;2c')
 elapsed=time.monotonic()-start
 if stage==0 and elapsed>3: os.write(fd,b'iAlpha Helix smoke\x1b');stage=1
 if stage==1 and elapsed>5: os.write(fd,b':wq\r');stage=2
 done,st=os.waitpid(pid,os.WNOHANG)
 if done: status=st;break
if status is None:
 done,status=os.waitpid(pid,os.WNOHANG)
 if not done:
  os.kill(pid,signal.SIGTERM);os.waitpid(pid,0);raise RuntimeError('Helix smoke timed out')
(base/'terminal.log').write_bytes(output)
assert os.waitstatus_to_exitcode(status)==0, status
assert target.read_text()=='Alpha Helix smoke\n',repr(target.read_text())
print('Helix PTY insert/save/quit PASS')
