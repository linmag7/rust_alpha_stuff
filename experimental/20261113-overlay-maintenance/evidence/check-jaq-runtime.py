#!/usr/bin/env python3
import json, subprocess, tempfile
from pathlib import Path
assert subprocess.check_output(['hostname'], text=True).strip() == 'ansgar'
assert subprocess.check_output(['uname','-m'], text=True).strip() == 'alpha'
b=Path('/root/rust-alpha-overnight-20261111')
assert (b/'jaq-retry1-exit-status').read_text().strip() == '0'
w=Path(tempfile.mkdtemp(prefix='jaq-smoke-',dir=b))
def run(label,query,data,expected):
 p=subprocess.run(['/usr/bin/jaq','-c',query],input=data,text=True,capture_output=True,timeout=60)
 assert p.returncode==0,(label,p.returncode,p.stderr)
 got=[json.loads(x) for x in p.stdout.splitlines()]
 assert got==expected,(label,got,expected)
 print(label+': PASS',flush=True)
 return p.stdout
print(subprocess.check_output(['/usr/bin/jaq','--version'],text=True).strip(),flush=True)
run('filter and map','[.items[] | select(.active) | .name]',json.dumps({'items':[{'name':'alpha','active':True},{'name':'skip','active':False}]}),[['alpha']])
run('reduction','reduce .[] as $x (0; . + $x)','[1,2,3,4]',[10])
run('Unicode round trip','.text','{"text":"Ångström Ελληνικά 日本語 😀"}',['Ångström Ελληνικά 日本語 😀'])
run('integer boundary round trip','.', '9007199254740991\n-9007199254740991\n0\n',[9007199254740991,-9007199254740991,0])
run('floating arithmetic','[. / 2, . * 2]','1.5',[[0.75,3.0]])
run('streamed documents','.n + 1','{"n":1}\n{"n":41}\n',[2,42])
f=w/'input.json';f.write_text('{"file":"alpha"}')
p=subprocess.run(['/usr/bin/jaq','-r','.file',str(f)],text=True,capture_output=True,timeout=60)
assert p.returncode==0 and p.stdout=='alpha\n',(p.returncode,p.stdout,p.stderr)
print('file input: PASS',flush=True)
p=subprocess.run(['/usr/bin/jaq','.'],input='{"broken":',text=True,capture_output=True,timeout=60)
assert p.returncode!=0 and p.stderr,(p.returncode,p.stdout,p.stderr)
(w/'invalid-input.stderr').write_text(p.stderr)
print('invalid input nonzero status: PASS',flush=True)
print('jaq runtime smoke: PASS; fixture '+str(w),flush=True)
print('No full test-suite or exhaustive floating-point/ABI validation.')
