#!/usr/bin/env python3
"""Repack prepared SDK directory; no compilation or host installation."""
import hashlib,subprocess,sys
from pathlib import Path
stage=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.parent.mkdir(parents=True,exist_ok=True)
prefix=stage/'opt/rust-alpha-experimental'
files=sorted(p for p in prefix.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='MANIFEST.sha256')
(prefix/'MANIFEST.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(prefix))+'\n' for p in files))
with out.open('wb') as f:
 a=subprocess.Popen(['tar','--sort=name','--owner=0','--group=0','--numeric-owner','-C',str(stage),'-c','opt/rust-alpha-experimental'],stdout=subprocess.PIPE)
 z=subprocess.run(['xz','-T4','-3'],stdin=a.stdout,stdout=f,check=True);a.stdout.close();assert a.wait()==0
subprocess.run(['xz','-t',str(out)],check=True)
out.with_suffix(out.suffix+'.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+out.name+'\n')
print(out)
