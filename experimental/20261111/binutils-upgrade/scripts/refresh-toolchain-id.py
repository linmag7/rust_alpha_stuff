from pathlib import Path
import hashlib,json
w=Path(__file__).resolve().parent
stage=w/'rust-build/x86_64-unknown-linux-gnu/stage2'
files=[stage/'bin/rustc',*sorted((stage/'lib').glob('librustc_driver*.so')),w/'backend-target/release/librustc_codegen_gcc.so',w.parent/'rebase-20261111/build/gcc/gcc/libgccjit.so']
binutils=w.parent/'binutils-20261111'
files += [binutils/'install/bin'/('alphaev6-unknown-linux-gnu-'+tool) for tool in ('as','ld','ar','ranlib')]
files += [binutils/'alpha-cc',w/'cargo-alpha']
checks={}
for f in files:
    with f.open('rb') as stream: checks[str(f)]=hashlib.file_digest(stream,'sha256').hexdigest()
identifier=hashlib.sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest()[:16]
(w/'toolchain-id').write_text(identifier+'\n')
(w/'toolchain-sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
print(identifier)
