"""Evaluate libc macro names with the Alpha C compiler and QEMU.
The source modules supply names/types only; their numeric values are never copied.
"""
import json, os, pathlib, re, subprocess
repo=pathlib.Path(__file__).resolve().parent.parent
work=repo.parent/'work'
cross=pathlib.Path(os.environ.get('ALPHA_CROSS','/home/lindholm/x-tools/alphaev6-unknown-linux-gnu'))
root=repo/'libc-alpha/src/unix'
arch=root/'linux_like/linux/arch/generic/mod.rs'
gnu=root/'linux_like/linux/gnu/b64/riscv64/mod.rs'
shared=[root/'mod.rs',root/'linux_like/mod.rs',root/'linux_like/linux/mod.rs',root/'linux_like/linux/gnu/mod.rs']
pattern=re.compile(r'pub const (\w+):\s*([^=;]+?)\s*=')
groups={}
all_names={}
for label,paths in [('arch',[arch]),('gnu',[gnu]),('shared',shared)]:
    group={}
    for p in paths:
        for n,t in pattern.findall(p.read_text()):
            if not re.fullmatch(r'(?:crate::)?\w+',t): continue
            if t in ('pthread_mutex_t','pthread_cond_t','pthread_rwlock_t','sigval','in_addr','in6_addr'): continue
            group[n]=t
    groups[label]=group
    all_names.update(group)
# Common b64 declarations absent from the separate Alpha module.
for n,t in [('__SIZEOF_PTHREAD_RWLOCKATTR_T','usize'),('O_LARGEFILE','c_int')]:
    groups['gnu'][n]=t;all_names[n]=t
headers=(repo/'libc_probe/headers.c').read_text().split('int main')[0]
# ioctl macros refer to the kernel's distinct termios2; mirror its named layout
# from Alpha asm/termbits.h without redefining glibc's termios structure.
headers+='struct termios2 { unsigned int c_iflag,c_oflag,c_cflag,c_lflag; unsigned char c_cc[19],c_line; unsigned int c_ispeed,c_ospeed; };\n'
headers+='struct termio { unsigned short c_iflag,c_oflag,c_cflag,c_lflag; unsigned char c_line,c_cc[8]; };\n'
headers+='#include <linux/serial.h>\n#include <sys/wait.h>\n#include <sys/prctl.h>\n#include <sys/auxv.h>\n#include <sys/utsname.h>\n'
lines=[headers,'int main(void) {']
for n in sorted(all_names):
    # Match the legacy cf*speed@GLIBC_2.0 bindings, not glibc 2.42's new ABI.
    if re.fullmatch(r'B[0-9]+|EXT[AB]', n):
        lines += [f'#ifdef __{n}', f'printf("{n} %llu\\n", (unsigned long long)(__{n}));', f'#elif defined({n})']
    else:
        lines += [f'#ifdef {n}']
    lines += [f'printf("{n} %llu\\n", (unsigned long long)(long long)({n}));', '#endif']
lines+=['return 0;','}']
source=work/'alpha-constants.c';binary=work/'alpha-constants'
source.write_text('\n'.join(lines))
subprocess.run([str(cross/'bin/alphaev6-unknown-linux-gnu-gcc'),'-no-pie',str(source),'-o',str(binary)],check=True)
data=subprocess.check_output(['qemu-alpha-static','-L',str(cross/'alphaev6-unknown-linux-gnu/sysroot'),str(binary)],text=True)
values={n:int(v) for n,v in (line.split() for line in data.splitlines())}
(work/'alpha-c-constants.json').write_text(json.dumps(values,indent=2)+'\n')
(work/'alpha-constant-groups.json').write_text(json.dumps(groups,indent=2)+'\n')
for label,path in [('arch',root/'linux_like/linux/arch/alpha.rs'),('gnu',root/'linux_like/linux/gnu/alpha/constants.rs')]:
    lines=['// Values evaluated using Alpha glibc/kernel headers. See libc_probe/generate_constants.py.', 'use crate::prelude::*;']
    if label=='arch': lines.append('use crate::Ioctl;')
    for n,t in sorted(groups[label].items()):
        if label=='arch' and n in ('SCM_TIMESTAMP','SO_DEBUG'): continue
        if n in values: lines.append(f'pub const {n}: {t} = 0x{values[n]:x}u64 as {t};')
    path.write_text('\n'.join(lines)+'\n')
print(f'Evaluated {len(values)} Alpha C constants')
