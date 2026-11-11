import pathlib,re
repo=pathlib.Path(__file__).resolve().parent.parent
source=(repo/'libc-alpha/src/unix/linux_like/linux/gnu/alpha/types.rs').read_text() + (repo/'libc-alpha/src/unix/linux_like/linux/gnu/alpha/mod.rs').read_text()
records={n:re.findall(r'pub (\w+):',body) for n,body in re.findall(r'pub struct (\w+)\s*\{(.*?)\}',source,re.S)}
records.pop('termios2') # kernel-only layout; not part of this glibc layout suite
for n in ['pthread_mutex_t','pthread_cond_t','pthread_rwlock_t','pthread_mutexattr_t','pthread_condattr_t','pthread_rwlockattr_t','pthread_barrier_t','pthread_barrierattr_t','timespec','timeval','pollfd','iovec','msghdr','cmsghdr','sockaddr','sockaddr_in','sockaddr_in6','sockaddr_storage','epoll_event','dirent','dirent64','rlimit','rusage']:
    records.setdefault(n,[])
typedefs=set('stack_t sigset_t siginfo_t mcontext_t ucontext_t pthread_attr_t pthread_mutex_t pthread_cond_t pthread_rwlock_t pthread_mutexattr_t pthread_condattr_t pthread_rwlockattr_t pthread_barrier_t pthread_barrierattr_t'.split())
checks=[]
for n,fields in records.items():
    ctype=n if n in typedefs else 'struct '+n
    checks.append((f'sizeof({ctype})',f'core::mem::size_of::<libc::{n}>()',f'{n} size'))
    checks.append((f'_Alignof({ctype})',f'core::mem::align_of::<libc::{n}>()',f'{n} alignment'))
    for field in fields:
        if field.startswith('_'): continue
        cfield=field
        if n in ('stat','stat64') and field.endswith('_nsec'): cfield=field[:-6]+'.tv_nsec'
        checks.append((f'offsetof({ctype},{cfield})',f'core::mem::offset_of!(libc::{n},{field})',f'{n}.{field} offset'))
aliases=re.findall(r'pub type (\w+) =',source)
for n in aliases:
    if n in ('__u64','__s64'):continue
    checks.extend([(f'sizeof({n})',f'core::mem::size_of::<libc::{n}>()',f'{n} size'),
                   (f'(({n})-1 < ({n})0)',f'(((-1i64 as libc::{n}) as i128) < 0) as usize',f'{n} signedness')])
headers=(repo/'libc_probe/headers.c').read_text().split('int main')[0]+'#include <dirent.h>\n'
c=[headers,'static const uint64_t values[] = {']+[f'    {expr}, /* {label} */' for expr,_,label in checks]+['};','uint64_t c_layout_value(unsigned i) { return values[i]; }']
(repo/'libc_probe/layouts.c').write_text('\n'.join(c)+'\n')
rs=['#![no_std]','#![no_main]','use alpha_libc_probe::*;', 'unsafe extern "C" { fn c_layout_value(i:u32)->u64; }', '#[unsafe(no_mangle)]','pub extern "C" fn _start()->! {', '    let mut failed=0;']
for i,(_,expr,label) in enumerate(checks):
    rs += [f'    if ({expr}) as u64 != unsafe {{c_layout_value({i})}} {{ print(b"FAIL {label}\\n"); failed+=1; }}']
rs+=['    if failed==0 { print(b"libc layouts: PASS\\n"); }', '    finish(if failed==0 {0} else {1})','}']
(repo/'libc_probe/src/bin/layouts.rs').write_text('\n'.join(rs)+'\n')
print(f'Generated {len(checks)} independent layout checks')
