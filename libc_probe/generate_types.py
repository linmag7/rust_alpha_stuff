"""Generate Alpha-specific libc records from GCC DWARF, not host layouts.
Usage: PYTHONPATH=.../pyelftools python3 generate_types.py linked-header-executable output.rs
The output is checked into libc-alpha; normal Cargo builds do not run this tool.
"""
import sys
from elftools.elf.elffile import ELFFile

records = 'stat stat64 statfs statfs64 statvfs statvfs64 stack_t ipc_perm shmid_ds semid_ds msqid_ds flock flock64 sysinfo timex mcontext_t ucontext_t'.split()
aliases = 'ino_t off_t blkcnt_t shmatt_t msgqnum_t msglen_t fsblkcnt_t fsfilcnt_t rlim_t __syscall_ulong_t clock_t time_t __fsword_t wchar_t nlink_t blksize_t fsblkcnt64_t fsfilcnt64_t suseconds_t'.split()
f = open(sys.argv[1], 'rb')
dwarf = ELFFile(f).get_dwarf_info()
dies = [die for cu in dwarf.iter_CUs() for die in cu.iter_DIEs()]
def name(d):
    a = d.attributes.get('DW_AT_name')
    return a.value.decode() if a else ''
def resolve(d):
    while d.tag in ('DW_TAG_typedef', 'DW_TAG_const_type', 'DW_TAG_volatile_type'):
        d = d.get_DIE_from_attribute('DW_AT_type')
    return d
def lookup(n):
    found = [d for d in dies if name(d) == n and 'DW_AT_declaration' not in d.attributes]
    assert found, n
    return resolve(found[-1])
known = {lookup(n).offset: n for n in records + ['sigset_t', 'fsid_t', 'timeval', 'timespec']}
base = {'char':'c_char', 'signed char':'c_schar','unsigned char':'c_uchar', 'short int':'c_short',
        'short unsigned int':'c_ushort','int':'c_int','unsigned int':'c_uint',
        'long int':'c_long','long unsigned int':'c_ulong','long long int':'c_longlong',
        'long long unsigned int':'c_ulonglong','float':'c_float','double':'c_double'}
def rust_type(d):
    if d.tag == "DW_TAG_typedef" and name(d) in ("size_t", "ssize_t"):
        return "crate::" + name(d)
    d=resolve(d)
    if d.tag=='DW_TAG_base_type': return base[name(d)]
    if d.offset in known: return 'crate::'+known[d.offset]
    if d.tag=='DW_TAG_pointer_type':
        return '*mut '+(rust_type(d.get_DIE_from_attribute('DW_AT_type')) if 'DW_AT_type' in d.attributes else 'c_void')
    if d.tag=='DW_TAG_array_type':
        bounds=[c for c in d.iter_children() if c.tag=='DW_TAG_subrange_type']
        assert len(bounds)==1
        n=bounds[0].attributes.get('DW_AT_upper_bound')
        count=n.value+1 if n else 0
        return f'[{rust_type(d.get_DIE_from_attribute("DW_AT_type"))}; {count}]'
    raise ValueError((d.tag,name(d),d.offset))
def size(d):
    d=resolve(d)
    if 'DW_AT_byte_size' in d.attributes: return d.attributes['DW_AT_byte_size'].value
    if d.tag=='DW_TAG_pointer_type': return 8
    if d.tag=='DW_TAG_array_type':
        n=next(d.iter_children()).attributes.get('DW_AT_upper_bound')
        return size(d.get_DIE_from_attribute('DW_AT_type'))*(n.value+1 if n else 0)
    raise ValueError(d)
lines=['// Generated from the Alpha glibc headers using GCC DWARF. See libc_probe/.', 'use crate::prelude::*;', '']
for n in aliases: lines.append(f'pub type {n} = {rust_type(lookup(n))};')
lines += ['pub type __u64 = c_ulong;', 'pub type __s64 = c_long;', '', 's! {']
for n in records:
    d=lookup(n)
    lines.append(f'    pub struct {n} {{')
    end=0
    for m in d.iter_children():
        if m.tag!='DW_TAG_member' or not name(m): continue
        assert 'DW_AT_bit_size' not in m.attributes, (n,name(m))
        field=name(m); typ=m.get_DIE_from_attribute('DW_AT_type')
        end=m.attributes['DW_AT_data_member_location'].value+size(typ)
        if n in ('stat','stat64') and field in ('st_atim','st_mtim','st_ctim'):
            lines.append(f'        pub {field}e: crate::time_t,')
            lines.append(f'        pub {field}e_nsec: c_long,')
        else: lines.append(f'        pub {field}: {rust_type(typ)},')
    if size(d)>end: lines.append(f'        __tail_padding: [u8; {size(d)-end}],')
    lines.append('    }')
lines += ['    pub struct sigset_t { __val: [c_ulong; 16] }',
          '    pub struct pthread_attr_t { __size: [c_ulong; 7] }',
          '    pub struct siginfo_t { pub si_signo: c_int, pub si_errno: c_int, pub si_code: c_int, _pad: [c_int; 29], _align: [c_ulong; 0] }',
          '    pub struct sigaction { pub sa_sigaction: crate::sighandler_t, pub sa_mask: sigset_t, pub sa_flags: c_int }',
          '}', '']
open(sys.argv[2],'w').write('\n'.join(lines))
