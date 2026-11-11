#define _GNU_SOURCE
#include <netdb.h>
#include <sys/mount.h>
#include <linux/fs.h>
#include <asm/termbits.h>
#include <sys/ptrace.h>
#include <stdio.h>
_Static_assert(__builtin_types_compatible_p(__typeof__(((struct netent *)0)->n_net), unsigned long), "netent.n_net");
_Static_assert((unsigned long)MS_NOUSER == 0xffffffff80000000UL, "MS_NOUSER");
_Static_assert(BLKIOMIN == 0x20001278UL, "BLKIOMIN");
_Static_assert(BLKIOOPT == 0x20001279UL, "BLKIOOPT");
_Static_assert(BLKPBSZGET == 0x2000127bUL, "BLKPBSZGET");
_Static_assert(BLKSSZGET == 0x20001268UL, "BLKSSZGET");
_Static_assert(PTRACE_DETACH == 17, "PTRACE_DETACH");
_Static_assert(EXTPROC == 0x10000000, "EXTPROC");
_Static_assert(XCASE == 0x4000, "XCASE");
_Static_assert(CIBAUD == 0x1f0000, "CIBAUD");
int main(void) { puts("Native Alpha headers: PASS"); }
