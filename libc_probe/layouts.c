#define _GNU_SOURCE 1
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/epoll.h>
#include <sys/sysinfo.h>
#include <sys/timex.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/msg.h>
#include <sys/syscall.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <sys/inotify.h>
#include <sys/ptrace.h>
#include <pthread.h>
#include <signal.h>
#include <ucontext.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sched.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

static const uint64_t values[] = {
    sizeof(struct stat), /* stat size */
    _Alignof(struct stat), /* stat alignment */
    offsetof(struct stat,st_dev), /* stat.st_dev offset */
    offsetof(struct stat,st_ino), /* stat.st_ino offset */
    offsetof(struct stat,st_rdev), /* stat.st_rdev offset */
    offsetof(struct stat,st_size), /* stat.st_size offset */
    offsetof(struct stat,st_blocks), /* stat.st_blocks offset */
    offsetof(struct stat,st_mode), /* stat.st_mode offset */
    offsetof(struct stat,st_uid), /* stat.st_uid offset */
    offsetof(struct stat,st_gid), /* stat.st_gid offset */
    offsetof(struct stat,st_blksize), /* stat.st_blksize offset */
    offsetof(struct stat,st_nlink), /* stat.st_nlink offset */
    offsetof(struct stat,st_atime), /* stat.st_atime offset */
    offsetof(struct stat,st_atim.tv_nsec), /* stat.st_atime_nsec offset */
    offsetof(struct stat,st_mtime), /* stat.st_mtime offset */
    offsetof(struct stat,st_mtim.tv_nsec), /* stat.st_mtime_nsec offset */
    offsetof(struct stat,st_ctime), /* stat.st_ctime offset */
    offsetof(struct stat,st_ctim.tv_nsec), /* stat.st_ctime_nsec offset */
    sizeof(struct stat64), /* stat64 size */
    _Alignof(struct stat64), /* stat64 alignment */
    offsetof(struct stat64,st_dev), /* stat64.st_dev offset */
    offsetof(struct stat64,st_ino), /* stat64.st_ino offset */
    offsetof(struct stat64,st_rdev), /* stat64.st_rdev offset */
    offsetof(struct stat64,st_size), /* stat64.st_size offset */
    offsetof(struct stat64,st_blocks), /* stat64.st_blocks offset */
    offsetof(struct stat64,st_mode), /* stat64.st_mode offset */
    offsetof(struct stat64,st_uid), /* stat64.st_uid offset */
    offsetof(struct stat64,st_gid), /* stat64.st_gid offset */
    offsetof(struct stat64,st_blksize), /* stat64.st_blksize offset */
    offsetof(struct stat64,st_nlink), /* stat64.st_nlink offset */
    offsetof(struct stat64,st_atime), /* stat64.st_atime offset */
    offsetof(struct stat64,st_atim.tv_nsec), /* stat64.st_atime_nsec offset */
    offsetof(struct stat64,st_mtime), /* stat64.st_mtime offset */
    offsetof(struct stat64,st_mtim.tv_nsec), /* stat64.st_mtime_nsec offset */
    offsetof(struct stat64,st_ctime), /* stat64.st_ctime offset */
    offsetof(struct stat64,st_ctim.tv_nsec), /* stat64.st_ctime_nsec offset */
    sizeof(struct statfs), /* statfs size */
    _Alignof(struct statfs), /* statfs alignment */
    offsetof(struct statfs,f_type), /* statfs.f_type offset */
    offsetof(struct statfs,f_bsize), /* statfs.f_bsize offset */
    offsetof(struct statfs,f_blocks), /* statfs.f_blocks offset */
    offsetof(struct statfs,f_bfree), /* statfs.f_bfree offset */
    offsetof(struct statfs,f_bavail), /* statfs.f_bavail offset */
    offsetof(struct statfs,f_files), /* statfs.f_files offset */
    offsetof(struct statfs,f_ffree), /* statfs.f_ffree offset */
    offsetof(struct statfs,f_fsid), /* statfs.f_fsid offset */
    offsetof(struct statfs,f_namelen), /* statfs.f_namelen offset */
    offsetof(struct statfs,f_frsize), /* statfs.f_frsize offset */
    offsetof(struct statfs,f_flags), /* statfs.f_flags offset */
    offsetof(struct statfs,f_spare), /* statfs.f_spare offset */
    sizeof(struct statfs64), /* statfs64 size */
    _Alignof(struct statfs64), /* statfs64 alignment */
    offsetof(struct statfs64,f_type), /* statfs64.f_type offset */
    offsetof(struct statfs64,f_bsize), /* statfs64.f_bsize offset */
    offsetof(struct statfs64,f_blocks), /* statfs64.f_blocks offset */
    offsetof(struct statfs64,f_bfree), /* statfs64.f_bfree offset */
    offsetof(struct statfs64,f_bavail), /* statfs64.f_bavail offset */
    offsetof(struct statfs64,f_files), /* statfs64.f_files offset */
    offsetof(struct statfs64,f_ffree), /* statfs64.f_ffree offset */
    offsetof(struct statfs64,f_fsid), /* statfs64.f_fsid offset */
    offsetof(struct statfs64,f_namelen), /* statfs64.f_namelen offset */
    offsetof(struct statfs64,f_frsize), /* statfs64.f_frsize offset */
    offsetof(struct statfs64,f_flags), /* statfs64.f_flags offset */
    offsetof(struct statfs64,f_spare), /* statfs64.f_spare offset */
    sizeof(struct statvfs), /* statvfs size */
    _Alignof(struct statvfs), /* statvfs alignment */
    offsetof(struct statvfs,f_bsize), /* statvfs.f_bsize offset */
    offsetof(struct statvfs,f_frsize), /* statvfs.f_frsize offset */
    offsetof(struct statvfs,f_blocks), /* statvfs.f_blocks offset */
    offsetof(struct statvfs,f_bfree), /* statvfs.f_bfree offset */
    offsetof(struct statvfs,f_bavail), /* statvfs.f_bavail offset */
    offsetof(struct statvfs,f_files), /* statvfs.f_files offset */
    offsetof(struct statvfs,f_ffree), /* statvfs.f_ffree offset */
    offsetof(struct statvfs,f_favail), /* statvfs.f_favail offset */
    offsetof(struct statvfs,f_fsid), /* statvfs.f_fsid offset */
    offsetof(struct statvfs,f_flag), /* statvfs.f_flag offset */
    offsetof(struct statvfs,f_namemax), /* statvfs.f_namemax offset */
    offsetof(struct statvfs,f_type), /* statvfs.f_type offset */
    sizeof(struct statvfs64), /* statvfs64 size */
    _Alignof(struct statvfs64), /* statvfs64 alignment */
    offsetof(struct statvfs64,f_bsize), /* statvfs64.f_bsize offset */
    offsetof(struct statvfs64,f_frsize), /* statvfs64.f_frsize offset */
    offsetof(struct statvfs64,f_blocks), /* statvfs64.f_blocks offset */
    offsetof(struct statvfs64,f_bfree), /* statvfs64.f_bfree offset */
    offsetof(struct statvfs64,f_bavail), /* statvfs64.f_bavail offset */
    offsetof(struct statvfs64,f_files), /* statvfs64.f_files offset */
    offsetof(struct statvfs64,f_ffree), /* statvfs64.f_ffree offset */
    offsetof(struct statvfs64,f_favail), /* statvfs64.f_favail offset */
    offsetof(struct statvfs64,f_fsid), /* statvfs64.f_fsid offset */
    offsetof(struct statvfs64,f_flag), /* statvfs64.f_flag offset */
    offsetof(struct statvfs64,f_namemax), /* statvfs64.f_namemax offset */
    offsetof(struct statvfs64,f_type), /* statvfs64.f_type offset */
    sizeof(stack_t), /* stack_t size */
    _Alignof(stack_t), /* stack_t alignment */
    offsetof(stack_t,ss_sp), /* stack_t.ss_sp offset */
    offsetof(stack_t,ss_flags), /* stack_t.ss_flags offset */
    offsetof(stack_t,ss_size), /* stack_t.ss_size offset */
    sizeof(struct ipc_perm), /* ipc_perm size */
    _Alignof(struct ipc_perm), /* ipc_perm alignment */
    offsetof(struct ipc_perm,uid), /* ipc_perm.uid offset */
    offsetof(struct ipc_perm,gid), /* ipc_perm.gid offset */
    offsetof(struct ipc_perm,cuid), /* ipc_perm.cuid offset */
    offsetof(struct ipc_perm,cgid), /* ipc_perm.cgid offset */
    offsetof(struct ipc_perm,mode), /* ipc_perm.mode offset */
    sizeof(struct shmid_ds), /* shmid_ds size */
    _Alignof(struct shmid_ds), /* shmid_ds alignment */
    offsetof(struct shmid_ds,shm_perm), /* shmid_ds.shm_perm offset */
    offsetof(struct shmid_ds,shm_segsz), /* shmid_ds.shm_segsz offset */
    offsetof(struct shmid_ds,shm_atime), /* shmid_ds.shm_atime offset */
    offsetof(struct shmid_ds,shm_dtime), /* shmid_ds.shm_dtime offset */
    offsetof(struct shmid_ds,shm_ctime), /* shmid_ds.shm_ctime offset */
    offsetof(struct shmid_ds,shm_cpid), /* shmid_ds.shm_cpid offset */
    offsetof(struct shmid_ds,shm_lpid), /* shmid_ds.shm_lpid offset */
    offsetof(struct shmid_ds,shm_nattch), /* shmid_ds.shm_nattch offset */
    sizeof(struct semid_ds), /* semid_ds size */
    _Alignof(struct semid_ds), /* semid_ds alignment */
    offsetof(struct semid_ds,sem_perm), /* semid_ds.sem_perm offset */
    offsetof(struct semid_ds,sem_otime), /* semid_ds.sem_otime offset */
    offsetof(struct semid_ds,sem_ctime), /* semid_ds.sem_ctime offset */
    offsetof(struct semid_ds,sem_nsems), /* semid_ds.sem_nsems offset */
    sizeof(struct msqid_ds), /* msqid_ds size */
    _Alignof(struct msqid_ds), /* msqid_ds alignment */
    offsetof(struct msqid_ds,msg_perm), /* msqid_ds.msg_perm offset */
    offsetof(struct msqid_ds,msg_stime), /* msqid_ds.msg_stime offset */
    offsetof(struct msqid_ds,msg_rtime), /* msqid_ds.msg_rtime offset */
    offsetof(struct msqid_ds,msg_ctime), /* msqid_ds.msg_ctime offset */
    offsetof(struct msqid_ds,msg_qnum), /* msqid_ds.msg_qnum offset */
    offsetof(struct msqid_ds,msg_qbytes), /* msqid_ds.msg_qbytes offset */
    offsetof(struct msqid_ds,msg_lspid), /* msqid_ds.msg_lspid offset */
    offsetof(struct msqid_ds,msg_lrpid), /* msqid_ds.msg_lrpid offset */
    sizeof(struct flock), /* flock size */
    _Alignof(struct flock), /* flock alignment */
    offsetof(struct flock,l_type), /* flock.l_type offset */
    offsetof(struct flock,l_whence), /* flock.l_whence offset */
    offsetof(struct flock,l_start), /* flock.l_start offset */
    offsetof(struct flock,l_len), /* flock.l_len offset */
    offsetof(struct flock,l_pid), /* flock.l_pid offset */
    sizeof(struct flock64), /* flock64 size */
    _Alignof(struct flock64), /* flock64 alignment */
    offsetof(struct flock64,l_type), /* flock64.l_type offset */
    offsetof(struct flock64,l_whence), /* flock64.l_whence offset */
    offsetof(struct flock64,l_start), /* flock64.l_start offset */
    offsetof(struct flock64,l_len), /* flock64.l_len offset */
    offsetof(struct flock64,l_pid), /* flock64.l_pid offset */
    sizeof(struct sysinfo), /* sysinfo size */
    _Alignof(struct sysinfo), /* sysinfo alignment */
    offsetof(struct sysinfo,uptime), /* sysinfo.uptime offset */
    offsetof(struct sysinfo,loads), /* sysinfo.loads offset */
    offsetof(struct sysinfo,totalram), /* sysinfo.totalram offset */
    offsetof(struct sysinfo,freeram), /* sysinfo.freeram offset */
    offsetof(struct sysinfo,sharedram), /* sysinfo.sharedram offset */
    offsetof(struct sysinfo,bufferram), /* sysinfo.bufferram offset */
    offsetof(struct sysinfo,totalswap), /* sysinfo.totalswap offset */
    offsetof(struct sysinfo,freeswap), /* sysinfo.freeswap offset */
    offsetof(struct sysinfo,procs), /* sysinfo.procs offset */
    offsetof(struct sysinfo,pad), /* sysinfo.pad offset */
    offsetof(struct sysinfo,totalhigh), /* sysinfo.totalhigh offset */
    offsetof(struct sysinfo,freehigh), /* sysinfo.freehigh offset */
    offsetof(struct sysinfo,mem_unit), /* sysinfo.mem_unit offset */
    sizeof(struct timex), /* timex size */
    _Alignof(struct timex), /* timex alignment */
    offsetof(struct timex,modes), /* timex.modes offset */
    offsetof(struct timex,offset), /* timex.offset offset */
    offsetof(struct timex,freq), /* timex.freq offset */
    offsetof(struct timex,maxerror), /* timex.maxerror offset */
    offsetof(struct timex,esterror), /* timex.esterror offset */
    offsetof(struct timex,status), /* timex.status offset */
    offsetof(struct timex,constant), /* timex.constant offset */
    offsetof(struct timex,precision), /* timex.precision offset */
    offsetof(struct timex,tolerance), /* timex.tolerance offset */
    offsetof(struct timex,time), /* timex.time offset */
    offsetof(struct timex,tick), /* timex.tick offset */
    offsetof(struct timex,ppsfreq), /* timex.ppsfreq offset */
    offsetof(struct timex,jitter), /* timex.jitter offset */
    offsetof(struct timex,shift), /* timex.shift offset */
    offsetof(struct timex,stabil), /* timex.stabil offset */
    offsetof(struct timex,jitcnt), /* timex.jitcnt offset */
    offsetof(struct timex,calcnt), /* timex.calcnt offset */
    offsetof(struct timex,errcnt), /* timex.errcnt offset */
    offsetof(struct timex,stbcnt), /* timex.stbcnt offset */
    offsetof(struct timex,tai), /* timex.tai offset */
    sizeof(mcontext_t), /* mcontext_t size */
    _Alignof(mcontext_t), /* mcontext_t alignment */
    offsetof(mcontext_t,sc_onstack), /* mcontext_t.sc_onstack offset */
    offsetof(mcontext_t,sc_mask), /* mcontext_t.sc_mask offset */
    offsetof(mcontext_t,sc_pc), /* mcontext_t.sc_pc offset */
    offsetof(mcontext_t,sc_ps), /* mcontext_t.sc_ps offset */
    offsetof(mcontext_t,sc_regs), /* mcontext_t.sc_regs offset */
    offsetof(mcontext_t,sc_ownedfp), /* mcontext_t.sc_ownedfp offset */
    offsetof(mcontext_t,sc_fpregs), /* mcontext_t.sc_fpregs offset */
    offsetof(mcontext_t,sc_fpcr), /* mcontext_t.sc_fpcr offset */
    offsetof(mcontext_t,sc_fp_control), /* mcontext_t.sc_fp_control offset */
    offsetof(mcontext_t,sc_ssize), /* mcontext_t.sc_ssize offset */
    offsetof(mcontext_t,sc_sbase), /* mcontext_t.sc_sbase offset */
    offsetof(mcontext_t,sc_traparg_a0), /* mcontext_t.sc_traparg_a0 offset */
    offsetof(mcontext_t,sc_traparg_a1), /* mcontext_t.sc_traparg_a1 offset */
    offsetof(mcontext_t,sc_traparg_a2), /* mcontext_t.sc_traparg_a2 offset */
    offsetof(mcontext_t,sc_fp_trap_pc), /* mcontext_t.sc_fp_trap_pc offset */
    offsetof(mcontext_t,sc_fp_trigger_sum), /* mcontext_t.sc_fp_trigger_sum offset */
    offsetof(mcontext_t,sc_fp_trigger_inst), /* mcontext_t.sc_fp_trigger_inst offset */
    sizeof(ucontext_t), /* ucontext_t size */
    _Alignof(ucontext_t), /* ucontext_t alignment */
    offsetof(ucontext_t,uc_flags), /* ucontext_t.uc_flags offset */
    offsetof(ucontext_t,uc_link), /* ucontext_t.uc_link offset */
    offsetof(ucontext_t,uc_stack), /* ucontext_t.uc_stack offset */
    offsetof(ucontext_t,uc_mcontext), /* ucontext_t.uc_mcontext offset */
    offsetof(ucontext_t,uc_sigmask), /* ucontext_t.uc_sigmask offset */
    sizeof(sigset_t), /* sigset_t size */
    _Alignof(sigset_t), /* sigset_t alignment */
    sizeof(pthread_attr_t), /* pthread_attr_t size */
    _Alignof(pthread_attr_t), /* pthread_attr_t alignment */
    sizeof(siginfo_t), /* siginfo_t size */
    _Alignof(siginfo_t), /* siginfo_t alignment */
    offsetof(siginfo_t,si_signo), /* siginfo_t.si_signo offset */
    offsetof(siginfo_t,si_errno), /* siginfo_t.si_errno offset */
    offsetof(siginfo_t,si_code), /* siginfo_t.si_code offset */
    sizeof(struct sigaction), /* sigaction size */
    _Alignof(struct sigaction), /* sigaction alignment */
    offsetof(struct sigaction,sa_sigaction), /* sigaction.sa_sigaction offset */
    offsetof(struct sigaction,sa_mask), /* sigaction.sa_mask offset */
    offsetof(struct sigaction,sa_flags), /* sigaction.sa_flags offset */
    sizeof(struct termios), /* termios size */
    _Alignof(struct termios), /* termios alignment */
    offsetof(struct termios,c_iflag), /* termios.c_iflag offset */
    offsetof(struct termios,c_oflag), /* termios.c_oflag offset */
    offsetof(struct termios,c_cflag), /* termios.c_cflag offset */
    offsetof(struct termios,c_lflag), /* termios.c_lflag offset */
    offsetof(struct termios,c_cc), /* termios.c_cc offset */
    offsetof(struct termios,c_line), /* termios.c_line offset */
    offsetof(struct termios,c_ispeed), /* termios.c_ispeed offset */
    offsetof(struct termios,c_ospeed), /* termios.c_ospeed offset */
    sizeof(pthread_mutex_t), /* pthread_mutex_t size */
    _Alignof(pthread_mutex_t), /* pthread_mutex_t alignment */
    sizeof(pthread_cond_t), /* pthread_cond_t size */
    _Alignof(pthread_cond_t), /* pthread_cond_t alignment */
    sizeof(pthread_rwlock_t), /* pthread_rwlock_t size */
    _Alignof(pthread_rwlock_t), /* pthread_rwlock_t alignment */
    sizeof(pthread_mutexattr_t), /* pthread_mutexattr_t size */
    _Alignof(pthread_mutexattr_t), /* pthread_mutexattr_t alignment */
    sizeof(pthread_condattr_t), /* pthread_condattr_t size */
    _Alignof(pthread_condattr_t), /* pthread_condattr_t alignment */
    sizeof(pthread_rwlockattr_t), /* pthread_rwlockattr_t size */
    _Alignof(pthread_rwlockattr_t), /* pthread_rwlockattr_t alignment */
    sizeof(pthread_barrier_t), /* pthread_barrier_t size */
    _Alignof(pthread_barrier_t), /* pthread_barrier_t alignment */
    sizeof(pthread_barrierattr_t), /* pthread_barrierattr_t size */
    _Alignof(pthread_barrierattr_t), /* pthread_barrierattr_t alignment */
    sizeof(struct timespec), /* timespec size */
    _Alignof(struct timespec), /* timespec alignment */
    sizeof(struct timeval), /* timeval size */
    _Alignof(struct timeval), /* timeval alignment */
    sizeof(struct pollfd), /* pollfd size */
    _Alignof(struct pollfd), /* pollfd alignment */
    sizeof(struct iovec), /* iovec size */
    _Alignof(struct iovec), /* iovec alignment */
    sizeof(struct msghdr), /* msghdr size */
    _Alignof(struct msghdr), /* msghdr alignment */
    sizeof(struct cmsghdr), /* cmsghdr size */
    _Alignof(struct cmsghdr), /* cmsghdr alignment */
    sizeof(struct sockaddr), /* sockaddr size */
    _Alignof(struct sockaddr), /* sockaddr alignment */
    sizeof(struct sockaddr_in), /* sockaddr_in size */
    _Alignof(struct sockaddr_in), /* sockaddr_in alignment */
    sizeof(struct sockaddr_in6), /* sockaddr_in6 size */
    _Alignof(struct sockaddr_in6), /* sockaddr_in6 alignment */
    sizeof(struct sockaddr_storage), /* sockaddr_storage size */
    _Alignof(struct sockaddr_storage), /* sockaddr_storage alignment */
    sizeof(struct epoll_event), /* epoll_event size */
    _Alignof(struct epoll_event), /* epoll_event alignment */
    sizeof(struct dirent), /* dirent size */
    _Alignof(struct dirent), /* dirent alignment */
    sizeof(struct dirent64), /* dirent64 size */
    _Alignof(struct dirent64), /* dirent64 alignment */
    sizeof(struct rlimit), /* rlimit size */
    _Alignof(struct rlimit), /* rlimit alignment */
    sizeof(struct rusage), /* rusage size */
    _Alignof(struct rusage), /* rusage alignment */
    sizeof(ino_t), /* ino_t size */
    ((ino_t)-1 < (ino_t)0), /* ino_t signedness */
    sizeof(off_t), /* off_t size */
    ((off_t)-1 < (off_t)0), /* off_t signedness */
    sizeof(blkcnt_t), /* blkcnt_t size */
    ((blkcnt_t)-1 < (blkcnt_t)0), /* blkcnt_t signedness */
    sizeof(shmatt_t), /* shmatt_t size */
    ((shmatt_t)-1 < (shmatt_t)0), /* shmatt_t signedness */
    sizeof(msgqnum_t), /* msgqnum_t size */
    ((msgqnum_t)-1 < (msgqnum_t)0), /* msgqnum_t signedness */
    sizeof(msglen_t), /* msglen_t size */
    ((msglen_t)-1 < (msglen_t)0), /* msglen_t signedness */
    sizeof(fsblkcnt_t), /* fsblkcnt_t size */
    ((fsblkcnt_t)-1 < (fsblkcnt_t)0), /* fsblkcnt_t signedness */
    sizeof(fsfilcnt_t), /* fsfilcnt_t size */
    ((fsfilcnt_t)-1 < (fsfilcnt_t)0), /* fsfilcnt_t signedness */
    sizeof(rlim_t), /* rlim_t size */
    ((rlim_t)-1 < (rlim_t)0), /* rlim_t signedness */
    sizeof(__syscall_ulong_t), /* __syscall_ulong_t size */
    ((__syscall_ulong_t)-1 < (__syscall_ulong_t)0), /* __syscall_ulong_t signedness */
    sizeof(clock_t), /* clock_t size */
    ((clock_t)-1 < (clock_t)0), /* clock_t signedness */
    sizeof(time_t), /* time_t size */
    ((time_t)-1 < (time_t)0), /* time_t signedness */
    sizeof(__fsword_t), /* __fsword_t size */
    ((__fsword_t)-1 < (__fsword_t)0), /* __fsword_t signedness */
    sizeof(wchar_t), /* wchar_t size */
    ((wchar_t)-1 < (wchar_t)0), /* wchar_t signedness */
    sizeof(nlink_t), /* nlink_t size */
    ((nlink_t)-1 < (nlink_t)0), /* nlink_t signedness */
    sizeof(blksize_t), /* blksize_t size */
    ((blksize_t)-1 < (blksize_t)0), /* blksize_t signedness */
    sizeof(fsblkcnt64_t), /* fsblkcnt64_t size */
    ((fsblkcnt64_t)-1 < (fsblkcnt64_t)0), /* fsblkcnt64_t signedness */
    sizeof(fsfilcnt64_t), /* fsfilcnt64_t size */
    ((fsfilcnt64_t)-1 < (fsfilcnt64_t)0), /* fsfilcnt64_t signedness */
    sizeof(suseconds_t), /* suseconds_t size */
    ((suseconds_t)-1 < (suseconds_t)0), /* suseconds_t signedness */
};
uint64_t c_layout_value(unsigned i) { return values[i]; }
