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
struct termios2 { unsigned int c_iflag,c_oflag,c_cflag,c_lflag; unsigned char c_cc[19],c_line; unsigned int c_ispeed,c_ospeed; };
struct termio { unsigned short c_iflag,c_oflag,c_cflag,c_lflag; unsigned char c_line,c_cc[8]; };
#include <linux/serial.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
#include <sys/utsname.h>

int main(void) {
#ifdef ABDAY_1
printf("ABDAY_1 %llu\n", (unsigned long long)(long long)(ABDAY_1));
#endif
#ifdef ABDAY_2
printf("ABDAY_2 %llu\n", (unsigned long long)(long long)(ABDAY_2));
#endif
#ifdef ABDAY_3
printf("ABDAY_3 %llu\n", (unsigned long long)(long long)(ABDAY_3));
#endif
#ifdef ABDAY_4
printf("ABDAY_4 %llu\n", (unsigned long long)(long long)(ABDAY_4));
#endif
#ifdef ABDAY_5
printf("ABDAY_5 %llu\n", (unsigned long long)(long long)(ABDAY_5));
#endif
#ifdef ABDAY_6
printf("ABDAY_6 %llu\n", (unsigned long long)(long long)(ABDAY_6));
#endif
#ifdef ABDAY_7
printf("ABDAY_7 %llu\n", (unsigned long long)(long long)(ABDAY_7));
#endif
#ifdef ABMON_1
printf("ABMON_1 %llu\n", (unsigned long long)(long long)(ABMON_1));
#endif
#ifdef ABMON_10
printf("ABMON_10 %llu\n", (unsigned long long)(long long)(ABMON_10));
#endif
#ifdef ABMON_11
printf("ABMON_11 %llu\n", (unsigned long long)(long long)(ABMON_11));
#endif
#ifdef ABMON_12
printf("ABMON_12 %llu\n", (unsigned long long)(long long)(ABMON_12));
#endif
#ifdef ABMON_2
printf("ABMON_2 %llu\n", (unsigned long long)(long long)(ABMON_2));
#endif
#ifdef ABMON_3
printf("ABMON_3 %llu\n", (unsigned long long)(long long)(ABMON_3));
#endif
#ifdef ABMON_4
printf("ABMON_4 %llu\n", (unsigned long long)(long long)(ABMON_4));
#endif
#ifdef ABMON_5
printf("ABMON_5 %llu\n", (unsigned long long)(long long)(ABMON_5));
#endif
#ifdef ABMON_6
printf("ABMON_6 %llu\n", (unsigned long long)(long long)(ABMON_6));
#endif
#ifdef ABMON_7
printf("ABMON_7 %llu\n", (unsigned long long)(long long)(ABMON_7));
#endif
#ifdef ABMON_8
printf("ABMON_8 %llu\n", (unsigned long long)(long long)(ABMON_8));
#endif
#ifdef ABMON_9
printf("ABMON_9 %llu\n", (unsigned long long)(long long)(ABMON_9));
#endif
#ifdef ABS_CNT
printf("ABS_CNT %llu\n", (unsigned long long)(long long)(ABS_CNT));
#endif
#ifdef ABS_MAX
printf("ABS_MAX %llu\n", (unsigned long long)(long long)(ABS_MAX));
#endif
#ifdef ACCOUNTING
printf("ACCOUNTING %llu\n", (unsigned long long)(long long)(ACCOUNTING));
#endif
#ifdef ADDR_COMPAT_LAYOUT
printf("ADDR_COMPAT_LAYOUT %llu\n", (unsigned long long)(long long)(ADDR_COMPAT_LAYOUT));
#endif
#ifdef ADDR_LIMIT_32BIT
printf("ADDR_LIMIT_32BIT %llu\n", (unsigned long long)(long long)(ADDR_LIMIT_32BIT));
#endif
#ifdef ADDR_LIMIT_3GB
printf("ADDR_LIMIT_3GB %llu\n", (unsigned long long)(long long)(ADDR_LIMIT_3GB));
#endif
#ifdef ADDR_NO_RANDOMIZE
printf("ADDR_NO_RANDOMIZE %llu\n", (unsigned long long)(long long)(ADDR_NO_RANDOMIZE));
#endif
#ifdef ADFS_SUPER_MAGIC
printf("ADFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(ADFS_SUPER_MAGIC));
#endif
#ifdef ADJ_ESTERROR
printf("ADJ_ESTERROR %llu\n", (unsigned long long)(long long)(ADJ_ESTERROR));
#endif
#ifdef ADJ_FREQUENCY
printf("ADJ_FREQUENCY %llu\n", (unsigned long long)(long long)(ADJ_FREQUENCY));
#endif
#ifdef ADJ_MAXERROR
printf("ADJ_MAXERROR %llu\n", (unsigned long long)(long long)(ADJ_MAXERROR));
#endif
#ifdef ADJ_MICRO
printf("ADJ_MICRO %llu\n", (unsigned long long)(long long)(ADJ_MICRO));
#endif
#ifdef ADJ_NANO
printf("ADJ_NANO %llu\n", (unsigned long long)(long long)(ADJ_NANO));
#endif
#ifdef ADJ_OFFSET
printf("ADJ_OFFSET %llu\n", (unsigned long long)(long long)(ADJ_OFFSET));
#endif
#ifdef ADJ_OFFSET_SINGLESHOT
printf("ADJ_OFFSET_SINGLESHOT %llu\n", (unsigned long long)(long long)(ADJ_OFFSET_SINGLESHOT));
#endif
#ifdef ADJ_OFFSET_SS_READ
printf("ADJ_OFFSET_SS_READ %llu\n", (unsigned long long)(long long)(ADJ_OFFSET_SS_READ));
#endif
#ifdef ADJ_SETOFFSET
printf("ADJ_SETOFFSET %llu\n", (unsigned long long)(long long)(ADJ_SETOFFSET));
#endif
#ifdef ADJ_STATUS
printf("ADJ_STATUS %llu\n", (unsigned long long)(long long)(ADJ_STATUS));
#endif
#ifdef ADJ_TAI
printf("ADJ_TAI %llu\n", (unsigned long long)(long long)(ADJ_TAI));
#endif
#ifdef ADJ_TICK
printf("ADJ_TICK %llu\n", (unsigned long long)(long long)(ADJ_TICK));
#endif
#ifdef ADJ_TIMECONST
printf("ADJ_TIMECONST %llu\n", (unsigned long long)(long long)(ADJ_TIMECONST));
#endif
#ifdef AFFS_SUPER_MAGIC
printf("AFFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(AFFS_SUPER_MAGIC));
#endif
#ifdef AFS_SUPER_MAGIC
printf("AFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(AFS_SUPER_MAGIC));
#endif
#ifdef AF_ALG
printf("AF_ALG %llu\n", (unsigned long long)(long long)(AF_ALG));
#endif
#ifdef AF_APPLETALK
printf("AF_APPLETALK %llu\n", (unsigned long long)(long long)(AF_APPLETALK));
#endif
#ifdef AF_ASH
printf("AF_ASH %llu\n", (unsigned long long)(long long)(AF_ASH));
#endif
#ifdef AF_ATMPVC
printf("AF_ATMPVC %llu\n", (unsigned long long)(long long)(AF_ATMPVC));
#endif
#ifdef AF_ATMSVC
printf("AF_ATMSVC %llu\n", (unsigned long long)(long long)(AF_ATMSVC));
#endif
#ifdef AF_AX25
printf("AF_AX25 %llu\n", (unsigned long long)(long long)(AF_AX25));
#endif
#ifdef AF_BLUETOOTH
printf("AF_BLUETOOTH %llu\n", (unsigned long long)(long long)(AF_BLUETOOTH));
#endif
#ifdef AF_BRIDGE
printf("AF_BRIDGE %llu\n", (unsigned long long)(long long)(AF_BRIDGE));
#endif
#ifdef AF_CAIF
printf("AF_CAIF %llu\n", (unsigned long long)(long long)(AF_CAIF));
#endif
#ifdef AF_CAN
printf("AF_CAN %llu\n", (unsigned long long)(long long)(AF_CAN));
#endif
#ifdef AF_DECnet
printf("AF_DECnet %llu\n", (unsigned long long)(long long)(AF_DECnet));
#endif
#ifdef AF_ECONET
printf("AF_ECONET %llu\n", (unsigned long long)(long long)(AF_ECONET));
#endif
#ifdef AF_IB
printf("AF_IB %llu\n", (unsigned long long)(long long)(AF_IB));
#endif
#ifdef AF_IEEE802154
printf("AF_IEEE802154 %llu\n", (unsigned long long)(long long)(AF_IEEE802154));
#endif
#ifdef AF_INET
printf("AF_INET %llu\n", (unsigned long long)(long long)(AF_INET));
#endif
#ifdef AF_INET6
printf("AF_INET6 %llu\n", (unsigned long long)(long long)(AF_INET6));
#endif
#ifdef AF_IPX
printf("AF_IPX %llu\n", (unsigned long long)(long long)(AF_IPX));
#endif
#ifdef AF_IRDA
printf("AF_IRDA %llu\n", (unsigned long long)(long long)(AF_IRDA));
#endif
#ifdef AF_ISDN
printf("AF_ISDN %llu\n", (unsigned long long)(long long)(AF_ISDN));
#endif
#ifdef AF_IUCV
printf("AF_IUCV %llu\n", (unsigned long long)(long long)(AF_IUCV));
#endif
#ifdef AF_KEY
printf("AF_KEY %llu\n", (unsigned long long)(long long)(AF_KEY));
#endif
#ifdef AF_LLC
printf("AF_LLC %llu\n", (unsigned long long)(long long)(AF_LLC));
#endif
#ifdef AF_LOCAL
printf("AF_LOCAL %llu\n", (unsigned long long)(long long)(AF_LOCAL));
#endif
#ifdef AF_MPLS
printf("AF_MPLS %llu\n", (unsigned long long)(long long)(AF_MPLS));
#endif
#ifdef AF_NETBEUI
printf("AF_NETBEUI %llu\n", (unsigned long long)(long long)(AF_NETBEUI));
#endif
#ifdef AF_NETLINK
printf("AF_NETLINK %llu\n", (unsigned long long)(long long)(AF_NETLINK));
#endif
#ifdef AF_NETROM
printf("AF_NETROM %llu\n", (unsigned long long)(long long)(AF_NETROM));
#endif
#ifdef AF_NFC
printf("AF_NFC %llu\n", (unsigned long long)(long long)(AF_NFC));
#endif
#ifdef AF_PACKET
printf("AF_PACKET %llu\n", (unsigned long long)(long long)(AF_PACKET));
#endif
#ifdef AF_PHONET
printf("AF_PHONET %llu\n", (unsigned long long)(long long)(AF_PHONET));
#endif
#ifdef AF_PPPOX
printf("AF_PPPOX %llu\n", (unsigned long long)(long long)(AF_PPPOX));
#endif
#ifdef AF_RDS
printf("AF_RDS %llu\n", (unsigned long long)(long long)(AF_RDS));
#endif
#ifdef AF_ROSE
printf("AF_ROSE %llu\n", (unsigned long long)(long long)(AF_ROSE));
#endif
#ifdef AF_ROUTE
printf("AF_ROUTE %llu\n", (unsigned long long)(long long)(AF_ROUTE));
#endif
#ifdef AF_RXRPC
printf("AF_RXRPC %llu\n", (unsigned long long)(long long)(AF_RXRPC));
#endif
#ifdef AF_SECURITY
printf("AF_SECURITY %llu\n", (unsigned long long)(long long)(AF_SECURITY));
#endif
#ifdef AF_SNA
printf("AF_SNA %llu\n", (unsigned long long)(long long)(AF_SNA));
#endif
#ifdef AF_TIPC
printf("AF_TIPC %llu\n", (unsigned long long)(long long)(AF_TIPC));
#endif
#ifdef AF_UNIX
printf("AF_UNIX %llu\n", (unsigned long long)(long long)(AF_UNIX));
#endif
#ifdef AF_UNSPEC
printf("AF_UNSPEC %llu\n", (unsigned long long)(long long)(AF_UNSPEC));
#endif
#ifdef AF_VSOCK
printf("AF_VSOCK %llu\n", (unsigned long long)(long long)(AF_VSOCK));
#endif
#ifdef AF_WANPIPE
printf("AF_WANPIPE %llu\n", (unsigned long long)(long long)(AF_WANPIPE));
#endif
#ifdef AF_X25
printf("AF_X25 %llu\n", (unsigned long long)(long long)(AF_X25));
#endif
#ifdef AF_XDP
printf("AF_XDP %llu\n", (unsigned long long)(long long)(AF_XDP));
#endif
#ifdef AIO_ALLDONE
printf("AIO_ALLDONE %llu\n", (unsigned long long)(long long)(AIO_ALLDONE));
#endif
#ifdef AIO_CANCELED
printf("AIO_CANCELED %llu\n", (unsigned long long)(long long)(AIO_CANCELED));
#endif
#ifdef AIO_NOTCANCELED
printf("AIO_NOTCANCELED %llu\n", (unsigned long long)(long long)(AIO_NOTCANCELED));
#endif
#ifdef AI_ADDRCONFIG
printf("AI_ADDRCONFIG %llu\n", (unsigned long long)(long long)(AI_ADDRCONFIG));
#endif
#ifdef AI_ALL
printf("AI_ALL %llu\n", (unsigned long long)(long long)(AI_ALL));
#endif
#ifdef AI_CANONNAME
printf("AI_CANONNAME %llu\n", (unsigned long long)(long long)(AI_CANONNAME));
#endif
#ifdef AI_NUMERICHOST
printf("AI_NUMERICHOST %llu\n", (unsigned long long)(long long)(AI_NUMERICHOST));
#endif
#ifdef AI_NUMERICSERV
printf("AI_NUMERICSERV %llu\n", (unsigned long long)(long long)(AI_NUMERICSERV));
#endif
#ifdef AI_PASSIVE
printf("AI_PASSIVE %llu\n", (unsigned long long)(long long)(AI_PASSIVE));
#endif
#ifdef AI_V4MAPPED
printf("AI_V4MAPPED %llu\n", (unsigned long long)(long long)(AI_V4MAPPED));
#endif
#ifdef ALG_OP_DECRYPT
printf("ALG_OP_DECRYPT %llu\n", (unsigned long long)(long long)(ALG_OP_DECRYPT));
#endif
#ifdef ALG_OP_ENCRYPT
printf("ALG_OP_ENCRYPT %llu\n", (unsigned long long)(long long)(ALG_OP_ENCRYPT));
#endif
#ifdef ALG_SET_AEAD_ASSOCLEN
printf("ALG_SET_AEAD_ASSOCLEN %llu\n", (unsigned long long)(long long)(ALG_SET_AEAD_ASSOCLEN));
#endif
#ifdef ALG_SET_AEAD_AUTHSIZE
printf("ALG_SET_AEAD_AUTHSIZE %llu\n", (unsigned long long)(long long)(ALG_SET_AEAD_AUTHSIZE));
#endif
#ifdef ALG_SET_DRBG_ENTROPY
printf("ALG_SET_DRBG_ENTROPY %llu\n", (unsigned long long)(long long)(ALG_SET_DRBG_ENTROPY));
#endif
#ifdef ALG_SET_IV
printf("ALG_SET_IV %llu\n", (unsigned long long)(long long)(ALG_SET_IV));
#endif
#ifdef ALG_SET_KEY
printf("ALG_SET_KEY %llu\n", (unsigned long long)(long long)(ALG_SET_KEY));
#endif
#ifdef ALG_SET_KEY_BY_KEY_SERIAL
printf("ALG_SET_KEY_BY_KEY_SERIAL %llu\n", (unsigned long long)(long long)(ALG_SET_KEY_BY_KEY_SERIAL));
#endif
#ifdef ALG_SET_OP
printf("ALG_SET_OP %llu\n", (unsigned long long)(long long)(ALG_SET_OP));
#endif
#ifdef ALT_DIGITS
printf("ALT_DIGITS %llu\n", (unsigned long long)(long long)(ALT_DIGITS));
#endif
#ifdef AM_STR
printf("AM_STR %llu\n", (unsigned long long)(long long)(AM_STR));
#endif
#ifdef ARPD_FLUSH
printf("ARPD_FLUSH %llu\n", (unsigned long long)(long long)(ARPD_FLUSH));
#endif
#ifdef ARPD_LOOKUP
printf("ARPD_LOOKUP %llu\n", (unsigned long long)(long long)(ARPD_LOOKUP));
#endif
#ifdef ARPD_UPDATE
printf("ARPD_UPDATE %llu\n", (unsigned long long)(long long)(ARPD_UPDATE));
#endif
#ifdef ARPHRD_ADAPT
printf("ARPHRD_ADAPT %llu\n", (unsigned long long)(long long)(ARPHRD_ADAPT));
#endif
#ifdef ARPHRD_APPLETLK
printf("ARPHRD_APPLETLK %llu\n", (unsigned long long)(long long)(ARPHRD_APPLETLK));
#endif
#ifdef ARPHRD_ARCNET
printf("ARPHRD_ARCNET %llu\n", (unsigned long long)(long long)(ARPHRD_ARCNET));
#endif
#ifdef ARPHRD_ASH
printf("ARPHRD_ASH %llu\n", (unsigned long long)(long long)(ARPHRD_ASH));
#endif
#ifdef ARPHRD_ATM
printf("ARPHRD_ATM %llu\n", (unsigned long long)(long long)(ARPHRD_ATM));
#endif
#ifdef ARPHRD_AX25
printf("ARPHRD_AX25 %llu\n", (unsigned long long)(long long)(ARPHRD_AX25));
#endif
#ifdef ARPHRD_BIF
printf("ARPHRD_BIF %llu\n", (unsigned long long)(long long)(ARPHRD_BIF));
#endif
#ifdef ARPHRD_CAN
printf("ARPHRD_CAN %llu\n", (unsigned long long)(long long)(ARPHRD_CAN));
#endif
#ifdef ARPHRD_CHAOS
printf("ARPHRD_CHAOS %llu\n", (unsigned long long)(long long)(ARPHRD_CHAOS));
#endif
#ifdef ARPHRD_CISCO
printf("ARPHRD_CISCO %llu\n", (unsigned long long)(long long)(ARPHRD_CISCO));
#endif
#ifdef ARPHRD_CSLIP
printf("ARPHRD_CSLIP %llu\n", (unsigned long long)(long long)(ARPHRD_CSLIP));
#endif
#ifdef ARPHRD_CSLIP6
printf("ARPHRD_CSLIP6 %llu\n", (unsigned long long)(long long)(ARPHRD_CSLIP6));
#endif
#ifdef ARPHRD_DDCMP
printf("ARPHRD_DDCMP %llu\n", (unsigned long long)(long long)(ARPHRD_DDCMP));
#endif
#ifdef ARPHRD_DLCI
printf("ARPHRD_DLCI %llu\n", (unsigned long long)(long long)(ARPHRD_DLCI));
#endif
#ifdef ARPHRD_ECONET
printf("ARPHRD_ECONET %llu\n", (unsigned long long)(long long)(ARPHRD_ECONET));
#endif
#ifdef ARPHRD_EETHER
printf("ARPHRD_EETHER %llu\n", (unsigned long long)(long long)(ARPHRD_EETHER));
#endif
#ifdef ARPHRD_ETHER
printf("ARPHRD_ETHER %llu\n", (unsigned long long)(long long)(ARPHRD_ETHER));
#endif
#ifdef ARPHRD_EUI64
printf("ARPHRD_EUI64 %llu\n", (unsigned long long)(long long)(ARPHRD_EUI64));
#endif
#ifdef ARPHRD_FCAL
printf("ARPHRD_FCAL %llu\n", (unsigned long long)(long long)(ARPHRD_FCAL));
#endif
#ifdef ARPHRD_FCFABRIC
printf("ARPHRD_FCFABRIC %llu\n", (unsigned long long)(long long)(ARPHRD_FCFABRIC));
#endif
#ifdef ARPHRD_FCPL
printf("ARPHRD_FCPL %llu\n", (unsigned long long)(long long)(ARPHRD_FCPL));
#endif
#ifdef ARPHRD_FCPP
printf("ARPHRD_FCPP %llu\n", (unsigned long long)(long long)(ARPHRD_FCPP));
#endif
#ifdef ARPHRD_FDDI
printf("ARPHRD_FDDI %llu\n", (unsigned long long)(long long)(ARPHRD_FDDI));
#endif
#ifdef ARPHRD_FRAD
printf("ARPHRD_FRAD %llu\n", (unsigned long long)(long long)(ARPHRD_FRAD));
#endif
#ifdef ARPHRD_HDLC
printf("ARPHRD_HDLC %llu\n", (unsigned long long)(long long)(ARPHRD_HDLC));
#endif
#ifdef ARPHRD_HIPPI
printf("ARPHRD_HIPPI %llu\n", (unsigned long long)(long long)(ARPHRD_HIPPI));
#endif
#ifdef ARPHRD_HWX25
printf("ARPHRD_HWX25 %llu\n", (unsigned long long)(long long)(ARPHRD_HWX25));
#endif
#ifdef ARPHRD_IEEE1394
printf("ARPHRD_IEEE1394 %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE1394));
#endif
#ifdef ARPHRD_IEEE802
printf("ARPHRD_IEEE802 %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE802));
#endif
#ifdef ARPHRD_IEEE80211
printf("ARPHRD_IEEE80211 %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE80211));
#endif
#ifdef ARPHRD_IEEE80211_PRISM
printf("ARPHRD_IEEE80211_PRISM %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE80211_PRISM));
#endif
#ifdef ARPHRD_IEEE80211_RADIOTAP
printf("ARPHRD_IEEE80211_RADIOTAP %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE80211_RADIOTAP));
#endif
#ifdef ARPHRD_IEEE802154
printf("ARPHRD_IEEE802154 %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE802154));
#endif
#ifdef ARPHRD_IEEE802_TR
printf("ARPHRD_IEEE802_TR %llu\n", (unsigned long long)(long long)(ARPHRD_IEEE802_TR));
#endif
#ifdef ARPHRD_INFINIBAND
printf("ARPHRD_INFINIBAND %llu\n", (unsigned long long)(long long)(ARPHRD_INFINIBAND));
#endif
#ifdef ARPHRD_IPDDP
printf("ARPHRD_IPDDP %llu\n", (unsigned long long)(long long)(ARPHRD_IPDDP));
#endif
#ifdef ARPHRD_IPGRE
printf("ARPHRD_IPGRE %llu\n", (unsigned long long)(long long)(ARPHRD_IPGRE));
#endif
#ifdef ARPHRD_IRDA
printf("ARPHRD_IRDA %llu\n", (unsigned long long)(long long)(ARPHRD_IRDA));
#endif
#ifdef ARPHRD_LAPB
printf("ARPHRD_LAPB %llu\n", (unsigned long long)(long long)(ARPHRD_LAPB));
#endif
#ifdef ARPHRD_LOCALTLK
printf("ARPHRD_LOCALTLK %llu\n", (unsigned long long)(long long)(ARPHRD_LOCALTLK));
#endif
#ifdef ARPHRD_LOOPBACK
printf("ARPHRD_LOOPBACK %llu\n", (unsigned long long)(long long)(ARPHRD_LOOPBACK));
#endif
#ifdef ARPHRD_METRICOM
printf("ARPHRD_METRICOM %llu\n", (unsigned long long)(long long)(ARPHRD_METRICOM));
#endif
#ifdef ARPHRD_NETROM
printf("ARPHRD_NETROM %llu\n", (unsigned long long)(long long)(ARPHRD_NETROM));
#endif
#ifdef ARPHRD_NONE
printf("ARPHRD_NONE %llu\n", (unsigned long long)(long long)(ARPHRD_NONE));
#endif
#ifdef ARPHRD_PIMREG
printf("ARPHRD_PIMREG %llu\n", (unsigned long long)(long long)(ARPHRD_PIMREG));
#endif
#ifdef ARPHRD_PPP
printf("ARPHRD_PPP %llu\n", (unsigned long long)(long long)(ARPHRD_PPP));
#endif
#ifdef ARPHRD_PRONET
printf("ARPHRD_PRONET %llu\n", (unsigned long long)(long long)(ARPHRD_PRONET));
#endif
#ifdef ARPHRD_RAWHDLC
printf("ARPHRD_RAWHDLC %llu\n", (unsigned long long)(long long)(ARPHRD_RAWHDLC));
#endif
#ifdef ARPHRD_ROSE
printf("ARPHRD_ROSE %llu\n", (unsigned long long)(long long)(ARPHRD_ROSE));
#endif
#ifdef ARPHRD_RSRVD
printf("ARPHRD_RSRVD %llu\n", (unsigned long long)(long long)(ARPHRD_RSRVD));
#endif
#ifdef ARPHRD_SIT
printf("ARPHRD_SIT %llu\n", (unsigned long long)(long long)(ARPHRD_SIT));
#endif
#ifdef ARPHRD_SKIP
printf("ARPHRD_SKIP %llu\n", (unsigned long long)(long long)(ARPHRD_SKIP));
#endif
#ifdef ARPHRD_SLIP
printf("ARPHRD_SLIP %llu\n", (unsigned long long)(long long)(ARPHRD_SLIP));
#endif
#ifdef ARPHRD_SLIP6
printf("ARPHRD_SLIP6 %llu\n", (unsigned long long)(long long)(ARPHRD_SLIP6));
#endif
#ifdef ARPHRD_TUNNEL
printf("ARPHRD_TUNNEL %llu\n", (unsigned long long)(long long)(ARPHRD_TUNNEL));
#endif
#ifdef ARPHRD_TUNNEL6
printf("ARPHRD_TUNNEL6 %llu\n", (unsigned long long)(long long)(ARPHRD_TUNNEL6));
#endif
#ifdef ARPHRD_VOID
printf("ARPHRD_VOID %llu\n", (unsigned long long)(long long)(ARPHRD_VOID));
#endif
#ifdef ARPHRD_X25
printf("ARPHRD_X25 %llu\n", (unsigned long long)(long long)(ARPHRD_X25));
#endif
#ifdef ARPOP_InREPLY
printf("ARPOP_InREPLY %llu\n", (unsigned long long)(long long)(ARPOP_InREPLY));
#endif
#ifdef ARPOP_InREQUEST
printf("ARPOP_InREQUEST %llu\n", (unsigned long long)(long long)(ARPOP_InREQUEST));
#endif
#ifdef ARPOP_NAK
printf("ARPOP_NAK %llu\n", (unsigned long long)(long long)(ARPOP_NAK));
#endif
#ifdef ARPOP_REPLY
printf("ARPOP_REPLY %llu\n", (unsigned long long)(long long)(ARPOP_REPLY));
#endif
#ifdef ARPOP_REQUEST
printf("ARPOP_REQUEST %llu\n", (unsigned long long)(long long)(ARPOP_REQUEST));
#endif
#ifdef ARPOP_RREPLY
printf("ARPOP_RREPLY %llu\n", (unsigned long long)(long long)(ARPOP_RREPLY));
#endif
#ifdef ARPOP_RREQUEST
printf("ARPOP_RREQUEST %llu\n", (unsigned long long)(long long)(ARPOP_RREQUEST));
#endif
#ifdef ATF_COM
printf("ATF_COM %llu\n", (unsigned long long)(long long)(ATF_COM));
#endif
#ifdef ATF_DONTPUB
printf("ATF_DONTPUB %llu\n", (unsigned long long)(long long)(ATF_DONTPUB));
#endif
#ifdef ATF_MAGIC
printf("ATF_MAGIC %llu\n", (unsigned long long)(long long)(ATF_MAGIC));
#endif
#ifdef ATF_NETMASK
printf("ATF_NETMASK %llu\n", (unsigned long long)(long long)(ATF_NETMASK));
#endif
#ifdef ATF_PERM
printf("ATF_PERM %llu\n", (unsigned long long)(long long)(ATF_PERM));
#endif
#ifdef ATF_PUBL
printf("ATF_PUBL %llu\n", (unsigned long long)(long long)(ATF_PUBL));
#endif
#ifdef ATF_USETRAILERS
printf("ATF_USETRAILERS %llu\n", (unsigned long long)(long long)(ATF_USETRAILERS));
#endif
#ifdef AT_BASE
printf("AT_BASE %llu\n", (unsigned long long)(long long)(AT_BASE));
#endif
#ifdef AT_BASE_PLATFORM
printf("AT_BASE_PLATFORM %llu\n", (unsigned long long)(long long)(AT_BASE_PLATFORM));
#endif
#ifdef AT_CLKTCK
printf("AT_CLKTCK %llu\n", (unsigned long long)(long long)(AT_CLKTCK));
#endif
#ifdef AT_EACCESS
printf("AT_EACCESS %llu\n", (unsigned long long)(long long)(AT_EACCESS));
#endif
#ifdef AT_EGID
printf("AT_EGID %llu\n", (unsigned long long)(long long)(AT_EGID));
#endif
#ifdef AT_EMPTY_PATH
printf("AT_EMPTY_PATH %llu\n", (unsigned long long)(long long)(AT_EMPTY_PATH));
#endif
#ifdef AT_ENTRY
printf("AT_ENTRY %llu\n", (unsigned long long)(long long)(AT_ENTRY));
#endif
#ifdef AT_EUID
printf("AT_EUID %llu\n", (unsigned long long)(long long)(AT_EUID));
#endif
#ifdef AT_EXECFD
printf("AT_EXECFD %llu\n", (unsigned long long)(long long)(AT_EXECFD));
#endif
#ifdef AT_EXECFN
printf("AT_EXECFN %llu\n", (unsigned long long)(long long)(AT_EXECFN));
#endif
#ifdef AT_FDCWD
printf("AT_FDCWD %llu\n", (unsigned long long)(long long)(AT_FDCWD));
#endif
#ifdef AT_FLAGS
printf("AT_FLAGS %llu\n", (unsigned long long)(long long)(AT_FLAGS));
#endif
#ifdef AT_GID
printf("AT_GID %llu\n", (unsigned long long)(long long)(AT_GID));
#endif
#ifdef AT_HWCAP
printf("AT_HWCAP %llu\n", (unsigned long long)(long long)(AT_HWCAP));
#endif
#ifdef AT_HWCAP2
printf("AT_HWCAP2 %llu\n", (unsigned long long)(long long)(AT_HWCAP2));
#endif
#ifdef AT_HWCAP3
printf("AT_HWCAP3 %llu\n", (unsigned long long)(long long)(AT_HWCAP3));
#endif
#ifdef AT_HWCAP4
printf("AT_HWCAP4 %llu\n", (unsigned long long)(long long)(AT_HWCAP4));
#endif
#ifdef AT_IGNORE
printf("AT_IGNORE %llu\n", (unsigned long long)(long long)(AT_IGNORE));
#endif
#ifdef AT_MINSIGSTKSZ
printf("AT_MINSIGSTKSZ %llu\n", (unsigned long long)(long long)(AT_MINSIGSTKSZ));
#endif
#ifdef AT_NOTELF
printf("AT_NOTELF %llu\n", (unsigned long long)(long long)(AT_NOTELF));
#endif
#ifdef AT_NO_AUTOMOUNT
printf("AT_NO_AUTOMOUNT %llu\n", (unsigned long long)(long long)(AT_NO_AUTOMOUNT));
#endif
#ifdef AT_NULL
printf("AT_NULL %llu\n", (unsigned long long)(long long)(AT_NULL));
#endif
#ifdef AT_PAGESZ
printf("AT_PAGESZ %llu\n", (unsigned long long)(long long)(AT_PAGESZ));
#endif
#ifdef AT_PHDR
printf("AT_PHDR %llu\n", (unsigned long long)(long long)(AT_PHDR));
#endif
#ifdef AT_PHENT
printf("AT_PHENT %llu\n", (unsigned long long)(long long)(AT_PHENT));
#endif
#ifdef AT_PHNUM
printf("AT_PHNUM %llu\n", (unsigned long long)(long long)(AT_PHNUM));
#endif
#ifdef AT_PLATFORM
printf("AT_PLATFORM %llu\n", (unsigned long long)(long long)(AT_PLATFORM));
#endif
#ifdef AT_RANDOM
printf("AT_RANDOM %llu\n", (unsigned long long)(long long)(AT_RANDOM));
#endif
#ifdef AT_RECURSIVE
printf("AT_RECURSIVE %llu\n", (unsigned long long)(long long)(AT_RECURSIVE));
#endif
#ifdef AT_REMOVEDIR
printf("AT_REMOVEDIR %llu\n", (unsigned long long)(long long)(AT_REMOVEDIR));
#endif
#ifdef AT_SECURE
printf("AT_SECURE %llu\n", (unsigned long long)(long long)(AT_SECURE));
#endif
#ifdef AT_STATX_DONT_SYNC
printf("AT_STATX_DONT_SYNC %llu\n", (unsigned long long)(long long)(AT_STATX_DONT_SYNC));
#endif
#ifdef AT_STATX_FORCE_SYNC
printf("AT_STATX_FORCE_SYNC %llu\n", (unsigned long long)(long long)(AT_STATX_FORCE_SYNC));
#endif
#ifdef AT_STATX_SYNC_AS_STAT
printf("AT_STATX_SYNC_AS_STAT %llu\n", (unsigned long long)(long long)(AT_STATX_SYNC_AS_STAT));
#endif
#ifdef AT_STATX_SYNC_TYPE
printf("AT_STATX_SYNC_TYPE %llu\n", (unsigned long long)(long long)(AT_STATX_SYNC_TYPE));
#endif
#ifdef AT_SYMLINK_FOLLOW
printf("AT_SYMLINK_FOLLOW %llu\n", (unsigned long long)(long long)(AT_SYMLINK_FOLLOW));
#endif
#ifdef AT_SYMLINK_NOFOLLOW
printf("AT_SYMLINK_NOFOLLOW %llu\n", (unsigned long long)(long long)(AT_SYMLINK_NOFOLLOW));
#endif
#ifdef AT_SYSINFO_EHDR
printf("AT_SYSINFO_EHDR %llu\n", (unsigned long long)(long long)(AT_SYSINFO_EHDR));
#endif
#ifdef AT_UID
printf("AT_UID %llu\n", (unsigned long long)(long long)(AT_UID));
#endif
#ifdef AUTOFS_SUPER_MAGIC
printf("AUTOFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(AUTOFS_SUPER_MAGIC));
#endif
#ifdef __B0
printf("B0 %llu\n", (unsigned long long)(__B0));
#elif defined(B0)
printf("B0 %llu\n", (unsigned long long)(long long)(B0));
#endif
#ifdef __B1000000
printf("B1000000 %llu\n", (unsigned long long)(__B1000000));
#elif defined(B1000000)
printf("B1000000 %llu\n", (unsigned long long)(long long)(B1000000));
#endif
#ifdef __B110
printf("B110 %llu\n", (unsigned long long)(__B110));
#elif defined(B110)
printf("B110 %llu\n", (unsigned long long)(long long)(B110));
#endif
#ifdef __B115200
printf("B115200 %llu\n", (unsigned long long)(__B115200));
#elif defined(B115200)
printf("B115200 %llu\n", (unsigned long long)(long long)(B115200));
#endif
#ifdef __B1152000
printf("B1152000 %llu\n", (unsigned long long)(__B1152000));
#elif defined(B1152000)
printf("B1152000 %llu\n", (unsigned long long)(long long)(B1152000));
#endif
#ifdef __B1200
printf("B1200 %llu\n", (unsigned long long)(__B1200));
#elif defined(B1200)
printf("B1200 %llu\n", (unsigned long long)(long long)(B1200));
#endif
#ifdef __B134
printf("B134 %llu\n", (unsigned long long)(__B134));
#elif defined(B134)
printf("B134 %llu\n", (unsigned long long)(long long)(B134));
#endif
#ifdef __B150
printf("B150 %llu\n", (unsigned long long)(__B150));
#elif defined(B150)
printf("B150 %llu\n", (unsigned long long)(long long)(B150));
#endif
#ifdef __B1500000
printf("B1500000 %llu\n", (unsigned long long)(__B1500000));
#elif defined(B1500000)
printf("B1500000 %llu\n", (unsigned long long)(long long)(B1500000));
#endif
#ifdef __B1800
printf("B1800 %llu\n", (unsigned long long)(__B1800));
#elif defined(B1800)
printf("B1800 %llu\n", (unsigned long long)(long long)(B1800));
#endif
#ifdef __B19200
printf("B19200 %llu\n", (unsigned long long)(__B19200));
#elif defined(B19200)
printf("B19200 %llu\n", (unsigned long long)(long long)(B19200));
#endif
#ifdef __B200
printf("B200 %llu\n", (unsigned long long)(__B200));
#elif defined(B200)
printf("B200 %llu\n", (unsigned long long)(long long)(B200));
#endif
#ifdef __B2000000
printf("B2000000 %llu\n", (unsigned long long)(__B2000000));
#elif defined(B2000000)
printf("B2000000 %llu\n", (unsigned long long)(long long)(B2000000));
#endif
#ifdef __B230400
printf("B230400 %llu\n", (unsigned long long)(__B230400));
#elif defined(B230400)
printf("B230400 %llu\n", (unsigned long long)(long long)(B230400));
#endif
#ifdef __B2400
printf("B2400 %llu\n", (unsigned long long)(__B2400));
#elif defined(B2400)
printf("B2400 %llu\n", (unsigned long long)(long long)(B2400));
#endif
#ifdef __B2500000
printf("B2500000 %llu\n", (unsigned long long)(__B2500000));
#elif defined(B2500000)
printf("B2500000 %llu\n", (unsigned long long)(long long)(B2500000));
#endif
#ifdef __B300
printf("B300 %llu\n", (unsigned long long)(__B300));
#elif defined(B300)
printf("B300 %llu\n", (unsigned long long)(long long)(B300));
#endif
#ifdef __B3000000
printf("B3000000 %llu\n", (unsigned long long)(__B3000000));
#elif defined(B3000000)
printf("B3000000 %llu\n", (unsigned long long)(long long)(B3000000));
#endif
#ifdef __B3500000
printf("B3500000 %llu\n", (unsigned long long)(__B3500000));
#elif defined(B3500000)
printf("B3500000 %llu\n", (unsigned long long)(long long)(B3500000));
#endif
#ifdef __B38400
printf("B38400 %llu\n", (unsigned long long)(__B38400));
#elif defined(B38400)
printf("B38400 %llu\n", (unsigned long long)(long long)(B38400));
#endif
#ifdef __B4000000
printf("B4000000 %llu\n", (unsigned long long)(__B4000000));
#elif defined(B4000000)
printf("B4000000 %llu\n", (unsigned long long)(long long)(B4000000));
#endif
#ifdef __B460800
printf("B460800 %llu\n", (unsigned long long)(__B460800));
#elif defined(B460800)
printf("B460800 %llu\n", (unsigned long long)(long long)(B460800));
#endif
#ifdef __B4800
printf("B4800 %llu\n", (unsigned long long)(__B4800));
#elif defined(B4800)
printf("B4800 %llu\n", (unsigned long long)(long long)(B4800));
#endif
#ifdef __B50
printf("B50 %llu\n", (unsigned long long)(__B50));
#elif defined(B50)
printf("B50 %llu\n", (unsigned long long)(long long)(B50));
#endif
#ifdef __B500000
printf("B500000 %llu\n", (unsigned long long)(__B500000));
#elif defined(B500000)
printf("B500000 %llu\n", (unsigned long long)(long long)(B500000));
#endif
#ifdef __B57600
printf("B57600 %llu\n", (unsigned long long)(__B57600));
#elif defined(B57600)
printf("B57600 %llu\n", (unsigned long long)(long long)(B57600));
#endif
#ifdef __B576000
printf("B576000 %llu\n", (unsigned long long)(__B576000));
#elif defined(B576000)
printf("B576000 %llu\n", (unsigned long long)(long long)(B576000));
#endif
#ifdef __B600
printf("B600 %llu\n", (unsigned long long)(__B600));
#elif defined(B600)
printf("B600 %llu\n", (unsigned long long)(long long)(B600));
#endif
#ifdef __B75
printf("B75 %llu\n", (unsigned long long)(__B75));
#elif defined(B75)
printf("B75 %llu\n", (unsigned long long)(long long)(B75));
#endif
#ifdef __B921600
printf("B921600 %llu\n", (unsigned long long)(__B921600));
#elif defined(B921600)
printf("B921600 %llu\n", (unsigned long long)(long long)(B921600));
#endif
#ifdef __B9600
printf("B9600 %llu\n", (unsigned long long)(__B9600));
#elif defined(B9600)
printf("B9600 %llu\n", (unsigned long long)(long long)(B9600));
#endif
#ifdef BINDERFS_SUPER_MAGIC
printf("BINDERFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(BINDERFS_SUPER_MAGIC));
#endif
#ifdef BLKIOMIN
printf("BLKIOMIN %llu\n", (unsigned long long)(long long)(BLKIOMIN));
#endif
#ifdef BLKIOOPT
printf("BLKIOOPT %llu\n", (unsigned long long)(long long)(BLKIOOPT));
#endif
#ifdef BLKPBSZGET
printf("BLKPBSZGET %llu\n", (unsigned long long)(long long)(BLKPBSZGET));
#endif
#ifdef BLKSSZGET
printf("BLKSSZGET %llu\n", (unsigned long long)(long long)(BLKSSZGET));
#endif
#ifdef BOOT_TIME
printf("BOOT_TIME %llu\n", (unsigned long long)(long long)(BOOT_TIME));
#endif
#ifdef BOTHER
printf("BOTHER %llu\n", (unsigned long long)(long long)(BOTHER));
#endif
#ifdef BPF_A
printf("BPF_A %llu\n", (unsigned long long)(long long)(BPF_A));
#endif
#ifdef BPF_ABS
printf("BPF_ABS %llu\n", (unsigned long long)(long long)(BPF_ABS));
#endif
#ifdef BPF_ADD
printf("BPF_ADD %llu\n", (unsigned long long)(long long)(BPF_ADD));
#endif
#ifdef BPF_ALU
printf("BPF_ALU %llu\n", (unsigned long long)(long long)(BPF_ALU));
#endif
#ifdef BPF_AND
printf("BPF_AND %llu\n", (unsigned long long)(long long)(BPF_AND));
#endif
#ifdef BPF_B
printf("BPF_B %llu\n", (unsigned long long)(long long)(BPF_B));
#endif
#ifdef BPF_DIV
printf("BPF_DIV %llu\n", (unsigned long long)(long long)(BPF_DIV));
#endif
#ifdef BPF_FS_MAGIC
printf("BPF_FS_MAGIC %llu\n", (unsigned long long)(long long)(BPF_FS_MAGIC));
#endif
#ifdef BPF_H
printf("BPF_H %llu\n", (unsigned long long)(long long)(BPF_H));
#endif
#ifdef BPF_IMM
printf("BPF_IMM %llu\n", (unsigned long long)(long long)(BPF_IMM));
#endif
#ifdef BPF_IND
printf("BPF_IND %llu\n", (unsigned long long)(long long)(BPF_IND));
#endif
#ifdef BPF_JA
printf("BPF_JA %llu\n", (unsigned long long)(long long)(BPF_JA));
#endif
#ifdef BPF_JEQ
printf("BPF_JEQ %llu\n", (unsigned long long)(long long)(BPF_JEQ));
#endif
#ifdef BPF_JGE
printf("BPF_JGE %llu\n", (unsigned long long)(long long)(BPF_JGE));
#endif
#ifdef BPF_JGT
printf("BPF_JGT %llu\n", (unsigned long long)(long long)(BPF_JGT));
#endif
#ifdef BPF_JMP
printf("BPF_JMP %llu\n", (unsigned long long)(long long)(BPF_JMP));
#endif
#ifdef BPF_JSET
printf("BPF_JSET %llu\n", (unsigned long long)(long long)(BPF_JSET));
#endif
#ifdef BPF_K
printf("BPF_K %llu\n", (unsigned long long)(long long)(BPF_K));
#endif
#ifdef BPF_LD
printf("BPF_LD %llu\n", (unsigned long long)(long long)(BPF_LD));
#endif
#ifdef BPF_LDX
printf("BPF_LDX %llu\n", (unsigned long long)(long long)(BPF_LDX));
#endif
#ifdef BPF_LEN
printf("BPF_LEN %llu\n", (unsigned long long)(long long)(BPF_LEN));
#endif
#ifdef BPF_LL_OFF
printf("BPF_LL_OFF %llu\n", (unsigned long long)(long long)(BPF_LL_OFF));
#endif
#ifdef BPF_LSH
printf("BPF_LSH %llu\n", (unsigned long long)(long long)(BPF_LSH));
#endif
#ifdef BPF_MAXINSNS
printf("BPF_MAXINSNS %llu\n", (unsigned long long)(long long)(BPF_MAXINSNS));
#endif
#ifdef BPF_MEM
printf("BPF_MEM %llu\n", (unsigned long long)(long long)(BPF_MEM));
#endif
#ifdef BPF_MEMWORDS
printf("BPF_MEMWORDS %llu\n", (unsigned long long)(long long)(BPF_MEMWORDS));
#endif
#ifdef BPF_MISC
printf("BPF_MISC %llu\n", (unsigned long long)(long long)(BPF_MISC));
#endif
#ifdef BPF_MOD
printf("BPF_MOD %llu\n", (unsigned long long)(long long)(BPF_MOD));
#endif
#ifdef BPF_MSH
printf("BPF_MSH %llu\n", (unsigned long long)(long long)(BPF_MSH));
#endif
#ifdef BPF_MUL
printf("BPF_MUL %llu\n", (unsigned long long)(long long)(BPF_MUL));
#endif
#ifdef BPF_NEG
printf("BPF_NEG %llu\n", (unsigned long long)(long long)(BPF_NEG));
#endif
#ifdef BPF_NET_OFF
printf("BPF_NET_OFF %llu\n", (unsigned long long)(long long)(BPF_NET_OFF));
#endif
#ifdef BPF_OR
printf("BPF_OR %llu\n", (unsigned long long)(long long)(BPF_OR));
#endif
#ifdef BPF_RET
printf("BPF_RET %llu\n", (unsigned long long)(long long)(BPF_RET));
#endif
#ifdef BPF_RSH
printf("BPF_RSH %llu\n", (unsigned long long)(long long)(BPF_RSH));
#endif
#ifdef BPF_ST
printf("BPF_ST %llu\n", (unsigned long long)(long long)(BPF_ST));
#endif
#ifdef BPF_STX
printf("BPF_STX %llu\n", (unsigned long long)(long long)(BPF_STX));
#endif
#ifdef BPF_SUB
printf("BPF_SUB %llu\n", (unsigned long long)(long long)(BPF_SUB));
#endif
#ifdef BPF_TAX
printf("BPF_TAX %llu\n", (unsigned long long)(long long)(BPF_TAX));
#endif
#ifdef BPF_TXA
printf("BPF_TXA %llu\n", (unsigned long long)(long long)(BPF_TXA));
#endif
#ifdef BPF_W
printf("BPF_W %llu\n", (unsigned long long)(long long)(BPF_W));
#endif
#ifdef BPF_X
printf("BPF_X %llu\n", (unsigned long long)(long long)(BPF_X));
#endif
#ifdef BPF_XOR
printf("BPF_XOR %llu\n", (unsigned long long)(long long)(BPF_XOR));
#endif
#ifdef BRKINT
printf("BRKINT %llu\n", (unsigned long long)(long long)(BRKINT));
#endif
#ifdef BS0
printf("BS0 %llu\n", (unsigned long long)(long long)(BS0));
#endif
#ifdef BS1
printf("BS1 %llu\n", (unsigned long long)(long long)(BS1));
#endif
#ifdef BSDLY
printf("BSDLY %llu\n", (unsigned long long)(long long)(BSDLY));
#endif
#ifdef BTRFS_SUPER_MAGIC
printf("BTRFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(BTRFS_SUPER_MAGIC));
#endif
#ifdef BUFSIZ
printf("BUFSIZ %llu\n", (unsigned long long)(long long)(BUFSIZ));
#endif
#ifdef BUS_ADRALN
printf("BUS_ADRALN %llu\n", (unsigned long long)(long long)(BUS_ADRALN));
#endif
#ifdef BUS_ADRERR
printf("BUS_ADRERR %llu\n", (unsigned long long)(long long)(BUS_ADRERR));
#endif
#ifdef BUS_MCEERR_AO
printf("BUS_MCEERR_AO %llu\n", (unsigned long long)(long long)(BUS_MCEERR_AO));
#endif
#ifdef BUS_MCEERR_AR
printf("BUS_MCEERR_AR %llu\n", (unsigned long long)(long long)(BUS_MCEERR_AR));
#endif
#ifdef BUS_OBJERR
printf("BUS_OBJERR %llu\n", (unsigned long long)(long long)(BUS_OBJERR));
#endif
#ifdef CBAUD
printf("CBAUD %llu\n", (unsigned long long)(long long)(CBAUD));
#endif
#ifdef CBAUDEX
printf("CBAUDEX %llu\n", (unsigned long long)(long long)(CBAUDEX));
#endif
#ifdef CGROUP2_SUPER_MAGIC
printf("CGROUP2_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(CGROUP2_SUPER_MAGIC));
#endif
#ifdef CGROUP_SUPER_MAGIC
printf("CGROUP_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(CGROUP_SUPER_MAGIC));
#endif
#ifdef CIBAUD
printf("CIBAUD %llu\n", (unsigned long long)(long long)(CIBAUD));
#endif
#ifdef CLD_CONTINUED
printf("CLD_CONTINUED %llu\n", (unsigned long long)(long long)(CLD_CONTINUED));
#endif
#ifdef CLD_DUMPED
printf("CLD_DUMPED %llu\n", (unsigned long long)(long long)(CLD_DUMPED));
#endif
#ifdef CLD_EXITED
printf("CLD_EXITED %llu\n", (unsigned long long)(long long)(CLD_EXITED));
#endif
#ifdef CLD_KILLED
printf("CLD_KILLED %llu\n", (unsigned long long)(long long)(CLD_KILLED));
#endif
#ifdef CLD_STOPPED
printf("CLD_STOPPED %llu\n", (unsigned long long)(long long)(CLD_STOPPED));
#endif
#ifdef CLD_TRAPPED
printf("CLD_TRAPPED %llu\n", (unsigned long long)(long long)(CLD_TRAPPED));
#endif
#ifdef CLOCAL
printf("CLOCAL %llu\n", (unsigned long long)(long long)(CLOCAL));
#endif
#ifdef CLOCK_BOOTTIME
printf("CLOCK_BOOTTIME %llu\n", (unsigned long long)(long long)(CLOCK_BOOTTIME));
#endif
#ifdef CLOCK_BOOTTIME_ALARM
printf("CLOCK_BOOTTIME_ALARM %llu\n", (unsigned long long)(long long)(CLOCK_BOOTTIME_ALARM));
#endif
#ifdef CLOCK_MONOTONIC
printf("CLOCK_MONOTONIC %llu\n", (unsigned long long)(long long)(CLOCK_MONOTONIC));
#endif
#ifdef CLOCK_MONOTONIC_COARSE
printf("CLOCK_MONOTONIC_COARSE %llu\n", (unsigned long long)(long long)(CLOCK_MONOTONIC_COARSE));
#endif
#ifdef CLOCK_MONOTONIC_RAW
printf("CLOCK_MONOTONIC_RAW %llu\n", (unsigned long long)(long long)(CLOCK_MONOTONIC_RAW));
#endif
#ifdef CLOCK_PROCESS_CPUTIME_ID
printf("CLOCK_PROCESS_CPUTIME_ID %llu\n", (unsigned long long)(long long)(CLOCK_PROCESS_CPUTIME_ID));
#endif
#ifdef CLOCK_REALTIME
printf("CLOCK_REALTIME %llu\n", (unsigned long long)(long long)(CLOCK_REALTIME));
#endif
#ifdef CLOCK_REALTIME_ALARM
printf("CLOCK_REALTIME_ALARM %llu\n", (unsigned long long)(long long)(CLOCK_REALTIME_ALARM));
#endif
#ifdef CLOCK_REALTIME_COARSE
printf("CLOCK_REALTIME_COARSE %llu\n", (unsigned long long)(long long)(CLOCK_REALTIME_COARSE));
#endif
#ifdef CLOCK_TAI
printf("CLOCK_TAI %llu\n", (unsigned long long)(long long)(CLOCK_TAI));
#endif
#ifdef CLOCK_THREAD_CPUTIME_ID
printf("CLOCK_THREAD_CPUTIME_ID %llu\n", (unsigned long long)(long long)(CLOCK_THREAD_CPUTIME_ID));
#endif
#ifdef CLONE_CHILD_CLEARTID
printf("CLONE_CHILD_CLEARTID %llu\n", (unsigned long long)(long long)(CLONE_CHILD_CLEARTID));
#endif
#ifdef CLONE_CHILD_SETTID
printf("CLONE_CHILD_SETTID %llu\n", (unsigned long long)(long long)(CLONE_CHILD_SETTID));
#endif
#ifdef CLONE_CLEAR_SIGHAND
printf("CLONE_CLEAR_SIGHAND %llu\n", (unsigned long long)(long long)(CLONE_CLEAR_SIGHAND));
#endif
#ifdef CLONE_DETACHED
printf("CLONE_DETACHED %llu\n", (unsigned long long)(long long)(CLONE_DETACHED));
#endif
#ifdef CLONE_FILES
printf("CLONE_FILES %llu\n", (unsigned long long)(long long)(CLONE_FILES));
#endif
#ifdef CLONE_FS
printf("CLONE_FS %llu\n", (unsigned long long)(long long)(CLONE_FS));
#endif
#ifdef CLONE_INTO_CGROUP
printf("CLONE_INTO_CGROUP %llu\n", (unsigned long long)(long long)(CLONE_INTO_CGROUP));
#endif
#ifdef CLONE_IO
printf("CLONE_IO %llu\n", (unsigned long long)(long long)(CLONE_IO));
#endif
#ifdef CLONE_NEWCGROUP
printf("CLONE_NEWCGROUP %llu\n", (unsigned long long)(long long)(CLONE_NEWCGROUP));
#endif
#ifdef CLONE_NEWIPC
printf("CLONE_NEWIPC %llu\n", (unsigned long long)(long long)(CLONE_NEWIPC));
#endif
#ifdef CLONE_NEWNET
printf("CLONE_NEWNET %llu\n", (unsigned long long)(long long)(CLONE_NEWNET));
#endif
#ifdef CLONE_NEWNS
printf("CLONE_NEWNS %llu\n", (unsigned long long)(long long)(CLONE_NEWNS));
#endif
#ifdef CLONE_NEWPID
printf("CLONE_NEWPID %llu\n", (unsigned long long)(long long)(CLONE_NEWPID));
#endif
#ifdef CLONE_NEWTIME
printf("CLONE_NEWTIME %llu\n", (unsigned long long)(long long)(CLONE_NEWTIME));
#endif
#ifdef CLONE_NEWUSER
printf("CLONE_NEWUSER %llu\n", (unsigned long long)(long long)(CLONE_NEWUSER));
#endif
#ifdef CLONE_NEWUTS
printf("CLONE_NEWUTS %llu\n", (unsigned long long)(long long)(CLONE_NEWUTS));
#endif
#ifdef CLONE_PARENT
printf("CLONE_PARENT %llu\n", (unsigned long long)(long long)(CLONE_PARENT));
#endif
#ifdef CLONE_PARENT_SETTID
printf("CLONE_PARENT_SETTID %llu\n", (unsigned long long)(long long)(CLONE_PARENT_SETTID));
#endif
#ifdef CLONE_PIDFD
printf("CLONE_PIDFD %llu\n", (unsigned long long)(long long)(CLONE_PIDFD));
#endif
#ifdef CLONE_PTRACE
printf("CLONE_PTRACE %llu\n", (unsigned long long)(long long)(CLONE_PTRACE));
#endif
#ifdef CLONE_SETTLS
printf("CLONE_SETTLS %llu\n", (unsigned long long)(long long)(CLONE_SETTLS));
#endif
#ifdef CLONE_SIGHAND
printf("CLONE_SIGHAND %llu\n", (unsigned long long)(long long)(CLONE_SIGHAND));
#endif
#ifdef CLONE_SYSVSEM
printf("CLONE_SYSVSEM %llu\n", (unsigned long long)(long long)(CLONE_SYSVSEM));
#endif
#ifdef CLONE_THREAD
printf("CLONE_THREAD %llu\n", (unsigned long long)(long long)(CLONE_THREAD));
#endif
#ifdef CLONE_UNTRACED
printf("CLONE_UNTRACED %llu\n", (unsigned long long)(long long)(CLONE_UNTRACED));
#endif
#ifdef CLONE_VFORK
printf("CLONE_VFORK %llu\n", (unsigned long long)(long long)(CLONE_VFORK));
#endif
#ifdef CLONE_VM
printf("CLONE_VM %llu\n", (unsigned long long)(long long)(CLONE_VM));
#endif
#ifdef CLOSE_RANGE_CLOEXEC
printf("CLOSE_RANGE_CLOEXEC %llu\n", (unsigned long long)(long long)(CLOSE_RANGE_CLOEXEC));
#endif
#ifdef CLOSE_RANGE_UNSHARE
printf("CLOSE_RANGE_UNSHARE %llu\n", (unsigned long long)(long long)(CLOSE_RANGE_UNSHARE));
#endif
#ifdef CMSPAR
printf("CMSPAR %llu\n", (unsigned long long)(long long)(CMSPAR));
#endif
#ifdef CN_DST_IDX
printf("CN_DST_IDX %llu\n", (unsigned long long)(long long)(CN_DST_IDX));
#endif
#ifdef CN_DST_VAL
printf("CN_DST_VAL %llu\n", (unsigned long long)(long long)(CN_DST_VAL));
#endif
#ifdef CN_IDX_BB
printf("CN_IDX_BB %llu\n", (unsigned long long)(long long)(CN_IDX_BB));
#endif
#ifdef CN_IDX_CIFS
printf("CN_IDX_CIFS %llu\n", (unsigned long long)(long long)(CN_IDX_CIFS));
#endif
#ifdef CN_IDX_DM
printf("CN_IDX_DM %llu\n", (unsigned long long)(long long)(CN_IDX_DM));
#endif
#ifdef CN_IDX_DRBD
printf("CN_IDX_DRBD %llu\n", (unsigned long long)(long long)(CN_IDX_DRBD));
#endif
#ifdef CN_IDX_PROC
printf("CN_IDX_PROC %llu\n", (unsigned long long)(long long)(CN_IDX_PROC));
#endif
#ifdef CN_IDX_V86D
printf("CN_IDX_V86D %llu\n", (unsigned long long)(long long)(CN_IDX_V86D));
#endif
#ifdef CN_KVP_IDX
printf("CN_KVP_IDX %llu\n", (unsigned long long)(long long)(CN_KVP_IDX));
#endif
#ifdef CN_KVP_VAL
printf("CN_KVP_VAL %llu\n", (unsigned long long)(long long)(CN_KVP_VAL));
#endif
#ifdef CN_VAL_CIFS
printf("CN_VAL_CIFS %llu\n", (unsigned long long)(long long)(CN_VAL_CIFS));
#endif
#ifdef CN_VAL_DM_USERSPACE_LOG
printf("CN_VAL_DM_USERSPACE_LOG %llu\n", (unsigned long long)(long long)(CN_VAL_DM_USERSPACE_LOG));
#endif
#ifdef CN_VAL_DRBD
printf("CN_VAL_DRBD %llu\n", (unsigned long long)(long long)(CN_VAL_DRBD));
#endif
#ifdef CN_VAL_PROC
printf("CN_VAL_PROC %llu\n", (unsigned long long)(long long)(CN_VAL_PROC));
#endif
#ifdef CN_VAL_V86D_UVESAFB
printf("CN_VAL_V86D_UVESAFB %llu\n", (unsigned long long)(long long)(CN_VAL_V86D_UVESAFB));
#endif
#ifdef CN_VSS_IDX
printf("CN_VSS_IDX %llu\n", (unsigned long long)(long long)(CN_VSS_IDX));
#endif
#ifdef CN_VSS_VAL
printf("CN_VSS_VAL %llu\n", (unsigned long long)(long long)(CN_VSS_VAL));
#endif
#ifdef CN_W1_IDX
printf("CN_W1_IDX %llu\n", (unsigned long long)(long long)(CN_W1_IDX));
#endif
#ifdef CN_W1_VAL
printf("CN_W1_VAL %llu\n", (unsigned long long)(long long)(CN_W1_VAL));
#endif
#ifdef CODA_SUPER_MAGIC
printf("CODA_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(CODA_SUPER_MAGIC));
#endif
#ifdef CODESET
printf("CODESET %llu\n", (unsigned long long)(long long)(CODESET));
#endif
#ifdef COMPAT_HWCAP_ISA_A
printf("COMPAT_HWCAP_ISA_A %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_A));
#endif
#ifdef COMPAT_HWCAP_ISA_C
printf("COMPAT_HWCAP_ISA_C %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_C));
#endif
#ifdef COMPAT_HWCAP_ISA_D
printf("COMPAT_HWCAP_ISA_D %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_D));
#endif
#ifdef COMPAT_HWCAP_ISA_F
printf("COMPAT_HWCAP_ISA_F %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_F));
#endif
#ifdef COMPAT_HWCAP_ISA_I
printf("COMPAT_HWCAP_ISA_I %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_I));
#endif
#ifdef COMPAT_HWCAP_ISA_M
printf("COMPAT_HWCAP_ISA_M %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_M));
#endif
#ifdef COMPAT_HWCAP_ISA_V
printf("COMPAT_HWCAP_ISA_V %llu\n", (unsigned long long)(long long)(COMPAT_HWCAP_ISA_V));
#endif
#ifdef CPU_SETSIZE
printf("CPU_SETSIZE %llu\n", (unsigned long long)(long long)(CPU_SETSIZE));
#endif
#ifdef CR0
printf("CR0 %llu\n", (unsigned long long)(long long)(CR0));
#endif
#ifdef CR1
printf("CR1 %llu\n", (unsigned long long)(long long)(CR1));
#endif
#ifdef CR2
printf("CR2 %llu\n", (unsigned long long)(long long)(CR2));
#endif
#ifdef CR3
printf("CR3 %llu\n", (unsigned long long)(long long)(CR3));
#endif
#ifdef CRAMFS_MAGIC
printf("CRAMFS_MAGIC %llu\n", (unsigned long long)(long long)(CRAMFS_MAGIC));
#endif
#ifdef CRDLY
printf("CRDLY %llu\n", (unsigned long long)(long long)(CRDLY));
#endif
#ifdef CREAD
printf("CREAD %llu\n", (unsigned long long)(long long)(CREAD));
#endif
#ifdef CRNCYSTR
printf("CRNCYSTR %llu\n", (unsigned long long)(long long)(CRNCYSTR));
#endif
#ifdef CRTSCTS
printf("CRTSCTS %llu\n", (unsigned long long)(long long)(CRTSCTS));
#endif
#ifdef CS5
printf("CS5 %llu\n", (unsigned long long)(long long)(CS5));
#endif
#ifdef CS6
printf("CS6 %llu\n", (unsigned long long)(long long)(CS6));
#endif
#ifdef CS7
printf("CS7 %llu\n", (unsigned long long)(long long)(CS7));
#endif
#ifdef CS8
printf("CS8 %llu\n", (unsigned long long)(long long)(CS8));
#endif
#ifdef CSIGNAL
printf("CSIGNAL %llu\n", (unsigned long long)(long long)(CSIGNAL));
#endif
#ifdef CSIZE
printf("CSIZE %llu\n", (unsigned long long)(long long)(CSIZE));
#endif
#ifdef CSTOPB
printf("CSTOPB %llu\n", (unsigned long long)(long long)(CSTOPB));
#endif
#ifdef CTL_ABI
printf("CTL_ABI %llu\n", (unsigned long long)(long long)(CTL_ABI));
#endif
#ifdef CTL_BUS
printf("CTL_BUS %llu\n", (unsigned long long)(long long)(CTL_BUS));
#endif
#ifdef CTL_BUS_ISA
printf("CTL_BUS_ISA %llu\n", (unsigned long long)(long long)(CTL_BUS_ISA));
#endif
#ifdef CTL_CPU
printf("CTL_CPU %llu\n", (unsigned long long)(long long)(CTL_CPU));
#endif
#ifdef CTL_DEBUG
printf("CTL_DEBUG %llu\n", (unsigned long long)(long long)(CTL_DEBUG));
#endif
#ifdef CTL_DEV
printf("CTL_DEV %llu\n", (unsigned long long)(long long)(CTL_DEV));
#endif
#ifdef CTL_FS
printf("CTL_FS %llu\n", (unsigned long long)(long long)(CTL_FS));
#endif
#ifdef CTL_KERN
printf("CTL_KERN %llu\n", (unsigned long long)(long long)(CTL_KERN));
#endif
#ifdef CTL_NET
printf("CTL_NET %llu\n", (unsigned long long)(long long)(CTL_NET));
#endif
#ifdef CTL_VM
printf("CTL_VM %llu\n", (unsigned long long)(long long)(CTL_VM));
#endif
#ifdef CTRL_ATTR_FAMILY_ID
printf("CTRL_ATTR_FAMILY_ID %llu\n", (unsigned long long)(long long)(CTRL_ATTR_FAMILY_ID));
#endif
#ifdef CTRL_ATTR_FAMILY_NAME
printf("CTRL_ATTR_FAMILY_NAME %llu\n", (unsigned long long)(long long)(CTRL_ATTR_FAMILY_NAME));
#endif
#ifdef CTRL_ATTR_HDRSIZE
printf("CTRL_ATTR_HDRSIZE %llu\n", (unsigned long long)(long long)(CTRL_ATTR_HDRSIZE));
#endif
#ifdef CTRL_ATTR_MAXATTR
printf("CTRL_ATTR_MAXATTR %llu\n", (unsigned long long)(long long)(CTRL_ATTR_MAXATTR));
#endif
#ifdef CTRL_ATTR_MCAST_GROUPS
printf("CTRL_ATTR_MCAST_GROUPS %llu\n", (unsigned long long)(long long)(CTRL_ATTR_MCAST_GROUPS));
#endif
#ifdef CTRL_ATTR_MCAST_GRP_ID
printf("CTRL_ATTR_MCAST_GRP_ID %llu\n", (unsigned long long)(long long)(CTRL_ATTR_MCAST_GRP_ID));
#endif
#ifdef CTRL_ATTR_MCAST_GRP_NAME
printf("CTRL_ATTR_MCAST_GRP_NAME %llu\n", (unsigned long long)(long long)(CTRL_ATTR_MCAST_GRP_NAME));
#endif
#ifdef CTRL_ATTR_MCAST_GRP_UNSPEC
printf("CTRL_ATTR_MCAST_GRP_UNSPEC %llu\n", (unsigned long long)(long long)(CTRL_ATTR_MCAST_GRP_UNSPEC));
#endif
#ifdef CTRL_ATTR_OPS
printf("CTRL_ATTR_OPS %llu\n", (unsigned long long)(long long)(CTRL_ATTR_OPS));
#endif
#ifdef CTRL_ATTR_OP_FLAGS
printf("CTRL_ATTR_OP_FLAGS %llu\n", (unsigned long long)(long long)(CTRL_ATTR_OP_FLAGS));
#endif
#ifdef CTRL_ATTR_OP_ID
printf("CTRL_ATTR_OP_ID %llu\n", (unsigned long long)(long long)(CTRL_ATTR_OP_ID));
#endif
#ifdef CTRL_ATTR_OP_UNSPEC
printf("CTRL_ATTR_OP_UNSPEC %llu\n", (unsigned long long)(long long)(CTRL_ATTR_OP_UNSPEC));
#endif
#ifdef CTRL_ATTR_UNSPEC
printf("CTRL_ATTR_UNSPEC %llu\n", (unsigned long long)(long long)(CTRL_ATTR_UNSPEC));
#endif
#ifdef CTRL_ATTR_VERSION
printf("CTRL_ATTR_VERSION %llu\n", (unsigned long long)(long long)(CTRL_ATTR_VERSION));
#endif
#ifdef CTRL_CMD_DELFAMILY
printf("CTRL_CMD_DELFAMILY %llu\n", (unsigned long long)(long long)(CTRL_CMD_DELFAMILY));
#endif
#ifdef CTRL_CMD_DELMCAST_GRP
printf("CTRL_CMD_DELMCAST_GRP %llu\n", (unsigned long long)(long long)(CTRL_CMD_DELMCAST_GRP));
#endif
#ifdef CTRL_CMD_DELOPS
printf("CTRL_CMD_DELOPS %llu\n", (unsigned long long)(long long)(CTRL_CMD_DELOPS));
#endif
#ifdef CTRL_CMD_GETFAMILY
printf("CTRL_CMD_GETFAMILY %llu\n", (unsigned long long)(long long)(CTRL_CMD_GETFAMILY));
#endif
#ifdef CTRL_CMD_GETMCAST_GRP
printf("CTRL_CMD_GETMCAST_GRP %llu\n", (unsigned long long)(long long)(CTRL_CMD_GETMCAST_GRP));
#endif
#ifdef CTRL_CMD_GETOPS
printf("CTRL_CMD_GETOPS %llu\n", (unsigned long long)(long long)(CTRL_CMD_GETOPS));
#endif
#ifdef CTRL_CMD_NEWFAMILY
printf("CTRL_CMD_NEWFAMILY %llu\n", (unsigned long long)(long long)(CTRL_CMD_NEWFAMILY));
#endif
#ifdef CTRL_CMD_NEWMCAST_GRP
printf("CTRL_CMD_NEWMCAST_GRP %llu\n", (unsigned long long)(long long)(CTRL_CMD_NEWMCAST_GRP));
#endif
#ifdef CTRL_CMD_NEWOPS
printf("CTRL_CMD_NEWOPS %llu\n", (unsigned long long)(long long)(CTRL_CMD_NEWOPS));
#endif
#ifdef CTRL_CMD_UNSPEC
printf("CTRL_CMD_UNSPEC %llu\n", (unsigned long long)(long long)(CTRL_CMD_UNSPEC));
#endif
#ifdef DAY_1
printf("DAY_1 %llu\n", (unsigned long long)(long long)(DAY_1));
#endif
#ifdef DAY_2
printf("DAY_2 %llu\n", (unsigned long long)(long long)(DAY_2));
#endif
#ifdef DAY_3
printf("DAY_3 %llu\n", (unsigned long long)(long long)(DAY_3));
#endif
#ifdef DAY_4
printf("DAY_4 %llu\n", (unsigned long long)(long long)(DAY_4));
#endif
#ifdef DAY_5
printf("DAY_5 %llu\n", (unsigned long long)(long long)(DAY_5));
#endif
#ifdef DAY_6
printf("DAY_6 %llu\n", (unsigned long long)(long long)(DAY_6));
#endif
#ifdef DAY_7
printf("DAY_7 %llu\n", (unsigned long long)(long long)(DAY_7));
#endif
#ifdef DCCP_SERVICE_LIST_MAX_LEN
printf("DCCP_SERVICE_LIST_MAX_LEN %llu\n", (unsigned long long)(long long)(DCCP_SERVICE_LIST_MAX_LEN));
#endif
#ifdef DCCP_SOCKOPT_AVAILABLE_CCIDS
printf("DCCP_SOCKOPT_AVAILABLE_CCIDS %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_AVAILABLE_CCIDS));
#endif
#ifdef DCCP_SOCKOPT_CCID
printf("DCCP_SOCKOPT_CCID %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_CCID));
#endif
#ifdef DCCP_SOCKOPT_CCID_RX_INFO
printf("DCCP_SOCKOPT_CCID_RX_INFO %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_CCID_RX_INFO));
#endif
#ifdef DCCP_SOCKOPT_CCID_TX_INFO
printf("DCCP_SOCKOPT_CCID_TX_INFO %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_CCID_TX_INFO));
#endif
#ifdef DCCP_SOCKOPT_CHANGE_L
printf("DCCP_SOCKOPT_CHANGE_L %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_CHANGE_L));
#endif
#ifdef DCCP_SOCKOPT_CHANGE_R
printf("DCCP_SOCKOPT_CHANGE_R %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_CHANGE_R));
#endif
#ifdef DCCP_SOCKOPT_GET_CUR_MPS
printf("DCCP_SOCKOPT_GET_CUR_MPS %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_GET_CUR_MPS));
#endif
#ifdef DCCP_SOCKOPT_PACKET_SIZE
printf("DCCP_SOCKOPT_PACKET_SIZE %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_PACKET_SIZE));
#endif
#ifdef DCCP_SOCKOPT_QPOLICY_ID
printf("DCCP_SOCKOPT_QPOLICY_ID %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_QPOLICY_ID));
#endif
#ifdef DCCP_SOCKOPT_QPOLICY_TXQLEN
printf("DCCP_SOCKOPT_QPOLICY_TXQLEN %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_QPOLICY_TXQLEN));
#endif
#ifdef DCCP_SOCKOPT_RECV_CSCOV
printf("DCCP_SOCKOPT_RECV_CSCOV %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_RECV_CSCOV));
#endif
#ifdef DCCP_SOCKOPT_RX_CCID
printf("DCCP_SOCKOPT_RX_CCID %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_RX_CCID));
#endif
#ifdef DCCP_SOCKOPT_SEND_CSCOV
printf("DCCP_SOCKOPT_SEND_CSCOV %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_SEND_CSCOV));
#endif
#ifdef DCCP_SOCKOPT_SERVER_TIMEWAIT
printf("DCCP_SOCKOPT_SERVER_TIMEWAIT %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_SERVER_TIMEWAIT));
#endif
#ifdef DCCP_SOCKOPT_SERVICE
printf("DCCP_SOCKOPT_SERVICE %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_SERVICE));
#endif
#ifdef DCCP_SOCKOPT_TX_CCID
printf("DCCP_SOCKOPT_TX_CCID %llu\n", (unsigned long long)(long long)(DCCP_SOCKOPT_TX_CCID));
#endif
#ifdef DEAD_PROCESS
printf("DEAD_PROCESS %llu\n", (unsigned long long)(long long)(DEAD_PROCESS));
#endif
#ifdef DEBUGFS_MAGIC
printf("DEBUGFS_MAGIC %llu\n", (unsigned long long)(long long)(DEBUGFS_MAGIC));
#endif
#ifdef DEVPTS_SUPER_MAGIC
printf("DEVPTS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(DEVPTS_SUPER_MAGIC));
#endif
#ifdef DT_BLK
printf("DT_BLK %llu\n", (unsigned long long)(long long)(DT_BLK));
#endif
#ifdef DT_CHR
printf("DT_CHR %llu\n", (unsigned long long)(long long)(DT_CHR));
#endif
#ifdef DT_DIR
printf("DT_DIR %llu\n", (unsigned long long)(long long)(DT_DIR));
#endif
#ifdef DT_FIFO
printf("DT_FIFO %llu\n", (unsigned long long)(long long)(DT_FIFO));
#endif
#ifdef DT_LNK
printf("DT_LNK %llu\n", (unsigned long long)(long long)(DT_LNK));
#endif
#ifdef DT_REG
printf("DT_REG %llu\n", (unsigned long long)(long long)(DT_REG));
#endif
#ifdef DT_SOCK
printf("DT_SOCK %llu\n", (unsigned long long)(long long)(DT_SOCK));
#endif
#ifdef DT_UNKNOWN
printf("DT_UNKNOWN %llu\n", (unsigned long long)(long long)(DT_UNKNOWN));
#endif
#ifdef D_FMT
printf("D_FMT %llu\n", (unsigned long long)(long long)(D_FMT));
#endif
#ifdef D_T_FMT
printf("D_T_FMT %llu\n", (unsigned long long)(long long)(D_T_FMT));
#endif
#ifdef E2BIG
printf("E2BIG %llu\n", (unsigned long long)(long long)(E2BIG));
#endif
#ifdef EACCES
printf("EACCES %llu\n", (unsigned long long)(long long)(EACCES));
#endif
#ifdef EADDRINUSE
printf("EADDRINUSE %llu\n", (unsigned long long)(long long)(EADDRINUSE));
#endif
#ifdef EADDRNOTAVAIL
printf("EADDRNOTAVAIL %llu\n", (unsigned long long)(long long)(EADDRNOTAVAIL));
#endif
#ifdef EADV
printf("EADV %llu\n", (unsigned long long)(long long)(EADV));
#endif
#ifdef EAFNOSUPPORT
printf("EAFNOSUPPORT %llu\n", (unsigned long long)(long long)(EAFNOSUPPORT));
#endif
#ifdef EAGAIN
printf("EAGAIN %llu\n", (unsigned long long)(long long)(EAGAIN));
#endif
#ifdef EAI_AGAIN
printf("EAI_AGAIN %llu\n", (unsigned long long)(long long)(EAI_AGAIN));
#endif
#ifdef EAI_BADFLAGS
printf("EAI_BADFLAGS %llu\n", (unsigned long long)(long long)(EAI_BADFLAGS));
#endif
#ifdef EAI_FAIL
printf("EAI_FAIL %llu\n", (unsigned long long)(long long)(EAI_FAIL));
#endif
#ifdef EAI_FAMILY
printf("EAI_FAMILY %llu\n", (unsigned long long)(long long)(EAI_FAMILY));
#endif
#ifdef EAI_MEMORY
printf("EAI_MEMORY %llu\n", (unsigned long long)(long long)(EAI_MEMORY));
#endif
#ifdef EAI_NODATA
printf("EAI_NODATA %llu\n", (unsigned long long)(long long)(EAI_NODATA));
#endif
#ifdef EAI_NONAME
printf("EAI_NONAME %llu\n", (unsigned long long)(long long)(EAI_NONAME));
#endif
#ifdef EAI_OVERFLOW
printf("EAI_OVERFLOW %llu\n", (unsigned long long)(long long)(EAI_OVERFLOW));
#endif
#ifdef EAI_SERVICE
printf("EAI_SERVICE %llu\n", (unsigned long long)(long long)(EAI_SERVICE));
#endif
#ifdef EAI_SOCKTYPE
printf("EAI_SOCKTYPE %llu\n", (unsigned long long)(long long)(EAI_SOCKTYPE));
#endif
#ifdef EAI_SYSTEM
printf("EAI_SYSTEM %llu\n", (unsigned long long)(long long)(EAI_SYSTEM));
#endif
#ifdef EALREADY
printf("EALREADY %llu\n", (unsigned long long)(long long)(EALREADY));
#endif
#ifdef EBADE
printf("EBADE %llu\n", (unsigned long long)(long long)(EBADE));
#endif
#ifdef EBADF
printf("EBADF %llu\n", (unsigned long long)(long long)(EBADF));
#endif
#ifdef EBADFD
printf("EBADFD %llu\n", (unsigned long long)(long long)(EBADFD));
#endif
#ifdef EBADMSG
printf("EBADMSG %llu\n", (unsigned long long)(long long)(EBADMSG));
#endif
#ifdef EBADR
printf("EBADR %llu\n", (unsigned long long)(long long)(EBADR));
#endif
#ifdef EBADRQC
printf("EBADRQC %llu\n", (unsigned long long)(long long)(EBADRQC));
#endif
#ifdef EBADSLT
printf("EBADSLT %llu\n", (unsigned long long)(long long)(EBADSLT));
#endif
#ifdef EBFONT
printf("EBFONT %llu\n", (unsigned long long)(long long)(EBFONT));
#endif
#ifdef EBUSY
printf("EBUSY %llu\n", (unsigned long long)(long long)(EBUSY));
#endif
#ifdef ECANCELED
printf("ECANCELED %llu\n", (unsigned long long)(long long)(ECANCELED));
#endif
#ifdef ECHILD
printf("ECHILD %llu\n", (unsigned long long)(long long)(ECHILD));
#endif
#ifdef ECHO
printf("ECHO %llu\n", (unsigned long long)(long long)(ECHO));
#endif
#ifdef ECHOCTL
printf("ECHOCTL %llu\n", (unsigned long long)(long long)(ECHOCTL));
#endif
#ifdef ECHOE
printf("ECHOE %llu\n", (unsigned long long)(long long)(ECHOE));
#endif
#ifdef ECHOK
printf("ECHOK %llu\n", (unsigned long long)(long long)(ECHOK));
#endif
#ifdef ECHOKE
printf("ECHOKE %llu\n", (unsigned long long)(long long)(ECHOKE));
#endif
#ifdef ECHONL
printf("ECHONL %llu\n", (unsigned long long)(long long)(ECHONL));
#endif
#ifdef ECHOPRT
printf("ECHOPRT %llu\n", (unsigned long long)(long long)(ECHOPRT));
#endif
#ifdef ECHRNG
printf("ECHRNG %llu\n", (unsigned long long)(long long)(ECHRNG));
#endif
#ifdef ECOMM
printf("ECOMM %llu\n", (unsigned long long)(long long)(ECOMM));
#endif
#ifdef ECONNABORTED
printf("ECONNABORTED %llu\n", (unsigned long long)(long long)(ECONNABORTED));
#endif
#ifdef ECONNREFUSED
printf("ECONNREFUSED %llu\n", (unsigned long long)(long long)(ECONNREFUSED));
#endif
#ifdef ECONNRESET
printf("ECONNRESET %llu\n", (unsigned long long)(long long)(ECONNRESET));
#endif
#ifdef ECRYPTFS_SUPER_MAGIC
printf("ECRYPTFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(ECRYPTFS_SUPER_MAGIC));
#endif
#ifdef EDEADLK
printf("EDEADLK %llu\n", (unsigned long long)(long long)(EDEADLK));
#endif
#ifdef EDEADLOCK
printf("EDEADLOCK %llu\n", (unsigned long long)(long long)(EDEADLOCK));
#endif
#ifdef EDESTADDRREQ
printf("EDESTADDRREQ %llu\n", (unsigned long long)(long long)(EDESTADDRREQ));
#endif
#ifdef EDOM
printf("EDOM %llu\n", (unsigned long long)(long long)(EDOM));
#endif
#ifdef EDOTDOT
printf("EDOTDOT %llu\n", (unsigned long long)(long long)(EDOTDOT));
#endif
#ifdef EDQUOT
printf("EDQUOT %llu\n", (unsigned long long)(long long)(EDQUOT));
#endif
#ifdef EEXIST
printf("EEXIST %llu\n", (unsigned long long)(long long)(EEXIST));
#endif
#ifdef EFAULT
printf("EFAULT %llu\n", (unsigned long long)(long long)(EFAULT));
#endif
#ifdef EFBIG
printf("EFBIG %llu\n", (unsigned long long)(long long)(EFBIG));
#endif
#ifdef EFD_CLOEXEC
printf("EFD_CLOEXEC %llu\n", (unsigned long long)(long long)(EFD_CLOEXEC));
#endif
#ifdef EFD_NONBLOCK
printf("EFD_NONBLOCK %llu\n", (unsigned long long)(long long)(EFD_NONBLOCK));
#endif
#ifdef EFD_SEMAPHORE
printf("EFD_SEMAPHORE %llu\n", (unsigned long long)(long long)(EFD_SEMAPHORE));
#endif
#ifdef EFS_SUPER_MAGIC
printf("EFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(EFS_SUPER_MAGIC));
#endif
#ifdef EHOSTDOWN
printf("EHOSTDOWN %llu\n", (unsigned long long)(long long)(EHOSTDOWN));
#endif
#ifdef EHOSTUNREACH
printf("EHOSTUNREACH %llu\n", (unsigned long long)(long long)(EHOSTUNREACH));
#endif
#ifdef EHWPOISON
printf("EHWPOISON %llu\n", (unsigned long long)(long long)(EHWPOISON));
#endif
#ifdef EIDRM
printf("EIDRM %llu\n", (unsigned long long)(long long)(EIDRM));
#endif
#ifdef EILSEQ
printf("EILSEQ %llu\n", (unsigned long long)(long long)(EILSEQ));
#endif
#ifdef EINPROGRESS
printf("EINPROGRESS %llu\n", (unsigned long long)(long long)(EINPROGRESS));
#endif
#ifdef EINTR
printf("EINTR %llu\n", (unsigned long long)(long long)(EINTR));
#endif
#ifdef EINVAL
printf("EINVAL %llu\n", (unsigned long long)(long long)(EINVAL));
#endif
#ifdef EIO
printf("EIO %llu\n", (unsigned long long)(long long)(EIO));
#endif
#ifdef EISCONN
printf("EISCONN %llu\n", (unsigned long long)(long long)(EISCONN));
#endif
#ifdef EISDIR
printf("EISDIR %llu\n", (unsigned long long)(long long)(EISDIR));
#endif
#ifdef EISNAM
printf("EISNAM %llu\n", (unsigned long long)(long long)(EISNAM));
#endif
#ifdef EI_ABIVERSION
printf("EI_ABIVERSION %llu\n", (unsigned long long)(long long)(EI_ABIVERSION));
#endif
#ifdef EI_CLASS
printf("EI_CLASS %llu\n", (unsigned long long)(long long)(EI_CLASS));
#endif
#ifdef EI_DATA
printf("EI_DATA %llu\n", (unsigned long long)(long long)(EI_DATA));
#endif
#ifdef EI_MAG0
printf("EI_MAG0 %llu\n", (unsigned long long)(long long)(EI_MAG0));
#endif
#ifdef EI_MAG1
printf("EI_MAG1 %llu\n", (unsigned long long)(long long)(EI_MAG1));
#endif
#ifdef EI_MAG2
printf("EI_MAG2 %llu\n", (unsigned long long)(long long)(EI_MAG2));
#endif
#ifdef EI_MAG3
printf("EI_MAG3 %llu\n", (unsigned long long)(long long)(EI_MAG3));
#endif
#ifdef EI_NIDENT
printf("EI_NIDENT %llu\n", (unsigned long long)(long long)(EI_NIDENT));
#endif
#ifdef EI_OSABI
printf("EI_OSABI %llu\n", (unsigned long long)(long long)(EI_OSABI));
#endif
#ifdef EI_PAD
printf("EI_PAD %llu\n", (unsigned long long)(long long)(EI_PAD));
#endif
#ifdef EI_VERSION
printf("EI_VERSION %llu\n", (unsigned long long)(long long)(EI_VERSION));
#endif
#ifdef EKEYEXPIRED
printf("EKEYEXPIRED %llu\n", (unsigned long long)(long long)(EKEYEXPIRED));
#endif
#ifdef EKEYREJECTED
printf("EKEYREJECTED %llu\n", (unsigned long long)(long long)(EKEYREJECTED));
#endif
#ifdef EKEYREVOKED
printf("EKEYREVOKED %llu\n", (unsigned long long)(long long)(EKEYREVOKED));
#endif
#ifdef EL2HLT
printf("EL2HLT %llu\n", (unsigned long long)(long long)(EL2HLT));
#endif
#ifdef EL2NSYNC
printf("EL2NSYNC %llu\n", (unsigned long long)(long long)(EL2NSYNC));
#endif
#ifdef EL3HLT
printf("EL3HLT %llu\n", (unsigned long long)(long long)(EL3HLT));
#endif
#ifdef EL3RST
printf("EL3RST %llu\n", (unsigned long long)(long long)(EL3RST));
#endif
#ifdef ELFCLASS32
printf("ELFCLASS32 %llu\n", (unsigned long long)(long long)(ELFCLASS32));
#endif
#ifdef ELFCLASS64
printf("ELFCLASS64 %llu\n", (unsigned long long)(long long)(ELFCLASS64));
#endif
#ifdef ELFCLASSNONE
printf("ELFCLASSNONE %llu\n", (unsigned long long)(long long)(ELFCLASSNONE));
#endif
#ifdef ELFCLASSNUM
printf("ELFCLASSNUM %llu\n", (unsigned long long)(long long)(ELFCLASSNUM));
#endif
#ifdef ELFDATA2LSB
printf("ELFDATA2LSB %llu\n", (unsigned long long)(long long)(ELFDATA2LSB));
#endif
#ifdef ELFDATA2MSB
printf("ELFDATA2MSB %llu\n", (unsigned long long)(long long)(ELFDATA2MSB));
#endif
#ifdef ELFDATANONE
printf("ELFDATANONE %llu\n", (unsigned long long)(long long)(ELFDATANONE));
#endif
#ifdef ELFDATANUM
printf("ELFDATANUM %llu\n", (unsigned long long)(long long)(ELFDATANUM));
#endif
#ifdef ELFMAG0
printf("ELFMAG0 %llu\n", (unsigned long long)(long long)(ELFMAG0));
#endif
#ifdef ELFMAG1
printf("ELFMAG1 %llu\n", (unsigned long long)(long long)(ELFMAG1));
#endif
#ifdef ELFMAG2
printf("ELFMAG2 %llu\n", (unsigned long long)(long long)(ELFMAG2));
#endif
#ifdef ELFMAG3
printf("ELFMAG3 %llu\n", (unsigned long long)(long long)(ELFMAG3));
#endif
#ifdef ELFOSABI_AIX
printf("ELFOSABI_AIX %llu\n", (unsigned long long)(long long)(ELFOSABI_AIX));
#endif
#ifdef ELFOSABI_ARM
printf("ELFOSABI_ARM %llu\n", (unsigned long long)(long long)(ELFOSABI_ARM));
#endif
#ifdef ELFOSABI_ARM_AEABI
printf("ELFOSABI_ARM_AEABI %llu\n", (unsigned long long)(long long)(ELFOSABI_ARM_AEABI));
#endif
#ifdef ELFOSABI_FREEBSD
printf("ELFOSABI_FREEBSD %llu\n", (unsigned long long)(long long)(ELFOSABI_FREEBSD));
#endif
#ifdef ELFOSABI_GNU
printf("ELFOSABI_GNU %llu\n", (unsigned long long)(long long)(ELFOSABI_GNU));
#endif
#ifdef ELFOSABI_HPUX
printf("ELFOSABI_HPUX %llu\n", (unsigned long long)(long long)(ELFOSABI_HPUX));
#endif
#ifdef ELFOSABI_IRIX
printf("ELFOSABI_IRIX %llu\n", (unsigned long long)(long long)(ELFOSABI_IRIX));
#endif
#ifdef ELFOSABI_LINUX
printf("ELFOSABI_LINUX %llu\n", (unsigned long long)(long long)(ELFOSABI_LINUX));
#endif
#ifdef ELFOSABI_MODESTO
printf("ELFOSABI_MODESTO %llu\n", (unsigned long long)(long long)(ELFOSABI_MODESTO));
#endif
#ifdef ELFOSABI_NETBSD
printf("ELFOSABI_NETBSD %llu\n", (unsigned long long)(long long)(ELFOSABI_NETBSD));
#endif
#ifdef ELFOSABI_NONE
printf("ELFOSABI_NONE %llu\n", (unsigned long long)(long long)(ELFOSABI_NONE));
#endif
#ifdef ELFOSABI_OPENBSD
printf("ELFOSABI_OPENBSD %llu\n", (unsigned long long)(long long)(ELFOSABI_OPENBSD));
#endif
#ifdef ELFOSABI_SOLARIS
printf("ELFOSABI_SOLARIS %llu\n", (unsigned long long)(long long)(ELFOSABI_SOLARIS));
#endif
#ifdef ELFOSABI_STANDALONE
printf("ELFOSABI_STANDALONE %llu\n", (unsigned long long)(long long)(ELFOSABI_STANDALONE));
#endif
#ifdef ELFOSABI_SYSV
printf("ELFOSABI_SYSV %llu\n", (unsigned long long)(long long)(ELFOSABI_SYSV));
#endif
#ifdef ELFOSABI_TRU64
printf("ELFOSABI_TRU64 %llu\n", (unsigned long long)(long long)(ELFOSABI_TRU64));
#endif
#ifdef ELIBACC
printf("ELIBACC %llu\n", (unsigned long long)(long long)(ELIBACC));
#endif
#ifdef ELIBBAD
printf("ELIBBAD %llu\n", (unsigned long long)(long long)(ELIBBAD));
#endif
#ifdef ELIBEXEC
printf("ELIBEXEC %llu\n", (unsigned long long)(long long)(ELIBEXEC));
#endif
#ifdef ELIBMAX
printf("ELIBMAX %llu\n", (unsigned long long)(long long)(ELIBMAX));
#endif
#ifdef ELIBSCN
printf("ELIBSCN %llu\n", (unsigned long long)(long long)(ELIBSCN));
#endif
#ifdef ELNRNG
printf("ELNRNG %llu\n", (unsigned long long)(long long)(ELNRNG));
#endif
#ifdef ELOOP
printf("ELOOP %llu\n", (unsigned long long)(long long)(ELOOP));
#endif
#ifdef EMEDIUMTYPE
printf("EMEDIUMTYPE %llu\n", (unsigned long long)(long long)(EMEDIUMTYPE));
#endif
#ifdef EMFILE
printf("EMFILE %llu\n", (unsigned long long)(long long)(EMFILE));
#endif
#ifdef EMLINK
printf("EMLINK %llu\n", (unsigned long long)(long long)(EMLINK));
#endif
#ifdef EMPTY
printf("EMPTY %llu\n", (unsigned long long)(long long)(EMPTY));
#endif
#ifdef EMSGSIZE
printf("EMSGSIZE %llu\n", (unsigned long long)(long long)(EMSGSIZE));
#endif
#ifdef EMULTIHOP
printf("EMULTIHOP %llu\n", (unsigned long long)(long long)(EMULTIHOP));
#endif
#ifdef EM_386
printf("EM_386 %llu\n", (unsigned long long)(long long)(EM_386));
#endif
#ifdef EM_68HC05
printf("EM_68HC05 %llu\n", (unsigned long long)(long long)(EM_68HC05));
#endif
#ifdef EM_68HC08
printf("EM_68HC08 %llu\n", (unsigned long long)(long long)(EM_68HC08));
#endif
#ifdef EM_68HC11
printf("EM_68HC11 %llu\n", (unsigned long long)(long long)(EM_68HC11));
#endif
#ifdef EM_68HC12
printf("EM_68HC12 %llu\n", (unsigned long long)(long long)(EM_68HC12));
#endif
#ifdef EM_68HC16
printf("EM_68HC16 %llu\n", (unsigned long long)(long long)(EM_68HC16));
#endif
#ifdef EM_68K
printf("EM_68K %llu\n", (unsigned long long)(long long)(EM_68K));
#endif
#ifdef EM_860
printf("EM_860 %llu\n", (unsigned long long)(long long)(EM_860));
#endif
#ifdef EM_88K
printf("EM_88K %llu\n", (unsigned long long)(long long)(EM_88K));
#endif
#ifdef EM_960
printf("EM_960 %llu\n", (unsigned long long)(long long)(EM_960));
#endif
#ifdef EM_AARCH64
printf("EM_AARCH64 %llu\n", (unsigned long long)(long long)(EM_AARCH64));
#endif
#ifdef EM_ALPHA
printf("EM_ALPHA %llu\n", (unsigned long long)(long long)(EM_ALPHA));
#endif
#ifdef EM_ARC
printf("EM_ARC %llu\n", (unsigned long long)(long long)(EM_ARC));
#endif
#ifdef EM_ARC_A5
printf("EM_ARC_A5 %llu\n", (unsigned long long)(long long)(EM_ARC_A5));
#endif
#ifdef EM_ARM
printf("EM_ARM %llu\n", (unsigned long long)(long long)(EM_ARM));
#endif
#ifdef EM_AVR
printf("EM_AVR %llu\n", (unsigned long long)(long long)(EM_AVR));
#endif
#ifdef EM_COLDFIRE
printf("EM_COLDFIRE %llu\n", (unsigned long long)(long long)(EM_COLDFIRE));
#endif
#ifdef EM_CRIS
printf("EM_CRIS %llu\n", (unsigned long long)(long long)(EM_CRIS));
#endif
#ifdef EM_D10V
printf("EM_D10V %llu\n", (unsigned long long)(long long)(EM_D10V));
#endif
#ifdef EM_D30V
printf("EM_D30V %llu\n", (unsigned long long)(long long)(EM_D30V));
#endif
#ifdef EM_FAKE_ALPHA
printf("EM_FAKE_ALPHA %llu\n", (unsigned long long)(long long)(EM_FAKE_ALPHA));
#endif
#ifdef EM_FIREPATH
printf("EM_FIREPATH %llu\n", (unsigned long long)(long long)(EM_FIREPATH));
#endif
#ifdef EM_FR20
printf("EM_FR20 %llu\n", (unsigned long long)(long long)(EM_FR20));
#endif
#ifdef EM_FR30
printf("EM_FR30 %llu\n", (unsigned long long)(long long)(EM_FR30));
#endif
#ifdef EM_FX66
printf("EM_FX66 %llu\n", (unsigned long long)(long long)(EM_FX66));
#endif
#ifdef EM_H8S
printf("EM_H8S %llu\n", (unsigned long long)(long long)(EM_H8S));
#endif
#ifdef EM_H8_300
printf("EM_H8_300 %llu\n", (unsigned long long)(long long)(EM_H8_300));
#endif
#ifdef EM_H8_300H
printf("EM_H8_300H %llu\n", (unsigned long long)(long long)(EM_H8_300H));
#endif
#ifdef EM_H8_500
printf("EM_H8_500 %llu\n", (unsigned long long)(long long)(EM_H8_500));
#endif
#ifdef EM_HUANY
printf("EM_HUANY %llu\n", (unsigned long long)(long long)(EM_HUANY));
#endif
#ifdef EM_IA_64
printf("EM_IA_64 %llu\n", (unsigned long long)(long long)(EM_IA_64));
#endif
#ifdef EM_JAVELIN
printf("EM_JAVELIN %llu\n", (unsigned long long)(long long)(EM_JAVELIN));
#endif
#ifdef EM_M32
printf("EM_M32 %llu\n", (unsigned long long)(long long)(EM_M32));
#endif
#ifdef EM_M32R
printf("EM_M32R %llu\n", (unsigned long long)(long long)(EM_M32R));
#endif
#ifdef EM_ME16
printf("EM_ME16 %llu\n", (unsigned long long)(long long)(EM_ME16));
#endif
#ifdef EM_MIPS
printf("EM_MIPS %llu\n", (unsigned long long)(long long)(EM_MIPS));
#endif
#ifdef EM_MIPS_RS3_LE
printf("EM_MIPS_RS3_LE %llu\n", (unsigned long long)(long long)(EM_MIPS_RS3_LE));
#endif
#ifdef EM_MIPS_X
printf("EM_MIPS_X %llu\n", (unsigned long long)(long long)(EM_MIPS_X));
#endif
#ifdef EM_MMA
printf("EM_MMA %llu\n", (unsigned long long)(long long)(EM_MMA));
#endif
#ifdef EM_MMIX
printf("EM_MMIX %llu\n", (unsigned long long)(long long)(EM_MMIX));
#endif
#ifdef EM_MN10200
printf("EM_MN10200 %llu\n", (unsigned long long)(long long)(EM_MN10200));
#endif
#ifdef EM_MN10300
printf("EM_MN10300 %llu\n", (unsigned long long)(long long)(EM_MN10300));
#endif
#ifdef EM_NCPU
printf("EM_NCPU %llu\n", (unsigned long long)(long long)(EM_NCPU));
#endif
#ifdef EM_NDR1
printf("EM_NDR1 %llu\n", (unsigned long long)(long long)(EM_NDR1));
#endif
#ifdef EM_NONE
printf("EM_NONE %llu\n", (unsigned long long)(long long)(EM_NONE));
#endif
#ifdef EM_OPENRISC
printf("EM_OPENRISC %llu\n", (unsigned long long)(long long)(EM_OPENRISC));
#endif
#ifdef EM_PARISC
printf("EM_PARISC %llu\n", (unsigned long long)(long long)(EM_PARISC));
#endif
#ifdef EM_PCP
printf("EM_PCP %llu\n", (unsigned long long)(long long)(EM_PCP));
#endif
#ifdef EM_PDSP
printf("EM_PDSP %llu\n", (unsigned long long)(long long)(EM_PDSP));
#endif
#ifdef EM_PJ
printf("EM_PJ %llu\n", (unsigned long long)(long long)(EM_PJ));
#endif
#ifdef EM_PPC
printf("EM_PPC %llu\n", (unsigned long long)(long long)(EM_PPC));
#endif
#ifdef EM_PPC64
printf("EM_PPC64 %llu\n", (unsigned long long)(long long)(EM_PPC64));
#endif
#ifdef EM_PRISM
printf("EM_PRISM %llu\n", (unsigned long long)(long long)(EM_PRISM));
#endif
#ifdef EM_RCE
printf("EM_RCE %llu\n", (unsigned long long)(long long)(EM_RCE));
#endif
#ifdef EM_RH32
printf("EM_RH32 %llu\n", (unsigned long long)(long long)(EM_RH32));
#endif
#ifdef EM_RISCV
printf("EM_RISCV %llu\n", (unsigned long long)(long long)(EM_RISCV));
#endif
#ifdef EM_S370
printf("EM_S370 %llu\n", (unsigned long long)(long long)(EM_S370));
#endif
#ifdef EM_S390
printf("EM_S390 %llu\n", (unsigned long long)(long long)(EM_S390));
#endif
#ifdef EM_SH
printf("EM_SH %llu\n", (unsigned long long)(long long)(EM_SH));
#endif
#ifdef EM_SPARC
printf("EM_SPARC %llu\n", (unsigned long long)(long long)(EM_SPARC));
#endif
#ifdef EM_SPARC32PLUS
printf("EM_SPARC32PLUS %llu\n", (unsigned long long)(long long)(EM_SPARC32PLUS));
#endif
#ifdef EM_SPARCV9
printf("EM_SPARCV9 %llu\n", (unsigned long long)(long long)(EM_SPARCV9));
#endif
#ifdef EM_ST100
printf("EM_ST100 %llu\n", (unsigned long long)(long long)(EM_ST100));
#endif
#ifdef EM_ST19
printf("EM_ST19 %llu\n", (unsigned long long)(long long)(EM_ST19));
#endif
#ifdef EM_ST7
printf("EM_ST7 %llu\n", (unsigned long long)(long long)(EM_ST7));
#endif
#ifdef EM_ST9PLUS
printf("EM_ST9PLUS %llu\n", (unsigned long long)(long long)(EM_ST9PLUS));
#endif
#ifdef EM_STARCORE
printf("EM_STARCORE %llu\n", (unsigned long long)(long long)(EM_STARCORE));
#endif
#ifdef EM_SVX
printf("EM_SVX %llu\n", (unsigned long long)(long long)(EM_SVX));
#endif
#ifdef EM_TILEGX
printf("EM_TILEGX %llu\n", (unsigned long long)(long long)(EM_TILEGX));
#endif
#ifdef EM_TILEPRO
printf("EM_TILEPRO %llu\n", (unsigned long long)(long long)(EM_TILEPRO));
#endif
#ifdef EM_TINYJ
printf("EM_TINYJ %llu\n", (unsigned long long)(long long)(EM_TINYJ));
#endif
#ifdef EM_TRICORE
printf("EM_TRICORE %llu\n", (unsigned long long)(long long)(EM_TRICORE));
#endif
#ifdef EM_V800
printf("EM_V800 %llu\n", (unsigned long long)(long long)(EM_V800));
#endif
#ifdef EM_V850
printf("EM_V850 %llu\n", (unsigned long long)(long long)(EM_V850));
#endif
#ifdef EM_VAX
printf("EM_VAX %llu\n", (unsigned long long)(long long)(EM_VAX));
#endif
#ifdef EM_VPP500
printf("EM_VPP500 %llu\n", (unsigned long long)(long long)(EM_VPP500));
#endif
#ifdef EM_X86_64
printf("EM_X86_64 %llu\n", (unsigned long long)(long long)(EM_X86_64));
#endif
#ifdef EM_XTENSA
printf("EM_XTENSA %llu\n", (unsigned long long)(long long)(EM_XTENSA));
#endif
#ifdef EM_ZSP
printf("EM_ZSP %llu\n", (unsigned long long)(long long)(EM_ZSP));
#endif
#ifdef ENAMETOOLONG
printf("ENAMETOOLONG %llu\n", (unsigned long long)(long long)(ENAMETOOLONG));
#endif
#ifdef ENAVAIL
printf("ENAVAIL %llu\n", (unsigned long long)(long long)(ENAVAIL));
#endif
#ifdef ENETDOWN
printf("ENETDOWN %llu\n", (unsigned long long)(long long)(ENETDOWN));
#endif
#ifdef ENETRESET
printf("ENETRESET %llu\n", (unsigned long long)(long long)(ENETRESET));
#endif
#ifdef ENETUNREACH
printf("ENETUNREACH %llu\n", (unsigned long long)(long long)(ENETUNREACH));
#endif
#ifdef ENFILE
printf("ENFILE %llu\n", (unsigned long long)(long long)(ENFILE));
#endif
#ifdef ENOANO
printf("ENOANO %llu\n", (unsigned long long)(long long)(ENOANO));
#endif
#ifdef ENOATTR
printf("ENOATTR %llu\n", (unsigned long long)(long long)(ENOATTR));
#endif
#ifdef ENOBUFS
printf("ENOBUFS %llu\n", (unsigned long long)(long long)(ENOBUFS));
#endif
#ifdef ENOCSI
printf("ENOCSI %llu\n", (unsigned long long)(long long)(ENOCSI));
#endif
#ifdef ENODATA
printf("ENODATA %llu\n", (unsigned long long)(long long)(ENODATA));
#endif
#ifdef ENODEV
printf("ENODEV %llu\n", (unsigned long long)(long long)(ENODEV));
#endif
#ifdef ENOENT
printf("ENOENT %llu\n", (unsigned long long)(long long)(ENOENT));
#endif
#ifdef ENOEXEC
printf("ENOEXEC %llu\n", (unsigned long long)(long long)(ENOEXEC));
#endif
#ifdef ENOKEY
printf("ENOKEY %llu\n", (unsigned long long)(long long)(ENOKEY));
#endif
#ifdef ENOLCK
printf("ENOLCK %llu\n", (unsigned long long)(long long)(ENOLCK));
#endif
#ifdef ENOLINK
printf("ENOLINK %llu\n", (unsigned long long)(long long)(ENOLINK));
#endif
#ifdef ENOMEDIUM
printf("ENOMEDIUM %llu\n", (unsigned long long)(long long)(ENOMEDIUM));
#endif
#ifdef ENOMEM
printf("ENOMEM %llu\n", (unsigned long long)(long long)(ENOMEM));
#endif
#ifdef ENOMSG
printf("ENOMSG %llu\n", (unsigned long long)(long long)(ENOMSG));
#endif
#ifdef ENONET
printf("ENONET %llu\n", (unsigned long long)(long long)(ENONET));
#endif
#ifdef ENOPKG
printf("ENOPKG %llu\n", (unsigned long long)(long long)(ENOPKG));
#endif
#ifdef ENOPROTOOPT
printf("ENOPROTOOPT %llu\n", (unsigned long long)(long long)(ENOPROTOOPT));
#endif
#ifdef ENOSPC
printf("ENOSPC %llu\n", (unsigned long long)(long long)(ENOSPC));
#endif
#ifdef ENOSR
printf("ENOSR %llu\n", (unsigned long long)(long long)(ENOSR));
#endif
#ifdef ENOSTR
printf("ENOSTR %llu\n", (unsigned long long)(long long)(ENOSTR));
#endif
#ifdef ENOSYS
printf("ENOSYS %llu\n", (unsigned long long)(long long)(ENOSYS));
#endif
#ifdef ENOTBLK
printf("ENOTBLK %llu\n", (unsigned long long)(long long)(ENOTBLK));
#endif
#ifdef ENOTCONN
printf("ENOTCONN %llu\n", (unsigned long long)(long long)(ENOTCONN));
#endif
#ifdef ENOTDIR
printf("ENOTDIR %llu\n", (unsigned long long)(long long)(ENOTDIR));
#endif
#ifdef ENOTEMPTY
printf("ENOTEMPTY %llu\n", (unsigned long long)(long long)(ENOTEMPTY));
#endif
#ifdef ENOTNAM
printf("ENOTNAM %llu\n", (unsigned long long)(long long)(ENOTNAM));
#endif
#ifdef ENOTRECOVERABLE
printf("ENOTRECOVERABLE %llu\n", (unsigned long long)(long long)(ENOTRECOVERABLE));
#endif
#ifdef ENOTSOCK
printf("ENOTSOCK %llu\n", (unsigned long long)(long long)(ENOTSOCK));
#endif
#ifdef ENOTSUP
printf("ENOTSUP %llu\n", (unsigned long long)(long long)(ENOTSUP));
#endif
#ifdef ENOTTY
printf("ENOTTY %llu\n", (unsigned long long)(long long)(ENOTTY));
#endif
#ifdef ENOTUNIQ
printf("ENOTUNIQ %llu\n", (unsigned long long)(long long)(ENOTUNIQ));
#endif
#ifdef ENXIO
printf("ENXIO %llu\n", (unsigned long long)(long long)(ENXIO));
#endif
#ifdef EOF
printf("EOF %llu\n", (unsigned long long)(long long)(EOF));
#endif
#ifdef EOPNOTSUPP
printf("EOPNOTSUPP %llu\n", (unsigned long long)(long long)(EOPNOTSUPP));
#endif
#ifdef EOVERFLOW
printf("EOVERFLOW %llu\n", (unsigned long long)(long long)(EOVERFLOW));
#endif
#ifdef EOWNERDEAD
printf("EOWNERDEAD %llu\n", (unsigned long long)(long long)(EOWNERDEAD));
#endif
#ifdef EPERM
printf("EPERM %llu\n", (unsigned long long)(long long)(EPERM));
#endif
#ifdef EPFNOSUPPORT
printf("EPFNOSUPPORT %llu\n", (unsigned long long)(long long)(EPFNOSUPPORT));
#endif
#ifdef EPIOCGPARAMS
printf("EPIOCGPARAMS %llu\n", (unsigned long long)(long long)(EPIOCGPARAMS));
#endif
#ifdef EPIOCSPARAMS
printf("EPIOCSPARAMS %llu\n", (unsigned long long)(long long)(EPIOCSPARAMS));
#endif
#ifdef EPIPE
printf("EPIPE %llu\n", (unsigned long long)(long long)(EPIPE));
#endif
#ifdef EPOLLERR
printf("EPOLLERR %llu\n", (unsigned long long)(long long)(EPOLLERR));
#endif
#ifdef EPOLLET
printf("EPOLLET %llu\n", (unsigned long long)(long long)(EPOLLET));
#endif
#ifdef EPOLLEXCLUSIVE
printf("EPOLLEXCLUSIVE %llu\n", (unsigned long long)(long long)(EPOLLEXCLUSIVE));
#endif
#ifdef EPOLLHUP
printf("EPOLLHUP %llu\n", (unsigned long long)(long long)(EPOLLHUP));
#endif
#ifdef EPOLLIN
printf("EPOLLIN %llu\n", (unsigned long long)(long long)(EPOLLIN));
#endif
#ifdef EPOLLMSG
printf("EPOLLMSG %llu\n", (unsigned long long)(long long)(EPOLLMSG));
#endif
#ifdef EPOLLONESHOT
printf("EPOLLONESHOT %llu\n", (unsigned long long)(long long)(EPOLLONESHOT));
#endif
#ifdef EPOLLOUT
printf("EPOLLOUT %llu\n", (unsigned long long)(long long)(EPOLLOUT));
#endif
#ifdef EPOLLPRI
printf("EPOLLPRI %llu\n", (unsigned long long)(long long)(EPOLLPRI));
#endif
#ifdef EPOLLRDBAND
printf("EPOLLRDBAND %llu\n", (unsigned long long)(long long)(EPOLLRDBAND));
#endif
#ifdef EPOLLRDHUP
printf("EPOLLRDHUP %llu\n", (unsigned long long)(long long)(EPOLLRDHUP));
#endif
#ifdef EPOLLRDNORM
printf("EPOLLRDNORM %llu\n", (unsigned long long)(long long)(EPOLLRDNORM));
#endif
#ifdef EPOLLWAKEUP
printf("EPOLLWAKEUP %llu\n", (unsigned long long)(long long)(EPOLLWAKEUP));
#endif
#ifdef EPOLLWRBAND
printf("EPOLLWRBAND %llu\n", (unsigned long long)(long long)(EPOLLWRBAND));
#endif
#ifdef EPOLLWRNORM
printf("EPOLLWRNORM %llu\n", (unsigned long long)(long long)(EPOLLWRNORM));
#endif
#ifdef EPOLL_CLOEXEC
printf("EPOLL_CLOEXEC %llu\n", (unsigned long long)(long long)(EPOLL_CLOEXEC));
#endif
#ifdef EPOLL_CTL_ADD
printf("EPOLL_CTL_ADD %llu\n", (unsigned long long)(long long)(EPOLL_CTL_ADD));
#endif
#ifdef EPOLL_CTL_DEL
printf("EPOLL_CTL_DEL %llu\n", (unsigned long long)(long long)(EPOLL_CTL_DEL));
#endif
#ifdef EPOLL_CTL_MOD
printf("EPOLL_CTL_MOD %llu\n", (unsigned long long)(long long)(EPOLL_CTL_MOD));
#endif
#ifdef EPROTO
printf("EPROTO %llu\n", (unsigned long long)(long long)(EPROTO));
#endif
#ifdef EPROTONOSUPPORT
printf("EPROTONOSUPPORT %llu\n", (unsigned long long)(long long)(EPROTONOSUPPORT));
#endif
#ifdef EPROTOTYPE
printf("EPROTOTYPE %llu\n", (unsigned long long)(long long)(EPROTOTYPE));
#endif
#ifdef ERA
printf("ERA %llu\n", (unsigned long long)(long long)(ERA));
#endif
#ifdef ERANGE
printf("ERANGE %llu\n", (unsigned long long)(long long)(ERANGE));
#endif
#ifdef ERA_D_FMT
printf("ERA_D_FMT %llu\n", (unsigned long long)(long long)(ERA_D_FMT));
#endif
#ifdef ERA_D_T_FMT
printf("ERA_D_T_FMT %llu\n", (unsigned long long)(long long)(ERA_D_T_FMT));
#endif
#ifdef ERA_T_FMT
printf("ERA_T_FMT %llu\n", (unsigned long long)(long long)(ERA_T_FMT));
#endif
#ifdef EREMCHG
printf("EREMCHG %llu\n", (unsigned long long)(long long)(EREMCHG));
#endif
#ifdef EREMOTE
printf("EREMOTE %llu\n", (unsigned long long)(long long)(EREMOTE));
#endif
#ifdef EREMOTEIO
printf("EREMOTEIO %llu\n", (unsigned long long)(long long)(EREMOTEIO));
#endif
#ifdef ERESTART
printf("ERESTART %llu\n", (unsigned long long)(long long)(ERESTART));
#endif
#ifdef ERFKILL
printf("ERFKILL %llu\n", (unsigned long long)(long long)(ERFKILL));
#endif
#ifdef EROFS
printf("EROFS %llu\n", (unsigned long long)(long long)(EROFS));
#endif
#ifdef ESHUTDOWN
printf("ESHUTDOWN %llu\n", (unsigned long long)(long long)(ESHUTDOWN));
#endif
#ifdef ESOCKTNOSUPPORT
printf("ESOCKTNOSUPPORT %llu\n", (unsigned long long)(long long)(ESOCKTNOSUPPORT));
#endif
#ifdef ESPIPE
printf("ESPIPE %llu\n", (unsigned long long)(long long)(ESPIPE));
#endif
#ifdef ESRCH
printf("ESRCH %llu\n", (unsigned long long)(long long)(ESRCH));
#endif
#ifdef ESRMNT
printf("ESRMNT %llu\n", (unsigned long long)(long long)(ESRMNT));
#endif
#ifdef ESTALE
printf("ESTALE %llu\n", (unsigned long long)(long long)(ESTALE));
#endif
#ifdef ESTRPIPE
printf("ESTRPIPE %llu\n", (unsigned long long)(long long)(ESTRPIPE));
#endif
#ifdef ETH_ALEN
printf("ETH_ALEN %llu\n", (unsigned long long)(long long)(ETH_ALEN));
#endif
#ifdef ETH_DATA_LEN
printf("ETH_DATA_LEN %llu\n", (unsigned long long)(long long)(ETH_DATA_LEN));
#endif
#ifdef ETH_FCS_LEN
printf("ETH_FCS_LEN %llu\n", (unsigned long long)(long long)(ETH_FCS_LEN));
#endif
#ifdef ETH_FRAME_LEN
printf("ETH_FRAME_LEN %llu\n", (unsigned long long)(long long)(ETH_FRAME_LEN));
#endif
#ifdef ETH_HLEN
printf("ETH_HLEN %llu\n", (unsigned long long)(long long)(ETH_HLEN));
#endif
#ifdef ETH_P_1588
printf("ETH_P_1588 %llu\n", (unsigned long long)(long long)(ETH_P_1588));
#endif
#ifdef ETH_P_8021AD
printf("ETH_P_8021AD %llu\n", (unsigned long long)(long long)(ETH_P_8021AD));
#endif
#ifdef ETH_P_8021AH
printf("ETH_P_8021AH %llu\n", (unsigned long long)(long long)(ETH_P_8021AH));
#endif
#ifdef ETH_P_8021Q
printf("ETH_P_8021Q %llu\n", (unsigned long long)(long long)(ETH_P_8021Q));
#endif
#ifdef ETH_P_80221
printf("ETH_P_80221 %llu\n", (unsigned long long)(long long)(ETH_P_80221));
#endif
#ifdef ETH_P_802_2
printf("ETH_P_802_2 %llu\n", (unsigned long long)(long long)(ETH_P_802_2));
#endif
#ifdef ETH_P_802_3
printf("ETH_P_802_3 %llu\n", (unsigned long long)(long long)(ETH_P_802_3));
#endif
#ifdef ETH_P_802_3_MIN
printf("ETH_P_802_3_MIN %llu\n", (unsigned long long)(long long)(ETH_P_802_3_MIN));
#endif
#ifdef ETH_P_802_EX1
printf("ETH_P_802_EX1 %llu\n", (unsigned long long)(long long)(ETH_P_802_EX1));
#endif
#ifdef ETH_P_AARP
printf("ETH_P_AARP %llu\n", (unsigned long long)(long long)(ETH_P_AARP));
#endif
#ifdef ETH_P_AF_IUCV
printf("ETH_P_AF_IUCV %llu\n", (unsigned long long)(long long)(ETH_P_AF_IUCV));
#endif
#ifdef ETH_P_ALL
printf("ETH_P_ALL %llu\n", (unsigned long long)(long long)(ETH_P_ALL));
#endif
#ifdef ETH_P_AOE
printf("ETH_P_AOE %llu\n", (unsigned long long)(long long)(ETH_P_AOE));
#endif
#ifdef ETH_P_ARCNET
printf("ETH_P_ARCNET %llu\n", (unsigned long long)(long long)(ETH_P_ARCNET));
#endif
#ifdef ETH_P_ARP
printf("ETH_P_ARP %llu\n", (unsigned long long)(long long)(ETH_P_ARP));
#endif
#ifdef ETH_P_ATALK
printf("ETH_P_ATALK %llu\n", (unsigned long long)(long long)(ETH_P_ATALK));
#endif
#ifdef ETH_P_ATMFATE
printf("ETH_P_ATMFATE %llu\n", (unsigned long long)(long long)(ETH_P_ATMFATE));
#endif
#ifdef ETH_P_ATMMPOA
printf("ETH_P_ATMMPOA %llu\n", (unsigned long long)(long long)(ETH_P_ATMMPOA));
#endif
#ifdef ETH_P_AX25
printf("ETH_P_AX25 %llu\n", (unsigned long long)(long long)(ETH_P_AX25));
#endif
#ifdef ETH_P_BATMAN
printf("ETH_P_BATMAN %llu\n", (unsigned long long)(long long)(ETH_P_BATMAN));
#endif
#ifdef ETH_P_BPQ
printf("ETH_P_BPQ %llu\n", (unsigned long long)(long long)(ETH_P_BPQ));
#endif
#ifdef ETH_P_CAIF
printf("ETH_P_CAIF %llu\n", (unsigned long long)(long long)(ETH_P_CAIF));
#endif
#ifdef ETH_P_CANFD
printf("ETH_P_CANFD %llu\n", (unsigned long long)(long long)(ETH_P_CANFD));
#endif
#ifdef ETH_P_CONTROL
printf("ETH_P_CONTROL %llu\n", (unsigned long long)(long long)(ETH_P_CONTROL));
#endif
#ifdef ETH_P_CUST
printf("ETH_P_CUST %llu\n", (unsigned long long)(long long)(ETH_P_CUST));
#endif
#ifdef ETH_P_DDCMP
printf("ETH_P_DDCMP %llu\n", (unsigned long long)(long long)(ETH_P_DDCMP));
#endif
#ifdef ETH_P_DEC
printf("ETH_P_DEC %llu\n", (unsigned long long)(long long)(ETH_P_DEC));
#endif
#ifdef ETH_P_DIAG
printf("ETH_P_DIAG %llu\n", (unsigned long long)(long long)(ETH_P_DIAG));
#endif
#ifdef ETH_P_DNA_DL
printf("ETH_P_DNA_DL %llu\n", (unsigned long long)(long long)(ETH_P_DNA_DL));
#endif
#ifdef ETH_P_DNA_RC
printf("ETH_P_DNA_RC %llu\n", (unsigned long long)(long long)(ETH_P_DNA_RC));
#endif
#ifdef ETH_P_DNA_RT
printf("ETH_P_DNA_RT %llu\n", (unsigned long long)(long long)(ETH_P_DNA_RT));
#endif
#ifdef ETH_P_DSA
printf("ETH_P_DSA %llu\n", (unsigned long long)(long long)(ETH_P_DSA));
#endif
#ifdef ETH_P_ECONET
printf("ETH_P_ECONET %llu\n", (unsigned long long)(long long)(ETH_P_ECONET));
#endif
#ifdef ETH_P_EDSA
printf("ETH_P_EDSA %llu\n", (unsigned long long)(long long)(ETH_P_EDSA));
#endif
#ifdef ETH_P_FCOE
printf("ETH_P_FCOE %llu\n", (unsigned long long)(long long)(ETH_P_FCOE));
#endif
#ifdef ETH_P_FIP
printf("ETH_P_FIP %llu\n", (unsigned long long)(long long)(ETH_P_FIP));
#endif
#ifdef ETH_P_HDLC
printf("ETH_P_HDLC %llu\n", (unsigned long long)(long long)(ETH_P_HDLC));
#endif
#ifdef ETH_P_IEEE802154
printf("ETH_P_IEEE802154 %llu\n", (unsigned long long)(long long)(ETH_P_IEEE802154));
#endif
#ifdef ETH_P_IEEEPUP
printf("ETH_P_IEEEPUP %llu\n", (unsigned long long)(long long)(ETH_P_IEEEPUP));
#endif
#ifdef ETH_P_IEEEPUPAT
printf("ETH_P_IEEEPUPAT %llu\n", (unsigned long long)(long long)(ETH_P_IEEEPUPAT));
#endif
#ifdef ETH_P_IP
printf("ETH_P_IP %llu\n", (unsigned long long)(long long)(ETH_P_IP));
#endif
#ifdef ETH_P_IPV6
printf("ETH_P_IPV6 %llu\n", (unsigned long long)(long long)(ETH_P_IPV6));
#endif
#ifdef ETH_P_IPX
printf("ETH_P_IPX %llu\n", (unsigned long long)(long long)(ETH_P_IPX));
#endif
#ifdef ETH_P_IRDA
printf("ETH_P_IRDA %llu\n", (unsigned long long)(long long)(ETH_P_IRDA));
#endif
#ifdef ETH_P_LAT
printf("ETH_P_LAT %llu\n", (unsigned long long)(long long)(ETH_P_LAT));
#endif
#ifdef ETH_P_LINK_CTL
printf("ETH_P_LINK_CTL %llu\n", (unsigned long long)(long long)(ETH_P_LINK_CTL));
#endif
#ifdef ETH_P_LOCALTALK
printf("ETH_P_LOCALTALK %llu\n", (unsigned long long)(long long)(ETH_P_LOCALTALK));
#endif
#ifdef ETH_P_LOOP
printf("ETH_P_LOOP %llu\n", (unsigned long long)(long long)(ETH_P_LOOP));
#endif
#ifdef ETH_P_LOOPBACK
printf("ETH_P_LOOPBACK %llu\n", (unsigned long long)(long long)(ETH_P_LOOPBACK));
#endif
#ifdef ETH_P_MACSEC
printf("ETH_P_MACSEC %llu\n", (unsigned long long)(long long)(ETH_P_MACSEC));
#endif
#ifdef ETH_P_MOBITEX
printf("ETH_P_MOBITEX %llu\n", (unsigned long long)(long long)(ETH_P_MOBITEX));
#endif
#ifdef ETH_P_MPLS_MC
printf("ETH_P_MPLS_MC %llu\n", (unsigned long long)(long long)(ETH_P_MPLS_MC));
#endif
#ifdef ETH_P_MPLS_UC
printf("ETH_P_MPLS_UC %llu\n", (unsigned long long)(long long)(ETH_P_MPLS_UC));
#endif
#ifdef ETH_P_MVRP
printf("ETH_P_MVRP %llu\n", (unsigned long long)(long long)(ETH_P_MVRP));
#endif
#ifdef ETH_P_PAE
printf("ETH_P_PAE %llu\n", (unsigned long long)(long long)(ETH_P_PAE));
#endif
#ifdef ETH_P_PAUSE
printf("ETH_P_PAUSE %llu\n", (unsigned long long)(long long)(ETH_P_PAUSE));
#endif
#ifdef ETH_P_PHONET
printf("ETH_P_PHONET %llu\n", (unsigned long long)(long long)(ETH_P_PHONET));
#endif
#ifdef ETH_P_PPPTALK
printf("ETH_P_PPPTALK %llu\n", (unsigned long long)(long long)(ETH_P_PPPTALK));
#endif
#ifdef ETH_P_PPP_DISC
printf("ETH_P_PPP_DISC %llu\n", (unsigned long long)(long long)(ETH_P_PPP_DISC));
#endif
#ifdef ETH_P_PPP_MP
printf("ETH_P_PPP_MP %llu\n", (unsigned long long)(long long)(ETH_P_PPP_MP));
#endif
#ifdef ETH_P_PPP_SES
printf("ETH_P_PPP_SES %llu\n", (unsigned long long)(long long)(ETH_P_PPP_SES));
#endif
#ifdef ETH_P_PRP
printf("ETH_P_PRP %llu\n", (unsigned long long)(long long)(ETH_P_PRP));
#endif
#ifdef ETH_P_PUP
printf("ETH_P_PUP %llu\n", (unsigned long long)(long long)(ETH_P_PUP));
#endif
#ifdef ETH_P_PUPAT
printf("ETH_P_PUPAT %llu\n", (unsigned long long)(long long)(ETH_P_PUPAT));
#endif
#ifdef ETH_P_QINQ1
printf("ETH_P_QINQ1 %llu\n", (unsigned long long)(long long)(ETH_P_QINQ1));
#endif
#ifdef ETH_P_QINQ2
printf("ETH_P_QINQ2 %llu\n", (unsigned long long)(long long)(ETH_P_QINQ2));
#endif
#ifdef ETH_P_QINQ3
printf("ETH_P_QINQ3 %llu\n", (unsigned long long)(long long)(ETH_P_QINQ3));
#endif
#ifdef ETH_P_RARP
printf("ETH_P_RARP %llu\n", (unsigned long long)(long long)(ETH_P_RARP));
#endif
#ifdef ETH_P_SCA
printf("ETH_P_SCA %llu\n", (unsigned long long)(long long)(ETH_P_SCA));
#endif
#ifdef ETH_P_SLOW
printf("ETH_P_SLOW %llu\n", (unsigned long long)(long long)(ETH_P_SLOW));
#endif
#ifdef ETH_P_SNAP
printf("ETH_P_SNAP %llu\n", (unsigned long long)(long long)(ETH_P_SNAP));
#endif
#ifdef ETH_P_TDLS
printf("ETH_P_TDLS %llu\n", (unsigned long long)(long long)(ETH_P_TDLS));
#endif
#ifdef ETH_P_TEB
printf("ETH_P_TEB %llu\n", (unsigned long long)(long long)(ETH_P_TEB));
#endif
#ifdef ETH_P_TIPC
printf("ETH_P_TIPC %llu\n", (unsigned long long)(long long)(ETH_P_TIPC));
#endif
#ifdef ETH_P_TRAILER
printf("ETH_P_TRAILER %llu\n", (unsigned long long)(long long)(ETH_P_TRAILER));
#endif
#ifdef ETH_P_TR_802_2
printf("ETH_P_TR_802_2 %llu\n", (unsigned long long)(long long)(ETH_P_TR_802_2));
#endif
#ifdef ETH_P_WAN_PPP
printf("ETH_P_WAN_PPP %llu\n", (unsigned long long)(long long)(ETH_P_WAN_PPP));
#endif
#ifdef ETH_P_WCCP
printf("ETH_P_WCCP %llu\n", (unsigned long long)(long long)(ETH_P_WCCP));
#endif
#ifdef ETH_P_X25
printf("ETH_P_X25 %llu\n", (unsigned long long)(long long)(ETH_P_X25));
#endif
#ifdef ETH_ZLEN
printf("ETH_ZLEN %llu\n", (unsigned long long)(long long)(ETH_ZLEN));
#endif
#ifdef ETIME
printf("ETIME %llu\n", (unsigned long long)(long long)(ETIME));
#endif
#ifdef ETIMEDOUT
printf("ETIMEDOUT %llu\n", (unsigned long long)(long long)(ETIMEDOUT));
#endif
#ifdef ETOOMANYREFS
printf("ETOOMANYREFS %llu\n", (unsigned long long)(long long)(ETOOMANYREFS));
#endif
#ifdef ETXTBSY
printf("ETXTBSY %llu\n", (unsigned long long)(long long)(ETXTBSY));
#endif
#ifdef ET_CORE
printf("ET_CORE %llu\n", (unsigned long long)(long long)(ET_CORE));
#endif
#ifdef ET_DYN
printf("ET_DYN %llu\n", (unsigned long long)(long long)(ET_DYN));
#endif
#ifdef ET_EXEC
printf("ET_EXEC %llu\n", (unsigned long long)(long long)(ET_EXEC));
#endif
#ifdef ET_HIOS
printf("ET_HIOS %llu\n", (unsigned long long)(long long)(ET_HIOS));
#endif
#ifdef ET_HIPROC
printf("ET_HIPROC %llu\n", (unsigned long long)(long long)(ET_HIPROC));
#endif
#ifdef ET_LOOS
printf("ET_LOOS %llu\n", (unsigned long long)(long long)(ET_LOOS));
#endif
#ifdef ET_LOPROC
printf("ET_LOPROC %llu\n", (unsigned long long)(long long)(ET_LOPROC));
#endif
#ifdef ET_NONE
printf("ET_NONE %llu\n", (unsigned long long)(long long)(ET_NONE));
#endif
#ifdef ET_NUM
printf("ET_NUM %llu\n", (unsigned long long)(long long)(ET_NUM));
#endif
#ifdef ET_REL
printf("ET_REL %llu\n", (unsigned long long)(long long)(ET_REL));
#endif
#ifdef EUCLEAN
printf("EUCLEAN %llu\n", (unsigned long long)(long long)(EUCLEAN));
#endif
#ifdef EUNATCH
printf("EUNATCH %llu\n", (unsigned long long)(long long)(EUNATCH));
#endif
#ifdef EUSERS
printf("EUSERS %llu\n", (unsigned long long)(long long)(EUSERS));
#endif
#ifdef EV_CNT
printf("EV_CNT %llu\n", (unsigned long long)(long long)(EV_CNT));
#endif
#ifdef EV_CURRENT
printf("EV_CURRENT %llu\n", (unsigned long long)(long long)(EV_CURRENT));
#endif
#ifdef EV_MAX
printf("EV_MAX %llu\n", (unsigned long long)(long long)(EV_MAX));
#endif
#ifdef EV_NONE
printf("EV_NONE %llu\n", (unsigned long long)(long long)(EV_NONE));
#endif
#ifdef EV_NUM
printf("EV_NUM %llu\n", (unsigned long long)(long long)(EV_NUM));
#endif
#ifdef EWOULDBLOCK
printf("EWOULDBLOCK %llu\n", (unsigned long long)(long long)(EWOULDBLOCK));
#endif
#ifdef EXDEV
printf("EXDEV %llu\n", (unsigned long long)(long long)(EXDEV));
#endif
#ifdef EXFULL
printf("EXFULL %llu\n", (unsigned long long)(long long)(EXFULL));
#endif
#ifdef EXIT_FAILURE
printf("EXIT_FAILURE %llu\n", (unsigned long long)(long long)(EXIT_FAILURE));
#endif
#ifdef EXIT_SUCCESS
printf("EXIT_SUCCESS %llu\n", (unsigned long long)(long long)(EXIT_SUCCESS));
#endif
#ifdef EXT2_SUPER_MAGIC
printf("EXT2_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(EXT2_SUPER_MAGIC));
#endif
#ifdef EXT3_SUPER_MAGIC
printf("EXT3_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(EXT3_SUPER_MAGIC));
#endif
#ifdef EXT4_SUPER_MAGIC
printf("EXT4_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(EXT4_SUPER_MAGIC));
#endif
#ifdef __EXTA
printf("EXTA %llu\n", (unsigned long long)(__EXTA));
#elif defined(EXTA)
printf("EXTA %llu\n", (unsigned long long)(long long)(EXTA));
#endif
#ifdef __EXTB
printf("EXTB %llu\n", (unsigned long long)(__EXTB));
#elif defined(EXTB)
printf("EXTB %llu\n", (unsigned long long)(long long)(EXTB));
#endif
#ifdef EXTPROC
printf("EXTPROC %llu\n", (unsigned long long)(long long)(EXTPROC));
#endif
#ifdef F2FS_SUPER_MAGIC
printf("F2FS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(F2FS_SUPER_MAGIC));
#endif
#ifdef FALLOC_FL_COLLAPSE_RANGE
printf("FALLOC_FL_COLLAPSE_RANGE %llu\n", (unsigned long long)(long long)(FALLOC_FL_COLLAPSE_RANGE));
#endif
#ifdef FALLOC_FL_INSERT_RANGE
printf("FALLOC_FL_INSERT_RANGE %llu\n", (unsigned long long)(long long)(FALLOC_FL_INSERT_RANGE));
#endif
#ifdef FALLOC_FL_KEEP_SIZE
printf("FALLOC_FL_KEEP_SIZE %llu\n", (unsigned long long)(long long)(FALLOC_FL_KEEP_SIZE));
#endif
#ifdef FALLOC_FL_PUNCH_HOLE
printf("FALLOC_FL_PUNCH_HOLE %llu\n", (unsigned long long)(long long)(FALLOC_FL_PUNCH_HOLE));
#endif
#ifdef FALLOC_FL_UNSHARE_RANGE
printf("FALLOC_FL_UNSHARE_RANGE %llu\n", (unsigned long long)(long long)(FALLOC_FL_UNSHARE_RANGE));
#endif
#ifdef FALLOC_FL_ZERO_RANGE
printf("FALLOC_FL_ZERO_RANGE %llu\n", (unsigned long long)(long long)(FALLOC_FL_ZERO_RANGE));
#endif
#ifdef FANOTIFY_METADATA_VERSION
printf("FANOTIFY_METADATA_VERSION %llu\n", (unsigned long long)(long long)(FANOTIFY_METADATA_VERSION));
#endif
#ifdef FAN_ACCESS
printf("FAN_ACCESS %llu\n", (unsigned long long)(long long)(FAN_ACCESS));
#endif
#ifdef FAN_ACCESS_PERM
printf("FAN_ACCESS_PERM %llu\n", (unsigned long long)(long long)(FAN_ACCESS_PERM));
#endif
#ifdef FAN_ALLOW
printf("FAN_ALLOW %llu\n", (unsigned long long)(long long)(FAN_ALLOW));
#endif
#ifdef FAN_ATTRIB
printf("FAN_ATTRIB %llu\n", (unsigned long long)(long long)(FAN_ATTRIB));
#endif
#ifdef FAN_AUDIT
printf("FAN_AUDIT %llu\n", (unsigned long long)(long long)(FAN_AUDIT));
#endif
#ifdef FAN_CLASS_CONTENT
printf("FAN_CLASS_CONTENT %llu\n", (unsigned long long)(long long)(FAN_CLASS_CONTENT));
#endif
#ifdef FAN_CLASS_NOTIF
printf("FAN_CLASS_NOTIF %llu\n", (unsigned long long)(long long)(FAN_CLASS_NOTIF));
#endif
#ifdef FAN_CLASS_PRE_CONTENT
printf("FAN_CLASS_PRE_CONTENT %llu\n", (unsigned long long)(long long)(FAN_CLASS_PRE_CONTENT));
#endif
#ifdef FAN_CLOEXEC
printf("FAN_CLOEXEC %llu\n", (unsigned long long)(long long)(FAN_CLOEXEC));
#endif
#ifdef FAN_CLOSE
printf("FAN_CLOSE %llu\n", (unsigned long long)(long long)(FAN_CLOSE));
#endif
#ifdef FAN_CLOSE_NOWRITE
printf("FAN_CLOSE_NOWRITE %llu\n", (unsigned long long)(long long)(FAN_CLOSE_NOWRITE));
#endif
#ifdef FAN_CLOSE_WRITE
printf("FAN_CLOSE_WRITE %llu\n", (unsigned long long)(long long)(FAN_CLOSE_WRITE));
#endif
#ifdef FAN_CREATE
printf("FAN_CREATE %llu\n", (unsigned long long)(long long)(FAN_CREATE));
#endif
#ifdef FAN_DELETE
printf("FAN_DELETE %llu\n", (unsigned long long)(long long)(FAN_DELETE));
#endif
#ifdef FAN_DELETE_SELF
printf("FAN_DELETE_SELF %llu\n", (unsigned long long)(long long)(FAN_DELETE_SELF));
#endif
#ifdef FAN_DENY
printf("FAN_DENY %llu\n", (unsigned long long)(long long)(FAN_DENY));
#endif
#ifdef FAN_ENABLE_AUDIT
printf("FAN_ENABLE_AUDIT %llu\n", (unsigned long long)(long long)(FAN_ENABLE_AUDIT));
#endif
#ifdef FAN_EPIDFD
printf("FAN_EPIDFD %llu\n", (unsigned long long)(long long)(FAN_EPIDFD));
#endif
#ifdef FAN_EVENT_INFO_TYPE_DFID
printf("FAN_EVENT_INFO_TYPE_DFID %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_DFID));
#endif
#ifdef FAN_EVENT_INFO_TYPE_DFID_NAME
printf("FAN_EVENT_INFO_TYPE_DFID_NAME %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_DFID_NAME));
#endif
#ifdef FAN_EVENT_INFO_TYPE_ERROR
printf("FAN_EVENT_INFO_TYPE_ERROR %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_ERROR));
#endif
#ifdef FAN_EVENT_INFO_TYPE_FID
printf("FAN_EVENT_INFO_TYPE_FID %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_FID));
#endif
#ifdef FAN_EVENT_INFO_TYPE_NEW_DFID_NAME
printf("FAN_EVENT_INFO_TYPE_NEW_DFID_NAME %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_NEW_DFID_NAME));
#endif
#ifdef FAN_EVENT_INFO_TYPE_OLD_DFID_NAME
printf("FAN_EVENT_INFO_TYPE_OLD_DFID_NAME %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_OLD_DFID_NAME));
#endif
#ifdef FAN_EVENT_INFO_TYPE_PIDFD
printf("FAN_EVENT_INFO_TYPE_PIDFD %llu\n", (unsigned long long)(long long)(FAN_EVENT_INFO_TYPE_PIDFD));
#endif
#ifdef FAN_EVENT_ON_CHILD
printf("FAN_EVENT_ON_CHILD %llu\n", (unsigned long long)(long long)(FAN_EVENT_ON_CHILD));
#endif
#ifdef FAN_FS_ERROR
printf("FAN_FS_ERROR %llu\n", (unsigned long long)(long long)(FAN_FS_ERROR));
#endif
#ifdef FAN_INFO
printf("FAN_INFO %llu\n", (unsigned long long)(long long)(FAN_INFO));
#endif
#ifdef FAN_MARK_ADD
printf("FAN_MARK_ADD %llu\n", (unsigned long long)(long long)(FAN_MARK_ADD));
#endif
#ifdef FAN_MARK_DONT_FOLLOW
printf("FAN_MARK_DONT_FOLLOW %llu\n", (unsigned long long)(long long)(FAN_MARK_DONT_FOLLOW));
#endif
#ifdef FAN_MARK_EVICTABLE
printf("FAN_MARK_EVICTABLE %llu\n", (unsigned long long)(long long)(FAN_MARK_EVICTABLE));
#endif
#ifdef FAN_MARK_FILESYSTEM
printf("FAN_MARK_FILESYSTEM %llu\n", (unsigned long long)(long long)(FAN_MARK_FILESYSTEM));
#endif
#ifdef FAN_MARK_FLUSH
printf("FAN_MARK_FLUSH %llu\n", (unsigned long long)(long long)(FAN_MARK_FLUSH));
#endif
#ifdef FAN_MARK_IGNORE
printf("FAN_MARK_IGNORE %llu\n", (unsigned long long)(long long)(FAN_MARK_IGNORE));
#endif
#ifdef FAN_MARK_IGNORED_MASK
printf("FAN_MARK_IGNORED_MASK %llu\n", (unsigned long long)(long long)(FAN_MARK_IGNORED_MASK));
#endif
#ifdef FAN_MARK_IGNORED_SURV_MODIFY
printf("FAN_MARK_IGNORED_SURV_MODIFY %llu\n", (unsigned long long)(long long)(FAN_MARK_IGNORED_SURV_MODIFY));
#endif
#ifdef FAN_MARK_IGNORE_SURV
printf("FAN_MARK_IGNORE_SURV %llu\n", (unsigned long long)(long long)(FAN_MARK_IGNORE_SURV));
#endif
#ifdef FAN_MARK_INODE
printf("FAN_MARK_INODE %llu\n", (unsigned long long)(long long)(FAN_MARK_INODE));
#endif
#ifdef FAN_MARK_MOUNT
printf("FAN_MARK_MOUNT %llu\n", (unsigned long long)(long long)(FAN_MARK_MOUNT));
#endif
#ifdef FAN_MARK_ONLYDIR
printf("FAN_MARK_ONLYDIR %llu\n", (unsigned long long)(long long)(FAN_MARK_ONLYDIR));
#endif
#ifdef FAN_MARK_REMOVE
printf("FAN_MARK_REMOVE %llu\n", (unsigned long long)(long long)(FAN_MARK_REMOVE));
#endif
#ifdef FAN_MODIFY
printf("FAN_MODIFY %llu\n", (unsigned long long)(long long)(FAN_MODIFY));
#endif
#ifdef FAN_MOVE
printf("FAN_MOVE %llu\n", (unsigned long long)(long long)(FAN_MOVE));
#endif
#ifdef FAN_MOVED_FROM
printf("FAN_MOVED_FROM %llu\n", (unsigned long long)(long long)(FAN_MOVED_FROM));
#endif
#ifdef FAN_MOVED_TO
printf("FAN_MOVED_TO %llu\n", (unsigned long long)(long long)(FAN_MOVED_TO));
#endif
#ifdef FAN_MOVE_SELF
printf("FAN_MOVE_SELF %llu\n", (unsigned long long)(long long)(FAN_MOVE_SELF));
#endif
#ifdef FAN_NOFD
printf("FAN_NOFD %llu\n", (unsigned long long)(long long)(FAN_NOFD));
#endif
#ifdef FAN_NONBLOCK
printf("FAN_NONBLOCK %llu\n", (unsigned long long)(long long)(FAN_NONBLOCK));
#endif
#ifdef FAN_NOPIDFD
printf("FAN_NOPIDFD %llu\n", (unsigned long long)(long long)(FAN_NOPIDFD));
#endif
#ifdef FAN_ONDIR
printf("FAN_ONDIR %llu\n", (unsigned long long)(long long)(FAN_ONDIR));
#endif
#ifdef FAN_OPEN
printf("FAN_OPEN %llu\n", (unsigned long long)(long long)(FAN_OPEN));
#endif
#ifdef FAN_OPEN_EXEC
printf("FAN_OPEN_EXEC %llu\n", (unsigned long long)(long long)(FAN_OPEN_EXEC));
#endif
#ifdef FAN_OPEN_EXEC_PERM
printf("FAN_OPEN_EXEC_PERM %llu\n", (unsigned long long)(long long)(FAN_OPEN_EXEC_PERM));
#endif
#ifdef FAN_OPEN_PERM
printf("FAN_OPEN_PERM %llu\n", (unsigned long long)(long long)(FAN_OPEN_PERM));
#endif
#ifdef FAN_Q_OVERFLOW
printf("FAN_Q_OVERFLOW %llu\n", (unsigned long long)(long long)(FAN_Q_OVERFLOW));
#endif
#ifdef FAN_RENAME
printf("FAN_RENAME %llu\n", (unsigned long long)(long long)(FAN_RENAME));
#endif
#ifdef FAN_REPORT_DFID_NAME
printf("FAN_REPORT_DFID_NAME %llu\n", (unsigned long long)(long long)(FAN_REPORT_DFID_NAME));
#endif
#ifdef FAN_REPORT_DFID_NAME_TARGET
printf("FAN_REPORT_DFID_NAME_TARGET %llu\n", (unsigned long long)(long long)(FAN_REPORT_DFID_NAME_TARGET));
#endif
#ifdef FAN_REPORT_DIR_FID
printf("FAN_REPORT_DIR_FID %llu\n", (unsigned long long)(long long)(FAN_REPORT_DIR_FID));
#endif
#ifdef FAN_REPORT_FID
printf("FAN_REPORT_FID %llu\n", (unsigned long long)(long long)(FAN_REPORT_FID));
#endif
#ifdef FAN_REPORT_NAME
printf("FAN_REPORT_NAME %llu\n", (unsigned long long)(long long)(FAN_REPORT_NAME));
#endif
#ifdef FAN_REPORT_PIDFD
printf("FAN_REPORT_PIDFD %llu\n", (unsigned long long)(long long)(FAN_REPORT_PIDFD));
#endif
#ifdef FAN_REPORT_TARGET_FID
printf("FAN_REPORT_TARGET_FID %llu\n", (unsigned long long)(long long)(FAN_REPORT_TARGET_FID));
#endif
#ifdef FAN_REPORT_TID
printf("FAN_REPORT_TID %llu\n", (unsigned long long)(long long)(FAN_REPORT_TID));
#endif
#ifdef FAN_RESPONSE_INFO_AUDIT_RULE
printf("FAN_RESPONSE_INFO_AUDIT_RULE %llu\n", (unsigned long long)(long long)(FAN_RESPONSE_INFO_AUDIT_RULE));
#endif
#ifdef FAN_RESPONSE_INFO_NONE
printf("FAN_RESPONSE_INFO_NONE %llu\n", (unsigned long long)(long long)(FAN_RESPONSE_INFO_NONE));
#endif
#ifdef FAN_UNLIMITED_MARKS
printf("FAN_UNLIMITED_MARKS %llu\n", (unsigned long long)(long long)(FAN_UNLIMITED_MARKS));
#endif
#ifdef FAN_UNLIMITED_QUEUE
printf("FAN_UNLIMITED_QUEUE %llu\n", (unsigned long long)(long long)(FAN_UNLIMITED_QUEUE));
#endif
#ifdef FDPIC_FUNCPTRS
printf("FDPIC_FUNCPTRS %llu\n", (unsigned long long)(long long)(FDPIC_FUNCPTRS));
#endif
#ifdef FD_CLOEXEC
printf("FD_CLOEXEC %llu\n", (unsigned long long)(long long)(FD_CLOEXEC));
#endif
#ifdef FD_SETSIZE
printf("FD_SETSIZE %llu\n", (unsigned long long)(long long)(FD_SETSIZE));
#endif
#ifdef FF0
printf("FF0 %llu\n", (unsigned long long)(long long)(FF0));
#endif
#ifdef FF1
printf("FF1 %llu\n", (unsigned long long)(long long)(FF1));
#endif
#ifdef FFDLY
printf("FFDLY %llu\n", (unsigned long long)(long long)(FFDLY));
#endif
#ifdef FF_CNT
printf("FF_CNT %llu\n", (unsigned long long)(long long)(FF_CNT));
#endif
#ifdef FF_MAX
printf("FF_MAX %llu\n", (unsigned long long)(long long)(FF_MAX));
#endif
#ifdef FICLONE
printf("FICLONE %llu\n", (unsigned long long)(long long)(FICLONE));
#endif
#ifdef FICLONERANGE
printf("FICLONERANGE %llu\n", (unsigned long long)(long long)(FICLONERANGE));
#endif
#ifdef FILENAME_MAX
printf("FILENAME_MAX %llu\n", (unsigned long long)(long long)(FILENAME_MAX));
#endif
#ifdef FIOASYNC
printf("FIOASYNC %llu\n", (unsigned long long)(long long)(FIOASYNC));
#endif
#ifdef FIOCLEX
printf("FIOCLEX %llu\n", (unsigned long long)(long long)(FIOCLEX));
#endif
#ifdef FIONBIO
printf("FIONBIO %llu\n", (unsigned long long)(long long)(FIONBIO));
#endif
#ifdef FIONCLEX
printf("FIONCLEX %llu\n", (unsigned long long)(long long)(FIONCLEX));
#endif
#ifdef FIONREAD
printf("FIONREAD %llu\n", (unsigned long long)(long long)(FIONREAD));
#endif
#ifdef FIOQSIZE
printf("FIOQSIZE %llu\n", (unsigned long long)(long long)(FIOQSIZE));
#endif
#ifdef FLUSHO
printf("FLUSHO %llu\n", (unsigned long long)(long long)(FLUSHO));
#endif
#ifdef FNM_CASEFOLD
printf("FNM_CASEFOLD %llu\n", (unsigned long long)(long long)(FNM_CASEFOLD));
#endif
#ifdef FNM_NOESCAPE
printf("FNM_NOESCAPE %llu\n", (unsigned long long)(long long)(FNM_NOESCAPE));
#endif
#ifdef FNM_NOMATCH
printf("FNM_NOMATCH %llu\n", (unsigned long long)(long long)(FNM_NOMATCH));
#endif
#ifdef FNM_PATHNAME
printf("FNM_PATHNAME %llu\n", (unsigned long long)(long long)(FNM_PATHNAME));
#endif
#ifdef FNM_PERIOD
printf("FNM_PERIOD %llu\n", (unsigned long long)(long long)(FNM_PERIOD));
#endif
#ifdef FOPEN_MAX
printf("FOPEN_MAX %llu\n", (unsigned long long)(long long)(FOPEN_MAX));
#endif
#ifdef FS_IOC32_GETFLAGS
printf("FS_IOC32_GETFLAGS %llu\n", (unsigned long long)(long long)(FS_IOC32_GETFLAGS));
#endif
#ifdef FS_IOC32_GETVERSION
printf("FS_IOC32_GETVERSION %llu\n", (unsigned long long)(long long)(FS_IOC32_GETVERSION));
#endif
#ifdef FS_IOC32_SETFLAGS
printf("FS_IOC32_SETFLAGS %llu\n", (unsigned long long)(long long)(FS_IOC32_SETFLAGS));
#endif
#ifdef FS_IOC32_SETVERSION
printf("FS_IOC32_SETVERSION %llu\n", (unsigned long long)(long long)(FS_IOC32_SETVERSION));
#endif
#ifdef FS_IOC_GETFLAGS
printf("FS_IOC_GETFLAGS %llu\n", (unsigned long long)(long long)(FS_IOC_GETFLAGS));
#endif
#ifdef FS_IOC_GETVERSION
printf("FS_IOC_GETVERSION %llu\n", (unsigned long long)(long long)(FS_IOC_GETVERSION));
#endif
#ifdef FS_IOC_SETFLAGS
printf("FS_IOC_SETFLAGS %llu\n", (unsigned long long)(long long)(FS_IOC_SETFLAGS));
#endif
#ifdef FS_IOC_SETVERSION
printf("FS_IOC_SETVERSION %llu\n", (unsigned long long)(long long)(FS_IOC_SETVERSION));
#endif
#ifdef FUSE_SUPER_MAGIC
printf("FUSE_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(FUSE_SUPER_MAGIC));
#endif
#ifdef FUTEXFS_SUPER_MAGIC
printf("FUTEXFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(FUTEXFS_SUPER_MAGIC));
#endif
#ifdef FUTEX_BITSET_MATCH_ANY
printf("FUTEX_BITSET_MATCH_ANY %llu\n", (unsigned long long)(long long)(FUTEX_BITSET_MATCH_ANY));
#endif
#ifdef FUTEX_CLOCK_REALTIME
printf("FUTEX_CLOCK_REALTIME %llu\n", (unsigned long long)(long long)(FUTEX_CLOCK_REALTIME));
#endif
#ifdef FUTEX_CMD_MASK
printf("FUTEX_CMD_MASK %llu\n", (unsigned long long)(long long)(FUTEX_CMD_MASK));
#endif
#ifdef FUTEX_CMP_REQUEUE
printf("FUTEX_CMP_REQUEUE %llu\n", (unsigned long long)(long long)(FUTEX_CMP_REQUEUE));
#endif
#ifdef FUTEX_CMP_REQUEUE_PI
printf("FUTEX_CMP_REQUEUE_PI %llu\n", (unsigned long long)(long long)(FUTEX_CMP_REQUEUE_PI));
#endif
#ifdef FUTEX_FD
printf("FUTEX_FD %llu\n", (unsigned long long)(long long)(FUTEX_FD));
#endif
#ifdef FUTEX_LOCK_PI
printf("FUTEX_LOCK_PI %llu\n", (unsigned long long)(long long)(FUTEX_LOCK_PI));
#endif
#ifdef FUTEX_LOCK_PI2
printf("FUTEX_LOCK_PI2 %llu\n", (unsigned long long)(long long)(FUTEX_LOCK_PI2));
#endif
#ifdef FUTEX_OP_ADD
printf("FUTEX_OP_ADD %llu\n", (unsigned long long)(long long)(FUTEX_OP_ADD));
#endif
#ifdef FUTEX_OP_ANDN
printf("FUTEX_OP_ANDN %llu\n", (unsigned long long)(long long)(FUTEX_OP_ANDN));
#endif
#ifdef FUTEX_OP_CMP_EQ
printf("FUTEX_OP_CMP_EQ %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_EQ));
#endif
#ifdef FUTEX_OP_CMP_GE
printf("FUTEX_OP_CMP_GE %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_GE));
#endif
#ifdef FUTEX_OP_CMP_GT
printf("FUTEX_OP_CMP_GT %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_GT));
#endif
#ifdef FUTEX_OP_CMP_LE
printf("FUTEX_OP_CMP_LE %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_LE));
#endif
#ifdef FUTEX_OP_CMP_LT
printf("FUTEX_OP_CMP_LT %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_LT));
#endif
#ifdef FUTEX_OP_CMP_NE
printf("FUTEX_OP_CMP_NE %llu\n", (unsigned long long)(long long)(FUTEX_OP_CMP_NE));
#endif
#ifdef FUTEX_OP_OPARG_SHIFT
printf("FUTEX_OP_OPARG_SHIFT %llu\n", (unsigned long long)(long long)(FUTEX_OP_OPARG_SHIFT));
#endif
#ifdef FUTEX_OP_OR
printf("FUTEX_OP_OR %llu\n", (unsigned long long)(long long)(FUTEX_OP_OR));
#endif
#ifdef FUTEX_OP_SET
printf("FUTEX_OP_SET %llu\n", (unsigned long long)(long long)(FUTEX_OP_SET));
#endif
#ifdef FUTEX_OP_XOR
printf("FUTEX_OP_XOR %llu\n", (unsigned long long)(long long)(FUTEX_OP_XOR));
#endif
#ifdef FUTEX_OWNER_DIED
printf("FUTEX_OWNER_DIED %llu\n", (unsigned long long)(long long)(FUTEX_OWNER_DIED));
#endif
#ifdef FUTEX_PRIVATE_FLAG
printf("FUTEX_PRIVATE_FLAG %llu\n", (unsigned long long)(long long)(FUTEX_PRIVATE_FLAG));
#endif
#ifdef FUTEX_REQUEUE
printf("FUTEX_REQUEUE %llu\n", (unsigned long long)(long long)(FUTEX_REQUEUE));
#endif
#ifdef FUTEX_TID_MASK
printf("FUTEX_TID_MASK %llu\n", (unsigned long long)(long long)(FUTEX_TID_MASK));
#endif
#ifdef FUTEX_TRYLOCK_PI
printf("FUTEX_TRYLOCK_PI %llu\n", (unsigned long long)(long long)(FUTEX_TRYLOCK_PI));
#endif
#ifdef FUTEX_UNLOCK_PI
printf("FUTEX_UNLOCK_PI %llu\n", (unsigned long long)(long long)(FUTEX_UNLOCK_PI));
#endif
#ifdef FUTEX_WAIT
printf("FUTEX_WAIT %llu\n", (unsigned long long)(long long)(FUTEX_WAIT));
#endif
#ifdef FUTEX_WAITERS
printf("FUTEX_WAITERS %llu\n", (unsigned long long)(long long)(FUTEX_WAITERS));
#endif
#ifdef FUTEX_WAIT_BITSET
printf("FUTEX_WAIT_BITSET %llu\n", (unsigned long long)(long long)(FUTEX_WAIT_BITSET));
#endif
#ifdef FUTEX_WAIT_REQUEUE_PI
printf("FUTEX_WAIT_REQUEUE_PI %llu\n", (unsigned long long)(long long)(FUTEX_WAIT_REQUEUE_PI));
#endif
#ifdef FUTEX_WAKE
printf("FUTEX_WAKE %llu\n", (unsigned long long)(long long)(FUTEX_WAKE));
#endif
#ifdef FUTEX_WAKE_BITSET
printf("FUTEX_WAKE_BITSET %llu\n", (unsigned long long)(long long)(FUTEX_WAKE_BITSET));
#endif
#ifdef FUTEX_WAKE_OP
printf("FUTEX_WAKE_OP %llu\n", (unsigned long long)(long long)(FUTEX_WAKE_OP));
#endif
#ifdef F_ADD_SEALS
printf("F_ADD_SEALS %llu\n", (unsigned long long)(long long)(F_ADD_SEALS));
#endif
#ifdef F_CANCELLK
printf("F_CANCELLK %llu\n", (unsigned long long)(long long)(F_CANCELLK));
#endif
#ifdef F_DUPFD
printf("F_DUPFD %llu\n", (unsigned long long)(long long)(F_DUPFD));
#endif
#ifdef F_DUPFD_CLOEXEC
printf("F_DUPFD_CLOEXEC %llu\n", (unsigned long long)(long long)(F_DUPFD_CLOEXEC));
#endif
#ifdef F_GETFD
printf("F_GETFD %llu\n", (unsigned long long)(long long)(F_GETFD));
#endif
#ifdef F_GETFL
printf("F_GETFL %llu\n", (unsigned long long)(long long)(F_GETFL));
#endif
#ifdef F_GETLEASE
printf("F_GETLEASE %llu\n", (unsigned long long)(long long)(F_GETLEASE));
#endif
#ifdef F_GETLK
printf("F_GETLK %llu\n", (unsigned long long)(long long)(F_GETLK));
#endif
#ifdef F_GETOWN
printf("F_GETOWN %llu\n", (unsigned long long)(long long)(F_GETOWN));
#endif
#ifdef F_GETPIPE_SZ
printf("F_GETPIPE_SZ %llu\n", (unsigned long long)(long long)(F_GETPIPE_SZ));
#endif
#ifdef F_GET_SEALS
printf("F_GET_SEALS %llu\n", (unsigned long long)(long long)(F_GET_SEALS));
#endif
#ifdef F_LOCK
printf("F_LOCK %llu\n", (unsigned long long)(long long)(F_LOCK));
#endif
#ifdef F_NOTIFY
printf("F_NOTIFY %llu\n", (unsigned long long)(long long)(F_NOTIFY));
#endif
#ifdef F_OFD_GETLK
printf("F_OFD_GETLK %llu\n", (unsigned long long)(long long)(F_OFD_GETLK));
#endif
#ifdef F_OFD_SETLK
printf("F_OFD_SETLK %llu\n", (unsigned long long)(long long)(F_OFD_SETLK));
#endif
#ifdef F_OFD_SETLKW
printf("F_OFD_SETLKW %llu\n", (unsigned long long)(long long)(F_OFD_SETLKW));
#endif
#ifdef F_OK
printf("F_OK %llu\n", (unsigned long long)(long long)(F_OK));
#endif
#ifdef F_RDLCK
printf("F_RDLCK %llu\n", (unsigned long long)(long long)(F_RDLCK));
#endif
#ifdef F_SEAL_EXEC
printf("F_SEAL_EXEC %llu\n", (unsigned long long)(long long)(F_SEAL_EXEC));
#endif
#ifdef F_SEAL_FUTURE_WRITE
printf("F_SEAL_FUTURE_WRITE %llu\n", (unsigned long long)(long long)(F_SEAL_FUTURE_WRITE));
#endif
#ifdef F_SEAL_GROW
printf("F_SEAL_GROW %llu\n", (unsigned long long)(long long)(F_SEAL_GROW));
#endif
#ifdef F_SEAL_SEAL
printf("F_SEAL_SEAL %llu\n", (unsigned long long)(long long)(F_SEAL_SEAL));
#endif
#ifdef F_SEAL_SHRINK
printf("F_SEAL_SHRINK %llu\n", (unsigned long long)(long long)(F_SEAL_SHRINK));
#endif
#ifdef F_SEAL_WRITE
printf("F_SEAL_WRITE %llu\n", (unsigned long long)(long long)(F_SEAL_WRITE));
#endif
#ifdef F_SETFD
printf("F_SETFD %llu\n", (unsigned long long)(long long)(F_SETFD));
#endif
#ifdef F_SETFL
printf("F_SETFL %llu\n", (unsigned long long)(long long)(F_SETFL));
#endif
#ifdef F_SETLEASE
printf("F_SETLEASE %llu\n", (unsigned long long)(long long)(F_SETLEASE));
#endif
#ifdef F_SETLK
printf("F_SETLK %llu\n", (unsigned long long)(long long)(F_SETLK));
#endif
#ifdef F_SETLKW
printf("F_SETLKW %llu\n", (unsigned long long)(long long)(F_SETLKW));
#endif
#ifdef F_SETOWN
printf("F_SETOWN %llu\n", (unsigned long long)(long long)(F_SETOWN));
#endif
#ifdef F_SETPIPE_SZ
printf("F_SETPIPE_SZ %llu\n", (unsigned long long)(long long)(F_SETPIPE_SZ));
#endif
#ifdef F_TEST
printf("F_TEST %llu\n", (unsigned long long)(long long)(F_TEST));
#endif
#ifdef F_TLOCK
printf("F_TLOCK %llu\n", (unsigned long long)(long long)(F_TLOCK));
#endif
#ifdef F_ULOCK
printf("F_ULOCK %llu\n", (unsigned long long)(long long)(F_ULOCK));
#endif
#ifdef F_UNLCK
printf("F_UNLCK %llu\n", (unsigned long long)(long long)(F_UNLCK));
#endif
#ifdef F_WRLCK
printf("F_WRLCK %llu\n", (unsigned long long)(long long)(F_WRLCK));
#endif
#ifdef GENL_ADMIN_PERM
printf("GENL_ADMIN_PERM %llu\n", (unsigned long long)(long long)(GENL_ADMIN_PERM));
#endif
#ifdef GENL_CMD_CAP_DO
printf("GENL_CMD_CAP_DO %llu\n", (unsigned long long)(long long)(GENL_CMD_CAP_DO));
#endif
#ifdef GENL_CMD_CAP_DUMP
printf("GENL_CMD_CAP_DUMP %llu\n", (unsigned long long)(long long)(GENL_CMD_CAP_DUMP));
#endif
#ifdef GENL_CMD_CAP_HASPOL
printf("GENL_CMD_CAP_HASPOL %llu\n", (unsigned long long)(long long)(GENL_CMD_CAP_HASPOL));
#endif
#ifdef GENL_ID_CTRL
printf("GENL_ID_CTRL %llu\n", (unsigned long long)(long long)(GENL_ID_CTRL));
#endif
#ifdef GENL_ID_PMCRAID
printf("GENL_ID_PMCRAID %llu\n", (unsigned long long)(long long)(GENL_ID_PMCRAID));
#endif
#ifdef GENL_ID_VFS_DQUOT
printf("GENL_ID_VFS_DQUOT %llu\n", (unsigned long long)(long long)(GENL_ID_VFS_DQUOT));
#endif
#ifdef GENL_MAX_ID
printf("GENL_MAX_ID %llu\n", (unsigned long long)(long long)(GENL_MAX_ID));
#endif
#ifdef GENL_MIN_ID
printf("GENL_MIN_ID %llu\n", (unsigned long long)(long long)(GENL_MIN_ID));
#endif
#ifdef GENL_NAMSIZ
printf("GENL_NAMSIZ %llu\n", (unsigned long long)(long long)(GENL_NAMSIZ));
#endif
#ifdef GENL_UNS_ADMIN_PERM
printf("GENL_UNS_ADMIN_PERM %llu\n", (unsigned long long)(long long)(GENL_UNS_ADMIN_PERM));
#endif
#ifdef GETALL
printf("GETALL %llu\n", (unsigned long long)(long long)(GETALL));
#endif
#ifdef GETNCNT
printf("GETNCNT %llu\n", (unsigned long long)(long long)(GETNCNT));
#endif
#ifdef GETPID
printf("GETPID %llu\n", (unsigned long long)(long long)(GETPID));
#endif
#ifdef GETVAL
printf("GETVAL %llu\n", (unsigned long long)(long long)(GETVAL));
#endif
#ifdef GETZCNT
printf("GETZCNT %llu\n", (unsigned long long)(long long)(GETZCNT));
#endif
#ifdef GLOB_ABORTED
printf("GLOB_ABORTED %llu\n", (unsigned long long)(long long)(GLOB_ABORTED));
#endif
#ifdef GLOB_ALTDIRFUNC
printf("GLOB_ALTDIRFUNC %llu\n", (unsigned long long)(long long)(GLOB_ALTDIRFUNC));
#endif
#ifdef GLOB_APPEND
printf("GLOB_APPEND %llu\n", (unsigned long long)(long long)(GLOB_APPEND));
#endif
#ifdef GLOB_BRACE
printf("GLOB_BRACE %llu\n", (unsigned long long)(long long)(GLOB_BRACE));
#endif
#ifdef GLOB_DOOFFS
printf("GLOB_DOOFFS %llu\n", (unsigned long long)(long long)(GLOB_DOOFFS));
#endif
#ifdef GLOB_ERR
printf("GLOB_ERR %llu\n", (unsigned long long)(long long)(GLOB_ERR));
#endif
#ifdef GLOB_MARK
printf("GLOB_MARK %llu\n", (unsigned long long)(long long)(GLOB_MARK));
#endif
#ifdef GLOB_NOCHECK
printf("GLOB_NOCHECK %llu\n", (unsigned long long)(long long)(GLOB_NOCHECK));
#endif
#ifdef GLOB_NOESCAPE
printf("GLOB_NOESCAPE %llu\n", (unsigned long long)(long long)(GLOB_NOESCAPE));
#endif
#ifdef GLOB_NOMAGIC
printf("GLOB_NOMAGIC %llu\n", (unsigned long long)(long long)(GLOB_NOMAGIC));
#endif
#ifdef GLOB_NOMATCH
printf("GLOB_NOMATCH %llu\n", (unsigned long long)(long long)(GLOB_NOMATCH));
#endif
#ifdef GLOB_NOSORT
printf("GLOB_NOSORT %llu\n", (unsigned long long)(long long)(GLOB_NOSORT));
#endif
#ifdef GLOB_NOSPACE
printf("GLOB_NOSPACE %llu\n", (unsigned long long)(long long)(GLOB_NOSPACE));
#endif
#ifdef GLOB_ONLYDIR
printf("GLOB_ONLYDIR %llu\n", (unsigned long long)(long long)(GLOB_ONLYDIR));
#endif
#ifdef GLOB_PERIOD
printf("GLOB_PERIOD %llu\n", (unsigned long long)(long long)(GLOB_PERIOD));
#endif
#ifdef GLOB_TILDE
printf("GLOB_TILDE %llu\n", (unsigned long long)(long long)(GLOB_TILDE));
#endif
#ifdef GLOB_TILDE_CHECK
printf("GLOB_TILDE_CHECK %llu\n", (unsigned long long)(long long)(GLOB_TILDE_CHECK));
#endif
#ifdef GRND_INSECURE
printf("GRND_INSECURE %llu\n", (unsigned long long)(long long)(GRND_INSECURE));
#endif
#ifdef GRND_NONBLOCK
printf("GRND_NONBLOCK %llu\n", (unsigned long long)(long long)(GRND_NONBLOCK));
#endif
#ifdef GRND_RANDOM
printf("GRND_RANDOM %llu\n", (unsigned long long)(long long)(GRND_RANDOM));
#endif
#ifdef GRPQUOTA
printf("GRPQUOTA %llu\n", (unsigned long long)(long long)(GRPQUOTA));
#endif
#ifdef HOSTFS_SUPER_MAGIC
printf("HOSTFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(HOSTFS_SUPER_MAGIC));
#endif
#ifdef HPFS_SUPER_MAGIC
printf("HPFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(HPFS_SUPER_MAGIC));
#endif
#ifdef HUGETLBFS_MAGIC
printf("HUGETLBFS_MAGIC %llu\n", (unsigned long long)(long long)(HUGETLBFS_MAGIC));
#endif
#ifdef HUGETLB_FLAG_ENCODE_16GB
printf("HUGETLB_FLAG_ENCODE_16GB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_16GB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_16MB
printf("HUGETLB_FLAG_ENCODE_16MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_16MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_1GB
printf("HUGETLB_FLAG_ENCODE_1GB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_1GB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_1MB
printf("HUGETLB_FLAG_ENCODE_1MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_1MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_256MB
printf("HUGETLB_FLAG_ENCODE_256MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_256MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_2GB
printf("HUGETLB_FLAG_ENCODE_2GB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_2GB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_2MB
printf("HUGETLB_FLAG_ENCODE_2MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_2MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_32MB
printf("HUGETLB_FLAG_ENCODE_32MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_32MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_512KB
printf("HUGETLB_FLAG_ENCODE_512KB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_512KB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_512MB
printf("HUGETLB_FLAG_ENCODE_512MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_512MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_64KB
printf("HUGETLB_FLAG_ENCODE_64KB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_64KB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_8MB
printf("HUGETLB_FLAG_ENCODE_8MB %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_8MB));
#endif
#ifdef HUGETLB_FLAG_ENCODE_MASK
printf("HUGETLB_FLAG_ENCODE_MASK %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_MASK));
#endif
#ifdef HUGETLB_FLAG_ENCODE_SHIFT
printf("HUGETLB_FLAG_ENCODE_SHIFT %llu\n", (unsigned long long)(long long)(HUGETLB_FLAG_ENCODE_SHIFT));
#endif
#ifdef HUPCL
printf("HUPCL %llu\n", (unsigned long long)(long long)(HUPCL));
#endif
#ifdef HWTSTAMP_FILTER_ALL
printf("HWTSTAMP_FILTER_ALL %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_ALL));
#endif
#ifdef HWTSTAMP_FILTER_NONE
printf("HWTSTAMP_FILTER_NONE %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_NONE));
#endif
#ifdef HWTSTAMP_FILTER_NTP_ALL
printf("HWTSTAMP_FILTER_NTP_ALL %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_NTP_ALL));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ
printf("HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V1_L4_EVENT
printf("HWTSTAMP_FILTER_PTP_V1_L4_EVENT %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V1_L4_EVENT));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V1_L4_SYNC
printf("HWTSTAMP_FILTER_PTP_V1_L4_SYNC %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V1_L4_SYNC));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_DELAY_REQ
printf("HWTSTAMP_FILTER_PTP_V2_DELAY_REQ %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_DELAY_REQ));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_EVENT
printf("HWTSTAMP_FILTER_PTP_V2_EVENT %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_EVENT));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ
printf("HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L2_EVENT
printf("HWTSTAMP_FILTER_PTP_V2_L2_EVENT %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L2_EVENT));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L2_SYNC
printf("HWTSTAMP_FILTER_PTP_V2_L2_SYNC %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L2_SYNC));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ
printf("HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L4_EVENT
printf("HWTSTAMP_FILTER_PTP_V2_L4_EVENT %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L4_EVENT));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_L4_SYNC
printf("HWTSTAMP_FILTER_PTP_V2_L4_SYNC %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_L4_SYNC));
#endif
#ifdef HWTSTAMP_FILTER_PTP_V2_SYNC
printf("HWTSTAMP_FILTER_PTP_V2_SYNC %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_PTP_V2_SYNC));
#endif
#ifdef HWTSTAMP_FILTER_SOME
printf("HWTSTAMP_FILTER_SOME %llu\n", (unsigned long long)(long long)(HWTSTAMP_FILTER_SOME));
#endif
#ifdef HWTSTAMP_TX_OFF
printf("HWTSTAMP_TX_OFF %llu\n", (unsigned long long)(long long)(HWTSTAMP_TX_OFF));
#endif
#ifdef HWTSTAMP_TX_ON
printf("HWTSTAMP_TX_ON %llu\n", (unsigned long long)(long long)(HWTSTAMP_TX_ON));
#endif
#ifdef HWTSTAMP_TX_ONESTEP_P2P
printf("HWTSTAMP_TX_ONESTEP_P2P %llu\n", (unsigned long long)(long long)(HWTSTAMP_TX_ONESTEP_P2P));
#endif
#ifdef HWTSTAMP_TX_ONESTEP_SYNC
printf("HWTSTAMP_TX_ONESTEP_SYNC %llu\n", (unsigned long long)(long long)(HWTSTAMP_TX_ONESTEP_SYNC));
#endif
#ifdef IBSHIFT
printf("IBSHIFT %llu\n", (unsigned long long)(long long)(IBSHIFT));
#endif
#ifdef ICANON
printf("ICANON %llu\n", (unsigned long long)(long long)(ICANON));
#endif
#ifdef ICRNL
printf("ICRNL %llu\n", (unsigned long long)(long long)(ICRNL));
#endif
#ifdef IEXTEN
printf("IEXTEN %llu\n", (unsigned long long)(long long)(IEXTEN));
#endif
#ifdef IFA_ADDRESS
printf("IFA_ADDRESS %llu\n", (unsigned long long)(long long)(IFA_ADDRESS));
#endif
#ifdef IFA_ANYCAST
printf("IFA_ANYCAST %llu\n", (unsigned long long)(long long)(IFA_ANYCAST));
#endif
#ifdef IFA_BROADCAST
printf("IFA_BROADCAST %llu\n", (unsigned long long)(long long)(IFA_BROADCAST));
#endif
#ifdef IFA_CACHEINFO
printf("IFA_CACHEINFO %llu\n", (unsigned long long)(long long)(IFA_CACHEINFO));
#endif
#ifdef IFA_FLAGS
printf("IFA_FLAGS %llu\n", (unsigned long long)(long long)(IFA_FLAGS));
#endif
#ifdef IFA_F_DADFAILED
printf("IFA_F_DADFAILED %llu\n", (unsigned long long)(long long)(IFA_F_DADFAILED));
#endif
#ifdef IFA_F_DEPRECATED
printf("IFA_F_DEPRECATED %llu\n", (unsigned long long)(long long)(IFA_F_DEPRECATED));
#endif
#ifdef IFA_F_HOMEADDRESS
printf("IFA_F_HOMEADDRESS %llu\n", (unsigned long long)(long long)(IFA_F_HOMEADDRESS));
#endif
#ifdef IFA_F_MANAGETEMPADDR
printf("IFA_F_MANAGETEMPADDR %llu\n", (unsigned long long)(long long)(IFA_F_MANAGETEMPADDR));
#endif
#ifdef IFA_F_MCAUTOJOIN
printf("IFA_F_MCAUTOJOIN %llu\n", (unsigned long long)(long long)(IFA_F_MCAUTOJOIN));
#endif
#ifdef IFA_F_NODAD
printf("IFA_F_NODAD %llu\n", (unsigned long long)(long long)(IFA_F_NODAD));
#endif
#ifdef IFA_F_NOPREFIXROUTE
printf("IFA_F_NOPREFIXROUTE %llu\n", (unsigned long long)(long long)(IFA_F_NOPREFIXROUTE));
#endif
#ifdef IFA_F_OPTIMISTIC
printf("IFA_F_OPTIMISTIC %llu\n", (unsigned long long)(long long)(IFA_F_OPTIMISTIC));
#endif
#ifdef IFA_F_PERMANENT
printf("IFA_F_PERMANENT %llu\n", (unsigned long long)(long long)(IFA_F_PERMANENT));
#endif
#ifdef IFA_F_SECONDARY
printf("IFA_F_SECONDARY %llu\n", (unsigned long long)(long long)(IFA_F_SECONDARY));
#endif
#ifdef IFA_F_STABLE_PRIVACY
printf("IFA_F_STABLE_PRIVACY %llu\n", (unsigned long long)(long long)(IFA_F_STABLE_PRIVACY));
#endif
#ifdef IFA_F_TEMPORARY
printf("IFA_F_TEMPORARY %llu\n", (unsigned long long)(long long)(IFA_F_TEMPORARY));
#endif
#ifdef IFA_F_TENTATIVE
printf("IFA_F_TENTATIVE %llu\n", (unsigned long long)(long long)(IFA_F_TENTATIVE));
#endif
#ifdef IFA_LABEL
printf("IFA_LABEL %llu\n", (unsigned long long)(long long)(IFA_LABEL));
#endif
#ifdef IFA_LOCAL
printf("IFA_LOCAL %llu\n", (unsigned long long)(long long)(IFA_LOCAL));
#endif
#ifdef IFA_MULTICAST
printf("IFA_MULTICAST %llu\n", (unsigned long long)(long long)(IFA_MULTICAST));
#endif
#ifdef IFA_UNSPEC
printf("IFA_UNSPEC %llu\n", (unsigned long long)(long long)(IFA_UNSPEC));
#endif
#ifdef IFF_ALLMULTI
printf("IFF_ALLMULTI %llu\n", (unsigned long long)(long long)(IFF_ALLMULTI));
#endif
#ifdef IFF_ATTACH_QUEUE
printf("IFF_ATTACH_QUEUE %llu\n", (unsigned long long)(long long)(IFF_ATTACH_QUEUE));
#endif
#ifdef IFF_AUTOMEDIA
printf("IFF_AUTOMEDIA %llu\n", (unsigned long long)(long long)(IFF_AUTOMEDIA));
#endif
#ifdef IFF_BROADCAST
printf("IFF_BROADCAST %llu\n", (unsigned long long)(long long)(IFF_BROADCAST));
#endif
#ifdef IFF_DEBUG
printf("IFF_DEBUG %llu\n", (unsigned long long)(long long)(IFF_DEBUG));
#endif
#ifdef IFF_DETACH_QUEUE
printf("IFF_DETACH_QUEUE %llu\n", (unsigned long long)(long long)(IFF_DETACH_QUEUE));
#endif
#ifdef IFF_DORMANT
printf("IFF_DORMANT %llu\n", (unsigned long long)(long long)(IFF_DORMANT));
#endif
#ifdef IFF_DYNAMIC
printf("IFF_DYNAMIC %llu\n", (unsigned long long)(long long)(IFF_DYNAMIC));
#endif
#ifdef IFF_ECHO
printf("IFF_ECHO %llu\n", (unsigned long long)(long long)(IFF_ECHO));
#endif
#ifdef IFF_LOOPBACK
printf("IFF_LOOPBACK %llu\n", (unsigned long long)(long long)(IFF_LOOPBACK));
#endif
#ifdef IFF_LOWER_UP
printf("IFF_LOWER_UP %llu\n", (unsigned long long)(long long)(IFF_LOWER_UP));
#endif
#ifdef IFF_MASTER
printf("IFF_MASTER %llu\n", (unsigned long long)(long long)(IFF_MASTER));
#endif
#ifdef IFF_MULTICAST
printf("IFF_MULTICAST %llu\n", (unsigned long long)(long long)(IFF_MULTICAST));
#endif
#ifdef IFF_MULTI_QUEUE
printf("IFF_MULTI_QUEUE %llu\n", (unsigned long long)(long long)(IFF_MULTI_QUEUE));
#endif
#ifdef IFF_NAPI
printf("IFF_NAPI %llu\n", (unsigned long long)(long long)(IFF_NAPI));
#endif
#ifdef IFF_NAPI_FRAGS
printf("IFF_NAPI_FRAGS %llu\n", (unsigned long long)(long long)(IFF_NAPI_FRAGS));
#endif
#ifdef IFF_NOARP
printf("IFF_NOARP %llu\n", (unsigned long long)(long long)(IFF_NOARP));
#endif
#ifdef IFF_NOFILTER
printf("IFF_NOFILTER %llu\n", (unsigned long long)(long long)(IFF_NOFILTER));
#endif
#ifdef IFF_NOTRAILERS
printf("IFF_NOTRAILERS %llu\n", (unsigned long long)(long long)(IFF_NOTRAILERS));
#endif
#ifdef IFF_NO_CARRIER
printf("IFF_NO_CARRIER %llu\n", (unsigned long long)(long long)(IFF_NO_CARRIER));
#endif
#ifdef IFF_NO_PI
printf("IFF_NO_PI %llu\n", (unsigned long long)(long long)(IFF_NO_PI));
#endif
#ifdef IFF_ONE_QUEUE
printf("IFF_ONE_QUEUE %llu\n", (unsigned long long)(long long)(IFF_ONE_QUEUE));
#endif
#ifdef IFF_PERSIST
printf("IFF_PERSIST %llu\n", (unsigned long long)(long long)(IFF_PERSIST));
#endif
#ifdef IFF_POINTOPOINT
printf("IFF_POINTOPOINT %llu\n", (unsigned long long)(long long)(IFF_POINTOPOINT));
#endif
#ifdef IFF_PORTSEL
printf("IFF_PORTSEL %llu\n", (unsigned long long)(long long)(IFF_PORTSEL));
#endif
#ifdef IFF_PROMISC
printf("IFF_PROMISC %llu\n", (unsigned long long)(long long)(IFF_PROMISC));
#endif
#ifdef IFF_RUNNING
printf("IFF_RUNNING %llu\n", (unsigned long long)(long long)(IFF_RUNNING));
#endif
#ifdef IFF_SLAVE
printf("IFF_SLAVE %llu\n", (unsigned long long)(long long)(IFF_SLAVE));
#endif
#ifdef IFF_TAP
printf("IFF_TAP %llu\n", (unsigned long long)(long long)(IFF_TAP));
#endif
#ifdef IFF_TUN
printf("IFF_TUN %llu\n", (unsigned long long)(long long)(IFF_TUN));
#endif
#ifdef IFF_TUN_EXCL
printf("IFF_TUN_EXCL %llu\n", (unsigned long long)(long long)(IFF_TUN_EXCL));
#endif
#ifdef IFF_UP
printf("IFF_UP %llu\n", (unsigned long long)(long long)(IFF_UP));
#endif
#ifdef IFF_VNET_HDR
printf("IFF_VNET_HDR %llu\n", (unsigned long long)(long long)(IFF_VNET_HDR));
#endif
#ifdef IFLA_ADDRESS
printf("IFLA_ADDRESS %llu\n", (unsigned long long)(long long)(IFLA_ADDRESS));
#endif
#ifdef IFLA_AF_SPEC
printf("IFLA_AF_SPEC %llu\n", (unsigned long long)(long long)(IFLA_AF_SPEC));
#endif
#ifdef IFLA_ALLMULTI
printf("IFLA_ALLMULTI %llu\n", (unsigned long long)(long long)(IFLA_ALLMULTI));
#endif
#ifdef IFLA_ALT_IFNAME
printf("IFLA_ALT_IFNAME %llu\n", (unsigned long long)(long long)(IFLA_ALT_IFNAME));
#endif
#ifdef IFLA_BROADCAST
printf("IFLA_BROADCAST %llu\n", (unsigned long long)(long long)(IFLA_BROADCAST));
#endif
#ifdef IFLA_CARRIER
printf("IFLA_CARRIER %llu\n", (unsigned long long)(long long)(IFLA_CARRIER));
#endif
#ifdef IFLA_CARRIER_CHANGES
printf("IFLA_CARRIER_CHANGES %llu\n", (unsigned long long)(long long)(IFLA_CARRIER_CHANGES));
#endif
#ifdef IFLA_CARRIER_DOWN_COUNT
printf("IFLA_CARRIER_DOWN_COUNT %llu\n", (unsigned long long)(long long)(IFLA_CARRIER_DOWN_COUNT));
#endif
#ifdef IFLA_CARRIER_UP_COUNT
printf("IFLA_CARRIER_UP_COUNT %llu\n", (unsigned long long)(long long)(IFLA_CARRIER_UP_COUNT));
#endif
#ifdef IFLA_COST
printf("IFLA_COST %llu\n", (unsigned long long)(long long)(IFLA_COST));
#endif
#ifdef IFLA_EVENT
printf("IFLA_EVENT %llu\n", (unsigned long long)(long long)(IFLA_EVENT));
#endif
#ifdef IFLA_EXT_MASK
printf("IFLA_EXT_MASK %llu\n", (unsigned long long)(long long)(IFLA_EXT_MASK));
#endif
#ifdef IFLA_GROUP
printf("IFLA_GROUP %llu\n", (unsigned long long)(long long)(IFLA_GROUP));
#endif
#ifdef IFLA_GRO_MAX_SIZE
printf("IFLA_GRO_MAX_SIZE %llu\n", (unsigned long long)(long long)(IFLA_GRO_MAX_SIZE));
#endif
#ifdef IFLA_GSO_MAX_SEGS
printf("IFLA_GSO_MAX_SEGS %llu\n", (unsigned long long)(long long)(IFLA_GSO_MAX_SEGS));
#endif
#ifdef IFLA_GSO_MAX_SIZE
printf("IFLA_GSO_MAX_SIZE %llu\n", (unsigned long long)(long long)(IFLA_GSO_MAX_SIZE));
#endif
#ifdef IFLA_IFALIAS
printf("IFLA_IFALIAS %llu\n", (unsigned long long)(long long)(IFLA_IFALIAS));
#endif
#ifdef IFLA_IFNAME
printf("IFLA_IFNAME %llu\n", (unsigned long long)(long long)(IFLA_IFNAME));
#endif
#ifdef IFLA_IF_NETNSID
printf("IFLA_IF_NETNSID %llu\n", (unsigned long long)(long long)(IFLA_IF_NETNSID));
#endif
#ifdef IFLA_INFO_DATA
printf("IFLA_INFO_DATA %llu\n", (unsigned long long)(long long)(IFLA_INFO_DATA));
#endif
#ifdef IFLA_INFO_KIND
printf("IFLA_INFO_KIND %llu\n", (unsigned long long)(long long)(IFLA_INFO_KIND));
#endif
#ifdef IFLA_INFO_SLAVE_DATA
printf("IFLA_INFO_SLAVE_DATA %llu\n", (unsigned long long)(long long)(IFLA_INFO_SLAVE_DATA));
#endif
#ifdef IFLA_INFO_SLAVE_KIND
printf("IFLA_INFO_SLAVE_KIND %llu\n", (unsigned long long)(long long)(IFLA_INFO_SLAVE_KIND));
#endif
#ifdef IFLA_INFO_UNSPEC
printf("IFLA_INFO_UNSPEC %llu\n", (unsigned long long)(long long)(IFLA_INFO_UNSPEC));
#endif
#ifdef IFLA_INFO_XSTATS
printf("IFLA_INFO_XSTATS %llu\n", (unsigned long long)(long long)(IFLA_INFO_XSTATS));
#endif
#ifdef IFLA_LINK
printf("IFLA_LINK %llu\n", (unsigned long long)(long long)(IFLA_LINK));
#endif
#ifdef IFLA_LINKINFO
printf("IFLA_LINKINFO %llu\n", (unsigned long long)(long long)(IFLA_LINKINFO));
#endif
#ifdef IFLA_LINKMODE
printf("IFLA_LINKMODE %llu\n", (unsigned long long)(long long)(IFLA_LINKMODE));
#endif
#ifdef IFLA_LINK_NETNSID
printf("IFLA_LINK_NETNSID %llu\n", (unsigned long long)(long long)(IFLA_LINK_NETNSID));
#endif
#ifdef IFLA_MAP
printf("IFLA_MAP %llu\n", (unsigned long long)(long long)(IFLA_MAP));
#endif
#ifdef IFLA_MASTER
printf("IFLA_MASTER %llu\n", (unsigned long long)(long long)(IFLA_MASTER));
#endif
#ifdef IFLA_MAX_MTU
printf("IFLA_MAX_MTU %llu\n", (unsigned long long)(long long)(IFLA_MAX_MTU));
#endif
#ifdef IFLA_MIN_MTU
printf("IFLA_MIN_MTU %llu\n", (unsigned long long)(long long)(IFLA_MIN_MTU));
#endif
#ifdef IFLA_MTU
printf("IFLA_MTU %llu\n", (unsigned long long)(long long)(IFLA_MTU));
#endif
#ifdef IFLA_NET_NS_FD
printf("IFLA_NET_NS_FD %llu\n", (unsigned long long)(long long)(IFLA_NET_NS_FD));
#endif
#ifdef IFLA_NET_NS_PID
printf("IFLA_NET_NS_PID %llu\n", (unsigned long long)(long long)(IFLA_NET_NS_PID));
#endif
#ifdef IFLA_NEW_IFINDEX
printf("IFLA_NEW_IFINDEX %llu\n", (unsigned long long)(long long)(IFLA_NEW_IFINDEX));
#endif
#ifdef IFLA_NEW_NETNSID
printf("IFLA_NEW_NETNSID %llu\n", (unsigned long long)(long long)(IFLA_NEW_NETNSID));
#endif
#ifdef IFLA_NUM_RX_QUEUES
printf("IFLA_NUM_RX_QUEUES %llu\n", (unsigned long long)(long long)(IFLA_NUM_RX_QUEUES));
#endif
#ifdef IFLA_NUM_TX_QUEUES
printf("IFLA_NUM_TX_QUEUES %llu\n", (unsigned long long)(long long)(IFLA_NUM_TX_QUEUES));
#endif
#ifdef IFLA_NUM_VF
printf("IFLA_NUM_VF %llu\n", (unsigned long long)(long long)(IFLA_NUM_VF));
#endif
#ifdef IFLA_OPERSTATE
printf("IFLA_OPERSTATE %llu\n", (unsigned long long)(long long)(IFLA_OPERSTATE));
#endif
#ifdef IFLA_PAD
printf("IFLA_PAD %llu\n", (unsigned long long)(long long)(IFLA_PAD));
#endif
#ifdef IFLA_PARENT_DEV_BUS_NAME
printf("IFLA_PARENT_DEV_BUS_NAME %llu\n", (unsigned long long)(long long)(IFLA_PARENT_DEV_BUS_NAME));
#endif
#ifdef IFLA_PARENT_DEV_NAME
printf("IFLA_PARENT_DEV_NAME %llu\n", (unsigned long long)(long long)(IFLA_PARENT_DEV_NAME));
#endif
#ifdef IFLA_PERM_ADDRESS
printf("IFLA_PERM_ADDRESS %llu\n", (unsigned long long)(long long)(IFLA_PERM_ADDRESS));
#endif
#ifdef IFLA_PHYS_PORT_ID
printf("IFLA_PHYS_PORT_ID %llu\n", (unsigned long long)(long long)(IFLA_PHYS_PORT_ID));
#endif
#ifdef IFLA_PHYS_PORT_NAME
printf("IFLA_PHYS_PORT_NAME %llu\n", (unsigned long long)(long long)(IFLA_PHYS_PORT_NAME));
#endif
#ifdef IFLA_PHYS_SWITCH_ID
printf("IFLA_PHYS_SWITCH_ID %llu\n", (unsigned long long)(long long)(IFLA_PHYS_SWITCH_ID));
#endif
#ifdef IFLA_PORT_SELF
printf("IFLA_PORT_SELF %llu\n", (unsigned long long)(long long)(IFLA_PORT_SELF));
#endif
#ifdef IFLA_PRIORITY
printf("IFLA_PRIORITY %llu\n", (unsigned long long)(long long)(IFLA_PRIORITY));
#endif
#ifdef IFLA_PROMISCUITY
printf("IFLA_PROMISCUITY %llu\n", (unsigned long long)(long long)(IFLA_PROMISCUITY));
#endif
#ifdef IFLA_PROP_LIST
printf("IFLA_PROP_LIST %llu\n", (unsigned long long)(long long)(IFLA_PROP_LIST));
#endif
#ifdef IFLA_PROTINFO
printf("IFLA_PROTINFO %llu\n", (unsigned long long)(long long)(IFLA_PROTINFO));
#endif
#ifdef IFLA_PROTO_DOWN
printf("IFLA_PROTO_DOWN %llu\n", (unsigned long long)(long long)(IFLA_PROTO_DOWN));
#endif
#ifdef IFLA_PROTO_DOWN_REASON
printf("IFLA_PROTO_DOWN_REASON %llu\n", (unsigned long long)(long long)(IFLA_PROTO_DOWN_REASON));
#endif
#ifdef IFLA_QDISC
printf("IFLA_QDISC %llu\n", (unsigned long long)(long long)(IFLA_QDISC));
#endif
#ifdef IFLA_STATS
printf("IFLA_STATS %llu\n", (unsigned long long)(long long)(IFLA_STATS));
#endif
#ifdef IFLA_STATS64
printf("IFLA_STATS64 %llu\n", (unsigned long long)(long long)(IFLA_STATS64));
#endif
#ifdef IFLA_TARGET_NETNSID
printf("IFLA_TARGET_NETNSID %llu\n", (unsigned long long)(long long)(IFLA_TARGET_NETNSID));
#endif
#ifdef IFLA_TSO_MAX_SEGS
printf("IFLA_TSO_MAX_SEGS %llu\n", (unsigned long long)(long long)(IFLA_TSO_MAX_SEGS));
#endif
#ifdef IFLA_TSO_MAX_SIZE
printf("IFLA_TSO_MAX_SIZE %llu\n", (unsigned long long)(long long)(IFLA_TSO_MAX_SIZE));
#endif
#ifdef IFLA_TXQLEN
printf("IFLA_TXQLEN %llu\n", (unsigned long long)(long long)(IFLA_TXQLEN));
#endif
#ifdef IFLA_UNSPEC
printf("IFLA_UNSPEC %llu\n", (unsigned long long)(long long)(IFLA_UNSPEC));
#endif
#ifdef IFLA_VFINFO_LIST
printf("IFLA_VFINFO_LIST %llu\n", (unsigned long long)(long long)(IFLA_VFINFO_LIST));
#endif
#ifdef IFLA_VF_PORTS
printf("IFLA_VF_PORTS %llu\n", (unsigned long long)(long long)(IFLA_VF_PORTS));
#endif
#ifdef IFLA_WEIGHT
printf("IFLA_WEIGHT %llu\n", (unsigned long long)(long long)(IFLA_WEIGHT));
#endif
#ifdef IFLA_WIRELESS
printf("IFLA_WIRELESS %llu\n", (unsigned long long)(long long)(IFLA_WIRELESS));
#endif
#ifdef IFLA_XDP
printf("IFLA_XDP %llu\n", (unsigned long long)(long long)(IFLA_XDP));
#endif
#ifdef IFNAMSIZ
printf("IFNAMSIZ %llu\n", (unsigned long long)(long long)(IFNAMSIZ));
#endif
#ifdef IF_LINK_MODE_DEFAULT
printf("IF_LINK_MODE_DEFAULT %llu\n", (unsigned long long)(long long)(IF_LINK_MODE_DEFAULT));
#endif
#ifdef IF_LINK_MODE_DORMANT
printf("IF_LINK_MODE_DORMANT %llu\n", (unsigned long long)(long long)(IF_LINK_MODE_DORMANT));
#endif
#ifdef IF_LINK_MODE_TESTING
printf("IF_LINK_MODE_TESTING %llu\n", (unsigned long long)(long long)(IF_LINK_MODE_TESTING));
#endif
#ifdef IF_NAMESIZE
printf("IF_NAMESIZE %llu\n", (unsigned long long)(long long)(IF_NAMESIZE));
#endif
#ifdef IF_OPER_DORMANT
printf("IF_OPER_DORMANT %llu\n", (unsigned long long)(long long)(IF_OPER_DORMANT));
#endif
#ifdef IF_OPER_DOWN
printf("IF_OPER_DOWN %llu\n", (unsigned long long)(long long)(IF_OPER_DOWN));
#endif
#ifdef IF_OPER_LOWERLAYERDOWN
printf("IF_OPER_LOWERLAYERDOWN %llu\n", (unsigned long long)(long long)(IF_OPER_LOWERLAYERDOWN));
#endif
#ifdef IF_OPER_NOTPRESENT
printf("IF_OPER_NOTPRESENT %llu\n", (unsigned long long)(long long)(IF_OPER_NOTPRESENT));
#endif
#ifdef IF_OPER_TESTING
printf("IF_OPER_TESTING %llu\n", (unsigned long long)(long long)(IF_OPER_TESTING));
#endif
#ifdef IF_OPER_UNKNOWN
printf("IF_OPER_UNKNOWN %llu\n", (unsigned long long)(long long)(IF_OPER_UNKNOWN));
#endif
#ifdef IF_OPER_UP
printf("IF_OPER_UP %llu\n", (unsigned long long)(long long)(IF_OPER_UP));
#endif
#ifdef IGNBRK
printf("IGNBRK %llu\n", (unsigned long long)(long long)(IGNBRK));
#endif
#ifdef IGNCR
printf("IGNCR %llu\n", (unsigned long long)(long long)(IGNCR));
#endif
#ifdef IGNPAR
printf("IGNPAR %llu\n", (unsigned long long)(long long)(IGNPAR));
#endif
#ifdef IMAXBEL
printf("IMAXBEL %llu\n", (unsigned long long)(long long)(IMAXBEL));
#endif
#ifdef INADDR_ANY
printf("INADDR_ANY %llu\n", (unsigned long long)(long long)(INADDR_ANY));
#endif
#ifdef INADDR_BROADCAST
printf("INADDR_BROADCAST %llu\n", (unsigned long long)(long long)(INADDR_BROADCAST));
#endif
#ifdef INADDR_LOOPBACK
printf("INADDR_LOOPBACK %llu\n", (unsigned long long)(long long)(INADDR_LOOPBACK));
#endif
#ifdef INADDR_NONE
printf("INADDR_NONE %llu\n", (unsigned long long)(long long)(INADDR_NONE));
#endif
#ifdef INIT_PROCESS
printf("INIT_PROCESS %llu\n", (unsigned long long)(long long)(INIT_PROCESS));
#endif
#ifdef INLCR
printf("INLCR %llu\n", (unsigned long long)(long long)(INLCR));
#endif
#ifdef INOTIFY_MAX_QUEUED_EVENTS
printf("INOTIFY_MAX_QUEUED_EVENTS %llu\n", (unsigned long long)(long long)(INOTIFY_MAX_QUEUED_EVENTS));
#endif
#ifdef INOTIFY_MAX_USER_INSTANCES
printf("INOTIFY_MAX_USER_INSTANCES %llu\n", (unsigned long long)(long long)(INOTIFY_MAX_USER_INSTANCES));
#endif
#ifdef INOTIFY_MAX_USER_WATCHES
printf("INOTIFY_MAX_USER_WATCHES %llu\n", (unsigned long long)(long long)(INOTIFY_MAX_USER_WATCHES));
#endif
#ifdef INPCK
printf("INPCK %llu\n", (unsigned long long)(long long)(INPCK));
#endif
#ifdef INPUT_PROP_ACCELEROMETER
printf("INPUT_PROP_ACCELEROMETER %llu\n", (unsigned long long)(long long)(INPUT_PROP_ACCELEROMETER));
#endif
#ifdef INPUT_PROP_BUTTONPAD
printf("INPUT_PROP_BUTTONPAD %llu\n", (unsigned long long)(long long)(INPUT_PROP_BUTTONPAD));
#endif
#ifdef INPUT_PROP_CNT
printf("INPUT_PROP_CNT %llu\n", (unsigned long long)(long long)(INPUT_PROP_CNT));
#endif
#ifdef INPUT_PROP_DIRECT
printf("INPUT_PROP_DIRECT %llu\n", (unsigned long long)(long long)(INPUT_PROP_DIRECT));
#endif
#ifdef INPUT_PROP_MAX
printf("INPUT_PROP_MAX %llu\n", (unsigned long long)(long long)(INPUT_PROP_MAX));
#endif
#ifdef INPUT_PROP_POINTER
printf("INPUT_PROP_POINTER %llu\n", (unsigned long long)(long long)(INPUT_PROP_POINTER));
#endif
#ifdef INPUT_PROP_POINTING_STICK
printf("INPUT_PROP_POINTING_STICK %llu\n", (unsigned long long)(long long)(INPUT_PROP_POINTING_STICK));
#endif
#ifdef INPUT_PROP_SEMI_MT
printf("INPUT_PROP_SEMI_MT %llu\n", (unsigned long long)(long long)(INPUT_PROP_SEMI_MT));
#endif
#ifdef INPUT_PROP_TOPBUTTONPAD
printf("INPUT_PROP_TOPBUTTONPAD %llu\n", (unsigned long long)(long long)(INPUT_PROP_TOPBUTTONPAD));
#endif
#ifdef INT_MAX
printf("INT_MAX %llu\n", (unsigned long long)(long long)(INT_MAX));
#endif
#ifdef INT_MIN
printf("INT_MIN %llu\n", (unsigned long long)(long long)(INT_MIN));
#endif
#ifdef IN_ACCESS
printf("IN_ACCESS %llu\n", (unsigned long long)(long long)(IN_ACCESS));
#endif
#ifdef IN_ALL_EVENTS
printf("IN_ALL_EVENTS %llu\n", (unsigned long long)(long long)(IN_ALL_EVENTS));
#endif
#ifdef IN_ATTRIB
printf("IN_ATTRIB %llu\n", (unsigned long long)(long long)(IN_ATTRIB));
#endif
#ifdef IN_CLOEXEC
printf("IN_CLOEXEC %llu\n", (unsigned long long)(long long)(IN_CLOEXEC));
#endif
#ifdef IN_CLOSE
printf("IN_CLOSE %llu\n", (unsigned long long)(long long)(IN_CLOSE));
#endif
#ifdef IN_CLOSE_NOWRITE
printf("IN_CLOSE_NOWRITE %llu\n", (unsigned long long)(long long)(IN_CLOSE_NOWRITE));
#endif
#ifdef IN_CLOSE_WRITE
printf("IN_CLOSE_WRITE %llu\n", (unsigned long long)(long long)(IN_CLOSE_WRITE));
#endif
#ifdef IN_CREATE
printf("IN_CREATE %llu\n", (unsigned long long)(long long)(IN_CREATE));
#endif
#ifdef IN_DELETE
printf("IN_DELETE %llu\n", (unsigned long long)(long long)(IN_DELETE));
#endif
#ifdef IN_DELETE_SELF
printf("IN_DELETE_SELF %llu\n", (unsigned long long)(long long)(IN_DELETE_SELF));
#endif
#ifdef IN_DONT_FOLLOW
printf("IN_DONT_FOLLOW %llu\n", (unsigned long long)(long long)(IN_DONT_FOLLOW));
#endif
#ifdef IN_EXCL_UNLINK
printf("IN_EXCL_UNLINK %llu\n", (unsigned long long)(long long)(IN_EXCL_UNLINK));
#endif
#ifdef IN_IGNORED
printf("IN_IGNORED %llu\n", (unsigned long long)(long long)(IN_IGNORED));
#endif
#ifdef IN_ISDIR
printf("IN_ISDIR %llu\n", (unsigned long long)(long long)(IN_ISDIR));
#endif
#ifdef IN_MASK_ADD
printf("IN_MASK_ADD %llu\n", (unsigned long long)(long long)(IN_MASK_ADD));
#endif
#ifdef IN_MASK_CREATE
printf("IN_MASK_CREATE %llu\n", (unsigned long long)(long long)(IN_MASK_CREATE));
#endif
#ifdef IN_MODIFY
printf("IN_MODIFY %llu\n", (unsigned long long)(long long)(IN_MODIFY));
#endif
#ifdef IN_MOVE
printf("IN_MOVE %llu\n", (unsigned long long)(long long)(IN_MOVE));
#endif
#ifdef IN_MOVED_FROM
printf("IN_MOVED_FROM %llu\n", (unsigned long long)(long long)(IN_MOVED_FROM));
#endif
#ifdef IN_MOVED_TO
printf("IN_MOVED_TO %llu\n", (unsigned long long)(long long)(IN_MOVED_TO));
#endif
#ifdef IN_MOVE_SELF
printf("IN_MOVE_SELF %llu\n", (unsigned long long)(long long)(IN_MOVE_SELF));
#endif
#ifdef IN_NONBLOCK
printf("IN_NONBLOCK %llu\n", (unsigned long long)(long long)(IN_NONBLOCK));
#endif
#ifdef IN_ONESHOT
printf("IN_ONESHOT %llu\n", (unsigned long long)(long long)(IN_ONESHOT));
#endif
#ifdef IN_ONLYDIR
printf("IN_ONLYDIR %llu\n", (unsigned long long)(long long)(IN_ONLYDIR));
#endif
#ifdef IN_OPEN
printf("IN_OPEN %llu\n", (unsigned long long)(long long)(IN_OPEN));
#endif
#ifdef IN_Q_OVERFLOW
printf("IN_Q_OVERFLOW %llu\n", (unsigned long long)(long long)(IN_Q_OVERFLOW));
#endif
#ifdef IN_UNMOUNT
printf("IN_UNMOUNT %llu\n", (unsigned long long)(long long)(IN_UNMOUNT));
#endif
#ifdef IP6T_SO_ORIGINAL_DST
printf("IP6T_SO_ORIGINAL_DST %llu\n", (unsigned long long)(long long)(IP6T_SO_ORIGINAL_DST));
#endif
#ifdef IPC_CREAT
printf("IPC_CREAT %llu\n", (unsigned long long)(long long)(IPC_CREAT));
#endif
#ifdef IPC_EXCL
printf("IPC_EXCL %llu\n", (unsigned long long)(long long)(IPC_EXCL));
#endif
#ifdef IPC_INFO
printf("IPC_INFO %llu\n", (unsigned long long)(long long)(IPC_INFO));
#endif
#ifdef IPC_NOWAIT
printf("IPC_NOWAIT %llu\n", (unsigned long long)(long long)(IPC_NOWAIT));
#endif
#ifdef IPC_PRIVATE
printf("IPC_PRIVATE %llu\n", (unsigned long long)(long long)(IPC_PRIVATE));
#endif
#ifdef IPC_RMID
printf("IPC_RMID %llu\n", (unsigned long long)(long long)(IPC_RMID));
#endif
#ifdef IPC_SET
printf("IPC_SET %llu\n", (unsigned long long)(long long)(IPC_SET));
#endif
#ifdef IPC_STAT
printf("IPC_STAT %llu\n", (unsigned long long)(long long)(IPC_STAT));
#endif
#ifdef IPDEFTTL
printf("IPDEFTTL %llu\n", (unsigned long long)(long long)(IPDEFTTL));
#endif
#ifdef IPOPT_CLASS_MASK
printf("IPOPT_CLASS_MASK %llu\n", (unsigned long long)(long long)(IPOPT_CLASS_MASK));
#endif
#ifdef IPOPT_CONTROL
printf("IPOPT_CONTROL %llu\n", (unsigned long long)(long long)(IPOPT_CONTROL));
#endif
#ifdef IPOPT_COPY
printf("IPOPT_COPY %llu\n", (unsigned long long)(long long)(IPOPT_COPY));
#endif
#ifdef IPOPT_END
printf("IPOPT_END %llu\n", (unsigned long long)(long long)(IPOPT_END));
#endif
#ifdef IPOPT_EOL
printf("IPOPT_EOL %llu\n", (unsigned long long)(long long)(IPOPT_EOL));
#endif
#ifdef IPOPT_LSRR
printf("IPOPT_LSRR %llu\n", (unsigned long long)(long long)(IPOPT_LSRR));
#endif
#ifdef IPOPT_MEASUREMENT
printf("IPOPT_MEASUREMENT %llu\n", (unsigned long long)(long long)(IPOPT_MEASUREMENT));
#endif
#ifdef IPOPT_MINOFF
printf("IPOPT_MINOFF %llu\n", (unsigned long long)(long long)(IPOPT_MINOFF));
#endif
#ifdef IPOPT_NOOP
printf("IPOPT_NOOP %llu\n", (unsigned long long)(long long)(IPOPT_NOOP));
#endif
#ifdef IPOPT_NOP
printf("IPOPT_NOP %llu\n", (unsigned long long)(long long)(IPOPT_NOP));
#endif
#ifdef IPOPT_NUMBER_MASK
printf("IPOPT_NUMBER_MASK %llu\n", (unsigned long long)(long long)(IPOPT_NUMBER_MASK));
#endif
#ifdef IPOPT_OFFSET
printf("IPOPT_OFFSET %llu\n", (unsigned long long)(long long)(IPOPT_OFFSET));
#endif
#ifdef IPOPT_OLEN
printf("IPOPT_OLEN %llu\n", (unsigned long long)(long long)(IPOPT_OLEN));
#endif
#ifdef IPOPT_OPTVAL
printf("IPOPT_OPTVAL %llu\n", (unsigned long long)(long long)(IPOPT_OPTVAL));
#endif
#ifdef IPOPT_RA
printf("IPOPT_RA %llu\n", (unsigned long long)(long long)(IPOPT_RA));
#endif
#ifdef IPOPT_RESERVED1
printf("IPOPT_RESERVED1 %llu\n", (unsigned long long)(long long)(IPOPT_RESERVED1));
#endif
#ifdef IPOPT_RESERVED2
printf("IPOPT_RESERVED2 %llu\n", (unsigned long long)(long long)(IPOPT_RESERVED2));
#endif
#ifdef IPOPT_RR
printf("IPOPT_RR %llu\n", (unsigned long long)(long long)(IPOPT_RR));
#endif
#ifdef IPOPT_SEC
printf("IPOPT_SEC %llu\n", (unsigned long long)(long long)(IPOPT_SEC));
#endif
#ifdef IPOPT_SID
printf("IPOPT_SID %llu\n", (unsigned long long)(long long)(IPOPT_SID));
#endif
#ifdef IPOPT_SSRR
printf("IPOPT_SSRR %llu\n", (unsigned long long)(long long)(IPOPT_SSRR));
#endif
#ifdef IPOPT_TIMESTAMP
printf("IPOPT_TIMESTAMP %llu\n", (unsigned long long)(long long)(IPOPT_TIMESTAMP));
#endif
#ifdef IPOPT_TS
printf("IPOPT_TS %llu\n", (unsigned long long)(long long)(IPOPT_TS));
#endif
#ifdef IPOPT_TS_PRESPEC
printf("IPOPT_TS_PRESPEC %llu\n", (unsigned long long)(long long)(IPOPT_TS_PRESPEC));
#endif
#ifdef IPOPT_TS_TSANDADDR
printf("IPOPT_TS_TSANDADDR %llu\n", (unsigned long long)(long long)(IPOPT_TS_TSANDADDR));
#endif
#ifdef IPOPT_TS_TSONLY
printf("IPOPT_TS_TSONLY %llu\n", (unsigned long long)(long long)(IPOPT_TS_TSONLY));
#endif
#ifdef IPPROTO_AH
printf("IPPROTO_AH %llu\n", (unsigned long long)(long long)(IPPROTO_AH));
#endif
#ifdef IPPROTO_BEETPH
printf("IPPROTO_BEETPH %llu\n", (unsigned long long)(long long)(IPPROTO_BEETPH));
#endif
#ifdef IPPROTO_COMP
printf("IPPROTO_COMP %llu\n", (unsigned long long)(long long)(IPPROTO_COMP));
#endif
#ifdef IPPROTO_DCCP
printf("IPPROTO_DCCP %llu\n", (unsigned long long)(long long)(IPPROTO_DCCP));
#endif
#ifdef IPPROTO_DSTOPTS
printf("IPPROTO_DSTOPTS %llu\n", (unsigned long long)(long long)(IPPROTO_DSTOPTS));
#endif
#ifdef IPPROTO_EGP
printf("IPPROTO_EGP %llu\n", (unsigned long long)(long long)(IPPROTO_EGP));
#endif
#ifdef IPPROTO_ENCAP
printf("IPPROTO_ENCAP %llu\n", (unsigned long long)(long long)(IPPROTO_ENCAP));
#endif
#ifdef IPPROTO_ESP
printf("IPPROTO_ESP %llu\n", (unsigned long long)(long long)(IPPROTO_ESP));
#endif
#ifdef IPPROTO_ETHERNET
printf("IPPROTO_ETHERNET %llu\n", (unsigned long long)(long long)(IPPROTO_ETHERNET));
#endif
#ifdef IPPROTO_FRAGMENT
printf("IPPROTO_FRAGMENT %llu\n", (unsigned long long)(long long)(IPPROTO_FRAGMENT));
#endif
#ifdef IPPROTO_GRE
printf("IPPROTO_GRE %llu\n", (unsigned long long)(long long)(IPPROTO_GRE));
#endif
#ifdef IPPROTO_HOPOPTS
printf("IPPROTO_HOPOPTS %llu\n", (unsigned long long)(long long)(IPPROTO_HOPOPTS));
#endif
#ifdef IPPROTO_ICMP
printf("IPPROTO_ICMP %llu\n", (unsigned long long)(long long)(IPPROTO_ICMP));
#endif
#ifdef IPPROTO_ICMPV6
printf("IPPROTO_ICMPV6 %llu\n", (unsigned long long)(long long)(IPPROTO_ICMPV6));
#endif
#ifdef IPPROTO_IDP
printf("IPPROTO_IDP %llu\n", (unsigned long long)(long long)(IPPROTO_IDP));
#endif
#ifdef IPPROTO_IGMP
printf("IPPROTO_IGMP %llu\n", (unsigned long long)(long long)(IPPROTO_IGMP));
#endif
#ifdef IPPROTO_IP
printf("IPPROTO_IP %llu\n", (unsigned long long)(long long)(IPPROTO_IP));
#endif
#ifdef IPPROTO_IPIP
printf("IPPROTO_IPIP %llu\n", (unsigned long long)(long long)(IPPROTO_IPIP));
#endif
#ifdef IPPROTO_IPV6
printf("IPPROTO_IPV6 %llu\n", (unsigned long long)(long long)(IPPROTO_IPV6));
#endif
#ifdef IPPROTO_MAX
printf("IPPROTO_MAX %llu\n", (unsigned long long)(long long)(IPPROTO_MAX));
#endif
#ifdef IPPROTO_MH
printf("IPPROTO_MH %llu\n", (unsigned long long)(long long)(IPPROTO_MH));
#endif
#ifdef IPPROTO_MPLS
printf("IPPROTO_MPLS %llu\n", (unsigned long long)(long long)(IPPROTO_MPLS));
#endif
#ifdef IPPROTO_MPTCP
printf("IPPROTO_MPTCP %llu\n", (unsigned long long)(long long)(IPPROTO_MPTCP));
#endif
#ifdef IPPROTO_MTP
printf("IPPROTO_MTP %llu\n", (unsigned long long)(long long)(IPPROTO_MTP));
#endif
#ifdef IPPROTO_NONE
printf("IPPROTO_NONE %llu\n", (unsigned long long)(long long)(IPPROTO_NONE));
#endif
#ifdef IPPROTO_PIM
printf("IPPROTO_PIM %llu\n", (unsigned long long)(long long)(IPPROTO_PIM));
#endif
#ifdef IPPROTO_PUP
printf("IPPROTO_PUP %llu\n", (unsigned long long)(long long)(IPPROTO_PUP));
#endif
#ifdef IPPROTO_RAW
printf("IPPROTO_RAW %llu\n", (unsigned long long)(long long)(IPPROTO_RAW));
#endif
#ifdef IPPROTO_ROUTING
printf("IPPROTO_ROUTING %llu\n", (unsigned long long)(long long)(IPPROTO_ROUTING));
#endif
#ifdef IPPROTO_RSVP
printf("IPPROTO_RSVP %llu\n", (unsigned long long)(long long)(IPPROTO_RSVP));
#endif
#ifdef IPPROTO_SCTP
printf("IPPROTO_SCTP %llu\n", (unsigned long long)(long long)(IPPROTO_SCTP));
#endif
#ifdef IPPROTO_TCP
printf("IPPROTO_TCP %llu\n", (unsigned long long)(long long)(IPPROTO_TCP));
#endif
#ifdef IPPROTO_TP
printf("IPPROTO_TP %llu\n", (unsigned long long)(long long)(IPPROTO_TP));
#endif
#ifdef IPPROTO_UDP
printf("IPPROTO_UDP %llu\n", (unsigned long long)(long long)(IPPROTO_UDP));
#endif
#ifdef IPPROTO_UDPLITE
printf("IPPROTO_UDPLITE %llu\n", (unsigned long long)(long long)(IPPROTO_UDPLITE));
#endif
#ifdef IPTOS_ECN_CE
printf("IPTOS_ECN_CE %llu\n", (unsigned long long)(long long)(IPTOS_ECN_CE));
#endif
#ifdef IPTOS_ECN_ECT0
printf("IPTOS_ECN_ECT0 %llu\n", (unsigned long long)(long long)(IPTOS_ECN_ECT0));
#endif
#ifdef IPTOS_ECN_ECT1
printf("IPTOS_ECN_ECT1 %llu\n", (unsigned long long)(long long)(IPTOS_ECN_ECT1));
#endif
#ifdef IPTOS_ECN_MASK
printf("IPTOS_ECN_MASK %llu\n", (unsigned long long)(long long)(IPTOS_ECN_MASK));
#endif
#ifdef IPTOS_ECN_NOT_ECT
printf("IPTOS_ECN_NOT_ECT %llu\n", (unsigned long long)(long long)(IPTOS_ECN_NOT_ECT));
#endif
#ifdef IPTOS_LOWDELAY
printf("IPTOS_LOWDELAY %llu\n", (unsigned long long)(long long)(IPTOS_LOWDELAY));
#endif
#ifdef IPTOS_MINCOST
printf("IPTOS_MINCOST %llu\n", (unsigned long long)(long long)(IPTOS_MINCOST));
#endif
#ifdef IPTOS_PREC_CRITIC_ECP
printf("IPTOS_PREC_CRITIC_ECP %llu\n", (unsigned long long)(long long)(IPTOS_PREC_CRITIC_ECP));
#endif
#ifdef IPTOS_PREC_FLASH
printf("IPTOS_PREC_FLASH %llu\n", (unsigned long long)(long long)(IPTOS_PREC_FLASH));
#endif
#ifdef IPTOS_PREC_FLASHOVERRIDE
printf("IPTOS_PREC_FLASHOVERRIDE %llu\n", (unsigned long long)(long long)(IPTOS_PREC_FLASHOVERRIDE));
#endif
#ifdef IPTOS_PREC_IMMEDIATE
printf("IPTOS_PREC_IMMEDIATE %llu\n", (unsigned long long)(long long)(IPTOS_PREC_IMMEDIATE));
#endif
#ifdef IPTOS_PREC_INTERNETCONTROL
printf("IPTOS_PREC_INTERNETCONTROL %llu\n", (unsigned long long)(long long)(IPTOS_PREC_INTERNETCONTROL));
#endif
#ifdef IPTOS_PREC_MASK
printf("IPTOS_PREC_MASK %llu\n", (unsigned long long)(long long)(IPTOS_PREC_MASK));
#endif
#ifdef IPTOS_PREC_NETCONTROL
printf("IPTOS_PREC_NETCONTROL %llu\n", (unsigned long long)(long long)(IPTOS_PREC_NETCONTROL));
#endif
#ifdef IPTOS_PREC_PRIORITY
printf("IPTOS_PREC_PRIORITY %llu\n", (unsigned long long)(long long)(IPTOS_PREC_PRIORITY));
#endif
#ifdef IPTOS_PREC_ROUTINE
printf("IPTOS_PREC_ROUTINE %llu\n", (unsigned long long)(long long)(IPTOS_PREC_ROUTINE));
#endif
#ifdef IPTOS_RELIABILITY
printf("IPTOS_RELIABILITY %llu\n", (unsigned long long)(long long)(IPTOS_RELIABILITY));
#endif
#ifdef IPTOS_THROUGHPUT
printf("IPTOS_THROUGHPUT %llu\n", (unsigned long long)(long long)(IPTOS_THROUGHPUT));
#endif
#ifdef IPTOS_TOS_MASK
printf("IPTOS_TOS_MASK %llu\n", (unsigned long long)(long long)(IPTOS_TOS_MASK));
#endif
#ifdef IPV6_2292DSTOPTS
printf("IPV6_2292DSTOPTS %llu\n", (unsigned long long)(long long)(IPV6_2292DSTOPTS));
#endif
#ifdef IPV6_2292HOPLIMIT
printf("IPV6_2292HOPLIMIT %llu\n", (unsigned long long)(long long)(IPV6_2292HOPLIMIT));
#endif
#ifdef IPV6_2292HOPOPTS
printf("IPV6_2292HOPOPTS %llu\n", (unsigned long long)(long long)(IPV6_2292HOPOPTS));
#endif
#ifdef IPV6_2292PKTINFO
printf("IPV6_2292PKTINFO %llu\n", (unsigned long long)(long long)(IPV6_2292PKTINFO));
#endif
#ifdef IPV6_2292PKTOPTIONS
printf("IPV6_2292PKTOPTIONS %llu\n", (unsigned long long)(long long)(IPV6_2292PKTOPTIONS));
#endif
#ifdef IPV6_2292RTHDR
printf("IPV6_2292RTHDR %llu\n", (unsigned long long)(long long)(IPV6_2292RTHDR));
#endif
#ifdef IPV6_ADDRFORM
printf("IPV6_ADDRFORM %llu\n", (unsigned long long)(long long)(IPV6_ADDRFORM));
#endif
#ifdef IPV6_ADDR_PREFERENCES
printf("IPV6_ADDR_PREFERENCES %llu\n", (unsigned long long)(long long)(IPV6_ADDR_PREFERENCES));
#endif
#ifdef IPV6_ADD_MEMBERSHIP
printf("IPV6_ADD_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IPV6_ADD_MEMBERSHIP));
#endif
#ifdef IPV6_AUTHHDR
printf("IPV6_AUTHHDR %llu\n", (unsigned long long)(long long)(IPV6_AUTHHDR));
#endif
#ifdef IPV6_AUTOFLOWLABEL
printf("IPV6_AUTOFLOWLABEL %llu\n", (unsigned long long)(long long)(IPV6_AUTOFLOWLABEL));
#endif
#ifdef IPV6_CHECKSUM
printf("IPV6_CHECKSUM %llu\n", (unsigned long long)(long long)(IPV6_CHECKSUM));
#endif
#ifdef IPV6_DONTFRAG
printf("IPV6_DONTFRAG %llu\n", (unsigned long long)(long long)(IPV6_DONTFRAG));
#endif
#ifdef IPV6_DROP_MEMBERSHIP
printf("IPV6_DROP_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IPV6_DROP_MEMBERSHIP));
#endif
#ifdef IPV6_DSTOPTS
printf("IPV6_DSTOPTS %llu\n", (unsigned long long)(long long)(IPV6_DSTOPTS));
#endif
#ifdef IPV6_FLOWINFO
printf("IPV6_FLOWINFO %llu\n", (unsigned long long)(long long)(IPV6_FLOWINFO));
#endif
#ifdef IPV6_FLOWINFO_FLOWLABEL
printf("IPV6_FLOWINFO_FLOWLABEL %llu\n", (unsigned long long)(long long)(IPV6_FLOWINFO_FLOWLABEL));
#endif
#ifdef IPV6_FLOWINFO_PRIORITY
printf("IPV6_FLOWINFO_PRIORITY %llu\n", (unsigned long long)(long long)(IPV6_FLOWINFO_PRIORITY));
#endif
#ifdef IPV6_FLOWINFO_SEND
printf("IPV6_FLOWINFO_SEND %llu\n", (unsigned long long)(long long)(IPV6_FLOWINFO_SEND));
#endif
#ifdef IPV6_FLOWLABEL_MGR
printf("IPV6_FLOWLABEL_MGR %llu\n", (unsigned long long)(long long)(IPV6_FLOWLABEL_MGR));
#endif
#ifdef IPV6_FREEBIND
printf("IPV6_FREEBIND %llu\n", (unsigned long long)(long long)(IPV6_FREEBIND));
#endif
#ifdef IPV6_HDRINCL
printf("IPV6_HDRINCL %llu\n", (unsigned long long)(long long)(IPV6_HDRINCL));
#endif
#ifdef IPV6_HOPLIMIT
printf("IPV6_HOPLIMIT %llu\n", (unsigned long long)(long long)(IPV6_HOPLIMIT));
#endif
#ifdef IPV6_HOPOPTS
printf("IPV6_HOPOPTS %llu\n", (unsigned long long)(long long)(IPV6_HOPOPTS));
#endif
#ifdef IPV6_IPSEC_POLICY
printf("IPV6_IPSEC_POLICY %llu\n", (unsigned long long)(long long)(IPV6_IPSEC_POLICY));
#endif
#ifdef IPV6_JOIN_ANYCAST
printf("IPV6_JOIN_ANYCAST %llu\n", (unsigned long long)(long long)(IPV6_JOIN_ANYCAST));
#endif
#ifdef IPV6_LEAVE_ANYCAST
printf("IPV6_LEAVE_ANYCAST %llu\n", (unsigned long long)(long long)(IPV6_LEAVE_ANYCAST));
#endif
#ifdef IPV6_MINHOPCOUNT
printf("IPV6_MINHOPCOUNT %llu\n", (unsigned long long)(long long)(IPV6_MINHOPCOUNT));
#endif
#ifdef IPV6_MTU
printf("IPV6_MTU %llu\n", (unsigned long long)(long long)(IPV6_MTU));
#endif
#ifdef IPV6_MTU_DISCOVER
printf("IPV6_MTU_DISCOVER %llu\n", (unsigned long long)(long long)(IPV6_MTU_DISCOVER));
#endif
#ifdef IPV6_MULTICAST_ALL
printf("IPV6_MULTICAST_ALL %llu\n", (unsigned long long)(long long)(IPV6_MULTICAST_ALL));
#endif
#ifdef IPV6_MULTICAST_HOPS
printf("IPV6_MULTICAST_HOPS %llu\n", (unsigned long long)(long long)(IPV6_MULTICAST_HOPS));
#endif
#ifdef IPV6_MULTICAST_IF
printf("IPV6_MULTICAST_IF %llu\n", (unsigned long long)(long long)(IPV6_MULTICAST_IF));
#endif
#ifdef IPV6_MULTICAST_LOOP
printf("IPV6_MULTICAST_LOOP %llu\n", (unsigned long long)(long long)(IPV6_MULTICAST_LOOP));
#endif
#ifdef IPV6_NEXTHOP
printf("IPV6_NEXTHOP %llu\n", (unsigned long long)(long long)(IPV6_NEXTHOP));
#endif
#ifdef IPV6_ORIGDSTADDR
printf("IPV6_ORIGDSTADDR %llu\n", (unsigned long long)(long long)(IPV6_ORIGDSTADDR));
#endif
#ifdef IPV6_PATHMTU
printf("IPV6_PATHMTU %llu\n", (unsigned long long)(long long)(IPV6_PATHMTU));
#endif
#ifdef IPV6_PKTINFO
printf("IPV6_PKTINFO %llu\n", (unsigned long long)(long long)(IPV6_PKTINFO));
#endif
#ifdef IPV6_PMTUDISC_DO
printf("IPV6_PMTUDISC_DO %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_DO));
#endif
#ifdef IPV6_PMTUDISC_DONT
printf("IPV6_PMTUDISC_DONT %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_DONT));
#endif
#ifdef IPV6_PMTUDISC_INTERFACE
printf("IPV6_PMTUDISC_INTERFACE %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_INTERFACE));
#endif
#ifdef IPV6_PMTUDISC_OMIT
printf("IPV6_PMTUDISC_OMIT %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_OMIT));
#endif
#ifdef IPV6_PMTUDISC_PROBE
printf("IPV6_PMTUDISC_PROBE %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_PROBE));
#endif
#ifdef IPV6_PMTUDISC_WANT
printf("IPV6_PMTUDISC_WANT %llu\n", (unsigned long long)(long long)(IPV6_PMTUDISC_WANT));
#endif
#ifdef IPV6_PREFER_SRC_CGA
printf("IPV6_PREFER_SRC_CGA %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_CGA));
#endif
#ifdef IPV6_PREFER_SRC_COA
printf("IPV6_PREFER_SRC_COA %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_COA));
#endif
#ifdef IPV6_PREFER_SRC_HOME
printf("IPV6_PREFER_SRC_HOME %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_HOME));
#endif
#ifdef IPV6_PREFER_SRC_NONCGA
printf("IPV6_PREFER_SRC_NONCGA %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_NONCGA));
#endif
#ifdef IPV6_PREFER_SRC_PUBLIC
printf("IPV6_PREFER_SRC_PUBLIC %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_PUBLIC));
#endif
#ifdef IPV6_PREFER_SRC_PUBTMP_DEFAULT
printf("IPV6_PREFER_SRC_PUBTMP_DEFAULT %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_PUBTMP_DEFAULT));
#endif
#ifdef IPV6_PREFER_SRC_TMP
printf("IPV6_PREFER_SRC_TMP %llu\n", (unsigned long long)(long long)(IPV6_PREFER_SRC_TMP));
#endif
#ifdef IPV6_RECVDSTOPTS
printf("IPV6_RECVDSTOPTS %llu\n", (unsigned long long)(long long)(IPV6_RECVDSTOPTS));
#endif
#ifdef IPV6_RECVERR
printf("IPV6_RECVERR %llu\n", (unsigned long long)(long long)(IPV6_RECVERR));
#endif
#ifdef IPV6_RECVFRAGSIZE
printf("IPV6_RECVFRAGSIZE %llu\n", (unsigned long long)(long long)(IPV6_RECVFRAGSIZE));
#endif
#ifdef IPV6_RECVHOPLIMIT
printf("IPV6_RECVHOPLIMIT %llu\n", (unsigned long long)(long long)(IPV6_RECVHOPLIMIT));
#endif
#ifdef IPV6_RECVHOPOPTS
printf("IPV6_RECVHOPOPTS %llu\n", (unsigned long long)(long long)(IPV6_RECVHOPOPTS));
#endif
#ifdef IPV6_RECVORIGDSTADDR
printf("IPV6_RECVORIGDSTADDR %llu\n", (unsigned long long)(long long)(IPV6_RECVORIGDSTADDR));
#endif
#ifdef IPV6_RECVPATHMTU
printf("IPV6_RECVPATHMTU %llu\n", (unsigned long long)(long long)(IPV6_RECVPATHMTU));
#endif
#ifdef IPV6_RECVPKTINFO
printf("IPV6_RECVPKTINFO %llu\n", (unsigned long long)(long long)(IPV6_RECVPKTINFO));
#endif
#ifdef IPV6_RECVRTHDR
printf("IPV6_RECVRTHDR %llu\n", (unsigned long long)(long long)(IPV6_RECVRTHDR));
#endif
#ifdef IPV6_RECVTCLASS
printf("IPV6_RECVTCLASS %llu\n", (unsigned long long)(long long)(IPV6_RECVTCLASS));
#endif
#ifdef IPV6_ROUTER_ALERT
printf("IPV6_ROUTER_ALERT %llu\n", (unsigned long long)(long long)(IPV6_ROUTER_ALERT));
#endif
#ifdef IPV6_ROUTER_ALERT_ISOLATE
printf("IPV6_ROUTER_ALERT_ISOLATE %llu\n", (unsigned long long)(long long)(IPV6_ROUTER_ALERT_ISOLATE));
#endif
#ifdef IPV6_RTHDR
printf("IPV6_RTHDR %llu\n", (unsigned long long)(long long)(IPV6_RTHDR));
#endif
#ifdef IPV6_RTHDRDSTOPTS
printf("IPV6_RTHDRDSTOPTS %llu\n", (unsigned long long)(long long)(IPV6_RTHDRDSTOPTS));
#endif
#ifdef IPV6_RTHDR_LOOSE
printf("IPV6_RTHDR_LOOSE %llu\n", (unsigned long long)(long long)(IPV6_RTHDR_LOOSE));
#endif
#ifdef IPV6_RTHDR_STRICT
printf("IPV6_RTHDR_STRICT %llu\n", (unsigned long long)(long long)(IPV6_RTHDR_STRICT));
#endif
#ifdef IPV6_TCLASS
printf("IPV6_TCLASS %llu\n", (unsigned long long)(long long)(IPV6_TCLASS));
#endif
#ifdef IPV6_TRANSPARENT
printf("IPV6_TRANSPARENT %llu\n", (unsigned long long)(long long)(IPV6_TRANSPARENT));
#endif
#ifdef IPV6_UNICAST_HOPS
printf("IPV6_UNICAST_HOPS %llu\n", (unsigned long long)(long long)(IPV6_UNICAST_HOPS));
#endif
#ifdef IPV6_UNICAST_IF
printf("IPV6_UNICAST_IF %llu\n", (unsigned long long)(long long)(IPV6_UNICAST_IF));
#endif
#ifdef IPV6_V6ONLY
printf("IPV6_V6ONLY %llu\n", (unsigned long long)(long long)(IPV6_V6ONLY));
#endif
#ifdef IPV6_XFRM_POLICY
printf("IPV6_XFRM_POLICY %llu\n", (unsigned long long)(long long)(IPV6_XFRM_POLICY));
#endif
#ifdef IPVERSION
printf("IPVERSION %llu\n", (unsigned long long)(long long)(IPVERSION));
#endif
#ifdef IP_ADD_MEMBERSHIP
printf("IP_ADD_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IP_ADD_MEMBERSHIP));
#endif
#ifdef IP_ADD_SOURCE_MEMBERSHIP
printf("IP_ADD_SOURCE_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IP_ADD_SOURCE_MEMBERSHIP));
#endif
#ifdef IP_BIND_ADDRESS_NO_PORT
printf("IP_BIND_ADDRESS_NO_PORT %llu\n", (unsigned long long)(long long)(IP_BIND_ADDRESS_NO_PORT));
#endif
#ifdef IP_BLOCK_SOURCE
printf("IP_BLOCK_SOURCE %llu\n", (unsigned long long)(long long)(IP_BLOCK_SOURCE));
#endif
#ifdef IP_CHECKSUM
printf("IP_CHECKSUM %llu\n", (unsigned long long)(long long)(IP_CHECKSUM));
#endif
#ifdef IP_DEFAULT_MULTICAST_LOOP
printf("IP_DEFAULT_MULTICAST_LOOP %llu\n", (unsigned long long)(long long)(IP_DEFAULT_MULTICAST_LOOP));
#endif
#ifdef IP_DEFAULT_MULTICAST_TTL
printf("IP_DEFAULT_MULTICAST_TTL %llu\n", (unsigned long long)(long long)(IP_DEFAULT_MULTICAST_TTL));
#endif
#ifdef IP_DROP_MEMBERSHIP
printf("IP_DROP_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IP_DROP_MEMBERSHIP));
#endif
#ifdef IP_DROP_SOURCE_MEMBERSHIP
printf("IP_DROP_SOURCE_MEMBERSHIP %llu\n", (unsigned long long)(long long)(IP_DROP_SOURCE_MEMBERSHIP));
#endif
#ifdef IP_FREEBIND
printf("IP_FREEBIND %llu\n", (unsigned long long)(long long)(IP_FREEBIND));
#endif
#ifdef IP_HDRINCL
printf("IP_HDRINCL %llu\n", (unsigned long long)(long long)(IP_HDRINCL));
#endif
#ifdef IP_IPSEC_POLICY
printf("IP_IPSEC_POLICY %llu\n", (unsigned long long)(long long)(IP_IPSEC_POLICY));
#endif
#ifdef IP_MINTTL
printf("IP_MINTTL %llu\n", (unsigned long long)(long long)(IP_MINTTL));
#endif
#ifdef IP_MSFILTER
printf("IP_MSFILTER %llu\n", (unsigned long long)(long long)(IP_MSFILTER));
#endif
#ifdef IP_MTU
printf("IP_MTU %llu\n", (unsigned long long)(long long)(IP_MTU));
#endif
#ifdef IP_MTU_DISCOVER
printf("IP_MTU_DISCOVER %llu\n", (unsigned long long)(long long)(IP_MTU_DISCOVER));
#endif
#ifdef IP_MULTICAST_ALL
printf("IP_MULTICAST_ALL %llu\n", (unsigned long long)(long long)(IP_MULTICAST_ALL));
#endif
#ifdef IP_MULTICAST_IF
printf("IP_MULTICAST_IF %llu\n", (unsigned long long)(long long)(IP_MULTICAST_IF));
#endif
#ifdef IP_MULTICAST_LOOP
printf("IP_MULTICAST_LOOP %llu\n", (unsigned long long)(long long)(IP_MULTICAST_LOOP));
#endif
#ifdef IP_MULTICAST_TTL
printf("IP_MULTICAST_TTL %llu\n", (unsigned long long)(long long)(IP_MULTICAST_TTL));
#endif
#ifdef IP_NODEFRAG
printf("IP_NODEFRAG %llu\n", (unsigned long long)(long long)(IP_NODEFRAG));
#endif
#ifdef IP_OPTIONS
printf("IP_OPTIONS %llu\n", (unsigned long long)(long long)(IP_OPTIONS));
#endif
#ifdef IP_ORIGDSTADDR
printf("IP_ORIGDSTADDR %llu\n", (unsigned long long)(long long)(IP_ORIGDSTADDR));
#endif
#ifdef IP_PASSSEC
printf("IP_PASSSEC %llu\n", (unsigned long long)(long long)(IP_PASSSEC));
#endif
#ifdef IP_PKTINFO
printf("IP_PKTINFO %llu\n", (unsigned long long)(long long)(IP_PKTINFO));
#endif
#ifdef IP_PKTOPTIONS
printf("IP_PKTOPTIONS %llu\n", (unsigned long long)(long long)(IP_PKTOPTIONS));
#endif
#ifdef IP_PMTUDISC_DO
printf("IP_PMTUDISC_DO %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_DO));
#endif
#ifdef IP_PMTUDISC_DONT
printf("IP_PMTUDISC_DONT %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_DONT));
#endif
#ifdef IP_PMTUDISC_INTERFACE
printf("IP_PMTUDISC_INTERFACE %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_INTERFACE));
#endif
#ifdef IP_PMTUDISC_OMIT
printf("IP_PMTUDISC_OMIT %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_OMIT));
#endif
#ifdef IP_PMTUDISC_PROBE
printf("IP_PMTUDISC_PROBE %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_PROBE));
#endif
#ifdef IP_PMTUDISC_WANT
printf("IP_PMTUDISC_WANT %llu\n", (unsigned long long)(long long)(IP_PMTUDISC_WANT));
#endif
#ifdef IP_RECVERR
printf("IP_RECVERR %llu\n", (unsigned long long)(long long)(IP_RECVERR));
#endif
#ifdef IP_RECVFRAGSIZE
printf("IP_RECVFRAGSIZE %llu\n", (unsigned long long)(long long)(IP_RECVFRAGSIZE));
#endif
#ifdef IP_RECVOPTS
printf("IP_RECVOPTS %llu\n", (unsigned long long)(long long)(IP_RECVOPTS));
#endif
#ifdef IP_RECVORIGDSTADDR
printf("IP_RECVORIGDSTADDR %llu\n", (unsigned long long)(long long)(IP_RECVORIGDSTADDR));
#endif
#ifdef IP_RECVTOS
printf("IP_RECVTOS %llu\n", (unsigned long long)(long long)(IP_RECVTOS));
#endif
#ifdef IP_RECVTTL
printf("IP_RECVTTL %llu\n", (unsigned long long)(long long)(IP_RECVTTL));
#endif
#ifdef IP_RETOPTS
printf("IP_RETOPTS %llu\n", (unsigned long long)(long long)(IP_RETOPTS));
#endif
#ifdef IP_ROUTER_ALERT
printf("IP_ROUTER_ALERT %llu\n", (unsigned long long)(long long)(IP_ROUTER_ALERT));
#endif
#ifdef IP_TOS
printf("IP_TOS %llu\n", (unsigned long long)(long long)(IP_TOS));
#endif
#ifdef IP_TRANSPARENT
printf("IP_TRANSPARENT %llu\n", (unsigned long long)(long long)(IP_TRANSPARENT));
#endif
#ifdef IP_TTL
printf("IP_TTL %llu\n", (unsigned long long)(long long)(IP_TTL));
#endif
#ifdef IP_UNBLOCK_SOURCE
printf("IP_UNBLOCK_SOURCE %llu\n", (unsigned long long)(long long)(IP_UNBLOCK_SOURCE));
#endif
#ifdef IP_UNICAST_IF
printf("IP_UNICAST_IF %llu\n", (unsigned long long)(long long)(IP_UNICAST_IF));
#endif
#ifdef IP_XFRM_POLICY
printf("IP_XFRM_POLICY %llu\n", (unsigned long long)(long long)(IP_XFRM_POLICY));
#endif
#ifdef ISIG
printf("ISIG %llu\n", (unsigned long long)(long long)(ISIG));
#endif
#ifdef ISOFS_SUPER_MAGIC
printf("ISOFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(ISOFS_SUPER_MAGIC));
#endif
#ifdef ISTRIP
printf("ISTRIP %llu\n", (unsigned long long)(long long)(ISTRIP));
#endif
#ifdef ITIMER_PROF
printf("ITIMER_PROF %llu\n", (unsigned long long)(long long)(ITIMER_PROF));
#endif
#ifdef ITIMER_REAL
printf("ITIMER_REAL %llu\n", (unsigned long long)(long long)(ITIMER_REAL));
#endif
#ifdef ITIMER_VIRTUAL
printf("ITIMER_VIRTUAL %llu\n", (unsigned long long)(long long)(ITIMER_VIRTUAL));
#endif
#ifdef IUCLC
printf("IUCLC %llu\n", (unsigned long long)(long long)(IUCLC));
#endif
#ifdef IUTF8
printf("IUTF8 %llu\n", (unsigned long long)(long long)(IUTF8));
#endif
#ifdef IWEVASSOCREQIE
printf("IWEVASSOCREQIE %llu\n", (unsigned long long)(long long)(IWEVASSOCREQIE));
#endif
#ifdef IWEVASSOCRESPIE
printf("IWEVASSOCRESPIE %llu\n", (unsigned long long)(long long)(IWEVASSOCRESPIE));
#endif
#ifdef IWEVCUSTOM
printf("IWEVCUSTOM %llu\n", (unsigned long long)(long long)(IWEVCUSTOM));
#endif
#ifdef IWEVEXPIRED
printf("IWEVEXPIRED %llu\n", (unsigned long long)(long long)(IWEVEXPIRED));
#endif
#ifdef IWEVFIRST
printf("IWEVFIRST %llu\n", (unsigned long long)(long long)(IWEVFIRST));
#endif
#ifdef IWEVGENIE
printf("IWEVGENIE %llu\n", (unsigned long long)(long long)(IWEVGENIE));
#endif
#ifdef IWEVMICHAELMICFAILURE
printf("IWEVMICHAELMICFAILURE %llu\n", (unsigned long long)(long long)(IWEVMICHAELMICFAILURE));
#endif
#ifdef IWEVPMKIDCAND
printf("IWEVPMKIDCAND %llu\n", (unsigned long long)(long long)(IWEVPMKIDCAND));
#endif
#ifdef IWEVQUAL
printf("IWEVQUAL %llu\n", (unsigned long long)(long long)(IWEVQUAL));
#endif
#ifdef IWEVREGISTERED
printf("IWEVREGISTERED %llu\n", (unsigned long long)(long long)(IWEVREGISTERED));
#endif
#ifdef IWEVTXDROP
printf("IWEVTXDROP %llu\n", (unsigned long long)(long long)(IWEVTXDROP));
#endif
#ifdef IW_AUTH_80211_AUTH_ALG
printf("IW_AUTH_80211_AUTH_ALG %llu\n", (unsigned long long)(long long)(IW_AUTH_80211_AUTH_ALG));
#endif
#ifdef IW_AUTH_ALG_LEAP
printf("IW_AUTH_ALG_LEAP %llu\n", (unsigned long long)(long long)(IW_AUTH_ALG_LEAP));
#endif
#ifdef IW_AUTH_ALG_OPEN_SYSTEM
printf("IW_AUTH_ALG_OPEN_SYSTEM %llu\n", (unsigned long long)(long long)(IW_AUTH_ALG_OPEN_SYSTEM));
#endif
#ifdef IW_AUTH_ALG_SHARED_KEY
printf("IW_AUTH_ALG_SHARED_KEY %llu\n", (unsigned long long)(long long)(IW_AUTH_ALG_SHARED_KEY));
#endif
#ifdef IW_AUTH_CIPHER_AES_CMAC
printf("IW_AUTH_CIPHER_AES_CMAC %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_AES_CMAC));
#endif
#ifdef IW_AUTH_CIPHER_CCMP
printf("IW_AUTH_CIPHER_CCMP %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_CCMP));
#endif
#ifdef IW_AUTH_CIPHER_GROUP
printf("IW_AUTH_CIPHER_GROUP %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_GROUP));
#endif
#ifdef IW_AUTH_CIPHER_GROUP_MGMT
printf("IW_AUTH_CIPHER_GROUP_MGMT %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_GROUP_MGMT));
#endif
#ifdef IW_AUTH_CIPHER_NONE
printf("IW_AUTH_CIPHER_NONE %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_NONE));
#endif
#ifdef IW_AUTH_CIPHER_PAIRWISE
printf("IW_AUTH_CIPHER_PAIRWISE %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_PAIRWISE));
#endif
#ifdef IW_AUTH_CIPHER_TKIP
printf("IW_AUTH_CIPHER_TKIP %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_TKIP));
#endif
#ifdef IW_AUTH_CIPHER_WEP104
printf("IW_AUTH_CIPHER_WEP104 %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_WEP104));
#endif
#ifdef IW_AUTH_CIPHER_WEP40
printf("IW_AUTH_CIPHER_WEP40 %llu\n", (unsigned long long)(long long)(IW_AUTH_CIPHER_WEP40));
#endif
#ifdef IW_AUTH_DROP_UNENCRYPTED
printf("IW_AUTH_DROP_UNENCRYPTED %llu\n", (unsigned long long)(long long)(IW_AUTH_DROP_UNENCRYPTED));
#endif
#ifdef IW_AUTH_FLAGS
printf("IW_AUTH_FLAGS %llu\n", (unsigned long long)(long long)(IW_AUTH_FLAGS));
#endif
#ifdef IW_AUTH_INDEX
printf("IW_AUTH_INDEX %llu\n", (unsigned long long)(long long)(IW_AUTH_INDEX));
#endif
#ifdef IW_AUTH_KEY_MGMT
printf("IW_AUTH_KEY_MGMT %llu\n", (unsigned long long)(long long)(IW_AUTH_KEY_MGMT));
#endif
#ifdef IW_AUTH_KEY_MGMT_802_1X
printf("IW_AUTH_KEY_MGMT_802_1X %llu\n", (unsigned long long)(long long)(IW_AUTH_KEY_MGMT_802_1X));
#endif
#ifdef IW_AUTH_KEY_MGMT_PSK
printf("IW_AUTH_KEY_MGMT_PSK %llu\n", (unsigned long long)(long long)(IW_AUTH_KEY_MGMT_PSK));
#endif
#ifdef IW_AUTH_MFP
printf("IW_AUTH_MFP %llu\n", (unsigned long long)(long long)(IW_AUTH_MFP));
#endif
#ifdef IW_AUTH_MFP_DISABLED
printf("IW_AUTH_MFP_DISABLED %llu\n", (unsigned long long)(long long)(IW_AUTH_MFP_DISABLED));
#endif
#ifdef IW_AUTH_MFP_OPTIONAL
printf("IW_AUTH_MFP_OPTIONAL %llu\n", (unsigned long long)(long long)(IW_AUTH_MFP_OPTIONAL));
#endif
#ifdef IW_AUTH_MFP_REQUIRED
printf("IW_AUTH_MFP_REQUIRED %llu\n", (unsigned long long)(long long)(IW_AUTH_MFP_REQUIRED));
#endif
#ifdef IW_AUTH_PRIVACY_INVOKED
printf("IW_AUTH_PRIVACY_INVOKED %llu\n", (unsigned long long)(long long)(IW_AUTH_PRIVACY_INVOKED));
#endif
#ifdef IW_AUTH_ROAMING_CONTROL
printf("IW_AUTH_ROAMING_CONTROL %llu\n", (unsigned long long)(long long)(IW_AUTH_ROAMING_CONTROL));
#endif
#ifdef IW_AUTH_ROAMING_DISABLE
printf("IW_AUTH_ROAMING_DISABLE %llu\n", (unsigned long long)(long long)(IW_AUTH_ROAMING_DISABLE));
#endif
#ifdef IW_AUTH_ROAMING_ENABLE
printf("IW_AUTH_ROAMING_ENABLE %llu\n", (unsigned long long)(long long)(IW_AUTH_ROAMING_ENABLE));
#endif
#ifdef IW_AUTH_RX_UNENCRYPTED_EAPOL
printf("IW_AUTH_RX_UNENCRYPTED_EAPOL %llu\n", (unsigned long long)(long long)(IW_AUTH_RX_UNENCRYPTED_EAPOL));
#endif
#ifdef IW_AUTH_TKIP_COUNTERMEASURES
printf("IW_AUTH_TKIP_COUNTERMEASURES %llu\n", (unsigned long long)(long long)(IW_AUTH_TKIP_COUNTERMEASURES));
#endif
#ifdef IW_AUTH_WPA_ENABLED
printf("IW_AUTH_WPA_ENABLED %llu\n", (unsigned long long)(long long)(IW_AUTH_WPA_ENABLED));
#endif
#ifdef IW_AUTH_WPA_VERSION
printf("IW_AUTH_WPA_VERSION %llu\n", (unsigned long long)(long long)(IW_AUTH_WPA_VERSION));
#endif
#ifdef IW_AUTH_WPA_VERSION_DISABLED
printf("IW_AUTH_WPA_VERSION_DISABLED %llu\n", (unsigned long long)(long long)(IW_AUTH_WPA_VERSION_DISABLED));
#endif
#ifdef IW_AUTH_WPA_VERSION_WPA
printf("IW_AUTH_WPA_VERSION_WPA %llu\n", (unsigned long long)(long long)(IW_AUTH_WPA_VERSION_WPA));
#endif
#ifdef IW_AUTH_WPA_VERSION_WPA2
printf("IW_AUTH_WPA_VERSION_WPA2 %llu\n", (unsigned long long)(long long)(IW_AUTH_WPA_VERSION_WPA2));
#endif
#ifdef IW_CUSTOM_MAX
printf("IW_CUSTOM_MAX %llu\n", (unsigned long long)(long long)(IW_CUSTOM_MAX));
#endif
#ifdef IW_ENCODE_ALG_AES_CMAC
printf("IW_ENCODE_ALG_AES_CMAC %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_AES_CMAC));
#endif
#ifdef IW_ENCODE_ALG_CCMP
printf("IW_ENCODE_ALG_CCMP %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_CCMP));
#endif
#ifdef IW_ENCODE_ALG_NONE
printf("IW_ENCODE_ALG_NONE %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_NONE));
#endif
#ifdef IW_ENCODE_ALG_PMK
printf("IW_ENCODE_ALG_PMK %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_PMK));
#endif
#ifdef IW_ENCODE_ALG_TKIP
printf("IW_ENCODE_ALG_TKIP %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_TKIP));
#endif
#ifdef IW_ENCODE_ALG_WEP
printf("IW_ENCODE_ALG_WEP %llu\n", (unsigned long long)(long long)(IW_ENCODE_ALG_WEP));
#endif
#ifdef IW_ENCODE_DISABLED
printf("IW_ENCODE_DISABLED %llu\n", (unsigned long long)(long long)(IW_ENCODE_DISABLED));
#endif
#ifdef IW_ENCODE_ENABLED
printf("IW_ENCODE_ENABLED %llu\n", (unsigned long long)(long long)(IW_ENCODE_ENABLED));
#endif
#ifdef IW_ENCODE_EXT_GROUP_KEY
printf("IW_ENCODE_EXT_GROUP_KEY %llu\n", (unsigned long long)(long long)(IW_ENCODE_EXT_GROUP_KEY));
#endif
#ifdef IW_ENCODE_EXT_RX_SEQ_VALID
printf("IW_ENCODE_EXT_RX_SEQ_VALID %llu\n", (unsigned long long)(long long)(IW_ENCODE_EXT_RX_SEQ_VALID));
#endif
#ifdef IW_ENCODE_EXT_SET_TX_KEY
printf("IW_ENCODE_EXT_SET_TX_KEY %llu\n", (unsigned long long)(long long)(IW_ENCODE_EXT_SET_TX_KEY));
#endif
#ifdef IW_ENCODE_EXT_TX_SEQ_VALID
printf("IW_ENCODE_EXT_TX_SEQ_VALID %llu\n", (unsigned long long)(long long)(IW_ENCODE_EXT_TX_SEQ_VALID));
#endif
#ifdef IW_ENCODE_FLAGS
printf("IW_ENCODE_FLAGS %llu\n", (unsigned long long)(long long)(IW_ENCODE_FLAGS));
#endif
#ifdef IW_ENCODE_INDEX
printf("IW_ENCODE_INDEX %llu\n", (unsigned long long)(long long)(IW_ENCODE_INDEX));
#endif
#ifdef IW_ENCODE_MODE
printf("IW_ENCODE_MODE %llu\n", (unsigned long long)(long long)(IW_ENCODE_MODE));
#endif
#ifdef IW_ENCODE_NOKEY
printf("IW_ENCODE_NOKEY %llu\n", (unsigned long long)(long long)(IW_ENCODE_NOKEY));
#endif
#ifdef IW_ENCODE_OPEN
printf("IW_ENCODE_OPEN %llu\n", (unsigned long long)(long long)(IW_ENCODE_OPEN));
#endif
#ifdef IW_ENCODE_RESTRICTED
printf("IW_ENCODE_RESTRICTED %llu\n", (unsigned long long)(long long)(IW_ENCODE_RESTRICTED));
#endif
#ifdef IW_ENCODE_SEQ_MAX_SIZE
printf("IW_ENCODE_SEQ_MAX_SIZE %llu\n", (unsigned long long)(long long)(IW_ENCODE_SEQ_MAX_SIZE));
#endif
#ifdef IW_ENCODE_TEMP
printf("IW_ENCODE_TEMP %llu\n", (unsigned long long)(long long)(IW_ENCODE_TEMP));
#endif
#ifdef IW_ENCODING_TOKEN_MAX
printf("IW_ENCODING_TOKEN_MAX %llu\n", (unsigned long long)(long long)(IW_ENCODING_TOKEN_MAX));
#endif
#ifdef IW_ENC_CAPA_4WAY_HANDSHAKE
printf("IW_ENC_CAPA_4WAY_HANDSHAKE %llu\n", (unsigned long long)(long long)(IW_ENC_CAPA_4WAY_HANDSHAKE));
#endif
#ifdef IW_ENC_CAPA_CIPHER_CCMP
printf("IW_ENC_CAPA_CIPHER_CCMP %llu\n", (unsigned long long)(long long)(IW_ENC_CAPA_CIPHER_CCMP));
#endif
#ifdef IW_ENC_CAPA_CIPHER_TKIP
printf("IW_ENC_CAPA_CIPHER_TKIP %llu\n", (unsigned long long)(long long)(IW_ENC_CAPA_CIPHER_TKIP));
#endif
#ifdef IW_ENC_CAPA_WPA
printf("IW_ENC_CAPA_WPA %llu\n", (unsigned long long)(long long)(IW_ENC_CAPA_WPA));
#endif
#ifdef IW_ENC_CAPA_WPA2
printf("IW_ENC_CAPA_WPA2 %llu\n", (unsigned long long)(long long)(IW_ENC_CAPA_WPA2));
#endif
#ifdef IW_ESSID_MAX_SIZE
printf("IW_ESSID_MAX_SIZE %llu\n", (unsigned long long)(long long)(IW_ESSID_MAX_SIZE));
#endif
#ifdef IW_EVENT_CAPA_K_0
printf("IW_EVENT_CAPA_K_0 %llu\n", (unsigned long long)(long long)(IW_EVENT_CAPA_K_0));
#endif
#ifdef IW_EVENT_CAPA_K_1
printf("IW_EVENT_CAPA_K_1 %llu\n", (unsigned long long)(long long)(IW_EVENT_CAPA_K_1));
#endif
#ifdef IW_EV_ADDR_PK_LEN
printf("IW_EV_ADDR_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_ADDR_PK_LEN));
#endif
#ifdef IW_EV_CHAR_PK_LEN
printf("IW_EV_CHAR_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_CHAR_PK_LEN));
#endif
#ifdef IW_EV_FREQ_PK_LEN
printf("IW_EV_FREQ_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_FREQ_PK_LEN));
#endif
#ifdef IW_EV_LCP_PK_LEN
printf("IW_EV_LCP_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_LCP_PK_LEN));
#endif
#ifdef IW_EV_PARAM_PK_LEN
printf("IW_EV_PARAM_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_PARAM_PK_LEN));
#endif
#ifdef IW_EV_POINT_PK_LEN
printf("IW_EV_POINT_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_POINT_PK_LEN));
#endif
#ifdef IW_EV_QUAL_PK_LEN
printf("IW_EV_QUAL_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_QUAL_PK_LEN));
#endif
#ifdef IW_EV_UINT_PK_LEN
printf("IW_EV_UINT_PK_LEN %llu\n", (unsigned long long)(long long)(IW_EV_UINT_PK_LEN));
#endif
#ifdef IW_FREQ_AUTO
printf("IW_FREQ_AUTO %llu\n", (unsigned long long)(long long)(IW_FREQ_AUTO));
#endif
#ifdef IW_FREQ_FIXED
printf("IW_FREQ_FIXED %llu\n", (unsigned long long)(long long)(IW_FREQ_FIXED));
#endif
#ifdef IW_GENERIC_IE_MAX
printf("IW_GENERIC_IE_MAX %llu\n", (unsigned long long)(long long)(IW_GENERIC_IE_MAX));
#endif
#ifdef IW_MAX_AP
printf("IW_MAX_AP %llu\n", (unsigned long long)(long long)(IW_MAX_AP));
#endif
#ifdef IW_MAX_BITRATES
printf("IW_MAX_BITRATES %llu\n", (unsigned long long)(long long)(IW_MAX_BITRATES));
#endif
#ifdef IW_MAX_ENCODING_SIZES
printf("IW_MAX_ENCODING_SIZES %llu\n", (unsigned long long)(long long)(IW_MAX_ENCODING_SIZES));
#endif
#ifdef IW_MAX_FREQUENCIES
printf("IW_MAX_FREQUENCIES %llu\n", (unsigned long long)(long long)(IW_MAX_FREQUENCIES));
#endif
#ifdef IW_MAX_SPY
printf("IW_MAX_SPY %llu\n", (unsigned long long)(long long)(IW_MAX_SPY));
#endif
#ifdef IW_MAX_TXPOWER
printf("IW_MAX_TXPOWER %llu\n", (unsigned long long)(long long)(IW_MAX_TXPOWER));
#endif
#ifdef IW_MICFAILURE_COUNT
printf("IW_MICFAILURE_COUNT %llu\n", (unsigned long long)(long long)(IW_MICFAILURE_COUNT));
#endif
#ifdef IW_MICFAILURE_GROUP
printf("IW_MICFAILURE_GROUP %llu\n", (unsigned long long)(long long)(IW_MICFAILURE_GROUP));
#endif
#ifdef IW_MICFAILURE_KEY_ID
printf("IW_MICFAILURE_KEY_ID %llu\n", (unsigned long long)(long long)(IW_MICFAILURE_KEY_ID));
#endif
#ifdef IW_MICFAILURE_PAIRWISE
printf("IW_MICFAILURE_PAIRWISE %llu\n", (unsigned long long)(long long)(IW_MICFAILURE_PAIRWISE));
#endif
#ifdef IW_MICFAILURE_STAKEY
printf("IW_MICFAILURE_STAKEY %llu\n", (unsigned long long)(long long)(IW_MICFAILURE_STAKEY));
#endif
#ifdef IW_MLME_ASSOC
printf("IW_MLME_ASSOC %llu\n", (unsigned long long)(long long)(IW_MLME_ASSOC));
#endif
#ifdef IW_MLME_AUTH
printf("IW_MLME_AUTH %llu\n", (unsigned long long)(long long)(IW_MLME_AUTH));
#endif
#ifdef IW_MLME_DEAUTH
printf("IW_MLME_DEAUTH %llu\n", (unsigned long long)(long long)(IW_MLME_DEAUTH));
#endif
#ifdef IW_MLME_DISASSOC
printf("IW_MLME_DISASSOC %llu\n", (unsigned long long)(long long)(IW_MLME_DISASSOC));
#endif
#ifdef IW_MODE_ADHOC
printf("IW_MODE_ADHOC %llu\n", (unsigned long long)(long long)(IW_MODE_ADHOC));
#endif
#ifdef IW_MODE_AUTO
printf("IW_MODE_AUTO %llu\n", (unsigned long long)(long long)(IW_MODE_AUTO));
#endif
#ifdef IW_MODE_INFRA
printf("IW_MODE_INFRA %llu\n", (unsigned long long)(long long)(IW_MODE_INFRA));
#endif
#ifdef IW_MODE_MASTER
printf("IW_MODE_MASTER %llu\n", (unsigned long long)(long long)(IW_MODE_MASTER));
#endif
#ifdef IW_MODE_MESH
printf("IW_MODE_MESH %llu\n", (unsigned long long)(long long)(IW_MODE_MESH));
#endif
#ifdef IW_MODE_MONITOR
printf("IW_MODE_MONITOR %llu\n", (unsigned long long)(long long)(IW_MODE_MONITOR));
#endif
#ifdef IW_MODE_REPEAT
printf("IW_MODE_REPEAT %llu\n", (unsigned long long)(long long)(IW_MODE_REPEAT));
#endif
#ifdef IW_MODE_SECOND
printf("IW_MODE_SECOND %llu\n", (unsigned long long)(long long)(IW_MODE_SECOND));
#endif
#ifdef IW_PMKID_CAND_PREAUTH
printf("IW_PMKID_CAND_PREAUTH %llu\n", (unsigned long long)(long long)(IW_PMKID_CAND_PREAUTH));
#endif
#ifdef IW_PMKID_LEN
printf("IW_PMKID_LEN %llu\n", (unsigned long long)(long long)(IW_PMKID_LEN));
#endif
#ifdef IW_PMKSA_ADD
printf("IW_PMKSA_ADD %llu\n", (unsigned long long)(long long)(IW_PMKSA_ADD));
#endif
#ifdef IW_PMKSA_FLUSH
printf("IW_PMKSA_FLUSH %llu\n", (unsigned long long)(long long)(IW_PMKSA_FLUSH));
#endif
#ifdef IW_PMKSA_REMOVE
printf("IW_PMKSA_REMOVE %llu\n", (unsigned long long)(long long)(IW_PMKSA_REMOVE));
#endif
#ifdef IW_POWER_ALL_R
printf("IW_POWER_ALL_R %llu\n", (unsigned long long)(long long)(IW_POWER_ALL_R));
#endif
#ifdef IW_POWER_FORCE_S
printf("IW_POWER_FORCE_S %llu\n", (unsigned long long)(long long)(IW_POWER_FORCE_S));
#endif
#ifdef IW_POWER_MAX
printf("IW_POWER_MAX %llu\n", (unsigned long long)(long long)(IW_POWER_MAX));
#endif
#ifdef IW_POWER_MIN
printf("IW_POWER_MIN %llu\n", (unsigned long long)(long long)(IW_POWER_MIN));
#endif
#ifdef IW_POWER_MODE
printf("IW_POWER_MODE %llu\n", (unsigned long long)(long long)(IW_POWER_MODE));
#endif
#ifdef IW_POWER_MODIFIER
printf("IW_POWER_MODIFIER %llu\n", (unsigned long long)(long long)(IW_POWER_MODIFIER));
#endif
#ifdef IW_POWER_MULTICAST_R
printf("IW_POWER_MULTICAST_R %llu\n", (unsigned long long)(long long)(IW_POWER_MULTICAST_R));
#endif
#ifdef IW_POWER_ON
printf("IW_POWER_ON %llu\n", (unsigned long long)(long long)(IW_POWER_ON));
#endif
#ifdef IW_POWER_PERIOD
printf("IW_POWER_PERIOD %llu\n", (unsigned long long)(long long)(IW_POWER_PERIOD));
#endif
#ifdef IW_POWER_RELATIVE
printf("IW_POWER_RELATIVE %llu\n", (unsigned long long)(long long)(IW_POWER_RELATIVE));
#endif
#ifdef IW_POWER_REPEATER
printf("IW_POWER_REPEATER %llu\n", (unsigned long long)(long long)(IW_POWER_REPEATER));
#endif
#ifdef IW_POWER_TIMEOUT
printf("IW_POWER_TIMEOUT %llu\n", (unsigned long long)(long long)(IW_POWER_TIMEOUT));
#endif
#ifdef IW_POWER_TYPE
printf("IW_POWER_TYPE %llu\n", (unsigned long long)(long long)(IW_POWER_TYPE));
#endif
#ifdef IW_POWER_UNICAST_R
printf("IW_POWER_UNICAST_R %llu\n", (unsigned long long)(long long)(IW_POWER_UNICAST_R));
#endif
#ifdef IW_PRIV_SIZE_FIXED
printf("IW_PRIV_SIZE_FIXED %llu\n", (unsigned long long)(long long)(IW_PRIV_SIZE_FIXED));
#endif
#ifdef IW_PRIV_SIZE_MASK
printf("IW_PRIV_SIZE_MASK %llu\n", (unsigned long long)(long long)(IW_PRIV_SIZE_MASK));
#endif
#ifdef IW_PRIV_TYPE_ADDR
printf("IW_PRIV_TYPE_ADDR %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_ADDR));
#endif
#ifdef IW_PRIV_TYPE_BYTE
printf("IW_PRIV_TYPE_BYTE %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_BYTE));
#endif
#ifdef IW_PRIV_TYPE_CHAR
printf("IW_PRIV_TYPE_CHAR %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_CHAR));
#endif
#ifdef IW_PRIV_TYPE_FLOAT
printf("IW_PRIV_TYPE_FLOAT %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_FLOAT));
#endif
#ifdef IW_PRIV_TYPE_INT
printf("IW_PRIV_TYPE_INT %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_INT));
#endif
#ifdef IW_PRIV_TYPE_MASK
printf("IW_PRIV_TYPE_MASK %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_MASK));
#endif
#ifdef IW_PRIV_TYPE_NONE
printf("IW_PRIV_TYPE_NONE %llu\n", (unsigned long long)(long long)(IW_PRIV_TYPE_NONE));
#endif
#ifdef IW_QUAL_ALL_INVALID
printf("IW_QUAL_ALL_INVALID %llu\n", (unsigned long long)(long long)(IW_QUAL_ALL_INVALID));
#endif
#ifdef IW_QUAL_ALL_UPDATED
printf("IW_QUAL_ALL_UPDATED %llu\n", (unsigned long long)(long long)(IW_QUAL_ALL_UPDATED));
#endif
#ifdef IW_QUAL_DBM
printf("IW_QUAL_DBM %llu\n", (unsigned long long)(long long)(IW_QUAL_DBM));
#endif
#ifdef IW_QUAL_LEVEL_INVALID
printf("IW_QUAL_LEVEL_INVALID %llu\n", (unsigned long long)(long long)(IW_QUAL_LEVEL_INVALID));
#endif
#ifdef IW_QUAL_LEVEL_UPDATED
printf("IW_QUAL_LEVEL_UPDATED %llu\n", (unsigned long long)(long long)(IW_QUAL_LEVEL_UPDATED));
#endif
#ifdef IW_QUAL_NOISE_INVALID
printf("IW_QUAL_NOISE_INVALID %llu\n", (unsigned long long)(long long)(IW_QUAL_NOISE_INVALID));
#endif
#ifdef IW_QUAL_NOISE_UPDATED
printf("IW_QUAL_NOISE_UPDATED %llu\n", (unsigned long long)(long long)(IW_QUAL_NOISE_UPDATED));
#endif
#ifdef IW_QUAL_QUAL_INVALID
printf("IW_QUAL_QUAL_INVALID %llu\n", (unsigned long long)(long long)(IW_QUAL_QUAL_INVALID));
#endif
#ifdef IW_QUAL_QUAL_UPDATED
printf("IW_QUAL_QUAL_UPDATED %llu\n", (unsigned long long)(long long)(IW_QUAL_QUAL_UPDATED));
#endif
#ifdef IW_QUAL_RCPI
printf("IW_QUAL_RCPI %llu\n", (unsigned long long)(long long)(IW_QUAL_RCPI));
#endif
#ifdef IW_RETRY_LIFETIME
printf("IW_RETRY_LIFETIME %llu\n", (unsigned long long)(long long)(IW_RETRY_LIFETIME));
#endif
#ifdef IW_RETRY_LIMIT
printf("IW_RETRY_LIMIT %llu\n", (unsigned long long)(long long)(IW_RETRY_LIMIT));
#endif
#ifdef IW_RETRY_LONG
printf("IW_RETRY_LONG %llu\n", (unsigned long long)(long long)(IW_RETRY_LONG));
#endif
#ifdef IW_RETRY_MAX
printf("IW_RETRY_MAX %llu\n", (unsigned long long)(long long)(IW_RETRY_MAX));
#endif
#ifdef IW_RETRY_MIN
printf("IW_RETRY_MIN %llu\n", (unsigned long long)(long long)(IW_RETRY_MIN));
#endif
#ifdef IW_RETRY_MODIFIER
printf("IW_RETRY_MODIFIER %llu\n", (unsigned long long)(long long)(IW_RETRY_MODIFIER));
#endif
#ifdef IW_RETRY_ON
printf("IW_RETRY_ON %llu\n", (unsigned long long)(long long)(IW_RETRY_ON));
#endif
#ifdef IW_RETRY_RELATIVE
printf("IW_RETRY_RELATIVE %llu\n", (unsigned long long)(long long)(IW_RETRY_RELATIVE));
#endif
#ifdef IW_RETRY_SHORT
printf("IW_RETRY_SHORT %llu\n", (unsigned long long)(long long)(IW_RETRY_SHORT));
#endif
#ifdef IW_RETRY_TYPE
printf("IW_RETRY_TYPE %llu\n", (unsigned long long)(long long)(IW_RETRY_TYPE));
#endif
#ifdef IW_SCAN_ALL_ESSID
printf("IW_SCAN_ALL_ESSID %llu\n", (unsigned long long)(long long)(IW_SCAN_ALL_ESSID));
#endif
#ifdef IW_SCAN_ALL_FREQ
printf("IW_SCAN_ALL_FREQ %llu\n", (unsigned long long)(long long)(IW_SCAN_ALL_FREQ));
#endif
#ifdef IW_SCAN_ALL_MODE
printf("IW_SCAN_ALL_MODE %llu\n", (unsigned long long)(long long)(IW_SCAN_ALL_MODE));
#endif
#ifdef IW_SCAN_ALL_RATE
printf("IW_SCAN_ALL_RATE %llu\n", (unsigned long long)(long long)(IW_SCAN_ALL_RATE));
#endif
#ifdef IW_SCAN_CAPA_BSSID
printf("IW_SCAN_CAPA_BSSID %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_BSSID));
#endif
#ifdef IW_SCAN_CAPA_CHANNEL
printf("IW_SCAN_CAPA_CHANNEL %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_CHANNEL));
#endif
#ifdef IW_SCAN_CAPA_ESSID
printf("IW_SCAN_CAPA_ESSID %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_ESSID));
#endif
#ifdef IW_SCAN_CAPA_MODE
printf("IW_SCAN_CAPA_MODE %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_MODE));
#endif
#ifdef IW_SCAN_CAPA_NONE
printf("IW_SCAN_CAPA_NONE %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_NONE));
#endif
#ifdef IW_SCAN_CAPA_RATE
printf("IW_SCAN_CAPA_RATE %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_RATE));
#endif
#ifdef IW_SCAN_CAPA_TIME
printf("IW_SCAN_CAPA_TIME %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_TIME));
#endif
#ifdef IW_SCAN_CAPA_TYPE
printf("IW_SCAN_CAPA_TYPE %llu\n", (unsigned long long)(long long)(IW_SCAN_CAPA_TYPE));
#endif
#ifdef IW_SCAN_DEFAULT
printf("IW_SCAN_DEFAULT %llu\n", (unsigned long long)(long long)(IW_SCAN_DEFAULT));
#endif
#ifdef IW_SCAN_MAX_DATA
printf("IW_SCAN_MAX_DATA %llu\n", (unsigned long long)(long long)(IW_SCAN_MAX_DATA));
#endif
#ifdef IW_SCAN_THIS_ESSID
printf("IW_SCAN_THIS_ESSID %llu\n", (unsigned long long)(long long)(IW_SCAN_THIS_ESSID));
#endif
#ifdef IW_SCAN_THIS_FREQ
printf("IW_SCAN_THIS_FREQ %llu\n", (unsigned long long)(long long)(IW_SCAN_THIS_FREQ));
#endif
#ifdef IW_SCAN_THIS_MODE
printf("IW_SCAN_THIS_MODE %llu\n", (unsigned long long)(long long)(IW_SCAN_THIS_MODE));
#endif
#ifdef IW_SCAN_THIS_RATE
printf("IW_SCAN_THIS_RATE %llu\n", (unsigned long long)(long long)(IW_SCAN_THIS_RATE));
#endif
#ifdef IW_SCAN_TYPE_ACTIVE
printf("IW_SCAN_TYPE_ACTIVE %llu\n", (unsigned long long)(long long)(IW_SCAN_TYPE_ACTIVE));
#endif
#ifdef IW_SCAN_TYPE_PASSIVE
printf("IW_SCAN_TYPE_PASSIVE %llu\n", (unsigned long long)(long long)(IW_SCAN_TYPE_PASSIVE));
#endif
#ifdef IW_TXPOW_DBM
printf("IW_TXPOW_DBM %llu\n", (unsigned long long)(long long)(IW_TXPOW_DBM));
#endif
#ifdef IW_TXPOW_MWATT
printf("IW_TXPOW_MWATT %llu\n", (unsigned long long)(long long)(IW_TXPOW_MWATT));
#endif
#ifdef IW_TXPOW_RANGE
printf("IW_TXPOW_RANGE %llu\n", (unsigned long long)(long long)(IW_TXPOW_RANGE));
#endif
#ifdef IW_TXPOW_RELATIVE
printf("IW_TXPOW_RELATIVE %llu\n", (unsigned long long)(long long)(IW_TXPOW_RELATIVE));
#endif
#ifdef IW_TXPOW_TYPE
printf("IW_TXPOW_TYPE %llu\n", (unsigned long long)(long long)(IW_TXPOW_TYPE));
#endif
#ifdef IXANY
printf("IXANY %llu\n", (unsigned long long)(long long)(IXANY));
#endif
#ifdef IXOFF
printf("IXOFF %llu\n", (unsigned long long)(long long)(IXOFF));
#endif
#ifdef IXON
printf("IXON %llu\n", (unsigned long long)(long long)(IXON));
#endif
#ifdef JFFS2_SUPER_MAGIC
printf("JFFS2_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(JFFS2_SUPER_MAGIC));
#endif
#ifdef KERN_ACCT
printf("KERN_ACCT %llu\n", (unsigned long long)(long long)(KERN_ACCT));
#endif
#ifdef KERN_ACPI_VIDEO_FLAGS
printf("KERN_ACPI_VIDEO_FLAGS %llu\n", (unsigned long long)(long long)(KERN_ACPI_VIDEO_FLAGS));
#endif
#ifdef KERN_BOOTLOADER_TYPE
printf("KERN_BOOTLOADER_TYPE %llu\n", (unsigned long long)(long long)(KERN_BOOTLOADER_TYPE));
#endif
#ifdef KERN_CADPID
printf("KERN_CADPID %llu\n", (unsigned long long)(long long)(KERN_CADPID));
#endif
#ifdef KERN_COMPAT_LOG
printf("KERN_COMPAT_LOG %llu\n", (unsigned long long)(long long)(KERN_COMPAT_LOG));
#endif
#ifdef KERN_CORE_PATTERN
printf("KERN_CORE_PATTERN %llu\n", (unsigned long long)(long long)(KERN_CORE_PATTERN));
#endif
#ifdef KERN_CORE_USES_PID
printf("KERN_CORE_USES_PID %llu\n", (unsigned long long)(long long)(KERN_CORE_USES_PID));
#endif
#ifdef KERN_CTLALTDEL
printf("KERN_CTLALTDEL %llu\n", (unsigned long long)(long long)(KERN_CTLALTDEL));
#endif
#ifdef KERN_DOMAINNAME
printf("KERN_DOMAINNAME %llu\n", (unsigned long long)(long long)(KERN_DOMAINNAME));
#endif
#ifdef KERN_HOTPLUG
printf("KERN_HOTPLUG %llu\n", (unsigned long long)(long long)(KERN_HOTPLUG));
#endif
#ifdef KERN_HPPA_PWRSW
printf("KERN_HPPA_PWRSW %llu\n", (unsigned long long)(long long)(KERN_HPPA_PWRSW));
#endif
#ifdef KERN_HPPA_UNALIGNED
printf("KERN_HPPA_UNALIGNED %llu\n", (unsigned long long)(long long)(KERN_HPPA_UNALIGNED));
#endif
#ifdef KERN_HZ_TIMER
printf("KERN_HZ_TIMER %llu\n", (unsigned long long)(long long)(KERN_HZ_TIMER));
#endif
#ifdef KERN_IA64_UNALIGNED
printf("KERN_IA64_UNALIGNED %llu\n", (unsigned long long)(long long)(KERN_IA64_UNALIGNED));
#endif
#ifdef KERN_IEEE_EMULATION_WARNINGS
printf("KERN_IEEE_EMULATION_WARNINGS %llu\n", (unsigned long long)(long long)(KERN_IEEE_EMULATION_WARNINGS));
#endif
#ifdef KERN_MAX_LOCK_DEPTH
printf("KERN_MAX_LOCK_DEPTH %llu\n", (unsigned long long)(long long)(KERN_MAX_LOCK_DEPTH));
#endif
#ifdef KERN_MAX_THREADS
printf("KERN_MAX_THREADS %llu\n", (unsigned long long)(long long)(KERN_MAX_THREADS));
#endif
#ifdef KERN_MODPROBE
printf("KERN_MODPROBE %llu\n", (unsigned long long)(long long)(KERN_MODPROBE));
#endif
#ifdef KERN_MSGMAX
printf("KERN_MSGMAX %llu\n", (unsigned long long)(long long)(KERN_MSGMAX));
#endif
#ifdef KERN_MSGMNB
printf("KERN_MSGMNB %llu\n", (unsigned long long)(long long)(KERN_MSGMNB));
#endif
#ifdef KERN_MSGMNI
printf("KERN_MSGMNI %llu\n", (unsigned long long)(long long)(KERN_MSGMNI));
#endif
#ifdef KERN_MSGPOOL
printf("KERN_MSGPOOL %llu\n", (unsigned long long)(long long)(KERN_MSGPOOL));
#endif
#ifdef KERN_NAMETRANS
printf("KERN_NAMETRANS %llu\n", (unsigned long long)(long long)(KERN_NAMETRANS));
#endif
#ifdef KERN_NGROUPS_MAX
printf("KERN_NGROUPS_MAX %llu\n", (unsigned long long)(long long)(KERN_NGROUPS_MAX));
#endif
#ifdef KERN_NMI_WATCHDOG
printf("KERN_NMI_WATCHDOG %llu\n", (unsigned long long)(long long)(KERN_NMI_WATCHDOG));
#endif
#ifdef KERN_NODENAME
printf("KERN_NODENAME %llu\n", (unsigned long long)(long long)(KERN_NODENAME));
#endif
#ifdef KERN_OSRELEASE
printf("KERN_OSRELEASE %llu\n", (unsigned long long)(long long)(KERN_OSRELEASE));
#endif
#ifdef KERN_OSREV
printf("KERN_OSREV %llu\n", (unsigned long long)(long long)(KERN_OSREV));
#endif
#ifdef KERN_OSTYPE
printf("KERN_OSTYPE %llu\n", (unsigned long long)(long long)(KERN_OSTYPE));
#endif
#ifdef KERN_OVERFLOWGID
printf("KERN_OVERFLOWGID %llu\n", (unsigned long long)(long long)(KERN_OVERFLOWGID));
#endif
#ifdef KERN_OVERFLOWUID
printf("KERN_OVERFLOWUID %llu\n", (unsigned long long)(long long)(KERN_OVERFLOWUID));
#endif
#ifdef KERN_PANIC
printf("KERN_PANIC %llu\n", (unsigned long long)(long long)(KERN_PANIC));
#endif
#ifdef KERN_PANIC_ON_NMI
printf("KERN_PANIC_ON_NMI %llu\n", (unsigned long long)(long long)(KERN_PANIC_ON_NMI));
#endif
#ifdef KERN_PANIC_ON_OOPS
printf("KERN_PANIC_ON_OOPS %llu\n", (unsigned long long)(long long)(KERN_PANIC_ON_OOPS));
#endif
#ifdef KERN_PIDMAX
printf("KERN_PIDMAX %llu\n", (unsigned long long)(long long)(KERN_PIDMAX));
#endif
#ifdef KERN_PPC_HTABRECLAIM
printf("KERN_PPC_HTABRECLAIM %llu\n", (unsigned long long)(long long)(KERN_PPC_HTABRECLAIM));
#endif
#ifdef KERN_PPC_L2CR
printf("KERN_PPC_L2CR %llu\n", (unsigned long long)(long long)(KERN_PPC_L2CR));
#endif
#ifdef KERN_PPC_POWERSAVE_NAP
printf("KERN_PPC_POWERSAVE_NAP %llu\n", (unsigned long long)(long long)(KERN_PPC_POWERSAVE_NAP));
#endif
#ifdef KERN_PPC_ZEROPAGED
printf("KERN_PPC_ZEROPAGED %llu\n", (unsigned long long)(long long)(KERN_PPC_ZEROPAGED));
#endif
#ifdef KERN_PRINTK
printf("KERN_PRINTK %llu\n", (unsigned long long)(long long)(KERN_PRINTK));
#endif
#ifdef KERN_PRINTK_RATELIMIT
printf("KERN_PRINTK_RATELIMIT %llu\n", (unsigned long long)(long long)(KERN_PRINTK_RATELIMIT));
#endif
#ifdef KERN_PRINTK_RATELIMIT_BURST
printf("KERN_PRINTK_RATELIMIT_BURST %llu\n", (unsigned long long)(long long)(KERN_PRINTK_RATELIMIT_BURST));
#endif
#ifdef KERN_PROF
printf("KERN_PROF %llu\n", (unsigned long long)(long long)(KERN_PROF));
#endif
#ifdef KERN_PTY
printf("KERN_PTY %llu\n", (unsigned long long)(long long)(KERN_PTY));
#endif
#ifdef KERN_RANDOM
printf("KERN_RANDOM %llu\n", (unsigned long long)(long long)(KERN_RANDOM));
#endif
#ifdef KERN_RANDOMIZE
printf("KERN_RANDOMIZE %llu\n", (unsigned long long)(long long)(KERN_RANDOMIZE));
#endif
#ifdef KERN_REALROOTDEV
printf("KERN_REALROOTDEV %llu\n", (unsigned long long)(long long)(KERN_REALROOTDEV));
#endif
#ifdef KERN_RTSIGMAX
printf("KERN_RTSIGMAX %llu\n", (unsigned long long)(long long)(KERN_RTSIGMAX));
#endif
#ifdef KERN_RTSIGNR
printf("KERN_RTSIGNR %llu\n", (unsigned long long)(long long)(KERN_RTSIGNR));
#endif
#ifdef KERN_S390_USER_DEBUG_LOGGING
printf("KERN_S390_USER_DEBUG_LOGGING %llu\n", (unsigned long long)(long long)(KERN_S390_USER_DEBUG_LOGGING));
#endif
#ifdef KERN_SECUREMASK
printf("KERN_SECUREMASK %llu\n", (unsigned long long)(long long)(KERN_SECUREMASK));
#endif
#ifdef KERN_SEM
printf("KERN_SEM %llu\n", (unsigned long long)(long long)(KERN_SEM));
#endif
#ifdef KERN_SETUID_DUMPABLE
printf("KERN_SETUID_DUMPABLE %llu\n", (unsigned long long)(long long)(KERN_SETUID_DUMPABLE));
#endif
#ifdef KERN_SG_BIG_BUFF
printf("KERN_SG_BIG_BUFF %llu\n", (unsigned long long)(long long)(KERN_SG_BIG_BUFF));
#endif
#ifdef KERN_SHMALL
printf("KERN_SHMALL %llu\n", (unsigned long long)(long long)(KERN_SHMALL));
#endif
#ifdef KERN_SHMMAX
printf("KERN_SHMMAX %llu\n", (unsigned long long)(long long)(KERN_SHMMAX));
#endif
#ifdef KERN_SHMMNI
printf("KERN_SHMMNI %llu\n", (unsigned long long)(long long)(KERN_SHMMNI));
#endif
#ifdef KERN_SHMPATH
printf("KERN_SHMPATH %llu\n", (unsigned long long)(long long)(KERN_SHMPATH));
#endif
#ifdef KERN_SPARC_REBOOT
printf("KERN_SPARC_REBOOT %llu\n", (unsigned long long)(long long)(KERN_SPARC_REBOOT));
#endif
#ifdef KERN_SPARC_SCONS_PWROFF
printf("KERN_SPARC_SCONS_PWROFF %llu\n", (unsigned long long)(long long)(KERN_SPARC_SCONS_PWROFF));
#endif
#ifdef KERN_SPARC_STOP_A
printf("KERN_SPARC_STOP_A %llu\n", (unsigned long long)(long long)(KERN_SPARC_STOP_A));
#endif
#ifdef KERN_SPIN_RETRY
printf("KERN_SPIN_RETRY %llu\n", (unsigned long long)(long long)(KERN_SPIN_RETRY));
#endif
#ifdef KERN_SYSRQ
printf("KERN_SYSRQ %llu\n", (unsigned long long)(long long)(KERN_SYSRQ));
#endif
#ifdef KERN_TAINTED
printf("KERN_TAINTED %llu\n", (unsigned long long)(long long)(KERN_TAINTED));
#endif
#ifdef KERN_UNKNOWN_NMI_PANIC
printf("KERN_UNKNOWN_NMI_PANIC %llu\n", (unsigned long long)(long long)(KERN_UNKNOWN_NMI_PANIC));
#endif
#ifdef KERN_VERSION
printf("KERN_VERSION %llu\n", (unsigned long long)(long long)(KERN_VERSION));
#endif
#ifdef KEXEC_ARCH_MASK
printf("KEXEC_ARCH_MASK %llu\n", (unsigned long long)(long long)(KEXEC_ARCH_MASK));
#endif
#ifdef KEXEC_FILE_NO_INITRAMFS
printf("KEXEC_FILE_NO_INITRAMFS %llu\n", (unsigned long long)(long long)(KEXEC_FILE_NO_INITRAMFS));
#endif
#ifdef KEXEC_FILE_ON_CRASH
printf("KEXEC_FILE_ON_CRASH %llu\n", (unsigned long long)(long long)(KEXEC_FILE_ON_CRASH));
#endif
#ifdef KEXEC_FILE_UNLOAD
printf("KEXEC_FILE_UNLOAD %llu\n", (unsigned long long)(long long)(KEXEC_FILE_UNLOAD));
#endif
#ifdef KEXEC_ON_CRASH
printf("KEXEC_ON_CRASH %llu\n", (unsigned long long)(long long)(KEXEC_ON_CRASH));
#endif
#ifdef KEXEC_PRESERVE_CONTEXT
printf("KEXEC_PRESERVE_CONTEXT %llu\n", (unsigned long long)(long long)(KEXEC_PRESERVE_CONTEXT));
#endif
#ifdef KEY_CNT
printf("KEY_CNT %llu\n", (unsigned long long)(long long)(KEY_CNT));
#endif
#ifdef KEY_MAX
printf("KEY_MAX %llu\n", (unsigned long long)(long long)(KEY_MAX));
#endif
#ifdef LC_ADDRESS
printf("LC_ADDRESS %llu\n", (unsigned long long)(long long)(LC_ADDRESS));
#endif
#ifdef LC_ADDRESS_MASK
printf("LC_ADDRESS_MASK %llu\n", (unsigned long long)(long long)(LC_ADDRESS_MASK));
#endif
#ifdef LC_ALL
printf("LC_ALL %llu\n", (unsigned long long)(long long)(LC_ALL));
#endif
#ifdef LC_ALL_MASK
printf("LC_ALL_MASK %llu\n", (unsigned long long)(long long)(LC_ALL_MASK));
#endif
#ifdef LC_COLLATE
printf("LC_COLLATE %llu\n", (unsigned long long)(long long)(LC_COLLATE));
#endif
#ifdef LC_COLLATE_MASK
printf("LC_COLLATE_MASK %llu\n", (unsigned long long)(long long)(LC_COLLATE_MASK));
#endif
#ifdef LC_CTYPE
printf("LC_CTYPE %llu\n", (unsigned long long)(long long)(LC_CTYPE));
#endif
#ifdef LC_CTYPE_MASK
printf("LC_CTYPE_MASK %llu\n", (unsigned long long)(long long)(LC_CTYPE_MASK));
#endif
#ifdef LC_IDENTIFICATION
printf("LC_IDENTIFICATION %llu\n", (unsigned long long)(long long)(LC_IDENTIFICATION));
#endif
#ifdef LC_IDENTIFICATION_MASK
printf("LC_IDENTIFICATION_MASK %llu\n", (unsigned long long)(long long)(LC_IDENTIFICATION_MASK));
#endif
#ifdef LC_MEASUREMENT
printf("LC_MEASUREMENT %llu\n", (unsigned long long)(long long)(LC_MEASUREMENT));
#endif
#ifdef LC_MEASUREMENT_MASK
printf("LC_MEASUREMENT_MASK %llu\n", (unsigned long long)(long long)(LC_MEASUREMENT_MASK));
#endif
#ifdef LC_MESSAGES
printf("LC_MESSAGES %llu\n", (unsigned long long)(long long)(LC_MESSAGES));
#endif
#ifdef LC_MESSAGES_MASK
printf("LC_MESSAGES_MASK %llu\n", (unsigned long long)(long long)(LC_MESSAGES_MASK));
#endif
#ifdef LC_MONETARY
printf("LC_MONETARY %llu\n", (unsigned long long)(long long)(LC_MONETARY));
#endif
#ifdef LC_MONETARY_MASK
printf("LC_MONETARY_MASK %llu\n", (unsigned long long)(long long)(LC_MONETARY_MASK));
#endif
#ifdef LC_NAME
printf("LC_NAME %llu\n", (unsigned long long)(long long)(LC_NAME));
#endif
#ifdef LC_NAME_MASK
printf("LC_NAME_MASK %llu\n", (unsigned long long)(long long)(LC_NAME_MASK));
#endif
#ifdef LC_NUMERIC
printf("LC_NUMERIC %llu\n", (unsigned long long)(long long)(LC_NUMERIC));
#endif
#ifdef LC_NUMERIC_MASK
printf("LC_NUMERIC_MASK %llu\n", (unsigned long long)(long long)(LC_NUMERIC_MASK));
#endif
#ifdef LC_PAPER
printf("LC_PAPER %llu\n", (unsigned long long)(long long)(LC_PAPER));
#endif
#ifdef LC_PAPER_MASK
printf("LC_PAPER_MASK %llu\n", (unsigned long long)(long long)(LC_PAPER_MASK));
#endif
#ifdef LC_TELEPHONE
printf("LC_TELEPHONE %llu\n", (unsigned long long)(long long)(LC_TELEPHONE));
#endif
#ifdef LC_TELEPHONE_MASK
printf("LC_TELEPHONE_MASK %llu\n", (unsigned long long)(long long)(LC_TELEPHONE_MASK));
#endif
#ifdef LC_TIME
printf("LC_TIME %llu\n", (unsigned long long)(long long)(LC_TIME));
#endif
#ifdef LC_TIME_MASK
printf("LC_TIME_MASK %llu\n", (unsigned long long)(long long)(LC_TIME_MASK));
#endif
#ifdef LED_CNT
printf("LED_CNT %llu\n", (unsigned long long)(long long)(LED_CNT));
#endif
#ifdef LED_MAX
printf("LED_MAX %llu\n", (unsigned long long)(long long)(LED_MAX));
#endif
#ifdef LINUX_REBOOT_CMD_CAD_OFF
printf("LINUX_REBOOT_CMD_CAD_OFF %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_CAD_OFF));
#endif
#ifdef LINUX_REBOOT_CMD_CAD_ON
printf("LINUX_REBOOT_CMD_CAD_ON %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_CAD_ON));
#endif
#ifdef LINUX_REBOOT_CMD_HALT
printf("LINUX_REBOOT_CMD_HALT %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_HALT));
#endif
#ifdef LINUX_REBOOT_CMD_KEXEC
printf("LINUX_REBOOT_CMD_KEXEC %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_KEXEC));
#endif
#ifdef LINUX_REBOOT_CMD_POWER_OFF
printf("LINUX_REBOOT_CMD_POWER_OFF %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_POWER_OFF));
#endif
#ifdef LINUX_REBOOT_CMD_RESTART
printf("LINUX_REBOOT_CMD_RESTART %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_RESTART));
#endif
#ifdef LINUX_REBOOT_CMD_RESTART2
printf("LINUX_REBOOT_CMD_RESTART2 %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_RESTART2));
#endif
#ifdef LINUX_REBOOT_CMD_SW_SUSPEND
printf("LINUX_REBOOT_CMD_SW_SUSPEND %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_CMD_SW_SUSPEND));
#endif
#ifdef LINUX_REBOOT_MAGIC1
printf("LINUX_REBOOT_MAGIC1 %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_MAGIC1));
#endif
#ifdef LINUX_REBOOT_MAGIC2
printf("LINUX_REBOOT_MAGIC2 %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_MAGIC2));
#endif
#ifdef LINUX_REBOOT_MAGIC2A
printf("LINUX_REBOOT_MAGIC2A %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_MAGIC2A));
#endif
#ifdef LINUX_REBOOT_MAGIC2B
printf("LINUX_REBOOT_MAGIC2B %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_MAGIC2B));
#endif
#ifdef LINUX_REBOOT_MAGIC2C
printf("LINUX_REBOOT_MAGIC2C %llu\n", (unsigned long long)(long long)(LINUX_REBOOT_MAGIC2C));
#endif
#ifdef LIO_NOP
printf("LIO_NOP %llu\n", (unsigned long long)(long long)(LIO_NOP));
#endif
#ifdef LIO_NOWAIT
printf("LIO_NOWAIT %llu\n", (unsigned long long)(long long)(LIO_NOWAIT));
#endif
#ifdef LIO_READ
printf("LIO_READ %llu\n", (unsigned long long)(long long)(LIO_READ));
#endif
#ifdef LIO_WAIT
printf("LIO_WAIT %llu\n", (unsigned long long)(long long)(LIO_WAIT));
#endif
#ifdef LIO_WRITE
printf("LIO_WRITE %llu\n", (unsigned long long)(long long)(LIO_WRITE));
#endif
#ifdef LM_ID_BASE
printf("LM_ID_BASE %llu\n", (unsigned long long)(long long)(LM_ID_BASE));
#endif
#ifdef LM_ID_NEWLM
printf("LM_ID_NEWLM %llu\n", (unsigned long long)(long long)(LM_ID_NEWLM));
#endif
#ifdef LOCK_EX
printf("LOCK_EX %llu\n", (unsigned long long)(long long)(LOCK_EX));
#endif
#ifdef LOCK_NB
printf("LOCK_NB %llu\n", (unsigned long long)(long long)(LOCK_NB));
#endif
#ifdef LOCK_SH
printf("LOCK_SH %llu\n", (unsigned long long)(long long)(LOCK_SH));
#endif
#ifdef LOCK_UN
printf("LOCK_UN %llu\n", (unsigned long long)(long long)(LOCK_UN));
#endif
#ifdef LOGIN_PROCESS
printf("LOGIN_PROCESS %llu\n", (unsigned long long)(long long)(LOGIN_PROCESS));
#endif
#ifdef LOG_ALERT
printf("LOG_ALERT %llu\n", (unsigned long long)(long long)(LOG_ALERT));
#endif
#ifdef LOG_AUTH
printf("LOG_AUTH %llu\n", (unsigned long long)(long long)(LOG_AUTH));
#endif
#ifdef LOG_AUTHPRIV
printf("LOG_AUTHPRIV %llu\n", (unsigned long long)(long long)(LOG_AUTHPRIV));
#endif
#ifdef LOG_CONS
printf("LOG_CONS %llu\n", (unsigned long long)(long long)(LOG_CONS));
#endif
#ifdef LOG_CRIT
printf("LOG_CRIT %llu\n", (unsigned long long)(long long)(LOG_CRIT));
#endif
#ifdef LOG_CRON
printf("LOG_CRON %llu\n", (unsigned long long)(long long)(LOG_CRON));
#endif
#ifdef LOG_DAEMON
printf("LOG_DAEMON %llu\n", (unsigned long long)(long long)(LOG_DAEMON));
#endif
#ifdef LOG_DEBUG
printf("LOG_DEBUG %llu\n", (unsigned long long)(long long)(LOG_DEBUG));
#endif
#ifdef LOG_EMERG
printf("LOG_EMERG %llu\n", (unsigned long long)(long long)(LOG_EMERG));
#endif
#ifdef LOG_ERR
printf("LOG_ERR %llu\n", (unsigned long long)(long long)(LOG_ERR));
#endif
#ifdef LOG_FACMASK
printf("LOG_FACMASK %llu\n", (unsigned long long)(long long)(LOG_FACMASK));
#endif
#ifdef LOG_FTP
printf("LOG_FTP %llu\n", (unsigned long long)(long long)(LOG_FTP));
#endif
#ifdef LOG_INFO
printf("LOG_INFO %llu\n", (unsigned long long)(long long)(LOG_INFO));
#endif
#ifdef LOG_KERN
printf("LOG_KERN %llu\n", (unsigned long long)(long long)(LOG_KERN));
#endif
#ifdef LOG_LOCAL0
printf("LOG_LOCAL0 %llu\n", (unsigned long long)(long long)(LOG_LOCAL0));
#endif
#ifdef LOG_LOCAL1
printf("LOG_LOCAL1 %llu\n", (unsigned long long)(long long)(LOG_LOCAL1));
#endif
#ifdef LOG_LOCAL2
printf("LOG_LOCAL2 %llu\n", (unsigned long long)(long long)(LOG_LOCAL2));
#endif
#ifdef LOG_LOCAL3
printf("LOG_LOCAL3 %llu\n", (unsigned long long)(long long)(LOG_LOCAL3));
#endif
#ifdef LOG_LOCAL4
printf("LOG_LOCAL4 %llu\n", (unsigned long long)(long long)(LOG_LOCAL4));
#endif
#ifdef LOG_LOCAL5
printf("LOG_LOCAL5 %llu\n", (unsigned long long)(long long)(LOG_LOCAL5));
#endif
#ifdef LOG_LOCAL6
printf("LOG_LOCAL6 %llu\n", (unsigned long long)(long long)(LOG_LOCAL6));
#endif
#ifdef LOG_LOCAL7
printf("LOG_LOCAL7 %llu\n", (unsigned long long)(long long)(LOG_LOCAL7));
#endif
#ifdef LOG_LPR
printf("LOG_LPR %llu\n", (unsigned long long)(long long)(LOG_LPR));
#endif
#ifdef LOG_MAIL
printf("LOG_MAIL %llu\n", (unsigned long long)(long long)(LOG_MAIL));
#endif
#ifdef LOG_NDELAY
printf("LOG_NDELAY %llu\n", (unsigned long long)(long long)(LOG_NDELAY));
#endif
#ifdef LOG_NEWS
printf("LOG_NEWS %llu\n", (unsigned long long)(long long)(LOG_NEWS));
#endif
#ifdef LOG_NFACILITIES
printf("LOG_NFACILITIES %llu\n", (unsigned long long)(long long)(LOG_NFACILITIES));
#endif
#ifdef LOG_NOTICE
printf("LOG_NOTICE %llu\n", (unsigned long long)(long long)(LOG_NOTICE));
#endif
#ifdef LOG_NOWAIT
printf("LOG_NOWAIT %llu\n", (unsigned long long)(long long)(LOG_NOWAIT));
#endif
#ifdef LOG_ODELAY
printf("LOG_ODELAY %llu\n", (unsigned long long)(long long)(LOG_ODELAY));
#endif
#ifdef LOG_PERROR
printf("LOG_PERROR %llu\n", (unsigned long long)(long long)(LOG_PERROR));
#endif
#ifdef LOG_PID
printf("LOG_PID %llu\n", (unsigned long long)(long long)(LOG_PID));
#endif
#ifdef LOG_PRIMASK
printf("LOG_PRIMASK %llu\n", (unsigned long long)(long long)(LOG_PRIMASK));
#endif
#ifdef LOG_SYSLOG
printf("LOG_SYSLOG %llu\n", (unsigned long long)(long long)(LOG_SYSLOG));
#endif
#ifdef LOG_USER
printf("LOG_USER %llu\n", (unsigned long long)(long long)(LOG_USER));
#endif
#ifdef LOG_UUCP
printf("LOG_UUCP %llu\n", (unsigned long long)(long long)(LOG_UUCP));
#endif
#ifdef LOG_WARNING
printf("LOG_WARNING %llu\n", (unsigned long long)(long long)(LOG_WARNING));
#endif
#ifdef L_tmpnam
printf("L_tmpnam %llu\n", (unsigned long long)(long long)(L_tmpnam));
#endif
#ifdef MADV_COLD
printf("MADV_COLD %llu\n", (unsigned long long)(long long)(MADV_COLD));
#endif
#ifdef MADV_COLLAPSE
printf("MADV_COLLAPSE %llu\n", (unsigned long long)(long long)(MADV_COLLAPSE));
#endif
#ifdef MADV_DODUMP
printf("MADV_DODUMP %llu\n", (unsigned long long)(long long)(MADV_DODUMP));
#endif
#ifdef MADV_DOFORK
printf("MADV_DOFORK %llu\n", (unsigned long long)(long long)(MADV_DOFORK));
#endif
#ifdef MADV_DONTDUMP
printf("MADV_DONTDUMP %llu\n", (unsigned long long)(long long)(MADV_DONTDUMP));
#endif
#ifdef MADV_DONTFORK
printf("MADV_DONTFORK %llu\n", (unsigned long long)(long long)(MADV_DONTFORK));
#endif
#ifdef MADV_DONTNEED
printf("MADV_DONTNEED %llu\n", (unsigned long long)(long long)(MADV_DONTNEED));
#endif
#ifdef MADV_DONTNEED_LOCKED
printf("MADV_DONTNEED_LOCKED %llu\n", (unsigned long long)(long long)(MADV_DONTNEED_LOCKED));
#endif
#ifdef MADV_FREE
printf("MADV_FREE %llu\n", (unsigned long long)(long long)(MADV_FREE));
#endif
#ifdef MADV_HUGEPAGE
printf("MADV_HUGEPAGE %llu\n", (unsigned long long)(long long)(MADV_HUGEPAGE));
#endif
#ifdef MADV_HWPOISON
printf("MADV_HWPOISON %llu\n", (unsigned long long)(long long)(MADV_HWPOISON));
#endif
#ifdef MADV_KEEPONFORK
printf("MADV_KEEPONFORK %llu\n", (unsigned long long)(long long)(MADV_KEEPONFORK));
#endif
#ifdef MADV_MERGEABLE
printf("MADV_MERGEABLE %llu\n", (unsigned long long)(long long)(MADV_MERGEABLE));
#endif
#ifdef MADV_NOHUGEPAGE
printf("MADV_NOHUGEPAGE %llu\n", (unsigned long long)(long long)(MADV_NOHUGEPAGE));
#endif
#ifdef MADV_NORMAL
printf("MADV_NORMAL %llu\n", (unsigned long long)(long long)(MADV_NORMAL));
#endif
#ifdef MADV_PAGEOUT
printf("MADV_PAGEOUT %llu\n", (unsigned long long)(long long)(MADV_PAGEOUT));
#endif
#ifdef MADV_POPULATE_READ
printf("MADV_POPULATE_READ %llu\n", (unsigned long long)(long long)(MADV_POPULATE_READ));
#endif
#ifdef MADV_POPULATE_WRITE
printf("MADV_POPULATE_WRITE %llu\n", (unsigned long long)(long long)(MADV_POPULATE_WRITE));
#endif
#ifdef MADV_RANDOM
printf("MADV_RANDOM %llu\n", (unsigned long long)(long long)(MADV_RANDOM));
#endif
#ifdef MADV_REMOVE
printf("MADV_REMOVE %llu\n", (unsigned long long)(long long)(MADV_REMOVE));
#endif
#ifdef MADV_SEQUENTIAL
printf("MADV_SEQUENTIAL %llu\n", (unsigned long long)(long long)(MADV_SEQUENTIAL));
#endif
#ifdef MADV_SOFT_OFFLINE
printf("MADV_SOFT_OFFLINE %llu\n", (unsigned long long)(long long)(MADV_SOFT_OFFLINE));
#endif
#ifdef MADV_UNMERGEABLE
printf("MADV_UNMERGEABLE %llu\n", (unsigned long long)(long long)(MADV_UNMERGEABLE));
#endif
#ifdef MADV_WILLNEED
printf("MADV_WILLNEED %llu\n", (unsigned long long)(long long)(MADV_WILLNEED));
#endif
#ifdef MADV_WIPEONFORK
printf("MADV_WIPEONFORK %llu\n", (unsigned long long)(long long)(MADV_WIPEONFORK));
#endif
#ifdef MAP_ANON
printf("MAP_ANON %llu\n", (unsigned long long)(long long)(MAP_ANON));
#endif
#ifdef MAP_ANONYMOUS
printf("MAP_ANONYMOUS %llu\n", (unsigned long long)(long long)(MAP_ANONYMOUS));
#endif
#ifdef MAP_DENYWRITE
printf("MAP_DENYWRITE %llu\n", (unsigned long long)(long long)(MAP_DENYWRITE));
#endif
#ifdef MAP_DROPPABLE
printf("MAP_DROPPABLE %llu\n", (unsigned long long)(long long)(MAP_DROPPABLE));
#endif
#ifdef MAP_EXECUTABLE
printf("MAP_EXECUTABLE %llu\n", (unsigned long long)(long long)(MAP_EXECUTABLE));
#endif
#ifdef MAP_FILE
printf("MAP_FILE %llu\n", (unsigned long long)(long long)(MAP_FILE));
#endif
#ifdef MAP_FIXED
printf("MAP_FIXED %llu\n", (unsigned long long)(long long)(MAP_FIXED));
#endif
#ifdef MAP_FIXED_NOREPLACE
printf("MAP_FIXED_NOREPLACE %llu\n", (unsigned long long)(long long)(MAP_FIXED_NOREPLACE));
#endif
#ifdef MAP_GROWSDOWN
printf("MAP_GROWSDOWN %llu\n", (unsigned long long)(long long)(MAP_GROWSDOWN));
#endif
#ifdef MAP_HUGETLB
printf("MAP_HUGETLB %llu\n", (unsigned long long)(long long)(MAP_HUGETLB));
#endif
#ifdef MAP_HUGE_16GB
printf("MAP_HUGE_16GB %llu\n", (unsigned long long)(long long)(MAP_HUGE_16GB));
#endif
#ifdef MAP_HUGE_16MB
printf("MAP_HUGE_16MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_16MB));
#endif
#ifdef MAP_HUGE_1GB
printf("MAP_HUGE_1GB %llu\n", (unsigned long long)(long long)(MAP_HUGE_1GB));
#endif
#ifdef MAP_HUGE_1MB
printf("MAP_HUGE_1MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_1MB));
#endif
#ifdef MAP_HUGE_256MB
printf("MAP_HUGE_256MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_256MB));
#endif
#ifdef MAP_HUGE_2GB
printf("MAP_HUGE_2GB %llu\n", (unsigned long long)(long long)(MAP_HUGE_2GB));
#endif
#ifdef MAP_HUGE_2MB
printf("MAP_HUGE_2MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_2MB));
#endif
#ifdef MAP_HUGE_32MB
printf("MAP_HUGE_32MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_32MB));
#endif
#ifdef MAP_HUGE_512KB
printf("MAP_HUGE_512KB %llu\n", (unsigned long long)(long long)(MAP_HUGE_512KB));
#endif
#ifdef MAP_HUGE_512MB
printf("MAP_HUGE_512MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_512MB));
#endif
#ifdef MAP_HUGE_64KB
printf("MAP_HUGE_64KB %llu\n", (unsigned long long)(long long)(MAP_HUGE_64KB));
#endif
#ifdef MAP_HUGE_8MB
printf("MAP_HUGE_8MB %llu\n", (unsigned long long)(long long)(MAP_HUGE_8MB));
#endif
#ifdef MAP_HUGE_MASK
printf("MAP_HUGE_MASK %llu\n", (unsigned long long)(long long)(MAP_HUGE_MASK));
#endif
#ifdef MAP_HUGE_SHIFT
printf("MAP_HUGE_SHIFT %llu\n", (unsigned long long)(long long)(MAP_HUGE_SHIFT));
#endif
#ifdef MAP_LOCKED
printf("MAP_LOCKED %llu\n", (unsigned long long)(long long)(MAP_LOCKED));
#endif
#ifdef MAP_NONBLOCK
printf("MAP_NONBLOCK %llu\n", (unsigned long long)(long long)(MAP_NONBLOCK));
#endif
#ifdef MAP_NORESERVE
printf("MAP_NORESERVE %llu\n", (unsigned long long)(long long)(MAP_NORESERVE));
#endif
#ifdef MAP_POPULATE
printf("MAP_POPULATE %llu\n", (unsigned long long)(long long)(MAP_POPULATE));
#endif
#ifdef MAP_PRIVATE
printf("MAP_PRIVATE %llu\n", (unsigned long long)(long long)(MAP_PRIVATE));
#endif
#ifdef MAP_SHARED
printf("MAP_SHARED %llu\n", (unsigned long long)(long long)(MAP_SHARED));
#endif
#ifdef MAP_SHARED_VALIDATE
printf("MAP_SHARED_VALIDATE %llu\n", (unsigned long long)(long long)(MAP_SHARED_VALIDATE));
#endif
#ifdef MAP_STACK
printf("MAP_STACK %llu\n", (unsigned long long)(long long)(MAP_STACK));
#endif
#ifdef MAP_SYNC
printf("MAP_SYNC %llu\n", (unsigned long long)(long long)(MAP_SYNC));
#endif
#ifdef MAP_TYPE
printf("MAP_TYPE %llu\n", (unsigned long long)(long long)(MAP_TYPE));
#endif
#ifdef MAXTC
printf("MAXTC %llu\n", (unsigned long long)(long long)(MAXTC));
#endif
#ifdef MAXTTL
printf("MAXTTL %llu\n", (unsigned long long)(long long)(MAXTTL));
#endif
#ifdef MAX_ADDR_LEN
printf("MAX_ADDR_LEN %llu\n", (unsigned long long)(long long)(MAX_ADDR_LEN));
#endif
#ifdef MAX_IPOPTLEN
printf("MAX_IPOPTLEN %llu\n", (unsigned long long)(long long)(MAX_IPOPTLEN));
#endif
#ifdef MAX_LINKS
printf("MAX_LINKS %llu\n", (unsigned long long)(long long)(MAX_LINKS));
#endif
#ifdef MCAST_BLOCK_SOURCE
printf("MCAST_BLOCK_SOURCE %llu\n", (unsigned long long)(long long)(MCAST_BLOCK_SOURCE));
#endif
#ifdef MCAST_EXCLUDE
printf("MCAST_EXCLUDE %llu\n", (unsigned long long)(long long)(MCAST_EXCLUDE));
#endif
#ifdef MCAST_INCLUDE
printf("MCAST_INCLUDE %llu\n", (unsigned long long)(long long)(MCAST_INCLUDE));
#endif
#ifdef MCAST_JOIN_GROUP
printf("MCAST_JOIN_GROUP %llu\n", (unsigned long long)(long long)(MCAST_JOIN_GROUP));
#endif
#ifdef MCAST_JOIN_SOURCE_GROUP
printf("MCAST_JOIN_SOURCE_GROUP %llu\n", (unsigned long long)(long long)(MCAST_JOIN_SOURCE_GROUP));
#endif
#ifdef MCAST_LEAVE_GROUP
printf("MCAST_LEAVE_GROUP %llu\n", (unsigned long long)(long long)(MCAST_LEAVE_GROUP));
#endif
#ifdef MCAST_LEAVE_SOURCE_GROUP
printf("MCAST_LEAVE_SOURCE_GROUP %llu\n", (unsigned long long)(long long)(MCAST_LEAVE_SOURCE_GROUP));
#endif
#ifdef MCAST_MSFILTER
printf("MCAST_MSFILTER %llu\n", (unsigned long long)(long long)(MCAST_MSFILTER));
#endif
#ifdef MCAST_UNBLOCK_SOURCE
printf("MCAST_UNBLOCK_SOURCE %llu\n", (unsigned long long)(long long)(MCAST_UNBLOCK_SOURCE));
#endif
#ifdef MCL_CURRENT
printf("MCL_CURRENT %llu\n", (unsigned long long)(long long)(MCL_CURRENT));
#endif
#ifdef MCL_FUTURE
printf("MCL_FUTURE %llu\n", (unsigned long long)(long long)(MCL_FUTURE));
#endif
#ifdef MCL_ONFAULT
printf("MCL_ONFAULT %llu\n", (unsigned long long)(long long)(MCL_ONFAULT));
#endif
#ifdef MEMBARRIER_CMD_GLOBAL
printf("MEMBARRIER_CMD_GLOBAL %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_GLOBAL));
#endif
#ifdef MEMBARRIER_CMD_GLOBAL_EXPEDITED
printf("MEMBARRIER_CMD_GLOBAL_EXPEDITED %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_GLOBAL_EXPEDITED));
#endif
#ifdef MEMBARRIER_CMD_PRIVATE_EXPEDITED
printf("MEMBARRIER_CMD_PRIVATE_EXPEDITED %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_PRIVATE_EXPEDITED));
#endif
#ifdef MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ
printf("MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ));
#endif
#ifdef MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE
printf("MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_PRIVATE_EXPEDITED_SYNC_CORE));
#endif
#ifdef MEMBARRIER_CMD_QUERY
printf("MEMBARRIER_CMD_QUERY %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_QUERY));
#endif
#ifdef MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED
printf("MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_REGISTER_GLOBAL_EXPEDITED));
#endif
#ifdef MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED
printf("MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED));
#endif
#ifdef MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ
printf("MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ));
#endif
#ifdef MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE
printf("MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE %llu\n", (unsigned long long)(long long)(MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED_SYNC_CORE));
#endif
#ifdef MFD_ALLOW_SEALING
printf("MFD_ALLOW_SEALING %llu\n", (unsigned long long)(long long)(MFD_ALLOW_SEALING));
#endif
#ifdef MFD_CLOEXEC
printf("MFD_CLOEXEC %llu\n", (unsigned long long)(long long)(MFD_CLOEXEC));
#endif
#ifdef MFD_EXEC
printf("MFD_EXEC %llu\n", (unsigned long long)(long long)(MFD_EXEC));
#endif
#ifdef MFD_HUGETLB
printf("MFD_HUGETLB %llu\n", (unsigned long long)(long long)(MFD_HUGETLB));
#endif
#ifdef MFD_HUGE_16GB
printf("MFD_HUGE_16GB %llu\n", (unsigned long long)(long long)(MFD_HUGE_16GB));
#endif
#ifdef MFD_HUGE_16MB
printf("MFD_HUGE_16MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_16MB));
#endif
#ifdef MFD_HUGE_1GB
printf("MFD_HUGE_1GB %llu\n", (unsigned long long)(long long)(MFD_HUGE_1GB));
#endif
#ifdef MFD_HUGE_1MB
printf("MFD_HUGE_1MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_1MB));
#endif
#ifdef MFD_HUGE_256MB
printf("MFD_HUGE_256MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_256MB));
#endif
#ifdef MFD_HUGE_2GB
printf("MFD_HUGE_2GB %llu\n", (unsigned long long)(long long)(MFD_HUGE_2GB));
#endif
#ifdef MFD_HUGE_2MB
printf("MFD_HUGE_2MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_2MB));
#endif
#ifdef MFD_HUGE_32MB
printf("MFD_HUGE_32MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_32MB));
#endif
#ifdef MFD_HUGE_512KB
printf("MFD_HUGE_512KB %llu\n", (unsigned long long)(long long)(MFD_HUGE_512KB));
#endif
#ifdef MFD_HUGE_512MB
printf("MFD_HUGE_512MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_512MB));
#endif
#ifdef MFD_HUGE_64KB
printf("MFD_HUGE_64KB %llu\n", (unsigned long long)(long long)(MFD_HUGE_64KB));
#endif
#ifdef MFD_HUGE_8MB
printf("MFD_HUGE_8MB %llu\n", (unsigned long long)(long long)(MFD_HUGE_8MB));
#endif
#ifdef MFD_HUGE_MASK
printf("MFD_HUGE_MASK %llu\n", (unsigned long long)(long long)(MFD_HUGE_MASK));
#endif
#ifdef MFD_HUGE_SHIFT
printf("MFD_HUGE_SHIFT %llu\n", (unsigned long long)(long long)(MFD_HUGE_SHIFT));
#endif
#ifdef MFD_NOEXEC_SEAL
printf("MFD_NOEXEC_SEAL %llu\n", (unsigned long long)(long long)(MFD_NOEXEC_SEAL));
#endif
#ifdef MINIX2_SUPER_MAGIC
printf("MINIX2_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(MINIX2_SUPER_MAGIC));
#endif
#ifdef MINIX2_SUPER_MAGIC2
printf("MINIX2_SUPER_MAGIC2 %llu\n", (unsigned long long)(long long)(MINIX2_SUPER_MAGIC2));
#endif
#ifdef MINIX3_SUPER_MAGIC
printf("MINIX3_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(MINIX3_SUPER_MAGIC));
#endif
#ifdef MINIX_SUPER_MAGIC
printf("MINIX_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(MINIX_SUPER_MAGIC));
#endif
#ifdef MINIX_SUPER_MAGIC2
printf("MINIX_SUPER_MAGIC2 %llu\n", (unsigned long long)(long long)(MINIX_SUPER_MAGIC2));
#endif
#ifdef MINSIGSTKSZ
printf("MINSIGSTKSZ %llu\n", (unsigned long long)(long long)(MINSIGSTKSZ));
#endif
#ifdef MLOCK_ONFAULT
printf("MLOCK_ONFAULT %llu\n", (unsigned long long)(long long)(MLOCK_ONFAULT));
#endif
#ifdef MMAP_PAGE_ZERO
printf("MMAP_PAGE_ZERO %llu\n", (unsigned long long)(long long)(MMAP_PAGE_ZERO));
#endif
#ifdef MNT_DETACH
printf("MNT_DETACH %llu\n", (unsigned long long)(long long)(MNT_DETACH));
#endif
#ifdef MNT_EXPIRE
printf("MNT_EXPIRE %llu\n", (unsigned long long)(long long)(MNT_EXPIRE));
#endif
#ifdef MNT_FORCE
printf("MNT_FORCE %llu\n", (unsigned long long)(long long)(MNT_FORCE));
#endif
#ifdef MNT_NS_INFO_SIZE_VER0
printf("MNT_NS_INFO_SIZE_VER0 %llu\n", (unsigned long long)(long long)(MNT_NS_INFO_SIZE_VER0));
#endif
#ifdef MODULE_INIT_IGNORE_MODVERSIONS
printf("MODULE_INIT_IGNORE_MODVERSIONS %llu\n", (unsigned long long)(long long)(MODULE_INIT_IGNORE_MODVERSIONS));
#endif
#ifdef MODULE_INIT_IGNORE_VERMAGIC
printf("MODULE_INIT_IGNORE_VERMAGIC %llu\n", (unsigned long long)(long long)(MODULE_INIT_IGNORE_VERMAGIC));
#endif
#ifdef MOD_CLKA
printf("MOD_CLKA %llu\n", (unsigned long long)(long long)(MOD_CLKA));
#endif
#ifdef MOD_CLKB
printf("MOD_CLKB %llu\n", (unsigned long long)(long long)(MOD_CLKB));
#endif
#ifdef MOD_ESTERROR
printf("MOD_ESTERROR %llu\n", (unsigned long long)(long long)(MOD_ESTERROR));
#endif
#ifdef MOD_FREQUENCY
printf("MOD_FREQUENCY %llu\n", (unsigned long long)(long long)(MOD_FREQUENCY));
#endif
#ifdef MOD_MAXERROR
printf("MOD_MAXERROR %llu\n", (unsigned long long)(long long)(MOD_MAXERROR));
#endif
#ifdef MOD_MICRO
printf("MOD_MICRO %llu\n", (unsigned long long)(long long)(MOD_MICRO));
#endif
#ifdef MOD_NANO
printf("MOD_NANO %llu\n", (unsigned long long)(long long)(MOD_NANO));
#endif
#ifdef MOD_OFFSET
printf("MOD_OFFSET %llu\n", (unsigned long long)(long long)(MOD_OFFSET));
#endif
#ifdef MOD_STATUS
printf("MOD_STATUS %llu\n", (unsigned long long)(long long)(MOD_STATUS));
#endif
#ifdef MOD_TAI
printf("MOD_TAI %llu\n", (unsigned long long)(long long)(MOD_TAI));
#endif
#ifdef MOD_TIMECONST
printf("MOD_TIMECONST %llu\n", (unsigned long long)(long long)(MOD_TIMECONST));
#endif
#ifdef MON_1
printf("MON_1 %llu\n", (unsigned long long)(long long)(MON_1));
#endif
#ifdef MON_10
printf("MON_10 %llu\n", (unsigned long long)(long long)(MON_10));
#endif
#ifdef MON_11
printf("MON_11 %llu\n", (unsigned long long)(long long)(MON_11));
#endif
#ifdef MON_12
printf("MON_12 %llu\n", (unsigned long long)(long long)(MON_12));
#endif
#ifdef MON_2
printf("MON_2 %llu\n", (unsigned long long)(long long)(MON_2));
#endif
#ifdef MON_3
printf("MON_3 %llu\n", (unsigned long long)(long long)(MON_3));
#endif
#ifdef MON_4
printf("MON_4 %llu\n", (unsigned long long)(long long)(MON_4));
#endif
#ifdef MON_5
printf("MON_5 %llu\n", (unsigned long long)(long long)(MON_5));
#endif
#ifdef MON_6
printf("MON_6 %llu\n", (unsigned long long)(long long)(MON_6));
#endif
#ifdef MON_7
printf("MON_7 %llu\n", (unsigned long long)(long long)(MON_7));
#endif
#ifdef MON_8
printf("MON_8 %llu\n", (unsigned long long)(long long)(MON_8));
#endif
#ifdef MON_9
printf("MON_9 %llu\n", (unsigned long long)(long long)(MON_9));
#endif
#ifdef MOUNT_ATTR_IDMAP
printf("MOUNT_ATTR_IDMAP %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_IDMAP));
#endif
#ifdef MOUNT_ATTR_NOATIME
printf("MOUNT_ATTR_NOATIME %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NOATIME));
#endif
#ifdef MOUNT_ATTR_NODEV
printf("MOUNT_ATTR_NODEV %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NODEV));
#endif
#ifdef MOUNT_ATTR_NODIRATIME
printf("MOUNT_ATTR_NODIRATIME %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NODIRATIME));
#endif
#ifdef MOUNT_ATTR_NOEXEC
printf("MOUNT_ATTR_NOEXEC %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NOEXEC));
#endif
#ifdef MOUNT_ATTR_NOSUID
printf("MOUNT_ATTR_NOSUID %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NOSUID));
#endif
#ifdef MOUNT_ATTR_NOSYMFOLLOW
printf("MOUNT_ATTR_NOSYMFOLLOW %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_NOSYMFOLLOW));
#endif
#ifdef MOUNT_ATTR_RDONLY
printf("MOUNT_ATTR_RDONLY %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_RDONLY));
#endif
#ifdef MOUNT_ATTR_RELATIME
printf("MOUNT_ATTR_RELATIME %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_RELATIME));
#endif
#ifdef MOUNT_ATTR_SIZE_VER0
printf("MOUNT_ATTR_SIZE_VER0 %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_SIZE_VER0));
#endif
#ifdef MOUNT_ATTR_STRICTATIME
printf("MOUNT_ATTR_STRICTATIME %llu\n", (unsigned long long)(long long)(MOUNT_ATTR_STRICTATIME));
#endif
#ifdef MOUNT_ATTR__ATIME
printf("MOUNT_ATTR__ATIME %llu\n", (unsigned long long)(long long)(MOUNT_ATTR__ATIME));
#endif
#ifdef MOVE_MOUNT_BENEATH
printf("MOVE_MOUNT_BENEATH %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_BENEATH));
#endif
#ifdef MOVE_MOUNT_F_AUTOMOUNTS
printf("MOVE_MOUNT_F_AUTOMOUNTS %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_F_AUTOMOUNTS));
#endif
#ifdef MOVE_MOUNT_F_EMPTY_PATH
printf("MOVE_MOUNT_F_EMPTY_PATH %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_F_EMPTY_PATH));
#endif
#ifdef MOVE_MOUNT_F_SYMLINKS
printf("MOVE_MOUNT_F_SYMLINKS %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_F_SYMLINKS));
#endif
#ifdef MOVE_MOUNT_SET_GROUP
printf("MOVE_MOUNT_SET_GROUP %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_SET_GROUP));
#endif
#ifdef MOVE_MOUNT_T_AUTOMOUNTS
printf("MOVE_MOUNT_T_AUTOMOUNTS %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_T_AUTOMOUNTS));
#endif
#ifdef MOVE_MOUNT_T_EMPTY_PATH
printf("MOVE_MOUNT_T_EMPTY_PATH %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_T_EMPTY_PATH));
#endif
#ifdef MOVE_MOUNT_T_SYMLINKS
printf("MOVE_MOUNT_T_SYMLINKS %llu\n", (unsigned long long)(long long)(MOVE_MOUNT_T_SYMLINKS));
#endif
#ifdef MPOL_BIND
printf("MPOL_BIND %llu\n", (unsigned long long)(long long)(MPOL_BIND));
#endif
#ifdef MPOL_DEFAULT
printf("MPOL_DEFAULT %llu\n", (unsigned long long)(long long)(MPOL_DEFAULT));
#endif
#ifdef MPOL_F_NUMA_BALANCING
printf("MPOL_F_NUMA_BALANCING %llu\n", (unsigned long long)(long long)(MPOL_F_NUMA_BALANCING));
#endif
#ifdef MPOL_F_RELATIVE_NODES
printf("MPOL_F_RELATIVE_NODES %llu\n", (unsigned long long)(long long)(MPOL_F_RELATIVE_NODES));
#endif
#ifdef MPOL_F_STATIC_NODES
printf("MPOL_F_STATIC_NODES %llu\n", (unsigned long long)(long long)(MPOL_F_STATIC_NODES));
#endif
#ifdef MPOL_INTERLEAVE
printf("MPOL_INTERLEAVE %llu\n", (unsigned long long)(long long)(MPOL_INTERLEAVE));
#endif
#ifdef MPOL_LOCAL
printf("MPOL_LOCAL %llu\n", (unsigned long long)(long long)(MPOL_LOCAL));
#endif
#ifdef MPOL_PREFERRED
printf("MPOL_PREFERRED %llu\n", (unsigned long long)(long long)(MPOL_PREFERRED));
#endif
#ifdef MREMAP_DONTUNMAP
printf("MREMAP_DONTUNMAP %llu\n", (unsigned long long)(long long)(MREMAP_DONTUNMAP));
#endif
#ifdef MREMAP_FIXED
printf("MREMAP_FIXED %llu\n", (unsigned long long)(long long)(MREMAP_FIXED));
#endif
#ifdef MREMAP_MAYMOVE
printf("MREMAP_MAYMOVE %llu\n", (unsigned long long)(long long)(MREMAP_MAYMOVE));
#endif
#ifdef MSC_CNT
printf("MSC_CNT %llu\n", (unsigned long long)(long long)(MSC_CNT));
#endif
#ifdef MSC_MAX
printf("MSC_MAX %llu\n", (unsigned long long)(long long)(MSC_MAX));
#endif
#ifdef MSDOS_SUPER_MAGIC
printf("MSDOS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(MSDOS_SUPER_MAGIC));
#endif
#ifdef MSG_CMSG_CLOEXEC
printf("MSG_CMSG_CLOEXEC %llu\n", (unsigned long long)(long long)(MSG_CMSG_CLOEXEC));
#endif
#ifdef MSG_CONFIRM
printf("MSG_CONFIRM %llu\n", (unsigned long long)(long long)(MSG_CONFIRM));
#endif
#ifdef MSG_COPY
printf("MSG_COPY %llu\n", (unsigned long long)(long long)(MSG_COPY));
#endif
#ifdef MSG_CTRUNC
printf("MSG_CTRUNC %llu\n", (unsigned long long)(long long)(MSG_CTRUNC));
#endif
#ifdef MSG_DONTROUTE
printf("MSG_DONTROUTE %llu\n", (unsigned long long)(long long)(MSG_DONTROUTE));
#endif
#ifdef MSG_DONTWAIT
printf("MSG_DONTWAIT %llu\n", (unsigned long long)(long long)(MSG_DONTWAIT));
#endif
#ifdef MSG_EOR
printf("MSG_EOR %llu\n", (unsigned long long)(long long)(MSG_EOR));
#endif
#ifdef MSG_ERRQUEUE
printf("MSG_ERRQUEUE %llu\n", (unsigned long long)(long long)(MSG_ERRQUEUE));
#endif
#ifdef MSG_EXCEPT
printf("MSG_EXCEPT %llu\n", (unsigned long long)(long long)(MSG_EXCEPT));
#endif
#ifdef MSG_FASTOPEN
printf("MSG_FASTOPEN %llu\n", (unsigned long long)(long long)(MSG_FASTOPEN));
#endif
#ifdef MSG_FIN
printf("MSG_FIN %llu\n", (unsigned long long)(long long)(MSG_FIN));
#endif
#ifdef MSG_INFO
printf("MSG_INFO %llu\n", (unsigned long long)(long long)(MSG_INFO));
#endif
#ifdef MSG_MORE
printf("MSG_MORE %llu\n", (unsigned long long)(long long)(MSG_MORE));
#endif
#ifdef MSG_NOERROR
printf("MSG_NOERROR %llu\n", (unsigned long long)(long long)(MSG_NOERROR));
#endif
#ifdef MSG_NOSIGNAL
printf("MSG_NOSIGNAL %llu\n", (unsigned long long)(long long)(MSG_NOSIGNAL));
#endif
#ifdef MSG_NOTIFICATION
printf("MSG_NOTIFICATION %llu\n", (unsigned long long)(long long)(MSG_NOTIFICATION));
#endif
#ifdef MSG_OOB
printf("MSG_OOB %llu\n", (unsigned long long)(long long)(MSG_OOB));
#endif
#ifdef MSG_PEEK
printf("MSG_PEEK %llu\n", (unsigned long long)(long long)(MSG_PEEK));
#endif
#ifdef MSG_RST
printf("MSG_RST %llu\n", (unsigned long long)(long long)(MSG_RST));
#endif
#ifdef MSG_STAT
printf("MSG_STAT %llu\n", (unsigned long long)(long long)(MSG_STAT));
#endif
#ifdef MSG_SYN
printf("MSG_SYN %llu\n", (unsigned long long)(long long)(MSG_SYN));
#endif
#ifdef MSG_TRUNC
printf("MSG_TRUNC %llu\n", (unsigned long long)(long long)(MSG_TRUNC));
#endif
#ifdef MSG_TRYHARD
printf("MSG_TRYHARD %llu\n", (unsigned long long)(long long)(MSG_TRYHARD));
#endif
#ifdef MSG_WAITALL
printf("MSG_WAITALL %llu\n", (unsigned long long)(long long)(MSG_WAITALL));
#endif
#ifdef MSG_WAITFORONE
printf("MSG_WAITFORONE %llu\n", (unsigned long long)(long long)(MSG_WAITFORONE));
#endif
#ifdef MSG_ZEROCOPY
printf("MSG_ZEROCOPY %llu\n", (unsigned long long)(long long)(MSG_ZEROCOPY));
#endif
#ifdef MS_ACTIVE
printf("MS_ACTIVE %llu\n", (unsigned long long)(long long)(MS_ACTIVE));
#endif
#ifdef MS_ASYNC
printf("MS_ASYNC %llu\n", (unsigned long long)(long long)(MS_ASYNC));
#endif
#ifdef MS_BIND
printf("MS_BIND %llu\n", (unsigned long long)(long long)(MS_BIND));
#endif
#ifdef MS_DIRSYNC
printf("MS_DIRSYNC %llu\n", (unsigned long long)(long long)(MS_DIRSYNC));
#endif
#ifdef MS_INVALIDATE
printf("MS_INVALIDATE %llu\n", (unsigned long long)(long long)(MS_INVALIDATE));
#endif
#ifdef MS_I_VERSION
printf("MS_I_VERSION %llu\n", (unsigned long long)(long long)(MS_I_VERSION));
#endif
#ifdef MS_KERNMOUNT
printf("MS_KERNMOUNT %llu\n", (unsigned long long)(long long)(MS_KERNMOUNT));
#endif
#ifdef MS_LAZYTIME
printf("MS_LAZYTIME %llu\n", (unsigned long long)(long long)(MS_LAZYTIME));
#endif
#ifdef MS_MANDLOCK
printf("MS_MANDLOCK %llu\n", (unsigned long long)(long long)(MS_MANDLOCK));
#endif
#ifdef MS_MGC_MSK
printf("MS_MGC_MSK %llu\n", (unsigned long long)(long long)(MS_MGC_MSK));
#endif
#ifdef MS_MGC_VAL
printf("MS_MGC_VAL %llu\n", (unsigned long long)(long long)(MS_MGC_VAL));
#endif
#ifdef MS_MOVE
printf("MS_MOVE %llu\n", (unsigned long long)(long long)(MS_MOVE));
#endif
#ifdef MS_NOATIME
printf("MS_NOATIME %llu\n", (unsigned long long)(long long)(MS_NOATIME));
#endif
#ifdef MS_NODEV
printf("MS_NODEV %llu\n", (unsigned long long)(long long)(MS_NODEV));
#endif
#ifdef MS_NODIRATIME
printf("MS_NODIRATIME %llu\n", (unsigned long long)(long long)(MS_NODIRATIME));
#endif
#ifdef MS_NOEXEC
printf("MS_NOEXEC %llu\n", (unsigned long long)(long long)(MS_NOEXEC));
#endif
#ifdef MS_NOSUID
printf("MS_NOSUID %llu\n", (unsigned long long)(long long)(MS_NOSUID));
#endif
#ifdef MS_NOSYMFOLLOW
printf("MS_NOSYMFOLLOW %llu\n", (unsigned long long)(long long)(MS_NOSYMFOLLOW));
#endif
#ifdef MS_NOUSER
printf("MS_NOUSER %llu\n", (unsigned long long)(long long)(MS_NOUSER));
#endif
#ifdef MS_POSIXACL
printf("MS_POSIXACL %llu\n", (unsigned long long)(long long)(MS_POSIXACL));
#endif
#ifdef MS_PRIVATE
printf("MS_PRIVATE %llu\n", (unsigned long long)(long long)(MS_PRIVATE));
#endif
#ifdef MS_RDONLY
printf("MS_RDONLY %llu\n", (unsigned long long)(long long)(MS_RDONLY));
#endif
#ifdef MS_REC
printf("MS_REC %llu\n", (unsigned long long)(long long)(MS_REC));
#endif
#ifdef MS_RELATIME
printf("MS_RELATIME %llu\n", (unsigned long long)(long long)(MS_RELATIME));
#endif
#ifdef MS_REMOUNT
printf("MS_REMOUNT %llu\n", (unsigned long long)(long long)(MS_REMOUNT));
#endif
#ifdef MS_RMT_MASK
printf("MS_RMT_MASK %llu\n", (unsigned long long)(long long)(MS_RMT_MASK));
#endif
#ifdef MS_SHARED
printf("MS_SHARED %llu\n", (unsigned long long)(long long)(MS_SHARED));
#endif
#ifdef MS_SILENT
printf("MS_SILENT %llu\n", (unsigned long long)(long long)(MS_SILENT));
#endif
#ifdef MS_SLAVE
printf("MS_SLAVE %llu\n", (unsigned long long)(long long)(MS_SLAVE));
#endif
#ifdef MS_STRICTATIME
printf("MS_STRICTATIME %llu\n", (unsigned long long)(long long)(MS_STRICTATIME));
#endif
#ifdef MS_SYNC
printf("MS_SYNC %llu\n", (unsigned long long)(long long)(MS_SYNC));
#endif
#ifdef MS_SYNCHRONOUS
printf("MS_SYNCHRONOUS %llu\n", (unsigned long long)(long long)(MS_SYNCHRONOUS));
#endif
#ifdef MS_UNBINDABLE
printf("MS_UNBINDABLE %llu\n", (unsigned long long)(long long)(MS_UNBINDABLE));
#endif
#ifdef M_ARENA_MAX
printf("M_ARENA_MAX %llu\n", (unsigned long long)(long long)(M_ARENA_MAX));
#endif
#ifdef M_ARENA_TEST
printf("M_ARENA_TEST %llu\n", (unsigned long long)(long long)(M_ARENA_TEST));
#endif
#ifdef M_CHECK_ACTION
printf("M_CHECK_ACTION %llu\n", (unsigned long long)(long long)(M_CHECK_ACTION));
#endif
#ifdef M_GRAIN
printf("M_GRAIN %llu\n", (unsigned long long)(long long)(M_GRAIN));
#endif
#ifdef M_KEEP
printf("M_KEEP %llu\n", (unsigned long long)(long long)(M_KEEP));
#endif
#ifdef M_MMAP_MAX
printf("M_MMAP_MAX %llu\n", (unsigned long long)(long long)(M_MMAP_MAX));
#endif
#ifdef M_MMAP_THRESHOLD
printf("M_MMAP_THRESHOLD %llu\n", (unsigned long long)(long long)(M_MMAP_THRESHOLD));
#endif
#ifdef M_MXFAST
printf("M_MXFAST %llu\n", (unsigned long long)(long long)(M_MXFAST));
#endif
#ifdef M_NLBLKS
printf("M_NLBLKS %llu\n", (unsigned long long)(long long)(M_NLBLKS));
#endif
#ifdef M_PERTURB
printf("M_PERTURB %llu\n", (unsigned long long)(long long)(M_PERTURB));
#endif
#ifdef M_TOP_PAD
printf("M_TOP_PAD %llu\n", (unsigned long long)(long long)(M_TOP_PAD));
#endif
#ifdef M_TRIM_THRESHOLD
printf("M_TRIM_THRESHOLD %llu\n", (unsigned long long)(long long)(M_TRIM_THRESHOLD));
#endif
#ifdef NCCS
printf("NCCS %llu\n", (unsigned long long)(long long)(NCCS));
#endif
#ifdef NCP_SUPER_MAGIC
printf("NCP_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(NCP_SUPER_MAGIC));
#endif
#ifdef NDA_CACHEINFO
printf("NDA_CACHEINFO %llu\n", (unsigned long long)(long long)(NDA_CACHEINFO));
#endif
#ifdef NDA_DST
printf("NDA_DST %llu\n", (unsigned long long)(long long)(NDA_DST));
#endif
#ifdef NDA_IFINDEX
printf("NDA_IFINDEX %llu\n", (unsigned long long)(long long)(NDA_IFINDEX));
#endif
#ifdef NDA_LINK_NETNSID
printf("NDA_LINK_NETNSID %llu\n", (unsigned long long)(long long)(NDA_LINK_NETNSID));
#endif
#ifdef NDA_LLADDR
printf("NDA_LLADDR %llu\n", (unsigned long long)(long long)(NDA_LLADDR));
#endif
#ifdef NDA_MASTER
printf("NDA_MASTER %llu\n", (unsigned long long)(long long)(NDA_MASTER));
#endif
#ifdef NDA_PORT
printf("NDA_PORT %llu\n", (unsigned long long)(long long)(NDA_PORT));
#endif
#ifdef NDA_PROBES
printf("NDA_PROBES %llu\n", (unsigned long long)(long long)(NDA_PROBES));
#endif
#ifdef NDA_SRC_VNI
printf("NDA_SRC_VNI %llu\n", (unsigned long long)(long long)(NDA_SRC_VNI));
#endif
#ifdef NDA_UNSPEC
printf("NDA_UNSPEC %llu\n", (unsigned long long)(long long)(NDA_UNSPEC));
#endif
#ifdef NDA_VLAN
printf("NDA_VLAN %llu\n", (unsigned long long)(long long)(NDA_VLAN));
#endif
#ifdef NDA_VNI
printf("NDA_VNI %llu\n", (unsigned long long)(long long)(NDA_VNI));
#endif
#ifdef NETLINK_ADD_MEMBERSHIP
printf("NETLINK_ADD_MEMBERSHIP %llu\n", (unsigned long long)(long long)(NETLINK_ADD_MEMBERSHIP));
#endif
#ifdef NETLINK_AUDIT
printf("NETLINK_AUDIT %llu\n", (unsigned long long)(long long)(NETLINK_AUDIT));
#endif
#ifdef NETLINK_BROADCAST_ERROR
printf("NETLINK_BROADCAST_ERROR %llu\n", (unsigned long long)(long long)(NETLINK_BROADCAST_ERROR));
#endif
#ifdef NETLINK_CAP_ACK
printf("NETLINK_CAP_ACK %llu\n", (unsigned long long)(long long)(NETLINK_CAP_ACK));
#endif
#ifdef NETLINK_CONNECTOR
printf("NETLINK_CONNECTOR %llu\n", (unsigned long long)(long long)(NETLINK_CONNECTOR));
#endif
#ifdef NETLINK_CRYPTO
printf("NETLINK_CRYPTO %llu\n", (unsigned long long)(long long)(NETLINK_CRYPTO));
#endif
#ifdef NETLINK_DNRTMSG
printf("NETLINK_DNRTMSG %llu\n", (unsigned long long)(long long)(NETLINK_DNRTMSG));
#endif
#ifdef NETLINK_DROP_MEMBERSHIP
printf("NETLINK_DROP_MEMBERSHIP %llu\n", (unsigned long long)(long long)(NETLINK_DROP_MEMBERSHIP));
#endif
#ifdef NETLINK_ECRYPTFS
printf("NETLINK_ECRYPTFS %llu\n", (unsigned long long)(long long)(NETLINK_ECRYPTFS));
#endif
#ifdef NETLINK_EXT_ACK
printf("NETLINK_EXT_ACK %llu\n", (unsigned long long)(long long)(NETLINK_EXT_ACK));
#endif
#ifdef NETLINK_FIB_LOOKUP
printf("NETLINK_FIB_LOOKUP %llu\n", (unsigned long long)(long long)(NETLINK_FIB_LOOKUP));
#endif
#ifdef NETLINK_FIREWALL
printf("NETLINK_FIREWALL %llu\n", (unsigned long long)(long long)(NETLINK_FIREWALL));
#endif
#ifdef NETLINK_GENERIC
printf("NETLINK_GENERIC %llu\n", (unsigned long long)(long long)(NETLINK_GENERIC));
#endif
#ifdef NETLINK_GET_STRICT_CHK
printf("NETLINK_GET_STRICT_CHK %llu\n", (unsigned long long)(long long)(NETLINK_GET_STRICT_CHK));
#endif
#ifdef NETLINK_INET_DIAG
printf("NETLINK_INET_DIAG %llu\n", (unsigned long long)(long long)(NETLINK_INET_DIAG));
#endif
#ifdef NETLINK_IP6_FW
printf("NETLINK_IP6_FW %llu\n", (unsigned long long)(long long)(NETLINK_IP6_FW));
#endif
#ifdef NETLINK_ISCSI
printf("NETLINK_ISCSI %llu\n", (unsigned long long)(long long)(NETLINK_ISCSI));
#endif
#ifdef NETLINK_KOBJECT_UEVENT
printf("NETLINK_KOBJECT_UEVENT %llu\n", (unsigned long long)(long long)(NETLINK_KOBJECT_UEVENT));
#endif
#ifdef NETLINK_LISTEN_ALL_NSID
printf("NETLINK_LISTEN_ALL_NSID %llu\n", (unsigned long long)(long long)(NETLINK_LISTEN_ALL_NSID));
#endif
#ifdef NETLINK_LIST_MEMBERSHIPS
printf("NETLINK_LIST_MEMBERSHIPS %llu\n", (unsigned long long)(long long)(NETLINK_LIST_MEMBERSHIPS));
#endif
#ifdef NETLINK_NETFILTER
printf("NETLINK_NETFILTER %llu\n", (unsigned long long)(long long)(NETLINK_NETFILTER));
#endif
#ifdef NETLINK_NFLOG
printf("NETLINK_NFLOG %llu\n", (unsigned long long)(long long)(NETLINK_NFLOG));
#endif
#ifdef NETLINK_NO_ENOBUFS
printf("NETLINK_NO_ENOBUFS %llu\n", (unsigned long long)(long long)(NETLINK_NO_ENOBUFS));
#endif
#ifdef NETLINK_PKTINFO
printf("NETLINK_PKTINFO %llu\n", (unsigned long long)(long long)(NETLINK_PKTINFO));
#endif
#ifdef NETLINK_RDMA
printf("NETLINK_RDMA %llu\n", (unsigned long long)(long long)(NETLINK_RDMA));
#endif
#ifdef NETLINK_ROUTE
printf("NETLINK_ROUTE %llu\n", (unsigned long long)(long long)(NETLINK_ROUTE));
#endif
#ifdef NETLINK_RX_RING
printf("NETLINK_RX_RING %llu\n", (unsigned long long)(long long)(NETLINK_RX_RING));
#endif
#ifdef NETLINK_SCSITRANSPORT
printf("NETLINK_SCSITRANSPORT %llu\n", (unsigned long long)(long long)(NETLINK_SCSITRANSPORT));
#endif
#ifdef NETLINK_SELINUX
printf("NETLINK_SELINUX %llu\n", (unsigned long long)(long long)(NETLINK_SELINUX));
#endif
#ifdef NETLINK_SOCK_DIAG
printf("NETLINK_SOCK_DIAG %llu\n", (unsigned long long)(long long)(NETLINK_SOCK_DIAG));
#endif
#ifdef NETLINK_TX_RING
printf("NETLINK_TX_RING %llu\n", (unsigned long long)(long long)(NETLINK_TX_RING));
#endif
#ifdef NETLINK_UNUSED
printf("NETLINK_UNUSED %llu\n", (unsigned long long)(long long)(NETLINK_UNUSED));
#endif
#ifdef NETLINK_USERSOCK
printf("NETLINK_USERSOCK %llu\n", (unsigned long long)(long long)(NETLINK_USERSOCK));
#endif
#ifdef NETLINK_XFRM
printf("NETLINK_XFRM %llu\n", (unsigned long long)(long long)(NETLINK_XFRM));
#endif
#ifdef NET_802
printf("NET_802 %llu\n", (unsigned long long)(long long)(NET_802));
#endif
#ifdef NET_ATALK
printf("NET_ATALK %llu\n", (unsigned long long)(long long)(NET_ATALK));
#endif
#ifdef NET_AX25
printf("NET_AX25 %llu\n", (unsigned long long)(long long)(NET_AX25));
#endif
#ifdef NET_BRIDGE
printf("NET_BRIDGE %llu\n", (unsigned long long)(long long)(NET_BRIDGE));
#endif
#ifdef NET_CORE
printf("NET_CORE %llu\n", (unsigned long long)(long long)(NET_CORE));
#endif
#ifdef NET_DCCP
printf("NET_DCCP %llu\n", (unsigned long long)(long long)(NET_DCCP));
#endif
#ifdef NET_DECNET
printf("NET_DECNET %llu\n", (unsigned long long)(long long)(NET_DECNET));
#endif
#ifdef NET_ECONET
printf("NET_ECONET %llu\n", (unsigned long long)(long long)(NET_ECONET));
#endif
#ifdef NET_ETHER
printf("NET_ETHER %llu\n", (unsigned long long)(long long)(NET_ETHER));
#endif
#ifdef NET_IPV4
printf("NET_IPV4 %llu\n", (unsigned long long)(long long)(NET_IPV4));
#endif
#ifdef NET_IPV6
printf("NET_IPV6 %llu\n", (unsigned long long)(long long)(NET_IPV6));
#endif
#ifdef NET_IPX
printf("NET_IPX %llu\n", (unsigned long long)(long long)(NET_IPX));
#endif
#ifdef NET_IRDA
printf("NET_IRDA %llu\n", (unsigned long long)(long long)(NET_IRDA));
#endif
#ifdef NET_LLC
printf("NET_LLC %llu\n", (unsigned long long)(long long)(NET_LLC));
#endif
#ifdef NET_NETFILTER
printf("NET_NETFILTER %llu\n", (unsigned long long)(long long)(NET_NETFILTER));
#endif
#ifdef NET_NETROM
printf("NET_NETROM %llu\n", (unsigned long long)(long long)(NET_NETROM));
#endif
#ifdef NET_ROSE
printf("NET_ROSE %llu\n", (unsigned long long)(long long)(NET_ROSE));
#endif
#ifdef NET_SCTP
printf("NET_SCTP %llu\n", (unsigned long long)(long long)(NET_SCTP));
#endif
#ifdef NET_TR
printf("NET_TR %llu\n", (unsigned long long)(long long)(NET_TR));
#endif
#ifdef NET_UNIX
printf("NET_UNIX %llu\n", (unsigned long long)(long long)(NET_UNIX));
#endif
#ifdef NET_X25
printf("NET_X25 %llu\n", (unsigned long long)(long long)(NET_X25));
#endif
#ifdef NEW_TIME
printf("NEW_TIME %llu\n", (unsigned long long)(long long)(NEW_TIME));
#endif
#ifdef NFNETLINK_V0
printf("NFNETLINK_V0 %llu\n", (unsigned long long)(long long)(NFNETLINK_V0));
#endif
#ifdef NFNLGRP_ACCT_QUOTA
printf("NFNLGRP_ACCT_QUOTA %llu\n", (unsigned long long)(long long)(NFNLGRP_ACCT_QUOTA));
#endif
#ifdef NFNLGRP_CONNTRACK_DESTROY
printf("NFNLGRP_CONNTRACK_DESTROY %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_DESTROY));
#endif
#ifdef NFNLGRP_CONNTRACK_EXP_DESTROY
printf("NFNLGRP_CONNTRACK_EXP_DESTROY %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_EXP_DESTROY));
#endif
#ifdef NFNLGRP_CONNTRACK_EXP_NEW
printf("NFNLGRP_CONNTRACK_EXP_NEW %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_EXP_NEW));
#endif
#ifdef NFNLGRP_CONNTRACK_EXP_UPDATE
printf("NFNLGRP_CONNTRACK_EXP_UPDATE %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_EXP_UPDATE));
#endif
#ifdef NFNLGRP_CONNTRACK_NEW
printf("NFNLGRP_CONNTRACK_NEW %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_NEW));
#endif
#ifdef NFNLGRP_CONNTRACK_UPDATE
printf("NFNLGRP_CONNTRACK_UPDATE %llu\n", (unsigned long long)(long long)(NFNLGRP_CONNTRACK_UPDATE));
#endif
#ifdef NFNLGRP_NFTABLES
printf("NFNLGRP_NFTABLES %llu\n", (unsigned long long)(long long)(NFNLGRP_NFTABLES));
#endif
#ifdef NFNLGRP_NFTRACE
printf("NFNLGRP_NFTRACE %llu\n", (unsigned long long)(long long)(NFNLGRP_NFTRACE));
#endif
#ifdef NFNLGRP_NONE
printf("NFNLGRP_NONE %llu\n", (unsigned long long)(long long)(NFNLGRP_NONE));
#endif
#ifdef NFNL_BATCH_GENID
printf("NFNL_BATCH_GENID %llu\n", (unsigned long long)(long long)(NFNL_BATCH_GENID));
#endif
#ifdef NFNL_BATCH_UNSPEC
printf("NFNL_BATCH_UNSPEC %llu\n", (unsigned long long)(long long)(NFNL_BATCH_UNSPEC));
#endif
#ifdef NFNL_MSG_BATCH_BEGIN
printf("NFNL_MSG_BATCH_BEGIN %llu\n", (unsigned long long)(long long)(NFNL_MSG_BATCH_BEGIN));
#endif
#ifdef NFNL_MSG_BATCH_END
printf("NFNL_MSG_BATCH_END %llu\n", (unsigned long long)(long long)(NFNL_MSG_BATCH_END));
#endif
#ifdef NFNL_SUBSYS_ACCT
printf("NFNL_SUBSYS_ACCT %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_ACCT));
#endif
#ifdef NFNL_SUBSYS_COUNT
printf("NFNL_SUBSYS_COUNT %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_COUNT));
#endif
#ifdef NFNL_SUBSYS_CTHELPER
printf("NFNL_SUBSYS_CTHELPER %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_CTHELPER));
#endif
#ifdef NFNL_SUBSYS_CTNETLINK
printf("NFNL_SUBSYS_CTNETLINK %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_CTNETLINK));
#endif
#ifdef NFNL_SUBSYS_CTNETLINK_EXP
printf("NFNL_SUBSYS_CTNETLINK_EXP %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_CTNETLINK_EXP));
#endif
#ifdef NFNL_SUBSYS_CTNETLINK_TIMEOUT
printf("NFNL_SUBSYS_CTNETLINK_TIMEOUT %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_CTNETLINK_TIMEOUT));
#endif
#ifdef NFNL_SUBSYS_HOOK
printf("NFNL_SUBSYS_HOOK %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_HOOK));
#endif
#ifdef NFNL_SUBSYS_IPSET
printf("NFNL_SUBSYS_IPSET %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_IPSET));
#endif
#ifdef NFNL_SUBSYS_NFTABLES
printf("NFNL_SUBSYS_NFTABLES %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_NFTABLES));
#endif
#ifdef NFNL_SUBSYS_NFT_COMPAT
printf("NFNL_SUBSYS_NFT_COMPAT %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_NFT_COMPAT));
#endif
#ifdef NFNL_SUBSYS_NONE
printf("NFNL_SUBSYS_NONE %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_NONE));
#endif
#ifdef NFNL_SUBSYS_OSF
printf("NFNL_SUBSYS_OSF %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_OSF));
#endif
#ifdef NFNL_SUBSYS_QUEUE
printf("NFNL_SUBSYS_QUEUE %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_QUEUE));
#endif
#ifdef NFNL_SUBSYS_ULOG
printf("NFNL_SUBSYS_ULOG %llu\n", (unsigned long long)(long long)(NFNL_SUBSYS_ULOG));
#endif
#ifdef NFPROTO_ARP
printf("NFPROTO_ARP %llu\n", (unsigned long long)(long long)(NFPROTO_ARP));
#endif
#ifdef NFPROTO_BRIDGE
printf("NFPROTO_BRIDGE %llu\n", (unsigned long long)(long long)(NFPROTO_BRIDGE));
#endif
#ifdef NFPROTO_DECNET
printf("NFPROTO_DECNET %llu\n", (unsigned long long)(long long)(NFPROTO_DECNET));
#endif
#ifdef NFPROTO_INET
printf("NFPROTO_INET %llu\n", (unsigned long long)(long long)(NFPROTO_INET));
#endif
#ifdef NFPROTO_IPV4
printf("NFPROTO_IPV4 %llu\n", (unsigned long long)(long long)(NFPROTO_IPV4));
#endif
#ifdef NFPROTO_IPV6
printf("NFPROTO_IPV6 %llu\n", (unsigned long long)(long long)(NFPROTO_IPV6));
#endif
#ifdef NFPROTO_NETDEV
printf("NFPROTO_NETDEV %llu\n", (unsigned long long)(long long)(NFPROTO_NETDEV));
#endif
#ifdef NFPROTO_NUMPROTO
printf("NFPROTO_NUMPROTO %llu\n", (unsigned long long)(long long)(NFPROTO_NUMPROTO));
#endif
#ifdef NFPROTO_UNSPEC
printf("NFPROTO_UNSPEC %llu\n", (unsigned long long)(long long)(NFPROTO_UNSPEC));
#endif
#ifdef NFQA_CAP_LEN
printf("NFQA_CAP_LEN %llu\n", (unsigned long long)(long long)(NFQA_CAP_LEN));
#endif
#ifdef NFQA_CFG_CMD
printf("NFQA_CFG_CMD %llu\n", (unsigned long long)(long long)(NFQA_CFG_CMD));
#endif
#ifdef NFQA_CFG_FLAGS
printf("NFQA_CFG_FLAGS %llu\n", (unsigned long long)(long long)(NFQA_CFG_FLAGS));
#endif
#ifdef NFQA_CFG_F_CONNTRACK
printf("NFQA_CFG_F_CONNTRACK %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_CONNTRACK));
#endif
#ifdef NFQA_CFG_F_FAIL_OPEN
printf("NFQA_CFG_F_FAIL_OPEN %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_FAIL_OPEN));
#endif
#ifdef NFQA_CFG_F_GSO
printf("NFQA_CFG_F_GSO %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_GSO));
#endif
#ifdef NFQA_CFG_F_MAX
printf("NFQA_CFG_F_MAX %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_MAX));
#endif
#ifdef NFQA_CFG_F_SECCTX
printf("NFQA_CFG_F_SECCTX %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_SECCTX));
#endif
#ifdef NFQA_CFG_F_UID_GID
printf("NFQA_CFG_F_UID_GID %llu\n", (unsigned long long)(long long)(NFQA_CFG_F_UID_GID));
#endif
#ifdef NFQA_CFG_MASK
printf("NFQA_CFG_MASK %llu\n", (unsigned long long)(long long)(NFQA_CFG_MASK));
#endif
#ifdef NFQA_CFG_PARAMS
printf("NFQA_CFG_PARAMS %llu\n", (unsigned long long)(long long)(NFQA_CFG_PARAMS));
#endif
#ifdef NFQA_CFG_QUEUE_MAXLEN
printf("NFQA_CFG_QUEUE_MAXLEN %llu\n", (unsigned long long)(long long)(NFQA_CFG_QUEUE_MAXLEN));
#endif
#ifdef NFQA_CFG_UNSPEC
printf("NFQA_CFG_UNSPEC %llu\n", (unsigned long long)(long long)(NFQA_CFG_UNSPEC));
#endif
#ifdef NFQA_CT
printf("NFQA_CT %llu\n", (unsigned long long)(long long)(NFQA_CT));
#endif
#ifdef NFQA_CT_INFO
printf("NFQA_CT_INFO %llu\n", (unsigned long long)(long long)(NFQA_CT_INFO));
#endif
#ifdef NFQA_EXP
printf("NFQA_EXP %llu\n", (unsigned long long)(long long)(NFQA_EXP));
#endif
#ifdef NFQA_GID
printf("NFQA_GID %llu\n", (unsigned long long)(long long)(NFQA_GID));
#endif
#ifdef NFQA_HWADDR
printf("NFQA_HWADDR %llu\n", (unsigned long long)(long long)(NFQA_HWADDR));
#endif
#ifdef NFQA_IFINDEX_INDEV
printf("NFQA_IFINDEX_INDEV %llu\n", (unsigned long long)(long long)(NFQA_IFINDEX_INDEV));
#endif
#ifdef NFQA_IFINDEX_OUTDEV
printf("NFQA_IFINDEX_OUTDEV %llu\n", (unsigned long long)(long long)(NFQA_IFINDEX_OUTDEV));
#endif
#ifdef NFQA_IFINDEX_PHYSINDEV
printf("NFQA_IFINDEX_PHYSINDEV %llu\n", (unsigned long long)(long long)(NFQA_IFINDEX_PHYSINDEV));
#endif
#ifdef NFQA_IFINDEX_PHYSOUTDEV
printf("NFQA_IFINDEX_PHYSOUTDEV %llu\n", (unsigned long long)(long long)(NFQA_IFINDEX_PHYSOUTDEV));
#endif
#ifdef NFQA_L2HDR
printf("NFQA_L2HDR %llu\n", (unsigned long long)(long long)(NFQA_L2HDR));
#endif
#ifdef NFQA_MARK
printf("NFQA_MARK %llu\n", (unsigned long long)(long long)(NFQA_MARK));
#endif
#ifdef NFQA_PACKET_HDR
printf("NFQA_PACKET_HDR %llu\n", (unsigned long long)(long long)(NFQA_PACKET_HDR));
#endif
#ifdef NFQA_PAYLOAD
printf("NFQA_PAYLOAD %llu\n", (unsigned long long)(long long)(NFQA_PAYLOAD));
#endif
#ifdef NFQA_PRIORITY
printf("NFQA_PRIORITY %llu\n", (unsigned long long)(long long)(NFQA_PRIORITY));
#endif
#ifdef NFQA_SECCTX
printf("NFQA_SECCTX %llu\n", (unsigned long long)(long long)(NFQA_SECCTX));
#endif
#ifdef NFQA_SKB_CSUMNOTREADY
printf("NFQA_SKB_CSUMNOTREADY %llu\n", (unsigned long long)(long long)(NFQA_SKB_CSUMNOTREADY));
#endif
#ifdef NFQA_SKB_CSUM_NOTVERIFIED
printf("NFQA_SKB_CSUM_NOTVERIFIED %llu\n", (unsigned long long)(long long)(NFQA_SKB_CSUM_NOTVERIFIED));
#endif
#ifdef NFQA_SKB_GSO
printf("NFQA_SKB_GSO %llu\n", (unsigned long long)(long long)(NFQA_SKB_GSO));
#endif
#ifdef NFQA_SKB_INFO
printf("NFQA_SKB_INFO %llu\n", (unsigned long long)(long long)(NFQA_SKB_INFO));
#endif
#ifdef NFQA_TIMESTAMP
printf("NFQA_TIMESTAMP %llu\n", (unsigned long long)(long long)(NFQA_TIMESTAMP));
#endif
#ifdef NFQA_UID
printf("NFQA_UID %llu\n", (unsigned long long)(long long)(NFQA_UID));
#endif
#ifdef NFQA_UNSPEC
printf("NFQA_UNSPEC %llu\n", (unsigned long long)(long long)(NFQA_UNSPEC));
#endif
#ifdef NFQA_VERDICT_HDR
printf("NFQA_VERDICT_HDR %llu\n", (unsigned long long)(long long)(NFQA_VERDICT_HDR));
#endif
#ifdef NFQA_VLAN
printf("NFQA_VLAN %llu\n", (unsigned long long)(long long)(NFQA_VLAN));
#endif
#ifdef NFQA_VLAN_PROTO
printf("NFQA_VLAN_PROTO %llu\n", (unsigned long long)(long long)(NFQA_VLAN_PROTO));
#endif
#ifdef NFQA_VLAN_TCI
printf("NFQA_VLAN_TCI %llu\n", (unsigned long long)(long long)(NFQA_VLAN_TCI));
#endif
#ifdef NFQA_VLAN_UNSPEC
printf("NFQA_VLAN_UNSPEC %llu\n", (unsigned long long)(long long)(NFQA_VLAN_UNSPEC));
#endif
#ifdef NFQNL_CFG_CMD_BIND
printf("NFQNL_CFG_CMD_BIND %llu\n", (unsigned long long)(long long)(NFQNL_CFG_CMD_BIND));
#endif
#ifdef NFQNL_CFG_CMD_NONE
printf("NFQNL_CFG_CMD_NONE %llu\n", (unsigned long long)(long long)(NFQNL_CFG_CMD_NONE));
#endif
#ifdef NFQNL_CFG_CMD_PF_BIND
printf("NFQNL_CFG_CMD_PF_BIND %llu\n", (unsigned long long)(long long)(NFQNL_CFG_CMD_PF_BIND));
#endif
#ifdef NFQNL_CFG_CMD_PF_UNBIND
printf("NFQNL_CFG_CMD_PF_UNBIND %llu\n", (unsigned long long)(long long)(NFQNL_CFG_CMD_PF_UNBIND));
#endif
#ifdef NFQNL_CFG_CMD_UNBIND
printf("NFQNL_CFG_CMD_UNBIND %llu\n", (unsigned long long)(long long)(NFQNL_CFG_CMD_UNBIND));
#endif
#ifdef NFQNL_COPY_META
printf("NFQNL_COPY_META %llu\n", (unsigned long long)(long long)(NFQNL_COPY_META));
#endif
#ifdef NFQNL_COPY_NONE
printf("NFQNL_COPY_NONE %llu\n", (unsigned long long)(long long)(NFQNL_COPY_NONE));
#endif
#ifdef NFQNL_COPY_PACKET
printf("NFQNL_COPY_PACKET %llu\n", (unsigned long long)(long long)(NFQNL_COPY_PACKET));
#endif
#ifdef NFQNL_MSG_CONFIG
printf("NFQNL_MSG_CONFIG %llu\n", (unsigned long long)(long long)(NFQNL_MSG_CONFIG));
#endif
#ifdef NFQNL_MSG_PACKET
printf("NFQNL_MSG_PACKET %llu\n", (unsigned long long)(long long)(NFQNL_MSG_PACKET));
#endif
#ifdef NFQNL_MSG_VERDICT
printf("NFQNL_MSG_VERDICT %llu\n", (unsigned long long)(long long)(NFQNL_MSG_VERDICT));
#endif
#ifdef NFQNL_MSG_VERDICT_BATCH
printf("NFQNL_MSG_VERDICT_BATCH %llu\n", (unsigned long long)(long long)(NFQNL_MSG_VERDICT_BATCH));
#endif
#ifdef NFS_SUPER_MAGIC
printf("NFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(NFS_SUPER_MAGIC));
#endif
#ifdef NFT_BREAK
printf("NFT_BREAK %llu\n", (unsigned long long)(long long)(NFT_BREAK));
#endif
#ifdef NFT_BYTEORDER_HTON
printf("NFT_BYTEORDER_HTON %llu\n", (unsigned long long)(long long)(NFT_BYTEORDER_HTON));
#endif
#ifdef NFT_BYTEORDER_NTOH
printf("NFT_BYTEORDER_NTOH %llu\n", (unsigned long long)(long long)(NFT_BYTEORDER_NTOH));
#endif
#ifdef NFT_CHAIN_MAXNAMELEN
printf("NFT_CHAIN_MAXNAMELEN %llu\n", (unsigned long long)(long long)(NFT_CHAIN_MAXNAMELEN));
#endif
#ifdef NFT_CMP_EQ
printf("NFT_CMP_EQ %llu\n", (unsigned long long)(long long)(NFT_CMP_EQ));
#endif
#ifdef NFT_CMP_GT
printf("NFT_CMP_GT %llu\n", (unsigned long long)(long long)(NFT_CMP_GT));
#endif
#ifdef NFT_CMP_GTE
printf("NFT_CMP_GTE %llu\n", (unsigned long long)(long long)(NFT_CMP_GTE));
#endif
#ifdef NFT_CMP_LT
printf("NFT_CMP_LT %llu\n", (unsigned long long)(long long)(NFT_CMP_LT));
#endif
#ifdef NFT_CMP_LTE
printf("NFT_CMP_LTE %llu\n", (unsigned long long)(long long)(NFT_CMP_LTE));
#endif
#ifdef NFT_CMP_NEQ
printf("NFT_CMP_NEQ %llu\n", (unsigned long long)(long long)(NFT_CMP_NEQ));
#endif
#ifdef NFT_CONTINUE
printf("NFT_CONTINUE %llu\n", (unsigned long long)(long long)(NFT_CONTINUE));
#endif
#ifdef NFT_CT_AVGPKT
printf("NFT_CT_AVGPKT %llu\n", (unsigned long long)(long long)(NFT_CT_AVGPKT));
#endif
#ifdef NFT_CT_BYTES
printf("NFT_CT_BYTES %llu\n", (unsigned long long)(long long)(NFT_CT_BYTES));
#endif
#ifdef NFT_CT_DIRECTION
printf("NFT_CT_DIRECTION %llu\n", (unsigned long long)(long long)(NFT_CT_DIRECTION));
#endif
#ifdef NFT_CT_DST
printf("NFT_CT_DST %llu\n", (unsigned long long)(long long)(NFT_CT_DST));
#endif
#ifdef NFT_CT_DST_IP
printf("NFT_CT_DST_IP %llu\n", (unsigned long long)(long long)(NFT_CT_DST_IP));
#endif
#ifdef NFT_CT_DST_IP6
printf("NFT_CT_DST_IP6 %llu\n", (unsigned long long)(long long)(NFT_CT_DST_IP6));
#endif
#ifdef NFT_CT_EVENTMASK
printf("NFT_CT_EVENTMASK %llu\n", (unsigned long long)(long long)(NFT_CT_EVENTMASK));
#endif
#ifdef NFT_CT_EXPIRATION
printf("NFT_CT_EXPIRATION %llu\n", (unsigned long long)(long long)(NFT_CT_EXPIRATION));
#endif
#ifdef NFT_CT_HELPER
printf("NFT_CT_HELPER %llu\n", (unsigned long long)(long long)(NFT_CT_HELPER));
#endif
#ifdef NFT_CT_L3PROTOCOL
printf("NFT_CT_L3PROTOCOL %llu\n", (unsigned long long)(long long)(NFT_CT_L3PROTOCOL));
#endif
#ifdef NFT_CT_LABELS
printf("NFT_CT_LABELS %llu\n", (unsigned long long)(long long)(NFT_CT_LABELS));
#endif
#ifdef NFT_CT_MARK
printf("NFT_CT_MARK %llu\n", (unsigned long long)(long long)(NFT_CT_MARK));
#endif
#ifdef NFT_CT_PKTS
printf("NFT_CT_PKTS %llu\n", (unsigned long long)(long long)(NFT_CT_PKTS));
#endif
#ifdef NFT_CT_PROTOCOL
printf("NFT_CT_PROTOCOL %llu\n", (unsigned long long)(long long)(NFT_CT_PROTOCOL));
#endif
#ifdef NFT_CT_PROTO_DST
printf("NFT_CT_PROTO_DST %llu\n", (unsigned long long)(long long)(NFT_CT_PROTO_DST));
#endif
#ifdef NFT_CT_PROTO_SRC
printf("NFT_CT_PROTO_SRC %llu\n", (unsigned long long)(long long)(NFT_CT_PROTO_SRC));
#endif
#ifdef NFT_CT_SECMARK
printf("NFT_CT_SECMARK %llu\n", (unsigned long long)(long long)(NFT_CT_SECMARK));
#endif
#ifdef NFT_CT_SRC
printf("NFT_CT_SRC %llu\n", (unsigned long long)(long long)(NFT_CT_SRC));
#endif
#ifdef NFT_CT_SRC_IP
printf("NFT_CT_SRC_IP %llu\n", (unsigned long long)(long long)(NFT_CT_SRC_IP));
#endif
#ifdef NFT_CT_SRC_IP6
printf("NFT_CT_SRC_IP6 %llu\n", (unsigned long long)(long long)(NFT_CT_SRC_IP6));
#endif
#ifdef NFT_CT_STATE
printf("NFT_CT_STATE %llu\n", (unsigned long long)(long long)(NFT_CT_STATE));
#endif
#ifdef NFT_CT_STATUS
printf("NFT_CT_STATUS %llu\n", (unsigned long long)(long long)(NFT_CT_STATUS));
#endif
#ifdef NFT_CT_ZONE
printf("NFT_CT_ZONE %llu\n", (unsigned long long)(long long)(NFT_CT_ZONE));
#endif
#ifdef NFT_DATA_RESERVED_MASK
printf("NFT_DATA_RESERVED_MASK %llu\n", (unsigned long long)(long long)(NFT_DATA_RESERVED_MASK));
#endif
#ifdef NFT_DATA_VALUE
printf("NFT_DATA_VALUE %llu\n", (unsigned long long)(long long)(NFT_DATA_VALUE));
#endif
#ifdef NFT_DATA_VALUE_MAXLEN
printf("NFT_DATA_VALUE_MAXLEN %llu\n", (unsigned long long)(long long)(NFT_DATA_VALUE_MAXLEN));
#endif
#ifdef NFT_DATA_VERDICT
printf("NFT_DATA_VERDICT %llu\n", (unsigned long long)(long long)(NFT_DATA_VERDICT));
#endif
#ifdef NFT_DYNSET_F_INV
printf("NFT_DYNSET_F_INV %llu\n", (unsigned long long)(long long)(NFT_DYNSET_F_INV));
#endif
#ifdef NFT_DYNSET_OP_ADD
printf("NFT_DYNSET_OP_ADD %llu\n", (unsigned long long)(long long)(NFT_DYNSET_OP_ADD));
#endif
#ifdef NFT_DYNSET_OP_UPDATE
printf("NFT_DYNSET_OP_UPDATE %llu\n", (unsigned long long)(long long)(NFT_DYNSET_OP_UPDATE));
#endif
#ifdef NFT_GOTO
printf("NFT_GOTO %llu\n", (unsigned long long)(long long)(NFT_GOTO));
#endif
#ifdef NFT_JUMP
printf("NFT_JUMP %llu\n", (unsigned long long)(long long)(NFT_JUMP));
#endif
#ifdef NFT_LIMIT_F_INV
printf("NFT_LIMIT_F_INV %llu\n", (unsigned long long)(long long)(NFT_LIMIT_F_INV));
#endif
#ifdef NFT_LIMIT_PKTS
printf("NFT_LIMIT_PKTS %llu\n", (unsigned long long)(long long)(NFT_LIMIT_PKTS));
#endif
#ifdef NFT_LIMIT_PKT_BYTES
printf("NFT_LIMIT_PKT_BYTES %llu\n", (unsigned long long)(long long)(NFT_LIMIT_PKT_BYTES));
#endif
#ifdef NFT_LOOKUP_F_INV
printf("NFT_LOOKUP_F_INV %llu\n", (unsigned long long)(long long)(NFT_LOOKUP_F_INV));
#endif
#ifdef NFT_META_BRI_IIFNAME
printf("NFT_META_BRI_IIFNAME %llu\n", (unsigned long long)(long long)(NFT_META_BRI_IIFNAME));
#endif
#ifdef NFT_META_BRI_OIFNAME
printf("NFT_META_BRI_OIFNAME %llu\n", (unsigned long long)(long long)(NFT_META_BRI_OIFNAME));
#endif
#ifdef NFT_META_CGROUP
printf("NFT_META_CGROUP %llu\n", (unsigned long long)(long long)(NFT_META_CGROUP));
#endif
#ifdef NFT_META_CPU
printf("NFT_META_CPU %llu\n", (unsigned long long)(long long)(NFT_META_CPU));
#endif
#ifdef NFT_META_IIF
printf("NFT_META_IIF %llu\n", (unsigned long long)(long long)(NFT_META_IIF));
#endif
#ifdef NFT_META_IIFGROUP
printf("NFT_META_IIFGROUP %llu\n", (unsigned long long)(long long)(NFT_META_IIFGROUP));
#endif
#ifdef NFT_META_IIFNAME
printf("NFT_META_IIFNAME %llu\n", (unsigned long long)(long long)(NFT_META_IIFNAME));
#endif
#ifdef NFT_META_IIFTYPE
printf("NFT_META_IIFTYPE %llu\n", (unsigned long long)(long long)(NFT_META_IIFTYPE));
#endif
#ifdef NFT_META_L4PROTO
printf("NFT_META_L4PROTO %llu\n", (unsigned long long)(long long)(NFT_META_L4PROTO));
#endif
#ifdef NFT_META_LEN
printf("NFT_META_LEN %llu\n", (unsigned long long)(long long)(NFT_META_LEN));
#endif
#ifdef NFT_META_MARK
printf("NFT_META_MARK %llu\n", (unsigned long long)(long long)(NFT_META_MARK));
#endif
#ifdef NFT_META_NFPROTO
printf("NFT_META_NFPROTO %llu\n", (unsigned long long)(long long)(NFT_META_NFPROTO));
#endif
#ifdef NFT_META_NFTRACE
printf("NFT_META_NFTRACE %llu\n", (unsigned long long)(long long)(NFT_META_NFTRACE));
#endif
#ifdef NFT_META_OIF
printf("NFT_META_OIF %llu\n", (unsigned long long)(long long)(NFT_META_OIF));
#endif
#ifdef NFT_META_OIFGROUP
printf("NFT_META_OIFGROUP %llu\n", (unsigned long long)(long long)(NFT_META_OIFGROUP));
#endif
#ifdef NFT_META_OIFNAME
printf("NFT_META_OIFNAME %llu\n", (unsigned long long)(long long)(NFT_META_OIFNAME));
#endif
#ifdef NFT_META_OIFTYPE
printf("NFT_META_OIFTYPE %llu\n", (unsigned long long)(long long)(NFT_META_OIFTYPE));
#endif
#ifdef NFT_META_PKTTYPE
printf("NFT_META_PKTTYPE %llu\n", (unsigned long long)(long long)(NFT_META_PKTTYPE));
#endif
#ifdef NFT_META_PRANDOM
printf("NFT_META_PRANDOM %llu\n", (unsigned long long)(long long)(NFT_META_PRANDOM));
#endif
#ifdef NFT_META_PRIORITY
printf("NFT_META_PRIORITY %llu\n", (unsigned long long)(long long)(NFT_META_PRIORITY));
#endif
#ifdef NFT_META_PROTOCOL
printf("NFT_META_PROTOCOL %llu\n", (unsigned long long)(long long)(NFT_META_PROTOCOL));
#endif
#ifdef NFT_META_RTCLASSID
printf("NFT_META_RTCLASSID %llu\n", (unsigned long long)(long long)(NFT_META_RTCLASSID));
#endif
#ifdef NFT_META_SECMARK
printf("NFT_META_SECMARK %llu\n", (unsigned long long)(long long)(NFT_META_SECMARK));
#endif
#ifdef NFT_META_SKGID
printf("NFT_META_SKGID %llu\n", (unsigned long long)(long long)(NFT_META_SKGID));
#endif
#ifdef NFT_META_SKUID
printf("NFT_META_SKUID %llu\n", (unsigned long long)(long long)(NFT_META_SKUID));
#endif
#ifdef NFT_MSG_DELCHAIN
printf("NFT_MSG_DELCHAIN %llu\n", (unsigned long long)(long long)(NFT_MSG_DELCHAIN));
#endif
#ifdef NFT_MSG_DELOBJ
printf("NFT_MSG_DELOBJ %llu\n", (unsigned long long)(long long)(NFT_MSG_DELOBJ));
#endif
#ifdef NFT_MSG_DELRULE
printf("NFT_MSG_DELRULE %llu\n", (unsigned long long)(long long)(NFT_MSG_DELRULE));
#endif
#ifdef NFT_MSG_DELSET
printf("NFT_MSG_DELSET %llu\n", (unsigned long long)(long long)(NFT_MSG_DELSET));
#endif
#ifdef NFT_MSG_DELSETELEM
printf("NFT_MSG_DELSETELEM %llu\n", (unsigned long long)(long long)(NFT_MSG_DELSETELEM));
#endif
#ifdef NFT_MSG_DELTABLE
printf("NFT_MSG_DELTABLE %llu\n", (unsigned long long)(long long)(NFT_MSG_DELTABLE));
#endif
#ifdef NFT_MSG_GETCHAIN
printf("NFT_MSG_GETCHAIN %llu\n", (unsigned long long)(long long)(NFT_MSG_GETCHAIN));
#endif
#ifdef NFT_MSG_GETGEN
printf("NFT_MSG_GETGEN %llu\n", (unsigned long long)(long long)(NFT_MSG_GETGEN));
#endif
#ifdef NFT_MSG_GETOBJ
printf("NFT_MSG_GETOBJ %llu\n", (unsigned long long)(long long)(NFT_MSG_GETOBJ));
#endif
#ifdef NFT_MSG_GETOBJ_RESET
printf("NFT_MSG_GETOBJ_RESET %llu\n", (unsigned long long)(long long)(NFT_MSG_GETOBJ_RESET));
#endif
#ifdef NFT_MSG_GETRULE
printf("NFT_MSG_GETRULE %llu\n", (unsigned long long)(long long)(NFT_MSG_GETRULE));
#endif
#ifdef NFT_MSG_GETSET
printf("NFT_MSG_GETSET %llu\n", (unsigned long long)(long long)(NFT_MSG_GETSET));
#endif
#ifdef NFT_MSG_GETSETELEM
printf("NFT_MSG_GETSETELEM %llu\n", (unsigned long long)(long long)(NFT_MSG_GETSETELEM));
#endif
#ifdef NFT_MSG_GETTABLE
printf("NFT_MSG_GETTABLE %llu\n", (unsigned long long)(long long)(NFT_MSG_GETTABLE));
#endif
#ifdef NFT_MSG_MAX
printf("NFT_MSG_MAX %llu\n", (unsigned long long)(long long)(NFT_MSG_MAX));
#endif
#ifdef NFT_MSG_NEWCHAIN
printf("NFT_MSG_NEWCHAIN %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWCHAIN));
#endif
#ifdef NFT_MSG_NEWGEN
printf("NFT_MSG_NEWGEN %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWGEN));
#endif
#ifdef NFT_MSG_NEWOBJ
printf("NFT_MSG_NEWOBJ %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWOBJ));
#endif
#ifdef NFT_MSG_NEWRULE
printf("NFT_MSG_NEWRULE %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWRULE));
#endif
#ifdef NFT_MSG_NEWSET
printf("NFT_MSG_NEWSET %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWSET));
#endif
#ifdef NFT_MSG_NEWSETELEM
printf("NFT_MSG_NEWSETELEM %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWSETELEM));
#endif
#ifdef NFT_MSG_NEWTABLE
printf("NFT_MSG_NEWTABLE %llu\n", (unsigned long long)(long long)(NFT_MSG_NEWTABLE));
#endif
#ifdef NFT_MSG_TRACE
printf("NFT_MSG_TRACE %llu\n", (unsigned long long)(long long)(NFT_MSG_TRACE));
#endif
#ifdef NFT_NAT_DNAT
printf("NFT_NAT_DNAT %llu\n", (unsigned long long)(long long)(NFT_NAT_DNAT));
#endif
#ifdef NFT_NAT_SNAT
printf("NFT_NAT_SNAT %llu\n", (unsigned long long)(long long)(NFT_NAT_SNAT));
#endif
#ifdef NFT_NG_INCREMENTAL
printf("NFT_NG_INCREMENTAL %llu\n", (unsigned long long)(long long)(NFT_NG_INCREMENTAL));
#endif
#ifdef NFT_NG_RANDOM
printf("NFT_NG_RANDOM %llu\n", (unsigned long long)(long long)(NFT_NG_RANDOM));
#endif
#ifdef NFT_OBJ_MAXNAMELEN
printf("NFT_OBJ_MAXNAMELEN %llu\n", (unsigned long long)(long long)(NFT_OBJ_MAXNAMELEN));
#endif
#ifdef NFT_PAYLOAD_CSUM_INET
printf("NFT_PAYLOAD_CSUM_INET %llu\n", (unsigned long long)(long long)(NFT_PAYLOAD_CSUM_INET));
#endif
#ifdef NFT_PAYLOAD_CSUM_NONE
printf("NFT_PAYLOAD_CSUM_NONE %llu\n", (unsigned long long)(long long)(NFT_PAYLOAD_CSUM_NONE));
#endif
#ifdef NFT_PAYLOAD_LL_HEADER
printf("NFT_PAYLOAD_LL_HEADER %llu\n", (unsigned long long)(long long)(NFT_PAYLOAD_LL_HEADER));
#endif
#ifdef NFT_PAYLOAD_NETWORK_HEADER
printf("NFT_PAYLOAD_NETWORK_HEADER %llu\n", (unsigned long long)(long long)(NFT_PAYLOAD_NETWORK_HEADER));
#endif
#ifdef NFT_PAYLOAD_TRANSPORT_HEADER
printf("NFT_PAYLOAD_TRANSPORT_HEADER %llu\n", (unsigned long long)(long long)(NFT_PAYLOAD_TRANSPORT_HEADER));
#endif
#ifdef NFT_QUEUE_FLAG_BYPASS
printf("NFT_QUEUE_FLAG_BYPASS %llu\n", (unsigned long long)(long long)(NFT_QUEUE_FLAG_BYPASS));
#endif
#ifdef NFT_QUEUE_FLAG_CPU_FANOUT
printf("NFT_QUEUE_FLAG_CPU_FANOUT %llu\n", (unsigned long long)(long long)(NFT_QUEUE_FLAG_CPU_FANOUT));
#endif
#ifdef NFT_QUEUE_FLAG_MASK
printf("NFT_QUEUE_FLAG_MASK %llu\n", (unsigned long long)(long long)(NFT_QUEUE_FLAG_MASK));
#endif
#ifdef NFT_QUOTA_F_INV
printf("NFT_QUOTA_F_INV %llu\n", (unsigned long long)(long long)(NFT_QUOTA_F_INV));
#endif
#ifdef NFT_RANGE_EQ
printf("NFT_RANGE_EQ %llu\n", (unsigned long long)(long long)(NFT_RANGE_EQ));
#endif
#ifdef NFT_RANGE_NEQ
printf("NFT_RANGE_NEQ %llu\n", (unsigned long long)(long long)(NFT_RANGE_NEQ));
#endif
#ifdef NFT_REG32_00
printf("NFT_REG32_00 %llu\n", (unsigned long long)(long long)(NFT_REG32_00));
#endif
#ifdef NFT_REG32_01
printf("NFT_REG32_01 %llu\n", (unsigned long long)(long long)(NFT_REG32_01));
#endif
#ifdef NFT_REG32_02
printf("NFT_REG32_02 %llu\n", (unsigned long long)(long long)(NFT_REG32_02));
#endif
#ifdef NFT_REG32_03
printf("NFT_REG32_03 %llu\n", (unsigned long long)(long long)(NFT_REG32_03));
#endif
#ifdef NFT_REG32_04
printf("NFT_REG32_04 %llu\n", (unsigned long long)(long long)(NFT_REG32_04));
#endif
#ifdef NFT_REG32_05
printf("NFT_REG32_05 %llu\n", (unsigned long long)(long long)(NFT_REG32_05));
#endif
#ifdef NFT_REG32_06
printf("NFT_REG32_06 %llu\n", (unsigned long long)(long long)(NFT_REG32_06));
#endif
#ifdef NFT_REG32_07
printf("NFT_REG32_07 %llu\n", (unsigned long long)(long long)(NFT_REG32_07));
#endif
#ifdef NFT_REG32_08
printf("NFT_REG32_08 %llu\n", (unsigned long long)(long long)(NFT_REG32_08));
#endif
#ifdef NFT_REG32_09
printf("NFT_REG32_09 %llu\n", (unsigned long long)(long long)(NFT_REG32_09));
#endif
#ifdef NFT_REG32_10
printf("NFT_REG32_10 %llu\n", (unsigned long long)(long long)(NFT_REG32_10));
#endif
#ifdef NFT_REG32_11
printf("NFT_REG32_11 %llu\n", (unsigned long long)(long long)(NFT_REG32_11));
#endif
#ifdef NFT_REG32_12
printf("NFT_REG32_12 %llu\n", (unsigned long long)(long long)(NFT_REG32_12));
#endif
#ifdef NFT_REG32_13
printf("NFT_REG32_13 %llu\n", (unsigned long long)(long long)(NFT_REG32_13));
#endif
#ifdef NFT_REG32_14
printf("NFT_REG32_14 %llu\n", (unsigned long long)(long long)(NFT_REG32_14));
#endif
#ifdef NFT_REG32_15
printf("NFT_REG32_15 %llu\n", (unsigned long long)(long long)(NFT_REG32_15));
#endif
#ifdef NFT_REG32_SIZE
printf("NFT_REG32_SIZE %llu\n", (unsigned long long)(long long)(NFT_REG32_SIZE));
#endif
#ifdef NFT_REG_1
printf("NFT_REG_1 %llu\n", (unsigned long long)(long long)(NFT_REG_1));
#endif
#ifdef NFT_REG_2
printf("NFT_REG_2 %llu\n", (unsigned long long)(long long)(NFT_REG_2));
#endif
#ifdef NFT_REG_3
printf("NFT_REG_3 %llu\n", (unsigned long long)(long long)(NFT_REG_3));
#endif
#ifdef NFT_REG_4
printf("NFT_REG_4 %llu\n", (unsigned long long)(long long)(NFT_REG_4));
#endif
#ifdef NFT_REG_SIZE
printf("NFT_REG_SIZE %llu\n", (unsigned long long)(long long)(NFT_REG_SIZE));
#endif
#ifdef NFT_REG_VERDICT
printf("NFT_REG_VERDICT %llu\n", (unsigned long long)(long long)(NFT_REG_VERDICT));
#endif
#ifdef NFT_REJECT_ICMPX_ADMIN_PROHIBITED
printf("NFT_REJECT_ICMPX_ADMIN_PROHIBITED %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMPX_ADMIN_PROHIBITED));
#endif
#ifdef NFT_REJECT_ICMPX_HOST_UNREACH
printf("NFT_REJECT_ICMPX_HOST_UNREACH %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMPX_HOST_UNREACH));
#endif
#ifdef NFT_REJECT_ICMPX_NO_ROUTE
printf("NFT_REJECT_ICMPX_NO_ROUTE %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMPX_NO_ROUTE));
#endif
#ifdef NFT_REJECT_ICMPX_PORT_UNREACH
printf("NFT_REJECT_ICMPX_PORT_UNREACH %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMPX_PORT_UNREACH));
#endif
#ifdef NFT_REJECT_ICMPX_UNREACH
printf("NFT_REJECT_ICMPX_UNREACH %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMPX_UNREACH));
#endif
#ifdef NFT_REJECT_ICMP_UNREACH
printf("NFT_REJECT_ICMP_UNREACH %llu\n", (unsigned long long)(long long)(NFT_REJECT_ICMP_UNREACH));
#endif
#ifdef NFT_REJECT_TCP_RST
printf("NFT_REJECT_TCP_RST %llu\n", (unsigned long long)(long long)(NFT_REJECT_TCP_RST));
#endif
#ifdef NFT_RETURN
printf("NFT_RETURN %llu\n", (unsigned long long)(long long)(NFT_RETURN));
#endif
#ifdef NFT_SET_ANONYMOUS
printf("NFT_SET_ANONYMOUS %llu\n", (unsigned long long)(long long)(NFT_SET_ANONYMOUS));
#endif
#ifdef NFT_SET_CONSTANT
printf("NFT_SET_CONSTANT %llu\n", (unsigned long long)(long long)(NFT_SET_CONSTANT));
#endif
#ifdef NFT_SET_ELEM_INTERVAL_END
printf("NFT_SET_ELEM_INTERVAL_END %llu\n", (unsigned long long)(long long)(NFT_SET_ELEM_INTERVAL_END));
#endif
#ifdef NFT_SET_EVAL
printf("NFT_SET_EVAL %llu\n", (unsigned long long)(long long)(NFT_SET_EVAL));
#endif
#ifdef NFT_SET_INTERVAL
printf("NFT_SET_INTERVAL %llu\n", (unsigned long long)(long long)(NFT_SET_INTERVAL));
#endif
#ifdef NFT_SET_MAP
printf("NFT_SET_MAP %llu\n", (unsigned long long)(long long)(NFT_SET_MAP));
#endif
#ifdef NFT_SET_MAXNAMELEN
printf("NFT_SET_MAXNAMELEN %llu\n", (unsigned long long)(long long)(NFT_SET_MAXNAMELEN));
#endif
#ifdef NFT_SET_POL_MEMORY
printf("NFT_SET_POL_MEMORY %llu\n", (unsigned long long)(long long)(NFT_SET_POL_MEMORY));
#endif
#ifdef NFT_SET_POL_PERFORMANCE
printf("NFT_SET_POL_PERFORMANCE %llu\n", (unsigned long long)(long long)(NFT_SET_POL_PERFORMANCE));
#endif
#ifdef NFT_SET_TIMEOUT
printf("NFT_SET_TIMEOUT %llu\n", (unsigned long long)(long long)(NFT_SET_TIMEOUT));
#endif
#ifdef NFT_TABLE_MAXNAMELEN
printf("NFT_TABLE_MAXNAMELEN %llu\n", (unsigned long long)(long long)(NFT_TABLE_MAXNAMELEN));
#endif
#ifdef NFT_TRACETYPE_POLICY
printf("NFT_TRACETYPE_POLICY %llu\n", (unsigned long long)(long long)(NFT_TRACETYPE_POLICY));
#endif
#ifdef NFT_TRACETYPE_RETURN
printf("NFT_TRACETYPE_RETURN %llu\n", (unsigned long long)(long long)(NFT_TRACETYPE_RETURN));
#endif
#ifdef NFT_TRACETYPE_RULE
printf("NFT_TRACETYPE_RULE %llu\n", (unsigned long long)(long long)(NFT_TRACETYPE_RULE));
#endif
#ifdef NFT_TRACETYPE_UNSPEC
printf("NFT_TRACETYPE_UNSPEC %llu\n", (unsigned long long)(long long)(NFT_TRACETYPE_UNSPEC));
#endif
#ifdef NFT_USERDATA_MAXLEN
printf("NFT_USERDATA_MAXLEN %llu\n", (unsigned long long)(long long)(NFT_USERDATA_MAXLEN));
#endif
#ifdef NFULA_CFG_CMD
printf("NFULA_CFG_CMD %llu\n", (unsigned long long)(long long)(NFULA_CFG_CMD));
#endif
#ifdef NFULA_CFG_FLAGS
printf("NFULA_CFG_FLAGS %llu\n", (unsigned long long)(long long)(NFULA_CFG_FLAGS));
#endif
#ifdef NFULA_CFG_MODE
printf("NFULA_CFG_MODE %llu\n", (unsigned long long)(long long)(NFULA_CFG_MODE));
#endif
#ifdef NFULA_CFG_NLBUFSIZ
printf("NFULA_CFG_NLBUFSIZ %llu\n", (unsigned long long)(long long)(NFULA_CFG_NLBUFSIZ));
#endif
#ifdef NFULA_CFG_QTHRESH
printf("NFULA_CFG_QTHRESH %llu\n", (unsigned long long)(long long)(NFULA_CFG_QTHRESH));
#endif
#ifdef NFULA_CFG_TIMEOUT
printf("NFULA_CFG_TIMEOUT %llu\n", (unsigned long long)(long long)(NFULA_CFG_TIMEOUT));
#endif
#ifdef NFULA_CFG_UNSPEC
printf("NFULA_CFG_UNSPEC %llu\n", (unsigned long long)(long long)(NFULA_CFG_UNSPEC));
#endif
#ifdef NFULA_CT
printf("NFULA_CT %llu\n", (unsigned long long)(long long)(NFULA_CT));
#endif
#ifdef NFULA_CT_INFO
printf("NFULA_CT_INFO %llu\n", (unsigned long long)(long long)(NFULA_CT_INFO));
#endif
#ifdef NFULA_GID
printf("NFULA_GID %llu\n", (unsigned long long)(long long)(NFULA_GID));
#endif
#ifdef NFULA_HWADDR
printf("NFULA_HWADDR %llu\n", (unsigned long long)(long long)(NFULA_HWADDR));
#endif
#ifdef NFULA_HWHEADER
printf("NFULA_HWHEADER %llu\n", (unsigned long long)(long long)(NFULA_HWHEADER));
#endif
#ifdef NFULA_HWLEN
printf("NFULA_HWLEN %llu\n", (unsigned long long)(long long)(NFULA_HWLEN));
#endif
#ifdef NFULA_HWTYPE
printf("NFULA_HWTYPE %llu\n", (unsigned long long)(long long)(NFULA_HWTYPE));
#endif
#ifdef NFULA_IFINDEX_INDEV
printf("NFULA_IFINDEX_INDEV %llu\n", (unsigned long long)(long long)(NFULA_IFINDEX_INDEV));
#endif
#ifdef NFULA_IFINDEX_OUTDEV
printf("NFULA_IFINDEX_OUTDEV %llu\n", (unsigned long long)(long long)(NFULA_IFINDEX_OUTDEV));
#endif
#ifdef NFULA_IFINDEX_PHYSINDEV
printf("NFULA_IFINDEX_PHYSINDEV %llu\n", (unsigned long long)(long long)(NFULA_IFINDEX_PHYSINDEV));
#endif
#ifdef NFULA_IFINDEX_PHYSOUTDEV
printf("NFULA_IFINDEX_PHYSOUTDEV %llu\n", (unsigned long long)(long long)(NFULA_IFINDEX_PHYSOUTDEV));
#endif
#ifdef NFULA_L2HDR
printf("NFULA_L2HDR %llu\n", (unsigned long long)(long long)(NFULA_L2HDR));
#endif
#ifdef NFULA_MARK
printf("NFULA_MARK %llu\n", (unsigned long long)(long long)(NFULA_MARK));
#endif
#ifdef NFULA_PACKET_HDR
printf("NFULA_PACKET_HDR %llu\n", (unsigned long long)(long long)(NFULA_PACKET_HDR));
#endif
#ifdef NFULA_PAYLOAD
printf("NFULA_PAYLOAD %llu\n", (unsigned long long)(long long)(NFULA_PAYLOAD));
#endif
#ifdef NFULA_PREFIX
printf("NFULA_PREFIX %llu\n", (unsigned long long)(long long)(NFULA_PREFIX));
#endif
#ifdef NFULA_SEQ
printf("NFULA_SEQ %llu\n", (unsigned long long)(long long)(NFULA_SEQ));
#endif
#ifdef NFULA_SEQ_GLOBAL
printf("NFULA_SEQ_GLOBAL %llu\n", (unsigned long long)(long long)(NFULA_SEQ_GLOBAL));
#endif
#ifdef NFULA_TIMESTAMP
printf("NFULA_TIMESTAMP %llu\n", (unsigned long long)(long long)(NFULA_TIMESTAMP));
#endif
#ifdef NFULA_UID
printf("NFULA_UID %llu\n", (unsigned long long)(long long)(NFULA_UID));
#endif
#ifdef NFULA_UNSPEC
printf("NFULA_UNSPEC %llu\n", (unsigned long long)(long long)(NFULA_UNSPEC));
#endif
#ifdef NFULA_VLAN
printf("NFULA_VLAN %llu\n", (unsigned long long)(long long)(NFULA_VLAN));
#endif
#ifdef NFULA_VLAN_PROTO
printf("NFULA_VLAN_PROTO %llu\n", (unsigned long long)(long long)(NFULA_VLAN_PROTO));
#endif
#ifdef NFULA_VLAN_TCI
printf("NFULA_VLAN_TCI %llu\n", (unsigned long long)(long long)(NFULA_VLAN_TCI));
#endif
#ifdef NFULA_VLAN_UNSPEC
printf("NFULA_VLAN_UNSPEC %llu\n", (unsigned long long)(long long)(NFULA_VLAN_UNSPEC));
#endif
#ifdef NFULNL_CFG_CMD_BIND
printf("NFULNL_CFG_CMD_BIND %llu\n", (unsigned long long)(long long)(NFULNL_CFG_CMD_BIND));
#endif
#ifdef NFULNL_CFG_CMD_NONE
printf("NFULNL_CFG_CMD_NONE %llu\n", (unsigned long long)(long long)(NFULNL_CFG_CMD_NONE));
#endif
#ifdef NFULNL_CFG_CMD_PF_BIND
printf("NFULNL_CFG_CMD_PF_BIND %llu\n", (unsigned long long)(long long)(NFULNL_CFG_CMD_PF_BIND));
#endif
#ifdef NFULNL_CFG_CMD_PF_UNBIND
printf("NFULNL_CFG_CMD_PF_UNBIND %llu\n", (unsigned long long)(long long)(NFULNL_CFG_CMD_PF_UNBIND));
#endif
#ifdef NFULNL_CFG_CMD_UNBIND
printf("NFULNL_CFG_CMD_UNBIND %llu\n", (unsigned long long)(long long)(NFULNL_CFG_CMD_UNBIND));
#endif
#ifdef NFULNL_CFG_F_CONNTRACK
printf("NFULNL_CFG_F_CONNTRACK %llu\n", (unsigned long long)(long long)(NFULNL_CFG_F_CONNTRACK));
#endif
#ifdef NFULNL_CFG_F_SEQ
printf("NFULNL_CFG_F_SEQ %llu\n", (unsigned long long)(long long)(NFULNL_CFG_F_SEQ));
#endif
#ifdef NFULNL_CFG_F_SEQ_GLOBAL
printf("NFULNL_CFG_F_SEQ_GLOBAL %llu\n", (unsigned long long)(long long)(NFULNL_CFG_F_SEQ_GLOBAL));
#endif
#ifdef NFULNL_COPY_META
printf("NFULNL_COPY_META %llu\n", (unsigned long long)(long long)(NFULNL_COPY_META));
#endif
#ifdef NFULNL_COPY_NONE
printf("NFULNL_COPY_NONE %llu\n", (unsigned long long)(long long)(NFULNL_COPY_NONE));
#endif
#ifdef NFULNL_COPY_PACKET
printf("NFULNL_COPY_PACKET %llu\n", (unsigned long long)(long long)(NFULNL_COPY_PACKET));
#endif
#ifdef NFULNL_MSG_CONFIG
printf("NFULNL_MSG_CONFIG %llu\n", (unsigned long long)(long long)(NFULNL_MSG_CONFIG));
#endif
#ifdef NFULNL_MSG_PACKET
printf("NFULNL_MSG_PACKET %llu\n", (unsigned long long)(long long)(NFULNL_MSG_PACKET));
#endif
#ifdef NF_ACCEPT
printf("NF_ACCEPT %llu\n", (unsigned long long)(long long)(NF_ACCEPT));
#endif
#ifdef NF_ARP
printf("NF_ARP %llu\n", (unsigned long long)(long long)(NF_ARP));
#endif
#ifdef NF_ARP_FORWARD
printf("NF_ARP_FORWARD %llu\n", (unsigned long long)(long long)(NF_ARP_FORWARD));
#endif
#ifdef NF_ARP_IN
printf("NF_ARP_IN %llu\n", (unsigned long long)(long long)(NF_ARP_IN));
#endif
#ifdef NF_ARP_NUMHOOKS
printf("NF_ARP_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_ARP_NUMHOOKS));
#endif
#ifdef NF_ARP_OUT
printf("NF_ARP_OUT %llu\n", (unsigned long long)(long long)(NF_ARP_OUT));
#endif
#ifdef NF_BR_BROUTING
printf("NF_BR_BROUTING %llu\n", (unsigned long long)(long long)(NF_BR_BROUTING));
#endif
#ifdef NF_BR_FORWARD
printf("NF_BR_FORWARD %llu\n", (unsigned long long)(long long)(NF_BR_FORWARD));
#endif
#ifdef NF_BR_LOCAL_IN
printf("NF_BR_LOCAL_IN %llu\n", (unsigned long long)(long long)(NF_BR_LOCAL_IN));
#endif
#ifdef NF_BR_LOCAL_OUT
printf("NF_BR_LOCAL_OUT %llu\n", (unsigned long long)(long long)(NF_BR_LOCAL_OUT));
#endif
#ifdef NF_BR_NUMHOOKS
printf("NF_BR_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_BR_NUMHOOKS));
#endif
#ifdef NF_BR_POST_ROUTING
printf("NF_BR_POST_ROUTING %llu\n", (unsigned long long)(long long)(NF_BR_POST_ROUTING));
#endif
#ifdef NF_BR_PRE_ROUTING
printf("NF_BR_PRE_ROUTING %llu\n", (unsigned long long)(long long)(NF_BR_PRE_ROUTING));
#endif
#ifdef NF_BR_PRI_BRNF
printf("NF_BR_PRI_BRNF %llu\n", (unsigned long long)(long long)(NF_BR_PRI_BRNF));
#endif
#ifdef NF_BR_PRI_FILTER_BRIDGED
printf("NF_BR_PRI_FILTER_BRIDGED %llu\n", (unsigned long long)(long long)(NF_BR_PRI_FILTER_BRIDGED));
#endif
#ifdef NF_BR_PRI_FILTER_OTHER
printf("NF_BR_PRI_FILTER_OTHER %llu\n", (unsigned long long)(long long)(NF_BR_PRI_FILTER_OTHER));
#endif
#ifdef NF_BR_PRI_FIRST
printf("NF_BR_PRI_FIRST %llu\n", (unsigned long long)(long long)(NF_BR_PRI_FIRST));
#endif
#ifdef NF_BR_PRI_LAST
printf("NF_BR_PRI_LAST %llu\n", (unsigned long long)(long long)(NF_BR_PRI_LAST));
#endif
#ifdef NF_BR_PRI_NAT_DST_BRIDGED
printf("NF_BR_PRI_NAT_DST_BRIDGED %llu\n", (unsigned long long)(long long)(NF_BR_PRI_NAT_DST_BRIDGED));
#endif
#ifdef NF_BR_PRI_NAT_DST_OTHER
printf("NF_BR_PRI_NAT_DST_OTHER %llu\n", (unsigned long long)(long long)(NF_BR_PRI_NAT_DST_OTHER));
#endif
#ifdef NF_BR_PRI_NAT_SRC
printf("NF_BR_PRI_NAT_SRC %llu\n", (unsigned long long)(long long)(NF_BR_PRI_NAT_SRC));
#endif
#ifdef NF_DROP
printf("NF_DROP %llu\n", (unsigned long long)(long long)(NF_DROP));
#endif
#ifdef NF_INET_FORWARD
printf("NF_INET_FORWARD %llu\n", (unsigned long long)(long long)(NF_INET_FORWARD));
#endif
#ifdef NF_INET_INGRESS
printf("NF_INET_INGRESS %llu\n", (unsigned long long)(long long)(NF_INET_INGRESS));
#endif
#ifdef NF_INET_LOCAL_IN
printf("NF_INET_LOCAL_IN %llu\n", (unsigned long long)(long long)(NF_INET_LOCAL_IN));
#endif
#ifdef NF_INET_LOCAL_OUT
printf("NF_INET_LOCAL_OUT %llu\n", (unsigned long long)(long long)(NF_INET_LOCAL_OUT));
#endif
#ifdef NF_INET_NUMHOOKS
printf("NF_INET_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_INET_NUMHOOKS));
#endif
#ifdef NF_INET_POST_ROUTING
printf("NF_INET_POST_ROUTING %llu\n", (unsigned long long)(long long)(NF_INET_POST_ROUTING));
#endif
#ifdef NF_INET_PRE_ROUTING
printf("NF_INET_PRE_ROUTING %llu\n", (unsigned long long)(long long)(NF_INET_PRE_ROUTING));
#endif
#ifdef NF_IP6_FORWARD
printf("NF_IP6_FORWARD %llu\n", (unsigned long long)(long long)(NF_IP6_FORWARD));
#endif
#ifdef NF_IP6_LOCAL_IN
printf("NF_IP6_LOCAL_IN %llu\n", (unsigned long long)(long long)(NF_IP6_LOCAL_IN));
#endif
#ifdef NF_IP6_LOCAL_OUT
printf("NF_IP6_LOCAL_OUT %llu\n", (unsigned long long)(long long)(NF_IP6_LOCAL_OUT));
#endif
#ifdef NF_IP6_NUMHOOKS
printf("NF_IP6_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_IP6_NUMHOOKS));
#endif
#ifdef NF_IP6_POST_ROUTING
printf("NF_IP6_POST_ROUTING %llu\n", (unsigned long long)(long long)(NF_IP6_POST_ROUTING));
#endif
#ifdef NF_IP6_PRE_ROUTING
printf("NF_IP6_PRE_ROUTING %llu\n", (unsigned long long)(long long)(NF_IP6_PRE_ROUTING));
#endif
#ifdef NF_IP6_PRI_CONNTRACK
printf("NF_IP6_PRI_CONNTRACK %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_CONNTRACK));
#endif
#ifdef NF_IP6_PRI_CONNTRACK_DEFRAG
printf("NF_IP6_PRI_CONNTRACK_DEFRAG %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_CONNTRACK_DEFRAG));
#endif
#ifdef NF_IP6_PRI_CONNTRACK_HELPER
printf("NF_IP6_PRI_CONNTRACK_HELPER %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_CONNTRACK_HELPER));
#endif
#ifdef NF_IP6_PRI_FILTER
printf("NF_IP6_PRI_FILTER %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_FILTER));
#endif
#ifdef NF_IP6_PRI_FIRST
printf("NF_IP6_PRI_FIRST %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_FIRST));
#endif
#ifdef NF_IP6_PRI_LAST
printf("NF_IP6_PRI_LAST %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_LAST));
#endif
#ifdef NF_IP6_PRI_MANGLE
printf("NF_IP6_PRI_MANGLE %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_MANGLE));
#endif
#ifdef NF_IP6_PRI_NAT_DST
printf("NF_IP6_PRI_NAT_DST %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_NAT_DST));
#endif
#ifdef NF_IP6_PRI_NAT_SRC
printf("NF_IP6_PRI_NAT_SRC %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_NAT_SRC));
#endif
#ifdef NF_IP6_PRI_RAW
printf("NF_IP6_PRI_RAW %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_RAW));
#endif
#ifdef NF_IP6_PRI_RAW_BEFORE_DEFRAG
printf("NF_IP6_PRI_RAW_BEFORE_DEFRAG %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_RAW_BEFORE_DEFRAG));
#endif
#ifdef NF_IP6_PRI_SECURITY
printf("NF_IP6_PRI_SECURITY %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_SECURITY));
#endif
#ifdef NF_IP6_PRI_SELINUX_FIRST
printf("NF_IP6_PRI_SELINUX_FIRST %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_SELINUX_FIRST));
#endif
#ifdef NF_IP6_PRI_SELINUX_LAST
printf("NF_IP6_PRI_SELINUX_LAST %llu\n", (unsigned long long)(long long)(NF_IP6_PRI_SELINUX_LAST));
#endif
#ifdef NF_IP_FORWARD
printf("NF_IP_FORWARD %llu\n", (unsigned long long)(long long)(NF_IP_FORWARD));
#endif
#ifdef NF_IP_LOCAL_IN
printf("NF_IP_LOCAL_IN %llu\n", (unsigned long long)(long long)(NF_IP_LOCAL_IN));
#endif
#ifdef NF_IP_LOCAL_OUT
printf("NF_IP_LOCAL_OUT %llu\n", (unsigned long long)(long long)(NF_IP_LOCAL_OUT));
#endif
#ifdef NF_IP_NUMHOOKS
printf("NF_IP_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_IP_NUMHOOKS));
#endif
#ifdef NF_IP_POST_ROUTING
printf("NF_IP_POST_ROUTING %llu\n", (unsigned long long)(long long)(NF_IP_POST_ROUTING));
#endif
#ifdef NF_IP_PRE_ROUTING
printf("NF_IP_PRE_ROUTING %llu\n", (unsigned long long)(long long)(NF_IP_PRE_ROUTING));
#endif
#ifdef NF_IP_PRI_CONNTRACK
printf("NF_IP_PRI_CONNTRACK %llu\n", (unsigned long long)(long long)(NF_IP_PRI_CONNTRACK));
#endif
#ifdef NF_IP_PRI_CONNTRACK_CONFIRM
printf("NF_IP_PRI_CONNTRACK_CONFIRM %llu\n", (unsigned long long)(long long)(NF_IP_PRI_CONNTRACK_CONFIRM));
#endif
#ifdef NF_IP_PRI_CONNTRACK_DEFRAG
printf("NF_IP_PRI_CONNTRACK_DEFRAG %llu\n", (unsigned long long)(long long)(NF_IP_PRI_CONNTRACK_DEFRAG));
#endif
#ifdef NF_IP_PRI_CONNTRACK_HELPER
printf("NF_IP_PRI_CONNTRACK_HELPER %llu\n", (unsigned long long)(long long)(NF_IP_PRI_CONNTRACK_HELPER));
#endif
#ifdef NF_IP_PRI_FILTER
printf("NF_IP_PRI_FILTER %llu\n", (unsigned long long)(long long)(NF_IP_PRI_FILTER));
#endif
#ifdef NF_IP_PRI_FIRST
printf("NF_IP_PRI_FIRST %llu\n", (unsigned long long)(long long)(NF_IP_PRI_FIRST));
#endif
#ifdef NF_IP_PRI_LAST
printf("NF_IP_PRI_LAST %llu\n", (unsigned long long)(long long)(NF_IP_PRI_LAST));
#endif
#ifdef NF_IP_PRI_MANGLE
printf("NF_IP_PRI_MANGLE %llu\n", (unsigned long long)(long long)(NF_IP_PRI_MANGLE));
#endif
#ifdef NF_IP_PRI_NAT_DST
printf("NF_IP_PRI_NAT_DST %llu\n", (unsigned long long)(long long)(NF_IP_PRI_NAT_DST));
#endif
#ifdef NF_IP_PRI_NAT_SRC
printf("NF_IP_PRI_NAT_SRC %llu\n", (unsigned long long)(long long)(NF_IP_PRI_NAT_SRC));
#endif
#ifdef NF_IP_PRI_RAW
printf("NF_IP_PRI_RAW %llu\n", (unsigned long long)(long long)(NF_IP_PRI_RAW));
#endif
#ifdef NF_IP_PRI_RAW_BEFORE_DEFRAG
printf("NF_IP_PRI_RAW_BEFORE_DEFRAG %llu\n", (unsigned long long)(long long)(NF_IP_PRI_RAW_BEFORE_DEFRAG));
#endif
#ifdef NF_IP_PRI_SECURITY
printf("NF_IP_PRI_SECURITY %llu\n", (unsigned long long)(long long)(NF_IP_PRI_SECURITY));
#endif
#ifdef NF_IP_PRI_SELINUX_FIRST
printf("NF_IP_PRI_SELINUX_FIRST %llu\n", (unsigned long long)(long long)(NF_IP_PRI_SELINUX_FIRST));
#endif
#ifdef NF_IP_PRI_SELINUX_LAST
printf("NF_IP_PRI_SELINUX_LAST %llu\n", (unsigned long long)(long long)(NF_IP_PRI_SELINUX_LAST));
#endif
#ifdef NF_MAX_VERDICT
printf("NF_MAX_VERDICT %llu\n", (unsigned long long)(long long)(NF_MAX_VERDICT));
#endif
#ifdef NF_NETDEV_EGRESS
printf("NF_NETDEV_EGRESS %llu\n", (unsigned long long)(long long)(NF_NETDEV_EGRESS));
#endif
#ifdef NF_NETDEV_INGRESS
printf("NF_NETDEV_INGRESS %llu\n", (unsigned long long)(long long)(NF_NETDEV_INGRESS));
#endif
#ifdef NF_NETDEV_NUMHOOKS
printf("NF_NETDEV_NUMHOOKS %llu\n", (unsigned long long)(long long)(NF_NETDEV_NUMHOOKS));
#endif
#ifdef NF_QUEUE
printf("NF_QUEUE %llu\n", (unsigned long long)(long long)(NF_QUEUE));
#endif
#ifdef NF_REPEAT
printf("NF_REPEAT %llu\n", (unsigned long long)(long long)(NF_REPEAT));
#endif
#ifdef NF_STOLEN
printf("NF_STOLEN %llu\n", (unsigned long long)(long long)(NF_STOLEN));
#endif
#ifdef NF_STOP
printf("NF_STOP %llu\n", (unsigned long long)(long long)(NF_STOP));
#endif
#ifdef NF_VERDICT_BITS
printf("NF_VERDICT_BITS %llu\n", (unsigned long long)(long long)(NF_VERDICT_BITS));
#endif
#ifdef NF_VERDICT_FLAG_QUEUE_BYPASS
printf("NF_VERDICT_FLAG_QUEUE_BYPASS %llu\n", (unsigned long long)(long long)(NF_VERDICT_FLAG_QUEUE_BYPASS));
#endif
#ifdef NF_VERDICT_MASK
printf("NF_VERDICT_MASK %llu\n", (unsigned long long)(long long)(NF_VERDICT_MASK));
#endif
#ifdef NF_VERDICT_QBITS
printf("NF_VERDICT_QBITS %llu\n", (unsigned long long)(long long)(NF_VERDICT_QBITS));
#endif
#ifdef NF_VERDICT_QMASK
printf("NF_VERDICT_QMASK %llu\n", (unsigned long long)(long long)(NF_VERDICT_QMASK));
#endif
#ifdef NGREG
printf("NGREG %llu\n", (unsigned long long)(long long)(NGREG));
#endif
#ifdef NILFS_SUPER_MAGIC
printf("NILFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(NILFS_SUPER_MAGIC));
#endif
#ifdef NI_DGRAM
printf("NI_DGRAM %llu\n", (unsigned long long)(long long)(NI_DGRAM));
#endif
#ifdef NI_IDN
printf("NI_IDN %llu\n", (unsigned long long)(long long)(NI_IDN));
#endif
#ifdef NI_MAXHOST
printf("NI_MAXHOST %llu\n", (unsigned long long)(long long)(NI_MAXHOST));
#endif
#ifdef NI_NAMEREQD
printf("NI_NAMEREQD %llu\n", (unsigned long long)(long long)(NI_NAMEREQD));
#endif
#ifdef NI_NOFQDN
printf("NI_NOFQDN %llu\n", (unsigned long long)(long long)(NI_NOFQDN));
#endif
#ifdef NI_NUMERICHOST
printf("NI_NUMERICHOST %llu\n", (unsigned long long)(long long)(NI_NUMERICHOST));
#endif
#ifdef NI_NUMERICSERV
printf("NI_NUMERICSERV %llu\n", (unsigned long long)(long long)(NI_NUMERICSERV));
#endif
#ifdef NL0
printf("NL0 %llu\n", (unsigned long long)(long long)(NL0));
#endif
#ifdef NL1
printf("NL1 %llu\n", (unsigned long long)(long long)(NL1));
#endif
#ifdef NLA_ALIGNTO
printf("NLA_ALIGNTO %llu\n", (unsigned long long)(long long)(NLA_ALIGNTO));
#endif
#ifdef NLA_F_NESTED
printf("NLA_F_NESTED %llu\n", (unsigned long long)(long long)(NLA_F_NESTED));
#endif
#ifdef NLA_F_NET_BYTEORDER
printf("NLA_F_NET_BYTEORDER %llu\n", (unsigned long long)(long long)(NLA_F_NET_BYTEORDER));
#endif
#ifdef NLA_TYPE_MASK
printf("NLA_TYPE_MASK %llu\n", (unsigned long long)(long long)(NLA_TYPE_MASK));
#endif
#ifdef NLDLY
printf("NLDLY %llu\n", (unsigned long long)(long long)(NLDLY));
#endif
#ifdef NLMSG_DONE
printf("NLMSG_DONE %llu\n", (unsigned long long)(long long)(NLMSG_DONE));
#endif
#ifdef NLMSG_ERROR
printf("NLMSG_ERROR %llu\n", (unsigned long long)(long long)(NLMSG_ERROR));
#endif
#ifdef NLMSG_MIN_TYPE
printf("NLMSG_MIN_TYPE %llu\n", (unsigned long long)(long long)(NLMSG_MIN_TYPE));
#endif
#ifdef NLMSG_NOOP
printf("NLMSG_NOOP %llu\n", (unsigned long long)(long long)(NLMSG_NOOP));
#endif
#ifdef NLMSG_OVERRUN
printf("NLMSG_OVERRUN %llu\n", (unsigned long long)(long long)(NLMSG_OVERRUN));
#endif
#ifdef NLM_F_ACK
printf("NLM_F_ACK %llu\n", (unsigned long long)(long long)(NLM_F_ACK));
#endif
#ifdef NLM_F_ACK_TLVS
printf("NLM_F_ACK_TLVS %llu\n", (unsigned long long)(long long)(NLM_F_ACK_TLVS));
#endif
#ifdef NLM_F_APPEND
printf("NLM_F_APPEND %llu\n", (unsigned long long)(long long)(NLM_F_APPEND));
#endif
#ifdef NLM_F_ATOMIC
printf("NLM_F_ATOMIC %llu\n", (unsigned long long)(long long)(NLM_F_ATOMIC));
#endif
#ifdef NLM_F_BULK
printf("NLM_F_BULK %llu\n", (unsigned long long)(long long)(NLM_F_BULK));
#endif
#ifdef NLM_F_CAPPED
printf("NLM_F_CAPPED %llu\n", (unsigned long long)(long long)(NLM_F_CAPPED));
#endif
#ifdef NLM_F_CREATE
printf("NLM_F_CREATE %llu\n", (unsigned long long)(long long)(NLM_F_CREATE));
#endif
#ifdef NLM_F_DUMP
printf("NLM_F_DUMP %llu\n", (unsigned long long)(long long)(NLM_F_DUMP));
#endif
#ifdef NLM_F_DUMP_FILTERED
printf("NLM_F_DUMP_FILTERED %llu\n", (unsigned long long)(long long)(NLM_F_DUMP_FILTERED));
#endif
#ifdef NLM_F_DUMP_INTR
printf("NLM_F_DUMP_INTR %llu\n", (unsigned long long)(long long)(NLM_F_DUMP_INTR));
#endif
#ifdef NLM_F_ECHO
printf("NLM_F_ECHO %llu\n", (unsigned long long)(long long)(NLM_F_ECHO));
#endif
#ifdef NLM_F_EXCL
printf("NLM_F_EXCL %llu\n", (unsigned long long)(long long)(NLM_F_EXCL));
#endif
#ifdef NLM_F_MATCH
printf("NLM_F_MATCH %llu\n", (unsigned long long)(long long)(NLM_F_MATCH));
#endif
#ifdef NLM_F_MULTI
printf("NLM_F_MULTI %llu\n", (unsigned long long)(long long)(NLM_F_MULTI));
#endif
#ifdef NLM_F_NONREC
printf("NLM_F_NONREC %llu\n", (unsigned long long)(long long)(NLM_F_NONREC));
#endif
#ifdef NLM_F_REPLACE
printf("NLM_F_REPLACE %llu\n", (unsigned long long)(long long)(NLM_F_REPLACE));
#endif
#ifdef NLM_F_REQUEST
printf("NLM_F_REQUEST %llu\n", (unsigned long long)(long long)(NLM_F_REQUEST));
#endif
#ifdef NLM_F_ROOT
printf("NLM_F_ROOT %llu\n", (unsigned long long)(long long)(NLM_F_ROOT));
#endif
#ifdef NOEXPR
printf("NOEXPR %llu\n", (unsigned long long)(long long)(NOEXPR));
#endif
#ifdef NOFLSH
printf("NOFLSH %llu\n", (unsigned long long)(long long)(NOFLSH));
#endif
#ifdef NOSTR
printf("NOSTR %llu\n", (unsigned long long)(long long)(NOSTR));
#endif
#ifdef NSFS_MAGIC
printf("NSFS_MAGIC %llu\n", (unsigned long long)(long long)(NSFS_MAGIC));
#endif
#ifdef NS_GET_MNTNS_ID
printf("NS_GET_MNTNS_ID %llu\n", (unsigned long long)(long long)(NS_GET_MNTNS_ID));
#endif
#ifdef NS_GET_NSTYPE
printf("NS_GET_NSTYPE %llu\n", (unsigned long long)(long long)(NS_GET_NSTYPE));
#endif
#ifdef NS_GET_OWNER_UID
printf("NS_GET_OWNER_UID %llu\n", (unsigned long long)(long long)(NS_GET_OWNER_UID));
#endif
#ifdef NS_GET_PARENT
printf("NS_GET_PARENT %llu\n", (unsigned long long)(long long)(NS_GET_PARENT));
#endif
#ifdef NS_GET_PID_FROM_PIDNS
printf("NS_GET_PID_FROM_PIDNS %llu\n", (unsigned long long)(long long)(NS_GET_PID_FROM_PIDNS));
#endif
#ifdef NS_GET_PID_IN_PIDNS
printf("NS_GET_PID_IN_PIDNS %llu\n", (unsigned long long)(long long)(NS_GET_PID_IN_PIDNS));
#endif
#ifdef NS_GET_TGID_FROM_PIDNS
printf("NS_GET_TGID_FROM_PIDNS %llu\n", (unsigned long long)(long long)(NS_GET_TGID_FROM_PIDNS));
#endif
#ifdef NS_GET_TGID_IN_PIDNS
printf("NS_GET_TGID_IN_PIDNS %llu\n", (unsigned long long)(long long)(NS_GET_TGID_IN_PIDNS));
#endif
#ifdef NS_GET_USERNS
printf("NS_GET_USERNS %llu\n", (unsigned long long)(long long)(NS_GET_USERNS));
#endif
#ifdef NS_MNT_GET_INFO
printf("NS_MNT_GET_INFO %llu\n", (unsigned long long)(long long)(NS_MNT_GET_INFO));
#endif
#ifdef NS_MNT_GET_NEXT
printf("NS_MNT_GET_NEXT %llu\n", (unsigned long long)(long long)(NS_MNT_GET_NEXT));
#endif
#ifdef NS_MNT_GET_PREV
printf("NS_MNT_GET_PREV %llu\n", (unsigned long long)(long long)(NS_MNT_GET_PREV));
#endif
#ifdef NTF_EXT_LEARNED
printf("NTF_EXT_LEARNED %llu\n", (unsigned long long)(long long)(NTF_EXT_LEARNED));
#endif
#ifdef NTF_MASTER
printf("NTF_MASTER %llu\n", (unsigned long long)(long long)(NTF_MASTER));
#endif
#ifdef NTF_OFFLOADED
printf("NTF_OFFLOADED %llu\n", (unsigned long long)(long long)(NTF_OFFLOADED));
#endif
#ifdef NTF_PROXY
printf("NTF_PROXY %llu\n", (unsigned long long)(long long)(NTF_PROXY));
#endif
#ifdef NTF_ROUTER
printf("NTF_ROUTER %llu\n", (unsigned long long)(long long)(NTF_ROUTER));
#endif
#ifdef NTF_SELF
printf("NTF_SELF %llu\n", (unsigned long long)(long long)(NTF_SELF));
#endif
#ifdef NTF_USE
printf("NTF_USE %llu\n", (unsigned long long)(long long)(NTF_USE));
#endif
#ifdef NTP_API
printf("NTP_API %llu\n", (unsigned long long)(long long)(NTP_API));
#endif
#ifdef NT_ASRS
printf("NT_ASRS %llu\n", (unsigned long long)(long long)(NT_ASRS));
#endif
#ifdef NT_AUXV
printf("NT_AUXV %llu\n", (unsigned long long)(long long)(NT_AUXV));
#endif
#ifdef NT_FPREGSET
printf("NT_FPREGSET %llu\n", (unsigned long long)(long long)(NT_FPREGSET));
#endif
#ifdef NT_GWINDOWS
printf("NT_GWINDOWS %llu\n", (unsigned long long)(long long)(NT_GWINDOWS));
#endif
#ifdef NT_LWPSINFO
printf("NT_LWPSINFO %llu\n", (unsigned long long)(long long)(NT_LWPSINFO));
#endif
#ifdef NT_LWPSTATUS
printf("NT_LWPSTATUS %llu\n", (unsigned long long)(long long)(NT_LWPSTATUS));
#endif
#ifdef NT_PLATFORM
printf("NT_PLATFORM %llu\n", (unsigned long long)(long long)(NT_PLATFORM));
#endif
#ifdef NT_PRCRED
printf("NT_PRCRED %llu\n", (unsigned long long)(long long)(NT_PRCRED));
#endif
#ifdef NT_PRFPREG
printf("NT_PRFPREG %llu\n", (unsigned long long)(long long)(NT_PRFPREG));
#endif
#ifdef NT_PRFPXREG
printf("NT_PRFPXREG %llu\n", (unsigned long long)(long long)(NT_PRFPXREG));
#endif
#ifdef NT_PRPSINFO
printf("NT_PRPSINFO %llu\n", (unsigned long long)(long long)(NT_PRPSINFO));
#endif
#ifdef NT_PRSTATUS
printf("NT_PRSTATUS %llu\n", (unsigned long long)(long long)(NT_PRSTATUS));
#endif
#ifdef NT_PRXREG
printf("NT_PRXREG %llu\n", (unsigned long long)(long long)(NT_PRXREG));
#endif
#ifdef NT_PSINFO
printf("NT_PSINFO %llu\n", (unsigned long long)(long long)(NT_PSINFO));
#endif
#ifdef NT_PSTATUS
printf("NT_PSTATUS %llu\n", (unsigned long long)(long long)(NT_PSTATUS));
#endif
#ifdef NT_TASKSTRUCT
printf("NT_TASKSTRUCT %llu\n", (unsigned long long)(long long)(NT_TASKSTRUCT));
#endif
#ifdef NT_UTSNAME
printf("NT_UTSNAME %llu\n", (unsigned long long)(long long)(NT_UTSNAME));
#endif
#ifdef NUD_DELAY
printf("NUD_DELAY %llu\n", (unsigned long long)(long long)(NUD_DELAY));
#endif
#ifdef NUD_FAILED
printf("NUD_FAILED %llu\n", (unsigned long long)(long long)(NUD_FAILED));
#endif
#ifdef NUD_INCOMPLETE
printf("NUD_INCOMPLETE %llu\n", (unsigned long long)(long long)(NUD_INCOMPLETE));
#endif
#ifdef NUD_NOARP
printf("NUD_NOARP %llu\n", (unsigned long long)(long long)(NUD_NOARP));
#endif
#ifdef NUD_NONE
printf("NUD_NONE %llu\n", (unsigned long long)(long long)(NUD_NONE));
#endif
#ifdef NUD_PERMANENT
printf("NUD_PERMANENT %llu\n", (unsigned long long)(long long)(NUD_PERMANENT));
#endif
#ifdef NUD_PROBE
printf("NUD_PROBE %llu\n", (unsigned long long)(long long)(NUD_PROBE));
#endif
#ifdef NUD_REACHABLE
printf("NUD_REACHABLE %llu\n", (unsigned long long)(long long)(NUD_REACHABLE));
#endif
#ifdef NUD_STALE
printf("NUD_STALE %llu\n", (unsigned long long)(long long)(NUD_STALE));
#endif
#ifdef OCFS2_SUPER_MAGIC
printf("OCFS2_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(OCFS2_SUPER_MAGIC));
#endif
#ifdef OCRNL
printf("OCRNL %llu\n", (unsigned long long)(long long)(OCRNL));
#endif
#ifdef OFDEL
printf("OFDEL %llu\n", (unsigned long long)(long long)(OFDEL));
#endif
#ifdef OFILL
printf("OFILL %llu\n", (unsigned long long)(long long)(OFILL));
#endif
#ifdef OLCUC
printf("OLCUC %llu\n", (unsigned long long)(long long)(OLCUC));
#endif
#ifdef OLD_TIME
printf("OLD_TIME %llu\n", (unsigned long long)(long long)(OLD_TIME));
#endif
#ifdef ONLCR
printf("ONLCR %llu\n", (unsigned long long)(long long)(ONLCR));
#endif
#ifdef ONLRET
printf("ONLRET %llu\n", (unsigned long long)(long long)(ONLRET));
#endif
#ifdef ONOCR
printf("ONOCR %llu\n", (unsigned long long)(long long)(ONOCR));
#endif
#ifdef OPENPROM_SUPER_MAGIC
printf("OPENPROM_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(OPENPROM_SUPER_MAGIC));
#endif
#ifdef OPEN_TREE_CLOEXEC
printf("OPEN_TREE_CLOEXEC %llu\n", (unsigned long long)(long long)(OPEN_TREE_CLOEXEC));
#endif
#ifdef OPEN_TREE_CLONE
printf("OPEN_TREE_CLONE %llu\n", (unsigned long long)(long long)(OPEN_TREE_CLONE));
#endif
#ifdef OPOST
printf("OPOST %llu\n", (unsigned long long)(long long)(OPOST));
#endif
#ifdef OVERLAYFS_SUPER_MAGIC
printf("OVERLAYFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(OVERLAYFS_SUPER_MAGIC));
#endif
#ifdef O_ACCMODE
printf("O_ACCMODE %llu\n", (unsigned long long)(long long)(O_ACCMODE));
#endif
#ifdef O_APPEND
printf("O_APPEND %llu\n", (unsigned long long)(long long)(O_APPEND));
#endif
#ifdef O_ASYNC
printf("O_ASYNC %llu\n", (unsigned long long)(long long)(O_ASYNC));
#endif
#ifdef O_CLOEXEC
printf("O_CLOEXEC %llu\n", (unsigned long long)(long long)(O_CLOEXEC));
#endif
#ifdef O_CREAT
printf("O_CREAT %llu\n", (unsigned long long)(long long)(O_CREAT));
#endif
#ifdef O_DIRECT
printf("O_DIRECT %llu\n", (unsigned long long)(long long)(O_DIRECT));
#endif
#ifdef O_DIRECTORY
printf("O_DIRECTORY %llu\n", (unsigned long long)(long long)(O_DIRECTORY));
#endif
#ifdef O_DSYNC
printf("O_DSYNC %llu\n", (unsigned long long)(long long)(O_DSYNC));
#endif
#ifdef O_EXCL
printf("O_EXCL %llu\n", (unsigned long long)(long long)(O_EXCL));
#endif
#ifdef O_FSYNC
printf("O_FSYNC %llu\n", (unsigned long long)(long long)(O_FSYNC));
#endif
#ifdef O_LARGEFILE
printf("O_LARGEFILE %llu\n", (unsigned long long)(long long)(O_LARGEFILE));
#endif
#ifdef O_NDELAY
printf("O_NDELAY %llu\n", (unsigned long long)(long long)(O_NDELAY));
#endif
#ifdef O_NOATIME
printf("O_NOATIME %llu\n", (unsigned long long)(long long)(O_NOATIME));
#endif
#ifdef O_NOCTTY
printf("O_NOCTTY %llu\n", (unsigned long long)(long long)(O_NOCTTY));
#endif
#ifdef O_NOFOLLOW
printf("O_NOFOLLOW %llu\n", (unsigned long long)(long long)(O_NOFOLLOW));
#endif
#ifdef O_NONBLOCK
printf("O_NONBLOCK %llu\n", (unsigned long long)(long long)(O_NONBLOCK));
#endif
#ifdef O_PATH
printf("O_PATH %llu\n", (unsigned long long)(long long)(O_PATH));
#endif
#ifdef O_RDONLY
printf("O_RDONLY %llu\n", (unsigned long long)(long long)(O_RDONLY));
#endif
#ifdef O_RDWR
printf("O_RDWR %llu\n", (unsigned long long)(long long)(O_RDWR));
#endif
#ifdef O_RSYNC
printf("O_RSYNC %llu\n", (unsigned long long)(long long)(O_RSYNC));
#endif
#ifdef O_SYNC
printf("O_SYNC %llu\n", (unsigned long long)(long long)(O_SYNC));
#endif
#ifdef O_TMPFILE
printf("O_TMPFILE %llu\n", (unsigned long long)(long long)(O_TMPFILE));
#endif
#ifdef O_TRUNC
printf("O_TRUNC %llu\n", (unsigned long long)(long long)(O_TRUNC));
#endif
#ifdef O_WRONLY
printf("O_WRONLY %llu\n", (unsigned long long)(long long)(O_WRONLY));
#endif
#ifdef PACKET_ADD_MEMBERSHIP
printf("PACKET_ADD_MEMBERSHIP %llu\n", (unsigned long long)(long long)(PACKET_ADD_MEMBERSHIP));
#endif
#ifdef PACKET_AUXDATA
printf("PACKET_AUXDATA %llu\n", (unsigned long long)(long long)(PACKET_AUXDATA));
#endif
#ifdef PACKET_BROADCAST
printf("PACKET_BROADCAST %llu\n", (unsigned long long)(long long)(PACKET_BROADCAST));
#endif
#ifdef PACKET_COPY_THRESH
printf("PACKET_COPY_THRESH %llu\n", (unsigned long long)(long long)(PACKET_COPY_THRESH));
#endif
#ifdef PACKET_DROP_MEMBERSHIP
printf("PACKET_DROP_MEMBERSHIP %llu\n", (unsigned long long)(long long)(PACKET_DROP_MEMBERSHIP));
#endif
#ifdef PACKET_FANOUT
printf("PACKET_FANOUT %llu\n", (unsigned long long)(long long)(PACKET_FANOUT));
#endif
#ifdef PACKET_FANOUT_CBPF
printf("PACKET_FANOUT_CBPF %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_CBPF));
#endif
#ifdef PACKET_FANOUT_CPU
printf("PACKET_FANOUT_CPU %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_CPU));
#endif
#ifdef PACKET_FANOUT_DATA
printf("PACKET_FANOUT_DATA %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_DATA));
#endif
#ifdef PACKET_FANOUT_EBPF
printf("PACKET_FANOUT_EBPF %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_EBPF));
#endif
#ifdef PACKET_FANOUT_FLAG_DEFRAG
printf("PACKET_FANOUT_FLAG_DEFRAG %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_FLAG_DEFRAG));
#endif
#ifdef PACKET_FANOUT_FLAG_IGNORE_OUTGOING
printf("PACKET_FANOUT_FLAG_IGNORE_OUTGOING %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_FLAG_IGNORE_OUTGOING));
#endif
#ifdef PACKET_FANOUT_FLAG_ROLLOVER
printf("PACKET_FANOUT_FLAG_ROLLOVER %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_FLAG_ROLLOVER));
#endif
#ifdef PACKET_FANOUT_FLAG_UNIQUEID
printf("PACKET_FANOUT_FLAG_UNIQUEID %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_FLAG_UNIQUEID));
#endif
#ifdef PACKET_FANOUT_HASH
printf("PACKET_FANOUT_HASH %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_HASH));
#endif
#ifdef PACKET_FANOUT_LB
printf("PACKET_FANOUT_LB %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_LB));
#endif
#ifdef PACKET_FANOUT_QM
printf("PACKET_FANOUT_QM %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_QM));
#endif
#ifdef PACKET_FANOUT_RND
printf("PACKET_FANOUT_RND %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_RND));
#endif
#ifdef PACKET_FANOUT_ROLLOVER
printf("PACKET_FANOUT_ROLLOVER %llu\n", (unsigned long long)(long long)(PACKET_FANOUT_ROLLOVER));
#endif
#ifdef PACKET_HDRLEN
printf("PACKET_HDRLEN %llu\n", (unsigned long long)(long long)(PACKET_HDRLEN));
#endif
#ifdef PACKET_HOST
printf("PACKET_HOST %llu\n", (unsigned long long)(long long)(PACKET_HOST));
#endif
#ifdef PACKET_IGNORE_OUTGOING
printf("PACKET_IGNORE_OUTGOING %llu\n", (unsigned long long)(long long)(PACKET_IGNORE_OUTGOING));
#endif
#ifdef PACKET_KERNEL
printf("PACKET_KERNEL %llu\n", (unsigned long long)(long long)(PACKET_KERNEL));
#endif
#ifdef PACKET_LOOPBACK
printf("PACKET_LOOPBACK %llu\n", (unsigned long long)(long long)(PACKET_LOOPBACK));
#endif
#ifdef PACKET_LOSS
printf("PACKET_LOSS %llu\n", (unsigned long long)(long long)(PACKET_LOSS));
#endif
#ifdef PACKET_MR_ALLMULTI
printf("PACKET_MR_ALLMULTI %llu\n", (unsigned long long)(long long)(PACKET_MR_ALLMULTI));
#endif
#ifdef PACKET_MR_MULTICAST
printf("PACKET_MR_MULTICAST %llu\n", (unsigned long long)(long long)(PACKET_MR_MULTICAST));
#endif
#ifdef PACKET_MR_PROMISC
printf("PACKET_MR_PROMISC %llu\n", (unsigned long long)(long long)(PACKET_MR_PROMISC));
#endif
#ifdef PACKET_MR_UNICAST
printf("PACKET_MR_UNICAST %llu\n", (unsigned long long)(long long)(PACKET_MR_UNICAST));
#endif
#ifdef PACKET_MULTICAST
printf("PACKET_MULTICAST %llu\n", (unsigned long long)(long long)(PACKET_MULTICAST));
#endif
#ifdef PACKET_ORIGDEV
printf("PACKET_ORIGDEV %llu\n", (unsigned long long)(long long)(PACKET_ORIGDEV));
#endif
#ifdef PACKET_OTHERHOST
printf("PACKET_OTHERHOST %llu\n", (unsigned long long)(long long)(PACKET_OTHERHOST));
#endif
#ifdef PACKET_OUTGOING
printf("PACKET_OUTGOING %llu\n", (unsigned long long)(long long)(PACKET_OUTGOING));
#endif
#ifdef PACKET_QDISC_BYPASS
printf("PACKET_QDISC_BYPASS %llu\n", (unsigned long long)(long long)(PACKET_QDISC_BYPASS));
#endif
#ifdef PACKET_RECV_OUTPUT
printf("PACKET_RECV_OUTPUT %llu\n", (unsigned long long)(long long)(PACKET_RECV_OUTPUT));
#endif
#ifdef PACKET_RESERVE
printf("PACKET_RESERVE %llu\n", (unsigned long long)(long long)(PACKET_RESERVE));
#endif
#ifdef PACKET_ROLLOVER_STATS
printf("PACKET_ROLLOVER_STATS %llu\n", (unsigned long long)(long long)(PACKET_ROLLOVER_STATS));
#endif
#ifdef PACKET_RX_RING
printf("PACKET_RX_RING %llu\n", (unsigned long long)(long long)(PACKET_RX_RING));
#endif
#ifdef PACKET_STATISTICS
printf("PACKET_STATISTICS %llu\n", (unsigned long long)(long long)(PACKET_STATISTICS));
#endif
#ifdef PACKET_TIMESTAMP
printf("PACKET_TIMESTAMP %llu\n", (unsigned long long)(long long)(PACKET_TIMESTAMP));
#endif
#ifdef PACKET_TX_HAS_OFF
printf("PACKET_TX_HAS_OFF %llu\n", (unsigned long long)(long long)(PACKET_TX_HAS_OFF));
#endif
#ifdef PACKET_TX_RING
printf("PACKET_TX_RING %llu\n", (unsigned long long)(long long)(PACKET_TX_RING));
#endif
#ifdef PACKET_TX_TIMESTAMP
printf("PACKET_TX_TIMESTAMP %llu\n", (unsigned long long)(long long)(PACKET_TX_TIMESTAMP));
#endif
#ifdef PACKET_USER
printf("PACKET_USER %llu\n", (unsigned long long)(long long)(PACKET_USER));
#endif
#ifdef PACKET_VERSION
printf("PACKET_VERSION %llu\n", (unsigned long long)(long long)(PACKET_VERSION));
#endif
#ifdef PACKET_VNET_HDR
printf("PACKET_VNET_HDR %llu\n", (unsigned long long)(long long)(PACKET_VNET_HDR));
#endif
#ifdef PACKET_VNET_HDR_SZ
printf("PACKET_VNET_HDR_SZ %llu\n", (unsigned long long)(long long)(PACKET_VNET_HDR_SZ));
#endif
#ifdef PARENB
printf("PARENB %llu\n", (unsigned long long)(long long)(PARENB));
#endif
#ifdef PARMRK
printf("PARMRK %llu\n", (unsigned long long)(long long)(PARMRK));
#endif
#ifdef PARODD
printf("PARODD %llu\n", (unsigned long long)(long long)(PARODD));
#endif
#ifdef PATH_MAX
printf("PATH_MAX %llu\n", (unsigned long long)(long long)(PATH_MAX));
#endif
#ifdef PENDIN
printf("PENDIN %llu\n", (unsigned long long)(long long)(PENDIN));
#endif
#ifdef PF_ALG
printf("PF_ALG %llu\n", (unsigned long long)(long long)(PF_ALG));
#endif
#ifdef PF_APPLETALK
printf("PF_APPLETALK %llu\n", (unsigned long long)(long long)(PF_APPLETALK));
#endif
#ifdef PF_ASH
printf("PF_ASH %llu\n", (unsigned long long)(long long)(PF_ASH));
#endif
#ifdef PF_ATMPVC
printf("PF_ATMPVC %llu\n", (unsigned long long)(long long)(PF_ATMPVC));
#endif
#ifdef PF_ATMSVC
printf("PF_ATMSVC %llu\n", (unsigned long long)(long long)(PF_ATMSVC));
#endif
#ifdef PF_AX25
printf("PF_AX25 %llu\n", (unsigned long long)(long long)(PF_AX25));
#endif
#ifdef PF_BLOCK_TS
printf("PF_BLOCK_TS %llu\n", (unsigned long long)(long long)(PF_BLOCK_TS));
#endif
#ifdef PF_BLUETOOTH
printf("PF_BLUETOOTH %llu\n", (unsigned long long)(long long)(PF_BLUETOOTH));
#endif
#ifdef PF_BRIDGE
printf("PF_BRIDGE %llu\n", (unsigned long long)(long long)(PF_BRIDGE));
#endif
#ifdef PF_CAIF
printf("PF_CAIF %llu\n", (unsigned long long)(long long)(PF_CAIF));
#endif
#ifdef PF_CAN
printf("PF_CAN %llu\n", (unsigned long long)(long long)(PF_CAN));
#endif
#ifdef PF_DECnet
printf("PF_DECnet %llu\n", (unsigned long long)(long long)(PF_DECnet));
#endif
#ifdef PF_DUMPCORE
printf("PF_DUMPCORE %llu\n", (unsigned long long)(long long)(PF_DUMPCORE));
#endif
#ifdef PF_ECONET
printf("PF_ECONET %llu\n", (unsigned long long)(long long)(PF_ECONET));
#endif
#ifdef PF_EXITING
printf("PF_EXITING %llu\n", (unsigned long long)(long long)(PF_EXITING));
#endif
#ifdef PF_FORKNOEXEC
printf("PF_FORKNOEXEC %llu\n", (unsigned long long)(long long)(PF_FORKNOEXEC));
#endif
#ifdef PF_IB
printf("PF_IB %llu\n", (unsigned long long)(long long)(PF_IB));
#endif
#ifdef PF_IDLE
printf("PF_IDLE %llu\n", (unsigned long long)(long long)(PF_IDLE));
#endif
#ifdef PF_IEEE802154
printf("PF_IEEE802154 %llu\n", (unsigned long long)(long long)(PF_IEEE802154));
#endif
#ifdef PF_INET
printf("PF_INET %llu\n", (unsigned long long)(long long)(PF_INET));
#endif
#ifdef PF_INET6
printf("PF_INET6 %llu\n", (unsigned long long)(long long)(PF_INET6));
#endif
#ifdef PF_IO_WORKER
printf("PF_IO_WORKER %llu\n", (unsigned long long)(long long)(PF_IO_WORKER));
#endif
#ifdef PF_IPX
printf("PF_IPX %llu\n", (unsigned long long)(long long)(PF_IPX));
#endif
#ifdef PF_IRDA
printf("PF_IRDA %llu\n", (unsigned long long)(long long)(PF_IRDA));
#endif
#ifdef PF_ISDN
printf("PF_ISDN %llu\n", (unsigned long long)(long long)(PF_ISDN));
#endif
#ifdef PF_IUCV
printf("PF_IUCV %llu\n", (unsigned long long)(long long)(PF_IUCV));
#endif
#ifdef PF_KEY
printf("PF_KEY %llu\n", (unsigned long long)(long long)(PF_KEY));
#endif
#ifdef PF_KSWAPD
printf("PF_KSWAPD %llu\n", (unsigned long long)(long long)(PF_KSWAPD));
#endif
#ifdef PF_KTHREAD
printf("PF_KTHREAD %llu\n", (unsigned long long)(long long)(PF_KTHREAD));
#endif
#ifdef PF_LLC
printf("PF_LLC %llu\n", (unsigned long long)(long long)(PF_LLC));
#endif
#ifdef PF_LOCAL
printf("PF_LOCAL %llu\n", (unsigned long long)(long long)(PF_LOCAL));
#endif
#ifdef PF_LOCAL_THROTTLE
printf("PF_LOCAL_THROTTLE %llu\n", (unsigned long long)(long long)(PF_LOCAL_THROTTLE));
#endif
#ifdef PF_MASKOS
printf("PF_MASKOS %llu\n", (unsigned long long)(long long)(PF_MASKOS));
#endif
#ifdef PF_MASKPROC
printf("PF_MASKPROC %llu\n", (unsigned long long)(long long)(PF_MASKPROC));
#endif
#ifdef PF_MCE_EARLY
printf("PF_MCE_EARLY %llu\n", (unsigned long long)(long long)(PF_MCE_EARLY));
#endif
#ifdef PF_MCE_PROCESS
printf("PF_MCE_PROCESS %llu\n", (unsigned long long)(long long)(PF_MCE_PROCESS));
#endif
#ifdef PF_MEMALLOC
printf("PF_MEMALLOC %llu\n", (unsigned long long)(long long)(PF_MEMALLOC));
#endif
#ifdef PF_MEMALLOC_NOFS
printf("PF_MEMALLOC_NOFS %llu\n", (unsigned long long)(long long)(PF_MEMALLOC_NOFS));
#endif
#ifdef PF_MEMALLOC_NOIO
printf("PF_MEMALLOC_NOIO %llu\n", (unsigned long long)(long long)(PF_MEMALLOC_NOIO));
#endif
#ifdef PF_MEMALLOC_PIN
printf("PF_MEMALLOC_PIN %llu\n", (unsigned long long)(long long)(PF_MEMALLOC_PIN));
#endif
#ifdef PF_MPLS
printf("PF_MPLS %llu\n", (unsigned long long)(long long)(PF_MPLS));
#endif
#ifdef PF_NETBEUI
printf("PF_NETBEUI %llu\n", (unsigned long long)(long long)(PF_NETBEUI));
#endif
#ifdef PF_NETLINK
printf("PF_NETLINK %llu\n", (unsigned long long)(long long)(PF_NETLINK));
#endif
#ifdef PF_NETROM
printf("PF_NETROM %llu\n", (unsigned long long)(long long)(PF_NETROM));
#endif
#ifdef PF_NFC
printf("PF_NFC %llu\n", (unsigned long long)(long long)(PF_NFC));
#endif
#ifdef PF_NOFREEZE
printf("PF_NOFREEZE %llu\n", (unsigned long long)(long long)(PF_NOFREEZE));
#endif
#ifdef PF_NO_SETAFFINITY
printf("PF_NO_SETAFFINITY %llu\n", (unsigned long long)(long long)(PF_NO_SETAFFINITY));
#endif
#ifdef PF_NPROC_EXCEEDED
printf("PF_NPROC_EXCEEDED %llu\n", (unsigned long long)(long long)(PF_NPROC_EXCEEDED));
#endif
#ifdef PF_PACKET
printf("PF_PACKET %llu\n", (unsigned long long)(long long)(PF_PACKET));
#endif
#ifdef PF_PHONET
printf("PF_PHONET %llu\n", (unsigned long long)(long long)(PF_PHONET));
#endif
#ifdef PF_POSTCOREDUMP
printf("PF_POSTCOREDUMP %llu\n", (unsigned long long)(long long)(PF_POSTCOREDUMP));
#endif
#ifdef PF_PPPOX
printf("PF_PPPOX %llu\n", (unsigned long long)(long long)(PF_PPPOX));
#endif
#ifdef PF_R
printf("PF_R %llu\n", (unsigned long long)(long long)(PF_R));
#endif
#ifdef PF_RANDOMIZE
printf("PF_RANDOMIZE %llu\n", (unsigned long long)(long long)(PF_RANDOMIZE));
#endif
#ifdef PF_RDS
printf("PF_RDS %llu\n", (unsigned long long)(long long)(PF_RDS));
#endif
#ifdef PF_ROSE
printf("PF_ROSE %llu\n", (unsigned long long)(long long)(PF_ROSE));
#endif
#ifdef PF_ROUTE
printf("PF_ROUTE %llu\n", (unsigned long long)(long long)(PF_ROUTE));
#endif
#ifdef PF_RXRPC
printf("PF_RXRPC %llu\n", (unsigned long long)(long long)(PF_RXRPC));
#endif
#ifdef PF_SECURITY
printf("PF_SECURITY %llu\n", (unsigned long long)(long long)(PF_SECURITY));
#endif
#ifdef PF_SIGNALED
printf("PF_SIGNALED %llu\n", (unsigned long long)(long long)(PF_SIGNALED));
#endif
#ifdef PF_SNA
printf("PF_SNA %llu\n", (unsigned long long)(long long)(PF_SNA));
#endif
#ifdef PF_SUPERPRIV
printf("PF_SUPERPRIV %llu\n", (unsigned long long)(long long)(PF_SUPERPRIV));
#endif
#ifdef PF_SUSPEND_TASK
printf("PF_SUSPEND_TASK %llu\n", (unsigned long long)(long long)(PF_SUSPEND_TASK));
#endif
#ifdef PF_TIPC
printf("PF_TIPC %llu\n", (unsigned long long)(long long)(PF_TIPC));
#endif
#ifdef PF_UNIX
printf("PF_UNIX %llu\n", (unsigned long long)(long long)(PF_UNIX));
#endif
#ifdef PF_UNSPEC
printf("PF_UNSPEC %llu\n", (unsigned long long)(long long)(PF_UNSPEC));
#endif
#ifdef PF_USED_MATH
printf("PF_USED_MATH %llu\n", (unsigned long long)(long long)(PF_USED_MATH));
#endif
#ifdef PF_USER_WORKER
printf("PF_USER_WORKER %llu\n", (unsigned long long)(long long)(PF_USER_WORKER));
#endif
#ifdef PF_VCPU
printf("PF_VCPU %llu\n", (unsigned long long)(long long)(PF_VCPU));
#endif
#ifdef PF_VSOCK
printf("PF_VSOCK %llu\n", (unsigned long long)(long long)(PF_VSOCK));
#endif
#ifdef PF_W
printf("PF_W %llu\n", (unsigned long long)(long long)(PF_W));
#endif
#ifdef PF_WANPIPE
printf("PF_WANPIPE %llu\n", (unsigned long long)(long long)(PF_WANPIPE));
#endif
#ifdef PF_WQ_WORKER
printf("PF_WQ_WORKER %llu\n", (unsigned long long)(long long)(PF_WQ_WORKER));
#endif
#ifdef PF_X
printf("PF_X %llu\n", (unsigned long long)(long long)(PF_X));
#endif
#ifdef PF_X25
printf("PF_X25 %llu\n", (unsigned long long)(long long)(PF_X25));
#endif
#ifdef PF_XDP
printf("PF_XDP %llu\n", (unsigned long long)(long long)(PF_XDP));
#endif
#ifdef PIDFD_GET_CGROUP_NAMESPACE
printf("PIDFD_GET_CGROUP_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_CGROUP_NAMESPACE));
#endif
#ifdef PIDFD_GET_INFO
printf("PIDFD_GET_INFO %llu\n", (unsigned long long)(long long)(PIDFD_GET_INFO));
#endif
#ifdef PIDFD_GET_IPC_NAMESPACE
printf("PIDFD_GET_IPC_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_IPC_NAMESPACE));
#endif
#ifdef PIDFD_GET_MNT_NAMESPACE
printf("PIDFD_GET_MNT_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_MNT_NAMESPACE));
#endif
#ifdef PIDFD_GET_NET_NAMESPACE
printf("PIDFD_GET_NET_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_NET_NAMESPACE));
#endif
#ifdef PIDFD_GET_PID_FOR_CHILDREN_NAMESPACE
printf("PIDFD_GET_PID_FOR_CHILDREN_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_PID_FOR_CHILDREN_NAMESPACE));
#endif
#ifdef PIDFD_GET_PID_NAMESPACE
printf("PIDFD_GET_PID_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_PID_NAMESPACE));
#endif
#ifdef PIDFD_GET_TIME_FOR_CHILDREN_NAMESPACE
printf("PIDFD_GET_TIME_FOR_CHILDREN_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_TIME_FOR_CHILDREN_NAMESPACE));
#endif
#ifdef PIDFD_GET_TIME_NAMESPACE
printf("PIDFD_GET_TIME_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_TIME_NAMESPACE));
#endif
#ifdef PIDFD_GET_USER_NAMESPACE
printf("PIDFD_GET_USER_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_USER_NAMESPACE));
#endif
#ifdef PIDFD_GET_UTS_NAMESPACE
printf("PIDFD_GET_UTS_NAMESPACE %llu\n", (unsigned long long)(long long)(PIDFD_GET_UTS_NAMESPACE));
#endif
#ifdef PIDFD_INFO_CGROUPID
printf("PIDFD_INFO_CGROUPID %llu\n", (unsigned long long)(long long)(PIDFD_INFO_CGROUPID));
#endif
#ifdef PIDFD_INFO_CREDS
printf("PIDFD_INFO_CREDS %llu\n", (unsigned long long)(long long)(PIDFD_INFO_CREDS));
#endif
#ifdef PIDFD_INFO_EXIT
printf("PIDFD_INFO_EXIT %llu\n", (unsigned long long)(long long)(PIDFD_INFO_EXIT));
#endif
#ifdef PIDFD_INFO_PID
printf("PIDFD_INFO_PID %llu\n", (unsigned long long)(long long)(PIDFD_INFO_PID));
#endif
#ifdef PIDFD_INFO_SIZE_VER0
printf("PIDFD_INFO_SIZE_VER0 %llu\n", (unsigned long long)(long long)(PIDFD_INFO_SIZE_VER0));
#endif
#ifdef PIDFD_NONBLOCK
printf("PIDFD_NONBLOCK %llu\n", (unsigned long long)(long long)(PIDFD_NONBLOCK));
#endif
#ifdef PIDFD_SIGNAL_PROCESS_GROUP
printf("PIDFD_SIGNAL_PROCESS_GROUP %llu\n", (unsigned long long)(long long)(PIDFD_SIGNAL_PROCESS_GROUP));
#endif
#ifdef PIDFD_SIGNAL_THREAD
printf("PIDFD_SIGNAL_THREAD %llu\n", (unsigned long long)(long long)(PIDFD_SIGNAL_THREAD));
#endif
#ifdef PIDFD_SIGNAL_THREAD_GROUP
printf("PIDFD_SIGNAL_THREAD_GROUP %llu\n", (unsigned long long)(long long)(PIDFD_SIGNAL_THREAD_GROUP));
#endif
#ifdef PIDFD_THREAD
printf("PIDFD_THREAD %llu\n", (unsigned long long)(long long)(PIDFD_THREAD));
#endif
#ifdef PIPE_BUF
printf("PIPE_BUF %llu\n", (unsigned long long)(long long)(PIPE_BUF));
#endif
#ifdef PM_STR
printf("PM_STR %llu\n", (unsigned long long)(long long)(PM_STR));
#endif
#ifdef POLLERR
printf("POLLERR %llu\n", (unsigned long long)(long long)(POLLERR));
#endif
#ifdef POLLHUP
printf("POLLHUP %llu\n", (unsigned long long)(long long)(POLLHUP));
#endif
#ifdef POLLIN
printf("POLLIN %llu\n", (unsigned long long)(long long)(POLLIN));
#endif
#ifdef POLLNVAL
printf("POLLNVAL %llu\n", (unsigned long long)(long long)(POLLNVAL));
#endif
#ifdef POLLOUT
printf("POLLOUT %llu\n", (unsigned long long)(long long)(POLLOUT));
#endif
#ifdef POLLPRI
printf("POLLPRI %llu\n", (unsigned long long)(long long)(POLLPRI));
#endif
#ifdef POLLRDBAND
printf("POLLRDBAND %llu\n", (unsigned long long)(long long)(POLLRDBAND));
#endif
#ifdef POLLRDHUP
printf("POLLRDHUP %llu\n", (unsigned long long)(long long)(POLLRDHUP));
#endif
#ifdef POLLRDNORM
printf("POLLRDNORM %llu\n", (unsigned long long)(long long)(POLLRDNORM));
#endif
#ifdef POLLWRBAND
printf("POLLWRBAND %llu\n", (unsigned long long)(long long)(POLLWRBAND));
#endif
#ifdef POLLWRNORM
printf("POLLWRNORM %llu\n", (unsigned long long)(long long)(POLLWRNORM));
#endif
#ifdef POSIX_FADV_DONTNEED
printf("POSIX_FADV_DONTNEED %llu\n", (unsigned long long)(long long)(POSIX_FADV_DONTNEED));
#endif
#ifdef POSIX_FADV_NOREUSE
printf("POSIX_FADV_NOREUSE %llu\n", (unsigned long long)(long long)(POSIX_FADV_NOREUSE));
#endif
#ifdef POSIX_FADV_NORMAL
printf("POSIX_FADV_NORMAL %llu\n", (unsigned long long)(long long)(POSIX_FADV_NORMAL));
#endif
#ifdef POSIX_FADV_RANDOM
printf("POSIX_FADV_RANDOM %llu\n", (unsigned long long)(long long)(POSIX_FADV_RANDOM));
#endif
#ifdef POSIX_FADV_SEQUENTIAL
printf("POSIX_FADV_SEQUENTIAL %llu\n", (unsigned long long)(long long)(POSIX_FADV_SEQUENTIAL));
#endif
#ifdef POSIX_FADV_WILLNEED
printf("POSIX_FADV_WILLNEED %llu\n", (unsigned long long)(long long)(POSIX_FADV_WILLNEED));
#endif
#ifdef POSIX_MADV_DONTNEED
printf("POSIX_MADV_DONTNEED %llu\n", (unsigned long long)(long long)(POSIX_MADV_DONTNEED));
#endif
#ifdef POSIX_MADV_NORMAL
printf("POSIX_MADV_NORMAL %llu\n", (unsigned long long)(long long)(POSIX_MADV_NORMAL));
#endif
#ifdef POSIX_MADV_RANDOM
printf("POSIX_MADV_RANDOM %llu\n", (unsigned long long)(long long)(POSIX_MADV_RANDOM));
#endif
#ifdef POSIX_MADV_SEQUENTIAL
printf("POSIX_MADV_SEQUENTIAL %llu\n", (unsigned long long)(long long)(POSIX_MADV_SEQUENTIAL));
#endif
#ifdef POSIX_MADV_WILLNEED
printf("POSIX_MADV_WILLNEED %llu\n", (unsigned long long)(long long)(POSIX_MADV_WILLNEED));
#endif
#ifdef POSIX_SPAWN_RESETIDS
printf("POSIX_SPAWN_RESETIDS %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_RESETIDS));
#endif
#ifdef POSIX_SPAWN_SETPGROUP
printf("POSIX_SPAWN_SETPGROUP %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETPGROUP));
#endif
#ifdef POSIX_SPAWN_SETSCHEDPARAM
printf("POSIX_SPAWN_SETSCHEDPARAM %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETSCHEDPARAM));
#endif
#ifdef POSIX_SPAWN_SETSCHEDULER
printf("POSIX_SPAWN_SETSCHEDULER %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETSCHEDULER));
#endif
#ifdef POSIX_SPAWN_SETSID
printf("POSIX_SPAWN_SETSID %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETSID));
#endif
#ifdef POSIX_SPAWN_SETSIGDEF
printf("POSIX_SPAWN_SETSIGDEF %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETSIGDEF));
#endif
#ifdef POSIX_SPAWN_SETSIGMASK
printf("POSIX_SPAWN_SETSIGMASK %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_SETSIGMASK));
#endif
#ifdef POSIX_SPAWN_USEVFORK
printf("POSIX_SPAWN_USEVFORK %llu\n", (unsigned long long)(long long)(POSIX_SPAWN_USEVFORK));
#endif
#ifdef PRIO_MAX
printf("PRIO_MAX %llu\n", (unsigned long long)(long long)(PRIO_MAX));
#endif
#ifdef PRIO_MIN
printf("PRIO_MIN %llu\n", (unsigned long long)(long long)(PRIO_MIN));
#endif
#ifdef PRIO_PGRP
printf("PRIO_PGRP %llu\n", (unsigned long long)(long long)(PRIO_PGRP));
#endif
#ifdef PRIO_PROCESS
printf("PRIO_PROCESS %llu\n", (unsigned long long)(long long)(PRIO_PROCESS));
#endif
#ifdef PRIO_USER
printf("PRIO_USER %llu\n", (unsigned long long)(long long)(PRIO_USER));
#endif
#ifdef PROC_SUPER_MAGIC
printf("PROC_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(PROC_SUPER_MAGIC));
#endif
#ifdef PROT_EXEC
printf("PROT_EXEC %llu\n", (unsigned long long)(long long)(PROT_EXEC));
#endif
#ifdef PROT_GROWSDOWN
printf("PROT_GROWSDOWN %llu\n", (unsigned long long)(long long)(PROT_GROWSDOWN));
#endif
#ifdef PROT_GROWSUP
printf("PROT_GROWSUP %llu\n", (unsigned long long)(long long)(PROT_GROWSUP));
#endif
#ifdef PROT_NONE
printf("PROT_NONE %llu\n", (unsigned long long)(long long)(PROT_NONE));
#endif
#ifdef PROT_READ
printf("PROT_READ %llu\n", (unsigned long long)(long long)(PROT_READ));
#endif
#ifdef PROT_WRITE
printf("PROT_WRITE %llu\n", (unsigned long long)(long long)(PROT_WRITE));
#endif
#ifdef PR_CAPBSET_DROP
printf("PR_CAPBSET_DROP %llu\n", (unsigned long long)(long long)(PR_CAPBSET_DROP));
#endif
#ifdef PR_CAPBSET_READ
printf("PR_CAPBSET_READ %llu\n", (unsigned long long)(long long)(PR_CAPBSET_READ));
#endif
#ifdef PR_CAP_AMBIENT
printf("PR_CAP_AMBIENT %llu\n", (unsigned long long)(long long)(PR_CAP_AMBIENT));
#endif
#ifdef PR_CAP_AMBIENT_CLEAR_ALL
printf("PR_CAP_AMBIENT_CLEAR_ALL %llu\n", (unsigned long long)(long long)(PR_CAP_AMBIENT_CLEAR_ALL));
#endif
#ifdef PR_CAP_AMBIENT_IS_SET
printf("PR_CAP_AMBIENT_IS_SET %llu\n", (unsigned long long)(long long)(PR_CAP_AMBIENT_IS_SET));
#endif
#ifdef PR_CAP_AMBIENT_LOWER
printf("PR_CAP_AMBIENT_LOWER %llu\n", (unsigned long long)(long long)(PR_CAP_AMBIENT_LOWER));
#endif
#ifdef PR_CAP_AMBIENT_RAISE
printf("PR_CAP_AMBIENT_RAISE %llu\n", (unsigned long long)(long long)(PR_CAP_AMBIENT_RAISE));
#endif
#ifdef PR_ENDIAN_BIG
printf("PR_ENDIAN_BIG %llu\n", (unsigned long long)(long long)(PR_ENDIAN_BIG));
#endif
#ifdef PR_ENDIAN_LITTLE
printf("PR_ENDIAN_LITTLE %llu\n", (unsigned long long)(long long)(PR_ENDIAN_LITTLE));
#endif
#ifdef PR_ENDIAN_PPC_LITTLE
printf("PR_ENDIAN_PPC_LITTLE %llu\n", (unsigned long long)(long long)(PR_ENDIAN_PPC_LITTLE));
#endif
#ifdef PR_FPEMU_NOPRINT
printf("PR_FPEMU_NOPRINT %llu\n", (unsigned long long)(long long)(PR_FPEMU_NOPRINT));
#endif
#ifdef PR_FPEMU_SIGFPE
printf("PR_FPEMU_SIGFPE %llu\n", (unsigned long long)(long long)(PR_FPEMU_SIGFPE));
#endif
#ifdef PR_FP_EXC_ASYNC
printf("PR_FP_EXC_ASYNC %llu\n", (unsigned long long)(long long)(PR_FP_EXC_ASYNC));
#endif
#ifdef PR_FP_EXC_DISABLED
printf("PR_FP_EXC_DISABLED %llu\n", (unsigned long long)(long long)(PR_FP_EXC_DISABLED));
#endif
#ifdef PR_FP_EXC_DIV
printf("PR_FP_EXC_DIV %llu\n", (unsigned long long)(long long)(PR_FP_EXC_DIV));
#endif
#ifdef PR_FP_EXC_INV
printf("PR_FP_EXC_INV %llu\n", (unsigned long long)(long long)(PR_FP_EXC_INV));
#endif
#ifdef PR_FP_EXC_NONRECOV
printf("PR_FP_EXC_NONRECOV %llu\n", (unsigned long long)(long long)(PR_FP_EXC_NONRECOV));
#endif
#ifdef PR_FP_EXC_OVF
printf("PR_FP_EXC_OVF %llu\n", (unsigned long long)(long long)(PR_FP_EXC_OVF));
#endif
#ifdef PR_FP_EXC_PRECISE
printf("PR_FP_EXC_PRECISE %llu\n", (unsigned long long)(long long)(PR_FP_EXC_PRECISE));
#endif
#ifdef PR_FP_EXC_RES
printf("PR_FP_EXC_RES %llu\n", (unsigned long long)(long long)(PR_FP_EXC_RES));
#endif
#ifdef PR_FP_EXC_SW_ENABLE
printf("PR_FP_EXC_SW_ENABLE %llu\n", (unsigned long long)(long long)(PR_FP_EXC_SW_ENABLE));
#endif
#ifdef PR_FP_EXC_UND
printf("PR_FP_EXC_UND %llu\n", (unsigned long long)(long long)(PR_FP_EXC_UND));
#endif
#ifdef PR_FP_MODE_FR
printf("PR_FP_MODE_FR %llu\n", (unsigned long long)(long long)(PR_FP_MODE_FR));
#endif
#ifdef PR_FP_MODE_FRE
printf("PR_FP_MODE_FRE %llu\n", (unsigned long long)(long long)(PR_FP_MODE_FRE));
#endif
#ifdef PR_GET_CHILD_SUBREAPER
printf("PR_GET_CHILD_SUBREAPER %llu\n", (unsigned long long)(long long)(PR_GET_CHILD_SUBREAPER));
#endif
#ifdef PR_GET_DUMPABLE
printf("PR_GET_DUMPABLE %llu\n", (unsigned long long)(long long)(PR_GET_DUMPABLE));
#endif
#ifdef PR_GET_ENDIAN
printf("PR_GET_ENDIAN %llu\n", (unsigned long long)(long long)(PR_GET_ENDIAN));
#endif
#ifdef PR_GET_FPEMU
printf("PR_GET_FPEMU %llu\n", (unsigned long long)(long long)(PR_GET_FPEMU));
#endif
#ifdef PR_GET_FPEXC
printf("PR_GET_FPEXC %llu\n", (unsigned long long)(long long)(PR_GET_FPEXC));
#endif
#ifdef PR_GET_FP_MODE
printf("PR_GET_FP_MODE %llu\n", (unsigned long long)(long long)(PR_GET_FP_MODE));
#endif
#ifdef PR_GET_KEEPCAPS
printf("PR_GET_KEEPCAPS %llu\n", (unsigned long long)(long long)(PR_GET_KEEPCAPS));
#endif
#ifdef PR_GET_MDWE
printf("PR_GET_MDWE %llu\n", (unsigned long long)(long long)(PR_GET_MDWE));
#endif
#ifdef PR_GET_NAME
printf("PR_GET_NAME %llu\n", (unsigned long long)(long long)(PR_GET_NAME));
#endif
#ifdef PR_GET_NO_NEW_PRIVS
printf("PR_GET_NO_NEW_PRIVS %llu\n", (unsigned long long)(long long)(PR_GET_NO_NEW_PRIVS));
#endif
#ifdef PR_GET_PDEATHSIG
printf("PR_GET_PDEATHSIG %llu\n", (unsigned long long)(long long)(PR_GET_PDEATHSIG));
#endif
#ifdef PR_GET_SECCOMP
printf("PR_GET_SECCOMP %llu\n", (unsigned long long)(long long)(PR_GET_SECCOMP));
#endif
#ifdef PR_GET_SECUREBITS
printf("PR_GET_SECUREBITS %llu\n", (unsigned long long)(long long)(PR_GET_SECUREBITS));
#endif
#ifdef PR_GET_THP_DISABLE
printf("PR_GET_THP_DISABLE %llu\n", (unsigned long long)(long long)(PR_GET_THP_DISABLE));
#endif
#ifdef PR_GET_TID_ADDRESS
printf("PR_GET_TID_ADDRESS %llu\n", (unsigned long long)(long long)(PR_GET_TID_ADDRESS));
#endif
#ifdef PR_GET_TIMERSLACK
printf("PR_GET_TIMERSLACK %llu\n", (unsigned long long)(long long)(PR_GET_TIMERSLACK));
#endif
#ifdef PR_GET_TIMING
printf("PR_GET_TIMING %llu\n", (unsigned long long)(long long)(PR_GET_TIMING));
#endif
#ifdef PR_GET_TSC
printf("PR_GET_TSC %llu\n", (unsigned long long)(long long)(PR_GET_TSC));
#endif
#ifdef PR_GET_UNALIGN
printf("PR_GET_UNALIGN %llu\n", (unsigned long long)(long long)(PR_GET_UNALIGN));
#endif
#ifdef PR_MCE_KILL
printf("PR_MCE_KILL %llu\n", (unsigned long long)(long long)(PR_MCE_KILL));
#endif
#ifdef PR_MCE_KILL_CLEAR
printf("PR_MCE_KILL_CLEAR %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_CLEAR));
#endif
#ifdef PR_MCE_KILL_DEFAULT
printf("PR_MCE_KILL_DEFAULT %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_DEFAULT));
#endif
#ifdef PR_MCE_KILL_EARLY
printf("PR_MCE_KILL_EARLY %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_EARLY));
#endif
#ifdef PR_MCE_KILL_GET
printf("PR_MCE_KILL_GET %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_GET));
#endif
#ifdef PR_MCE_KILL_LATE
printf("PR_MCE_KILL_LATE %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_LATE));
#endif
#ifdef PR_MCE_KILL_SET
printf("PR_MCE_KILL_SET %llu\n", (unsigned long long)(long long)(PR_MCE_KILL_SET));
#endif
#ifdef PR_MDWE_NO_INHERIT
printf("PR_MDWE_NO_INHERIT %llu\n", (unsigned long long)(long long)(PR_MDWE_NO_INHERIT));
#endif
#ifdef PR_MDWE_REFUSE_EXEC_GAIN
printf("PR_MDWE_REFUSE_EXEC_GAIN %llu\n", (unsigned long long)(long long)(PR_MDWE_REFUSE_EXEC_GAIN));
#endif
#ifdef PR_MPX_DISABLE_MANAGEMENT
printf("PR_MPX_DISABLE_MANAGEMENT %llu\n", (unsigned long long)(long long)(PR_MPX_DISABLE_MANAGEMENT));
#endif
#ifdef PR_MPX_ENABLE_MANAGEMENT
printf("PR_MPX_ENABLE_MANAGEMENT %llu\n", (unsigned long long)(long long)(PR_MPX_ENABLE_MANAGEMENT));
#endif
#ifdef PR_SCHED_CORE
printf("PR_SCHED_CORE %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE));
#endif
#ifdef PR_SCHED_CORE_CREATE
printf("PR_SCHED_CORE_CREATE %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_CREATE));
#endif
#ifdef PR_SCHED_CORE_GET
printf("PR_SCHED_CORE_GET %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_GET));
#endif
#ifdef PR_SCHED_CORE_MAX
printf("PR_SCHED_CORE_MAX %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_MAX));
#endif
#ifdef PR_SCHED_CORE_SCOPE_PROCESS_GROUP
printf("PR_SCHED_CORE_SCOPE_PROCESS_GROUP %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_SCOPE_PROCESS_GROUP));
#endif
#ifdef PR_SCHED_CORE_SCOPE_THREAD
printf("PR_SCHED_CORE_SCOPE_THREAD %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_SCOPE_THREAD));
#endif
#ifdef PR_SCHED_CORE_SCOPE_THREAD_GROUP
printf("PR_SCHED_CORE_SCOPE_THREAD_GROUP %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_SCOPE_THREAD_GROUP));
#endif
#ifdef PR_SCHED_CORE_SHARE_FROM
printf("PR_SCHED_CORE_SHARE_FROM %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_SHARE_FROM));
#endif
#ifdef PR_SCHED_CORE_SHARE_TO
printf("PR_SCHED_CORE_SHARE_TO %llu\n", (unsigned long long)(long long)(PR_SCHED_CORE_SHARE_TO));
#endif
#ifdef PR_SET_CHILD_SUBREAPER
printf("PR_SET_CHILD_SUBREAPER %llu\n", (unsigned long long)(long long)(PR_SET_CHILD_SUBREAPER));
#endif
#ifdef PR_SET_DUMPABLE
printf("PR_SET_DUMPABLE %llu\n", (unsigned long long)(long long)(PR_SET_DUMPABLE));
#endif
#ifdef PR_SET_ENDIAN
printf("PR_SET_ENDIAN %llu\n", (unsigned long long)(long long)(PR_SET_ENDIAN));
#endif
#ifdef PR_SET_FPEMU
printf("PR_SET_FPEMU %llu\n", (unsigned long long)(long long)(PR_SET_FPEMU));
#endif
#ifdef PR_SET_FPEXC
printf("PR_SET_FPEXC %llu\n", (unsigned long long)(long long)(PR_SET_FPEXC));
#endif
#ifdef PR_SET_FP_MODE
printf("PR_SET_FP_MODE %llu\n", (unsigned long long)(long long)(PR_SET_FP_MODE));
#endif
#ifdef PR_SET_KEEPCAPS
printf("PR_SET_KEEPCAPS %llu\n", (unsigned long long)(long long)(PR_SET_KEEPCAPS));
#endif
#ifdef PR_SET_MDWE
printf("PR_SET_MDWE %llu\n", (unsigned long long)(long long)(PR_SET_MDWE));
#endif
#ifdef PR_SET_MM
printf("PR_SET_MM %llu\n", (unsigned long long)(long long)(PR_SET_MM));
#endif
#ifdef PR_SET_MM_ARG_END
printf("PR_SET_MM_ARG_END %llu\n", (unsigned long long)(long long)(PR_SET_MM_ARG_END));
#endif
#ifdef PR_SET_MM_ARG_START
printf("PR_SET_MM_ARG_START %llu\n", (unsigned long long)(long long)(PR_SET_MM_ARG_START));
#endif
#ifdef PR_SET_MM_AUXV
printf("PR_SET_MM_AUXV %llu\n", (unsigned long long)(long long)(PR_SET_MM_AUXV));
#endif
#ifdef PR_SET_MM_BRK
printf("PR_SET_MM_BRK %llu\n", (unsigned long long)(long long)(PR_SET_MM_BRK));
#endif
#ifdef PR_SET_MM_END_CODE
printf("PR_SET_MM_END_CODE %llu\n", (unsigned long long)(long long)(PR_SET_MM_END_CODE));
#endif
#ifdef PR_SET_MM_END_DATA
printf("PR_SET_MM_END_DATA %llu\n", (unsigned long long)(long long)(PR_SET_MM_END_DATA));
#endif
#ifdef PR_SET_MM_ENV_END
printf("PR_SET_MM_ENV_END %llu\n", (unsigned long long)(long long)(PR_SET_MM_ENV_END));
#endif
#ifdef PR_SET_MM_ENV_START
printf("PR_SET_MM_ENV_START %llu\n", (unsigned long long)(long long)(PR_SET_MM_ENV_START));
#endif
#ifdef PR_SET_MM_EXE_FILE
printf("PR_SET_MM_EXE_FILE %llu\n", (unsigned long long)(long long)(PR_SET_MM_EXE_FILE));
#endif
#ifdef PR_SET_MM_MAP
printf("PR_SET_MM_MAP %llu\n", (unsigned long long)(long long)(PR_SET_MM_MAP));
#endif
#ifdef PR_SET_MM_MAP_SIZE
printf("PR_SET_MM_MAP_SIZE %llu\n", (unsigned long long)(long long)(PR_SET_MM_MAP_SIZE));
#endif
#ifdef PR_SET_MM_START_BRK
printf("PR_SET_MM_START_BRK %llu\n", (unsigned long long)(long long)(PR_SET_MM_START_BRK));
#endif
#ifdef PR_SET_MM_START_CODE
printf("PR_SET_MM_START_CODE %llu\n", (unsigned long long)(long long)(PR_SET_MM_START_CODE));
#endif
#ifdef PR_SET_MM_START_DATA
printf("PR_SET_MM_START_DATA %llu\n", (unsigned long long)(long long)(PR_SET_MM_START_DATA));
#endif
#ifdef PR_SET_MM_START_STACK
printf("PR_SET_MM_START_STACK %llu\n", (unsigned long long)(long long)(PR_SET_MM_START_STACK));
#endif
#ifdef PR_SET_NAME
printf("PR_SET_NAME %llu\n", (unsigned long long)(long long)(PR_SET_NAME));
#endif
#ifdef PR_SET_NO_NEW_PRIVS
printf("PR_SET_NO_NEW_PRIVS %llu\n", (unsigned long long)(long long)(PR_SET_NO_NEW_PRIVS));
#endif
#ifdef PR_SET_PDEATHSIG
printf("PR_SET_PDEATHSIG %llu\n", (unsigned long long)(long long)(PR_SET_PDEATHSIG));
#endif
#ifdef PR_SET_PTRACER
printf("PR_SET_PTRACER %llu\n", (unsigned long long)(long long)(PR_SET_PTRACER));
#endif
#ifdef PR_SET_PTRACER_ANY
printf("PR_SET_PTRACER_ANY %llu\n", (unsigned long long)(long long)(PR_SET_PTRACER_ANY));
#endif
#ifdef PR_SET_SECCOMP
printf("PR_SET_SECCOMP %llu\n", (unsigned long long)(long long)(PR_SET_SECCOMP));
#endif
#ifdef PR_SET_SECUREBITS
printf("PR_SET_SECUREBITS %llu\n", (unsigned long long)(long long)(PR_SET_SECUREBITS));
#endif
#ifdef PR_SET_THP_DISABLE
printf("PR_SET_THP_DISABLE %llu\n", (unsigned long long)(long long)(PR_SET_THP_DISABLE));
#endif
#ifdef PR_SET_TIMERSLACK
printf("PR_SET_TIMERSLACK %llu\n", (unsigned long long)(long long)(PR_SET_TIMERSLACK));
#endif
#ifdef PR_SET_TIMING
printf("PR_SET_TIMING %llu\n", (unsigned long long)(long long)(PR_SET_TIMING));
#endif
#ifdef PR_SET_TSC
printf("PR_SET_TSC %llu\n", (unsigned long long)(long long)(PR_SET_TSC));
#endif
#ifdef PR_SET_UNALIGN
printf("PR_SET_UNALIGN %llu\n", (unsigned long long)(long long)(PR_SET_UNALIGN));
#endif
#ifdef PR_SET_VMA
printf("PR_SET_VMA %llu\n", (unsigned long long)(long long)(PR_SET_VMA));
#endif
#ifdef PR_SET_VMA_ANON_NAME
printf("PR_SET_VMA_ANON_NAME %llu\n", (unsigned long long)(long long)(PR_SET_VMA_ANON_NAME));
#endif
#ifdef PR_TASK_PERF_EVENTS_DISABLE
printf("PR_TASK_PERF_EVENTS_DISABLE %llu\n", (unsigned long long)(long long)(PR_TASK_PERF_EVENTS_DISABLE));
#endif
#ifdef PR_TASK_PERF_EVENTS_ENABLE
printf("PR_TASK_PERF_EVENTS_ENABLE %llu\n", (unsigned long long)(long long)(PR_TASK_PERF_EVENTS_ENABLE));
#endif
#ifdef PR_TIMING_STATISTICAL
printf("PR_TIMING_STATISTICAL %llu\n", (unsigned long long)(long long)(PR_TIMING_STATISTICAL));
#endif
#ifdef PR_TIMING_TIMESTAMP
printf("PR_TIMING_TIMESTAMP %llu\n", (unsigned long long)(long long)(PR_TIMING_TIMESTAMP));
#endif
#ifdef PR_TSC_ENABLE
printf("PR_TSC_ENABLE %llu\n", (unsigned long long)(long long)(PR_TSC_ENABLE));
#endif
#ifdef PR_TSC_SIGSEGV
printf("PR_TSC_SIGSEGV %llu\n", (unsigned long long)(long long)(PR_TSC_SIGSEGV));
#endif
#ifdef PR_UNALIGN_NOPRINT
printf("PR_UNALIGN_NOPRINT %llu\n", (unsigned long long)(long long)(PR_UNALIGN_NOPRINT));
#endif
#ifdef PR_UNALIGN_SIGBUS
printf("PR_UNALIGN_SIGBUS %llu\n", (unsigned long long)(long long)(PR_UNALIGN_SIGBUS));
#endif
#ifdef PTHREAD_BARRIER_SERIAL_THREAD
printf("PTHREAD_BARRIER_SERIAL_THREAD %llu\n", (unsigned long long)(long long)(PTHREAD_BARRIER_SERIAL_THREAD));
#endif
#ifdef PTHREAD_CREATE_DETACHED
printf("PTHREAD_CREATE_DETACHED %llu\n", (unsigned long long)(long long)(PTHREAD_CREATE_DETACHED));
#endif
#ifdef PTHREAD_CREATE_JOINABLE
printf("PTHREAD_CREATE_JOINABLE %llu\n", (unsigned long long)(long long)(PTHREAD_CREATE_JOINABLE));
#endif
#ifdef PTHREAD_EXPLICIT_SCHED
printf("PTHREAD_EXPLICIT_SCHED %llu\n", (unsigned long long)(long long)(PTHREAD_EXPLICIT_SCHED));
#endif
#ifdef PTHREAD_INHERIT_SCHED
printf("PTHREAD_INHERIT_SCHED %llu\n", (unsigned long long)(long long)(PTHREAD_INHERIT_SCHED));
#endif
#ifdef PTHREAD_MUTEX_ADAPTIVE_NP
printf("PTHREAD_MUTEX_ADAPTIVE_NP %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_ADAPTIVE_NP));
#endif
#ifdef PTHREAD_MUTEX_DEFAULT
printf("PTHREAD_MUTEX_DEFAULT %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_DEFAULT));
#endif
#ifdef PTHREAD_MUTEX_ERRORCHECK
printf("PTHREAD_MUTEX_ERRORCHECK %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_ERRORCHECK));
#endif
#ifdef PTHREAD_MUTEX_NORMAL
printf("PTHREAD_MUTEX_NORMAL %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_NORMAL));
#endif
#ifdef PTHREAD_MUTEX_RECURSIVE
printf("PTHREAD_MUTEX_RECURSIVE %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_RECURSIVE));
#endif
#ifdef PTHREAD_MUTEX_ROBUST
printf("PTHREAD_MUTEX_ROBUST %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_ROBUST));
#endif
#ifdef PTHREAD_MUTEX_STALLED
printf("PTHREAD_MUTEX_STALLED %llu\n", (unsigned long long)(long long)(PTHREAD_MUTEX_STALLED));
#endif
#ifdef PTHREAD_ONCE_INIT
printf("PTHREAD_ONCE_INIT %llu\n", (unsigned long long)(long long)(PTHREAD_ONCE_INIT));
#endif
#ifdef PTHREAD_PRIO_INHERIT
printf("PTHREAD_PRIO_INHERIT %llu\n", (unsigned long long)(long long)(PTHREAD_PRIO_INHERIT));
#endif
#ifdef PTHREAD_PRIO_NONE
printf("PTHREAD_PRIO_NONE %llu\n", (unsigned long long)(long long)(PTHREAD_PRIO_NONE));
#endif
#ifdef PTHREAD_PRIO_PROTECT
printf("PTHREAD_PRIO_PROTECT %llu\n", (unsigned long long)(long long)(PTHREAD_PRIO_PROTECT));
#endif
#ifdef PTHREAD_PROCESS_PRIVATE
printf("PTHREAD_PROCESS_PRIVATE %llu\n", (unsigned long long)(long long)(PTHREAD_PROCESS_PRIVATE));
#endif
#ifdef PTHREAD_PROCESS_SHARED
printf("PTHREAD_PROCESS_SHARED %llu\n", (unsigned long long)(long long)(PTHREAD_PROCESS_SHARED));
#endif
#ifdef PTHREAD_STACK_MIN
printf("PTHREAD_STACK_MIN %llu\n", (unsigned long long)(long long)(PTHREAD_STACK_MIN));
#endif
#ifdef PTP_CLOCK_GETCAPS
printf("PTP_CLOCK_GETCAPS %llu\n", (unsigned long long)(long long)(PTP_CLOCK_GETCAPS));
#endif
#ifdef PTP_CLOCK_GETCAPS2
printf("PTP_CLOCK_GETCAPS2 %llu\n", (unsigned long long)(long long)(PTP_CLOCK_GETCAPS2));
#endif
#ifdef PTP_ENABLE_PPS
printf("PTP_ENABLE_PPS %llu\n", (unsigned long long)(long long)(PTP_ENABLE_PPS));
#endif
#ifdef PTP_ENABLE_PPS2
printf("PTP_ENABLE_PPS2 %llu\n", (unsigned long long)(long long)(PTP_ENABLE_PPS2));
#endif
#ifdef PTP_EXTTS_REQUEST
printf("PTP_EXTTS_REQUEST %llu\n", (unsigned long long)(long long)(PTP_EXTTS_REQUEST));
#endif
#ifdef PTP_EXTTS_REQUEST2
printf("PTP_EXTTS_REQUEST2 %llu\n", (unsigned long long)(long long)(PTP_EXTTS_REQUEST2));
#endif
#ifdef PTP_MAX_SAMPLES
printf("PTP_MAX_SAMPLES %llu\n", (unsigned long long)(long long)(PTP_MAX_SAMPLES));
#endif
#ifdef PTP_PEROUT_REQUEST
printf("PTP_PEROUT_REQUEST %llu\n", (unsigned long long)(long long)(PTP_PEROUT_REQUEST));
#endif
#ifdef PTP_PEROUT_REQUEST2
printf("PTP_PEROUT_REQUEST2 %llu\n", (unsigned long long)(long long)(PTP_PEROUT_REQUEST2));
#endif
#ifdef PTP_PF_EXTTS
printf("PTP_PF_EXTTS %llu\n", (unsigned long long)(long long)(PTP_PF_EXTTS));
#endif
#ifdef PTP_PF_NONE
printf("PTP_PF_NONE %llu\n", (unsigned long long)(long long)(PTP_PF_NONE));
#endif
#ifdef PTP_PF_PEROUT
printf("PTP_PF_PEROUT %llu\n", (unsigned long long)(long long)(PTP_PF_PEROUT));
#endif
#ifdef PTP_PF_PHYSYNC
printf("PTP_PF_PHYSYNC %llu\n", (unsigned long long)(long long)(PTP_PF_PHYSYNC));
#endif
#ifdef PTP_PIN_GETFUNC
printf("PTP_PIN_GETFUNC %llu\n", (unsigned long long)(long long)(PTP_PIN_GETFUNC));
#endif
#ifdef PTP_PIN_GETFUNC2
printf("PTP_PIN_GETFUNC2 %llu\n", (unsigned long long)(long long)(PTP_PIN_GETFUNC2));
#endif
#ifdef PTP_PIN_SETFUNC
printf("PTP_PIN_SETFUNC %llu\n", (unsigned long long)(long long)(PTP_PIN_SETFUNC));
#endif
#ifdef PTP_PIN_SETFUNC2
printf("PTP_PIN_SETFUNC2 %llu\n", (unsigned long long)(long long)(PTP_PIN_SETFUNC2));
#endif
#ifdef PTP_SYS_OFFSET
printf("PTP_SYS_OFFSET %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET));
#endif
#ifdef PTP_SYS_OFFSET2
printf("PTP_SYS_OFFSET2 %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET2));
#endif
#ifdef PTP_SYS_OFFSET_EXTENDED
printf("PTP_SYS_OFFSET_EXTENDED %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET_EXTENDED));
#endif
#ifdef PTP_SYS_OFFSET_EXTENDED2
printf("PTP_SYS_OFFSET_EXTENDED2 %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET_EXTENDED2));
#endif
#ifdef PTP_SYS_OFFSET_PRECISE
printf("PTP_SYS_OFFSET_PRECISE %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET_PRECISE));
#endif
#ifdef PTP_SYS_OFFSET_PRECISE2
printf("PTP_SYS_OFFSET_PRECISE2 %llu\n", (unsigned long long)(long long)(PTP_SYS_OFFSET_PRECISE2));
#endif
#ifdef PTRACE_ATTACH
printf("PTRACE_ATTACH %llu\n", (unsigned long long)(long long)(PTRACE_ATTACH));
#endif
#ifdef PTRACE_CONT
printf("PTRACE_CONT %llu\n", (unsigned long long)(long long)(PTRACE_CONT));
#endif
#ifdef PTRACE_DETACH
printf("PTRACE_DETACH %llu\n", (unsigned long long)(long long)(PTRACE_DETACH));
#endif
#ifdef PTRACE_EVENT_CLONE
printf("PTRACE_EVENT_CLONE %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_CLONE));
#endif
#ifdef PTRACE_EVENT_EXEC
printf("PTRACE_EVENT_EXEC %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_EXEC));
#endif
#ifdef PTRACE_EVENT_EXIT
printf("PTRACE_EVENT_EXIT %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_EXIT));
#endif
#ifdef PTRACE_EVENT_FORK
printf("PTRACE_EVENT_FORK %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_FORK));
#endif
#ifdef PTRACE_EVENT_SECCOMP
printf("PTRACE_EVENT_SECCOMP %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_SECCOMP));
#endif
#ifdef PTRACE_EVENT_STOP
printf("PTRACE_EVENT_STOP %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_STOP));
#endif
#ifdef PTRACE_EVENT_VFORK
printf("PTRACE_EVENT_VFORK %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_VFORK));
#endif
#ifdef PTRACE_EVENT_VFORK_DONE
printf("PTRACE_EVENT_VFORK_DONE %llu\n", (unsigned long long)(long long)(PTRACE_EVENT_VFORK_DONE));
#endif
#ifdef PTRACE_GETEVENTMSG
printf("PTRACE_GETEVENTMSG %llu\n", (unsigned long long)(long long)(PTRACE_GETEVENTMSG));
#endif
#ifdef PTRACE_GETFPREGS
printf("PTRACE_GETFPREGS %llu\n", (unsigned long long)(long long)(PTRACE_GETFPREGS));
#endif
#ifdef PTRACE_GETFPXREGS
printf("PTRACE_GETFPXREGS %llu\n", (unsigned long long)(long long)(PTRACE_GETFPXREGS));
#endif
#ifdef PTRACE_GETREGS
printf("PTRACE_GETREGS %llu\n", (unsigned long long)(long long)(PTRACE_GETREGS));
#endif
#ifdef PTRACE_GETREGSET
printf("PTRACE_GETREGSET %llu\n", (unsigned long long)(long long)(PTRACE_GETREGSET));
#endif
#ifdef PTRACE_GETSIGINFO
printf("PTRACE_GETSIGINFO %llu\n", (unsigned long long)(long long)(PTRACE_GETSIGINFO));
#endif
#ifdef PTRACE_GETSIGMASK
printf("PTRACE_GETSIGMASK %llu\n", (unsigned long long)(long long)(PTRACE_GETSIGMASK));
#endif
#ifdef PTRACE_GET_SYSCALL_INFO
printf("PTRACE_GET_SYSCALL_INFO %llu\n", (unsigned long long)(long long)(PTRACE_GET_SYSCALL_INFO));
#endif
#ifdef PTRACE_GET_SYSCALL_USER_DISPATCH_CONFIG
printf("PTRACE_GET_SYSCALL_USER_DISPATCH_CONFIG %llu\n", (unsigned long long)(long long)(PTRACE_GET_SYSCALL_USER_DISPATCH_CONFIG));
#endif
#ifdef PTRACE_INTERRUPT
printf("PTRACE_INTERRUPT %llu\n", (unsigned long long)(long long)(PTRACE_INTERRUPT));
#endif
#ifdef PTRACE_KILL
printf("PTRACE_KILL %llu\n", (unsigned long long)(long long)(PTRACE_KILL));
#endif
#ifdef PTRACE_LISTEN
printf("PTRACE_LISTEN %llu\n", (unsigned long long)(long long)(PTRACE_LISTEN));
#endif
#ifdef PTRACE_O_EXITKILL
printf("PTRACE_O_EXITKILL %llu\n", (unsigned long long)(long long)(PTRACE_O_EXITKILL));
#endif
#ifdef PTRACE_O_MASK
printf("PTRACE_O_MASK %llu\n", (unsigned long long)(long long)(PTRACE_O_MASK));
#endif
#ifdef PTRACE_O_SUSPEND_SECCOMP
printf("PTRACE_O_SUSPEND_SECCOMP %llu\n", (unsigned long long)(long long)(PTRACE_O_SUSPEND_SECCOMP));
#endif
#ifdef PTRACE_O_TRACECLONE
printf("PTRACE_O_TRACECLONE %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACECLONE));
#endif
#ifdef PTRACE_O_TRACEEXEC
printf("PTRACE_O_TRACEEXEC %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACEEXEC));
#endif
#ifdef PTRACE_O_TRACEEXIT
printf("PTRACE_O_TRACEEXIT %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACEEXIT));
#endif
#ifdef PTRACE_O_TRACEFORK
printf("PTRACE_O_TRACEFORK %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACEFORK));
#endif
#ifdef PTRACE_O_TRACESECCOMP
printf("PTRACE_O_TRACESECCOMP %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACESECCOMP));
#endif
#ifdef PTRACE_O_TRACESYSGOOD
printf("PTRACE_O_TRACESYSGOOD %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACESYSGOOD));
#endif
#ifdef PTRACE_O_TRACEVFORK
printf("PTRACE_O_TRACEVFORK %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACEVFORK));
#endif
#ifdef PTRACE_O_TRACEVFORKDONE
printf("PTRACE_O_TRACEVFORKDONE %llu\n", (unsigned long long)(long long)(PTRACE_O_TRACEVFORKDONE));
#endif
#ifdef PTRACE_PEEKDATA
printf("PTRACE_PEEKDATA %llu\n", (unsigned long long)(long long)(PTRACE_PEEKDATA));
#endif
#ifdef PTRACE_PEEKSIGINFO
printf("PTRACE_PEEKSIGINFO %llu\n", (unsigned long long)(long long)(PTRACE_PEEKSIGINFO));
#endif
#ifdef PTRACE_PEEKTEXT
printf("PTRACE_PEEKTEXT %llu\n", (unsigned long long)(long long)(PTRACE_PEEKTEXT));
#endif
#ifdef PTRACE_PEEKUSER
printf("PTRACE_PEEKUSER %llu\n", (unsigned long long)(long long)(PTRACE_PEEKUSER));
#endif
#ifdef PTRACE_POKEDATA
printf("PTRACE_POKEDATA %llu\n", (unsigned long long)(long long)(PTRACE_POKEDATA));
#endif
#ifdef PTRACE_POKETEXT
printf("PTRACE_POKETEXT %llu\n", (unsigned long long)(long long)(PTRACE_POKETEXT));
#endif
#ifdef PTRACE_POKEUSER
printf("PTRACE_POKEUSER %llu\n", (unsigned long long)(long long)(PTRACE_POKEUSER));
#endif
#ifdef PTRACE_SEIZE
printf("PTRACE_SEIZE %llu\n", (unsigned long long)(long long)(PTRACE_SEIZE));
#endif
#ifdef PTRACE_SETFPREGS
printf("PTRACE_SETFPREGS %llu\n", (unsigned long long)(long long)(PTRACE_SETFPREGS));
#endif
#ifdef PTRACE_SETFPXREGS
printf("PTRACE_SETFPXREGS %llu\n", (unsigned long long)(long long)(PTRACE_SETFPXREGS));
#endif
#ifdef PTRACE_SETOPTIONS
printf("PTRACE_SETOPTIONS %llu\n", (unsigned long long)(long long)(PTRACE_SETOPTIONS));
#endif
#ifdef PTRACE_SETREGS
printf("PTRACE_SETREGS %llu\n", (unsigned long long)(long long)(PTRACE_SETREGS));
#endif
#ifdef PTRACE_SETREGSET
printf("PTRACE_SETREGSET %llu\n", (unsigned long long)(long long)(PTRACE_SETREGSET));
#endif
#ifdef PTRACE_SETSIGINFO
printf("PTRACE_SETSIGINFO %llu\n", (unsigned long long)(long long)(PTRACE_SETSIGINFO));
#endif
#ifdef PTRACE_SETSIGMASK
printf("PTRACE_SETSIGMASK %llu\n", (unsigned long long)(long long)(PTRACE_SETSIGMASK));
#endif
#ifdef PTRACE_SET_SYSCALL_USER_DISPATCH_CONFIG
printf("PTRACE_SET_SYSCALL_USER_DISPATCH_CONFIG %llu\n", (unsigned long long)(long long)(PTRACE_SET_SYSCALL_USER_DISPATCH_CONFIG));
#endif
#ifdef PTRACE_SINGLESTEP
printf("PTRACE_SINGLESTEP %llu\n", (unsigned long long)(long long)(PTRACE_SINGLESTEP));
#endif
#ifdef PTRACE_SYSCALL
printf("PTRACE_SYSCALL %llu\n", (unsigned long long)(long long)(PTRACE_SYSCALL));
#endif
#ifdef PTRACE_SYSCALL_INFO_ENTRY
printf("PTRACE_SYSCALL_INFO_ENTRY %llu\n", (unsigned long long)(long long)(PTRACE_SYSCALL_INFO_ENTRY));
#endif
#ifdef PTRACE_SYSCALL_INFO_EXIT
printf("PTRACE_SYSCALL_INFO_EXIT %llu\n", (unsigned long long)(long long)(PTRACE_SYSCALL_INFO_EXIT));
#endif
#ifdef PTRACE_SYSCALL_INFO_NONE
printf("PTRACE_SYSCALL_INFO_NONE %llu\n", (unsigned long long)(long long)(PTRACE_SYSCALL_INFO_NONE));
#endif
#ifdef PTRACE_SYSCALL_INFO_SECCOMP
printf("PTRACE_SYSCALL_INFO_SECCOMP %llu\n", (unsigned long long)(long long)(PTRACE_SYSCALL_INFO_SECCOMP));
#endif
#ifdef PTRACE_TRACEME
printf("PTRACE_TRACEME %llu\n", (unsigned long long)(long long)(PTRACE_TRACEME));
#endif
#ifdef PT_DYNAMIC
printf("PT_DYNAMIC %llu\n", (unsigned long long)(long long)(PT_DYNAMIC));
#endif
#ifdef PT_GNU_EH_FRAME
printf("PT_GNU_EH_FRAME %llu\n", (unsigned long long)(long long)(PT_GNU_EH_FRAME));
#endif
#ifdef PT_GNU_RELRO
printf("PT_GNU_RELRO %llu\n", (unsigned long long)(long long)(PT_GNU_RELRO));
#endif
#ifdef PT_GNU_STACK
printf("PT_GNU_STACK %llu\n", (unsigned long long)(long long)(PT_GNU_STACK));
#endif
#ifdef PT_HIOS
printf("PT_HIOS %llu\n", (unsigned long long)(long long)(PT_HIOS));
#endif
#ifdef PT_HIPROC
printf("PT_HIPROC %llu\n", (unsigned long long)(long long)(PT_HIPROC));
#endif
#ifdef PT_HISUNW
printf("PT_HISUNW %llu\n", (unsigned long long)(long long)(PT_HISUNW));
#endif
#ifdef PT_INTERP
printf("PT_INTERP %llu\n", (unsigned long long)(long long)(PT_INTERP));
#endif
#ifdef PT_LOAD
printf("PT_LOAD %llu\n", (unsigned long long)(long long)(PT_LOAD));
#endif
#ifdef PT_LOOS
printf("PT_LOOS %llu\n", (unsigned long long)(long long)(PT_LOOS));
#endif
#ifdef PT_LOPROC
printf("PT_LOPROC %llu\n", (unsigned long long)(long long)(PT_LOPROC));
#endif
#ifdef PT_LOSUNW
printf("PT_LOSUNW %llu\n", (unsigned long long)(long long)(PT_LOSUNW));
#endif
#ifdef PT_NOTE
printf("PT_NOTE %llu\n", (unsigned long long)(long long)(PT_NOTE));
#endif
#ifdef PT_NULL
printf("PT_NULL %llu\n", (unsigned long long)(long long)(PT_NULL));
#endif
#ifdef PT_NUM
printf("PT_NUM %llu\n", (unsigned long long)(long long)(PT_NUM));
#endif
#ifdef PT_PHDR
printf("PT_PHDR %llu\n", (unsigned long long)(long long)(PT_PHDR));
#endif
#ifdef PT_SHLIB
printf("PT_SHLIB %llu\n", (unsigned long long)(long long)(PT_SHLIB));
#endif
#ifdef PT_SUNWBSS
printf("PT_SUNWBSS %llu\n", (unsigned long long)(long long)(PT_SUNWBSS));
#endif
#ifdef PT_SUNWSTACK
printf("PT_SUNWSTACK %llu\n", (unsigned long long)(long long)(PT_SUNWSTACK));
#endif
#ifdef PT_TLS
printf("PT_TLS %llu\n", (unsigned long long)(long long)(PT_TLS));
#endif
#ifdef P_ALL
printf("P_ALL %llu\n", (unsigned long long)(long long)(P_ALL));
#endif
#ifdef P_PGID
printf("P_PGID %llu\n", (unsigned long long)(long long)(P_PGID));
#endif
#ifdef P_PID
printf("P_PID %llu\n", (unsigned long long)(long long)(P_PID));
#endif
#ifdef P_PIDFD
printf("P_PIDFD %llu\n", (unsigned long long)(long long)(P_PIDFD));
#endif
#ifdef QFMT_VFS_OLD
printf("QFMT_VFS_OLD %llu\n", (unsigned long long)(long long)(QFMT_VFS_OLD));
#endif
#ifdef QFMT_VFS_V0
printf("QFMT_VFS_V0 %llu\n", (unsigned long long)(long long)(QFMT_VFS_V0));
#endif
#ifdef QFMT_VFS_V1
printf("QFMT_VFS_V1 %llu\n", (unsigned long long)(long long)(QFMT_VFS_V1));
#endif
#ifdef QIF_ALL
printf("QIF_ALL %llu\n", (unsigned long long)(long long)(QIF_ALL));
#endif
#ifdef QIF_BLIMITS
printf("QIF_BLIMITS %llu\n", (unsigned long long)(long long)(QIF_BLIMITS));
#endif
#ifdef QIF_BTIME
printf("QIF_BTIME %llu\n", (unsigned long long)(long long)(QIF_BTIME));
#endif
#ifdef QIF_ILIMITS
printf("QIF_ILIMITS %llu\n", (unsigned long long)(long long)(QIF_ILIMITS));
#endif
#ifdef QIF_INODES
printf("QIF_INODES %llu\n", (unsigned long long)(long long)(QIF_INODES));
#endif
#ifdef QIF_ITIME
printf("QIF_ITIME %llu\n", (unsigned long long)(long long)(QIF_ITIME));
#endif
#ifdef QIF_LIMITS
printf("QIF_LIMITS %llu\n", (unsigned long long)(long long)(QIF_LIMITS));
#endif
#ifdef QIF_SPACE
printf("QIF_SPACE %llu\n", (unsigned long long)(long long)(QIF_SPACE));
#endif
#ifdef QIF_TIMES
printf("QIF_TIMES %llu\n", (unsigned long long)(long long)(QIF_TIMES));
#endif
#ifdef QIF_USAGE
printf("QIF_USAGE %llu\n", (unsigned long long)(long long)(QIF_USAGE));
#endif
#ifdef QNX4_SUPER_MAGIC
printf("QNX4_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(QNX4_SUPER_MAGIC));
#endif
#ifdef QNX6_SUPER_MAGIC
printf("QNX6_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(QNX6_SUPER_MAGIC));
#endif
#ifdef Q_GETFMT
printf("Q_GETFMT %llu\n", (unsigned long long)(long long)(Q_GETFMT));
#endif
#ifdef Q_GETINFO
printf("Q_GETINFO %llu\n", (unsigned long long)(long long)(Q_GETINFO));
#endif
#ifdef Q_GETQUOTA
printf("Q_GETQUOTA %llu\n", (unsigned long long)(long long)(Q_GETQUOTA));
#endif
#ifdef Q_QUOTAOFF
printf("Q_QUOTAOFF %llu\n", (unsigned long long)(long long)(Q_QUOTAOFF));
#endif
#ifdef Q_QUOTAON
printf("Q_QUOTAON %llu\n", (unsigned long long)(long long)(Q_QUOTAON));
#endif
#ifdef Q_SETINFO
printf("Q_SETINFO %llu\n", (unsigned long long)(long long)(Q_SETINFO));
#endif
#ifdef Q_SETQUOTA
printf("Q_SETQUOTA %llu\n", (unsigned long long)(long long)(Q_SETQUOTA));
#endif
#ifdef Q_SYNC
printf("Q_SYNC %llu\n", (unsigned long long)(long long)(Q_SYNC));
#endif
#ifdef RADIXCHAR
printf("RADIXCHAR %llu\n", (unsigned long long)(long long)(RADIXCHAR));
#endif
#ifdef RAND_MAX
printf("RAND_MAX %llu\n", (unsigned long long)(long long)(RAND_MAX));
#endif
#ifdef RB_AUTOBOOT
printf("RB_AUTOBOOT %llu\n", (unsigned long long)(long long)(RB_AUTOBOOT));
#endif
#ifdef RB_DISABLE_CAD
printf("RB_DISABLE_CAD %llu\n", (unsigned long long)(long long)(RB_DISABLE_CAD));
#endif
#ifdef RB_ENABLE_CAD
printf("RB_ENABLE_CAD %llu\n", (unsigned long long)(long long)(RB_ENABLE_CAD));
#endif
#ifdef RB_HALT_SYSTEM
printf("RB_HALT_SYSTEM %llu\n", (unsigned long long)(long long)(RB_HALT_SYSTEM));
#endif
#ifdef RB_KEXEC
printf("RB_KEXEC %llu\n", (unsigned long long)(long long)(RB_KEXEC));
#endif
#ifdef RB_POWER_OFF
printf("RB_POWER_OFF %llu\n", (unsigned long long)(long long)(RB_POWER_OFF));
#endif
#ifdef RB_SW_SUSPEND
printf("RB_SW_SUSPEND %llu\n", (unsigned long long)(long long)(RB_SW_SUSPEND));
#endif
#ifdef RDTGROUP_SUPER_MAGIC
printf("RDTGROUP_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(RDTGROUP_SUPER_MAGIC));
#endif
#ifdef READ_IMPLIES_EXEC
printf("READ_IMPLIES_EXEC %llu\n", (unsigned long long)(long long)(READ_IMPLIES_EXEC));
#endif
#ifdef REG_A0
printf("REG_A0 %llu\n", (unsigned long long)(long long)(REG_A0));
#endif
#ifdef REG_BADBR
printf("REG_BADBR %llu\n", (unsigned long long)(long long)(REG_BADBR));
#endif
#ifdef REG_BADPAT
printf("REG_BADPAT %llu\n", (unsigned long long)(long long)(REG_BADPAT));
#endif
#ifdef REG_BADRPT
printf("REG_BADRPT %llu\n", (unsigned long long)(long long)(REG_BADRPT));
#endif
#ifdef REG_EBRACE
printf("REG_EBRACE %llu\n", (unsigned long long)(long long)(REG_EBRACE));
#endif
#ifdef REG_EBRACK
printf("REG_EBRACK %llu\n", (unsigned long long)(long long)(REG_EBRACK));
#endif
#ifdef REG_ECOLLATE
printf("REG_ECOLLATE %llu\n", (unsigned long long)(long long)(REG_ECOLLATE));
#endif
#ifdef REG_ECTYPE
printf("REG_ECTYPE %llu\n", (unsigned long long)(long long)(REG_ECTYPE));
#endif
#ifdef REG_EEND
printf("REG_EEND %llu\n", (unsigned long long)(long long)(REG_EEND));
#endif
#ifdef REG_EESCAPE
printf("REG_EESCAPE %llu\n", (unsigned long long)(long long)(REG_EESCAPE));
#endif
#ifdef REG_ENOSYS
printf("REG_ENOSYS %llu\n", (unsigned long long)(long long)(REG_ENOSYS));
#endif
#ifdef REG_EPAREN
printf("REG_EPAREN %llu\n", (unsigned long long)(long long)(REG_EPAREN));
#endif
#ifdef REG_ERANGE
printf("REG_ERANGE %llu\n", (unsigned long long)(long long)(REG_ERANGE));
#endif
#ifdef REG_ERPAREN
printf("REG_ERPAREN %llu\n", (unsigned long long)(long long)(REG_ERPAREN));
#endif
#ifdef REG_ESIZE
printf("REG_ESIZE %llu\n", (unsigned long long)(long long)(REG_ESIZE));
#endif
#ifdef REG_ESPACE
printf("REG_ESPACE %llu\n", (unsigned long long)(long long)(REG_ESPACE));
#endif
#ifdef REG_ESUBREG
printf("REG_ESUBREG %llu\n", (unsigned long long)(long long)(REG_ESUBREG));
#endif
#ifdef REG_EXTENDED
printf("REG_EXTENDED %llu\n", (unsigned long long)(long long)(REG_EXTENDED));
#endif
#ifdef REG_ICASE
printf("REG_ICASE %llu\n", (unsigned long long)(long long)(REG_ICASE));
#endif
#ifdef REG_NARGS
printf("REG_NARGS %llu\n", (unsigned long long)(long long)(REG_NARGS));
#endif
#ifdef REG_NEWLINE
printf("REG_NEWLINE %llu\n", (unsigned long long)(long long)(REG_NEWLINE));
#endif
#ifdef REG_NOMATCH
printf("REG_NOMATCH %llu\n", (unsigned long long)(long long)(REG_NOMATCH));
#endif
#ifdef REG_NOSUB
printf("REG_NOSUB %llu\n", (unsigned long long)(long long)(REG_NOSUB));
#endif
#ifdef REG_NOTBOL
printf("REG_NOTBOL %llu\n", (unsigned long long)(long long)(REG_NOTBOL));
#endif
#ifdef REG_NOTEOL
printf("REG_NOTEOL %llu\n", (unsigned long long)(long long)(REG_NOTEOL));
#endif
#ifdef REG_PC
printf("REG_PC %llu\n", (unsigned long long)(long long)(REG_PC));
#endif
#ifdef REG_RA
printf("REG_RA %llu\n", (unsigned long long)(long long)(REG_RA));
#endif
#ifdef REG_S0
printf("REG_S0 %llu\n", (unsigned long long)(long long)(REG_S0));
#endif
#ifdef REG_S1
printf("REG_S1 %llu\n", (unsigned long long)(long long)(REG_S1));
#endif
#ifdef REG_S2
printf("REG_S2 %llu\n", (unsigned long long)(long long)(REG_S2));
#endif
#ifdef REG_SP
printf("REG_SP %llu\n", (unsigned long long)(long long)(REG_SP));
#endif
#ifdef REG_STARTEND
printf("REG_STARTEND %llu\n", (unsigned long long)(long long)(REG_STARTEND));
#endif
#ifdef REG_TP
printf("REG_TP %llu\n", (unsigned long long)(long long)(REG_TP));
#endif
#ifdef REISERFS_SUPER_MAGIC
printf("REISERFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(REISERFS_SUPER_MAGIC));
#endif
#ifdef REL_CNT
printf("REL_CNT %llu\n", (unsigned long long)(long long)(REL_CNT));
#endif
#ifdef REL_MAX
printf("REL_MAX %llu\n", (unsigned long long)(long long)(REL_MAX));
#endif
#ifdef RENAME_EXCHANGE
printf("RENAME_EXCHANGE %llu\n", (unsigned long long)(long long)(RENAME_EXCHANGE));
#endif
#ifdef RENAME_NOREPLACE
printf("RENAME_NOREPLACE %llu\n", (unsigned long long)(long long)(RENAME_NOREPLACE));
#endif
#ifdef RENAME_WHITEOUT
printf("RENAME_WHITEOUT %llu\n", (unsigned long long)(long long)(RENAME_WHITEOUT));
#endif
#ifdef REP_CNT
printf("REP_CNT %llu\n", (unsigned long long)(long long)(REP_CNT));
#endif
#ifdef REP_MAX
printf("REP_MAX %llu\n", (unsigned long long)(long long)(REP_MAX));
#endif
#ifdef RESOLVE_BENEATH
printf("RESOLVE_BENEATH %llu\n", (unsigned long long)(long long)(RESOLVE_BENEATH));
#endif
#ifdef RESOLVE_CACHED
printf("RESOLVE_CACHED %llu\n", (unsigned long long)(long long)(RESOLVE_CACHED));
#endif
#ifdef RESOLVE_IN_ROOT
printf("RESOLVE_IN_ROOT %llu\n", (unsigned long long)(long long)(RESOLVE_IN_ROOT));
#endif
#ifdef RESOLVE_NO_MAGICLINKS
printf("RESOLVE_NO_MAGICLINKS %llu\n", (unsigned long long)(long long)(RESOLVE_NO_MAGICLINKS));
#endif
#ifdef RESOLVE_NO_SYMLINKS
printf("RESOLVE_NO_SYMLINKS %llu\n", (unsigned long long)(long long)(RESOLVE_NO_SYMLINKS));
#endif
#ifdef RESOLVE_NO_XDEV
printf("RESOLVE_NO_XDEV %llu\n", (unsigned long long)(long long)(RESOLVE_NO_XDEV));
#endif
#ifdef RLIM64_INFINITY
printf("RLIM64_INFINITY %llu\n", (unsigned long long)(long long)(RLIM64_INFINITY));
#endif
#ifdef RLIMIT_AS
printf("RLIMIT_AS %llu\n", (unsigned long long)(long long)(RLIMIT_AS));
#endif
#ifdef RLIMIT_CORE
printf("RLIMIT_CORE %llu\n", (unsigned long long)(long long)(RLIMIT_CORE));
#endif
#ifdef RLIMIT_CPU
printf("RLIMIT_CPU %llu\n", (unsigned long long)(long long)(RLIMIT_CPU));
#endif
#ifdef RLIMIT_DATA
printf("RLIMIT_DATA %llu\n", (unsigned long long)(long long)(RLIMIT_DATA));
#endif
#ifdef RLIMIT_FSIZE
printf("RLIMIT_FSIZE %llu\n", (unsigned long long)(long long)(RLIMIT_FSIZE));
#endif
#ifdef RLIMIT_LOCKS
printf("RLIMIT_LOCKS %llu\n", (unsigned long long)(long long)(RLIMIT_LOCKS));
#endif
#ifdef RLIMIT_MEMLOCK
printf("RLIMIT_MEMLOCK %llu\n", (unsigned long long)(long long)(RLIMIT_MEMLOCK));
#endif
#ifdef RLIMIT_MSGQUEUE
printf("RLIMIT_MSGQUEUE %llu\n", (unsigned long long)(long long)(RLIMIT_MSGQUEUE));
#endif
#ifdef RLIMIT_NICE
printf("RLIMIT_NICE %llu\n", (unsigned long long)(long long)(RLIMIT_NICE));
#endif
#ifdef RLIMIT_NLIMITS
printf("RLIMIT_NLIMITS %llu\n", (unsigned long long)(long long)(RLIMIT_NLIMITS));
#endif
#ifdef RLIMIT_NOFILE
printf("RLIMIT_NOFILE %llu\n", (unsigned long long)(long long)(RLIMIT_NOFILE));
#endif
#ifdef RLIMIT_NPROC
printf("RLIMIT_NPROC %llu\n", (unsigned long long)(long long)(RLIMIT_NPROC));
#endif
#ifdef RLIMIT_RSS
printf("RLIMIT_RSS %llu\n", (unsigned long long)(long long)(RLIMIT_RSS));
#endif
#ifdef RLIMIT_RTPRIO
printf("RLIMIT_RTPRIO %llu\n", (unsigned long long)(long long)(RLIMIT_RTPRIO));
#endif
#ifdef RLIMIT_RTTIME
printf("RLIMIT_RTTIME %llu\n", (unsigned long long)(long long)(RLIMIT_RTTIME));
#endif
#ifdef RLIMIT_SIGPENDING
printf("RLIMIT_SIGPENDING %llu\n", (unsigned long long)(long long)(RLIMIT_SIGPENDING));
#endif
#ifdef RLIMIT_STACK
printf("RLIMIT_STACK %llu\n", (unsigned long long)(long long)(RLIMIT_STACK));
#endif
#ifdef RLIM_INFINITY
printf("RLIM_INFINITY %llu\n", (unsigned long long)(long long)(RLIM_INFINITY));
#endif
#ifdef RLIM_NLIMITS
printf("RLIM_NLIMITS %llu\n", (unsigned long long)(long long)(RLIM_NLIMITS));
#endif
#ifdef RLIM_SAVED_CUR
printf("RLIM_SAVED_CUR %llu\n", (unsigned long long)(long long)(RLIM_SAVED_CUR));
#endif
#ifdef RLIM_SAVED_MAX
printf("RLIM_SAVED_MAX %llu\n", (unsigned long long)(long long)(RLIM_SAVED_MAX));
#endif
#ifdef RTA_CACHEINFO
printf("RTA_CACHEINFO %llu\n", (unsigned long long)(long long)(RTA_CACHEINFO));
#endif
#ifdef RTA_DST
printf("RTA_DST %llu\n", (unsigned long long)(long long)(RTA_DST));
#endif
#ifdef RTA_ENCAP
printf("RTA_ENCAP %llu\n", (unsigned long long)(long long)(RTA_ENCAP));
#endif
#ifdef RTA_ENCAP_TYPE
printf("RTA_ENCAP_TYPE %llu\n", (unsigned long long)(long long)(RTA_ENCAP_TYPE));
#endif
#ifdef RTA_EXPIRES
printf("RTA_EXPIRES %llu\n", (unsigned long long)(long long)(RTA_EXPIRES));
#endif
#ifdef RTA_FLOW
printf("RTA_FLOW %llu\n", (unsigned long long)(long long)(RTA_FLOW));
#endif
#ifdef RTA_GATEWAY
printf("RTA_GATEWAY %llu\n", (unsigned long long)(long long)(RTA_GATEWAY));
#endif
#ifdef RTA_IIF
printf("RTA_IIF %llu\n", (unsigned long long)(long long)(RTA_IIF));
#endif
#ifdef RTA_MARK
printf("RTA_MARK %llu\n", (unsigned long long)(long long)(RTA_MARK));
#endif
#ifdef RTA_METRICS
printf("RTA_METRICS %llu\n", (unsigned long long)(long long)(RTA_METRICS));
#endif
#ifdef RTA_MFC_STATS
printf("RTA_MFC_STATS %llu\n", (unsigned long long)(long long)(RTA_MFC_STATS));
#endif
#ifdef RTA_MP_ALGO
printf("RTA_MP_ALGO %llu\n", (unsigned long long)(long long)(RTA_MP_ALGO));
#endif
#ifdef RTA_MULTIPATH
printf("RTA_MULTIPATH %llu\n", (unsigned long long)(long long)(RTA_MULTIPATH));
#endif
#ifdef RTA_NEWDST
printf("RTA_NEWDST %llu\n", (unsigned long long)(long long)(RTA_NEWDST));
#endif
#ifdef RTA_OIF
printf("RTA_OIF %llu\n", (unsigned long long)(long long)(RTA_OIF));
#endif
#ifdef RTA_PAD
printf("RTA_PAD %llu\n", (unsigned long long)(long long)(RTA_PAD));
#endif
#ifdef RTA_PREF
printf("RTA_PREF %llu\n", (unsigned long long)(long long)(RTA_PREF));
#endif
#ifdef RTA_PREFSRC
printf("RTA_PREFSRC %llu\n", (unsigned long long)(long long)(RTA_PREFSRC));
#endif
#ifdef RTA_PRIORITY
printf("RTA_PRIORITY %llu\n", (unsigned long long)(long long)(RTA_PRIORITY));
#endif
#ifdef RTA_PROTOINFO
printf("RTA_PROTOINFO %llu\n", (unsigned long long)(long long)(RTA_PROTOINFO));
#endif
#ifdef RTA_SESSION
printf("RTA_SESSION %llu\n", (unsigned long long)(long long)(RTA_SESSION));
#endif
#ifdef RTA_SRC
printf("RTA_SRC %llu\n", (unsigned long long)(long long)(RTA_SRC));
#endif
#ifdef RTA_TABLE
printf("RTA_TABLE %llu\n", (unsigned long long)(long long)(RTA_TABLE));
#endif
#ifdef RTA_TTL_PROPAGATE
printf("RTA_TTL_PROPAGATE %llu\n", (unsigned long long)(long long)(RTA_TTL_PROPAGATE));
#endif
#ifdef RTA_UID
printf("RTA_UID %llu\n", (unsigned long long)(long long)(RTA_UID));
#endif
#ifdef RTA_UNSPEC
printf("RTA_UNSPEC %llu\n", (unsigned long long)(long long)(RTA_UNSPEC));
#endif
#ifdef RTA_VIA
printf("RTA_VIA %llu\n", (unsigned long long)(long long)(RTA_VIA));
#endif
#ifdef RTCF_DIRECTSRC
printf("RTCF_DIRECTSRC %llu\n", (unsigned long long)(long long)(RTCF_DIRECTSRC));
#endif
#ifdef RTCF_DOREDIRECT
printf("RTCF_DOREDIRECT %llu\n", (unsigned long long)(long long)(RTCF_DOREDIRECT));
#endif
#ifdef RTCF_LOG
printf("RTCF_LOG %llu\n", (unsigned long long)(long long)(RTCF_LOG));
#endif
#ifdef RTCF_MASQ
printf("RTCF_MASQ %llu\n", (unsigned long long)(long long)(RTCF_MASQ));
#endif
#ifdef RTCF_NAT
printf("RTCF_NAT %llu\n", (unsigned long long)(long long)(RTCF_NAT));
#endif
#ifdef RTCF_VALVE
printf("RTCF_VALVE %llu\n", (unsigned long long)(long long)(RTCF_VALVE));
#endif
#ifdef RTEXT_FILTER_BRVLAN
printf("RTEXT_FILTER_BRVLAN %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_BRVLAN));
#endif
#ifdef RTEXT_FILTER_BRVLAN_COMPRESSED
printf("RTEXT_FILTER_BRVLAN_COMPRESSED %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_BRVLAN_COMPRESSED));
#endif
#ifdef RTEXT_FILTER_CFM_CONFIG
printf("RTEXT_FILTER_CFM_CONFIG %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_CFM_CONFIG));
#endif
#ifdef RTEXT_FILTER_CFM_STATUS
printf("RTEXT_FILTER_CFM_STATUS %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_CFM_STATUS));
#endif
#ifdef RTEXT_FILTER_MRP
printf("RTEXT_FILTER_MRP %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_MRP));
#endif
#ifdef RTEXT_FILTER_SKIP_STATS
printf("RTEXT_FILTER_SKIP_STATS %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_SKIP_STATS));
#endif
#ifdef RTEXT_FILTER_VF
printf("RTEXT_FILTER_VF %llu\n", (unsigned long long)(long long)(RTEXT_FILTER_VF));
#endif
#ifdef RTF_ADDRCLASSMASK
printf("RTF_ADDRCLASSMASK %llu\n", (unsigned long long)(long long)(RTF_ADDRCLASSMASK));
#endif
#ifdef RTF_ADDRCONF
printf("RTF_ADDRCONF %llu\n", (unsigned long long)(long long)(RTF_ADDRCONF));
#endif
#ifdef RTF_ALLONLINK
printf("RTF_ALLONLINK %llu\n", (unsigned long long)(long long)(RTF_ALLONLINK));
#endif
#ifdef RTF_BROADCAST
printf("RTF_BROADCAST %llu\n", (unsigned long long)(long long)(RTF_BROADCAST));
#endif
#ifdef RTF_CACHE
printf("RTF_CACHE %llu\n", (unsigned long long)(long long)(RTF_CACHE));
#endif
#ifdef RTF_DEFAULT
printf("RTF_DEFAULT %llu\n", (unsigned long long)(long long)(RTF_DEFAULT));
#endif
#ifdef RTF_DYNAMIC
printf("RTF_DYNAMIC %llu\n", (unsigned long long)(long long)(RTF_DYNAMIC));
#endif
#ifdef RTF_FLOW
printf("RTF_FLOW %llu\n", (unsigned long long)(long long)(RTF_FLOW));
#endif
#ifdef RTF_GATEWAY
printf("RTF_GATEWAY %llu\n", (unsigned long long)(long long)(RTF_GATEWAY));
#endif
#ifdef RTF_HOST
printf("RTF_HOST %llu\n", (unsigned long long)(long long)(RTF_HOST));
#endif
#ifdef RTF_INTERFACE
printf("RTF_INTERFACE %llu\n", (unsigned long long)(long long)(RTF_INTERFACE));
#endif
#ifdef RTF_IRTT
printf("RTF_IRTT %llu\n", (unsigned long long)(long long)(RTF_IRTT));
#endif
#ifdef RTF_LINKRT
printf("RTF_LINKRT %llu\n", (unsigned long long)(long long)(RTF_LINKRT));
#endif
#ifdef RTF_LOCAL
printf("RTF_LOCAL %llu\n", (unsigned long long)(long long)(RTF_LOCAL));
#endif
#ifdef RTF_MODIFIED
printf("RTF_MODIFIED %llu\n", (unsigned long long)(long long)(RTF_MODIFIED));
#endif
#ifdef RTF_MSS
printf("RTF_MSS %llu\n", (unsigned long long)(long long)(RTF_MSS));
#endif
#ifdef RTF_MTU
printf("RTF_MTU %llu\n", (unsigned long long)(long long)(RTF_MTU));
#endif
#ifdef RTF_MULTICAST
printf("RTF_MULTICAST %llu\n", (unsigned long long)(long long)(RTF_MULTICAST));
#endif
#ifdef RTF_NAT
printf("RTF_NAT %llu\n", (unsigned long long)(long long)(RTF_NAT));
#endif
#ifdef RTF_NOFORWARD
printf("RTF_NOFORWARD %llu\n", (unsigned long long)(long long)(RTF_NOFORWARD));
#endif
#ifdef RTF_NONEXTHOP
printf("RTF_NONEXTHOP %llu\n", (unsigned long long)(long long)(RTF_NONEXTHOP));
#endif
#ifdef RTF_NOPMTUDISC
printf("RTF_NOPMTUDISC %llu\n", (unsigned long long)(long long)(RTF_NOPMTUDISC));
#endif
#ifdef RTF_POLICY
printf("RTF_POLICY %llu\n", (unsigned long long)(long long)(RTF_POLICY));
#endif
#ifdef RTF_REINSTATE
printf("RTF_REINSTATE %llu\n", (unsigned long long)(long long)(RTF_REINSTATE));
#endif
#ifdef RTF_REJECT
printf("RTF_REJECT %llu\n", (unsigned long long)(long long)(RTF_REJECT));
#endif
#ifdef RTF_STATIC
printf("RTF_STATIC %llu\n", (unsigned long long)(long long)(RTF_STATIC));
#endif
#ifdef RTF_THROW
printf("RTF_THROW %llu\n", (unsigned long long)(long long)(RTF_THROW));
#endif
#ifdef RTF_UP
printf("RTF_UP %llu\n", (unsigned long long)(long long)(RTF_UP));
#endif
#ifdef RTF_WINDOW
printf("RTF_WINDOW %llu\n", (unsigned long long)(long long)(RTF_WINDOW));
#endif
#ifdef RTF_XRESOLVE
printf("RTF_XRESOLVE %llu\n", (unsigned long long)(long long)(RTF_XRESOLVE));
#endif
#ifdef RTLD_DEEPBIND
printf("RTLD_DEEPBIND %llu\n", (unsigned long long)(long long)(RTLD_DEEPBIND));
#endif
#ifdef RTLD_DI_CONFIGADDR
printf("RTLD_DI_CONFIGADDR %llu\n", (unsigned long long)(long long)(RTLD_DI_CONFIGADDR));
#endif
#ifdef RTLD_DI_LINKMAP
printf("RTLD_DI_LINKMAP %llu\n", (unsigned long long)(long long)(RTLD_DI_LINKMAP));
#endif
#ifdef RTLD_DI_LMID
printf("RTLD_DI_LMID %llu\n", (unsigned long long)(long long)(RTLD_DI_LMID));
#endif
#ifdef RTLD_DI_ORIGIN
printf("RTLD_DI_ORIGIN %llu\n", (unsigned long long)(long long)(RTLD_DI_ORIGIN));
#endif
#ifdef RTLD_DI_PROFILENAME
printf("RTLD_DI_PROFILENAME %llu\n", (unsigned long long)(long long)(RTLD_DI_PROFILENAME));
#endif
#ifdef RTLD_DI_PROFILEOUT
printf("RTLD_DI_PROFILEOUT %llu\n", (unsigned long long)(long long)(RTLD_DI_PROFILEOUT));
#endif
#ifdef RTLD_DI_SERINFO
printf("RTLD_DI_SERINFO %llu\n", (unsigned long long)(long long)(RTLD_DI_SERINFO));
#endif
#ifdef RTLD_DI_SERINFOSIZE
printf("RTLD_DI_SERINFOSIZE %llu\n", (unsigned long long)(long long)(RTLD_DI_SERINFOSIZE));
#endif
#ifdef RTLD_DI_TLS_DATA
printf("RTLD_DI_TLS_DATA %llu\n", (unsigned long long)(long long)(RTLD_DI_TLS_DATA));
#endif
#ifdef RTLD_DI_TLS_MODID
printf("RTLD_DI_TLS_MODID %llu\n", (unsigned long long)(long long)(RTLD_DI_TLS_MODID));
#endif
#ifdef RTLD_GLOBAL
printf("RTLD_GLOBAL %llu\n", (unsigned long long)(long long)(RTLD_GLOBAL));
#endif
#ifdef RTLD_LAZY
printf("RTLD_LAZY %llu\n", (unsigned long long)(long long)(RTLD_LAZY));
#endif
#ifdef RTLD_LOCAL
printf("RTLD_LOCAL %llu\n", (unsigned long long)(long long)(RTLD_LOCAL));
#endif
#ifdef RTLD_NODELETE
printf("RTLD_NODELETE %llu\n", (unsigned long long)(long long)(RTLD_NODELETE));
#endif
#ifdef RTLD_NOLOAD
printf("RTLD_NOLOAD %llu\n", (unsigned long long)(long long)(RTLD_NOLOAD));
#endif
#ifdef RTLD_NOW
printf("RTLD_NOW %llu\n", (unsigned long long)(long long)(RTLD_NOW));
#endif
#ifdef RTMGRP_DECnet_IFADDR
printf("RTMGRP_DECnet_IFADDR %llu\n", (unsigned long long)(long long)(RTMGRP_DECnet_IFADDR));
#endif
#ifdef RTMGRP_DECnet_ROUTE
printf("RTMGRP_DECnet_ROUTE %llu\n", (unsigned long long)(long long)(RTMGRP_DECnet_ROUTE));
#endif
#ifdef RTMGRP_IPV4_IFADDR
printf("RTMGRP_IPV4_IFADDR %llu\n", (unsigned long long)(long long)(RTMGRP_IPV4_IFADDR));
#endif
#ifdef RTMGRP_IPV4_MROUTE
printf("RTMGRP_IPV4_MROUTE %llu\n", (unsigned long long)(long long)(RTMGRP_IPV4_MROUTE));
#endif
#ifdef RTMGRP_IPV4_ROUTE
printf("RTMGRP_IPV4_ROUTE %llu\n", (unsigned long long)(long long)(RTMGRP_IPV4_ROUTE));
#endif
#ifdef RTMGRP_IPV4_RULE
printf("RTMGRP_IPV4_RULE %llu\n", (unsigned long long)(long long)(RTMGRP_IPV4_RULE));
#endif
#ifdef RTMGRP_IPV6_IFADDR
printf("RTMGRP_IPV6_IFADDR %llu\n", (unsigned long long)(long long)(RTMGRP_IPV6_IFADDR));
#endif
#ifdef RTMGRP_IPV6_IFINFO
printf("RTMGRP_IPV6_IFINFO %llu\n", (unsigned long long)(long long)(RTMGRP_IPV6_IFINFO));
#endif
#ifdef RTMGRP_IPV6_MROUTE
printf("RTMGRP_IPV6_MROUTE %llu\n", (unsigned long long)(long long)(RTMGRP_IPV6_MROUTE));
#endif
#ifdef RTMGRP_IPV6_PREFIX
printf("RTMGRP_IPV6_PREFIX %llu\n", (unsigned long long)(long long)(RTMGRP_IPV6_PREFIX));
#endif
#ifdef RTMGRP_IPV6_ROUTE
printf("RTMGRP_IPV6_ROUTE %llu\n", (unsigned long long)(long long)(RTMGRP_IPV6_ROUTE));
#endif
#ifdef RTMGRP_LINK
printf("RTMGRP_LINK %llu\n", (unsigned long long)(long long)(RTMGRP_LINK));
#endif
#ifdef RTMGRP_NEIGH
printf("RTMGRP_NEIGH %llu\n", (unsigned long long)(long long)(RTMGRP_NEIGH));
#endif
#ifdef RTMGRP_NOTIFY
printf("RTMGRP_NOTIFY %llu\n", (unsigned long long)(long long)(RTMGRP_NOTIFY));
#endif
#ifdef RTMGRP_TC
printf("RTMGRP_TC %llu\n", (unsigned long long)(long long)(RTMGRP_TC));
#endif
#ifdef RTMSG_AR_FAILED
printf("RTMSG_AR_FAILED %llu\n", (unsigned long long)(long long)(RTMSG_AR_FAILED));
#endif
#ifdef RTMSG_CONTROL
printf("RTMSG_CONTROL %llu\n", (unsigned long long)(long long)(RTMSG_CONTROL));
#endif
#ifdef RTMSG_DELDEVICE
printf("RTMSG_DELDEVICE %llu\n", (unsigned long long)(long long)(RTMSG_DELDEVICE));
#endif
#ifdef RTMSG_DELROUTE
printf("RTMSG_DELROUTE %llu\n", (unsigned long long)(long long)(RTMSG_DELROUTE));
#endif
#ifdef RTMSG_DELRULE
printf("RTMSG_DELRULE %llu\n", (unsigned long long)(long long)(RTMSG_DELRULE));
#endif
#ifdef RTMSG_NEWDEVICE
printf("RTMSG_NEWDEVICE %llu\n", (unsigned long long)(long long)(RTMSG_NEWDEVICE));
#endif
#ifdef RTMSG_NEWROUTE
printf("RTMSG_NEWROUTE %llu\n", (unsigned long long)(long long)(RTMSG_NEWROUTE));
#endif
#ifdef RTMSG_NEWRULE
printf("RTMSG_NEWRULE %llu\n", (unsigned long long)(long long)(RTMSG_NEWRULE));
#endif
#ifdef RTMSG_OVERRUN
printf("RTMSG_OVERRUN %llu\n", (unsigned long long)(long long)(RTMSG_OVERRUN));
#endif
#ifdef RTM_DELACTION
printf("RTM_DELACTION %llu\n", (unsigned long long)(long long)(RTM_DELACTION));
#endif
#ifdef RTM_DELADDR
printf("RTM_DELADDR %llu\n", (unsigned long long)(long long)(RTM_DELADDR));
#endif
#ifdef RTM_DELADDRLABEL
printf("RTM_DELADDRLABEL %llu\n", (unsigned long long)(long long)(RTM_DELADDRLABEL));
#endif
#ifdef RTM_DELLINK
printf("RTM_DELLINK %llu\n", (unsigned long long)(long long)(RTM_DELLINK));
#endif
#ifdef RTM_DELMDB
printf("RTM_DELMDB %llu\n", (unsigned long long)(long long)(RTM_DELMDB));
#endif
#ifdef RTM_DELNEIGH
printf("RTM_DELNEIGH %llu\n", (unsigned long long)(long long)(RTM_DELNEIGH));
#endif
#ifdef RTM_DELNETCONF
printf("RTM_DELNETCONF %llu\n", (unsigned long long)(long long)(RTM_DELNETCONF));
#endif
#ifdef RTM_DELNSID
printf("RTM_DELNSID %llu\n", (unsigned long long)(long long)(RTM_DELNSID));
#endif
#ifdef RTM_DELQDISC
printf("RTM_DELQDISC %llu\n", (unsigned long long)(long long)(RTM_DELQDISC));
#endif
#ifdef RTM_DELROUTE
printf("RTM_DELROUTE %llu\n", (unsigned long long)(long long)(RTM_DELROUTE));
#endif
#ifdef RTM_DELRULE
printf("RTM_DELRULE %llu\n", (unsigned long long)(long long)(RTM_DELRULE));
#endif
#ifdef RTM_DELTCLASS
printf("RTM_DELTCLASS %llu\n", (unsigned long long)(long long)(RTM_DELTCLASS));
#endif
#ifdef RTM_DELTFILTER
printf("RTM_DELTFILTER %llu\n", (unsigned long long)(long long)(RTM_DELTFILTER));
#endif
#ifdef RTM_F_CLONED
printf("RTM_F_CLONED %llu\n", (unsigned long long)(long long)(RTM_F_CLONED));
#endif
#ifdef RTM_F_EQUALIZE
printf("RTM_F_EQUALIZE %llu\n", (unsigned long long)(long long)(RTM_F_EQUALIZE));
#endif
#ifdef RTM_F_FIB_MATCH
printf("RTM_F_FIB_MATCH %llu\n", (unsigned long long)(long long)(RTM_F_FIB_MATCH));
#endif
#ifdef RTM_F_LOOKUP_TABLE
printf("RTM_F_LOOKUP_TABLE %llu\n", (unsigned long long)(long long)(RTM_F_LOOKUP_TABLE));
#endif
#ifdef RTM_F_NOTIFY
printf("RTM_F_NOTIFY %llu\n", (unsigned long long)(long long)(RTM_F_NOTIFY));
#endif
#ifdef RTM_F_PREFIX
printf("RTM_F_PREFIX %llu\n", (unsigned long long)(long long)(RTM_F_PREFIX));
#endif
#ifdef RTM_GETACTION
printf("RTM_GETACTION %llu\n", (unsigned long long)(long long)(RTM_GETACTION));
#endif
#ifdef RTM_GETADDR
printf("RTM_GETADDR %llu\n", (unsigned long long)(long long)(RTM_GETADDR));
#endif
#ifdef RTM_GETADDRLABEL
printf("RTM_GETADDRLABEL %llu\n", (unsigned long long)(long long)(RTM_GETADDRLABEL));
#endif
#ifdef RTM_GETANYCAST
printf("RTM_GETANYCAST %llu\n", (unsigned long long)(long long)(RTM_GETANYCAST));
#endif
#ifdef RTM_GETDCB
printf("RTM_GETDCB %llu\n", (unsigned long long)(long long)(RTM_GETDCB));
#endif
#ifdef RTM_GETLINK
printf("RTM_GETLINK %llu\n", (unsigned long long)(long long)(RTM_GETLINK));
#endif
#ifdef RTM_GETMDB
printf("RTM_GETMDB %llu\n", (unsigned long long)(long long)(RTM_GETMDB));
#endif
#ifdef RTM_GETMULTICAST
printf("RTM_GETMULTICAST %llu\n", (unsigned long long)(long long)(RTM_GETMULTICAST));
#endif
#ifdef RTM_GETNEIGH
printf("RTM_GETNEIGH %llu\n", (unsigned long long)(long long)(RTM_GETNEIGH));
#endif
#ifdef RTM_GETNEIGHTBL
printf("RTM_GETNEIGHTBL %llu\n", (unsigned long long)(long long)(RTM_GETNEIGHTBL));
#endif
#ifdef RTM_GETNETCONF
printf("RTM_GETNETCONF %llu\n", (unsigned long long)(long long)(RTM_GETNETCONF));
#endif
#ifdef RTM_GETNSID
printf("RTM_GETNSID %llu\n", (unsigned long long)(long long)(RTM_GETNSID));
#endif
#ifdef RTM_GETQDISC
printf("RTM_GETQDISC %llu\n", (unsigned long long)(long long)(RTM_GETQDISC));
#endif
#ifdef RTM_GETROUTE
printf("RTM_GETROUTE %llu\n", (unsigned long long)(long long)(RTM_GETROUTE));
#endif
#ifdef RTM_GETRULE
printf("RTM_GETRULE %llu\n", (unsigned long long)(long long)(RTM_GETRULE));
#endif
#ifdef RTM_GETSTATS
printf("RTM_GETSTATS %llu\n", (unsigned long long)(long long)(RTM_GETSTATS));
#endif
#ifdef RTM_GETTCLASS
printf("RTM_GETTCLASS %llu\n", (unsigned long long)(long long)(RTM_GETTCLASS));
#endif
#ifdef RTM_GETTFILTER
printf("RTM_GETTFILTER %llu\n", (unsigned long long)(long long)(RTM_GETTFILTER));
#endif
#ifdef RTM_NEWACTION
printf("RTM_NEWACTION %llu\n", (unsigned long long)(long long)(RTM_NEWACTION));
#endif
#ifdef RTM_NEWADDR
printf("RTM_NEWADDR %llu\n", (unsigned long long)(long long)(RTM_NEWADDR));
#endif
#ifdef RTM_NEWADDRLABEL
printf("RTM_NEWADDRLABEL %llu\n", (unsigned long long)(long long)(RTM_NEWADDRLABEL));
#endif
#ifdef RTM_NEWCACHEREPORT
printf("RTM_NEWCACHEREPORT %llu\n", (unsigned long long)(long long)(RTM_NEWCACHEREPORT));
#endif
#ifdef RTM_NEWLINK
printf("RTM_NEWLINK %llu\n", (unsigned long long)(long long)(RTM_NEWLINK));
#endif
#ifdef RTM_NEWMDB
printf("RTM_NEWMDB %llu\n", (unsigned long long)(long long)(RTM_NEWMDB));
#endif
#ifdef RTM_NEWNDUSEROPT
printf("RTM_NEWNDUSEROPT %llu\n", (unsigned long long)(long long)(RTM_NEWNDUSEROPT));
#endif
#ifdef RTM_NEWNEIGH
printf("RTM_NEWNEIGH %llu\n", (unsigned long long)(long long)(RTM_NEWNEIGH));
#endif
#ifdef RTM_NEWNEIGHTBL
printf("RTM_NEWNEIGHTBL %llu\n", (unsigned long long)(long long)(RTM_NEWNEIGHTBL));
#endif
#ifdef RTM_NEWNETCONF
printf("RTM_NEWNETCONF %llu\n", (unsigned long long)(long long)(RTM_NEWNETCONF));
#endif
#ifdef RTM_NEWNSID
printf("RTM_NEWNSID %llu\n", (unsigned long long)(long long)(RTM_NEWNSID));
#endif
#ifdef RTM_NEWPREFIX
printf("RTM_NEWPREFIX %llu\n", (unsigned long long)(long long)(RTM_NEWPREFIX));
#endif
#ifdef RTM_NEWQDISC
printf("RTM_NEWQDISC %llu\n", (unsigned long long)(long long)(RTM_NEWQDISC));
#endif
#ifdef RTM_NEWROUTE
printf("RTM_NEWROUTE %llu\n", (unsigned long long)(long long)(RTM_NEWROUTE));
#endif
#ifdef RTM_NEWRULE
printf("RTM_NEWRULE %llu\n", (unsigned long long)(long long)(RTM_NEWRULE));
#endif
#ifdef RTM_NEWSTATS
printf("RTM_NEWSTATS %llu\n", (unsigned long long)(long long)(RTM_NEWSTATS));
#endif
#ifdef RTM_NEWTCLASS
printf("RTM_NEWTCLASS %llu\n", (unsigned long long)(long long)(RTM_NEWTCLASS));
#endif
#ifdef RTM_NEWTFILTER
printf("RTM_NEWTFILTER %llu\n", (unsigned long long)(long long)(RTM_NEWTFILTER));
#endif
#ifdef RTM_SETDCB
printf("RTM_SETDCB %llu\n", (unsigned long long)(long long)(RTM_SETDCB));
#endif
#ifdef RTM_SETLINK
printf("RTM_SETLINK %llu\n", (unsigned long long)(long long)(RTM_SETLINK));
#endif
#ifdef RTM_SETNEIGHTBL
printf("RTM_SETNEIGHTBL %llu\n", (unsigned long long)(long long)(RTM_SETNEIGHTBL));
#endif
#ifdef RTNLGRP_BRVLAN
printf("RTNLGRP_BRVLAN %llu\n", (unsigned long long)(long long)(RTNLGRP_BRVLAN));
#endif
#ifdef RTNLGRP_DCB
printf("RTNLGRP_DCB %llu\n", (unsigned long long)(long long)(RTNLGRP_DCB));
#endif
#ifdef RTNLGRP_DECnet_IFADDR
printf("RTNLGRP_DECnet_IFADDR %llu\n", (unsigned long long)(long long)(RTNLGRP_DECnet_IFADDR));
#endif
#ifdef RTNLGRP_DECnet_ROUTE
printf("RTNLGRP_DECnet_ROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_DECnet_ROUTE));
#endif
#ifdef RTNLGRP_DECnet_RULE
printf("RTNLGRP_DECnet_RULE %llu\n", (unsigned long long)(long long)(RTNLGRP_DECnet_RULE));
#endif
#ifdef RTNLGRP_IPV4_IFADDR
printf("RTNLGRP_IPV4_IFADDR %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_IFADDR));
#endif
#ifdef RTNLGRP_IPV4_MROUTE
printf("RTNLGRP_IPV4_MROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_MROUTE));
#endif
#ifdef RTNLGRP_IPV4_MROUTE_R
printf("RTNLGRP_IPV4_MROUTE_R %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_MROUTE_R));
#endif
#ifdef RTNLGRP_IPV4_NETCONF
printf("RTNLGRP_IPV4_NETCONF %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_NETCONF));
#endif
#ifdef RTNLGRP_IPV4_ROUTE
printf("RTNLGRP_IPV4_ROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_ROUTE));
#endif
#ifdef RTNLGRP_IPV4_RULE
printf("RTNLGRP_IPV4_RULE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV4_RULE));
#endif
#ifdef RTNLGRP_IPV6_IFADDR
printf("RTNLGRP_IPV6_IFADDR %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_IFADDR));
#endif
#ifdef RTNLGRP_IPV6_IFINFO
printf("RTNLGRP_IPV6_IFINFO %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_IFINFO));
#endif
#ifdef RTNLGRP_IPV6_MROUTE
printf("RTNLGRP_IPV6_MROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_MROUTE));
#endif
#ifdef RTNLGRP_IPV6_MROUTE_R
printf("RTNLGRP_IPV6_MROUTE_R %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_MROUTE_R));
#endif
#ifdef RTNLGRP_IPV6_NETCONF
printf("RTNLGRP_IPV6_NETCONF %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_NETCONF));
#endif
#ifdef RTNLGRP_IPV6_PREFIX
printf("RTNLGRP_IPV6_PREFIX %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_PREFIX));
#endif
#ifdef RTNLGRP_IPV6_ROUTE
printf("RTNLGRP_IPV6_ROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_ROUTE));
#endif
#ifdef RTNLGRP_IPV6_RULE
printf("RTNLGRP_IPV6_RULE %llu\n", (unsigned long long)(long long)(RTNLGRP_IPV6_RULE));
#endif
#ifdef RTNLGRP_LINK
printf("RTNLGRP_LINK %llu\n", (unsigned long long)(long long)(RTNLGRP_LINK));
#endif
#ifdef RTNLGRP_MCTP_IFADDR
printf("RTNLGRP_MCTP_IFADDR %llu\n", (unsigned long long)(long long)(RTNLGRP_MCTP_IFADDR));
#endif
#ifdef RTNLGRP_MDB
printf("RTNLGRP_MDB %llu\n", (unsigned long long)(long long)(RTNLGRP_MDB));
#endif
#ifdef RTNLGRP_MPLS_NETCONF
printf("RTNLGRP_MPLS_NETCONF %llu\n", (unsigned long long)(long long)(RTNLGRP_MPLS_NETCONF));
#endif
#ifdef RTNLGRP_MPLS_ROUTE
printf("RTNLGRP_MPLS_ROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_MPLS_ROUTE));
#endif
#ifdef RTNLGRP_ND_USEROPT
printf("RTNLGRP_ND_USEROPT %llu\n", (unsigned long long)(long long)(RTNLGRP_ND_USEROPT));
#endif
#ifdef RTNLGRP_NEIGH
printf("RTNLGRP_NEIGH %llu\n", (unsigned long long)(long long)(RTNLGRP_NEIGH));
#endif
#ifdef RTNLGRP_NEXTHOP
printf("RTNLGRP_NEXTHOP %llu\n", (unsigned long long)(long long)(RTNLGRP_NEXTHOP));
#endif
#ifdef RTNLGRP_NONE
printf("RTNLGRP_NONE %llu\n", (unsigned long long)(long long)(RTNLGRP_NONE));
#endif
#ifdef RTNLGRP_NOP2
printf("RTNLGRP_NOP2 %llu\n", (unsigned long long)(long long)(RTNLGRP_NOP2));
#endif
#ifdef RTNLGRP_NOP4
printf("RTNLGRP_NOP4 %llu\n", (unsigned long long)(long long)(RTNLGRP_NOP4));
#endif
#ifdef RTNLGRP_NOTIFY
printf("RTNLGRP_NOTIFY %llu\n", (unsigned long long)(long long)(RTNLGRP_NOTIFY));
#endif
#ifdef RTNLGRP_NSID
printf("RTNLGRP_NSID %llu\n", (unsigned long long)(long long)(RTNLGRP_NSID));
#endif
#ifdef RTNLGRP_PHONET_IFADDR
printf("RTNLGRP_PHONET_IFADDR %llu\n", (unsigned long long)(long long)(RTNLGRP_PHONET_IFADDR));
#endif
#ifdef RTNLGRP_PHONET_ROUTE
printf("RTNLGRP_PHONET_ROUTE %llu\n", (unsigned long long)(long long)(RTNLGRP_PHONET_ROUTE));
#endif
#ifdef RTNLGRP_STATS
printf("RTNLGRP_STATS %llu\n", (unsigned long long)(long long)(RTNLGRP_STATS));
#endif
#ifdef RTNLGRP_TC
printf("RTNLGRP_TC %llu\n", (unsigned long long)(long long)(RTNLGRP_TC));
#endif
#ifdef RTNLGRP_TUNNEL
printf("RTNLGRP_TUNNEL %llu\n", (unsigned long long)(long long)(RTNLGRP_TUNNEL));
#endif
#ifdef RTN_ANYCAST
printf("RTN_ANYCAST %llu\n", (unsigned long long)(long long)(RTN_ANYCAST));
#endif
#ifdef RTN_BLACKHOLE
printf("RTN_BLACKHOLE %llu\n", (unsigned long long)(long long)(RTN_BLACKHOLE));
#endif
#ifdef RTN_BROADCAST
printf("RTN_BROADCAST %llu\n", (unsigned long long)(long long)(RTN_BROADCAST));
#endif
#ifdef RTN_LOCAL
printf("RTN_LOCAL %llu\n", (unsigned long long)(long long)(RTN_LOCAL));
#endif
#ifdef RTN_MULTICAST
printf("RTN_MULTICAST %llu\n", (unsigned long long)(long long)(RTN_MULTICAST));
#endif
#ifdef RTN_NAT
printf("RTN_NAT %llu\n", (unsigned long long)(long long)(RTN_NAT));
#endif
#ifdef RTN_PROHIBIT
printf("RTN_PROHIBIT %llu\n", (unsigned long long)(long long)(RTN_PROHIBIT));
#endif
#ifdef RTN_THROW
printf("RTN_THROW %llu\n", (unsigned long long)(long long)(RTN_THROW));
#endif
#ifdef RTN_UNICAST
printf("RTN_UNICAST %llu\n", (unsigned long long)(long long)(RTN_UNICAST));
#endif
#ifdef RTN_UNREACHABLE
printf("RTN_UNREACHABLE %llu\n", (unsigned long long)(long long)(RTN_UNREACHABLE));
#endif
#ifdef RTN_UNSPEC
printf("RTN_UNSPEC %llu\n", (unsigned long long)(long long)(RTN_UNSPEC));
#endif
#ifdef RTN_XRESOLVE
printf("RTN_XRESOLVE %llu\n", (unsigned long long)(long long)(RTN_XRESOLVE));
#endif
#ifdef RTPROT_BOOT
printf("RTPROT_BOOT %llu\n", (unsigned long long)(long long)(RTPROT_BOOT));
#endif
#ifdef RTPROT_KERNEL
printf("RTPROT_KERNEL %llu\n", (unsigned long long)(long long)(RTPROT_KERNEL));
#endif
#ifdef RTPROT_REDIRECT
printf("RTPROT_REDIRECT %llu\n", (unsigned long long)(long long)(RTPROT_REDIRECT));
#endif
#ifdef RTPROT_STATIC
printf("RTPROT_STATIC %llu\n", (unsigned long long)(long long)(RTPROT_STATIC));
#endif
#ifdef RTPROT_UNSPEC
printf("RTPROT_UNSPEC %llu\n", (unsigned long long)(long long)(RTPROT_UNSPEC));
#endif
#ifdef RT_CLASS_DEFAULT
printf("RT_CLASS_DEFAULT %llu\n", (unsigned long long)(long long)(RT_CLASS_DEFAULT));
#endif
#ifdef RT_CLASS_LOCAL
printf("RT_CLASS_LOCAL %llu\n", (unsigned long long)(long long)(RT_CLASS_LOCAL));
#endif
#ifdef RT_CLASS_MAIN
printf("RT_CLASS_MAIN %llu\n", (unsigned long long)(long long)(RT_CLASS_MAIN));
#endif
#ifdef RT_CLASS_MAX
printf("RT_CLASS_MAX %llu\n", (unsigned long long)(long long)(RT_CLASS_MAX));
#endif
#ifdef RT_CLASS_UNSPEC
printf("RT_CLASS_UNSPEC %llu\n", (unsigned long long)(long long)(RT_CLASS_UNSPEC));
#endif
#ifdef RT_SCOPE_HOST
printf("RT_SCOPE_HOST %llu\n", (unsigned long long)(long long)(RT_SCOPE_HOST));
#endif
#ifdef RT_SCOPE_LINK
printf("RT_SCOPE_LINK %llu\n", (unsigned long long)(long long)(RT_SCOPE_LINK));
#endif
#ifdef RT_SCOPE_NOWHERE
printf("RT_SCOPE_NOWHERE %llu\n", (unsigned long long)(long long)(RT_SCOPE_NOWHERE));
#endif
#ifdef RT_SCOPE_SITE
printf("RT_SCOPE_SITE %llu\n", (unsigned long long)(long long)(RT_SCOPE_SITE));
#endif
#ifdef RT_SCOPE_UNIVERSE
printf("RT_SCOPE_UNIVERSE %llu\n", (unsigned long long)(long long)(RT_SCOPE_UNIVERSE));
#endif
#ifdef RT_TABLE_COMPAT
printf("RT_TABLE_COMPAT %llu\n", (unsigned long long)(long long)(RT_TABLE_COMPAT));
#endif
#ifdef RT_TABLE_DEFAULT
printf("RT_TABLE_DEFAULT %llu\n", (unsigned long long)(long long)(RT_TABLE_DEFAULT));
#endif
#ifdef RT_TABLE_LOCAL
printf("RT_TABLE_LOCAL %llu\n", (unsigned long long)(long long)(RT_TABLE_LOCAL));
#endif
#ifdef RT_TABLE_MAIN
printf("RT_TABLE_MAIN %llu\n", (unsigned long long)(long long)(RT_TABLE_MAIN));
#endif
#ifdef RT_TABLE_UNSPEC
printf("RT_TABLE_UNSPEC %llu\n", (unsigned long long)(long long)(RT_TABLE_UNSPEC));
#endif
#ifdef RUN_LVL
printf("RUN_LVL %llu\n", (unsigned long long)(long long)(RUN_LVL));
#endif
#ifdef RUSAGE_CHILDREN
printf("RUSAGE_CHILDREN %llu\n", (unsigned long long)(long long)(RUSAGE_CHILDREN));
#endif
#ifdef RUSAGE_SELF
printf("RUSAGE_SELF %llu\n", (unsigned long long)(long long)(RUSAGE_SELF));
#endif
#ifdef RUSAGE_THREAD
printf("RUSAGE_THREAD %llu\n", (unsigned long long)(long long)(RUSAGE_THREAD));
#endif
#ifdef RWF_APPEND
printf("RWF_APPEND %llu\n", (unsigned long long)(long long)(RWF_APPEND));
#endif
#ifdef RWF_ATOMIC
printf("RWF_ATOMIC %llu\n", (unsigned long long)(long long)(RWF_ATOMIC));
#endif
#ifdef RWF_DONTCACHE
printf("RWF_DONTCACHE %llu\n", (unsigned long long)(long long)(RWF_DONTCACHE));
#endif
#ifdef RWF_DSYNC
printf("RWF_DSYNC %llu\n", (unsigned long long)(long long)(RWF_DSYNC));
#endif
#ifdef RWF_HIPRI
printf("RWF_HIPRI %llu\n", (unsigned long long)(long long)(RWF_HIPRI));
#endif
#ifdef RWF_NOAPPEND
printf("RWF_NOAPPEND %llu\n", (unsigned long long)(long long)(RWF_NOAPPEND));
#endif
#ifdef RWF_NOWAIT
printf("RWF_NOWAIT %llu\n", (unsigned long long)(long long)(RWF_NOWAIT));
#endif
#ifdef RWF_SYNC
printf("RWF_SYNC %llu\n", (unsigned long long)(long long)(RWF_SYNC));
#endif
#ifdef R_OK
printf("R_OK %llu\n", (unsigned long long)(long long)(R_OK));
#endif
#ifdef SA_NOCLDSTOP
printf("SA_NOCLDSTOP %llu\n", (unsigned long long)(long long)(SA_NOCLDSTOP));
#endif
#ifdef SA_NOCLDWAIT
printf("SA_NOCLDWAIT %llu\n", (unsigned long long)(long long)(SA_NOCLDWAIT));
#endif
#ifdef SA_NODEFER
printf("SA_NODEFER %llu\n", (unsigned long long)(long long)(SA_NODEFER));
#endif
#ifdef SA_ONSTACK
printf("SA_ONSTACK %llu\n", (unsigned long long)(long long)(SA_ONSTACK));
#endif
#ifdef SA_RESETHAND
printf("SA_RESETHAND %llu\n", (unsigned long long)(long long)(SA_RESETHAND));
#endif
#ifdef SA_RESTART
printf("SA_RESTART %llu\n", (unsigned long long)(long long)(SA_RESTART));
#endif
#ifdef SA_SIGINFO
printf("SA_SIGINFO %llu\n", (unsigned long long)(long long)(SA_SIGINFO));
#endif
#ifdef SCHED_BATCH
printf("SCHED_BATCH %llu\n", (unsigned long long)(long long)(SCHED_BATCH));
#endif
#ifdef SCHED_DEADLINE
printf("SCHED_DEADLINE %llu\n", (unsigned long long)(long long)(SCHED_DEADLINE));
#endif
#ifdef SCHED_FIFO
printf("SCHED_FIFO %llu\n", (unsigned long long)(long long)(SCHED_FIFO));
#endif
#ifdef SCHED_FLAG_ALL
printf("SCHED_FLAG_ALL %llu\n", (unsigned long long)(long long)(SCHED_FLAG_ALL));
#endif
#ifdef SCHED_FLAG_DL_OVERRUN
printf("SCHED_FLAG_DL_OVERRUN %llu\n", (unsigned long long)(long long)(SCHED_FLAG_DL_OVERRUN));
#endif
#ifdef SCHED_FLAG_KEEP_ALL
printf("SCHED_FLAG_KEEP_ALL %llu\n", (unsigned long long)(long long)(SCHED_FLAG_KEEP_ALL));
#endif
#ifdef SCHED_FLAG_KEEP_PARAMS
printf("SCHED_FLAG_KEEP_PARAMS %llu\n", (unsigned long long)(long long)(SCHED_FLAG_KEEP_PARAMS));
#endif
#ifdef SCHED_FLAG_KEEP_POLICY
printf("SCHED_FLAG_KEEP_POLICY %llu\n", (unsigned long long)(long long)(SCHED_FLAG_KEEP_POLICY));
#endif
#ifdef SCHED_FLAG_RECLAIM
printf("SCHED_FLAG_RECLAIM %llu\n", (unsigned long long)(long long)(SCHED_FLAG_RECLAIM));
#endif
#ifdef SCHED_FLAG_RESET_ON_FORK
printf("SCHED_FLAG_RESET_ON_FORK %llu\n", (unsigned long long)(long long)(SCHED_FLAG_RESET_ON_FORK));
#endif
#ifdef SCHED_FLAG_UTIL_CLAMP
printf("SCHED_FLAG_UTIL_CLAMP %llu\n", (unsigned long long)(long long)(SCHED_FLAG_UTIL_CLAMP));
#endif
#ifdef SCHED_FLAG_UTIL_CLAMP_MAX
printf("SCHED_FLAG_UTIL_CLAMP_MAX %llu\n", (unsigned long long)(long long)(SCHED_FLAG_UTIL_CLAMP_MAX));
#endif
#ifdef SCHED_FLAG_UTIL_CLAMP_MIN
printf("SCHED_FLAG_UTIL_CLAMP_MIN %llu\n", (unsigned long long)(long long)(SCHED_FLAG_UTIL_CLAMP_MIN));
#endif
#ifdef SCHED_IDLE
printf("SCHED_IDLE %llu\n", (unsigned long long)(long long)(SCHED_IDLE));
#endif
#ifdef SCHED_NORMAL
printf("SCHED_NORMAL %llu\n", (unsigned long long)(long long)(SCHED_NORMAL));
#endif
#ifdef SCHED_OTHER
printf("SCHED_OTHER %llu\n", (unsigned long long)(long long)(SCHED_OTHER));
#endif
#ifdef SCHED_RESET_ON_FORK
printf("SCHED_RESET_ON_FORK %llu\n", (unsigned long long)(long long)(SCHED_RESET_ON_FORK));
#endif
#ifdef SCHED_RR
printf("SCHED_RR %llu\n", (unsigned long long)(long long)(SCHED_RR));
#endif
#ifdef SCM_CREDENTIALS
printf("SCM_CREDENTIALS %llu\n", (unsigned long long)(long long)(SCM_CREDENTIALS));
#endif
#ifdef SCM_DEVMEM_DMABUF
printf("SCM_DEVMEM_DMABUF %llu\n", (unsigned long long)(long long)(SCM_DEVMEM_DMABUF));
#endif
#ifdef SCM_DEVMEM_LINEAR
printf("SCM_DEVMEM_LINEAR %llu\n", (unsigned long long)(long long)(SCM_DEVMEM_LINEAR));
#endif
#ifdef SCM_RIGHTS
printf("SCM_RIGHTS %llu\n", (unsigned long long)(long long)(SCM_RIGHTS));
#endif
#ifdef SCM_TIMESTAMP
printf("SCM_TIMESTAMP %llu\n", (unsigned long long)(long long)(SCM_TIMESTAMP));
#endif
#ifdef SCM_TIMESTAMPING
printf("SCM_TIMESTAMPING %llu\n", (unsigned long long)(long long)(SCM_TIMESTAMPING));
#endif
#ifdef SCM_TIMESTAMPING_OPT_STATS
printf("SCM_TIMESTAMPING_OPT_STATS %llu\n", (unsigned long long)(long long)(SCM_TIMESTAMPING_OPT_STATS));
#endif
#ifdef SCM_TIMESTAMPING_PKTINFO
printf("SCM_TIMESTAMPING_PKTINFO %llu\n", (unsigned long long)(long long)(SCM_TIMESTAMPING_PKTINFO));
#endif
#ifdef SCM_TIMESTAMPNS
printf("SCM_TIMESTAMPNS %llu\n", (unsigned long long)(long long)(SCM_TIMESTAMPNS));
#endif
#ifdef SCM_TXTIME
printf("SCM_TXTIME %llu\n", (unsigned long long)(long long)(SCM_TXTIME));
#endif
#ifdef SCM_WIFI_STATUS
printf("SCM_WIFI_STATUS %llu\n", (unsigned long long)(long long)(SCM_WIFI_STATUS));
#endif
#ifdef SCTP_ABORT
printf("SCTP_ABORT %llu\n", (unsigned long long)(long long)(SCTP_ABORT));
#endif
#ifdef SCTP_ADAPTATION_LAYER
printf("SCTP_ADAPTATION_LAYER %llu\n", (unsigned long long)(long long)(SCTP_ADAPTATION_LAYER));
#endif
#ifdef SCTP_ADDR_OVER
printf("SCTP_ADDR_OVER %llu\n", (unsigned long long)(long long)(SCTP_ADDR_OVER));
#endif
#ifdef SCTP_ALL_ASSOC
printf("SCTP_ALL_ASSOC %llu\n", (unsigned long long)(long long)(SCTP_ALL_ASSOC));
#endif
#ifdef SCTP_ASSOCINFO
printf("SCTP_ASSOCINFO %llu\n", (unsigned long long)(long long)(SCTP_ASSOCINFO));
#endif
#ifdef SCTP_AUTHINFO
printf("SCTP_AUTHINFO %llu\n", (unsigned long long)(long long)(SCTP_AUTHINFO));
#endif
#ifdef SCTP_AUTH_ACTIVE_KEY
printf("SCTP_AUTH_ACTIVE_KEY %llu\n", (unsigned long long)(long long)(SCTP_AUTH_ACTIVE_KEY));
#endif
#ifdef SCTP_AUTH_CHUNK
printf("SCTP_AUTH_CHUNK %llu\n", (unsigned long long)(long long)(SCTP_AUTH_CHUNK));
#endif
#ifdef SCTP_AUTH_DEACTIVATE_KEY
printf("SCTP_AUTH_DEACTIVATE_KEY %llu\n", (unsigned long long)(long long)(SCTP_AUTH_DEACTIVATE_KEY));
#endif
#ifdef SCTP_AUTH_DELETE_KEY
printf("SCTP_AUTH_DELETE_KEY %llu\n", (unsigned long long)(long long)(SCTP_AUTH_DELETE_KEY));
#endif
#ifdef SCTP_AUTH_KEY
printf("SCTP_AUTH_KEY %llu\n", (unsigned long long)(long long)(SCTP_AUTH_KEY));
#endif
#ifdef SCTP_AUTOCLOSE
printf("SCTP_AUTOCLOSE %llu\n", (unsigned long long)(long long)(SCTP_AUTOCLOSE));
#endif
#ifdef SCTP_AUTO_ASCONF
printf("SCTP_AUTO_ASCONF %llu\n", (unsigned long long)(long long)(SCTP_AUTO_ASCONF));
#endif
#ifdef SCTP_CONTEXT
printf("SCTP_CONTEXT %llu\n", (unsigned long long)(long long)(SCTP_CONTEXT));
#endif
#ifdef SCTP_CURRENT_ASSOC
printf("SCTP_CURRENT_ASSOC %llu\n", (unsigned long long)(long long)(SCTP_CURRENT_ASSOC));
#endif
#ifdef SCTP_DEFAULT_SEND_PARAM
printf("SCTP_DEFAULT_SEND_PARAM %llu\n", (unsigned long long)(long long)(SCTP_DEFAULT_SEND_PARAM));
#endif
#ifdef SCTP_DEFAULT_SNDINFO
printf("SCTP_DEFAULT_SNDINFO %llu\n", (unsigned long long)(long long)(SCTP_DEFAULT_SNDINFO));
#endif
#ifdef SCTP_DELAYED_ACK
printf("SCTP_DELAYED_ACK %llu\n", (unsigned long long)(long long)(SCTP_DELAYED_ACK));
#endif
#ifdef SCTP_DELAYED_ACK_TIME
printf("SCTP_DELAYED_ACK_TIME %llu\n", (unsigned long long)(long long)(SCTP_DELAYED_ACK_TIME));
#endif
#ifdef SCTP_DELAYED_SACK
printf("SCTP_DELAYED_SACK %llu\n", (unsigned long long)(long long)(SCTP_DELAYED_SACK));
#endif
#ifdef SCTP_DISABLE_FRAGMENTS
printf("SCTP_DISABLE_FRAGMENTS %llu\n", (unsigned long long)(long long)(SCTP_DISABLE_FRAGMENTS));
#endif
#ifdef SCTP_DSTADDRV4
printf("SCTP_DSTADDRV4 %llu\n", (unsigned long long)(long long)(SCTP_DSTADDRV4));
#endif
#ifdef SCTP_DSTADDRV6
printf("SCTP_DSTADDRV6 %llu\n", (unsigned long long)(long long)(SCTP_DSTADDRV6));
#endif
#ifdef SCTP_ENABLE_CHANGE_ASSOC_REQ
printf("SCTP_ENABLE_CHANGE_ASSOC_REQ %llu\n", (unsigned long long)(long long)(SCTP_ENABLE_CHANGE_ASSOC_REQ));
#endif
#ifdef SCTP_ENABLE_RESET_ASSOC_REQ
printf("SCTP_ENABLE_RESET_ASSOC_REQ %llu\n", (unsigned long long)(long long)(SCTP_ENABLE_RESET_ASSOC_REQ));
#endif
#ifdef SCTP_ENABLE_RESET_STREAM_REQ
printf("SCTP_ENABLE_RESET_STREAM_REQ %llu\n", (unsigned long long)(long long)(SCTP_ENABLE_RESET_STREAM_REQ));
#endif
#ifdef SCTP_ENABLE_STRRESET_MASK
printf("SCTP_ENABLE_STRRESET_MASK %llu\n", (unsigned long long)(long long)(SCTP_ENABLE_STRRESET_MASK));
#endif
#ifdef SCTP_EOF
printf("SCTP_EOF %llu\n", (unsigned long long)(long long)(SCTP_EOF));
#endif
#ifdef SCTP_EVENTS
printf("SCTP_EVENTS %llu\n", (unsigned long long)(long long)(SCTP_EVENTS));
#endif
#ifdef SCTP_FRAGMENT_INTERLEAVE
printf("SCTP_FRAGMENT_INTERLEAVE %llu\n", (unsigned long long)(long long)(SCTP_FRAGMENT_INTERLEAVE));
#endif
#ifdef SCTP_FUTURE_ASSOC
printf("SCTP_FUTURE_ASSOC %llu\n", (unsigned long long)(long long)(SCTP_FUTURE_ASSOC));
#endif
#ifdef SCTP_GET_ASSOC_ID_LIST
printf("SCTP_GET_ASSOC_ID_LIST %llu\n", (unsigned long long)(long long)(SCTP_GET_ASSOC_ID_LIST));
#endif
#ifdef SCTP_GET_ASSOC_NUMBER
printf("SCTP_GET_ASSOC_NUMBER %llu\n", (unsigned long long)(long long)(SCTP_GET_ASSOC_NUMBER));
#endif
#ifdef SCTP_GET_PEER_ADDR_INFO
printf("SCTP_GET_PEER_ADDR_INFO %llu\n", (unsigned long long)(long long)(SCTP_GET_PEER_ADDR_INFO));
#endif
#ifdef SCTP_HMAC_IDENT
printf("SCTP_HMAC_IDENT %llu\n", (unsigned long long)(long long)(SCTP_HMAC_IDENT));
#endif
#ifdef SCTP_INIT
printf("SCTP_INIT %llu\n", (unsigned long long)(long long)(SCTP_INIT));
#endif
#ifdef SCTP_INITMSG
printf("SCTP_INITMSG %llu\n", (unsigned long long)(long long)(SCTP_INITMSG));
#endif
#ifdef SCTP_I_WANT_MAPPED_V4_ADDR
printf("SCTP_I_WANT_MAPPED_V4_ADDR %llu\n", (unsigned long long)(long long)(SCTP_I_WANT_MAPPED_V4_ADDR));
#endif
#ifdef SCTP_LOCAL_AUTH_CHUNKS
printf("SCTP_LOCAL_AUTH_CHUNKS %llu\n", (unsigned long long)(long long)(SCTP_LOCAL_AUTH_CHUNKS));
#endif
#ifdef SCTP_MAXSEG
printf("SCTP_MAXSEG %llu\n", (unsigned long long)(long long)(SCTP_MAXSEG));
#endif
#ifdef SCTP_MAX_BURST
printf("SCTP_MAX_BURST %llu\n", (unsigned long long)(long long)(SCTP_MAX_BURST));
#endif
#ifdef SCTP_NODELAY
printf("SCTP_NODELAY %llu\n", (unsigned long long)(long long)(SCTP_NODELAY));
#endif
#ifdef SCTP_NOTIFICATION
printf("SCTP_NOTIFICATION %llu\n", (unsigned long long)(long long)(SCTP_NOTIFICATION));
#endif
#ifdef SCTP_NXTINFO
printf("SCTP_NXTINFO %llu\n", (unsigned long long)(long long)(SCTP_NXTINFO));
#endif
#ifdef SCTP_PARTIAL_DELIVERY_POINT
printf("SCTP_PARTIAL_DELIVERY_POINT %llu\n", (unsigned long long)(long long)(SCTP_PARTIAL_DELIVERY_POINT));
#endif
#ifdef SCTP_PEER_ADDR_PARAMS
printf("SCTP_PEER_ADDR_PARAMS %llu\n", (unsigned long long)(long long)(SCTP_PEER_ADDR_PARAMS));
#endif
#ifdef SCTP_PEER_ADDR_THLDS
printf("SCTP_PEER_ADDR_THLDS %llu\n", (unsigned long long)(long long)(SCTP_PEER_ADDR_THLDS));
#endif
#ifdef SCTP_PEER_ADDR_THLDS_V2
printf("SCTP_PEER_ADDR_THLDS_V2 %llu\n", (unsigned long long)(long long)(SCTP_PEER_ADDR_THLDS_V2));
#endif
#ifdef SCTP_PEER_AUTH_CHUNKS
printf("SCTP_PEER_AUTH_CHUNKS %llu\n", (unsigned long long)(long long)(SCTP_PEER_AUTH_CHUNKS));
#endif
#ifdef SCTP_PRIMARY_ADDR
printf("SCTP_PRIMARY_ADDR %llu\n", (unsigned long long)(long long)(SCTP_PRIMARY_ADDR));
#endif
#ifdef SCTP_PRINFO
printf("SCTP_PRINFO %llu\n", (unsigned long long)(long long)(SCTP_PRINFO));
#endif
#ifdef SCTP_PR_SCTP_ALL
printf("SCTP_PR_SCTP_ALL %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_ALL));
#endif
#ifdef SCTP_PR_SCTP_MASK
printf("SCTP_PR_SCTP_MASK %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_MASK));
#endif
#ifdef SCTP_PR_SCTP_MAX
printf("SCTP_PR_SCTP_MAX %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_MAX));
#endif
#ifdef SCTP_PR_SCTP_NONE
printf("SCTP_PR_SCTP_NONE %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_NONE));
#endif
#ifdef SCTP_PR_SCTP_PRIO
printf("SCTP_PR_SCTP_PRIO %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_PRIO));
#endif
#ifdef SCTP_PR_SCTP_RTX
printf("SCTP_PR_SCTP_RTX %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_RTX));
#endif
#ifdef SCTP_PR_SCTP_TTL
printf("SCTP_PR_SCTP_TTL %llu\n", (unsigned long long)(long long)(SCTP_PR_SCTP_TTL));
#endif
#ifdef SCTP_RCVINFO
printf("SCTP_RCVINFO %llu\n", (unsigned long long)(long long)(SCTP_RCVINFO));
#endif
#ifdef SCTP_RECVNXTINFO
printf("SCTP_RECVNXTINFO %llu\n", (unsigned long long)(long long)(SCTP_RECVNXTINFO));
#endif
#ifdef SCTP_RECVRCVINFO
printf("SCTP_RECVRCVINFO %llu\n", (unsigned long long)(long long)(SCTP_RECVRCVINFO));
#endif
#ifdef SCTP_REUSE_PORT
printf("SCTP_REUSE_PORT %llu\n", (unsigned long long)(long long)(SCTP_REUSE_PORT));
#endif
#ifdef SCTP_RTOINFO
printf("SCTP_RTOINFO %llu\n", (unsigned long long)(long long)(SCTP_RTOINFO));
#endif
#ifdef SCTP_SACK_IMMEDIATELY
printf("SCTP_SACK_IMMEDIATELY %llu\n", (unsigned long long)(long long)(SCTP_SACK_IMMEDIATELY));
#endif
#ifdef SCTP_SENDALL
printf("SCTP_SENDALL %llu\n", (unsigned long long)(long long)(SCTP_SENDALL));
#endif
#ifdef SCTP_SET_PEER_PRIMARY_ADDR
printf("SCTP_SET_PEER_PRIMARY_ADDR %llu\n", (unsigned long long)(long long)(SCTP_SET_PEER_PRIMARY_ADDR));
#endif
#ifdef SCTP_SNDINFO
printf("SCTP_SNDINFO %llu\n", (unsigned long long)(long long)(SCTP_SNDINFO));
#endif
#ifdef SCTP_SNDRCV
printf("SCTP_SNDRCV %llu\n", (unsigned long long)(long long)(SCTP_SNDRCV));
#endif
#ifdef SCTP_STATUS
printf("SCTP_STATUS %llu\n", (unsigned long long)(long long)(SCTP_STATUS));
#endif
#ifdef SCTP_STREAM_RESET_INCOMING
printf("SCTP_STREAM_RESET_INCOMING %llu\n", (unsigned long long)(long long)(SCTP_STREAM_RESET_INCOMING));
#endif
#ifdef SCTP_STREAM_RESET_OUTGOING
printf("SCTP_STREAM_RESET_OUTGOING %llu\n", (unsigned long long)(long long)(SCTP_STREAM_RESET_OUTGOING));
#endif
#ifdef SCTP_UNORDERED
printf("SCTP_UNORDERED %llu\n", (unsigned long long)(long long)(SCTP_UNORDERED));
#endif
#ifdef SECBIT_EXEC_DENY_INTERACTIVE
printf("SECBIT_EXEC_DENY_INTERACTIVE %llu\n", (unsigned long long)(long long)(SECBIT_EXEC_DENY_INTERACTIVE));
#endif
#ifdef SECBIT_EXEC_DENY_INTERACTIVE_LOCKED
printf("SECBIT_EXEC_DENY_INTERACTIVE_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_EXEC_DENY_INTERACTIVE_LOCKED));
#endif
#ifdef SECBIT_EXEC_RESTRICT_FILE
printf("SECBIT_EXEC_RESTRICT_FILE %llu\n", (unsigned long long)(long long)(SECBIT_EXEC_RESTRICT_FILE));
#endif
#ifdef SECBIT_EXEC_RESTRICT_FILE_LOCKED
printf("SECBIT_EXEC_RESTRICT_FILE_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_EXEC_RESTRICT_FILE_LOCKED));
#endif
#ifdef SECBIT_KEEP_CAPS
printf("SECBIT_KEEP_CAPS %llu\n", (unsigned long long)(long long)(SECBIT_KEEP_CAPS));
#endif
#ifdef SECBIT_KEEP_CAPS_LOCKED
printf("SECBIT_KEEP_CAPS_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_KEEP_CAPS_LOCKED));
#endif
#ifdef SECBIT_NOROOT
printf("SECBIT_NOROOT %llu\n", (unsigned long long)(long long)(SECBIT_NOROOT));
#endif
#ifdef SECBIT_NOROOT_LOCKED
printf("SECBIT_NOROOT_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_NOROOT_LOCKED));
#endif
#ifdef SECBIT_NO_CAP_AMBIENT_RAISE
printf("SECBIT_NO_CAP_AMBIENT_RAISE %llu\n", (unsigned long long)(long long)(SECBIT_NO_CAP_AMBIENT_RAISE));
#endif
#ifdef SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED
printf("SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED));
#endif
#ifdef SECBIT_NO_SETUID_FIXUP
printf("SECBIT_NO_SETUID_FIXUP %llu\n", (unsigned long long)(long long)(SECBIT_NO_SETUID_FIXUP));
#endif
#ifdef SECBIT_NO_SETUID_FIXUP_LOCKED
printf("SECBIT_NO_SETUID_FIXUP_LOCKED %llu\n", (unsigned long long)(long long)(SECBIT_NO_SETUID_FIXUP_LOCKED));
#endif
#ifdef SECCOMP_ADDFD_FLAG_SEND
printf("SECCOMP_ADDFD_FLAG_SEND %llu\n", (unsigned long long)(long long)(SECCOMP_ADDFD_FLAG_SEND));
#endif
#ifdef SECCOMP_ADDFD_FLAG_SETFD
printf("SECCOMP_ADDFD_FLAG_SETFD %llu\n", (unsigned long long)(long long)(SECCOMP_ADDFD_FLAG_SETFD));
#endif
#ifdef SECCOMP_FILTER_FLAG_LOG
printf("SECCOMP_FILTER_FLAG_LOG %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_LOG));
#endif
#ifdef SECCOMP_FILTER_FLAG_NEW_LISTENER
printf("SECCOMP_FILTER_FLAG_NEW_LISTENER %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_NEW_LISTENER));
#endif
#ifdef SECCOMP_FILTER_FLAG_SPEC_ALLOW
printf("SECCOMP_FILTER_FLAG_SPEC_ALLOW %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_SPEC_ALLOW));
#endif
#ifdef SECCOMP_FILTER_FLAG_TSYNC
printf("SECCOMP_FILTER_FLAG_TSYNC %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_TSYNC));
#endif
#ifdef SECCOMP_FILTER_FLAG_TSYNC_ESRCH
printf("SECCOMP_FILTER_FLAG_TSYNC_ESRCH %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_TSYNC_ESRCH));
#endif
#ifdef SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV
printf("SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV %llu\n", (unsigned long long)(long long)(SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV));
#endif
#ifdef SECCOMP_GET_ACTION_AVAIL
printf("SECCOMP_GET_ACTION_AVAIL %llu\n", (unsigned long long)(long long)(SECCOMP_GET_ACTION_AVAIL));
#endif
#ifdef SECCOMP_GET_NOTIF_SIZES
printf("SECCOMP_GET_NOTIF_SIZES %llu\n", (unsigned long long)(long long)(SECCOMP_GET_NOTIF_SIZES));
#endif
#ifdef SECCOMP_MODE_DISABLED
printf("SECCOMP_MODE_DISABLED %llu\n", (unsigned long long)(long long)(SECCOMP_MODE_DISABLED));
#endif
#ifdef SECCOMP_MODE_FILTER
printf("SECCOMP_MODE_FILTER %llu\n", (unsigned long long)(long long)(SECCOMP_MODE_FILTER));
#endif
#ifdef SECCOMP_MODE_STRICT
printf("SECCOMP_MODE_STRICT %llu\n", (unsigned long long)(long long)(SECCOMP_MODE_STRICT));
#endif
#ifdef SECCOMP_RET_ACTION
printf("SECCOMP_RET_ACTION %llu\n", (unsigned long long)(long long)(SECCOMP_RET_ACTION));
#endif
#ifdef SECCOMP_RET_ACTION_FULL
printf("SECCOMP_RET_ACTION_FULL %llu\n", (unsigned long long)(long long)(SECCOMP_RET_ACTION_FULL));
#endif
#ifdef SECCOMP_RET_ALLOW
printf("SECCOMP_RET_ALLOW %llu\n", (unsigned long long)(long long)(SECCOMP_RET_ALLOW));
#endif
#ifdef SECCOMP_RET_DATA
printf("SECCOMP_RET_DATA %llu\n", (unsigned long long)(long long)(SECCOMP_RET_DATA));
#endif
#ifdef SECCOMP_RET_ERRNO
printf("SECCOMP_RET_ERRNO %llu\n", (unsigned long long)(long long)(SECCOMP_RET_ERRNO));
#endif
#ifdef SECCOMP_RET_KILL
printf("SECCOMP_RET_KILL %llu\n", (unsigned long long)(long long)(SECCOMP_RET_KILL));
#endif
#ifdef SECCOMP_RET_KILL_PROCESS
printf("SECCOMP_RET_KILL_PROCESS %llu\n", (unsigned long long)(long long)(SECCOMP_RET_KILL_PROCESS));
#endif
#ifdef SECCOMP_RET_KILL_THREAD
printf("SECCOMP_RET_KILL_THREAD %llu\n", (unsigned long long)(long long)(SECCOMP_RET_KILL_THREAD));
#endif
#ifdef SECCOMP_RET_LOG
printf("SECCOMP_RET_LOG %llu\n", (unsigned long long)(long long)(SECCOMP_RET_LOG));
#endif
#ifdef SECCOMP_RET_TRACE
printf("SECCOMP_RET_TRACE %llu\n", (unsigned long long)(long long)(SECCOMP_RET_TRACE));
#endif
#ifdef SECCOMP_RET_TRAP
printf("SECCOMP_RET_TRAP %llu\n", (unsigned long long)(long long)(SECCOMP_RET_TRAP));
#endif
#ifdef SECCOMP_RET_USER_NOTIF
printf("SECCOMP_RET_USER_NOTIF %llu\n", (unsigned long long)(long long)(SECCOMP_RET_USER_NOTIF));
#endif
#ifdef SECCOMP_SET_MODE_FILTER
printf("SECCOMP_SET_MODE_FILTER %llu\n", (unsigned long long)(long long)(SECCOMP_SET_MODE_FILTER));
#endif
#ifdef SECCOMP_SET_MODE_STRICT
printf("SECCOMP_SET_MODE_STRICT %llu\n", (unsigned long long)(long long)(SECCOMP_SET_MODE_STRICT));
#endif
#ifdef SECCOMP_USER_NOTIF_FLAG_CONTINUE
printf("SECCOMP_USER_NOTIF_FLAG_CONTINUE %llu\n", (unsigned long long)(long long)(SECCOMP_USER_NOTIF_FLAG_CONTINUE));
#endif
#ifdef SECUREBITS_DEFAULT
printf("SECUREBITS_DEFAULT %llu\n", (unsigned long long)(long long)(SECUREBITS_DEFAULT));
#endif
#ifdef SECURE_ALL_BITS
printf("SECURE_ALL_BITS %llu\n", (unsigned long long)(long long)(SECURE_ALL_BITS));
#endif
#ifdef SECURE_ALL_LOCKS
printf("SECURE_ALL_LOCKS %llu\n", (unsigned long long)(long long)(SECURE_ALL_LOCKS));
#endif
#ifdef SECURE_ALL_UNPRIVILEGED
printf("SECURE_ALL_UNPRIVILEGED %llu\n", (unsigned long long)(long long)(SECURE_ALL_UNPRIVILEGED));
#endif
#ifdef SECURITYFS_MAGIC
printf("SECURITYFS_MAGIC %llu\n", (unsigned long long)(long long)(SECURITYFS_MAGIC));
#endif
#ifdef SEEK_CUR
printf("SEEK_CUR %llu\n", (unsigned long long)(long long)(SEEK_CUR));
#endif
#ifdef SEEK_DATA
printf("SEEK_DATA %llu\n", (unsigned long long)(long long)(SEEK_DATA));
#endif
#ifdef SEEK_END
printf("SEEK_END %llu\n", (unsigned long long)(long long)(SEEK_END));
#endif
#ifdef SEEK_HOLE
printf("SEEK_HOLE %llu\n", (unsigned long long)(long long)(SEEK_HOLE));
#endif
#ifdef SEEK_SET
printf("SEEK_SET %llu\n", (unsigned long long)(long long)(SEEK_SET));
#endif
#ifdef SELFMAG
printf("SELFMAG %llu\n", (unsigned long long)(long long)(SELFMAG));
#endif
#ifdef SELINUX_MAGIC
printf("SELINUX_MAGIC %llu\n", (unsigned long long)(long long)(SELINUX_MAGIC));
#endif
#ifdef SEM_INFO
printf("SEM_INFO %llu\n", (unsigned long long)(long long)(SEM_INFO));
#endif
#ifdef SEM_STAT
printf("SEM_STAT %llu\n", (unsigned long long)(long long)(SEM_STAT));
#endif
#ifdef SEM_STAT_ANY
printf("SEM_STAT_ANY %llu\n", (unsigned long long)(long long)(SEM_STAT_ANY));
#endif
#ifdef SEM_UNDO
printf("SEM_UNDO %llu\n", (unsigned long long)(long long)(SEM_UNDO));
#endif
#ifdef SETALL
printf("SETALL %llu\n", (unsigned long long)(long long)(SETALL));
#endif
#ifdef SETVAL
printf("SETVAL %llu\n", (unsigned long long)(long long)(SETVAL));
#endif
#ifdef SFD_CLOEXEC
printf("SFD_CLOEXEC %llu\n", (unsigned long long)(long long)(SFD_CLOEXEC));
#endif
#ifdef SFD_NONBLOCK
printf("SFD_NONBLOCK %llu\n", (unsigned long long)(long long)(SFD_NONBLOCK));
#endif
#ifdef SHM_EXEC
printf("SHM_EXEC %llu\n", (unsigned long long)(long long)(SHM_EXEC));
#endif
#ifdef SHM_HUGETLB
printf("SHM_HUGETLB %llu\n", (unsigned long long)(long long)(SHM_HUGETLB));
#endif
#ifdef SHM_LOCK
printf("SHM_LOCK %llu\n", (unsigned long long)(long long)(SHM_LOCK));
#endif
#ifdef SHM_NORESERVE
printf("SHM_NORESERVE %llu\n", (unsigned long long)(long long)(SHM_NORESERVE));
#endif
#ifdef SHM_R
printf("SHM_R %llu\n", (unsigned long long)(long long)(SHM_R));
#endif
#ifdef SHM_RDONLY
printf("SHM_RDONLY %llu\n", (unsigned long long)(long long)(SHM_RDONLY));
#endif
#ifdef SHM_REMAP
printf("SHM_REMAP %llu\n", (unsigned long long)(long long)(SHM_REMAP));
#endif
#ifdef SHM_RND
printf("SHM_RND %llu\n", (unsigned long long)(long long)(SHM_RND));
#endif
#ifdef SHM_UNLOCK
printf("SHM_UNLOCK %llu\n", (unsigned long long)(long long)(SHM_UNLOCK));
#endif
#ifdef SHM_W
printf("SHM_W %llu\n", (unsigned long long)(long long)(SHM_W));
#endif
#ifdef SHORT_INODE
printf("SHORT_INODE %llu\n", (unsigned long long)(long long)(SHORT_INODE));
#endif
#ifdef SHUT_RD
printf("SHUT_RD %llu\n", (unsigned long long)(long long)(SHUT_RD));
#endif
#ifdef SHUT_RDWR
printf("SHUT_RDWR %llu\n", (unsigned long long)(long long)(SHUT_RDWR));
#endif
#ifdef SHUT_WR
printf("SHUT_WR %llu\n", (unsigned long long)(long long)(SHUT_WR));
#endif
#ifdef SIGABRT
printf("SIGABRT %llu\n", (unsigned long long)(long long)(SIGABRT));
#endif
#ifdef SIGALRM
printf("SIGALRM %llu\n", (unsigned long long)(long long)(SIGALRM));
#endif
#ifdef SIGBUS
printf("SIGBUS %llu\n", (unsigned long long)(long long)(SIGBUS));
#endif
#ifdef SIGCHLD
printf("SIGCHLD %llu\n", (unsigned long long)(long long)(SIGCHLD));
#endif
#ifdef SIGCONT
printf("SIGCONT %llu\n", (unsigned long long)(long long)(SIGCONT));
#endif
#ifdef SIGEV_NONE
printf("SIGEV_NONE %llu\n", (unsigned long long)(long long)(SIGEV_NONE));
#endif
#ifdef SIGEV_SIGNAL
printf("SIGEV_SIGNAL %llu\n", (unsigned long long)(long long)(SIGEV_SIGNAL));
#endif
#ifdef SIGEV_THREAD
printf("SIGEV_THREAD %llu\n", (unsigned long long)(long long)(SIGEV_THREAD));
#endif
#ifdef SIGEV_THREAD_ID
printf("SIGEV_THREAD_ID %llu\n", (unsigned long long)(long long)(SIGEV_THREAD_ID));
#endif
#ifdef SIGFPE
printf("SIGFPE %llu\n", (unsigned long long)(long long)(SIGFPE));
#endif
#ifdef SIGHUP
printf("SIGHUP %llu\n", (unsigned long long)(long long)(SIGHUP));
#endif
#ifdef SIGILL
printf("SIGILL %llu\n", (unsigned long long)(long long)(SIGILL));
#endif
#ifdef SIGINT
printf("SIGINT %llu\n", (unsigned long long)(long long)(SIGINT));
#endif
#ifdef SIGIO
printf("SIGIO %llu\n", (unsigned long long)(long long)(SIGIO));
#endif
#ifdef SIGIOT
printf("SIGIOT %llu\n", (unsigned long long)(long long)(SIGIOT));
#endif
#ifdef SIGKILL
printf("SIGKILL %llu\n", (unsigned long long)(long long)(SIGKILL));
#endif
#ifdef SIGPIPE
printf("SIGPIPE %llu\n", (unsigned long long)(long long)(SIGPIPE));
#endif
#ifdef SIGPOLL
printf("SIGPOLL %llu\n", (unsigned long long)(long long)(SIGPOLL));
#endif
#ifdef SIGPROF
printf("SIGPROF %llu\n", (unsigned long long)(long long)(SIGPROF));
#endif
#ifdef SIGPWR
printf("SIGPWR %llu\n", (unsigned long long)(long long)(SIGPWR));
#endif
#ifdef SIGQUIT
printf("SIGQUIT %llu\n", (unsigned long long)(long long)(SIGQUIT));
#endif
#ifdef SIGSEGV
printf("SIGSEGV %llu\n", (unsigned long long)(long long)(SIGSEGV));
#endif
#ifdef SIGSTKFLT
printf("SIGSTKFLT %llu\n", (unsigned long long)(long long)(SIGSTKFLT));
#endif
#ifdef SIGSTKSZ
printf("SIGSTKSZ %llu\n", (unsigned long long)(long long)(SIGSTKSZ));
#endif
#ifdef SIGSTOP
printf("SIGSTOP %llu\n", (unsigned long long)(long long)(SIGSTOP));
#endif
#ifdef SIGSYS
printf("SIGSYS %llu\n", (unsigned long long)(long long)(SIGSYS));
#endif
#ifdef SIGTERM
printf("SIGTERM %llu\n", (unsigned long long)(long long)(SIGTERM));
#endif
#ifdef SIGTRAP
printf("SIGTRAP %llu\n", (unsigned long long)(long long)(SIGTRAP));
#endif
#ifdef SIGTSTP
printf("SIGTSTP %llu\n", (unsigned long long)(long long)(SIGTSTP));
#endif
#ifdef SIGTTIN
printf("SIGTTIN %llu\n", (unsigned long long)(long long)(SIGTTIN));
#endif
#ifdef SIGTTOU
printf("SIGTTOU %llu\n", (unsigned long long)(long long)(SIGTTOU));
#endif
#ifdef SIGURG
printf("SIGURG %llu\n", (unsigned long long)(long long)(SIGURG));
#endif
#ifdef SIGUSR1
printf("SIGUSR1 %llu\n", (unsigned long long)(long long)(SIGUSR1));
#endif
#ifdef SIGUSR2
printf("SIGUSR2 %llu\n", (unsigned long long)(long long)(SIGUSR2));
#endif
#ifdef SIGVTALRM
printf("SIGVTALRM %llu\n", (unsigned long long)(long long)(SIGVTALRM));
#endif
#ifdef SIGWINCH
printf("SIGWINCH %llu\n", (unsigned long long)(long long)(SIGWINCH));
#endif
#ifdef SIGXCPU
printf("SIGXCPU %llu\n", (unsigned long long)(long long)(SIGXCPU));
#endif
#ifdef SIGXFSZ
printf("SIGXFSZ %llu\n", (unsigned long long)(long long)(SIGXFSZ));
#endif
#ifdef SIG_BLOCK
printf("SIG_BLOCK %llu\n", (unsigned long long)(long long)(SIG_BLOCK));
#endif
#ifdef SIG_DFL
printf("SIG_DFL %llu\n", (unsigned long long)(long long)(SIG_DFL));
#endif
#ifdef SIG_ERR
printf("SIG_ERR %llu\n", (unsigned long long)(long long)(SIG_ERR));
#endif
#ifdef SIG_IGN
printf("SIG_IGN %llu\n", (unsigned long long)(long long)(SIG_IGN));
#endif
#ifdef SIG_SETMASK
printf("SIG_SETMASK %llu\n", (unsigned long long)(long long)(SIG_SETMASK));
#endif
#ifdef SIG_UNBLOCK
printf("SIG_UNBLOCK %llu\n", (unsigned long long)(long long)(SIG_UNBLOCK));
#endif
#ifdef SIOCADDMULTI
printf("SIOCADDMULTI %llu\n", (unsigned long long)(long long)(SIOCADDMULTI));
#endif
#ifdef SIOCADDRT
printf("SIOCADDRT %llu\n", (unsigned long long)(long long)(SIOCADDRT));
#endif
#ifdef SIOCDARP
printf("SIOCDARP %llu\n", (unsigned long long)(long long)(SIOCDARP));
#endif
#ifdef SIOCDELMULTI
printf("SIOCDELMULTI %llu\n", (unsigned long long)(long long)(SIOCDELMULTI));
#endif
#ifdef SIOCDELRT
printf("SIOCDELRT %llu\n", (unsigned long long)(long long)(SIOCDELRT));
#endif
#ifdef SIOCDIFADDR
printf("SIOCDIFADDR %llu\n", (unsigned long long)(long long)(SIOCDIFADDR));
#endif
#ifdef SIOCDRARP
printf("SIOCDRARP %llu\n", (unsigned long long)(long long)(SIOCDRARP));
#endif
#ifdef SIOCETHTOOL
printf("SIOCETHTOOL %llu\n", (unsigned long long)(long long)(SIOCETHTOOL));
#endif
#ifdef SIOCGARP
printf("SIOCGARP %llu\n", (unsigned long long)(long long)(SIOCGARP));
#endif
#ifdef SIOCGHWTSTAMP
printf("SIOCGHWTSTAMP %llu\n", (unsigned long long)(long long)(SIOCGHWTSTAMP));
#endif
#ifdef SIOCGIFADDR
printf("SIOCGIFADDR %llu\n", (unsigned long long)(long long)(SIOCGIFADDR));
#endif
#ifdef SIOCGIFBR
printf("SIOCGIFBR %llu\n", (unsigned long long)(long long)(SIOCGIFBR));
#endif
#ifdef SIOCGIFBRDADDR
printf("SIOCGIFBRDADDR %llu\n", (unsigned long long)(long long)(SIOCGIFBRDADDR));
#endif
#ifdef SIOCGIFCONF
printf("SIOCGIFCONF %llu\n", (unsigned long long)(long long)(SIOCGIFCONF));
#endif
#ifdef SIOCGIFCOUNT
printf("SIOCGIFCOUNT %llu\n", (unsigned long long)(long long)(SIOCGIFCOUNT));
#endif
#ifdef SIOCGIFDSTADDR
printf("SIOCGIFDSTADDR %llu\n", (unsigned long long)(long long)(SIOCGIFDSTADDR));
#endif
#ifdef SIOCGIFENCAP
printf("SIOCGIFENCAP %llu\n", (unsigned long long)(long long)(SIOCGIFENCAP));
#endif
#ifdef SIOCGIFFLAGS
printf("SIOCGIFFLAGS %llu\n", (unsigned long long)(long long)(SIOCGIFFLAGS));
#endif
#ifdef SIOCGIFHWADDR
printf("SIOCGIFHWADDR %llu\n", (unsigned long long)(long long)(SIOCGIFHWADDR));
#endif
#ifdef SIOCGIFINDEX
printf("SIOCGIFINDEX %llu\n", (unsigned long long)(long long)(SIOCGIFINDEX));
#endif
#ifdef SIOCGIFMAP
printf("SIOCGIFMAP %llu\n", (unsigned long long)(long long)(SIOCGIFMAP));
#endif
#ifdef SIOCGIFMEM
printf("SIOCGIFMEM %llu\n", (unsigned long long)(long long)(SIOCGIFMEM));
#endif
#ifdef SIOCGIFMETRIC
printf("SIOCGIFMETRIC %llu\n", (unsigned long long)(long long)(SIOCGIFMETRIC));
#endif
#ifdef SIOCGIFMTU
printf("SIOCGIFMTU %llu\n", (unsigned long long)(long long)(SIOCGIFMTU));
#endif
#ifdef SIOCGIFNAME
printf("SIOCGIFNAME %llu\n", (unsigned long long)(long long)(SIOCGIFNAME));
#endif
#ifdef SIOCGIFNETMASK
printf("SIOCGIFNETMASK %llu\n", (unsigned long long)(long long)(SIOCGIFNETMASK));
#endif
#ifdef SIOCGIFPFLAGS
printf("SIOCGIFPFLAGS %llu\n", (unsigned long long)(long long)(SIOCGIFPFLAGS));
#endif
#ifdef SIOCGIFSLAVE
printf("SIOCGIFSLAVE %llu\n", (unsigned long long)(long long)(SIOCGIFSLAVE));
#endif
#ifdef SIOCGIFTXQLEN
printf("SIOCGIFTXQLEN %llu\n", (unsigned long long)(long long)(SIOCGIFTXQLEN));
#endif
#ifdef SIOCGIWAP
printf("SIOCGIWAP %llu\n", (unsigned long long)(long long)(SIOCGIWAP));
#endif
#ifdef SIOCGIWAPLIST
printf("SIOCGIWAPLIST %llu\n", (unsigned long long)(long long)(SIOCGIWAPLIST));
#endif
#ifdef SIOCGIWAUTH
printf("SIOCGIWAUTH %llu\n", (unsigned long long)(long long)(SIOCGIWAUTH));
#endif
#ifdef SIOCGIWENCODE
printf("SIOCGIWENCODE %llu\n", (unsigned long long)(long long)(SIOCGIWENCODE));
#endif
#ifdef SIOCGIWENCODEEXT
printf("SIOCGIWENCODEEXT %llu\n", (unsigned long long)(long long)(SIOCGIWENCODEEXT));
#endif
#ifdef SIOCGIWESSID
printf("SIOCGIWESSID %llu\n", (unsigned long long)(long long)(SIOCGIWESSID));
#endif
#ifdef SIOCGIWFRAG
printf("SIOCGIWFRAG %llu\n", (unsigned long long)(long long)(SIOCGIWFRAG));
#endif
#ifdef SIOCGIWFREQ
printf("SIOCGIWFREQ %llu\n", (unsigned long long)(long long)(SIOCGIWFREQ));
#endif
#ifdef SIOCGIWGENIE
printf("SIOCGIWGENIE %llu\n", (unsigned long long)(long long)(SIOCGIWGENIE));
#endif
#ifdef SIOCGIWMODE
printf("SIOCGIWMODE %llu\n", (unsigned long long)(long long)(SIOCGIWMODE));
#endif
#ifdef SIOCGIWNAME
printf("SIOCGIWNAME %llu\n", (unsigned long long)(long long)(SIOCGIWNAME));
#endif
#ifdef SIOCGIWNICKN
printf("SIOCGIWNICKN %llu\n", (unsigned long long)(long long)(SIOCGIWNICKN));
#endif
#ifdef SIOCGIWNWID
printf("SIOCGIWNWID %llu\n", (unsigned long long)(long long)(SIOCGIWNWID));
#endif
#ifdef SIOCGIWPOWER
printf("SIOCGIWPOWER %llu\n", (unsigned long long)(long long)(SIOCGIWPOWER));
#endif
#ifdef SIOCGIWPRIV
printf("SIOCGIWPRIV %llu\n", (unsigned long long)(long long)(SIOCGIWPRIV));
#endif
#ifdef SIOCGIWRANGE
printf("SIOCGIWRANGE %llu\n", (unsigned long long)(long long)(SIOCGIWRANGE));
#endif
#ifdef SIOCGIWRATE
printf("SIOCGIWRATE %llu\n", (unsigned long long)(long long)(SIOCGIWRATE));
#endif
#ifdef SIOCGIWRETRY
printf("SIOCGIWRETRY %llu\n", (unsigned long long)(long long)(SIOCGIWRETRY));
#endif
#ifdef SIOCGIWRTS
printf("SIOCGIWRTS %llu\n", (unsigned long long)(long long)(SIOCGIWRTS));
#endif
#ifdef SIOCGIWSCAN
printf("SIOCGIWSCAN %llu\n", (unsigned long long)(long long)(SIOCGIWSCAN));
#endif
#ifdef SIOCGIWSENS
printf("SIOCGIWSENS %llu\n", (unsigned long long)(long long)(SIOCGIWSENS));
#endif
#ifdef SIOCGIWSPY
printf("SIOCGIWSPY %llu\n", (unsigned long long)(long long)(SIOCGIWSPY));
#endif
#ifdef SIOCGIWSTATS
printf("SIOCGIWSTATS %llu\n", (unsigned long long)(long long)(SIOCGIWSTATS));
#endif
#ifdef SIOCGIWTHRSPY
printf("SIOCGIWTHRSPY %llu\n", (unsigned long long)(long long)(SIOCGIWTHRSPY));
#endif
#ifdef SIOCGIWTXPOW
printf("SIOCGIWTXPOW %llu\n", (unsigned long long)(long long)(SIOCGIWTXPOW));
#endif
#ifdef SIOCGMIIPHY
printf("SIOCGMIIPHY %llu\n", (unsigned long long)(long long)(SIOCGMIIPHY));
#endif
#ifdef SIOCGMIIREG
printf("SIOCGMIIREG %llu\n", (unsigned long long)(long long)(SIOCGMIIREG));
#endif
#ifdef SIOCGRARP
printf("SIOCGRARP %llu\n", (unsigned long long)(long long)(SIOCGRARP));
#endif
#ifdef SIOCGSKNS
printf("SIOCGSKNS %llu\n", (unsigned long long)(long long)(SIOCGSKNS));
#endif
#ifdef SIOCIWFIRST
printf("SIOCIWFIRST %llu\n", (unsigned long long)(long long)(SIOCIWFIRST));
#endif
#ifdef SIOCIWFIRSTPRIV
printf("SIOCIWFIRSTPRIV %llu\n", (unsigned long long)(long long)(SIOCIWFIRSTPRIV));
#endif
#ifdef SIOCIWLAST
printf("SIOCIWLAST %llu\n", (unsigned long long)(long long)(SIOCIWLAST));
#endif
#ifdef SIOCIWLASTPRIV
printf("SIOCIWLASTPRIV %llu\n", (unsigned long long)(long long)(SIOCIWLASTPRIV));
#endif
#ifdef SIOCOUTQNSD
printf("SIOCOUTQNSD %llu\n", (unsigned long long)(long long)(SIOCOUTQNSD));
#endif
#ifdef SIOCSARP
printf("SIOCSARP %llu\n", (unsigned long long)(long long)(SIOCSARP));
#endif
#ifdef SIOCSHWTSTAMP
printf("SIOCSHWTSTAMP %llu\n", (unsigned long long)(long long)(SIOCSHWTSTAMP));
#endif
#ifdef SIOCSIFADDR
printf("SIOCSIFADDR %llu\n", (unsigned long long)(long long)(SIOCSIFADDR));
#endif
#ifdef SIOCSIFBR
printf("SIOCSIFBR %llu\n", (unsigned long long)(long long)(SIOCSIFBR));
#endif
#ifdef SIOCSIFBRDADDR
printf("SIOCSIFBRDADDR %llu\n", (unsigned long long)(long long)(SIOCSIFBRDADDR));
#endif
#ifdef SIOCSIFDSTADDR
printf("SIOCSIFDSTADDR %llu\n", (unsigned long long)(long long)(SIOCSIFDSTADDR));
#endif
#ifdef SIOCSIFENCAP
printf("SIOCSIFENCAP %llu\n", (unsigned long long)(long long)(SIOCSIFENCAP));
#endif
#ifdef SIOCSIFFLAGS
printf("SIOCSIFFLAGS %llu\n", (unsigned long long)(long long)(SIOCSIFFLAGS));
#endif
#ifdef SIOCSIFHWADDR
printf("SIOCSIFHWADDR %llu\n", (unsigned long long)(long long)(SIOCSIFHWADDR));
#endif
#ifdef SIOCSIFHWBROADCAST
printf("SIOCSIFHWBROADCAST %llu\n", (unsigned long long)(long long)(SIOCSIFHWBROADCAST));
#endif
#ifdef SIOCSIFLINK
printf("SIOCSIFLINK %llu\n", (unsigned long long)(long long)(SIOCSIFLINK));
#endif
#ifdef SIOCSIFMAP
printf("SIOCSIFMAP %llu\n", (unsigned long long)(long long)(SIOCSIFMAP));
#endif
#ifdef SIOCSIFMEM
printf("SIOCSIFMEM %llu\n", (unsigned long long)(long long)(SIOCSIFMEM));
#endif
#ifdef SIOCSIFMETRIC
printf("SIOCSIFMETRIC %llu\n", (unsigned long long)(long long)(SIOCSIFMETRIC));
#endif
#ifdef SIOCSIFMTU
printf("SIOCSIFMTU %llu\n", (unsigned long long)(long long)(SIOCSIFMTU));
#endif
#ifdef SIOCSIFNAME
printf("SIOCSIFNAME %llu\n", (unsigned long long)(long long)(SIOCSIFNAME));
#endif
#ifdef SIOCSIFNETMASK
printf("SIOCSIFNETMASK %llu\n", (unsigned long long)(long long)(SIOCSIFNETMASK));
#endif
#ifdef SIOCSIFPFLAGS
printf("SIOCSIFPFLAGS %llu\n", (unsigned long long)(long long)(SIOCSIFPFLAGS));
#endif
#ifdef SIOCSIFSLAVE
printf("SIOCSIFSLAVE %llu\n", (unsigned long long)(long long)(SIOCSIFSLAVE));
#endif
#ifdef SIOCSIFTXQLEN
printf("SIOCSIFTXQLEN %llu\n", (unsigned long long)(long long)(SIOCSIFTXQLEN));
#endif
#ifdef SIOCSIWAP
printf("SIOCSIWAP %llu\n", (unsigned long long)(long long)(SIOCSIWAP));
#endif
#ifdef SIOCSIWAUTH
printf("SIOCSIWAUTH %llu\n", (unsigned long long)(long long)(SIOCSIWAUTH));
#endif
#ifdef SIOCSIWCOMMIT
printf("SIOCSIWCOMMIT %llu\n", (unsigned long long)(long long)(SIOCSIWCOMMIT));
#endif
#ifdef SIOCSIWENCODE
printf("SIOCSIWENCODE %llu\n", (unsigned long long)(long long)(SIOCSIWENCODE));
#endif
#ifdef SIOCSIWENCODEEXT
printf("SIOCSIWENCODEEXT %llu\n", (unsigned long long)(long long)(SIOCSIWENCODEEXT));
#endif
#ifdef SIOCSIWESSID
printf("SIOCSIWESSID %llu\n", (unsigned long long)(long long)(SIOCSIWESSID));
#endif
#ifdef SIOCSIWFRAG
printf("SIOCSIWFRAG %llu\n", (unsigned long long)(long long)(SIOCSIWFRAG));
#endif
#ifdef SIOCSIWFREQ
printf("SIOCSIWFREQ %llu\n", (unsigned long long)(long long)(SIOCSIWFREQ));
#endif
#ifdef SIOCSIWGENIE
printf("SIOCSIWGENIE %llu\n", (unsigned long long)(long long)(SIOCSIWGENIE));
#endif
#ifdef SIOCSIWMLME
printf("SIOCSIWMLME %llu\n", (unsigned long long)(long long)(SIOCSIWMLME));
#endif
#ifdef SIOCSIWMODE
printf("SIOCSIWMODE %llu\n", (unsigned long long)(long long)(SIOCSIWMODE));
#endif
#ifdef SIOCSIWNICKN
printf("SIOCSIWNICKN %llu\n", (unsigned long long)(long long)(SIOCSIWNICKN));
#endif
#ifdef SIOCSIWNWID
printf("SIOCSIWNWID %llu\n", (unsigned long long)(long long)(SIOCSIWNWID));
#endif
#ifdef SIOCSIWPMKSA
printf("SIOCSIWPMKSA %llu\n", (unsigned long long)(long long)(SIOCSIWPMKSA));
#endif
#ifdef SIOCSIWPOWER
printf("SIOCSIWPOWER %llu\n", (unsigned long long)(long long)(SIOCSIWPOWER));
#endif
#ifdef SIOCSIWPRIV
printf("SIOCSIWPRIV %llu\n", (unsigned long long)(long long)(SIOCSIWPRIV));
#endif
#ifdef SIOCSIWRANGE
printf("SIOCSIWRANGE %llu\n", (unsigned long long)(long long)(SIOCSIWRANGE));
#endif
#ifdef SIOCSIWRATE
printf("SIOCSIWRATE %llu\n", (unsigned long long)(long long)(SIOCSIWRATE));
#endif
#ifdef SIOCSIWRETRY
printf("SIOCSIWRETRY %llu\n", (unsigned long long)(long long)(SIOCSIWRETRY));
#endif
#ifdef SIOCSIWRTS
printf("SIOCSIWRTS %llu\n", (unsigned long long)(long long)(SIOCSIWRTS));
#endif
#ifdef SIOCSIWSCAN
printf("SIOCSIWSCAN %llu\n", (unsigned long long)(long long)(SIOCSIWSCAN));
#endif
#ifdef SIOCSIWSENS
printf("SIOCSIWSENS %llu\n", (unsigned long long)(long long)(SIOCSIWSENS));
#endif
#ifdef SIOCSIWSPY
printf("SIOCSIWSPY %llu\n", (unsigned long long)(long long)(SIOCSIWSPY));
#endif
#ifdef SIOCSIWSTATS
printf("SIOCSIWSTATS %llu\n", (unsigned long long)(long long)(SIOCSIWSTATS));
#endif
#ifdef SIOCSIWTHRSPY
printf("SIOCSIWTHRSPY %llu\n", (unsigned long long)(long long)(SIOCSIWTHRSPY));
#endif
#ifdef SIOCSIWTXPOW
printf("SIOCSIWTXPOW %llu\n", (unsigned long long)(long long)(SIOCSIWTXPOW));
#endif
#ifdef SIOCSMIIREG
printf("SIOCSMIIREG %llu\n", (unsigned long long)(long long)(SIOCSMIIREG));
#endif
#ifdef SIOCSRARP
printf("SIOCSRARP %llu\n", (unsigned long long)(long long)(SIOCSRARP));
#endif
#ifdef SIOCWANDEV
printf("SIOCWANDEV %llu\n", (unsigned long long)(long long)(SIOCWANDEV));
#endif
#ifdef SIOGIFINDEX
printf("SIOGIFINDEX %llu\n", (unsigned long long)(long long)(SIOGIFINDEX));
#endif
#ifdef SI_ASYNCIO
printf("SI_ASYNCIO %llu\n", (unsigned long long)(long long)(SI_ASYNCIO));
#endif
#ifdef SI_ASYNCNL
printf("SI_ASYNCNL %llu\n", (unsigned long long)(long long)(SI_ASYNCNL));
#endif
#ifdef SI_DETHREAD
printf("SI_DETHREAD %llu\n", (unsigned long long)(long long)(SI_DETHREAD));
#endif
#ifdef SI_KERNEL
printf("SI_KERNEL %llu\n", (unsigned long long)(long long)(SI_KERNEL));
#endif
#ifdef SI_LOAD_SHIFT
printf("SI_LOAD_SHIFT %llu\n", (unsigned long long)(long long)(SI_LOAD_SHIFT));
#endif
#ifdef SI_MESGQ
printf("SI_MESGQ %llu\n", (unsigned long long)(long long)(SI_MESGQ));
#endif
#ifdef SI_QUEUE
printf("SI_QUEUE %llu\n", (unsigned long long)(long long)(SI_QUEUE));
#endif
#ifdef SI_SIGIO
printf("SI_SIGIO %llu\n", (unsigned long long)(long long)(SI_SIGIO));
#endif
#ifdef SI_TIMER
printf("SI_TIMER %llu\n", (unsigned long long)(long long)(SI_TIMER));
#endif
#ifdef SI_TKILL
printf("SI_TKILL %llu\n", (unsigned long long)(long long)(SI_TKILL));
#endif
#ifdef SI_USER
printf("SI_USER %llu\n", (unsigned long long)(long long)(SI_USER));
#endif
#ifdef SKF_AD_ALU_XOR_X
printf("SKF_AD_ALU_XOR_X %llu\n", (unsigned long long)(long long)(SKF_AD_ALU_XOR_X));
#endif
#ifdef SKF_AD_CPU
printf("SKF_AD_CPU %llu\n", (unsigned long long)(long long)(SKF_AD_CPU));
#endif
#ifdef SKF_AD_HATYPE
printf("SKF_AD_HATYPE %llu\n", (unsigned long long)(long long)(SKF_AD_HATYPE));
#endif
#ifdef SKF_AD_IFINDEX
printf("SKF_AD_IFINDEX %llu\n", (unsigned long long)(long long)(SKF_AD_IFINDEX));
#endif
#ifdef SKF_AD_MARK
printf("SKF_AD_MARK %llu\n", (unsigned long long)(long long)(SKF_AD_MARK));
#endif
#ifdef SKF_AD_MAX
printf("SKF_AD_MAX %llu\n", (unsigned long long)(long long)(SKF_AD_MAX));
#endif
#ifdef SKF_AD_NLATTR
printf("SKF_AD_NLATTR %llu\n", (unsigned long long)(long long)(SKF_AD_NLATTR));
#endif
#ifdef SKF_AD_NLATTR_NEST
printf("SKF_AD_NLATTR_NEST %llu\n", (unsigned long long)(long long)(SKF_AD_NLATTR_NEST));
#endif
#ifdef SKF_AD_OFF
printf("SKF_AD_OFF %llu\n", (unsigned long long)(long long)(SKF_AD_OFF));
#endif
#ifdef SKF_AD_PAY_OFFSET
printf("SKF_AD_PAY_OFFSET %llu\n", (unsigned long long)(long long)(SKF_AD_PAY_OFFSET));
#endif
#ifdef SKF_AD_PKTTYPE
printf("SKF_AD_PKTTYPE %llu\n", (unsigned long long)(long long)(SKF_AD_PKTTYPE));
#endif
#ifdef SKF_AD_PROTOCOL
printf("SKF_AD_PROTOCOL %llu\n", (unsigned long long)(long long)(SKF_AD_PROTOCOL));
#endif
#ifdef SKF_AD_QUEUE
printf("SKF_AD_QUEUE %llu\n", (unsigned long long)(long long)(SKF_AD_QUEUE));
#endif
#ifdef SKF_AD_RANDOM
printf("SKF_AD_RANDOM %llu\n", (unsigned long long)(long long)(SKF_AD_RANDOM));
#endif
#ifdef SKF_AD_RXHASH
printf("SKF_AD_RXHASH %llu\n", (unsigned long long)(long long)(SKF_AD_RXHASH));
#endif
#ifdef SKF_AD_VLAN_TAG
printf("SKF_AD_VLAN_TAG %llu\n", (unsigned long long)(long long)(SKF_AD_VLAN_TAG));
#endif
#ifdef SKF_AD_VLAN_TAG_PRESENT
printf("SKF_AD_VLAN_TAG_PRESENT %llu\n", (unsigned long long)(long long)(SKF_AD_VLAN_TAG_PRESENT));
#endif
#ifdef SKF_AD_VLAN_TPID
printf("SKF_AD_VLAN_TPID %llu\n", (unsigned long long)(long long)(SKF_AD_VLAN_TPID));
#endif
#ifdef SKF_LL_OFF
printf("SKF_LL_OFF %llu\n", (unsigned long long)(long long)(SKF_LL_OFF));
#endif
#ifdef SKF_NET_OFF
printf("SKF_NET_OFF %llu\n", (unsigned long long)(long long)(SKF_NET_OFF));
#endif
#ifdef SK_MEMINFO_BACKLOG
printf("SK_MEMINFO_BACKLOG %llu\n", (unsigned long long)(long long)(SK_MEMINFO_BACKLOG));
#endif
#ifdef SK_MEMINFO_DROPS
printf("SK_MEMINFO_DROPS %llu\n", (unsigned long long)(long long)(SK_MEMINFO_DROPS));
#endif
#ifdef SK_MEMINFO_FWD_ALLOC
printf("SK_MEMINFO_FWD_ALLOC %llu\n", (unsigned long long)(long long)(SK_MEMINFO_FWD_ALLOC));
#endif
#ifdef SK_MEMINFO_OPTMEM
printf("SK_MEMINFO_OPTMEM %llu\n", (unsigned long long)(long long)(SK_MEMINFO_OPTMEM));
#endif
#ifdef SK_MEMINFO_RCVBUF
printf("SK_MEMINFO_RCVBUF %llu\n", (unsigned long long)(long long)(SK_MEMINFO_RCVBUF));
#endif
#ifdef SK_MEMINFO_RMEM_ALLOC
printf("SK_MEMINFO_RMEM_ALLOC %llu\n", (unsigned long long)(long long)(SK_MEMINFO_RMEM_ALLOC));
#endif
#ifdef SK_MEMINFO_SNDBUF
printf("SK_MEMINFO_SNDBUF %llu\n", (unsigned long long)(long long)(SK_MEMINFO_SNDBUF));
#endif
#ifdef SK_MEMINFO_WMEM_ALLOC
printf("SK_MEMINFO_WMEM_ALLOC %llu\n", (unsigned long long)(long long)(SK_MEMINFO_WMEM_ALLOC));
#endif
#ifdef SK_MEMINFO_WMEM_QUEUED
printf("SK_MEMINFO_WMEM_QUEUED %llu\n", (unsigned long long)(long long)(SK_MEMINFO_WMEM_QUEUED));
#endif
#ifdef SMACK_MAGIC
printf("SMACK_MAGIC %llu\n", (unsigned long long)(long long)(SMACK_MAGIC));
#endif
#ifdef SMB_SUPER_MAGIC
printf("SMB_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(SMB_SUPER_MAGIC));
#endif
#ifdef SND_CNT
printf("SND_CNT %llu\n", (unsigned long long)(long long)(SND_CNT));
#endif
#ifdef SND_MAX
printf("SND_MAX %llu\n", (unsigned long long)(long long)(SND_MAX));
#endif
#ifdef SOCK_CLOEXEC
printf("SOCK_CLOEXEC %llu\n", (unsigned long long)(long long)(SOCK_CLOEXEC));
#endif
#ifdef SOCK_DCCP
printf("SOCK_DCCP %llu\n", (unsigned long long)(long long)(SOCK_DCCP));
#endif
#ifdef SOCK_DGRAM
printf("SOCK_DGRAM %llu\n", (unsigned long long)(long long)(SOCK_DGRAM));
#endif
#ifdef SOCK_NONBLOCK
printf("SOCK_NONBLOCK %llu\n", (unsigned long long)(long long)(SOCK_NONBLOCK));
#endif
#ifdef SOCK_PACKET
printf("SOCK_PACKET %llu\n", (unsigned long long)(long long)(SOCK_PACKET));
#endif
#ifdef SOCK_RAW
printf("SOCK_RAW %llu\n", (unsigned long long)(long long)(SOCK_RAW));
#endif
#ifdef SOCK_RDM
printf("SOCK_RDM %llu\n", (unsigned long long)(long long)(SOCK_RDM));
#endif
#ifdef SOCK_SEQPACKET
printf("SOCK_SEQPACKET %llu\n", (unsigned long long)(long long)(SOCK_SEQPACKET));
#endif
#ifdef SOCK_STREAM
printf("SOCK_STREAM %llu\n", (unsigned long long)(long long)(SOCK_STREAM));
#endif
#ifdef SOF_TIMESTAMPING_BIND_PHC
printf("SOF_TIMESTAMPING_BIND_PHC %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_BIND_PHC));
#endif
#ifdef SOF_TIMESTAMPING_OPT_CMSG
printf("SOF_TIMESTAMPING_OPT_CMSG %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_CMSG));
#endif
#ifdef SOF_TIMESTAMPING_OPT_ID
printf("SOF_TIMESTAMPING_OPT_ID %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_ID));
#endif
#ifdef SOF_TIMESTAMPING_OPT_ID_TCP
printf("SOF_TIMESTAMPING_OPT_ID_TCP %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_ID_TCP));
#endif
#ifdef SOF_TIMESTAMPING_OPT_PKTINFO
printf("SOF_TIMESTAMPING_OPT_PKTINFO %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_PKTINFO));
#endif
#ifdef SOF_TIMESTAMPING_OPT_RX_FILTER
printf("SOF_TIMESTAMPING_OPT_RX_FILTER %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_RX_FILTER));
#endif
#ifdef SOF_TIMESTAMPING_OPT_STATS
printf("SOF_TIMESTAMPING_OPT_STATS %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_STATS));
#endif
#ifdef SOF_TIMESTAMPING_OPT_TSONLY
printf("SOF_TIMESTAMPING_OPT_TSONLY %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_TSONLY));
#endif
#ifdef SOF_TIMESTAMPING_OPT_TX_SWHW
printf("SOF_TIMESTAMPING_OPT_TX_SWHW %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_OPT_TX_SWHW));
#endif
#ifdef SOF_TIMESTAMPING_RAW_HARDWARE
printf("SOF_TIMESTAMPING_RAW_HARDWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_RAW_HARDWARE));
#endif
#ifdef SOF_TIMESTAMPING_RX_HARDWARE
printf("SOF_TIMESTAMPING_RX_HARDWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_RX_HARDWARE));
#endif
#ifdef SOF_TIMESTAMPING_RX_SOFTWARE
printf("SOF_TIMESTAMPING_RX_SOFTWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_RX_SOFTWARE));
#endif
#ifdef SOF_TIMESTAMPING_SOFTWARE
printf("SOF_TIMESTAMPING_SOFTWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_SOFTWARE));
#endif
#ifdef SOF_TIMESTAMPING_SYS_HARDWARE
printf("SOF_TIMESTAMPING_SYS_HARDWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_SYS_HARDWARE));
#endif
#ifdef SOF_TIMESTAMPING_TX_ACK
printf("SOF_TIMESTAMPING_TX_ACK %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_TX_ACK));
#endif
#ifdef SOF_TIMESTAMPING_TX_HARDWARE
printf("SOF_TIMESTAMPING_TX_HARDWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_TX_HARDWARE));
#endif
#ifdef SOF_TIMESTAMPING_TX_SCHED
printf("SOF_TIMESTAMPING_TX_SCHED %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_TX_SCHED));
#endif
#ifdef SOF_TIMESTAMPING_TX_SOFTWARE
printf("SOF_TIMESTAMPING_TX_SOFTWARE %llu\n", (unsigned long long)(long long)(SOF_TIMESTAMPING_TX_SOFTWARE));
#endif
#ifdef SOF_TXTIME_DEADLINE_MODE
printf("SOF_TXTIME_DEADLINE_MODE %llu\n", (unsigned long long)(long long)(SOF_TXTIME_DEADLINE_MODE));
#endif
#ifdef SOF_TXTIME_REPORT_ERRORS
printf("SOF_TXTIME_REPORT_ERRORS %llu\n", (unsigned long long)(long long)(SOF_TXTIME_REPORT_ERRORS));
#endif
#ifdef SOL_AAL
printf("SOL_AAL %llu\n", (unsigned long long)(long long)(SOL_AAL));
#endif
#ifdef SOL_ALG
printf("SOL_ALG %llu\n", (unsigned long long)(long long)(SOL_ALG));
#endif
#ifdef SOL_ATM
printf("SOL_ATM %llu\n", (unsigned long long)(long long)(SOL_ATM));
#endif
#ifdef SOL_BLUETOOTH
printf("SOL_BLUETOOTH %llu\n", (unsigned long long)(long long)(SOL_BLUETOOTH));
#endif
#ifdef SOL_CAIF
printf("SOL_CAIF %llu\n", (unsigned long long)(long long)(SOL_CAIF));
#endif
#ifdef SOL_DCCP
printf("SOL_DCCP %llu\n", (unsigned long long)(long long)(SOL_DCCP));
#endif
#ifdef SOL_DECNET
printf("SOL_DECNET %llu\n", (unsigned long long)(long long)(SOL_DECNET));
#endif
#ifdef SOL_ICMPV6
printf("SOL_ICMPV6 %llu\n", (unsigned long long)(long long)(SOL_ICMPV6));
#endif
#ifdef SOL_IP
printf("SOL_IP %llu\n", (unsigned long long)(long long)(SOL_IP));
#endif
#ifdef SOL_IPV6
printf("SOL_IPV6 %llu\n", (unsigned long long)(long long)(SOL_IPV6));
#endif
#ifdef SOL_IRDA
printf("SOL_IRDA %llu\n", (unsigned long long)(long long)(SOL_IRDA));
#endif
#ifdef SOL_IUCV
printf("SOL_IUCV %llu\n", (unsigned long long)(long long)(SOL_IUCV));
#endif
#ifdef SOL_LLC
printf("SOL_LLC %llu\n", (unsigned long long)(long long)(SOL_LLC));
#endif
#ifdef SOL_NETBEUI
printf("SOL_NETBEUI %llu\n", (unsigned long long)(long long)(SOL_NETBEUI));
#endif
#ifdef SOL_NETLINK
printf("SOL_NETLINK %llu\n", (unsigned long long)(long long)(SOL_NETLINK));
#endif
#ifdef SOL_NFC
printf("SOL_NFC %llu\n", (unsigned long long)(long long)(SOL_NFC));
#endif
#ifdef SOL_PACKET
printf("SOL_PACKET %llu\n", (unsigned long long)(long long)(SOL_PACKET));
#endif
#ifdef SOL_PNPIPE
printf("SOL_PNPIPE %llu\n", (unsigned long long)(long long)(SOL_PNPIPE));
#endif
#ifdef SOL_PPPOL2TP
printf("SOL_PPPOL2TP %llu\n", (unsigned long long)(long long)(SOL_PPPOL2TP));
#endif
#ifdef SOL_RAW
printf("SOL_RAW %llu\n", (unsigned long long)(long long)(SOL_RAW));
#endif
#ifdef SOL_RDS
printf("SOL_RDS %llu\n", (unsigned long long)(long long)(SOL_RDS));
#endif
#ifdef SOL_RXRPC
printf("SOL_RXRPC %llu\n", (unsigned long long)(long long)(SOL_RXRPC));
#endif
#ifdef SOL_SOCKET
printf("SOL_SOCKET %llu\n", (unsigned long long)(long long)(SOL_SOCKET));
#endif
#ifdef SOL_TCP
printf("SOL_TCP %llu\n", (unsigned long long)(long long)(SOL_TCP));
#endif
#ifdef SOL_TIPC
printf("SOL_TIPC %llu\n", (unsigned long long)(long long)(SOL_TIPC));
#endif
#ifdef SOL_TLS
printf("SOL_TLS %llu\n", (unsigned long long)(long long)(SOL_TLS));
#endif
#ifdef SOL_UDP
printf("SOL_UDP %llu\n", (unsigned long long)(long long)(SOL_UDP));
#endif
#ifdef SOL_X25
printf("SOL_X25 %llu\n", (unsigned long long)(long long)(SOL_X25));
#endif
#ifdef SOL_XDP
printf("SOL_XDP %llu\n", (unsigned long long)(long long)(SOL_XDP));
#endif
#ifdef SOMAXCONN
printf("SOMAXCONN %llu\n", (unsigned long long)(long long)(SOMAXCONN));
#endif
#ifdef SO_ACCEPTCONN
printf("SO_ACCEPTCONN %llu\n", (unsigned long long)(long long)(SO_ACCEPTCONN));
#endif
#ifdef SO_ATTACH_BPF
printf("SO_ATTACH_BPF %llu\n", (unsigned long long)(long long)(SO_ATTACH_BPF));
#endif
#ifdef SO_ATTACH_FILTER
printf("SO_ATTACH_FILTER %llu\n", (unsigned long long)(long long)(SO_ATTACH_FILTER));
#endif
#ifdef SO_ATTACH_REUSEPORT_CBPF
printf("SO_ATTACH_REUSEPORT_CBPF %llu\n", (unsigned long long)(long long)(SO_ATTACH_REUSEPORT_CBPF));
#endif
#ifdef SO_ATTACH_REUSEPORT_EBPF
printf("SO_ATTACH_REUSEPORT_EBPF %llu\n", (unsigned long long)(long long)(SO_ATTACH_REUSEPORT_EBPF));
#endif
#ifdef SO_BINDTODEVICE
printf("SO_BINDTODEVICE %llu\n", (unsigned long long)(long long)(SO_BINDTODEVICE));
#endif
#ifdef SO_BINDTOIFINDEX
printf("SO_BINDTOIFINDEX %llu\n", (unsigned long long)(long long)(SO_BINDTOIFINDEX));
#endif
#ifdef SO_BPF_EXTENSIONS
printf("SO_BPF_EXTENSIONS %llu\n", (unsigned long long)(long long)(SO_BPF_EXTENSIONS));
#endif
#ifdef SO_BROADCAST
printf("SO_BROADCAST %llu\n", (unsigned long long)(long long)(SO_BROADCAST));
#endif
#ifdef SO_BSDCOMPAT
printf("SO_BSDCOMPAT %llu\n", (unsigned long long)(long long)(SO_BSDCOMPAT));
#endif
#ifdef SO_BUF_LOCK
printf("SO_BUF_LOCK %llu\n", (unsigned long long)(long long)(SO_BUF_LOCK));
#endif
#ifdef SO_BUSY_POLL
printf("SO_BUSY_POLL %llu\n", (unsigned long long)(long long)(SO_BUSY_POLL));
#endif
#ifdef SO_BUSY_POLL_BUDGET
printf("SO_BUSY_POLL_BUDGET %llu\n", (unsigned long long)(long long)(SO_BUSY_POLL_BUDGET));
#endif
#ifdef SO_CNX_ADVICE
printf("SO_CNX_ADVICE %llu\n", (unsigned long long)(long long)(SO_CNX_ADVICE));
#endif
#ifdef SO_COOKIE
printf("SO_COOKIE %llu\n", (unsigned long long)(long long)(SO_COOKIE));
#endif
#ifdef SO_DEBUG
printf("SO_DEBUG %llu\n", (unsigned long long)(long long)(SO_DEBUG));
#endif
#ifdef SO_DETACH_BPF
printf("SO_DETACH_BPF %llu\n", (unsigned long long)(long long)(SO_DETACH_BPF));
#endif
#ifdef SO_DETACH_FILTER
printf("SO_DETACH_FILTER %llu\n", (unsigned long long)(long long)(SO_DETACH_FILTER));
#endif
#ifdef SO_DETACH_REUSEPORT_BPF
printf("SO_DETACH_REUSEPORT_BPF %llu\n", (unsigned long long)(long long)(SO_DETACH_REUSEPORT_BPF));
#endif
#ifdef SO_DEVMEM_DMABUF
printf("SO_DEVMEM_DMABUF %llu\n", (unsigned long long)(long long)(SO_DEVMEM_DMABUF));
#endif
#ifdef SO_DEVMEM_DONTNEED
printf("SO_DEVMEM_DONTNEED %llu\n", (unsigned long long)(long long)(SO_DEVMEM_DONTNEED));
#endif
#ifdef SO_DEVMEM_LINEAR
printf("SO_DEVMEM_LINEAR %llu\n", (unsigned long long)(long long)(SO_DEVMEM_LINEAR));
#endif
#ifdef SO_DOMAIN
printf("SO_DOMAIN %llu\n", (unsigned long long)(long long)(SO_DOMAIN));
#endif
#ifdef SO_DONTROUTE
printf("SO_DONTROUTE %llu\n", (unsigned long long)(long long)(SO_DONTROUTE));
#endif
#ifdef SO_EE_ORIGIN_ICMP
printf("SO_EE_ORIGIN_ICMP %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_ICMP));
#endif
#ifdef SO_EE_ORIGIN_ICMP6
printf("SO_EE_ORIGIN_ICMP6 %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_ICMP6));
#endif
#ifdef SO_EE_ORIGIN_LOCAL
printf("SO_EE_ORIGIN_LOCAL %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_LOCAL));
#endif
#ifdef SO_EE_ORIGIN_NONE
printf("SO_EE_ORIGIN_NONE %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_NONE));
#endif
#ifdef SO_EE_ORIGIN_TIMESTAMPING
printf("SO_EE_ORIGIN_TIMESTAMPING %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_TIMESTAMPING));
#endif
#ifdef SO_EE_ORIGIN_TXSTATUS
printf("SO_EE_ORIGIN_TXSTATUS %llu\n", (unsigned long long)(long long)(SO_EE_ORIGIN_TXSTATUS));
#endif
#ifdef SO_ERROR
printf("SO_ERROR %llu\n", (unsigned long long)(long long)(SO_ERROR));
#endif
#ifdef SO_GET_FILTER
printf("SO_GET_FILTER %llu\n", (unsigned long long)(long long)(SO_GET_FILTER));
#endif
#ifdef SO_INCOMING_CPU
printf("SO_INCOMING_CPU %llu\n", (unsigned long long)(long long)(SO_INCOMING_CPU));
#endif
#ifdef SO_INCOMING_NAPI_ID
printf("SO_INCOMING_NAPI_ID %llu\n", (unsigned long long)(long long)(SO_INCOMING_NAPI_ID));
#endif
#ifdef SO_KEEPALIVE
printf("SO_KEEPALIVE %llu\n", (unsigned long long)(long long)(SO_KEEPALIVE));
#endif
#ifdef SO_LINGER
printf("SO_LINGER %llu\n", (unsigned long long)(long long)(SO_LINGER));
#endif
#ifdef SO_LOCK_FILTER
printf("SO_LOCK_FILTER %llu\n", (unsigned long long)(long long)(SO_LOCK_FILTER));
#endif
#ifdef SO_MARK
printf("SO_MARK %llu\n", (unsigned long long)(long long)(SO_MARK));
#endif
#ifdef SO_MAX_PACING_RATE
printf("SO_MAX_PACING_RATE %llu\n", (unsigned long long)(long long)(SO_MAX_PACING_RATE));
#endif
#ifdef SO_MEMINFO
printf("SO_MEMINFO %llu\n", (unsigned long long)(long long)(SO_MEMINFO));
#endif
#ifdef SO_NETNS_COOKIE
printf("SO_NETNS_COOKIE %llu\n", (unsigned long long)(long long)(SO_NETNS_COOKIE));
#endif
#ifdef SO_NOFCS
printf("SO_NOFCS %llu\n", (unsigned long long)(long long)(SO_NOFCS));
#endif
#ifdef SO_NO_CHECK
printf("SO_NO_CHECK %llu\n", (unsigned long long)(long long)(SO_NO_CHECK));
#endif
#ifdef SO_OOBINLINE
printf("SO_OOBINLINE %llu\n", (unsigned long long)(long long)(SO_OOBINLINE));
#endif
#ifdef SO_ORIGINAL_DST
printf("SO_ORIGINAL_DST %llu\n", (unsigned long long)(long long)(SO_ORIGINAL_DST));
#endif
#ifdef SO_PASSCRED
printf("SO_PASSCRED %llu\n", (unsigned long long)(long long)(SO_PASSCRED));
#endif
#ifdef SO_PASSPIDFD
printf("SO_PASSPIDFD %llu\n", (unsigned long long)(long long)(SO_PASSPIDFD));
#endif
#ifdef SO_PASSSEC
printf("SO_PASSSEC %llu\n", (unsigned long long)(long long)(SO_PASSSEC));
#endif
#ifdef SO_PEEK_OFF
printf("SO_PEEK_OFF %llu\n", (unsigned long long)(long long)(SO_PEEK_OFF));
#endif
#ifdef SO_PEERCRED
printf("SO_PEERCRED %llu\n", (unsigned long long)(long long)(SO_PEERCRED));
#endif
#ifdef SO_PEERGROUPS
printf("SO_PEERGROUPS %llu\n", (unsigned long long)(long long)(SO_PEERGROUPS));
#endif
#ifdef SO_PEERNAME
printf("SO_PEERNAME %llu\n", (unsigned long long)(long long)(SO_PEERNAME));
#endif
#ifdef SO_PEERPIDFD
printf("SO_PEERPIDFD %llu\n", (unsigned long long)(long long)(SO_PEERPIDFD));
#endif
#ifdef SO_PEERSEC
printf("SO_PEERSEC %llu\n", (unsigned long long)(long long)(SO_PEERSEC));
#endif
#ifdef SO_PREFER_BUSY_POLL
printf("SO_PREFER_BUSY_POLL %llu\n", (unsigned long long)(long long)(SO_PREFER_BUSY_POLL));
#endif
#ifdef SO_PRIORITY
printf("SO_PRIORITY %llu\n", (unsigned long long)(long long)(SO_PRIORITY));
#endif
#ifdef SO_PROTOCOL
printf("SO_PROTOCOL %llu\n", (unsigned long long)(long long)(SO_PROTOCOL));
#endif
#ifdef SO_RCVBUF
printf("SO_RCVBUF %llu\n", (unsigned long long)(long long)(SO_RCVBUF));
#endif
#ifdef SO_RCVBUFFORCE
printf("SO_RCVBUFFORCE %llu\n", (unsigned long long)(long long)(SO_RCVBUFFORCE));
#endif
#ifdef SO_RCVLOWAT
printf("SO_RCVLOWAT %llu\n", (unsigned long long)(long long)(SO_RCVLOWAT));
#endif
#ifdef SO_RCVMARK
printf("SO_RCVMARK %llu\n", (unsigned long long)(long long)(SO_RCVMARK));
#endif
#ifdef SO_RCVTIMEO
printf("SO_RCVTIMEO %llu\n", (unsigned long long)(long long)(SO_RCVTIMEO));
#endif
#ifdef SO_RCVTIMEO_NEW
printf("SO_RCVTIMEO_NEW %llu\n", (unsigned long long)(long long)(SO_RCVTIMEO_NEW));
#endif
#ifdef SO_RESERVE_MEM
printf("SO_RESERVE_MEM %llu\n", (unsigned long long)(long long)(SO_RESERVE_MEM));
#endif
#ifdef SO_REUSEADDR
printf("SO_REUSEADDR %llu\n", (unsigned long long)(long long)(SO_REUSEADDR));
#endif
#ifdef SO_REUSEPORT
printf("SO_REUSEPORT %llu\n", (unsigned long long)(long long)(SO_REUSEPORT));
#endif
#ifdef SO_RXQ_OVFL
printf("SO_RXQ_OVFL %llu\n", (unsigned long long)(long long)(SO_RXQ_OVFL));
#endif
#ifdef SO_SECURITY_AUTHENTICATION
printf("SO_SECURITY_AUTHENTICATION %llu\n", (unsigned long long)(long long)(SO_SECURITY_AUTHENTICATION));
#endif
#ifdef SO_SECURITY_ENCRYPTION_NETWORK
printf("SO_SECURITY_ENCRYPTION_NETWORK %llu\n", (unsigned long long)(long long)(SO_SECURITY_ENCRYPTION_NETWORK));
#endif
#ifdef SO_SECURITY_ENCRYPTION_TRANSPORT
printf("SO_SECURITY_ENCRYPTION_TRANSPORT %llu\n", (unsigned long long)(long long)(SO_SECURITY_ENCRYPTION_TRANSPORT));
#endif
#ifdef SO_SELECT_ERR_QUEUE
printf("SO_SELECT_ERR_QUEUE %llu\n", (unsigned long long)(long long)(SO_SELECT_ERR_QUEUE));
#endif
#ifdef SO_SNDBUF
printf("SO_SNDBUF %llu\n", (unsigned long long)(long long)(SO_SNDBUF));
#endif
#ifdef SO_SNDBUFFORCE
printf("SO_SNDBUFFORCE %llu\n", (unsigned long long)(long long)(SO_SNDBUFFORCE));
#endif
#ifdef SO_SNDLOWAT
printf("SO_SNDLOWAT %llu\n", (unsigned long long)(long long)(SO_SNDLOWAT));
#endif
#ifdef SO_SNDTIMEO
printf("SO_SNDTIMEO %llu\n", (unsigned long long)(long long)(SO_SNDTIMEO));
#endif
#ifdef SO_SNDTIMEO_NEW
printf("SO_SNDTIMEO_NEW %llu\n", (unsigned long long)(long long)(SO_SNDTIMEO_NEW));
#endif
#ifdef SO_TIMESTAMP
printf("SO_TIMESTAMP %llu\n", (unsigned long long)(long long)(SO_TIMESTAMP));
#endif
#ifdef SO_TIMESTAMPING
printf("SO_TIMESTAMPING %llu\n", (unsigned long long)(long long)(SO_TIMESTAMPING));
#endif
#ifdef SO_TIMESTAMPING_NEW
printf("SO_TIMESTAMPING_NEW %llu\n", (unsigned long long)(long long)(SO_TIMESTAMPING_NEW));
#endif
#ifdef SO_TIMESTAMPNS
printf("SO_TIMESTAMPNS %llu\n", (unsigned long long)(long long)(SO_TIMESTAMPNS));
#endif
#ifdef SO_TIMESTAMPNS_NEW
printf("SO_TIMESTAMPNS_NEW %llu\n", (unsigned long long)(long long)(SO_TIMESTAMPNS_NEW));
#endif
#ifdef SO_TIMESTAMP_NEW
printf("SO_TIMESTAMP_NEW %llu\n", (unsigned long long)(long long)(SO_TIMESTAMP_NEW));
#endif
#ifdef SO_TXREHASH
printf("SO_TXREHASH %llu\n", (unsigned long long)(long long)(SO_TXREHASH));
#endif
#ifdef SO_TXTIME
printf("SO_TXTIME %llu\n", (unsigned long long)(long long)(SO_TXTIME));
#endif
#ifdef SO_TYPE
printf("SO_TYPE %llu\n", (unsigned long long)(long long)(SO_TYPE));
#endif
#ifdef SO_WIFI_STATUS
printf("SO_WIFI_STATUS %llu\n", (unsigned long long)(long long)(SO_WIFI_STATUS));
#endif
#ifdef SO_ZEROCOPY
printf("SO_ZEROCOPY %llu\n", (unsigned long long)(long long)(SO_ZEROCOPY));
#endif
#ifdef SPLICE_F_GIFT
printf("SPLICE_F_GIFT %llu\n", (unsigned long long)(long long)(SPLICE_F_GIFT));
#endif
#ifdef SPLICE_F_MORE
printf("SPLICE_F_MORE %llu\n", (unsigned long long)(long long)(SPLICE_F_MORE));
#endif
#ifdef SPLICE_F_MOVE
printf("SPLICE_F_MOVE %llu\n", (unsigned long long)(long long)(SPLICE_F_MOVE));
#endif
#ifdef SPLICE_F_NONBLOCK
printf("SPLICE_F_NONBLOCK %llu\n", (unsigned long long)(long long)(SPLICE_F_NONBLOCK));
#endif
#ifdef SS_DISABLE
printf("SS_DISABLE %llu\n", (unsigned long long)(long long)(SS_DISABLE));
#endif
#ifdef SS_ONSTACK
printf("SS_ONSTACK %llu\n", (unsigned long long)(long long)(SS_ONSTACK));
#endif
#ifdef STATX_ALL
printf("STATX_ALL %llu\n", (unsigned long long)(long long)(STATX_ALL));
#endif
#ifdef STATX_ATIME
printf("STATX_ATIME %llu\n", (unsigned long long)(long long)(STATX_ATIME));
#endif
#ifdef STATX_ATTR_APPEND
printf("STATX_ATTR_APPEND %llu\n", (unsigned long long)(long long)(STATX_ATTR_APPEND));
#endif
#ifdef STATX_ATTR_AUTOMOUNT
printf("STATX_ATTR_AUTOMOUNT %llu\n", (unsigned long long)(long long)(STATX_ATTR_AUTOMOUNT));
#endif
#ifdef STATX_ATTR_COMPRESSED
printf("STATX_ATTR_COMPRESSED %llu\n", (unsigned long long)(long long)(STATX_ATTR_COMPRESSED));
#endif
#ifdef STATX_ATTR_DAX
printf("STATX_ATTR_DAX %llu\n", (unsigned long long)(long long)(STATX_ATTR_DAX));
#endif
#ifdef STATX_ATTR_ENCRYPTED
printf("STATX_ATTR_ENCRYPTED %llu\n", (unsigned long long)(long long)(STATX_ATTR_ENCRYPTED));
#endif
#ifdef STATX_ATTR_IMMUTABLE
printf("STATX_ATTR_IMMUTABLE %llu\n", (unsigned long long)(long long)(STATX_ATTR_IMMUTABLE));
#endif
#ifdef STATX_ATTR_MOUNT_ROOT
printf("STATX_ATTR_MOUNT_ROOT %llu\n", (unsigned long long)(long long)(STATX_ATTR_MOUNT_ROOT));
#endif
#ifdef STATX_ATTR_NODUMP
printf("STATX_ATTR_NODUMP %llu\n", (unsigned long long)(long long)(STATX_ATTR_NODUMP));
#endif
#ifdef STATX_ATTR_VERITY
printf("STATX_ATTR_VERITY %llu\n", (unsigned long long)(long long)(STATX_ATTR_VERITY));
#endif
#ifdef STATX_BASIC_STATS
printf("STATX_BASIC_STATS %llu\n", (unsigned long long)(long long)(STATX_BASIC_STATS));
#endif
#ifdef STATX_BLOCKS
printf("STATX_BLOCKS %llu\n", (unsigned long long)(long long)(STATX_BLOCKS));
#endif
#ifdef STATX_BTIME
printf("STATX_BTIME %llu\n", (unsigned long long)(long long)(STATX_BTIME));
#endif
#ifdef STATX_CTIME
printf("STATX_CTIME %llu\n", (unsigned long long)(long long)(STATX_CTIME));
#endif
#ifdef STATX_DIOALIGN
printf("STATX_DIOALIGN %llu\n", (unsigned long long)(long long)(STATX_DIOALIGN));
#endif
#ifdef STATX_GID
printf("STATX_GID %llu\n", (unsigned long long)(long long)(STATX_GID));
#endif
#ifdef STATX_INO
printf("STATX_INO %llu\n", (unsigned long long)(long long)(STATX_INO));
#endif
#ifdef STATX_MNT_ID
printf("STATX_MNT_ID %llu\n", (unsigned long long)(long long)(STATX_MNT_ID));
#endif
#ifdef STATX_MODE
printf("STATX_MODE %llu\n", (unsigned long long)(long long)(STATX_MODE));
#endif
#ifdef STATX_MTIME
printf("STATX_MTIME %llu\n", (unsigned long long)(long long)(STATX_MTIME));
#endif
#ifdef STATX_NLINK
printf("STATX_NLINK %llu\n", (unsigned long long)(long long)(STATX_NLINK));
#endif
#ifdef STATX_SIZE
printf("STATX_SIZE %llu\n", (unsigned long long)(long long)(STATX_SIZE));
#endif
#ifdef STATX_TYPE
printf("STATX_TYPE %llu\n", (unsigned long long)(long long)(STATX_TYPE));
#endif
#ifdef STATX_UID
printf("STATX_UID %llu\n", (unsigned long long)(long long)(STATX_UID));
#endif
#ifdef STATX__RESERVED
printf("STATX__RESERVED %llu\n", (unsigned long long)(long long)(STATX__RESERVED));
#endif
#ifdef STA_CLK
printf("STA_CLK %llu\n", (unsigned long long)(long long)(STA_CLK));
#endif
#ifdef STA_CLOCKERR
printf("STA_CLOCKERR %llu\n", (unsigned long long)(long long)(STA_CLOCKERR));
#endif
#ifdef STA_DEL
printf("STA_DEL %llu\n", (unsigned long long)(long long)(STA_DEL));
#endif
#ifdef STA_FLL
printf("STA_FLL %llu\n", (unsigned long long)(long long)(STA_FLL));
#endif
#ifdef STA_FREQHOLD
printf("STA_FREQHOLD %llu\n", (unsigned long long)(long long)(STA_FREQHOLD));
#endif
#ifdef STA_INS
printf("STA_INS %llu\n", (unsigned long long)(long long)(STA_INS));
#endif
#ifdef STA_MODE
printf("STA_MODE %llu\n", (unsigned long long)(long long)(STA_MODE));
#endif
#ifdef STA_NANO
printf("STA_NANO %llu\n", (unsigned long long)(long long)(STA_NANO));
#endif
#ifdef STA_PLL
printf("STA_PLL %llu\n", (unsigned long long)(long long)(STA_PLL));
#endif
#ifdef STA_PPSERROR
printf("STA_PPSERROR %llu\n", (unsigned long long)(long long)(STA_PPSERROR));
#endif
#ifdef STA_PPSFREQ
printf("STA_PPSFREQ %llu\n", (unsigned long long)(long long)(STA_PPSFREQ));
#endif
#ifdef STA_PPSJITTER
printf("STA_PPSJITTER %llu\n", (unsigned long long)(long long)(STA_PPSJITTER));
#endif
#ifdef STA_PPSSIGNAL
printf("STA_PPSSIGNAL %llu\n", (unsigned long long)(long long)(STA_PPSSIGNAL));
#endif
#ifdef STA_PPSTIME
printf("STA_PPSTIME %llu\n", (unsigned long long)(long long)(STA_PPSTIME));
#endif
#ifdef STA_PPSWANDER
printf("STA_PPSWANDER %llu\n", (unsigned long long)(long long)(STA_PPSWANDER));
#endif
#ifdef STA_RONLY
printf("STA_RONLY %llu\n", (unsigned long long)(long long)(STA_RONLY));
#endif
#ifdef STA_UNSYNC
printf("STA_UNSYNC %llu\n", (unsigned long long)(long long)(STA_UNSYNC));
#endif
#ifdef STICKY_TIMEOUTS
printf("STICKY_TIMEOUTS %llu\n", (unsigned long long)(long long)(STICKY_TIMEOUTS));
#endif
#ifdef ST_APPEND
printf("ST_APPEND %llu\n", (unsigned long long)(long long)(ST_APPEND));
#endif
#ifdef ST_IMMUTABLE
printf("ST_IMMUTABLE %llu\n", (unsigned long long)(long long)(ST_IMMUTABLE));
#endif
#ifdef ST_MANDLOCK
printf("ST_MANDLOCK %llu\n", (unsigned long long)(long long)(ST_MANDLOCK));
#endif
#ifdef ST_NOATIME
printf("ST_NOATIME %llu\n", (unsigned long long)(long long)(ST_NOATIME));
#endif
#ifdef ST_NODEV
printf("ST_NODEV %llu\n", (unsigned long long)(long long)(ST_NODEV));
#endif
#ifdef ST_NODIRATIME
printf("ST_NODIRATIME %llu\n", (unsigned long long)(long long)(ST_NODIRATIME));
#endif
#ifdef ST_NOEXEC
printf("ST_NOEXEC %llu\n", (unsigned long long)(long long)(ST_NOEXEC));
#endif
#ifdef ST_NOSUID
printf("ST_NOSUID %llu\n", (unsigned long long)(long long)(ST_NOSUID));
#endif
#ifdef ST_RDONLY
printf("ST_RDONLY %llu\n", (unsigned long long)(long long)(ST_RDONLY));
#endif
#ifdef ST_RELATIME
printf("ST_RELATIME %llu\n", (unsigned long long)(long long)(ST_RELATIME));
#endif
#ifdef ST_SYNCHRONOUS
printf("ST_SYNCHRONOUS %llu\n", (unsigned long long)(long long)(ST_SYNCHRONOUS));
#endif
#ifdef ST_WRITE
printf("ST_WRITE %llu\n", (unsigned long long)(long long)(ST_WRITE));
#endif
#ifdef SW_CNT
printf("SW_CNT %llu\n", (unsigned long long)(long long)(SW_CNT));
#endif
#ifdef SW_MAX
printf("SW_MAX %llu\n", (unsigned long long)(long long)(SW_MAX));
#endif
#ifdef SYNC_FILE_RANGE_WAIT_AFTER
printf("SYNC_FILE_RANGE_WAIT_AFTER %llu\n", (unsigned long long)(long long)(SYNC_FILE_RANGE_WAIT_AFTER));
#endif
#ifdef SYNC_FILE_RANGE_WAIT_BEFORE
printf("SYNC_FILE_RANGE_WAIT_BEFORE %llu\n", (unsigned long long)(long long)(SYNC_FILE_RANGE_WAIT_BEFORE));
#endif
#ifdef SYNC_FILE_RANGE_WRITE
printf("SYNC_FILE_RANGE_WRITE %llu\n", (unsigned long long)(long long)(SYNC_FILE_RANGE_WRITE));
#endif
#ifdef SYN_CNT
printf("SYN_CNT %llu\n", (unsigned long long)(long long)(SYN_CNT));
#endif
#ifdef SYN_MAX
printf("SYN_MAX %llu\n", (unsigned long long)(long long)(SYN_MAX));
#endif
#ifdef SYSFS_MAGIC
printf("SYSFS_MAGIC %llu\n", (unsigned long long)(long long)(SYSFS_MAGIC));
#endif
#ifdef SYS_accept
printf("SYS_accept %llu\n", (unsigned long long)(long long)(SYS_accept));
#endif
#ifdef SYS_accept4
printf("SYS_accept4 %llu\n", (unsigned long long)(long long)(SYS_accept4));
#endif
#ifdef SYS_acct
printf("SYS_acct %llu\n", (unsigned long long)(long long)(SYS_acct));
#endif
#ifdef SYS_add_key
printf("SYS_add_key %llu\n", (unsigned long long)(long long)(SYS_add_key));
#endif
#ifdef SYS_adjtimex
printf("SYS_adjtimex %llu\n", (unsigned long long)(long long)(SYS_adjtimex));
#endif
#ifdef SYS_bind
printf("SYS_bind %llu\n", (unsigned long long)(long long)(SYS_bind));
#endif
#ifdef SYS_bpf
printf("SYS_bpf %llu\n", (unsigned long long)(long long)(SYS_bpf));
#endif
#ifdef SYS_brk
printf("SYS_brk %llu\n", (unsigned long long)(long long)(SYS_brk));
#endif
#ifdef SYS_capget
printf("SYS_capget %llu\n", (unsigned long long)(long long)(SYS_capget));
#endif
#ifdef SYS_capset
printf("SYS_capset %llu\n", (unsigned long long)(long long)(SYS_capset));
#endif
#ifdef SYS_chdir
printf("SYS_chdir %llu\n", (unsigned long long)(long long)(SYS_chdir));
#endif
#ifdef SYS_chroot
printf("SYS_chroot %llu\n", (unsigned long long)(long long)(SYS_chroot));
#endif
#ifdef SYS_clock_adjtime
printf("SYS_clock_adjtime %llu\n", (unsigned long long)(long long)(SYS_clock_adjtime));
#endif
#ifdef SYS_clock_getres
printf("SYS_clock_getres %llu\n", (unsigned long long)(long long)(SYS_clock_getres));
#endif
#ifdef SYS_clock_gettime
printf("SYS_clock_gettime %llu\n", (unsigned long long)(long long)(SYS_clock_gettime));
#endif
#ifdef SYS_clock_nanosleep
printf("SYS_clock_nanosleep %llu\n", (unsigned long long)(long long)(SYS_clock_nanosleep));
#endif
#ifdef SYS_clock_settime
printf("SYS_clock_settime %llu\n", (unsigned long long)(long long)(SYS_clock_settime));
#endif
#ifdef SYS_clone
printf("SYS_clone %llu\n", (unsigned long long)(long long)(SYS_clone));
#endif
#ifdef SYS_clone3
printf("SYS_clone3 %llu\n", (unsigned long long)(long long)(SYS_clone3));
#endif
#ifdef SYS_close
printf("SYS_close %llu\n", (unsigned long long)(long long)(SYS_close));
#endif
#ifdef SYS_close_range
printf("SYS_close_range %llu\n", (unsigned long long)(long long)(SYS_close_range));
#endif
#ifdef SYS_connect
printf("SYS_connect %llu\n", (unsigned long long)(long long)(SYS_connect));
#endif
#ifdef SYS_copy_file_range
printf("SYS_copy_file_range %llu\n", (unsigned long long)(long long)(SYS_copy_file_range));
#endif
#ifdef SYS_delete_module
printf("SYS_delete_module %llu\n", (unsigned long long)(long long)(SYS_delete_module));
#endif
#ifdef SYS_dup
printf("SYS_dup %llu\n", (unsigned long long)(long long)(SYS_dup));
#endif
#ifdef SYS_dup3
printf("SYS_dup3 %llu\n", (unsigned long long)(long long)(SYS_dup3));
#endif
#ifdef SYS_epoll_create1
printf("SYS_epoll_create1 %llu\n", (unsigned long long)(long long)(SYS_epoll_create1));
#endif
#ifdef SYS_epoll_ctl
printf("SYS_epoll_ctl %llu\n", (unsigned long long)(long long)(SYS_epoll_ctl));
#endif
#ifdef SYS_epoll_pwait
printf("SYS_epoll_pwait %llu\n", (unsigned long long)(long long)(SYS_epoll_pwait));
#endif
#ifdef SYS_epoll_pwait2
printf("SYS_epoll_pwait2 %llu\n", (unsigned long long)(long long)(SYS_epoll_pwait2));
#endif
#ifdef SYS_eventfd2
printf("SYS_eventfd2 %llu\n", (unsigned long long)(long long)(SYS_eventfd2));
#endif
#ifdef SYS_execve
printf("SYS_execve %llu\n", (unsigned long long)(long long)(SYS_execve));
#endif
#ifdef SYS_execveat
printf("SYS_execveat %llu\n", (unsigned long long)(long long)(SYS_execveat));
#endif
#ifdef SYS_exit
printf("SYS_exit %llu\n", (unsigned long long)(long long)(SYS_exit));
#endif
#ifdef SYS_exit_group
printf("SYS_exit_group %llu\n", (unsigned long long)(long long)(SYS_exit_group));
#endif
#ifdef SYS_faccessat
printf("SYS_faccessat %llu\n", (unsigned long long)(long long)(SYS_faccessat));
#endif
#ifdef SYS_faccessat2
printf("SYS_faccessat2 %llu\n", (unsigned long long)(long long)(SYS_faccessat2));
#endif
#ifdef SYS_fadvise64
printf("SYS_fadvise64 %llu\n", (unsigned long long)(long long)(SYS_fadvise64));
#endif
#ifdef SYS_fallocate
printf("SYS_fallocate %llu\n", (unsigned long long)(long long)(SYS_fallocate));
#endif
#ifdef SYS_fanotify_init
printf("SYS_fanotify_init %llu\n", (unsigned long long)(long long)(SYS_fanotify_init));
#endif
#ifdef SYS_fanotify_mark
printf("SYS_fanotify_mark %llu\n", (unsigned long long)(long long)(SYS_fanotify_mark));
#endif
#ifdef SYS_fchdir
printf("SYS_fchdir %llu\n", (unsigned long long)(long long)(SYS_fchdir));
#endif
#ifdef SYS_fchmod
printf("SYS_fchmod %llu\n", (unsigned long long)(long long)(SYS_fchmod));
#endif
#ifdef SYS_fchmodat
printf("SYS_fchmodat %llu\n", (unsigned long long)(long long)(SYS_fchmodat));
#endif
#ifdef SYS_fchown
printf("SYS_fchown %llu\n", (unsigned long long)(long long)(SYS_fchown));
#endif
#ifdef SYS_fchownat
printf("SYS_fchownat %llu\n", (unsigned long long)(long long)(SYS_fchownat));
#endif
#ifdef SYS_fcntl
printf("SYS_fcntl %llu\n", (unsigned long long)(long long)(SYS_fcntl));
#endif
#ifdef SYS_fdatasync
printf("SYS_fdatasync %llu\n", (unsigned long long)(long long)(SYS_fdatasync));
#endif
#ifdef SYS_fgetxattr
printf("SYS_fgetxattr %llu\n", (unsigned long long)(long long)(SYS_fgetxattr));
#endif
#ifdef SYS_finit_module
printf("SYS_finit_module %llu\n", (unsigned long long)(long long)(SYS_finit_module));
#endif
#ifdef SYS_flistxattr
printf("SYS_flistxattr %llu\n", (unsigned long long)(long long)(SYS_flistxattr));
#endif
#ifdef SYS_flock
printf("SYS_flock %llu\n", (unsigned long long)(long long)(SYS_flock));
#endif
#ifdef SYS_fremovexattr
printf("SYS_fremovexattr %llu\n", (unsigned long long)(long long)(SYS_fremovexattr));
#endif
#ifdef SYS_fsconfig
printf("SYS_fsconfig %llu\n", (unsigned long long)(long long)(SYS_fsconfig));
#endif
#ifdef SYS_fsetxattr
printf("SYS_fsetxattr %llu\n", (unsigned long long)(long long)(SYS_fsetxattr));
#endif
#ifdef SYS_fsmount
printf("SYS_fsmount %llu\n", (unsigned long long)(long long)(SYS_fsmount));
#endif
#ifdef SYS_fsopen
printf("SYS_fsopen %llu\n", (unsigned long long)(long long)(SYS_fsopen));
#endif
#ifdef SYS_fspick
printf("SYS_fspick %llu\n", (unsigned long long)(long long)(SYS_fspick));
#endif
#ifdef SYS_fstat
printf("SYS_fstat %llu\n", (unsigned long long)(long long)(SYS_fstat));
#endif
#ifdef SYS_fstatfs
printf("SYS_fstatfs %llu\n", (unsigned long long)(long long)(SYS_fstatfs));
#endif
#ifdef SYS_fsync
printf("SYS_fsync %llu\n", (unsigned long long)(long long)(SYS_fsync));
#endif
#ifdef SYS_ftruncate
printf("SYS_ftruncate %llu\n", (unsigned long long)(long long)(SYS_ftruncate));
#endif
#ifdef SYS_futex
printf("SYS_futex %llu\n", (unsigned long long)(long long)(SYS_futex));
#endif
#ifdef SYS_futex_waitv
printf("SYS_futex_waitv %llu\n", (unsigned long long)(long long)(SYS_futex_waitv));
#endif
#ifdef SYS_get_mempolicy
printf("SYS_get_mempolicy %llu\n", (unsigned long long)(long long)(SYS_get_mempolicy));
#endif
#ifdef SYS_get_robust_list
printf("SYS_get_robust_list %llu\n", (unsigned long long)(long long)(SYS_get_robust_list));
#endif
#ifdef SYS_getcpu
printf("SYS_getcpu %llu\n", (unsigned long long)(long long)(SYS_getcpu));
#endif
#ifdef SYS_getcwd
printf("SYS_getcwd %llu\n", (unsigned long long)(long long)(SYS_getcwd));
#endif
#ifdef SYS_getdents64
printf("SYS_getdents64 %llu\n", (unsigned long long)(long long)(SYS_getdents64));
#endif
#ifdef SYS_getegid
printf("SYS_getegid %llu\n", (unsigned long long)(long long)(SYS_getegid));
#endif
#ifdef SYS_geteuid
printf("SYS_geteuid %llu\n", (unsigned long long)(long long)(SYS_geteuid));
#endif
#ifdef SYS_getgid
printf("SYS_getgid %llu\n", (unsigned long long)(long long)(SYS_getgid));
#endif
#ifdef SYS_getgroups
printf("SYS_getgroups %llu\n", (unsigned long long)(long long)(SYS_getgroups));
#endif
#ifdef SYS_getitimer
printf("SYS_getitimer %llu\n", (unsigned long long)(long long)(SYS_getitimer));
#endif
#ifdef SYS_getpeername
printf("SYS_getpeername %llu\n", (unsigned long long)(long long)(SYS_getpeername));
#endif
#ifdef SYS_getpgid
printf("SYS_getpgid %llu\n", (unsigned long long)(long long)(SYS_getpgid));
#endif
#ifdef SYS_getpid
printf("SYS_getpid %llu\n", (unsigned long long)(long long)(SYS_getpid));
#endif
#ifdef SYS_getppid
printf("SYS_getppid %llu\n", (unsigned long long)(long long)(SYS_getppid));
#endif
#ifdef SYS_getpriority
printf("SYS_getpriority %llu\n", (unsigned long long)(long long)(SYS_getpriority));
#endif
#ifdef SYS_getrandom
printf("SYS_getrandom %llu\n", (unsigned long long)(long long)(SYS_getrandom));
#endif
#ifdef SYS_getresgid
printf("SYS_getresgid %llu\n", (unsigned long long)(long long)(SYS_getresgid));
#endif
#ifdef SYS_getresuid
printf("SYS_getresuid %llu\n", (unsigned long long)(long long)(SYS_getresuid));
#endif
#ifdef SYS_getrlimit
printf("SYS_getrlimit %llu\n", (unsigned long long)(long long)(SYS_getrlimit));
#endif
#ifdef SYS_getrusage
printf("SYS_getrusage %llu\n", (unsigned long long)(long long)(SYS_getrusage));
#endif
#ifdef SYS_getsid
printf("SYS_getsid %llu\n", (unsigned long long)(long long)(SYS_getsid));
#endif
#ifdef SYS_getsockname
printf("SYS_getsockname %llu\n", (unsigned long long)(long long)(SYS_getsockname));
#endif
#ifdef SYS_getsockopt
printf("SYS_getsockopt %llu\n", (unsigned long long)(long long)(SYS_getsockopt));
#endif
#ifdef SYS_gettid
printf("SYS_gettid %llu\n", (unsigned long long)(long long)(SYS_gettid));
#endif
#ifdef SYS_gettimeofday
printf("SYS_gettimeofday %llu\n", (unsigned long long)(long long)(SYS_gettimeofday));
#endif
#ifdef SYS_getuid
printf("SYS_getuid %llu\n", (unsigned long long)(long long)(SYS_getuid));
#endif
#ifdef SYS_getxattr
printf("SYS_getxattr %llu\n", (unsigned long long)(long long)(SYS_getxattr));
#endif
#ifdef SYS_init_module
printf("SYS_init_module %llu\n", (unsigned long long)(long long)(SYS_init_module));
#endif
#ifdef SYS_inotify_add_watch
printf("SYS_inotify_add_watch %llu\n", (unsigned long long)(long long)(SYS_inotify_add_watch));
#endif
#ifdef SYS_inotify_init1
printf("SYS_inotify_init1 %llu\n", (unsigned long long)(long long)(SYS_inotify_init1));
#endif
#ifdef SYS_inotify_rm_watch
printf("SYS_inotify_rm_watch %llu\n", (unsigned long long)(long long)(SYS_inotify_rm_watch));
#endif
#ifdef SYS_io_cancel
printf("SYS_io_cancel %llu\n", (unsigned long long)(long long)(SYS_io_cancel));
#endif
#ifdef SYS_io_destroy
printf("SYS_io_destroy %llu\n", (unsigned long long)(long long)(SYS_io_destroy));
#endif
#ifdef SYS_io_getevents
printf("SYS_io_getevents %llu\n", (unsigned long long)(long long)(SYS_io_getevents));
#endif
#ifdef SYS_io_setup
printf("SYS_io_setup %llu\n", (unsigned long long)(long long)(SYS_io_setup));
#endif
#ifdef SYS_io_submit
printf("SYS_io_submit %llu\n", (unsigned long long)(long long)(SYS_io_submit));
#endif
#ifdef SYS_io_uring_enter
printf("SYS_io_uring_enter %llu\n", (unsigned long long)(long long)(SYS_io_uring_enter));
#endif
#ifdef SYS_io_uring_register
printf("SYS_io_uring_register %llu\n", (unsigned long long)(long long)(SYS_io_uring_register));
#endif
#ifdef SYS_io_uring_setup
printf("SYS_io_uring_setup %llu\n", (unsigned long long)(long long)(SYS_io_uring_setup));
#endif
#ifdef SYS_ioctl
printf("SYS_ioctl %llu\n", (unsigned long long)(long long)(SYS_ioctl));
#endif
#ifdef SYS_ioprio_get
printf("SYS_ioprio_get %llu\n", (unsigned long long)(long long)(SYS_ioprio_get));
#endif
#ifdef SYS_ioprio_set
printf("SYS_ioprio_set %llu\n", (unsigned long long)(long long)(SYS_ioprio_set));
#endif
#ifdef SYS_kcmp
printf("SYS_kcmp %llu\n", (unsigned long long)(long long)(SYS_kcmp));
#endif
#ifdef SYS_kexec_load
printf("SYS_kexec_load %llu\n", (unsigned long long)(long long)(SYS_kexec_load));
#endif
#ifdef SYS_keyctl
printf("SYS_keyctl %llu\n", (unsigned long long)(long long)(SYS_keyctl));
#endif
#ifdef SYS_kill
printf("SYS_kill %llu\n", (unsigned long long)(long long)(SYS_kill));
#endif
#ifdef SYS_landlock_add_rule
printf("SYS_landlock_add_rule %llu\n", (unsigned long long)(long long)(SYS_landlock_add_rule));
#endif
#ifdef SYS_landlock_create_ruleset
printf("SYS_landlock_create_ruleset %llu\n", (unsigned long long)(long long)(SYS_landlock_create_ruleset));
#endif
#ifdef SYS_landlock_restrict_self
printf("SYS_landlock_restrict_self %llu\n", (unsigned long long)(long long)(SYS_landlock_restrict_self));
#endif
#ifdef SYS_lgetxattr
printf("SYS_lgetxattr %llu\n", (unsigned long long)(long long)(SYS_lgetxattr));
#endif
#ifdef SYS_linkat
printf("SYS_linkat %llu\n", (unsigned long long)(long long)(SYS_linkat));
#endif
#ifdef SYS_listen
printf("SYS_listen %llu\n", (unsigned long long)(long long)(SYS_listen));
#endif
#ifdef SYS_listxattr
printf("SYS_listxattr %llu\n", (unsigned long long)(long long)(SYS_listxattr));
#endif
#ifdef SYS_llistxattr
printf("SYS_llistxattr %llu\n", (unsigned long long)(long long)(SYS_llistxattr));
#endif
#ifdef SYS_lookup_dcookie
printf("SYS_lookup_dcookie %llu\n", (unsigned long long)(long long)(SYS_lookup_dcookie));
#endif
#ifdef SYS_lremovexattr
printf("SYS_lremovexattr %llu\n", (unsigned long long)(long long)(SYS_lremovexattr));
#endif
#ifdef SYS_lseek
printf("SYS_lseek %llu\n", (unsigned long long)(long long)(SYS_lseek));
#endif
#ifdef SYS_lsetxattr
printf("SYS_lsetxattr %llu\n", (unsigned long long)(long long)(SYS_lsetxattr));
#endif
#ifdef SYS_madvise
printf("SYS_madvise %llu\n", (unsigned long long)(long long)(SYS_madvise));
#endif
#ifdef SYS_mbind
printf("SYS_mbind %llu\n", (unsigned long long)(long long)(SYS_mbind));
#endif
#ifdef SYS_membarrier
printf("SYS_membarrier %llu\n", (unsigned long long)(long long)(SYS_membarrier));
#endif
#ifdef SYS_memfd_create
printf("SYS_memfd_create %llu\n", (unsigned long long)(long long)(SYS_memfd_create));
#endif
#ifdef SYS_memfd_secret
printf("SYS_memfd_secret %llu\n", (unsigned long long)(long long)(SYS_memfd_secret));
#endif
#ifdef SYS_migrate_pages
printf("SYS_migrate_pages %llu\n", (unsigned long long)(long long)(SYS_migrate_pages));
#endif
#ifdef SYS_mincore
printf("SYS_mincore %llu\n", (unsigned long long)(long long)(SYS_mincore));
#endif
#ifdef SYS_mkdirat
printf("SYS_mkdirat %llu\n", (unsigned long long)(long long)(SYS_mkdirat));
#endif
#ifdef SYS_mknodat
printf("SYS_mknodat %llu\n", (unsigned long long)(long long)(SYS_mknodat));
#endif
#ifdef SYS_mlock
printf("SYS_mlock %llu\n", (unsigned long long)(long long)(SYS_mlock));
#endif
#ifdef SYS_mlock2
printf("SYS_mlock2 %llu\n", (unsigned long long)(long long)(SYS_mlock2));
#endif
#ifdef SYS_mlockall
printf("SYS_mlockall %llu\n", (unsigned long long)(long long)(SYS_mlockall));
#endif
#ifdef SYS_mmap
printf("SYS_mmap %llu\n", (unsigned long long)(long long)(SYS_mmap));
#endif
#ifdef SYS_mount
printf("SYS_mount %llu\n", (unsigned long long)(long long)(SYS_mount));
#endif
#ifdef SYS_mount_setattr
printf("SYS_mount_setattr %llu\n", (unsigned long long)(long long)(SYS_mount_setattr));
#endif
#ifdef SYS_move_mount
printf("SYS_move_mount %llu\n", (unsigned long long)(long long)(SYS_move_mount));
#endif
#ifdef SYS_move_pages
printf("SYS_move_pages %llu\n", (unsigned long long)(long long)(SYS_move_pages));
#endif
#ifdef SYS_mprotect
printf("SYS_mprotect %llu\n", (unsigned long long)(long long)(SYS_mprotect));
#endif
#ifdef SYS_mq_getsetattr
printf("SYS_mq_getsetattr %llu\n", (unsigned long long)(long long)(SYS_mq_getsetattr));
#endif
#ifdef SYS_mq_notify
printf("SYS_mq_notify %llu\n", (unsigned long long)(long long)(SYS_mq_notify));
#endif
#ifdef SYS_mq_open
printf("SYS_mq_open %llu\n", (unsigned long long)(long long)(SYS_mq_open));
#endif
#ifdef SYS_mq_timedreceive
printf("SYS_mq_timedreceive %llu\n", (unsigned long long)(long long)(SYS_mq_timedreceive));
#endif
#ifdef SYS_mq_timedsend
printf("SYS_mq_timedsend %llu\n", (unsigned long long)(long long)(SYS_mq_timedsend));
#endif
#ifdef SYS_mq_unlink
printf("SYS_mq_unlink %llu\n", (unsigned long long)(long long)(SYS_mq_unlink));
#endif
#ifdef SYS_mremap
printf("SYS_mremap %llu\n", (unsigned long long)(long long)(SYS_mremap));
#endif
#ifdef SYS_msgctl
printf("SYS_msgctl %llu\n", (unsigned long long)(long long)(SYS_msgctl));
#endif
#ifdef SYS_msgget
printf("SYS_msgget %llu\n", (unsigned long long)(long long)(SYS_msgget));
#endif
#ifdef SYS_msgrcv
printf("SYS_msgrcv %llu\n", (unsigned long long)(long long)(SYS_msgrcv));
#endif
#ifdef SYS_msgsnd
printf("SYS_msgsnd %llu\n", (unsigned long long)(long long)(SYS_msgsnd));
#endif
#ifdef SYS_msync
printf("SYS_msync %llu\n", (unsigned long long)(long long)(SYS_msync));
#endif
#ifdef SYS_munlock
printf("SYS_munlock %llu\n", (unsigned long long)(long long)(SYS_munlock));
#endif
#ifdef SYS_munlockall
printf("SYS_munlockall %llu\n", (unsigned long long)(long long)(SYS_munlockall));
#endif
#ifdef SYS_munmap
printf("SYS_munmap %llu\n", (unsigned long long)(long long)(SYS_munmap));
#endif
#ifdef SYS_name_to_handle_at
printf("SYS_name_to_handle_at %llu\n", (unsigned long long)(long long)(SYS_name_to_handle_at));
#endif
#ifdef SYS_nanosleep
printf("SYS_nanosleep %llu\n", (unsigned long long)(long long)(SYS_nanosleep));
#endif
#ifdef SYS_newfstatat
printf("SYS_newfstatat %llu\n", (unsigned long long)(long long)(SYS_newfstatat));
#endif
#ifdef SYS_nfsservctl
printf("SYS_nfsservctl %llu\n", (unsigned long long)(long long)(SYS_nfsservctl));
#endif
#ifdef SYS_open_by_handle_at
printf("SYS_open_by_handle_at %llu\n", (unsigned long long)(long long)(SYS_open_by_handle_at));
#endif
#ifdef SYS_open_tree
printf("SYS_open_tree %llu\n", (unsigned long long)(long long)(SYS_open_tree));
#endif
#ifdef SYS_openat
printf("SYS_openat %llu\n", (unsigned long long)(long long)(SYS_openat));
#endif
#ifdef SYS_openat2
printf("SYS_openat2 %llu\n", (unsigned long long)(long long)(SYS_openat2));
#endif
#ifdef SYS_perf_event_open
printf("SYS_perf_event_open %llu\n", (unsigned long long)(long long)(SYS_perf_event_open));
#endif
#ifdef SYS_personality
printf("SYS_personality %llu\n", (unsigned long long)(long long)(SYS_personality));
#endif
#ifdef SYS_pidfd_getfd
printf("SYS_pidfd_getfd %llu\n", (unsigned long long)(long long)(SYS_pidfd_getfd));
#endif
#ifdef SYS_pidfd_open
printf("SYS_pidfd_open %llu\n", (unsigned long long)(long long)(SYS_pidfd_open));
#endif
#ifdef SYS_pidfd_send_signal
printf("SYS_pidfd_send_signal %llu\n", (unsigned long long)(long long)(SYS_pidfd_send_signal));
#endif
#ifdef SYS_pipe2
printf("SYS_pipe2 %llu\n", (unsigned long long)(long long)(SYS_pipe2));
#endif
#ifdef SYS_pivot_root
printf("SYS_pivot_root %llu\n", (unsigned long long)(long long)(SYS_pivot_root));
#endif
#ifdef SYS_pkey_alloc
printf("SYS_pkey_alloc %llu\n", (unsigned long long)(long long)(SYS_pkey_alloc));
#endif
#ifdef SYS_pkey_free
printf("SYS_pkey_free %llu\n", (unsigned long long)(long long)(SYS_pkey_free));
#endif
#ifdef SYS_pkey_mprotect
printf("SYS_pkey_mprotect %llu\n", (unsigned long long)(long long)(SYS_pkey_mprotect));
#endif
#ifdef SYS_ppoll
printf("SYS_ppoll %llu\n", (unsigned long long)(long long)(SYS_ppoll));
#endif
#ifdef SYS_prctl
printf("SYS_prctl %llu\n", (unsigned long long)(long long)(SYS_prctl));
#endif
#ifdef SYS_pread64
printf("SYS_pread64 %llu\n", (unsigned long long)(long long)(SYS_pread64));
#endif
#ifdef SYS_preadv
printf("SYS_preadv %llu\n", (unsigned long long)(long long)(SYS_preadv));
#endif
#ifdef SYS_preadv2
printf("SYS_preadv2 %llu\n", (unsigned long long)(long long)(SYS_preadv2));
#endif
#ifdef SYS_prlimit64
printf("SYS_prlimit64 %llu\n", (unsigned long long)(long long)(SYS_prlimit64));
#endif
#ifdef SYS_process_madvise
printf("SYS_process_madvise %llu\n", (unsigned long long)(long long)(SYS_process_madvise));
#endif
#ifdef SYS_process_mrelease
printf("SYS_process_mrelease %llu\n", (unsigned long long)(long long)(SYS_process_mrelease));
#endif
#ifdef SYS_process_vm_readv
printf("SYS_process_vm_readv %llu\n", (unsigned long long)(long long)(SYS_process_vm_readv));
#endif
#ifdef SYS_process_vm_writev
printf("SYS_process_vm_writev %llu\n", (unsigned long long)(long long)(SYS_process_vm_writev));
#endif
#ifdef SYS_pselect6
printf("SYS_pselect6 %llu\n", (unsigned long long)(long long)(SYS_pselect6));
#endif
#ifdef SYS_ptrace
printf("SYS_ptrace %llu\n", (unsigned long long)(long long)(SYS_ptrace));
#endif
#ifdef SYS_pwrite64
printf("SYS_pwrite64 %llu\n", (unsigned long long)(long long)(SYS_pwrite64));
#endif
#ifdef SYS_pwritev
printf("SYS_pwritev %llu\n", (unsigned long long)(long long)(SYS_pwritev));
#endif
#ifdef SYS_pwritev2
printf("SYS_pwritev2 %llu\n", (unsigned long long)(long long)(SYS_pwritev2));
#endif
#ifdef SYS_quotactl
printf("SYS_quotactl %llu\n", (unsigned long long)(long long)(SYS_quotactl));
#endif
#ifdef SYS_quotactl_fd
printf("SYS_quotactl_fd %llu\n", (unsigned long long)(long long)(SYS_quotactl_fd));
#endif
#ifdef SYS_read
printf("SYS_read %llu\n", (unsigned long long)(long long)(SYS_read));
#endif
#ifdef SYS_readahead
printf("SYS_readahead %llu\n", (unsigned long long)(long long)(SYS_readahead));
#endif
#ifdef SYS_readlinkat
printf("SYS_readlinkat %llu\n", (unsigned long long)(long long)(SYS_readlinkat));
#endif
#ifdef SYS_readv
printf("SYS_readv %llu\n", (unsigned long long)(long long)(SYS_readv));
#endif
#ifdef SYS_reboot
printf("SYS_reboot %llu\n", (unsigned long long)(long long)(SYS_reboot));
#endif
#ifdef SYS_recvfrom
printf("SYS_recvfrom %llu\n", (unsigned long long)(long long)(SYS_recvfrom));
#endif
#ifdef SYS_recvmmsg
printf("SYS_recvmmsg %llu\n", (unsigned long long)(long long)(SYS_recvmmsg));
#endif
#ifdef SYS_recvmsg
printf("SYS_recvmsg %llu\n", (unsigned long long)(long long)(SYS_recvmsg));
#endif
#ifdef SYS_remap_file_pages
printf("SYS_remap_file_pages %llu\n", (unsigned long long)(long long)(SYS_remap_file_pages));
#endif
#ifdef SYS_removexattr
printf("SYS_removexattr %llu\n", (unsigned long long)(long long)(SYS_removexattr));
#endif
#ifdef SYS_renameat2
printf("SYS_renameat2 %llu\n", (unsigned long long)(long long)(SYS_renameat2));
#endif
#ifdef SYS_request_key
printf("SYS_request_key %llu\n", (unsigned long long)(long long)(SYS_request_key));
#endif
#ifdef SYS_restart_syscall
printf("SYS_restart_syscall %llu\n", (unsigned long long)(long long)(SYS_restart_syscall));
#endif
#ifdef SYS_rseq
printf("SYS_rseq %llu\n", (unsigned long long)(long long)(SYS_rseq));
#endif
#ifdef SYS_rt_sigaction
printf("SYS_rt_sigaction %llu\n", (unsigned long long)(long long)(SYS_rt_sigaction));
#endif
#ifdef SYS_rt_sigpending
printf("SYS_rt_sigpending %llu\n", (unsigned long long)(long long)(SYS_rt_sigpending));
#endif
#ifdef SYS_rt_sigprocmask
printf("SYS_rt_sigprocmask %llu\n", (unsigned long long)(long long)(SYS_rt_sigprocmask));
#endif
#ifdef SYS_rt_sigqueueinfo
printf("SYS_rt_sigqueueinfo %llu\n", (unsigned long long)(long long)(SYS_rt_sigqueueinfo));
#endif
#ifdef SYS_rt_sigreturn
printf("SYS_rt_sigreturn %llu\n", (unsigned long long)(long long)(SYS_rt_sigreturn));
#endif
#ifdef SYS_rt_sigsuspend
printf("SYS_rt_sigsuspend %llu\n", (unsigned long long)(long long)(SYS_rt_sigsuspend));
#endif
#ifdef SYS_rt_sigtimedwait
printf("SYS_rt_sigtimedwait %llu\n", (unsigned long long)(long long)(SYS_rt_sigtimedwait));
#endif
#ifdef SYS_rt_tgsigqueueinfo
printf("SYS_rt_tgsigqueueinfo %llu\n", (unsigned long long)(long long)(SYS_rt_tgsigqueueinfo));
#endif
#ifdef SYS_sched_get_priority_max
printf("SYS_sched_get_priority_max %llu\n", (unsigned long long)(long long)(SYS_sched_get_priority_max));
#endif
#ifdef SYS_sched_get_priority_min
printf("SYS_sched_get_priority_min %llu\n", (unsigned long long)(long long)(SYS_sched_get_priority_min));
#endif
#ifdef SYS_sched_getaffinity
printf("SYS_sched_getaffinity %llu\n", (unsigned long long)(long long)(SYS_sched_getaffinity));
#endif
#ifdef SYS_sched_getattr
printf("SYS_sched_getattr %llu\n", (unsigned long long)(long long)(SYS_sched_getattr));
#endif
#ifdef SYS_sched_getparam
printf("SYS_sched_getparam %llu\n", (unsigned long long)(long long)(SYS_sched_getparam));
#endif
#ifdef SYS_sched_getscheduler
printf("SYS_sched_getscheduler %llu\n", (unsigned long long)(long long)(SYS_sched_getscheduler));
#endif
#ifdef SYS_sched_rr_get_interval
printf("SYS_sched_rr_get_interval %llu\n", (unsigned long long)(long long)(SYS_sched_rr_get_interval));
#endif
#ifdef SYS_sched_setaffinity
printf("SYS_sched_setaffinity %llu\n", (unsigned long long)(long long)(SYS_sched_setaffinity));
#endif
#ifdef SYS_sched_setattr
printf("SYS_sched_setattr %llu\n", (unsigned long long)(long long)(SYS_sched_setattr));
#endif
#ifdef SYS_sched_setparam
printf("SYS_sched_setparam %llu\n", (unsigned long long)(long long)(SYS_sched_setparam));
#endif
#ifdef SYS_sched_setscheduler
printf("SYS_sched_setscheduler %llu\n", (unsigned long long)(long long)(SYS_sched_setscheduler));
#endif
#ifdef SYS_sched_yield
printf("SYS_sched_yield %llu\n", (unsigned long long)(long long)(SYS_sched_yield));
#endif
#ifdef SYS_seccomp
printf("SYS_seccomp %llu\n", (unsigned long long)(long long)(SYS_seccomp));
#endif
#ifdef SYS_semctl
printf("SYS_semctl %llu\n", (unsigned long long)(long long)(SYS_semctl));
#endif
#ifdef SYS_semget
printf("SYS_semget %llu\n", (unsigned long long)(long long)(SYS_semget));
#endif
#ifdef SYS_semop
printf("SYS_semop %llu\n", (unsigned long long)(long long)(SYS_semop));
#endif
#ifdef SYS_semtimedop
printf("SYS_semtimedop %llu\n", (unsigned long long)(long long)(SYS_semtimedop));
#endif
#ifdef SYS_sendfile
printf("SYS_sendfile %llu\n", (unsigned long long)(long long)(SYS_sendfile));
#endif
#ifdef SYS_sendmmsg
printf("SYS_sendmmsg %llu\n", (unsigned long long)(long long)(SYS_sendmmsg));
#endif
#ifdef SYS_sendmsg
printf("SYS_sendmsg %llu\n", (unsigned long long)(long long)(SYS_sendmsg));
#endif
#ifdef SYS_sendto
printf("SYS_sendto %llu\n", (unsigned long long)(long long)(SYS_sendto));
#endif
#ifdef SYS_set_mempolicy
printf("SYS_set_mempolicy %llu\n", (unsigned long long)(long long)(SYS_set_mempolicy));
#endif
#ifdef SYS_set_mempolicy_home_node
printf("SYS_set_mempolicy_home_node %llu\n", (unsigned long long)(long long)(SYS_set_mempolicy_home_node));
#endif
#ifdef SYS_set_robust_list
printf("SYS_set_robust_list %llu\n", (unsigned long long)(long long)(SYS_set_robust_list));
#endif
#ifdef SYS_set_tid_address
printf("SYS_set_tid_address %llu\n", (unsigned long long)(long long)(SYS_set_tid_address));
#endif
#ifdef SYS_setdomainname
printf("SYS_setdomainname %llu\n", (unsigned long long)(long long)(SYS_setdomainname));
#endif
#ifdef SYS_setfsgid
printf("SYS_setfsgid %llu\n", (unsigned long long)(long long)(SYS_setfsgid));
#endif
#ifdef SYS_setfsuid
printf("SYS_setfsuid %llu\n", (unsigned long long)(long long)(SYS_setfsuid));
#endif
#ifdef SYS_setgid
printf("SYS_setgid %llu\n", (unsigned long long)(long long)(SYS_setgid));
#endif
#ifdef SYS_setgroups
printf("SYS_setgroups %llu\n", (unsigned long long)(long long)(SYS_setgroups));
#endif
#ifdef SYS_sethostname
printf("SYS_sethostname %llu\n", (unsigned long long)(long long)(SYS_sethostname));
#endif
#ifdef SYS_setitimer
printf("SYS_setitimer %llu\n", (unsigned long long)(long long)(SYS_setitimer));
#endif
#ifdef SYS_setns
printf("SYS_setns %llu\n", (unsigned long long)(long long)(SYS_setns));
#endif
#ifdef SYS_setpgid
printf("SYS_setpgid %llu\n", (unsigned long long)(long long)(SYS_setpgid));
#endif
#ifdef SYS_setpriority
printf("SYS_setpriority %llu\n", (unsigned long long)(long long)(SYS_setpriority));
#endif
#ifdef SYS_setregid
printf("SYS_setregid %llu\n", (unsigned long long)(long long)(SYS_setregid));
#endif
#ifdef SYS_setresgid
printf("SYS_setresgid %llu\n", (unsigned long long)(long long)(SYS_setresgid));
#endif
#ifdef SYS_setresuid
printf("SYS_setresuid %llu\n", (unsigned long long)(long long)(SYS_setresuid));
#endif
#ifdef SYS_setreuid
printf("SYS_setreuid %llu\n", (unsigned long long)(long long)(SYS_setreuid));
#endif
#ifdef SYS_setrlimit
printf("SYS_setrlimit %llu\n", (unsigned long long)(long long)(SYS_setrlimit));
#endif
#ifdef SYS_setsid
printf("SYS_setsid %llu\n", (unsigned long long)(long long)(SYS_setsid));
#endif
#ifdef SYS_setsockopt
printf("SYS_setsockopt %llu\n", (unsigned long long)(long long)(SYS_setsockopt));
#endif
#ifdef SYS_settimeofday
printf("SYS_settimeofday %llu\n", (unsigned long long)(long long)(SYS_settimeofday));
#endif
#ifdef SYS_setuid
printf("SYS_setuid %llu\n", (unsigned long long)(long long)(SYS_setuid));
#endif
#ifdef SYS_setxattr
printf("SYS_setxattr %llu\n", (unsigned long long)(long long)(SYS_setxattr));
#endif
#ifdef SYS_shmat
printf("SYS_shmat %llu\n", (unsigned long long)(long long)(SYS_shmat));
#endif
#ifdef SYS_shmctl
printf("SYS_shmctl %llu\n", (unsigned long long)(long long)(SYS_shmctl));
#endif
#ifdef SYS_shmdt
printf("SYS_shmdt %llu\n", (unsigned long long)(long long)(SYS_shmdt));
#endif
#ifdef SYS_shmget
printf("SYS_shmget %llu\n", (unsigned long long)(long long)(SYS_shmget));
#endif
#ifdef SYS_shutdown
printf("SYS_shutdown %llu\n", (unsigned long long)(long long)(SYS_shutdown));
#endif
#ifdef SYS_sigaltstack
printf("SYS_sigaltstack %llu\n", (unsigned long long)(long long)(SYS_sigaltstack));
#endif
#ifdef SYS_signalfd4
printf("SYS_signalfd4 %llu\n", (unsigned long long)(long long)(SYS_signalfd4));
#endif
#ifdef SYS_socket
printf("SYS_socket %llu\n", (unsigned long long)(long long)(SYS_socket));
#endif
#ifdef SYS_socketpair
printf("SYS_socketpair %llu\n", (unsigned long long)(long long)(SYS_socketpair));
#endif
#ifdef SYS_splice
printf("SYS_splice %llu\n", (unsigned long long)(long long)(SYS_splice));
#endif
#ifdef SYS_statfs
printf("SYS_statfs %llu\n", (unsigned long long)(long long)(SYS_statfs));
#endif
#ifdef SYS_statx
printf("SYS_statx %llu\n", (unsigned long long)(long long)(SYS_statx));
#endif
#ifdef SYS_swapoff
printf("SYS_swapoff %llu\n", (unsigned long long)(long long)(SYS_swapoff));
#endif
#ifdef SYS_swapon
printf("SYS_swapon %llu\n", (unsigned long long)(long long)(SYS_swapon));
#endif
#ifdef SYS_symlinkat
printf("SYS_symlinkat %llu\n", (unsigned long long)(long long)(SYS_symlinkat));
#endif
#ifdef SYS_sync
printf("SYS_sync %llu\n", (unsigned long long)(long long)(SYS_sync));
#endif
#ifdef SYS_sync_file_range
printf("SYS_sync_file_range %llu\n", (unsigned long long)(long long)(SYS_sync_file_range));
#endif
#ifdef SYS_syncfs
printf("SYS_syncfs %llu\n", (unsigned long long)(long long)(SYS_syncfs));
#endif
#ifdef SYS_sysinfo
printf("SYS_sysinfo %llu\n", (unsigned long long)(long long)(SYS_sysinfo));
#endif
#ifdef SYS_syslog
printf("SYS_syslog %llu\n", (unsigned long long)(long long)(SYS_syslog));
#endif
#ifdef SYS_tee
printf("SYS_tee %llu\n", (unsigned long long)(long long)(SYS_tee));
#endif
#ifdef SYS_tgkill
printf("SYS_tgkill %llu\n", (unsigned long long)(long long)(SYS_tgkill));
#endif
#ifdef SYS_timer_create
printf("SYS_timer_create %llu\n", (unsigned long long)(long long)(SYS_timer_create));
#endif
#ifdef SYS_timer_delete
printf("SYS_timer_delete %llu\n", (unsigned long long)(long long)(SYS_timer_delete));
#endif
#ifdef SYS_timer_getoverrun
printf("SYS_timer_getoverrun %llu\n", (unsigned long long)(long long)(SYS_timer_getoverrun));
#endif
#ifdef SYS_timer_gettime
printf("SYS_timer_gettime %llu\n", (unsigned long long)(long long)(SYS_timer_gettime));
#endif
#ifdef SYS_timer_settime
printf("SYS_timer_settime %llu\n", (unsigned long long)(long long)(SYS_timer_settime));
#endif
#ifdef SYS_timerfd_create
printf("SYS_timerfd_create %llu\n", (unsigned long long)(long long)(SYS_timerfd_create));
#endif
#ifdef SYS_timerfd_gettime
printf("SYS_timerfd_gettime %llu\n", (unsigned long long)(long long)(SYS_timerfd_gettime));
#endif
#ifdef SYS_timerfd_settime
printf("SYS_timerfd_settime %llu\n", (unsigned long long)(long long)(SYS_timerfd_settime));
#endif
#ifdef SYS_times
printf("SYS_times %llu\n", (unsigned long long)(long long)(SYS_times));
#endif
#ifdef SYS_tkill
printf("SYS_tkill %llu\n", (unsigned long long)(long long)(SYS_tkill));
#endif
#ifdef SYS_truncate
printf("SYS_truncate %llu\n", (unsigned long long)(long long)(SYS_truncate));
#endif
#ifdef SYS_umask
printf("SYS_umask %llu\n", (unsigned long long)(long long)(SYS_umask));
#endif
#ifdef SYS_umount2
printf("SYS_umount2 %llu\n", (unsigned long long)(long long)(SYS_umount2));
#endif
#ifdef SYS_uname
printf("SYS_uname %llu\n", (unsigned long long)(long long)(SYS_uname));
#endif
#ifdef SYS_unlinkat
printf("SYS_unlinkat %llu\n", (unsigned long long)(long long)(SYS_unlinkat));
#endif
#ifdef SYS_unshare
printf("SYS_unshare %llu\n", (unsigned long long)(long long)(SYS_unshare));
#endif
#ifdef SYS_userfaultfd
printf("SYS_userfaultfd %llu\n", (unsigned long long)(long long)(SYS_userfaultfd));
#endif
#ifdef SYS_utimensat
printf("SYS_utimensat %llu\n", (unsigned long long)(long long)(SYS_utimensat));
#endif
#ifdef SYS_vhangup
printf("SYS_vhangup %llu\n", (unsigned long long)(long long)(SYS_vhangup));
#endif
#ifdef SYS_vmsplice
printf("SYS_vmsplice %llu\n", (unsigned long long)(long long)(SYS_vmsplice));
#endif
#ifdef SYS_wait4
printf("SYS_wait4 %llu\n", (unsigned long long)(long long)(SYS_wait4));
#endif
#ifdef SYS_waitid
printf("SYS_waitid %llu\n", (unsigned long long)(long long)(SYS_waitid));
#endif
#ifdef SYS_write
printf("SYS_write %llu\n", (unsigned long long)(long long)(SYS_write));
#endif
#ifdef SYS_writev
printf("SYS_writev %llu\n", (unsigned long long)(long long)(SYS_writev));
#endif
#ifdef S_IEXEC
printf("S_IEXEC %llu\n", (unsigned long long)(long long)(S_IEXEC));
#endif
#ifdef S_IFBLK
printf("S_IFBLK %llu\n", (unsigned long long)(long long)(S_IFBLK));
#endif
#ifdef S_IFCHR
printf("S_IFCHR %llu\n", (unsigned long long)(long long)(S_IFCHR));
#endif
#ifdef S_IFDIR
printf("S_IFDIR %llu\n", (unsigned long long)(long long)(S_IFDIR));
#endif
#ifdef S_IFIFO
printf("S_IFIFO %llu\n", (unsigned long long)(long long)(S_IFIFO));
#endif
#ifdef S_IFLNK
printf("S_IFLNK %llu\n", (unsigned long long)(long long)(S_IFLNK));
#endif
#ifdef S_IFMT
printf("S_IFMT %llu\n", (unsigned long long)(long long)(S_IFMT));
#endif
#ifdef S_IFREG
printf("S_IFREG %llu\n", (unsigned long long)(long long)(S_IFREG));
#endif
#ifdef S_IFSOCK
printf("S_IFSOCK %llu\n", (unsigned long long)(long long)(S_IFSOCK));
#endif
#ifdef S_IREAD
printf("S_IREAD %llu\n", (unsigned long long)(long long)(S_IREAD));
#endif
#ifdef S_IRGRP
printf("S_IRGRP %llu\n", (unsigned long long)(long long)(S_IRGRP));
#endif
#ifdef S_IROTH
printf("S_IROTH %llu\n", (unsigned long long)(long long)(S_IROTH));
#endif
#ifdef S_IRUSR
printf("S_IRUSR %llu\n", (unsigned long long)(long long)(S_IRUSR));
#endif
#ifdef S_IRWXG
printf("S_IRWXG %llu\n", (unsigned long long)(long long)(S_IRWXG));
#endif
#ifdef S_IRWXO
printf("S_IRWXO %llu\n", (unsigned long long)(long long)(S_IRWXO));
#endif
#ifdef S_IRWXU
printf("S_IRWXU %llu\n", (unsigned long long)(long long)(S_IRWXU));
#endif
#ifdef S_ISGID
printf("S_ISGID %llu\n", (unsigned long long)(long long)(S_ISGID));
#endif
#ifdef S_ISUID
printf("S_ISUID %llu\n", (unsigned long long)(long long)(S_ISUID));
#endif
#ifdef S_ISVTX
printf("S_ISVTX %llu\n", (unsigned long long)(long long)(S_ISVTX));
#endif
#ifdef S_IWGRP
printf("S_IWGRP %llu\n", (unsigned long long)(long long)(S_IWGRP));
#endif
#ifdef S_IWOTH
printf("S_IWOTH %llu\n", (unsigned long long)(long long)(S_IWOTH));
#endif
#ifdef S_IWRITE
printf("S_IWRITE %llu\n", (unsigned long long)(long long)(S_IWRITE));
#endif
#ifdef S_IWUSR
printf("S_IWUSR %llu\n", (unsigned long long)(long long)(S_IWUSR));
#endif
#ifdef S_IXGRP
printf("S_IXGRP %llu\n", (unsigned long long)(long long)(S_IXGRP));
#endif
#ifdef S_IXOTH
printf("S_IXOTH %llu\n", (unsigned long long)(long long)(S_IXOTH));
#endif
#ifdef S_IXUSR
printf("S_IXUSR %llu\n", (unsigned long long)(long long)(S_IXUSR));
#endif
#ifdef TAB0
printf("TAB0 %llu\n", (unsigned long long)(long long)(TAB0));
#endif
#ifdef TAB1
printf("TAB1 %llu\n", (unsigned long long)(long long)(TAB1));
#endif
#ifdef TAB2
printf("TAB2 %llu\n", (unsigned long long)(long long)(TAB2));
#endif
#ifdef TAB3
printf("TAB3 %llu\n", (unsigned long long)(long long)(TAB3));
#endif
#ifdef TABDLY
printf("TABDLY %llu\n", (unsigned long long)(long long)(TABDLY));
#endif
#ifdef TCA_CHAIN
printf("TCA_CHAIN %llu\n", (unsigned long long)(long long)(TCA_CHAIN));
#endif
#ifdef TCA_DUMP_INVISIBLE
printf("TCA_DUMP_INVISIBLE %llu\n", (unsigned long long)(long long)(TCA_DUMP_INVISIBLE));
#endif
#ifdef TCA_FCNT
printf("TCA_FCNT %llu\n", (unsigned long long)(long long)(TCA_FCNT));
#endif
#ifdef TCA_HW_OFFLOAD
printf("TCA_HW_OFFLOAD %llu\n", (unsigned long long)(long long)(TCA_HW_OFFLOAD));
#endif
#ifdef TCA_KIND
printf("TCA_KIND %llu\n", (unsigned long long)(long long)(TCA_KIND));
#endif
#ifdef TCA_OPTIONS
printf("TCA_OPTIONS %llu\n", (unsigned long long)(long long)(TCA_OPTIONS));
#endif
#ifdef TCA_PAD
printf("TCA_PAD %llu\n", (unsigned long long)(long long)(TCA_PAD));
#endif
#ifdef TCA_RATE
printf("TCA_RATE %llu\n", (unsigned long long)(long long)(TCA_RATE));
#endif
#ifdef TCA_STAB
printf("TCA_STAB %llu\n", (unsigned long long)(long long)(TCA_STAB));
#endif
#ifdef TCA_STATS
printf("TCA_STATS %llu\n", (unsigned long long)(long long)(TCA_STATS));
#endif
#ifdef TCA_STATS2
printf("TCA_STATS2 %llu\n", (unsigned long long)(long long)(TCA_STATS2));
#endif
#ifdef TCA_UNSPEC
printf("TCA_UNSPEC %llu\n", (unsigned long long)(long long)(TCA_UNSPEC));
#endif
#ifdef TCA_XSTATS
printf("TCA_XSTATS %llu\n", (unsigned long long)(long long)(TCA_XSTATS));
#endif
#ifdef TCFLSH
printf("TCFLSH %llu\n", (unsigned long long)(long long)(TCFLSH));
#endif
#ifdef TCGETA
printf("TCGETA %llu\n", (unsigned long long)(long long)(TCGETA));
#endif
#ifdef TCGETS
printf("TCGETS %llu\n", (unsigned long long)(long long)(TCGETS));
#endif
#ifdef TCGETS2
printf("TCGETS2 %llu\n", (unsigned long long)(long long)(TCGETS2));
#endif
#ifdef TCGETX
printf("TCGETX %llu\n", (unsigned long long)(long long)(TCGETX));
#endif
#ifdef TCIFLUSH
printf("TCIFLUSH %llu\n", (unsigned long long)(long long)(TCIFLUSH));
#endif
#ifdef TCIOFF
printf("TCIOFF %llu\n", (unsigned long long)(long long)(TCIOFF));
#endif
#ifdef TCIOFLUSH
printf("TCIOFLUSH %llu\n", (unsigned long long)(long long)(TCIOFLUSH));
#endif
#ifdef TCION
printf("TCION %llu\n", (unsigned long long)(long long)(TCION));
#endif
#ifdef TCOFLUSH
printf("TCOFLUSH %llu\n", (unsigned long long)(long long)(TCOFLUSH));
#endif
#ifdef TCOOFF
printf("TCOOFF %llu\n", (unsigned long long)(long long)(TCOOFF));
#endif
#ifdef TCOON
printf("TCOON %llu\n", (unsigned long long)(long long)(TCOON));
#endif
#ifdef TCP_CC_INFO
printf("TCP_CC_INFO %llu\n", (unsigned long long)(long long)(TCP_CC_INFO));
#endif
#ifdef TCP_CM_INQ
printf("TCP_CM_INQ %llu\n", (unsigned long long)(long long)(TCP_CM_INQ));
#endif
#ifdef TCP_CONGESTION
printf("TCP_CONGESTION %llu\n", (unsigned long long)(long long)(TCP_CONGESTION));
#endif
#ifdef TCP_COOKIE_TRANSACTIONS
printf("TCP_COOKIE_TRANSACTIONS %llu\n", (unsigned long long)(long long)(TCP_COOKIE_TRANSACTIONS));
#endif
#ifdef TCP_CORK
printf("TCP_CORK %llu\n", (unsigned long long)(long long)(TCP_CORK));
#endif
#ifdef TCP_DEFER_ACCEPT
printf("TCP_DEFER_ACCEPT %llu\n", (unsigned long long)(long long)(TCP_DEFER_ACCEPT));
#endif
#ifdef TCP_FASTOPEN
printf("TCP_FASTOPEN %llu\n", (unsigned long long)(long long)(TCP_FASTOPEN));
#endif
#ifdef TCP_FASTOPEN_CONNECT
printf("TCP_FASTOPEN_CONNECT %llu\n", (unsigned long long)(long long)(TCP_FASTOPEN_CONNECT));
#endif
#ifdef TCP_FASTOPEN_KEY
printf("TCP_FASTOPEN_KEY %llu\n", (unsigned long long)(long long)(TCP_FASTOPEN_KEY));
#endif
#ifdef TCP_FASTOPEN_NO_COOKIE
printf("TCP_FASTOPEN_NO_COOKIE %llu\n", (unsigned long long)(long long)(TCP_FASTOPEN_NO_COOKIE));
#endif
#ifdef TCP_INFO
printf("TCP_INFO %llu\n", (unsigned long long)(long long)(TCP_INFO));
#endif
#ifdef TCP_INQ
printf("TCP_INQ %llu\n", (unsigned long long)(long long)(TCP_INQ));
#endif
#ifdef TCP_KEEPCNT
printf("TCP_KEEPCNT %llu\n", (unsigned long long)(long long)(TCP_KEEPCNT));
#endif
#ifdef TCP_KEEPIDLE
printf("TCP_KEEPIDLE %llu\n", (unsigned long long)(long long)(TCP_KEEPIDLE));
#endif
#ifdef TCP_KEEPINTVL
printf("TCP_KEEPINTVL %llu\n", (unsigned long long)(long long)(TCP_KEEPINTVL));
#endif
#ifdef TCP_LINGER2
printf("TCP_LINGER2 %llu\n", (unsigned long long)(long long)(TCP_LINGER2));
#endif
#ifdef TCP_MAXSEG
printf("TCP_MAXSEG %llu\n", (unsigned long long)(long long)(TCP_MAXSEG));
#endif
#ifdef TCP_MD5SIG
printf("TCP_MD5SIG %llu\n", (unsigned long long)(long long)(TCP_MD5SIG));
#endif
#ifdef TCP_MD5SIG_EXT
printf("TCP_MD5SIG_EXT %llu\n", (unsigned long long)(long long)(TCP_MD5SIG_EXT));
#endif
#ifdef TCP_MD5SIG_MAXKEYLEN
printf("TCP_MD5SIG_MAXKEYLEN %llu\n", (unsigned long long)(long long)(TCP_MD5SIG_MAXKEYLEN));
#endif
#ifdef TCP_NODELAY
printf("TCP_NODELAY %llu\n", (unsigned long long)(long long)(TCP_NODELAY));
#endif
#ifdef TCP_NOTSENT_LOWAT
printf("TCP_NOTSENT_LOWAT %llu\n", (unsigned long long)(long long)(TCP_NOTSENT_LOWAT));
#endif
#ifdef TCP_QUEUE_SEQ
printf("TCP_QUEUE_SEQ %llu\n", (unsigned long long)(long long)(TCP_QUEUE_SEQ));
#endif
#ifdef TCP_QUICKACK
printf("TCP_QUICKACK %llu\n", (unsigned long long)(long long)(TCP_QUICKACK));
#endif
#ifdef TCP_REPAIR
printf("TCP_REPAIR %llu\n", (unsigned long long)(long long)(TCP_REPAIR));
#endif
#ifdef TCP_REPAIR_OPTIONS
printf("TCP_REPAIR_OPTIONS %llu\n", (unsigned long long)(long long)(TCP_REPAIR_OPTIONS));
#endif
#ifdef TCP_REPAIR_QUEUE
printf("TCP_REPAIR_QUEUE %llu\n", (unsigned long long)(long long)(TCP_REPAIR_QUEUE));
#endif
#ifdef TCP_REPAIR_WINDOW
printf("TCP_REPAIR_WINDOW %llu\n", (unsigned long long)(long long)(TCP_REPAIR_WINDOW));
#endif
#ifdef TCP_SAVED_SYN
printf("TCP_SAVED_SYN %llu\n", (unsigned long long)(long long)(TCP_SAVED_SYN));
#endif
#ifdef TCP_SAVE_SYN
printf("TCP_SAVE_SYN %llu\n", (unsigned long long)(long long)(TCP_SAVE_SYN));
#endif
#ifdef TCP_SYNCNT
printf("TCP_SYNCNT %llu\n", (unsigned long long)(long long)(TCP_SYNCNT));
#endif
#ifdef TCP_THIN_DUPACK
printf("TCP_THIN_DUPACK %llu\n", (unsigned long long)(long long)(TCP_THIN_DUPACK));
#endif
#ifdef TCP_THIN_LINEAR_TIMEOUTS
printf("TCP_THIN_LINEAR_TIMEOUTS %llu\n", (unsigned long long)(long long)(TCP_THIN_LINEAR_TIMEOUTS));
#endif
#ifdef TCP_TIMESTAMP
printf("TCP_TIMESTAMP %llu\n", (unsigned long long)(long long)(TCP_TIMESTAMP));
#endif
#ifdef TCP_TX_DELAY
printf("TCP_TX_DELAY %llu\n", (unsigned long long)(long long)(TCP_TX_DELAY));
#endif
#ifdef TCP_ULP
printf("TCP_ULP %llu\n", (unsigned long long)(long long)(TCP_ULP));
#endif
#ifdef TCP_USER_TIMEOUT
printf("TCP_USER_TIMEOUT %llu\n", (unsigned long long)(long long)(TCP_USER_TIMEOUT));
#endif
#ifdef TCP_WINDOW_CLAMP
printf("TCP_WINDOW_CLAMP %llu\n", (unsigned long long)(long long)(TCP_WINDOW_CLAMP));
#endif
#ifdef TCP_ZEROCOPY_RECEIVE
printf("TCP_ZEROCOPY_RECEIVE %llu\n", (unsigned long long)(long long)(TCP_ZEROCOPY_RECEIVE));
#endif
#ifdef TCSADRAIN
printf("TCSADRAIN %llu\n", (unsigned long long)(long long)(TCSADRAIN));
#endif
#ifdef TCSAFLUSH
printf("TCSAFLUSH %llu\n", (unsigned long long)(long long)(TCSAFLUSH));
#endif
#ifdef TCSANOW
printf("TCSANOW %llu\n", (unsigned long long)(long long)(TCSANOW));
#endif
#ifdef TCSBRK
printf("TCSBRK %llu\n", (unsigned long long)(long long)(TCSBRK));
#endif
#ifdef TCSBRKP
printf("TCSBRKP %llu\n", (unsigned long long)(long long)(TCSBRKP));
#endif
#ifdef TCSETA
printf("TCSETA %llu\n", (unsigned long long)(long long)(TCSETA));
#endif
#ifdef TCSETAF
printf("TCSETAF %llu\n", (unsigned long long)(long long)(TCSETAF));
#endif
#ifdef TCSETAW
printf("TCSETAW %llu\n", (unsigned long long)(long long)(TCSETAW));
#endif
#ifdef TCSETS
printf("TCSETS %llu\n", (unsigned long long)(long long)(TCSETS));
#endif
#ifdef TCSETS2
printf("TCSETS2 %llu\n", (unsigned long long)(long long)(TCSETS2));
#endif
#ifdef TCSETSF
printf("TCSETSF %llu\n", (unsigned long long)(long long)(TCSETSF));
#endif
#ifdef TCSETSF2
printf("TCSETSF2 %llu\n", (unsigned long long)(long long)(TCSETSF2));
#endif
#ifdef TCSETSW
printf("TCSETSW %llu\n", (unsigned long long)(long long)(TCSETSW));
#endif
#ifdef TCSETSW2
printf("TCSETSW2 %llu\n", (unsigned long long)(long long)(TCSETSW2));
#endif
#ifdef TCSETX
printf("TCSETX %llu\n", (unsigned long long)(long long)(TCSETX));
#endif
#ifdef TCSETXF
printf("TCSETXF %llu\n", (unsigned long long)(long long)(TCSETXF));
#endif
#ifdef TCSETXW
printf("TCSETXW %llu\n", (unsigned long long)(long long)(TCSETXW));
#endif
#ifdef TCXONC
printf("TCXONC %llu\n", (unsigned long long)(long long)(TCXONC));
#endif
#ifdef TFD_CLOEXEC
printf("TFD_CLOEXEC %llu\n", (unsigned long long)(long long)(TFD_CLOEXEC));
#endif
#ifdef TFD_NONBLOCK
printf("TFD_NONBLOCK %llu\n", (unsigned long long)(long long)(TFD_NONBLOCK));
#endif
#ifdef TFD_TIMER_ABSTIME
printf("TFD_TIMER_ABSTIME %llu\n", (unsigned long long)(long long)(TFD_TIMER_ABSTIME));
#endif
#ifdef TFD_TIMER_CANCEL_ON_SET
printf("TFD_TIMER_CANCEL_ON_SET %llu\n", (unsigned long long)(long long)(TFD_TIMER_CANCEL_ON_SET));
#endif
#ifdef THOUSEP
printf("THOUSEP %llu\n", (unsigned long long)(long long)(THOUSEP));
#endif
#ifdef TIMER_ABSTIME
printf("TIMER_ABSTIME %llu\n", (unsigned long long)(long long)(TIMER_ABSTIME));
#endif
#ifdef TIME_BAD
printf("TIME_BAD %llu\n", (unsigned long long)(long long)(TIME_BAD));
#endif
#ifdef TIME_DEL
printf("TIME_DEL %llu\n", (unsigned long long)(long long)(TIME_DEL));
#endif
#ifdef TIME_ERROR
printf("TIME_ERROR %llu\n", (unsigned long long)(long long)(TIME_ERROR));
#endif
#ifdef TIME_INS
printf("TIME_INS %llu\n", (unsigned long long)(long long)(TIME_INS));
#endif
#ifdef TIME_OK
printf("TIME_OK %llu\n", (unsigned long long)(long long)(TIME_OK));
#endif
#ifdef TIME_OOP
printf("TIME_OOP %llu\n", (unsigned long long)(long long)(TIME_OOP));
#endif
#ifdef TIME_WAIT
printf("TIME_WAIT %llu\n", (unsigned long long)(long long)(TIME_WAIT));
#endif
#ifdef TIOCCBRK
printf("TIOCCBRK %llu\n", (unsigned long long)(long long)(TIOCCBRK));
#endif
#ifdef TIOCCONS
printf("TIOCCONS %llu\n", (unsigned long long)(long long)(TIOCCONS));
#endif
#ifdef TIOCEXCL
printf("TIOCEXCL %llu\n", (unsigned long long)(long long)(TIOCEXCL));
#endif
#ifdef TIOCGDEV
printf("TIOCGDEV %llu\n", (unsigned long long)(long long)(TIOCGDEV));
#endif
#ifdef TIOCGETD
printf("TIOCGETD %llu\n", (unsigned long long)(long long)(TIOCGETD));
#endif
#ifdef TIOCGEXCL
printf("TIOCGEXCL %llu\n", (unsigned long long)(long long)(TIOCGEXCL));
#endif
#ifdef TIOCGICOUNT
printf("TIOCGICOUNT %llu\n", (unsigned long long)(long long)(TIOCGICOUNT));
#endif
#ifdef TIOCGISO7816
printf("TIOCGISO7816 %llu\n", (unsigned long long)(long long)(TIOCGISO7816));
#endif
#ifdef TIOCGLCKTRMIOS
printf("TIOCGLCKTRMIOS %llu\n", (unsigned long long)(long long)(TIOCGLCKTRMIOS));
#endif
#ifdef TIOCGPGRP
printf("TIOCGPGRP %llu\n", (unsigned long long)(long long)(TIOCGPGRP));
#endif
#ifdef TIOCGPKT
printf("TIOCGPKT %llu\n", (unsigned long long)(long long)(TIOCGPKT));
#endif
#ifdef TIOCGPTLCK
printf("TIOCGPTLCK %llu\n", (unsigned long long)(long long)(TIOCGPTLCK));
#endif
#ifdef TIOCGPTN
printf("TIOCGPTN %llu\n", (unsigned long long)(long long)(TIOCGPTN));
#endif
#ifdef TIOCGPTPEER
printf("TIOCGPTPEER %llu\n", (unsigned long long)(long long)(TIOCGPTPEER));
#endif
#ifdef TIOCGRS485
printf("TIOCGRS485 %llu\n", (unsigned long long)(long long)(TIOCGRS485));
#endif
#ifdef TIOCGSERIAL
printf("TIOCGSERIAL %llu\n", (unsigned long long)(long long)(TIOCGSERIAL));
#endif
#ifdef TIOCGSID
printf("TIOCGSID %llu\n", (unsigned long long)(long long)(TIOCGSID));
#endif
#ifdef TIOCGSOFTCAR
printf("TIOCGSOFTCAR %llu\n", (unsigned long long)(long long)(TIOCGSOFTCAR));
#endif
#ifdef TIOCGWINSZ
printf("TIOCGWINSZ %llu\n", (unsigned long long)(long long)(TIOCGWINSZ));
#endif
#ifdef TIOCINQ
printf("TIOCINQ %llu\n", (unsigned long long)(long long)(TIOCINQ));
#endif
#ifdef TIOCLINUX
printf("TIOCLINUX %llu\n", (unsigned long long)(long long)(TIOCLINUX));
#endif
#ifdef TIOCMBIC
printf("TIOCMBIC %llu\n", (unsigned long long)(long long)(TIOCMBIC));
#endif
#ifdef TIOCMBIS
printf("TIOCMBIS %llu\n", (unsigned long long)(long long)(TIOCMBIS));
#endif
#ifdef TIOCMGET
printf("TIOCMGET %llu\n", (unsigned long long)(long long)(TIOCMGET));
#endif
#ifdef TIOCMIWAIT
printf("TIOCMIWAIT %llu\n", (unsigned long long)(long long)(TIOCMIWAIT));
#endif
#ifdef TIOCMSET
printf("TIOCMSET %llu\n", (unsigned long long)(long long)(TIOCMSET));
#endif
#ifdef TIOCM_CAR
printf("TIOCM_CAR %llu\n", (unsigned long long)(long long)(TIOCM_CAR));
#endif
#ifdef TIOCM_CD
printf("TIOCM_CD %llu\n", (unsigned long long)(long long)(TIOCM_CD));
#endif
#ifdef TIOCM_CTS
printf("TIOCM_CTS %llu\n", (unsigned long long)(long long)(TIOCM_CTS));
#endif
#ifdef TIOCM_DSR
printf("TIOCM_DSR %llu\n", (unsigned long long)(long long)(TIOCM_DSR));
#endif
#ifdef TIOCM_DTR
printf("TIOCM_DTR %llu\n", (unsigned long long)(long long)(TIOCM_DTR));
#endif
#ifdef TIOCM_LE
printf("TIOCM_LE %llu\n", (unsigned long long)(long long)(TIOCM_LE));
#endif
#ifdef TIOCM_RI
printf("TIOCM_RI %llu\n", (unsigned long long)(long long)(TIOCM_RI));
#endif
#ifdef TIOCM_RNG
printf("TIOCM_RNG %llu\n", (unsigned long long)(long long)(TIOCM_RNG));
#endif
#ifdef TIOCM_RTS
printf("TIOCM_RTS %llu\n", (unsigned long long)(long long)(TIOCM_RTS));
#endif
#ifdef TIOCM_SR
printf("TIOCM_SR %llu\n", (unsigned long long)(long long)(TIOCM_SR));
#endif
#ifdef TIOCM_ST
printf("TIOCM_ST %llu\n", (unsigned long long)(long long)(TIOCM_ST));
#endif
#ifdef TIOCNOTTY
printf("TIOCNOTTY %llu\n", (unsigned long long)(long long)(TIOCNOTTY));
#endif
#ifdef TIOCNXCL
printf("TIOCNXCL %llu\n", (unsigned long long)(long long)(TIOCNXCL));
#endif
#ifdef TIOCOUTQ
printf("TIOCOUTQ %llu\n", (unsigned long long)(long long)(TIOCOUTQ));
#endif
#ifdef TIOCPKT
printf("TIOCPKT %llu\n", (unsigned long long)(long long)(TIOCPKT));
#endif
#ifdef TIOCSBRK
printf("TIOCSBRK %llu\n", (unsigned long long)(long long)(TIOCSBRK));
#endif
#ifdef TIOCSCTTY
printf("TIOCSCTTY %llu\n", (unsigned long long)(long long)(TIOCSCTTY));
#endif
#ifdef TIOCSERCONFIG
printf("TIOCSERCONFIG %llu\n", (unsigned long long)(long long)(TIOCSERCONFIG));
#endif
#ifdef TIOCSERGETLSR
printf("TIOCSERGETLSR %llu\n", (unsigned long long)(long long)(TIOCSERGETLSR));
#endif
#ifdef TIOCSERGETMULTI
printf("TIOCSERGETMULTI %llu\n", (unsigned long long)(long long)(TIOCSERGETMULTI));
#endif
#ifdef TIOCSERGSTRUCT
printf("TIOCSERGSTRUCT %llu\n", (unsigned long long)(long long)(TIOCSERGSTRUCT));
#endif
#ifdef TIOCSERGWILD
printf("TIOCSERGWILD %llu\n", (unsigned long long)(long long)(TIOCSERGWILD));
#endif
#ifdef TIOCSERSETMULTI
printf("TIOCSERSETMULTI %llu\n", (unsigned long long)(long long)(TIOCSERSETMULTI));
#endif
#ifdef TIOCSERSWILD
printf("TIOCSERSWILD %llu\n", (unsigned long long)(long long)(TIOCSERSWILD));
#endif
#ifdef TIOCSETD
printf("TIOCSETD %llu\n", (unsigned long long)(long long)(TIOCSETD));
#endif
#ifdef TIOCSIG
printf("TIOCSIG %llu\n", (unsigned long long)(long long)(TIOCSIG));
#endif
#ifdef TIOCSISO7816
printf("TIOCSISO7816 %llu\n", (unsigned long long)(long long)(TIOCSISO7816));
#endif
#ifdef TIOCSLCKTRMIOS
printf("TIOCSLCKTRMIOS %llu\n", (unsigned long long)(long long)(TIOCSLCKTRMIOS));
#endif
#ifdef TIOCSPGRP
printf("TIOCSPGRP %llu\n", (unsigned long long)(long long)(TIOCSPGRP));
#endif
#ifdef TIOCSPTLCK
printf("TIOCSPTLCK %llu\n", (unsigned long long)(long long)(TIOCSPTLCK));
#endif
#ifdef TIOCSRS485
printf("TIOCSRS485 %llu\n", (unsigned long long)(long long)(TIOCSRS485));
#endif
#ifdef TIOCSSERIAL
printf("TIOCSSERIAL %llu\n", (unsigned long long)(long long)(TIOCSSERIAL));
#endif
#ifdef TIOCSSOFTCAR
printf("TIOCSSOFTCAR %llu\n", (unsigned long long)(long long)(TIOCSSOFTCAR));
#endif
#ifdef TIOCSTI
printf("TIOCSTI %llu\n", (unsigned long long)(long long)(TIOCSTI));
#endif
#ifdef TIOCSWINSZ
printf("TIOCSWINSZ %llu\n", (unsigned long long)(long long)(TIOCSWINSZ));
#endif
#ifdef TIOCVHANGUP
printf("TIOCVHANGUP %llu\n", (unsigned long long)(long long)(TIOCVHANGUP));
#endif
#ifdef TLS_1_2_VERSION
printf("TLS_1_2_VERSION %llu\n", (unsigned long long)(long long)(TLS_1_2_VERSION));
#endif
#ifdef TLS_1_2_VERSION_MAJOR
printf("TLS_1_2_VERSION_MAJOR %llu\n", (unsigned long long)(long long)(TLS_1_2_VERSION_MAJOR));
#endif
#ifdef TLS_1_2_VERSION_MINOR
printf("TLS_1_2_VERSION_MINOR %llu\n", (unsigned long long)(long long)(TLS_1_2_VERSION_MINOR));
#endif
#ifdef TLS_1_3_VERSION
printf("TLS_1_3_VERSION %llu\n", (unsigned long long)(long long)(TLS_1_3_VERSION));
#endif
#ifdef TLS_1_3_VERSION_MAJOR
printf("TLS_1_3_VERSION_MAJOR %llu\n", (unsigned long long)(long long)(TLS_1_3_VERSION_MAJOR));
#endif
#ifdef TLS_1_3_VERSION_MINOR
printf("TLS_1_3_VERSION_MINOR %llu\n", (unsigned long long)(long long)(TLS_1_3_VERSION_MINOR));
#endif
#ifdef TLS_CIPHER_AES_CCM_128
printf("TLS_CIPHER_AES_CCM_128 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128));
#endif
#ifdef TLS_CIPHER_AES_CCM_128_IV_SIZE
printf("TLS_CIPHER_AES_CCM_128_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128_IV_SIZE));
#endif
#ifdef TLS_CIPHER_AES_CCM_128_KEY_SIZE
printf("TLS_CIPHER_AES_CCM_128_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_AES_CCM_128_REC_SEQ_SIZE
printf("TLS_CIPHER_AES_CCM_128_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_AES_CCM_128_SALT_SIZE
printf("TLS_CIPHER_AES_CCM_128_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_AES_CCM_128_TAG_SIZE
printf("TLS_CIPHER_AES_CCM_128_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_CCM_128_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_128
printf("TLS_CIPHER_AES_GCM_128 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128));
#endif
#ifdef TLS_CIPHER_AES_GCM_128_IV_SIZE
printf("TLS_CIPHER_AES_GCM_128_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128_IV_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_128_KEY_SIZE
printf("TLS_CIPHER_AES_GCM_128_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_128_REC_SEQ_SIZE
printf("TLS_CIPHER_AES_GCM_128_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_128_SALT_SIZE
printf("TLS_CIPHER_AES_GCM_128_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_128_TAG_SIZE
printf("TLS_CIPHER_AES_GCM_128_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_128_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_256
printf("TLS_CIPHER_AES_GCM_256 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256));
#endif
#ifdef TLS_CIPHER_AES_GCM_256_IV_SIZE
printf("TLS_CIPHER_AES_GCM_256_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256_IV_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_256_KEY_SIZE
printf("TLS_CIPHER_AES_GCM_256_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_256_REC_SEQ_SIZE
printf("TLS_CIPHER_AES_GCM_256_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_256_SALT_SIZE
printf("TLS_CIPHER_AES_GCM_256_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_AES_GCM_256_TAG_SIZE
printf("TLS_CIPHER_AES_GCM_256_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_AES_GCM_256_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128
printf("TLS_CIPHER_ARIA_GCM_128 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128_IV_SIZE
printf("TLS_CIPHER_ARIA_GCM_128_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128_IV_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128_KEY_SIZE
printf("TLS_CIPHER_ARIA_GCM_128_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128_REC_SEQ_SIZE
printf("TLS_CIPHER_ARIA_GCM_128_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128_SALT_SIZE
printf("TLS_CIPHER_ARIA_GCM_128_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_128_TAG_SIZE
printf("TLS_CIPHER_ARIA_GCM_128_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_128_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256
printf("TLS_CIPHER_ARIA_GCM_256 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256_IV_SIZE
printf("TLS_CIPHER_ARIA_GCM_256_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256_IV_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256_KEY_SIZE
printf("TLS_CIPHER_ARIA_GCM_256_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256_REC_SEQ_SIZE
printf("TLS_CIPHER_ARIA_GCM_256_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256_SALT_SIZE
printf("TLS_CIPHER_ARIA_GCM_256_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_ARIA_GCM_256_TAG_SIZE
printf("TLS_CIPHER_ARIA_GCM_256_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_ARIA_GCM_256_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305
printf("TLS_CIPHER_CHACHA20_POLY1305 %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305_IV_SIZE
printf("TLS_CIPHER_CHACHA20_POLY1305_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305_IV_SIZE));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305_KEY_SIZE
printf("TLS_CIPHER_CHACHA20_POLY1305_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305_REC_SEQ_SIZE
printf("TLS_CIPHER_CHACHA20_POLY1305_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305_SALT_SIZE
printf("TLS_CIPHER_CHACHA20_POLY1305_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_CHACHA20_POLY1305_TAG_SIZE
printf("TLS_CIPHER_CHACHA20_POLY1305_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_CHACHA20_POLY1305_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_CCM
printf("TLS_CIPHER_SM4_CCM %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM));
#endif
#ifdef TLS_CIPHER_SM4_CCM_IV_SIZE
printf("TLS_CIPHER_SM4_CCM_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM_IV_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_CCM_KEY_SIZE
printf("TLS_CIPHER_SM4_CCM_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_CCM_REC_SEQ_SIZE
printf("TLS_CIPHER_SM4_CCM_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_CCM_SALT_SIZE
printf("TLS_CIPHER_SM4_CCM_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_CCM_TAG_SIZE
printf("TLS_CIPHER_SM4_CCM_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_CCM_TAG_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_GCM
printf("TLS_CIPHER_SM4_GCM %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM));
#endif
#ifdef TLS_CIPHER_SM4_GCM_IV_SIZE
printf("TLS_CIPHER_SM4_GCM_IV_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM_IV_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_GCM_KEY_SIZE
printf("TLS_CIPHER_SM4_GCM_KEY_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM_KEY_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_GCM_REC_SEQ_SIZE
printf("TLS_CIPHER_SM4_GCM_REC_SEQ_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM_REC_SEQ_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_GCM_SALT_SIZE
printf("TLS_CIPHER_SM4_GCM_SALT_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM_SALT_SIZE));
#endif
#ifdef TLS_CIPHER_SM4_GCM_TAG_SIZE
printf("TLS_CIPHER_SM4_GCM_TAG_SIZE %llu\n", (unsigned long long)(long long)(TLS_CIPHER_SM4_GCM_TAG_SIZE));
#endif
#ifdef TLS_CONF_BASE
printf("TLS_CONF_BASE %llu\n", (unsigned long long)(long long)(TLS_CONF_BASE));
#endif
#ifdef TLS_CONF_HW
printf("TLS_CONF_HW %llu\n", (unsigned long long)(long long)(TLS_CONF_HW));
#endif
#ifdef TLS_CONF_HW_RECORD
printf("TLS_CONF_HW_RECORD %llu\n", (unsigned long long)(long long)(TLS_CONF_HW_RECORD));
#endif
#ifdef TLS_CONF_SW
printf("TLS_CONF_SW %llu\n", (unsigned long long)(long long)(TLS_CONF_SW));
#endif
#ifdef TLS_GET_RECORD_TYPE
printf("TLS_GET_RECORD_TYPE %llu\n", (unsigned long long)(long long)(TLS_GET_RECORD_TYPE));
#endif
#ifdef TLS_INFO_CIPHER
printf("TLS_INFO_CIPHER %llu\n", (unsigned long long)(long long)(TLS_INFO_CIPHER));
#endif
#ifdef TLS_INFO_RXCONF
printf("TLS_INFO_RXCONF %llu\n", (unsigned long long)(long long)(TLS_INFO_RXCONF));
#endif
#ifdef TLS_INFO_RX_NO_PAD
printf("TLS_INFO_RX_NO_PAD %llu\n", (unsigned long long)(long long)(TLS_INFO_RX_NO_PAD));
#endif
#ifdef TLS_INFO_TXCONF
printf("TLS_INFO_TXCONF %llu\n", (unsigned long long)(long long)(TLS_INFO_TXCONF));
#endif
#ifdef TLS_INFO_UNSPEC
printf("TLS_INFO_UNSPEC %llu\n", (unsigned long long)(long long)(TLS_INFO_UNSPEC));
#endif
#ifdef TLS_INFO_VERSION
printf("TLS_INFO_VERSION %llu\n", (unsigned long long)(long long)(TLS_INFO_VERSION));
#endif
#ifdef TLS_INFO_ZC_RO_TX
printf("TLS_INFO_ZC_RO_TX %llu\n", (unsigned long long)(long long)(TLS_INFO_ZC_RO_TX));
#endif
#ifdef TLS_RX
printf("TLS_RX %llu\n", (unsigned long long)(long long)(TLS_RX));
#endif
#ifdef TLS_RX_EXPECT_NO_PAD
printf("TLS_RX_EXPECT_NO_PAD %llu\n", (unsigned long long)(long long)(TLS_RX_EXPECT_NO_PAD));
#endif
#ifdef TLS_SET_RECORD_TYPE
printf("TLS_SET_RECORD_TYPE %llu\n", (unsigned long long)(long long)(TLS_SET_RECORD_TYPE));
#endif
#ifdef TLS_TX
printf("TLS_TX %llu\n", (unsigned long long)(long long)(TLS_TX));
#endif
#ifdef TLS_TX_ZEROCOPY_RO
printf("TLS_TX_ZEROCOPY_RO %llu\n", (unsigned long long)(long long)(TLS_TX_ZEROCOPY_RO));
#endif
#ifdef TMPFS_MAGIC
printf("TMPFS_MAGIC %llu\n", (unsigned long long)(long long)(TMPFS_MAGIC));
#endif
#ifdef TMP_MAX
printf("TMP_MAX %llu\n", (unsigned long long)(long long)(TMP_MAX));
#endif
#ifdef TOSTOP
printf("TOSTOP %llu\n", (unsigned long long)(long long)(TOSTOP));
#endif
#ifdef TPACKET2_HDRLEN
printf("TPACKET2_HDRLEN %llu\n", (unsigned long long)(long long)(TPACKET2_HDRLEN));
#endif
#ifdef TPACKET3_HDRLEN
printf("TPACKET3_HDRLEN %llu\n", (unsigned long long)(long long)(TPACKET3_HDRLEN));
#endif
#ifdef TPACKET_ALIGNMENT
printf("TPACKET_ALIGNMENT %llu\n", (unsigned long long)(long long)(TPACKET_ALIGNMENT));
#endif
#ifdef TPACKET_HDRLEN
printf("TPACKET_HDRLEN %llu\n", (unsigned long long)(long long)(TPACKET_HDRLEN));
#endif
#ifdef TP_FT_REQ_FILL_RXHASH
printf("TP_FT_REQ_FILL_RXHASH %llu\n", (unsigned long long)(long long)(TP_FT_REQ_FILL_RXHASH));
#endif
#ifdef TP_STATUS_AVAILABLE
printf("TP_STATUS_AVAILABLE %llu\n", (unsigned long long)(long long)(TP_STATUS_AVAILABLE));
#endif
#ifdef TP_STATUS_BLK_TMO
printf("TP_STATUS_BLK_TMO %llu\n", (unsigned long long)(long long)(TP_STATUS_BLK_TMO));
#endif
#ifdef TP_STATUS_COPY
printf("TP_STATUS_COPY %llu\n", (unsigned long long)(long long)(TP_STATUS_COPY));
#endif
#ifdef TP_STATUS_CSUMNOTREADY
printf("TP_STATUS_CSUMNOTREADY %llu\n", (unsigned long long)(long long)(TP_STATUS_CSUMNOTREADY));
#endif
#ifdef TP_STATUS_CSUM_VALID
printf("TP_STATUS_CSUM_VALID %llu\n", (unsigned long long)(long long)(TP_STATUS_CSUM_VALID));
#endif
#ifdef TP_STATUS_KERNEL
printf("TP_STATUS_KERNEL %llu\n", (unsigned long long)(long long)(TP_STATUS_KERNEL));
#endif
#ifdef TP_STATUS_LOSING
printf("TP_STATUS_LOSING %llu\n", (unsigned long long)(long long)(TP_STATUS_LOSING));
#endif
#ifdef TP_STATUS_SENDING
printf("TP_STATUS_SENDING %llu\n", (unsigned long long)(long long)(TP_STATUS_SENDING));
#endif
#ifdef TP_STATUS_SEND_REQUEST
printf("TP_STATUS_SEND_REQUEST %llu\n", (unsigned long long)(long long)(TP_STATUS_SEND_REQUEST));
#endif
#ifdef TP_STATUS_TS_RAW_HARDWARE
printf("TP_STATUS_TS_RAW_HARDWARE %llu\n", (unsigned long long)(long long)(TP_STATUS_TS_RAW_HARDWARE));
#endif
#ifdef TP_STATUS_TS_SOFTWARE
printf("TP_STATUS_TS_SOFTWARE %llu\n", (unsigned long long)(long long)(TP_STATUS_TS_SOFTWARE));
#endif
#ifdef TP_STATUS_TS_SYS_HARDWARE
printf("TP_STATUS_TS_SYS_HARDWARE %llu\n", (unsigned long long)(long long)(TP_STATUS_TS_SYS_HARDWARE));
#endif
#ifdef TP_STATUS_USER
printf("TP_STATUS_USER %llu\n", (unsigned long long)(long long)(TP_STATUS_USER));
#endif
#ifdef TP_STATUS_VLAN_TPID_VALID
printf("TP_STATUS_VLAN_TPID_VALID %llu\n", (unsigned long long)(long long)(TP_STATUS_VLAN_TPID_VALID));
#endif
#ifdef TP_STATUS_VLAN_VALID
printf("TP_STATUS_VLAN_VALID %llu\n", (unsigned long long)(long long)(TP_STATUS_VLAN_VALID));
#endif
#ifdef TP_STATUS_WRONG_FORMAT
printf("TP_STATUS_WRONG_FORMAT %llu\n", (unsigned long long)(long long)(TP_STATUS_WRONG_FORMAT));
#endif
#ifdef TRACEFS_MAGIC
printf("TRACEFS_MAGIC %llu\n", (unsigned long long)(long long)(TRACEFS_MAGIC));
#endif
#ifdef TRAP_BRANCH
printf("TRAP_BRANCH %llu\n", (unsigned long long)(long long)(TRAP_BRANCH));
#endif
#ifdef TRAP_BRKPT
printf("TRAP_BRKPT %llu\n", (unsigned long long)(long long)(TRAP_BRKPT));
#endif
#ifdef TRAP_HWBKPT
printf("TRAP_HWBKPT %llu\n", (unsigned long long)(long long)(TRAP_HWBKPT));
#endif
#ifdef TRAP_PERF
printf("TRAP_PERF %llu\n", (unsigned long long)(long long)(TRAP_PERF));
#endif
#ifdef TRAP_TRACE
printf("TRAP_TRACE %llu\n", (unsigned long long)(long long)(TRAP_TRACE));
#endif
#ifdef TRAP_UNK
printf("TRAP_UNK %llu\n", (unsigned long long)(long long)(TRAP_UNK));
#endif
#ifdef TUNATTACHFILTER
printf("TUNATTACHFILTER %llu\n", (unsigned long long)(long long)(TUNATTACHFILTER));
#endif
#ifdef TUNDETACHFILTER
printf("TUNDETACHFILTER %llu\n", (unsigned long long)(long long)(TUNDETACHFILTER));
#endif
#ifdef TUNGETDEVNETNS
printf("TUNGETDEVNETNS %llu\n", (unsigned long long)(long long)(TUNGETDEVNETNS));
#endif
#ifdef TUNGETFEATURES
printf("TUNGETFEATURES %llu\n", (unsigned long long)(long long)(TUNGETFEATURES));
#endif
#ifdef TUNGETFILTER
printf("TUNGETFILTER %llu\n", (unsigned long long)(long long)(TUNGETFILTER));
#endif
#ifdef TUNGETIFF
printf("TUNGETIFF %llu\n", (unsigned long long)(long long)(TUNGETIFF));
#endif
#ifdef TUNGETSNDBUF
printf("TUNGETSNDBUF %llu\n", (unsigned long long)(long long)(TUNGETSNDBUF));
#endif
#ifdef TUNGETVNETBE
printf("TUNGETVNETBE %llu\n", (unsigned long long)(long long)(TUNGETVNETBE));
#endif
#ifdef TUNGETVNETHDRSZ
printf("TUNGETVNETHDRSZ %llu\n", (unsigned long long)(long long)(TUNGETVNETHDRSZ));
#endif
#ifdef TUNGETVNETLE
printf("TUNGETVNETLE %llu\n", (unsigned long long)(long long)(TUNGETVNETLE));
#endif
#ifdef TUNSETCARRIER
printf("TUNSETCARRIER %llu\n", (unsigned long long)(long long)(TUNSETCARRIER));
#endif
#ifdef TUNSETDEBUG
printf("TUNSETDEBUG %llu\n", (unsigned long long)(long long)(TUNSETDEBUG));
#endif
#ifdef TUNSETFILTEREBPF
printf("TUNSETFILTEREBPF %llu\n", (unsigned long long)(long long)(TUNSETFILTEREBPF));
#endif
#ifdef TUNSETGROUP
printf("TUNSETGROUP %llu\n", (unsigned long long)(long long)(TUNSETGROUP));
#endif
#ifdef TUNSETIFF
printf("TUNSETIFF %llu\n", (unsigned long long)(long long)(TUNSETIFF));
#endif
#ifdef TUNSETIFINDEX
printf("TUNSETIFINDEX %llu\n", (unsigned long long)(long long)(TUNSETIFINDEX));
#endif
#ifdef TUNSETLINK
printf("TUNSETLINK %llu\n", (unsigned long long)(long long)(TUNSETLINK));
#endif
#ifdef TUNSETNOCSUM
printf("TUNSETNOCSUM %llu\n", (unsigned long long)(long long)(TUNSETNOCSUM));
#endif
#ifdef TUNSETOFFLOAD
printf("TUNSETOFFLOAD %llu\n", (unsigned long long)(long long)(TUNSETOFFLOAD));
#endif
#ifdef TUNSETOWNER
printf("TUNSETOWNER %llu\n", (unsigned long long)(long long)(TUNSETOWNER));
#endif
#ifdef TUNSETPERSIST
printf("TUNSETPERSIST %llu\n", (unsigned long long)(long long)(TUNSETPERSIST));
#endif
#ifdef TUNSETQUEUE
printf("TUNSETQUEUE %llu\n", (unsigned long long)(long long)(TUNSETQUEUE));
#endif
#ifdef TUNSETSNDBUF
printf("TUNSETSNDBUF %llu\n", (unsigned long long)(long long)(TUNSETSNDBUF));
#endif
#ifdef TUNSETSTEERINGEBPF
printf("TUNSETSTEERINGEBPF %llu\n", (unsigned long long)(long long)(TUNSETSTEERINGEBPF));
#endif
#ifdef TUNSETTXFILTER
printf("TUNSETTXFILTER %llu\n", (unsigned long long)(long long)(TUNSETTXFILTER));
#endif
#ifdef TUNSETVNETBE
printf("TUNSETVNETBE %llu\n", (unsigned long long)(long long)(TUNSETVNETBE));
#endif
#ifdef TUNSETVNETHDRSZ
printf("TUNSETVNETHDRSZ %llu\n", (unsigned long long)(long long)(TUNSETVNETHDRSZ));
#endif
#ifdef TUNSETVNETLE
printf("TUNSETVNETLE %llu\n", (unsigned long long)(long long)(TUNSETVNETLE));
#endif
#ifdef TUN_FLT_ALLMULTI
printf("TUN_FLT_ALLMULTI %llu\n", (unsigned long long)(long long)(TUN_FLT_ALLMULTI));
#endif
#ifdef TUN_F_CSUM
printf("TUN_F_CSUM %llu\n", (unsigned long long)(long long)(TUN_F_CSUM));
#endif
#ifdef TUN_F_TSO4
printf("TUN_F_TSO4 %llu\n", (unsigned long long)(long long)(TUN_F_TSO4));
#endif
#ifdef TUN_F_TSO6
printf("TUN_F_TSO6 %llu\n", (unsigned long long)(long long)(TUN_F_TSO6));
#endif
#ifdef TUN_F_TSO_ECN
printf("TUN_F_TSO_ECN %llu\n", (unsigned long long)(long long)(TUN_F_TSO_ECN));
#endif
#ifdef TUN_F_UFO
printf("TUN_F_UFO %llu\n", (unsigned long long)(long long)(TUN_F_UFO));
#endif
#ifdef TUN_F_USO4
printf("TUN_F_USO4 %llu\n", (unsigned long long)(long long)(TUN_F_USO4));
#endif
#ifdef TUN_F_USO6
printf("TUN_F_USO6 %llu\n", (unsigned long long)(long long)(TUN_F_USO6));
#endif
#ifdef TUN_PKT_STRIP
printf("TUN_PKT_STRIP %llu\n", (unsigned long long)(long long)(TUN_PKT_STRIP));
#endif
#ifdef TUN_READQ_SIZE
printf("TUN_READQ_SIZE %llu\n", (unsigned long long)(long long)(TUN_READQ_SIZE));
#endif
#ifdef TUN_TAP_DEV
printf("TUN_TAP_DEV %llu\n", (unsigned long long)(long long)(TUN_TAP_DEV));
#endif
#ifdef TUN_TUN_DEV
printf("TUN_TUN_DEV %llu\n", (unsigned long long)(long long)(TUN_TUN_DEV));
#endif
#ifdef TUN_TX_TIMESTAMP
printf("TUN_TX_TIMESTAMP %llu\n", (unsigned long long)(long long)(TUN_TX_TIMESTAMP));
#endif
#ifdef TUN_TYPE_MASK
printf("TUN_TYPE_MASK %llu\n", (unsigned long long)(long long)(TUN_TYPE_MASK));
#endif
#ifdef T_FMT
printf("T_FMT %llu\n", (unsigned long long)(long long)(T_FMT));
#endif
#ifdef T_FMT_AMPM
printf("T_FMT_AMPM %llu\n", (unsigned long long)(long long)(T_FMT_AMPM));
#endif
#ifdef UDF_SUPER_MAGIC
printf("UDF_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(UDF_SUPER_MAGIC));
#endif
#ifdef UDP_CORK
printf("UDP_CORK %llu\n", (unsigned long long)(long long)(UDP_CORK));
#endif
#ifdef UDP_ENCAP
printf("UDP_ENCAP %llu\n", (unsigned long long)(long long)(UDP_ENCAP));
#endif
#ifdef UDP_GRO
printf("UDP_GRO %llu\n", (unsigned long long)(long long)(UDP_GRO));
#endif
#ifdef UDP_NO_CHECK6_RX
printf("UDP_NO_CHECK6_RX %llu\n", (unsigned long long)(long long)(UDP_NO_CHECK6_RX));
#endif
#ifdef UDP_NO_CHECK6_TX
printf("UDP_NO_CHECK6_TX %llu\n", (unsigned long long)(long long)(UDP_NO_CHECK6_TX));
#endif
#ifdef UDP_SEGMENT
printf("UDP_SEGMENT %llu\n", (unsigned long long)(long long)(UDP_SEGMENT));
#endif
#ifdef UINPUT_MAX_NAME_SIZE
printf("UINPUT_MAX_NAME_SIZE %llu\n", (unsigned long long)(long long)(UINPUT_MAX_NAME_SIZE));
#endif
#ifdef UINPUT_VERSION
printf("UINPUT_VERSION %llu\n", (unsigned long long)(long long)(UINPUT_VERSION));
#endif
#ifdef UIO_MAXIOV
printf("UIO_MAXIOV %llu\n", (unsigned long long)(long long)(UIO_MAXIOV));
#endif
#ifdef UMOUNT_NOFOLLOW
printf("UMOUNT_NOFOLLOW %llu\n", (unsigned long long)(long long)(UMOUNT_NOFOLLOW));
#endif
#ifdef UNAME26
printf("UNAME26 %llu\n", (unsigned long long)(long long)(UNAME26));
#endif
#ifdef USBDEVICE_SUPER_MAGIC
printf("USBDEVICE_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(USBDEVICE_SUPER_MAGIC));
#endif
#ifdef USER_PROCESS
printf("USER_PROCESS %llu\n", (unsigned long long)(long long)(USER_PROCESS));
#endif
#ifdef USRQUOTA
printf("USRQUOTA %llu\n", (unsigned long long)(long long)(USRQUOTA));
#endif
#ifdef UTIME_NOW
printf("UTIME_NOW %llu\n", (unsigned long long)(long long)(UTIME_NOW));
#endif
#ifdef UTIME_OMIT
printf("UTIME_OMIT %llu\n", (unsigned long long)(long long)(UTIME_OMIT));
#endif
#ifdef VDISCARD
printf("VDISCARD %llu\n", (unsigned long long)(long long)(VDISCARD));
#endif
#ifdef VEOF
printf("VEOF %llu\n", (unsigned long long)(long long)(VEOF));
#endif
#ifdef VEOL
printf("VEOL %llu\n", (unsigned long long)(long long)(VEOL));
#endif
#ifdef VEOL2
printf("VEOL2 %llu\n", (unsigned long long)(long long)(VEOL2));
#endif
#ifdef VERASE
printf("VERASE %llu\n", (unsigned long long)(long long)(VERASE));
#endif
#ifdef VINTR
printf("VINTR %llu\n", (unsigned long long)(long long)(VINTR));
#endif
#ifdef VKILL
printf("VKILL %llu\n", (unsigned long long)(long long)(VKILL));
#endif
#ifdef VLNEXT
printf("VLNEXT %llu\n", (unsigned long long)(long long)(VLNEXT));
#endif
#ifdef VMADDR_CID_ANY
printf("VMADDR_CID_ANY %llu\n", (unsigned long long)(long long)(VMADDR_CID_ANY));
#endif
#ifdef VMADDR_CID_HOST
printf("VMADDR_CID_HOST %llu\n", (unsigned long long)(long long)(VMADDR_CID_HOST));
#endif
#ifdef VMADDR_CID_HYPERVISOR
printf("VMADDR_CID_HYPERVISOR %llu\n", (unsigned long long)(long long)(VMADDR_CID_HYPERVISOR));
#endif
#ifdef VMADDR_CID_LOCAL
printf("VMADDR_CID_LOCAL %llu\n", (unsigned long long)(long long)(VMADDR_CID_LOCAL));
#endif
#ifdef VMADDR_CID_RESERVED
printf("VMADDR_CID_RESERVED %llu\n", (unsigned long long)(long long)(VMADDR_CID_RESERVED));
#endif
#ifdef VMADDR_PORT_ANY
printf("VMADDR_PORT_ANY %llu\n", (unsigned long long)(long long)(VMADDR_PORT_ANY));
#endif
#ifdef VMIN
printf("VMIN %llu\n", (unsigned long long)(long long)(VMIN));
#endif
#ifdef VM_BLOCK_DUMP
printf("VM_BLOCK_DUMP %llu\n", (unsigned long long)(long long)(VM_BLOCK_DUMP));
#endif
#ifdef VM_DIRTY_BACKGROUND
printf("VM_DIRTY_BACKGROUND %llu\n", (unsigned long long)(long long)(VM_DIRTY_BACKGROUND));
#endif
#ifdef VM_DIRTY_EXPIRE_CS
printf("VM_DIRTY_EXPIRE_CS %llu\n", (unsigned long long)(long long)(VM_DIRTY_EXPIRE_CS));
#endif
#ifdef VM_DIRTY_RATIO
printf("VM_DIRTY_RATIO %llu\n", (unsigned long long)(long long)(VM_DIRTY_RATIO));
#endif
#ifdef VM_DIRTY_WB_CS
printf("VM_DIRTY_WB_CS %llu\n", (unsigned long long)(long long)(VM_DIRTY_WB_CS));
#endif
#ifdef VM_DROP_PAGECACHE
printf("VM_DROP_PAGECACHE %llu\n", (unsigned long long)(long long)(VM_DROP_PAGECACHE));
#endif
#ifdef VM_HUGETLB_GROUP
printf("VM_HUGETLB_GROUP %llu\n", (unsigned long long)(long long)(VM_HUGETLB_GROUP));
#endif
#ifdef VM_HUGETLB_PAGES
printf("VM_HUGETLB_PAGES %llu\n", (unsigned long long)(long long)(VM_HUGETLB_PAGES));
#endif
#ifdef VM_LAPTOP_MODE
printf("VM_LAPTOP_MODE %llu\n", (unsigned long long)(long long)(VM_LAPTOP_MODE));
#endif
#ifdef VM_LEGACY_VA_LAYOUT
printf("VM_LEGACY_VA_LAYOUT %llu\n", (unsigned long long)(long long)(VM_LEGACY_VA_LAYOUT));
#endif
#ifdef VM_LOWMEM_RESERVE_RATIO
printf("VM_LOWMEM_RESERVE_RATIO %llu\n", (unsigned long long)(long long)(VM_LOWMEM_RESERVE_RATIO));
#endif
#ifdef VM_MAX_MAP_COUNT
printf("VM_MAX_MAP_COUNT %llu\n", (unsigned long long)(long long)(VM_MAX_MAP_COUNT));
#endif
#ifdef VM_MIN_FREE_KBYTES
printf("VM_MIN_FREE_KBYTES %llu\n", (unsigned long long)(long long)(VM_MIN_FREE_KBYTES));
#endif
#ifdef VM_MIN_SLAB
printf("VM_MIN_SLAB %llu\n", (unsigned long long)(long long)(VM_MIN_SLAB));
#endif
#ifdef VM_MIN_UNMAPPED
printf("VM_MIN_UNMAPPED %llu\n", (unsigned long long)(long long)(VM_MIN_UNMAPPED));
#endif
#ifdef VM_NR_PDFLUSH_THREADS
printf("VM_NR_PDFLUSH_THREADS %llu\n", (unsigned long long)(long long)(VM_NR_PDFLUSH_THREADS));
#endif
#ifdef VM_OVERCOMMIT_MEMORY
printf("VM_OVERCOMMIT_MEMORY %llu\n", (unsigned long long)(long long)(VM_OVERCOMMIT_MEMORY));
#endif
#ifdef VM_OVERCOMMIT_RATIO
printf("VM_OVERCOMMIT_RATIO %llu\n", (unsigned long long)(long long)(VM_OVERCOMMIT_RATIO));
#endif
#ifdef VM_PAGEBUF
printf("VM_PAGEBUF %llu\n", (unsigned long long)(long long)(VM_PAGEBUF));
#endif
#ifdef VM_PAGE_CLUSTER
printf("VM_PAGE_CLUSTER %llu\n", (unsigned long long)(long long)(VM_PAGE_CLUSTER));
#endif
#ifdef VM_PANIC_ON_OOM
printf("VM_PANIC_ON_OOM %llu\n", (unsigned long long)(long long)(VM_PANIC_ON_OOM));
#endif
#ifdef VM_PERCPU_PAGELIST_FRACTION
printf("VM_PERCPU_PAGELIST_FRACTION %llu\n", (unsigned long long)(long long)(VM_PERCPU_PAGELIST_FRACTION));
#endif
#ifdef VM_SWAPPINESS
printf("VM_SWAPPINESS %llu\n", (unsigned long long)(long long)(VM_SWAPPINESS));
#endif
#ifdef VM_SWAP_TOKEN_TIMEOUT
printf("VM_SWAP_TOKEN_TIMEOUT %llu\n", (unsigned long long)(long long)(VM_SWAP_TOKEN_TIMEOUT));
#endif
#ifdef VM_VDSO_ENABLED
printf("VM_VDSO_ENABLED %llu\n", (unsigned long long)(long long)(VM_VDSO_ENABLED));
#endif
#ifdef VM_VFS_CACHE_PRESSURE
printf("VM_VFS_CACHE_PRESSURE %llu\n", (unsigned long long)(long long)(VM_VFS_CACHE_PRESSURE));
#endif
#ifdef VM_ZONE_RECLAIM_MODE
printf("VM_ZONE_RECLAIM_MODE %llu\n", (unsigned long long)(long long)(VM_ZONE_RECLAIM_MODE));
#endif
#ifdef VQUIT
printf("VQUIT %llu\n", (unsigned long long)(long long)(VQUIT));
#endif
#ifdef VREPRINT
printf("VREPRINT %llu\n", (unsigned long long)(long long)(VREPRINT));
#endif
#ifdef VSTART
printf("VSTART %llu\n", (unsigned long long)(long long)(VSTART));
#endif
#ifdef VSTOP
printf("VSTOP %llu\n", (unsigned long long)(long long)(VSTOP));
#endif
#ifdef VSUSP
printf("VSUSP %llu\n", (unsigned long long)(long long)(VSUSP));
#endif
#ifdef VSWTC
printf("VSWTC %llu\n", (unsigned long long)(long long)(VSWTC));
#endif
#ifdef VT0
printf("VT0 %llu\n", (unsigned long long)(long long)(VT0));
#endif
#ifdef VT1
printf("VT1 %llu\n", (unsigned long long)(long long)(VT1));
#endif
#ifdef VTDLY
printf("VTDLY %llu\n", (unsigned long long)(long long)(VTDLY));
#endif
#ifdef VTIME
printf("VTIME %llu\n", (unsigned long long)(long long)(VTIME));
#endif
#ifdef VWERASE
printf("VWERASE %llu\n", (unsigned long long)(long long)(VWERASE));
#endif
#ifdef WCONTINUED
printf("WCONTINUED %llu\n", (unsigned long long)(long long)(WCONTINUED));
#endif
#ifdef WEXITED
printf("WEXITED %llu\n", (unsigned long long)(long long)(WEXITED));
#endif
#ifdef WHOLE_SECONDS
printf("WHOLE_SECONDS %llu\n", (unsigned long long)(long long)(WHOLE_SECONDS));
#endif
#ifdef WIRELESS_EXT
printf("WIRELESS_EXT %llu\n", (unsigned long long)(long long)(WIRELESS_EXT));
#endif
#ifdef WNOHANG
printf("WNOHANG %llu\n", (unsigned long long)(long long)(WNOHANG));
#endif
#ifdef WNOWAIT
printf("WNOWAIT %llu\n", (unsigned long long)(long long)(WNOWAIT));
#endif
#ifdef WSTOPPED
printf("WSTOPPED %llu\n", (unsigned long long)(long long)(WSTOPPED));
#endif
#ifdef WUNTRACED
printf("WUNTRACED %llu\n", (unsigned long long)(long long)(WUNTRACED));
#endif
#ifdef W_OK
printf("W_OK %llu\n", (unsigned long long)(long long)(W_OK));
#endif
#ifdef XATTR_CREATE
printf("XATTR_CREATE %llu\n", (unsigned long long)(long long)(XATTR_CREATE));
#endif
#ifdef XATTR_REPLACE
printf("XATTR_REPLACE %llu\n", (unsigned long long)(long long)(XATTR_REPLACE));
#endif
#ifdef XDP_COPY
printf("XDP_COPY %llu\n", (unsigned long long)(long long)(XDP_COPY));
#endif
#ifdef XDP_MMAP_OFFSETS
printf("XDP_MMAP_OFFSETS %llu\n", (unsigned long long)(long long)(XDP_MMAP_OFFSETS));
#endif
#ifdef XDP_OPTIONS
printf("XDP_OPTIONS %llu\n", (unsigned long long)(long long)(XDP_OPTIONS));
#endif
#ifdef XDP_OPTIONS_ZEROCOPY
printf("XDP_OPTIONS_ZEROCOPY %llu\n", (unsigned long long)(long long)(XDP_OPTIONS_ZEROCOPY));
#endif
#ifdef XDP_PGOFF_RX_RING
printf("XDP_PGOFF_RX_RING %llu\n", (unsigned long long)(long long)(XDP_PGOFF_RX_RING));
#endif
#ifdef XDP_PGOFF_TX_RING
printf("XDP_PGOFF_TX_RING %llu\n", (unsigned long long)(long long)(XDP_PGOFF_TX_RING));
#endif
#ifdef XDP_PKT_CONTD
printf("XDP_PKT_CONTD %llu\n", (unsigned long long)(long long)(XDP_PKT_CONTD));
#endif
#ifdef XDP_RING_NEED_WAKEUP
printf("XDP_RING_NEED_WAKEUP %llu\n", (unsigned long long)(long long)(XDP_RING_NEED_WAKEUP));
#endif
#ifdef XDP_RX_RING
printf("XDP_RX_RING %llu\n", (unsigned long long)(long long)(XDP_RX_RING));
#endif
#ifdef XDP_SHARED_UMEM
printf("XDP_SHARED_UMEM %llu\n", (unsigned long long)(long long)(XDP_SHARED_UMEM));
#endif
#ifdef XDP_STATISTICS
printf("XDP_STATISTICS %llu\n", (unsigned long long)(long long)(XDP_STATISTICS));
#endif
#ifdef XDP_TXMD_FLAGS_CHECKSUM
printf("XDP_TXMD_FLAGS_CHECKSUM %llu\n", (unsigned long long)(long long)(XDP_TXMD_FLAGS_CHECKSUM));
#endif
#ifdef XDP_TXMD_FLAGS_TIMESTAMP
printf("XDP_TXMD_FLAGS_TIMESTAMP %llu\n", (unsigned long long)(long long)(XDP_TXMD_FLAGS_TIMESTAMP));
#endif
#ifdef XDP_TX_METADATA
printf("XDP_TX_METADATA %llu\n", (unsigned long long)(long long)(XDP_TX_METADATA));
#endif
#ifdef XDP_TX_RING
printf("XDP_TX_RING %llu\n", (unsigned long long)(long long)(XDP_TX_RING));
#endif
#ifdef XDP_UMEM_COMPLETION_RING
printf("XDP_UMEM_COMPLETION_RING %llu\n", (unsigned long long)(long long)(XDP_UMEM_COMPLETION_RING));
#endif
#ifdef XDP_UMEM_FILL_RING
printf("XDP_UMEM_FILL_RING %llu\n", (unsigned long long)(long long)(XDP_UMEM_FILL_RING));
#endif
#ifdef XDP_UMEM_PGOFF_COMPLETION_RING
printf("XDP_UMEM_PGOFF_COMPLETION_RING %llu\n", (unsigned long long)(long long)(XDP_UMEM_PGOFF_COMPLETION_RING));
#endif
#ifdef XDP_UMEM_PGOFF_FILL_RING
printf("XDP_UMEM_PGOFF_FILL_RING %llu\n", (unsigned long long)(long long)(XDP_UMEM_PGOFF_FILL_RING));
#endif
#ifdef XDP_UMEM_REG
printf("XDP_UMEM_REG %llu\n", (unsigned long long)(long long)(XDP_UMEM_REG));
#endif
#ifdef XDP_UMEM_TX_METADATA_LEN
printf("XDP_UMEM_TX_METADATA_LEN %llu\n", (unsigned long long)(long long)(XDP_UMEM_TX_METADATA_LEN));
#endif
#ifdef XDP_UMEM_TX_SW_CSUM
printf("XDP_UMEM_TX_SW_CSUM %llu\n", (unsigned long long)(long long)(XDP_UMEM_TX_SW_CSUM));
#endif
#ifdef XDP_UMEM_UNALIGNED_CHUNK_FLAG
printf("XDP_UMEM_UNALIGNED_CHUNK_FLAG %llu\n", (unsigned long long)(long long)(XDP_UMEM_UNALIGNED_CHUNK_FLAG));
#endif
#ifdef XDP_USE_NEED_WAKEUP
printf("XDP_USE_NEED_WAKEUP %llu\n", (unsigned long long)(long long)(XDP_USE_NEED_WAKEUP));
#endif
#ifdef XDP_USE_SG
printf("XDP_USE_SG %llu\n", (unsigned long long)(long long)(XDP_USE_SG));
#endif
#ifdef XDP_ZEROCOPY
printf("XDP_ZEROCOPY %llu\n", (unsigned long long)(long long)(XDP_ZEROCOPY));
#endif
#ifdef XENFS_SUPER_MAGIC
printf("XENFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(XENFS_SUPER_MAGIC));
#endif
#ifdef XFS_SUPER_MAGIC
printf("XFS_SUPER_MAGIC %llu\n", (unsigned long long)(long long)(XFS_SUPER_MAGIC));
#endif
#ifdef XSK_UNALIGNED_BUF_ADDR_MASK
printf("XSK_UNALIGNED_BUF_ADDR_MASK %llu\n", (unsigned long long)(long long)(XSK_UNALIGNED_BUF_ADDR_MASK));
#endif
#ifdef XSK_UNALIGNED_BUF_OFFSET_SHIFT
printf("XSK_UNALIGNED_BUF_OFFSET_SHIFT %llu\n", (unsigned long long)(long long)(XSK_UNALIGNED_BUF_OFFSET_SHIFT));
#endif
#ifdef XTABS
printf("XTABS %llu\n", (unsigned long long)(long long)(XTABS));
#endif
#ifdef X_OK
printf("X_OK %llu\n", (unsigned long long)(long long)(X_OK));
#endif
#ifdef YESEXPR
printf("YESEXPR %llu\n", (unsigned long long)(long long)(YESEXPR));
#endif
#ifdef YESSTR
printf("YESSTR %llu\n", (unsigned long long)(long long)(YESSTR));
#endif
#ifdef _CS_GNU_LIBC_VERSION
printf("_CS_GNU_LIBC_VERSION %llu\n", (unsigned long long)(long long)(_CS_GNU_LIBC_VERSION));
#endif
#ifdef _CS_GNU_LIBPTHREAD_VERSION
printf("_CS_GNU_LIBPTHREAD_VERSION %llu\n", (unsigned long long)(long long)(_CS_GNU_LIBPTHREAD_VERSION));
#endif
#ifdef _CS_PATH
printf("_CS_PATH %llu\n", (unsigned long long)(long long)(_CS_PATH));
#endif
#ifdef _CS_POSIX_V5_WIDTH_RESTRICTED_ENVS
printf("_CS_POSIX_V5_WIDTH_RESTRICTED_ENVS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V5_WIDTH_RESTRICTED_ENVS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFF32_CFLAGS
printf("_CS_POSIX_V6_ILP32_OFF32_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFF32_CFLAGS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFF32_LDFLAGS
printf("_CS_POSIX_V6_ILP32_OFF32_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFF32_LDFLAGS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFF32_LIBS
printf("_CS_POSIX_V6_ILP32_OFF32_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFF32_LIBS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS
printf("_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS
printf("_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS
printf("_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFFBIG_LIBS
printf("_CS_POSIX_V6_ILP32_OFFBIG_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFFBIG_LIBS));
#endif
#ifdef _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS
printf("_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V6_LP64_OFF64_CFLAGS
printf("_CS_POSIX_V6_LP64_OFF64_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LP64_OFF64_CFLAGS));
#endif
#ifdef _CS_POSIX_V6_LP64_OFF64_LDFLAGS
printf("_CS_POSIX_V6_LP64_OFF64_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LP64_OFF64_LDFLAGS));
#endif
#ifdef _CS_POSIX_V6_LP64_OFF64_LIBS
printf("_CS_POSIX_V6_LP64_OFF64_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LP64_OFF64_LIBS));
#endif
#ifdef _CS_POSIX_V6_LP64_OFF64_LINTFLAGS
printf("_CS_POSIX_V6_LP64_OFF64_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LP64_OFF64_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS
printf("_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS));
#endif
#ifdef _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS
printf("_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS));
#endif
#ifdef _CS_POSIX_V6_LPBIG_OFFBIG_LIBS
printf("_CS_POSIX_V6_LPBIG_OFFBIG_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LPBIG_OFFBIG_LIBS));
#endif
#ifdef _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS
printf("_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V6_WIDTH_RESTRICTED_ENVS
printf("_CS_POSIX_V6_WIDTH_RESTRICTED_ENVS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V6_WIDTH_RESTRICTED_ENVS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFF32_CFLAGS
printf("_CS_POSIX_V7_ILP32_OFF32_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFF32_CFLAGS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFF32_LDFLAGS
printf("_CS_POSIX_V7_ILP32_OFF32_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFF32_LDFLAGS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFF32_LIBS
printf("_CS_POSIX_V7_ILP32_OFF32_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFF32_LIBS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS
printf("_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS
printf("_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS
printf("_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFFBIG_LIBS
printf("_CS_POSIX_V7_ILP32_OFFBIG_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFFBIG_LIBS));
#endif
#ifdef _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS
printf("_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V7_LP64_OFF64_CFLAGS
printf("_CS_POSIX_V7_LP64_OFF64_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LP64_OFF64_CFLAGS));
#endif
#ifdef _CS_POSIX_V7_LP64_OFF64_LDFLAGS
printf("_CS_POSIX_V7_LP64_OFF64_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LP64_OFF64_LDFLAGS));
#endif
#ifdef _CS_POSIX_V7_LP64_OFF64_LIBS
printf("_CS_POSIX_V7_LP64_OFF64_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LP64_OFF64_LIBS));
#endif
#ifdef _CS_POSIX_V7_LP64_OFF64_LINTFLAGS
printf("_CS_POSIX_V7_LP64_OFF64_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LP64_OFF64_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS
printf("_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS));
#endif
#ifdef _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS
printf("_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS));
#endif
#ifdef _CS_POSIX_V7_LPBIG_OFFBIG_LIBS
printf("_CS_POSIX_V7_LPBIG_OFFBIG_LIBS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LPBIG_OFFBIG_LIBS));
#endif
#ifdef _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS
printf("_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS));
#endif
#ifdef _CS_POSIX_V7_WIDTH_RESTRICTED_ENVS
printf("_CS_POSIX_V7_WIDTH_RESTRICTED_ENVS %llu\n", (unsigned long long)(long long)(_CS_POSIX_V7_WIDTH_RESTRICTED_ENVS));
#endif
#ifdef _CS_V6_ENV
printf("_CS_V6_ENV %llu\n", (unsigned long long)(long long)(_CS_V6_ENV));
#endif
#ifdef _CS_V7_ENV
printf("_CS_V7_ENV %llu\n", (unsigned long long)(long long)(_CS_V7_ENV));
#endif
#ifdef _IOFBF
printf("_IOFBF %llu\n", (unsigned long long)(long long)(_IOFBF));
#endif
#ifdef _IOLBF
printf("_IOLBF %llu\n", (unsigned long long)(long long)(_IOLBF));
#endif
#ifdef _IONBF
printf("_IONBF %llu\n", (unsigned long long)(long long)(_IONBF));
#endif
#ifdef _PC_2_SYMLINKS
printf("_PC_2_SYMLINKS %llu\n", (unsigned long long)(long long)(_PC_2_SYMLINKS));
#endif
#ifdef _PC_ALLOC_SIZE_MIN
printf("_PC_ALLOC_SIZE_MIN %llu\n", (unsigned long long)(long long)(_PC_ALLOC_SIZE_MIN));
#endif
#ifdef _PC_ASYNC_IO
printf("_PC_ASYNC_IO %llu\n", (unsigned long long)(long long)(_PC_ASYNC_IO));
#endif
#ifdef _PC_CHOWN_RESTRICTED
printf("_PC_CHOWN_RESTRICTED %llu\n", (unsigned long long)(long long)(_PC_CHOWN_RESTRICTED));
#endif
#ifdef _PC_FILESIZEBITS
printf("_PC_FILESIZEBITS %llu\n", (unsigned long long)(long long)(_PC_FILESIZEBITS));
#endif
#ifdef _PC_LINK_MAX
printf("_PC_LINK_MAX %llu\n", (unsigned long long)(long long)(_PC_LINK_MAX));
#endif
#ifdef _PC_MAX_CANON
printf("_PC_MAX_CANON %llu\n", (unsigned long long)(long long)(_PC_MAX_CANON));
#endif
#ifdef _PC_MAX_INPUT
printf("_PC_MAX_INPUT %llu\n", (unsigned long long)(long long)(_PC_MAX_INPUT));
#endif
#ifdef _PC_NAME_MAX
printf("_PC_NAME_MAX %llu\n", (unsigned long long)(long long)(_PC_NAME_MAX));
#endif
#ifdef _PC_NO_TRUNC
printf("_PC_NO_TRUNC %llu\n", (unsigned long long)(long long)(_PC_NO_TRUNC));
#endif
#ifdef _PC_PATH_MAX
printf("_PC_PATH_MAX %llu\n", (unsigned long long)(long long)(_PC_PATH_MAX));
#endif
#ifdef _PC_PIPE_BUF
printf("_PC_PIPE_BUF %llu\n", (unsigned long long)(long long)(_PC_PIPE_BUF));
#endif
#ifdef _PC_PRIO_IO
printf("_PC_PRIO_IO %llu\n", (unsigned long long)(long long)(_PC_PRIO_IO));
#endif
#ifdef _PC_REC_INCR_XFER_SIZE
printf("_PC_REC_INCR_XFER_SIZE %llu\n", (unsigned long long)(long long)(_PC_REC_INCR_XFER_SIZE));
#endif
#ifdef _PC_REC_MAX_XFER_SIZE
printf("_PC_REC_MAX_XFER_SIZE %llu\n", (unsigned long long)(long long)(_PC_REC_MAX_XFER_SIZE));
#endif
#ifdef _PC_REC_MIN_XFER_SIZE
printf("_PC_REC_MIN_XFER_SIZE %llu\n", (unsigned long long)(long long)(_PC_REC_MIN_XFER_SIZE));
#endif
#ifdef _PC_REC_XFER_ALIGN
printf("_PC_REC_XFER_ALIGN %llu\n", (unsigned long long)(long long)(_PC_REC_XFER_ALIGN));
#endif
#ifdef _PC_SOCK_MAXBUF
printf("_PC_SOCK_MAXBUF %llu\n", (unsigned long long)(long long)(_PC_SOCK_MAXBUF));
#endif
#ifdef _PC_SYMLINK_MAX
printf("_PC_SYMLINK_MAX %llu\n", (unsigned long long)(long long)(_PC_SYMLINK_MAX));
#endif
#ifdef _PC_SYNC_IO
printf("_PC_SYNC_IO %llu\n", (unsigned long long)(long long)(_PC_SYNC_IO));
#endif
#ifdef _PC_VDISABLE
printf("_PC_VDISABLE %llu\n", (unsigned long long)(long long)(_PC_VDISABLE));
#endif
#ifdef _POSIX_VDISABLE
printf("_POSIX_VDISABLE %llu\n", (unsigned long long)(long long)(_POSIX_VDISABLE));
#endif
#ifdef _SC_2_CHAR_TERM
printf("_SC_2_CHAR_TERM %llu\n", (unsigned long long)(long long)(_SC_2_CHAR_TERM));
#endif
#ifdef _SC_2_C_BIND
printf("_SC_2_C_BIND %llu\n", (unsigned long long)(long long)(_SC_2_C_BIND));
#endif
#ifdef _SC_2_C_DEV
printf("_SC_2_C_DEV %llu\n", (unsigned long long)(long long)(_SC_2_C_DEV));
#endif
#ifdef _SC_2_C_VERSION
printf("_SC_2_C_VERSION %llu\n", (unsigned long long)(long long)(_SC_2_C_VERSION));
#endif
#ifdef _SC_2_FORT_DEV
printf("_SC_2_FORT_DEV %llu\n", (unsigned long long)(long long)(_SC_2_FORT_DEV));
#endif
#ifdef _SC_2_FORT_RUN
printf("_SC_2_FORT_RUN %llu\n", (unsigned long long)(long long)(_SC_2_FORT_RUN));
#endif
#ifdef _SC_2_LOCALEDEF
printf("_SC_2_LOCALEDEF %llu\n", (unsigned long long)(long long)(_SC_2_LOCALEDEF));
#endif
#ifdef _SC_2_PBS
printf("_SC_2_PBS %llu\n", (unsigned long long)(long long)(_SC_2_PBS));
#endif
#ifdef _SC_2_PBS_ACCOUNTING
printf("_SC_2_PBS_ACCOUNTING %llu\n", (unsigned long long)(long long)(_SC_2_PBS_ACCOUNTING));
#endif
#ifdef _SC_2_PBS_CHECKPOINT
printf("_SC_2_PBS_CHECKPOINT %llu\n", (unsigned long long)(long long)(_SC_2_PBS_CHECKPOINT));
#endif
#ifdef _SC_2_PBS_LOCATE
printf("_SC_2_PBS_LOCATE %llu\n", (unsigned long long)(long long)(_SC_2_PBS_LOCATE));
#endif
#ifdef _SC_2_PBS_MESSAGE
printf("_SC_2_PBS_MESSAGE %llu\n", (unsigned long long)(long long)(_SC_2_PBS_MESSAGE));
#endif
#ifdef _SC_2_PBS_TRACK
printf("_SC_2_PBS_TRACK %llu\n", (unsigned long long)(long long)(_SC_2_PBS_TRACK));
#endif
#ifdef _SC_2_SW_DEV
printf("_SC_2_SW_DEV %llu\n", (unsigned long long)(long long)(_SC_2_SW_DEV));
#endif
#ifdef _SC_2_UPE
printf("_SC_2_UPE %llu\n", (unsigned long long)(long long)(_SC_2_UPE));
#endif
#ifdef _SC_2_VERSION
printf("_SC_2_VERSION %llu\n", (unsigned long long)(long long)(_SC_2_VERSION));
#endif
#ifdef _SC_ADVISORY_INFO
printf("_SC_ADVISORY_INFO %llu\n", (unsigned long long)(long long)(_SC_ADVISORY_INFO));
#endif
#ifdef _SC_AIO_LISTIO_MAX
printf("_SC_AIO_LISTIO_MAX %llu\n", (unsigned long long)(long long)(_SC_AIO_LISTIO_MAX));
#endif
#ifdef _SC_AIO_MAX
printf("_SC_AIO_MAX %llu\n", (unsigned long long)(long long)(_SC_AIO_MAX));
#endif
#ifdef _SC_AIO_PRIO_DELTA_MAX
printf("_SC_AIO_PRIO_DELTA_MAX %llu\n", (unsigned long long)(long long)(_SC_AIO_PRIO_DELTA_MAX));
#endif
#ifdef _SC_ARG_MAX
printf("_SC_ARG_MAX %llu\n", (unsigned long long)(long long)(_SC_ARG_MAX));
#endif
#ifdef _SC_ASYNCHRONOUS_IO
printf("_SC_ASYNCHRONOUS_IO %llu\n", (unsigned long long)(long long)(_SC_ASYNCHRONOUS_IO));
#endif
#ifdef _SC_ATEXIT_MAX
printf("_SC_ATEXIT_MAX %llu\n", (unsigned long long)(long long)(_SC_ATEXIT_MAX));
#endif
#ifdef _SC_AVPHYS_PAGES
printf("_SC_AVPHYS_PAGES %llu\n", (unsigned long long)(long long)(_SC_AVPHYS_PAGES));
#endif
#ifdef _SC_BARRIERS
printf("_SC_BARRIERS %llu\n", (unsigned long long)(long long)(_SC_BARRIERS));
#endif
#ifdef _SC_BASE
printf("_SC_BASE %llu\n", (unsigned long long)(long long)(_SC_BASE));
#endif
#ifdef _SC_BC_BASE_MAX
printf("_SC_BC_BASE_MAX %llu\n", (unsigned long long)(long long)(_SC_BC_BASE_MAX));
#endif
#ifdef _SC_BC_DIM_MAX
printf("_SC_BC_DIM_MAX %llu\n", (unsigned long long)(long long)(_SC_BC_DIM_MAX));
#endif
#ifdef _SC_BC_SCALE_MAX
printf("_SC_BC_SCALE_MAX %llu\n", (unsigned long long)(long long)(_SC_BC_SCALE_MAX));
#endif
#ifdef _SC_BC_STRING_MAX
printf("_SC_BC_STRING_MAX %llu\n", (unsigned long long)(long long)(_SC_BC_STRING_MAX));
#endif
#ifdef _SC_CHARCLASS_NAME_MAX
printf("_SC_CHARCLASS_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_CHARCLASS_NAME_MAX));
#endif
#ifdef _SC_CHAR_BIT
printf("_SC_CHAR_BIT %llu\n", (unsigned long long)(long long)(_SC_CHAR_BIT));
#endif
#ifdef _SC_CHAR_MAX
printf("_SC_CHAR_MAX %llu\n", (unsigned long long)(long long)(_SC_CHAR_MAX));
#endif
#ifdef _SC_CHAR_MIN
printf("_SC_CHAR_MIN %llu\n", (unsigned long long)(long long)(_SC_CHAR_MIN));
#endif
#ifdef _SC_CHILD_MAX
printf("_SC_CHILD_MAX %llu\n", (unsigned long long)(long long)(_SC_CHILD_MAX));
#endif
#ifdef _SC_CLK_TCK
printf("_SC_CLK_TCK %llu\n", (unsigned long long)(long long)(_SC_CLK_TCK));
#endif
#ifdef _SC_CLOCK_SELECTION
printf("_SC_CLOCK_SELECTION %llu\n", (unsigned long long)(long long)(_SC_CLOCK_SELECTION));
#endif
#ifdef _SC_COLL_WEIGHTS_MAX
printf("_SC_COLL_WEIGHTS_MAX %llu\n", (unsigned long long)(long long)(_SC_COLL_WEIGHTS_MAX));
#endif
#ifdef _SC_CPUTIME
printf("_SC_CPUTIME %llu\n", (unsigned long long)(long long)(_SC_CPUTIME));
#endif
#ifdef _SC_C_LANG_SUPPORT
printf("_SC_C_LANG_SUPPORT %llu\n", (unsigned long long)(long long)(_SC_C_LANG_SUPPORT));
#endif
#ifdef _SC_C_LANG_SUPPORT_R
printf("_SC_C_LANG_SUPPORT_R %llu\n", (unsigned long long)(long long)(_SC_C_LANG_SUPPORT_R));
#endif
#ifdef _SC_DELAYTIMER_MAX
printf("_SC_DELAYTIMER_MAX %llu\n", (unsigned long long)(long long)(_SC_DELAYTIMER_MAX));
#endif
#ifdef _SC_DEVICE_IO
printf("_SC_DEVICE_IO %llu\n", (unsigned long long)(long long)(_SC_DEVICE_IO));
#endif
#ifdef _SC_DEVICE_SPECIFIC
printf("_SC_DEVICE_SPECIFIC %llu\n", (unsigned long long)(long long)(_SC_DEVICE_SPECIFIC));
#endif
#ifdef _SC_DEVICE_SPECIFIC_R
printf("_SC_DEVICE_SPECIFIC_R %llu\n", (unsigned long long)(long long)(_SC_DEVICE_SPECIFIC_R));
#endif
#ifdef _SC_EQUIV_CLASS_MAX
printf("_SC_EQUIV_CLASS_MAX %llu\n", (unsigned long long)(long long)(_SC_EQUIV_CLASS_MAX));
#endif
#ifdef _SC_EXPR_NEST_MAX
printf("_SC_EXPR_NEST_MAX %llu\n", (unsigned long long)(long long)(_SC_EXPR_NEST_MAX));
#endif
#ifdef _SC_FD_MGMT
printf("_SC_FD_MGMT %llu\n", (unsigned long long)(long long)(_SC_FD_MGMT));
#endif
#ifdef _SC_FIFO
printf("_SC_FIFO %llu\n", (unsigned long long)(long long)(_SC_FIFO));
#endif
#ifdef _SC_FILE_ATTRIBUTES
printf("_SC_FILE_ATTRIBUTES %llu\n", (unsigned long long)(long long)(_SC_FILE_ATTRIBUTES));
#endif
#ifdef _SC_FILE_LOCKING
printf("_SC_FILE_LOCKING %llu\n", (unsigned long long)(long long)(_SC_FILE_LOCKING));
#endif
#ifdef _SC_FILE_SYSTEM
printf("_SC_FILE_SYSTEM %llu\n", (unsigned long long)(long long)(_SC_FILE_SYSTEM));
#endif
#ifdef _SC_FSYNC
printf("_SC_FSYNC %llu\n", (unsigned long long)(long long)(_SC_FSYNC));
#endif
#ifdef _SC_GETGR_R_SIZE_MAX
printf("_SC_GETGR_R_SIZE_MAX %llu\n", (unsigned long long)(long long)(_SC_GETGR_R_SIZE_MAX));
#endif
#ifdef _SC_GETPW_R_SIZE_MAX
printf("_SC_GETPW_R_SIZE_MAX %llu\n", (unsigned long long)(long long)(_SC_GETPW_R_SIZE_MAX));
#endif
#ifdef _SC_HOST_NAME_MAX
printf("_SC_HOST_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_HOST_NAME_MAX));
#endif
#ifdef _SC_INT_MAX
printf("_SC_INT_MAX %llu\n", (unsigned long long)(long long)(_SC_INT_MAX));
#endif
#ifdef _SC_INT_MIN
printf("_SC_INT_MIN %llu\n", (unsigned long long)(long long)(_SC_INT_MIN));
#endif
#ifdef _SC_IOV_MAX
printf("_SC_IOV_MAX %llu\n", (unsigned long long)(long long)(_SC_IOV_MAX));
#endif
#ifdef _SC_IPV6
printf("_SC_IPV6 %llu\n", (unsigned long long)(long long)(_SC_IPV6));
#endif
#ifdef _SC_JOB_CONTROL
printf("_SC_JOB_CONTROL %llu\n", (unsigned long long)(long long)(_SC_JOB_CONTROL));
#endif
#ifdef _SC_LEVEL1_DCACHE_ASSOC
printf("_SC_LEVEL1_DCACHE_ASSOC %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_DCACHE_ASSOC));
#endif
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
printf("_SC_LEVEL1_DCACHE_LINESIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_DCACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL1_DCACHE_SIZE
printf("_SC_LEVEL1_DCACHE_SIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_DCACHE_SIZE));
#endif
#ifdef _SC_LEVEL1_ICACHE_ASSOC
printf("_SC_LEVEL1_ICACHE_ASSOC %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_ICACHE_ASSOC));
#endif
#ifdef _SC_LEVEL1_ICACHE_LINESIZE
printf("_SC_LEVEL1_ICACHE_LINESIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_ICACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL1_ICACHE_SIZE
printf("_SC_LEVEL1_ICACHE_SIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL1_ICACHE_SIZE));
#endif
#ifdef _SC_LEVEL2_CACHE_ASSOC
printf("_SC_LEVEL2_CACHE_ASSOC %llu\n", (unsigned long long)(long long)(_SC_LEVEL2_CACHE_ASSOC));
#endif
#ifdef _SC_LEVEL2_CACHE_LINESIZE
printf("_SC_LEVEL2_CACHE_LINESIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL2_CACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
printf("_SC_LEVEL2_CACHE_SIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL2_CACHE_SIZE));
#endif
#ifdef _SC_LEVEL3_CACHE_ASSOC
printf("_SC_LEVEL3_CACHE_ASSOC %llu\n", (unsigned long long)(long long)(_SC_LEVEL3_CACHE_ASSOC));
#endif
#ifdef _SC_LEVEL3_CACHE_LINESIZE
printf("_SC_LEVEL3_CACHE_LINESIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL3_CACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
printf("_SC_LEVEL3_CACHE_SIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL3_CACHE_SIZE));
#endif
#ifdef _SC_LEVEL4_CACHE_ASSOC
printf("_SC_LEVEL4_CACHE_ASSOC %llu\n", (unsigned long long)(long long)(_SC_LEVEL4_CACHE_ASSOC));
#endif
#ifdef _SC_LEVEL4_CACHE_LINESIZE
printf("_SC_LEVEL4_CACHE_LINESIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL4_CACHE_LINESIZE));
#endif
#ifdef _SC_LEVEL4_CACHE_SIZE
printf("_SC_LEVEL4_CACHE_SIZE %llu\n", (unsigned long long)(long long)(_SC_LEVEL4_CACHE_SIZE));
#endif
#ifdef _SC_LINE_MAX
printf("_SC_LINE_MAX %llu\n", (unsigned long long)(long long)(_SC_LINE_MAX));
#endif
#ifdef _SC_LOGIN_NAME_MAX
printf("_SC_LOGIN_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_LOGIN_NAME_MAX));
#endif
#ifdef _SC_LONG_BIT
printf("_SC_LONG_BIT %llu\n", (unsigned long long)(long long)(_SC_LONG_BIT));
#endif
#ifdef _SC_MAPPED_FILES
printf("_SC_MAPPED_FILES %llu\n", (unsigned long long)(long long)(_SC_MAPPED_FILES));
#endif
#ifdef _SC_MB_LEN_MAX
printf("_SC_MB_LEN_MAX %llu\n", (unsigned long long)(long long)(_SC_MB_LEN_MAX));
#endif
#ifdef _SC_MEMLOCK
printf("_SC_MEMLOCK %llu\n", (unsigned long long)(long long)(_SC_MEMLOCK));
#endif
#ifdef _SC_MEMLOCK_RANGE
printf("_SC_MEMLOCK_RANGE %llu\n", (unsigned long long)(long long)(_SC_MEMLOCK_RANGE));
#endif
#ifdef _SC_MEMORY_PROTECTION
printf("_SC_MEMORY_PROTECTION %llu\n", (unsigned long long)(long long)(_SC_MEMORY_PROTECTION));
#endif
#ifdef _SC_MESSAGE_PASSING
printf("_SC_MESSAGE_PASSING %llu\n", (unsigned long long)(long long)(_SC_MESSAGE_PASSING));
#endif
#ifdef _SC_MONOTONIC_CLOCK
printf("_SC_MONOTONIC_CLOCK %llu\n", (unsigned long long)(long long)(_SC_MONOTONIC_CLOCK));
#endif
#ifdef _SC_MQ_OPEN_MAX
printf("_SC_MQ_OPEN_MAX %llu\n", (unsigned long long)(long long)(_SC_MQ_OPEN_MAX));
#endif
#ifdef _SC_MQ_PRIO_MAX
printf("_SC_MQ_PRIO_MAX %llu\n", (unsigned long long)(long long)(_SC_MQ_PRIO_MAX));
#endif
#ifdef _SC_MULTI_PROCESS
printf("_SC_MULTI_PROCESS %llu\n", (unsigned long long)(long long)(_SC_MULTI_PROCESS));
#endif
#ifdef _SC_NETWORKING
printf("_SC_NETWORKING %llu\n", (unsigned long long)(long long)(_SC_NETWORKING));
#endif
#ifdef _SC_NGROUPS_MAX
printf("_SC_NGROUPS_MAX %llu\n", (unsigned long long)(long long)(_SC_NGROUPS_MAX));
#endif
#ifdef _SC_NL_ARGMAX
printf("_SC_NL_ARGMAX %llu\n", (unsigned long long)(long long)(_SC_NL_ARGMAX));
#endif
#ifdef _SC_NL_LANGMAX
printf("_SC_NL_LANGMAX %llu\n", (unsigned long long)(long long)(_SC_NL_LANGMAX));
#endif
#ifdef _SC_NL_MSGMAX
printf("_SC_NL_MSGMAX %llu\n", (unsigned long long)(long long)(_SC_NL_MSGMAX));
#endif
#ifdef _SC_NL_NMAX
printf("_SC_NL_NMAX %llu\n", (unsigned long long)(long long)(_SC_NL_NMAX));
#endif
#ifdef _SC_NL_SETMAX
printf("_SC_NL_SETMAX %llu\n", (unsigned long long)(long long)(_SC_NL_SETMAX));
#endif
#ifdef _SC_NL_TEXTMAX
printf("_SC_NL_TEXTMAX %llu\n", (unsigned long long)(long long)(_SC_NL_TEXTMAX));
#endif
#ifdef _SC_NPROCESSORS_CONF
printf("_SC_NPROCESSORS_CONF %llu\n", (unsigned long long)(long long)(_SC_NPROCESSORS_CONF));
#endif
#ifdef _SC_NPROCESSORS_ONLN
printf("_SC_NPROCESSORS_ONLN %llu\n", (unsigned long long)(long long)(_SC_NPROCESSORS_ONLN));
#endif
#ifdef _SC_NZERO
printf("_SC_NZERO %llu\n", (unsigned long long)(long long)(_SC_NZERO));
#endif
#ifdef _SC_OPEN_MAX
printf("_SC_OPEN_MAX %llu\n", (unsigned long long)(long long)(_SC_OPEN_MAX));
#endif
#ifdef _SC_PAGESIZE
printf("_SC_PAGESIZE %llu\n", (unsigned long long)(long long)(_SC_PAGESIZE));
#endif
#ifdef _SC_PAGE_SIZE
printf("_SC_PAGE_SIZE %llu\n", (unsigned long long)(long long)(_SC_PAGE_SIZE));
#endif
#ifdef _SC_PASS_MAX
printf("_SC_PASS_MAX %llu\n", (unsigned long long)(long long)(_SC_PASS_MAX));
#endif
#ifdef _SC_PHYS_PAGES
printf("_SC_PHYS_PAGES %llu\n", (unsigned long long)(long long)(_SC_PHYS_PAGES));
#endif
#ifdef _SC_PII
printf("_SC_PII %llu\n", (unsigned long long)(long long)(_SC_PII));
#endif
#ifdef _SC_PII_INTERNET
printf("_SC_PII_INTERNET %llu\n", (unsigned long long)(long long)(_SC_PII_INTERNET));
#endif
#ifdef _SC_PII_INTERNET_DGRAM
printf("_SC_PII_INTERNET_DGRAM %llu\n", (unsigned long long)(long long)(_SC_PII_INTERNET_DGRAM));
#endif
#ifdef _SC_PII_INTERNET_STREAM
printf("_SC_PII_INTERNET_STREAM %llu\n", (unsigned long long)(long long)(_SC_PII_INTERNET_STREAM));
#endif
#ifdef _SC_PII_OSI
printf("_SC_PII_OSI %llu\n", (unsigned long long)(long long)(_SC_PII_OSI));
#endif
#ifdef _SC_PII_OSI_CLTS
printf("_SC_PII_OSI_CLTS %llu\n", (unsigned long long)(long long)(_SC_PII_OSI_CLTS));
#endif
#ifdef _SC_PII_OSI_COTS
printf("_SC_PII_OSI_COTS %llu\n", (unsigned long long)(long long)(_SC_PII_OSI_COTS));
#endif
#ifdef _SC_PII_OSI_M
printf("_SC_PII_OSI_M %llu\n", (unsigned long long)(long long)(_SC_PII_OSI_M));
#endif
#ifdef _SC_PII_SOCKET
printf("_SC_PII_SOCKET %llu\n", (unsigned long long)(long long)(_SC_PII_SOCKET));
#endif
#ifdef _SC_PII_XTI
printf("_SC_PII_XTI %llu\n", (unsigned long long)(long long)(_SC_PII_XTI));
#endif
#ifdef _SC_PIPE
printf("_SC_PIPE %llu\n", (unsigned long long)(long long)(_SC_PIPE));
#endif
#ifdef _SC_POLL
printf("_SC_POLL %llu\n", (unsigned long long)(long long)(_SC_POLL));
#endif
#ifdef _SC_PRIORITIZED_IO
printf("_SC_PRIORITIZED_IO %llu\n", (unsigned long long)(long long)(_SC_PRIORITIZED_IO));
#endif
#ifdef _SC_PRIORITY_SCHEDULING
printf("_SC_PRIORITY_SCHEDULING %llu\n", (unsigned long long)(long long)(_SC_PRIORITY_SCHEDULING));
#endif
#ifdef _SC_RAW_SOCKETS
printf("_SC_RAW_SOCKETS %llu\n", (unsigned long long)(long long)(_SC_RAW_SOCKETS));
#endif
#ifdef _SC_READER_WRITER_LOCKS
printf("_SC_READER_WRITER_LOCKS %llu\n", (unsigned long long)(long long)(_SC_READER_WRITER_LOCKS));
#endif
#ifdef _SC_REALTIME_SIGNALS
printf("_SC_REALTIME_SIGNALS %llu\n", (unsigned long long)(long long)(_SC_REALTIME_SIGNALS));
#endif
#ifdef _SC_REGEXP
printf("_SC_REGEXP %llu\n", (unsigned long long)(long long)(_SC_REGEXP));
#endif
#ifdef _SC_REGEX_VERSION
printf("_SC_REGEX_VERSION %llu\n", (unsigned long long)(long long)(_SC_REGEX_VERSION));
#endif
#ifdef _SC_RE_DUP_MAX
printf("_SC_RE_DUP_MAX %llu\n", (unsigned long long)(long long)(_SC_RE_DUP_MAX));
#endif
#ifdef _SC_RTSIG_MAX
printf("_SC_RTSIG_MAX %llu\n", (unsigned long long)(long long)(_SC_RTSIG_MAX));
#endif
#ifdef _SC_SAVED_IDS
printf("_SC_SAVED_IDS %llu\n", (unsigned long long)(long long)(_SC_SAVED_IDS));
#endif
#ifdef _SC_SCHAR_MAX
printf("_SC_SCHAR_MAX %llu\n", (unsigned long long)(long long)(_SC_SCHAR_MAX));
#endif
#ifdef _SC_SCHAR_MIN
printf("_SC_SCHAR_MIN %llu\n", (unsigned long long)(long long)(_SC_SCHAR_MIN));
#endif
#ifdef _SC_SELECT
printf("_SC_SELECT %llu\n", (unsigned long long)(long long)(_SC_SELECT));
#endif
#ifdef _SC_SEMAPHORES
printf("_SC_SEMAPHORES %llu\n", (unsigned long long)(long long)(_SC_SEMAPHORES));
#endif
#ifdef _SC_SEM_NSEMS_MAX
printf("_SC_SEM_NSEMS_MAX %llu\n", (unsigned long long)(long long)(_SC_SEM_NSEMS_MAX));
#endif
#ifdef _SC_SEM_VALUE_MAX
printf("_SC_SEM_VALUE_MAX %llu\n", (unsigned long long)(long long)(_SC_SEM_VALUE_MAX));
#endif
#ifdef _SC_SHARED_MEMORY_OBJECTS
printf("_SC_SHARED_MEMORY_OBJECTS %llu\n", (unsigned long long)(long long)(_SC_SHARED_MEMORY_OBJECTS));
#endif
#ifdef _SC_SHELL
printf("_SC_SHELL %llu\n", (unsigned long long)(long long)(_SC_SHELL));
#endif
#ifdef _SC_SHRT_MAX
printf("_SC_SHRT_MAX %llu\n", (unsigned long long)(long long)(_SC_SHRT_MAX));
#endif
#ifdef _SC_SHRT_MIN
printf("_SC_SHRT_MIN %llu\n", (unsigned long long)(long long)(_SC_SHRT_MIN));
#endif
#ifdef _SC_SIGNALS
printf("_SC_SIGNALS %llu\n", (unsigned long long)(long long)(_SC_SIGNALS));
#endif
#ifdef _SC_SIGQUEUE_MAX
printf("_SC_SIGQUEUE_MAX %llu\n", (unsigned long long)(long long)(_SC_SIGQUEUE_MAX));
#endif
#ifdef _SC_SINGLE_PROCESS
printf("_SC_SINGLE_PROCESS %llu\n", (unsigned long long)(long long)(_SC_SINGLE_PROCESS));
#endif
#ifdef _SC_SPAWN
printf("_SC_SPAWN %llu\n", (unsigned long long)(long long)(_SC_SPAWN));
#endif
#ifdef _SC_SPIN_LOCKS
printf("_SC_SPIN_LOCKS %llu\n", (unsigned long long)(long long)(_SC_SPIN_LOCKS));
#endif
#ifdef _SC_SPORADIC_SERVER
printf("_SC_SPORADIC_SERVER %llu\n", (unsigned long long)(long long)(_SC_SPORADIC_SERVER));
#endif
#ifdef _SC_SSIZE_MAX
printf("_SC_SSIZE_MAX %llu\n", (unsigned long long)(long long)(_SC_SSIZE_MAX));
#endif
#ifdef _SC_SS_REPL_MAX
printf("_SC_SS_REPL_MAX %llu\n", (unsigned long long)(long long)(_SC_SS_REPL_MAX));
#endif
#ifdef _SC_STREAMS
printf("_SC_STREAMS %llu\n", (unsigned long long)(long long)(_SC_STREAMS));
#endif
#ifdef _SC_STREAM_MAX
printf("_SC_STREAM_MAX %llu\n", (unsigned long long)(long long)(_SC_STREAM_MAX));
#endif
#ifdef _SC_SYMLOOP_MAX
printf("_SC_SYMLOOP_MAX %llu\n", (unsigned long long)(long long)(_SC_SYMLOOP_MAX));
#endif
#ifdef _SC_SYNCHRONIZED_IO
printf("_SC_SYNCHRONIZED_IO %llu\n", (unsigned long long)(long long)(_SC_SYNCHRONIZED_IO));
#endif
#ifdef _SC_SYSTEM_DATABASE
printf("_SC_SYSTEM_DATABASE %llu\n", (unsigned long long)(long long)(_SC_SYSTEM_DATABASE));
#endif
#ifdef _SC_SYSTEM_DATABASE_R
printf("_SC_SYSTEM_DATABASE_R %llu\n", (unsigned long long)(long long)(_SC_SYSTEM_DATABASE_R));
#endif
#ifdef _SC_THREADS
printf("_SC_THREADS %llu\n", (unsigned long long)(long long)(_SC_THREADS));
#endif
#ifdef _SC_THREAD_ATTR_STACKADDR
printf("_SC_THREAD_ATTR_STACKADDR %llu\n", (unsigned long long)(long long)(_SC_THREAD_ATTR_STACKADDR));
#endif
#ifdef _SC_THREAD_ATTR_STACKSIZE
printf("_SC_THREAD_ATTR_STACKSIZE %llu\n", (unsigned long long)(long long)(_SC_THREAD_ATTR_STACKSIZE));
#endif
#ifdef _SC_THREAD_CPUTIME
printf("_SC_THREAD_CPUTIME %llu\n", (unsigned long long)(long long)(_SC_THREAD_CPUTIME));
#endif
#ifdef _SC_THREAD_DESTRUCTOR_ITERATIONS
printf("_SC_THREAD_DESTRUCTOR_ITERATIONS %llu\n", (unsigned long long)(long long)(_SC_THREAD_DESTRUCTOR_ITERATIONS));
#endif
#ifdef _SC_THREAD_KEYS_MAX
printf("_SC_THREAD_KEYS_MAX %llu\n", (unsigned long long)(long long)(_SC_THREAD_KEYS_MAX));
#endif
#ifdef _SC_THREAD_PRIORITY_SCHEDULING
printf("_SC_THREAD_PRIORITY_SCHEDULING %llu\n", (unsigned long long)(long long)(_SC_THREAD_PRIORITY_SCHEDULING));
#endif
#ifdef _SC_THREAD_PRIO_INHERIT
printf("_SC_THREAD_PRIO_INHERIT %llu\n", (unsigned long long)(long long)(_SC_THREAD_PRIO_INHERIT));
#endif
#ifdef _SC_THREAD_PRIO_PROTECT
printf("_SC_THREAD_PRIO_PROTECT %llu\n", (unsigned long long)(long long)(_SC_THREAD_PRIO_PROTECT));
#endif
#ifdef _SC_THREAD_PROCESS_SHARED
printf("_SC_THREAD_PROCESS_SHARED %llu\n", (unsigned long long)(long long)(_SC_THREAD_PROCESS_SHARED));
#endif
#ifdef _SC_THREAD_ROBUST_PRIO_INHERIT
printf("_SC_THREAD_ROBUST_PRIO_INHERIT %llu\n", (unsigned long long)(long long)(_SC_THREAD_ROBUST_PRIO_INHERIT));
#endif
#ifdef _SC_THREAD_ROBUST_PRIO_PROTECT
printf("_SC_THREAD_ROBUST_PRIO_PROTECT %llu\n", (unsigned long long)(long long)(_SC_THREAD_ROBUST_PRIO_PROTECT));
#endif
#ifdef _SC_THREAD_SAFE_FUNCTIONS
printf("_SC_THREAD_SAFE_FUNCTIONS %llu\n", (unsigned long long)(long long)(_SC_THREAD_SAFE_FUNCTIONS));
#endif
#ifdef _SC_THREAD_SPORADIC_SERVER
printf("_SC_THREAD_SPORADIC_SERVER %llu\n", (unsigned long long)(long long)(_SC_THREAD_SPORADIC_SERVER));
#endif
#ifdef _SC_THREAD_STACK_MIN
printf("_SC_THREAD_STACK_MIN %llu\n", (unsigned long long)(long long)(_SC_THREAD_STACK_MIN));
#endif
#ifdef _SC_THREAD_THREADS_MAX
printf("_SC_THREAD_THREADS_MAX %llu\n", (unsigned long long)(long long)(_SC_THREAD_THREADS_MAX));
#endif
#ifdef _SC_TIMEOUTS
printf("_SC_TIMEOUTS %llu\n", (unsigned long long)(long long)(_SC_TIMEOUTS));
#endif
#ifdef _SC_TIMERS
printf("_SC_TIMERS %llu\n", (unsigned long long)(long long)(_SC_TIMERS));
#endif
#ifdef _SC_TIMER_MAX
printf("_SC_TIMER_MAX %llu\n", (unsigned long long)(long long)(_SC_TIMER_MAX));
#endif
#ifdef _SC_TRACE
printf("_SC_TRACE %llu\n", (unsigned long long)(long long)(_SC_TRACE));
#endif
#ifdef _SC_TRACE_EVENT_FILTER
printf("_SC_TRACE_EVENT_FILTER %llu\n", (unsigned long long)(long long)(_SC_TRACE_EVENT_FILTER));
#endif
#ifdef _SC_TRACE_EVENT_NAME_MAX
printf("_SC_TRACE_EVENT_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_TRACE_EVENT_NAME_MAX));
#endif
#ifdef _SC_TRACE_INHERIT
printf("_SC_TRACE_INHERIT %llu\n", (unsigned long long)(long long)(_SC_TRACE_INHERIT));
#endif
#ifdef _SC_TRACE_LOG
printf("_SC_TRACE_LOG %llu\n", (unsigned long long)(long long)(_SC_TRACE_LOG));
#endif
#ifdef _SC_TRACE_NAME_MAX
printf("_SC_TRACE_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_TRACE_NAME_MAX));
#endif
#ifdef _SC_TRACE_SYS_MAX
printf("_SC_TRACE_SYS_MAX %llu\n", (unsigned long long)(long long)(_SC_TRACE_SYS_MAX));
#endif
#ifdef _SC_TRACE_USER_EVENT_MAX
printf("_SC_TRACE_USER_EVENT_MAX %llu\n", (unsigned long long)(long long)(_SC_TRACE_USER_EVENT_MAX));
#endif
#ifdef _SC_TTY_NAME_MAX
printf("_SC_TTY_NAME_MAX %llu\n", (unsigned long long)(long long)(_SC_TTY_NAME_MAX));
#endif
#ifdef _SC_TYPED_MEMORY_OBJECTS
printf("_SC_TYPED_MEMORY_OBJECTS %llu\n", (unsigned long long)(long long)(_SC_TYPED_MEMORY_OBJECTS));
#endif
#ifdef _SC_TZNAME_MAX
printf("_SC_TZNAME_MAX %llu\n", (unsigned long long)(long long)(_SC_TZNAME_MAX));
#endif
#ifdef _SC_T_IOV_MAX
printf("_SC_T_IOV_MAX %llu\n", (unsigned long long)(long long)(_SC_T_IOV_MAX));
#endif
#ifdef _SC_UCHAR_MAX
printf("_SC_UCHAR_MAX %llu\n", (unsigned long long)(long long)(_SC_UCHAR_MAX));
#endif
#ifdef _SC_UINT_MAX
printf("_SC_UINT_MAX %llu\n", (unsigned long long)(long long)(_SC_UINT_MAX));
#endif
#ifdef _SC_UIO_MAXIOV
printf("_SC_UIO_MAXIOV %llu\n", (unsigned long long)(long long)(_SC_UIO_MAXIOV));
#endif
#ifdef _SC_ULONG_MAX
printf("_SC_ULONG_MAX %llu\n", (unsigned long long)(long long)(_SC_ULONG_MAX));
#endif
#ifdef _SC_USER_GROUPS
printf("_SC_USER_GROUPS %llu\n", (unsigned long long)(long long)(_SC_USER_GROUPS));
#endif
#ifdef _SC_USER_GROUPS_R
printf("_SC_USER_GROUPS_R %llu\n", (unsigned long long)(long long)(_SC_USER_GROUPS_R));
#endif
#ifdef _SC_USHRT_MAX
printf("_SC_USHRT_MAX %llu\n", (unsigned long long)(long long)(_SC_USHRT_MAX));
#endif
#ifdef _SC_V6_ILP32_OFF32
printf("_SC_V6_ILP32_OFF32 %llu\n", (unsigned long long)(long long)(_SC_V6_ILP32_OFF32));
#endif
#ifdef _SC_V6_ILP32_OFFBIG
printf("_SC_V6_ILP32_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_V6_ILP32_OFFBIG));
#endif
#ifdef _SC_V6_LP64_OFF64
printf("_SC_V6_LP64_OFF64 %llu\n", (unsigned long long)(long long)(_SC_V6_LP64_OFF64));
#endif
#ifdef _SC_V6_LPBIG_OFFBIG
printf("_SC_V6_LPBIG_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_V6_LPBIG_OFFBIG));
#endif
#ifdef _SC_V7_ILP32_OFF32
printf("_SC_V7_ILP32_OFF32 %llu\n", (unsigned long long)(long long)(_SC_V7_ILP32_OFF32));
#endif
#ifdef _SC_V7_ILP32_OFFBIG
printf("_SC_V7_ILP32_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_V7_ILP32_OFFBIG));
#endif
#ifdef _SC_V7_LP64_OFF64
printf("_SC_V7_LP64_OFF64 %llu\n", (unsigned long long)(long long)(_SC_V7_LP64_OFF64));
#endif
#ifdef _SC_V7_LPBIG_OFFBIG
printf("_SC_V7_LPBIG_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_V7_LPBIG_OFFBIG));
#endif
#ifdef _SC_VERSION
printf("_SC_VERSION %llu\n", (unsigned long long)(long long)(_SC_VERSION));
#endif
#ifdef _SC_WORD_BIT
printf("_SC_WORD_BIT %llu\n", (unsigned long long)(long long)(_SC_WORD_BIT));
#endif
#ifdef _SC_XBS5_ILP32_OFF32
printf("_SC_XBS5_ILP32_OFF32 %llu\n", (unsigned long long)(long long)(_SC_XBS5_ILP32_OFF32));
#endif
#ifdef _SC_XBS5_ILP32_OFFBIG
printf("_SC_XBS5_ILP32_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_XBS5_ILP32_OFFBIG));
#endif
#ifdef _SC_XBS5_LP64_OFF64
printf("_SC_XBS5_LP64_OFF64 %llu\n", (unsigned long long)(long long)(_SC_XBS5_LP64_OFF64));
#endif
#ifdef _SC_XBS5_LPBIG_OFFBIG
printf("_SC_XBS5_LPBIG_OFFBIG %llu\n", (unsigned long long)(long long)(_SC_XBS5_LPBIG_OFFBIG));
#endif
#ifdef _SC_XOPEN_CRYPT
printf("_SC_XOPEN_CRYPT %llu\n", (unsigned long long)(long long)(_SC_XOPEN_CRYPT));
#endif
#ifdef _SC_XOPEN_ENH_I18N
printf("_SC_XOPEN_ENH_I18N %llu\n", (unsigned long long)(long long)(_SC_XOPEN_ENH_I18N));
#endif
#ifdef _SC_XOPEN_LEGACY
printf("_SC_XOPEN_LEGACY %llu\n", (unsigned long long)(long long)(_SC_XOPEN_LEGACY));
#endif
#ifdef _SC_XOPEN_REALTIME
printf("_SC_XOPEN_REALTIME %llu\n", (unsigned long long)(long long)(_SC_XOPEN_REALTIME));
#endif
#ifdef _SC_XOPEN_REALTIME_THREADS
printf("_SC_XOPEN_REALTIME_THREADS %llu\n", (unsigned long long)(long long)(_SC_XOPEN_REALTIME_THREADS));
#endif
#ifdef _SC_XOPEN_SHM
printf("_SC_XOPEN_SHM %llu\n", (unsigned long long)(long long)(_SC_XOPEN_SHM));
#endif
#ifdef _SC_XOPEN_STREAMS
printf("_SC_XOPEN_STREAMS %llu\n", (unsigned long long)(long long)(_SC_XOPEN_STREAMS));
#endif
#ifdef _SC_XOPEN_UNIX
printf("_SC_XOPEN_UNIX %llu\n", (unsigned long long)(long long)(_SC_XOPEN_UNIX));
#endif
#ifdef _SC_XOPEN_VERSION
printf("_SC_XOPEN_VERSION %llu\n", (unsigned long long)(long long)(_SC_XOPEN_VERSION));
#endif
#ifdef _SC_XOPEN_XCU_VERSION
printf("_SC_XOPEN_XCU_VERSION %llu\n", (unsigned long long)(long long)(_SC_XOPEN_XCU_VERSION));
#endif
#ifdef _SC_XOPEN_XPG2
printf("_SC_XOPEN_XPG2 %llu\n", (unsigned long long)(long long)(_SC_XOPEN_XPG2));
#endif
#ifdef _SC_XOPEN_XPG3
printf("_SC_XOPEN_XPG3 %llu\n", (unsigned long long)(long long)(_SC_XOPEN_XPG3));
#endif
#ifdef _SC_XOPEN_XPG4
printf("_SC_XOPEN_XPG4 %llu\n", (unsigned long long)(long long)(_SC_XOPEN_XPG4));
#endif
#ifdef __NFT_REG_MAX
printf("__NFT_REG_MAX %llu\n", (unsigned long long)(long long)(__NFT_REG_MAX));
#endif
#ifdef __SIZEOF_PTHREAD_BARRIERATTR_T
printf("__SIZEOF_PTHREAD_BARRIERATTR_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_BARRIERATTR_T));
#endif
#ifdef __SIZEOF_PTHREAD_BARRIER_T
printf("__SIZEOF_PTHREAD_BARRIER_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_BARRIER_T));
#endif
#ifdef __SIZEOF_PTHREAD_CONDATTR_T
printf("__SIZEOF_PTHREAD_CONDATTR_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_CONDATTR_T));
#endif
#ifdef __SIZEOF_PTHREAD_COND_T
printf("__SIZEOF_PTHREAD_COND_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_COND_T));
#endif
#ifdef __SIZEOF_PTHREAD_MUTEXATTR_T
printf("__SIZEOF_PTHREAD_MUTEXATTR_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_MUTEXATTR_T));
#endif
#ifdef __SIZEOF_PTHREAD_MUTEX_T
printf("__SIZEOF_PTHREAD_MUTEX_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_MUTEX_T));
#endif
#ifdef __SIZEOF_PTHREAD_RWLOCKATTR_T
printf("__SIZEOF_PTHREAD_RWLOCKATTR_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_RWLOCKATTR_T));
#endif
#ifdef __SIZEOF_PTHREAD_RWLOCK_T
printf("__SIZEOF_PTHREAD_RWLOCK_T %llu\n", (unsigned long long)(long long)(__SIZEOF_PTHREAD_RWLOCK_T));
#endif
#ifdef __UT_HOSTSIZE
printf("__UT_HOSTSIZE %llu\n", (unsigned long long)(long long)(__UT_HOSTSIZE));
#endif
#ifdef __UT_LINESIZE
printf("__UT_LINESIZE %llu\n", (unsigned long long)(long long)(__UT_LINESIZE));
#endif
#ifdef __UT_NAMESIZE
printf("__UT_NAMESIZE %llu\n", (unsigned long long)(long long)(__UT_NAMESIZE));
#endif
#ifdef __WALL
printf("__WALL %llu\n", (unsigned long long)(long long)(__WALL));
#endif
#ifdef __WCLONE
printf("__WCLONE %llu\n", (unsigned long long)(long long)(__WCLONE));
#endif
#ifdef __WNOTHREAD
printf("__WNOTHREAD %llu\n", (unsigned long long)(long long)(__WNOTHREAD));
#endif
return 0;
}