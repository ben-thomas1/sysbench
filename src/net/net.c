#include "net/net.h"
#include "core/report.h"
#include "core/timer.h"
#include "sys/ipc.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* 1 MiB per write()/read(): large enough that syscall cost is amortised
 * (FINDINGS: 64 KiB and 1 MiB within 10% here), small enough to stay in the
 * L2. Socket buffers are left at their defaults so Linux TCP autotuning stays
 * on (setting SO_SNDBUF/SO_RCVBUF disables it). */
#define BW_IO_BYTES  (1U << 20)
#define BW_WINDOW_NS 1'000'000'000ULL

/* --- Socket setup (outside every timed window) --- */

/* No SO_RCVTIMEO/SO_SNDTIMEO: on macOS they cost ~8% throughput (TCP and
 * Unix, results/agent-sysnet). A stuck peer is instead woken by shutdown()
 * on every error path, which makes its blocking read/write return. */
static void set_nosigpipe(int fd) {
#if defined(SO_NOSIGPIPE)
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
    (void)fd; /* Linux: SIGPIPE is ignored process-wide in net_run */
#endif
}

static sb_status_e set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0 ? SB_OK : SB_ERR_SYS;
}

/* Connected TCP pair over 127.0.0.1: out[0] = accepted side, out[1] = client.
 * A blocking connect to loopback completes from the listen backlog before
 * accept() runs, so no helper thread is needed. */
static sb_status_e tcp_pair(bool nodelay, int out[2]) {
    sb_status_e s   = SB_ERR_SYS;
    int         srv = -1;
    int         cli = -1;
    int         con = -1;

    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_port        = 0,
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    socklen_t alen = sizeof(addr);
    int       one  = 1;

    srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { goto fail; }
    if (setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) { goto fail; }
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) { goto fail; }
    if (getsockname(srv, (struct sockaddr *)&addr, &alen) < 0) { goto fail; }
    if (listen(srv, 1) < 0) { goto fail; }

    cli = socket(AF_INET, SOCK_STREAM, 0);
    if (cli < 0) { goto fail; }
    if (connect(cli, (struct sockaddr *)&addr, sizeof(addr)) < 0) { goto fail; }

    struct pollfd pfd = { .fd = srv, .events = POLLIN };
    int           ready;
    do { ready = poll(&pfd, 1, (int)(SB_IPC_TIMEOUT_NS / 1'000'000ULL)); } while (ready < 0 && errno == EINTR);
    if (ready == 0) {
        s = SB_ERR_TIMEOUT;
        goto fail;
    }
    if (ready < 0) { goto fail; }
    con = accept(srv, NULL, NULL);
    if (con < 0) { goto fail; }

    set_nosigpipe(con);
    set_nosigpipe(cli);
    if (nodelay && (set_nodelay(con) != SB_OK || set_nodelay(cli) != SB_OK)) { goto fail; }
    close(srv);
    out[0] = con;
    out[1] = cli;
    return SB_OK;

fail:
    if (con >= 0) { close(con); }
    if (cli >= 0) { close(cli); }
    if (srv >= 0) { close(srv); }
    return s;
}

static sb_status_e unix_pair(int out[2]) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, out) < 0) { return SB_ERR_SYS; }
    set_nosigpipe(out[0]);
    set_nosigpipe(out[1]);
    return SB_OK;
}

static sb_status_e make_pair(bool tcp, bool nodelay, int out[2]) {
    return tcp ? tcp_pair(nodelay, out) : unix_pair(out);
}

/* --- Throughput: one writer thread streams, the reader thread measures ---
 * The reader warms up for SB_WARMUP_NS, counts bytes received in a fixed
 * window, then raises `stop` and drains to EOF so the writer never blocks. */

typedef struct {
    int          fd;
    atomic_bool *stop;
    sb_status_e  status;
    f64          gbps;
} bw_arg;

static void *bw_writer(void *p) {
    bw_arg *a = p;
    sb_ipc_place(1);
    char *buf = SB_MALLOC(BW_IO_BYTES);
    if (buf == NULL) {
        a->status = SB_ERR_NOMEM;
        (void)shutdown(a->fd, SHUT_RDWR);
        return NULL;
    }
    memset(buf, 0xAB, BW_IO_BYTES);
    a->status = SB_OK;
    while (!atomic_load_explicit(a->stop, memory_order_relaxed) && a->status == SB_OK) {
        size_t done = 0;
        while (done < BW_IO_BYTES) {
            ssize_t n = write(a->fd, buf + done, BW_IO_BYTES - done);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) {
                a->status = SB_ERR_IO; /* EPIPE after the reader shut down, or a real error */
                break;
            }
            done += (size_t)n;
        }
    }
    (void)shutdown(a->fd, SHUT_WR);
    SB_FREE(buf);
    return NULL;
}

/* Reads until `dur_ns` has passed; returns bytes and elapsed ns. */
static sb_status_e bw_read_for(int fd, char *buf, u64 dur_ns, u64 *out_bytes, u64 *out_ns) {
    u64 bytes = 0;
    u64 t0    = sb_timer_now_ns();
    u64 t     = t0;
    while (t - t0 < dur_ns) {
        ssize_t n = read(fd, buf, BW_IO_BYTES);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { return SB_ERR_IO; }
        bytes += (u64)n;
        t      = sb_timer_now_ns();
    }
    *out_bytes = bytes;
    *out_ns    = t - t0;
    return SB_OK;
}

static void *bw_reader(void *p) {
    bw_arg *a = p;
    sb_ipc_place(0);
    char *buf = SB_MALLOC(BW_IO_BYTES);
    if (buf == NULL) {
        a->status = SB_ERR_NOMEM;
        atomic_store(a->stop, true);
        (void)shutdown(a->fd, SHUT_RDWR);
        return NULL;
    }
    memset(buf, 0, BW_IO_BYTES);
    u64 bytes = 0;
    u64 ns    = 0;
    a->status = bw_read_for(a->fd, buf, SB_WARMUP_NS, &bytes, &ns);
    if (a->status == SB_OK) { a->status = bw_read_for(a->fd, buf, BW_WINDOW_NS, &bytes, &ns); }
    atomic_store(a->stop, true);
    if (a->status != SB_OK) {
        (void)shutdown(a->fd, SHUT_RDWR);
    } else {
        for (;;) { /* drain to the writer's EOF */
            ssize_t n = read(a->fd, buf, BW_IO_BYTES);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { break; }
        }
        if (ns == 0) { a->status = SB_ERR_RANGE; }
        else         { a->gbps = (f64)bytes / (f64)ns; }
    }
    SB_FREE(buf);
    return NULL;
}

static sb_status_e measure_bw(bool tcp, f64 *out_gbps) {
    int fds[2];
    sb_status_e s = make_pair(tcp, false, fds);
    if (s != SB_OK) { return s; }

    atomic_bool stop = false;
    bw_arg      ra   = { .fd = fds[0], .stop = &stop, .status = SB_ERR_SYS };
    bw_arg      wa   = { .fd = fds[1], .stop = &stop, .status = SB_ERR_SYS };
    pthread_t   tr;
    pthread_t   tw;
    if (pthread_create(&tw, NULL, bw_writer, &wa) != 0) {
        s = SB_ERR_SYS;
        goto out;
    }
    if (pthread_create(&tr, NULL, bw_reader, &ra) != 0) {
        atomic_store(&stop, true);
        (void)shutdown(fds[0], SHUT_RDWR);
        pthread_join(tw, NULL);
        s = SB_ERR_SYS;
        goto out;
    }
    pthread_join(tr, NULL);
    pthread_join(tw, NULL);
    if (ra.status != SB_OK) { s = ra.status; }
    else if (wa.status != SB_OK) { s = wa.status; }
    else { *out_gbps = ra.gbps; }

out:
    close(fds[0]);
    close(fds[1]);
    return s;
}

/* --- Latency: one-byte ping-pong between two placed threads --- */

static sb_status_e measure_rtt(bool tcp, bool spin, f64 *out_us) {
    int fds[2];
    sb_status_e s = make_pair(tcp, true, fds);
    if (s != SB_OK) { return s; }
    if (spin && (sb_ipc_set_nonblock(fds[0]) != SB_OK || sb_ipc_set_nonblock(fds[1]) != SB_OK)) {
        s = SB_ERR_SYS;
    }
    f64 rtt_ns = 0;
    if (s == SB_OK) { s = sb_ipc_pingpong_threads(fds[0], fds[0], fds[1], fds[1], spin, &rtt_ns); }
    close(fds[0]);
    close(fds[1]);
    if (s == SB_OK) { *out_us = rtt_ns / 1000.0; }
    return s;
}

/* --- Section --- */

static void report(const char *test, sb_status_e s, f64 v, const char *unit) {
    if (s == SB_OK) { sb_report_value(test, v, unit, SB_KIND_MEASURED); }
    else            { sb_report_error(test, s); }
}

static void info_default_buffers(void) {
    int fds[2];
    if (tcp_pair(false, fds) != SB_OK) { return; }
    int       snd = 0;
    int       rcv = 0;
    socklen_t len = sizeof(snd);
    (void)getsockopt(fds[1], SOL_SOCKET, SO_SNDBUF, &snd, &len);
    len = sizeof(rcv);
    (void)getsockopt(fds[0], SOL_SOCKET, SO_RCVBUF, &rcv, &len);
    close(fds[0]);
    close(fds[1]);
    sb_report_info("TCP initial buffers: SO_SNDBUF %.0f KiB, SO_RCVBUF %.0f KiB (defaults; autotuning untouched)",
                   (f64)snd / 1024.0, (f64)rcv / 1024.0);
}

static void run_family(bool tcp) {
    const char *bw_name    = tcp ? "TCP throughput" : "Unix throughput";
    const char *block_name = tcp ? "TCP RTT, blocking (incl. wakeup)" : "Unix RTT, blocking (incl. wakeup)";
    const char *spin_name  = tcp ? "TCP RTT, busy-poll" : "Unix RTT, busy-poll";

    f64 v = 0;
    sb_status_e s = measure_bw(tcp, &v);
    report(bw_name, s, v, "GB/s");
    s = measure_rtt(tcp, false, &v);
    report(block_name, s, v, "us/RTT");
    if (sb_ipc_can_spin()) {
        s = measure_rtt(tcp, true, &v);
        report(spin_name, s, v, "us/RTT");
    } else {
        sb_report_skip(spin_name, "needs 2 CPUs");
    }
}

static sb_status_e net_run(void) {
    signal(SIGPIPE, SIG_IGN);
    sb_ipc_peers_init();

    sb_report_info("Local OS stack only (127.0.0.1 TCP, AF_UNIX socketpair); no NIC involved");
    sb_report_info("Throughput: 1 MiB writes/reads, 250 ms warmup + 1 s window");
    sb_report_info("RTT: 1-byte ping-pong; blocking minus busy-poll = sleep/wakeup cost");
    sb_report_info("Peers: %s", sb_ipc_peers_desc());
    info_default_buffers();
    char exts[256];
    if (sb_ipc_active_exts("network_extension", exts, sizeof(exts)) > 0) {
        sb_report_info("Network-filter extensions active: %s", exts);
        sb_report_info("-> TCP rows include their overhead; compare busy-poll TCP vs Unix RTT");
    }

    sb_report_group("TCP loopback");
    run_family(true);
    sb_report_group("Unix socketpair (non-network reference)");
    run_family(false);
    return SB_OK;
}

const sb_section sb_section_net = {
    .name       = "net",
    .title      = "Network (loopback)",
    .help       = "  Local OS network stack, not NIC throughput. TCP over 127.0.0.1 and an\n"
                  "  AF_UNIX stream socketpair as a non-network IPC reference.\n"
                  "  Throughput: one thread streams 1 MiB writes, another reads; bytes per\n"
                  "  second over a 1 s window after a 250 ms warmup (default socket buffers).\n"
                  "  RTT: one-byte ping-pong (TCP_NODELAY), blocking (includes sleep/wakeup)\n"
                  "  and busy-poll (O_NONBLOCK spin, transport only). A busy-poll TCP RTT far\n"
                  "  above the Unix one points at filter software, not the stack. macOS lists\n"
                  "  active network-filter system extensions when present.\n",
    .run        = net_run,
    .repeatable = true,
};
