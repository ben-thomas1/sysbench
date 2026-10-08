#include "sys/sys.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"
#include "core/timer.h"
#include "sys/ipc.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <sys/wait.h>
#include <unistd.h>

#define CALL_WINDOW_NS 500'000'000ULL
#define SHM_BATCH      1024          /* round trips between clock reads */
#define SHM_POLL_MASK  0xFFFFF       /* check the timeout every 1 Mi empty polls */
#define SHM_STOP       UINT64_MAX

/* --- Per-call costs via the time-based runner (warmup + fixed window) --- */

/* getuid() is a real kernel trap on both platforms (getpid() is cached in
 * user space on macOS); nothing hooks it, so it is the bare trap cost. */
static u64 getuid_work(void *ctx, u32 tid, u64 chunk) {
    (void)ctx;
    (void)tid;
    for (u64 i = 0; i < chunk; i++) {
        uid_t u = getuid();
        __asm__ volatile("" : : "r"(u));
    }
    return chunk;
}

/* open()+close() of /dev/null: a short VFS path that endpoint-security
 * software intercepts (AUTH/NOTIFY_OPEN on macOS). Compared with getuid it
 * shows how much such hooks add to file operations. */
static u64 open_close_work(void *ctx, u32 tid, u64 chunk) {
    (void)tid;
    atomic_bool *failed = ctx;
    for (u64 i = 0; i < chunk; i++) {
        int fd = open("/dev/null", O_RDONLY);
        if (fd < 0) {
            atomic_store(failed, true);
            return i;
        }
        close(fd);
    }
    return chunk;
}

/* Thread create + join of an empty thread: stack allocation, kernel thread
 * creation, the first schedule of the new thread, its exit, and the joiner's
 * wakeup. */
static void *noop_thread(void *arg) { return arg; }

static u64 create_join_work(void *ctx, u32 tid, u64 chunk) {
    (void)tid;
    atomic_bool *failed = ctx;
    for (u64 i = 0; i < chunk; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, noop_thread, NULL) != 0) {
            atomic_store(failed, true);
            return i;
        }
        pthread_join(t, NULL);
    }
    return chunk;
}

static sb_status_e measure_per_call(sb_work_fn fn, u64 chunk, f64 *out_ns) {
    atomic_bool failed = false;
    sb_par_cfg  cfg = {
        .nthreads  = 1,
        .place     = SB_CORE_ANY,
        .window_ns = CALL_WINDOW_NS,
        .chunk     = chunk,
        .fn        = fn,
        .ctx       = &failed,
    };
    sb_par_result r;
    sb_status_e   s = sb_par_run(&cfg, &r);
    if (s != SB_OK) { return s; }
    if (atomic_load(&failed)) { return SB_ERR_SYS; }
    if (r.units_per_sec <= 0) { return SB_ERR_RANGE; }
    *out_ns = 1e9 / r.units_per_sec;
    return SB_OK;
}

/* --- Pipe ping-pong between two processes ---
 * a: driver -> echo, b: echo -> driver. Blocking reads sleep until the peer
 * writes, so each handoff includes a sleep and a wakeup; busy-poll spins on
 * O_NONBLOCK reads, leaving pipe transport and syscall cost. */

static void close_pair(int p[2]) {
    close(p[0]);
    close(p[1]);
}

static sb_status_e pipe_pingpong(bool spin, f64 *out_rtt_ns) {
    int a[2];
    int b[2];
    if (pipe(a) < 0) { return SB_ERR_SYS; }
    if (pipe(b) < 0) {
        close_pair(a);
        return SB_ERR_SYS;
    }
    if (spin && (sb_ipc_set_nonblock(a[0]) != SB_OK || sb_ipc_set_nonblock(b[0]) != SB_OK)) {
        close_pair(a);
        close_pair(b);
        return SB_ERR_SYS;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close_pair(a);
        close_pair(b);
        return SB_ERR_SYS;
    }
    if (pid == 0) {
        sb_ipc_place(1);
        close(a[1]);
        close(b[0]);
        _exit(sb_ipc_echo(a[0], b[1], spin) == SB_OK ? 0 : 1);
    }

    close(a[0]);
    close(b[1]);
    sb_status_e s = sb_ipc_drive(b[0], a[1], spin, SB_IPC_WINDOW_NS, out_rtt_ns);
    close(a[1]);  /* on error the child reads EOF and exits */
    close(b[0]);

    int   status = 0;
    pid_t w;
    do { w = waitpid(pid, &status, 0); } while (w < 0 && errno == EINTR);
    if (s != SB_OK) { return s; }
    if (w < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) { return SB_ERR_SYS; }
    return SB_OK;
}

typedef struct {
    bool        spin;
    f64         rtt_ns;
    sb_status_e status;
} pipe_arg;

/* The driver runs on a placed thread and forks from there, so the main
 * thread's affinity/QoS stays untouched and the child inherits side 0's. */
static void *pipe_thread(void *p) {
    pipe_arg *a = p;
    sb_ipc_place(0);
    a->status = pipe_pingpong(a->spin, &a->rtt_ns);
    return NULL;
}

static sb_status_e measure_pipe_handoff(bool spin, f64 *out_ns) {
    pipe_arg  a = { .spin = spin, .status = SB_ERR_SYS };
    pthread_t t;
    if (pthread_create(&t, NULL, pipe_thread, &a) != 0) { return SB_ERR_SYS; }
    pthread_join(t, NULL);
    if (a.status != SB_OK) { return a.status; }
    *out_ns = a.rtt_ns / 2.0;
    return SB_OK;
}

/* --- Shared-memory ping-pong between two threads ---
 * One atomic counter on its own cache line: the driver writes an odd value,
 * the peer answers with the next even one. Each handoff is one cache-line
 * transfer between cores, the floor under any IPC mechanism. */

typedef struct {
    _Atomic u64 *flag;
    sb_status_e  status;
} shm_arg;

static void *shm_peer(void *p) {
    shm_arg     *a    = p;
    _Atomic u64 *flag = a->flag;
    sb_ipc_place(1);
    u64 last    = 0;
    u32 polls   = 0;
    u64 t_start = 0;
    for (;;) {
        u64 v = atomic_load_explicit(flag, memory_order_acquire);
        if (v == last) {
            if ((++polls & SHM_POLL_MASK) == 0) {
                u64 t = sb_timer_now_ns();
                if (t_start == 0) { t_start = t; }
                else if (t - t_start > SB_IPC_TIMEOUT_NS) {
                    a->status = SB_ERR_TIMEOUT;
                    return NULL;
                }
            }
            continue;
        }
        if (v == SHM_STOP) { break; }
        last = v + 1;
        atomic_store_explicit(flag, last, memory_order_release);
        t_start = 0;
    }
    a->status = SB_OK;
    return NULL;
}

/* Round trips for at least `dur_ns`; `seq` is the last value seen. */
static sb_status_e shm_loop(_Atomic u64 *flag, u64 *seq, u64 dur_ns, u64 *out_n, u64 *out_ns) {
    u64 n  = 0;
    u64 s  = *seq;
    u64 t0 = sb_timer_now_ns();
    u64 t  = t0;
    do {
        for (u32 i = 0; i < SHM_BATCH; i++) {
            atomic_store_explicit(flag, s + 1, memory_order_release);
            u32 polls = 0;
            while (atomic_load_explicit(flag, memory_order_acquire) != s + 2) {
                if ((++polls & SHM_POLL_MASK) == 0 && sb_timer_now_ns() - t0 > dur_ns + SB_IPC_TIMEOUT_NS) {
                    *seq = s;
                    return SB_ERR_TIMEOUT;
                }
            }
            s += 2;
        }
        n += SHM_BATCH;
        t  = sb_timer_now_ns();
    } while (t - t0 < dur_ns);
    *seq    = s;
    *out_n  = n;
    *out_ns = t - t0;
    return SB_OK;
}

typedef struct {
    _Atomic u64 *flag;
    f64          rtt_ns;
    sb_status_e  status;
} shm_drv_arg;

static void *shm_driver(void *p) {
    shm_drv_arg *a = p;
    sb_ipc_place(0);
    u64 seq = 0;
    u64 n   = 0;
    u64 ns  = 0;
    a->status = shm_loop(a->flag, &seq, SB_WARMUP_NS, &n, &ns);
    if (a->status == SB_OK) { a->status = shm_loop(a->flag, &seq, SB_IPC_WINDOW_NS, &n, &ns); }
    atomic_store_explicit(a->flag, SHM_STOP, memory_order_release);
    if (a->status == SB_OK) {
        if (n == 0 || ns == 0) { a->status = SB_ERR_RANGE; }
        else                   { a->rtt_ns = (f64)ns / (f64)n; }
    }
    return NULL;
}

static sb_status_e measure_shm_handoff(f64 *out_ns) {
    /* The counter gets a whole cache line (size from the platform) so no
     * unrelated data shares it. */
    size_t line = sb_platform_get()->cache_line;
    if (line < 64 || (line & (line - 1)) != 0) { line = 128; }
    void *mem = aligned_alloc(line, line);
    if (mem == NULL) { return SB_ERR_NOMEM; }
    _Atomic u64 *flag = mem;
    atomic_init(flag, 0);

    shm_arg     pa = { .flag = flag, .status = SB_ERR_SYS };
    shm_drv_arg da = { .flag = flag, .status = SB_ERR_SYS };
    pthread_t   tp;
    pthread_t   td;
    sb_status_e s = SB_OK;
    if (pthread_create(&tp, NULL, shm_peer, &pa) != 0) {
        free(mem);
        return SB_ERR_SYS;
    }
    if (pthread_create(&td, NULL, shm_driver, &da) != 0) {
        atomic_store(flag, SHM_STOP);
        s = SB_ERR_SYS;
    } else {
        pthread_join(td, NULL);
    }
    pthread_join(tp, NULL);
    free(mem);
    if (s != SB_OK) { return s; }
    if (da.status != SB_OK) { return da.status; }
    if (pa.status != SB_OK) { return pa.status; }
    *out_ns = da.rtt_ns / 2.0;
    return SB_OK;
}

/* --- Section --- */

static void report(const char *test, sb_status_e s, f64 v, const char *unit) {
    if (s == SB_OK) { sb_report_value(test, v, unit, SB_KIND_MEASURED); }
    else            { sb_report_error(test, s); }
}

static sb_status_e sys_run(void) {
    signal(SIGPIPE, SIG_IGN);
    sb_ipc_peers_init();

    const sb_platform *p = sb_platform_get();
    sb_report_info("OS: %s (%s)", p->os[0] ? p->os : "unknown", p->os_release);
    char exts[256];
    u32  nes = sb_ipc_active_exts("endpoint_security", exts, sizeof(exts));
    if (nes > 0) {
        sb_report_info("Endpoint-security extensions active: %s", exts);
        sb_report_info("-> open, fork and exec costs include their hooks; not hardware ground truth");
    }

    f64 v = 0;
    sb_status_e s = measure_per_call(getuid_work, 1000, &v);
    report("Syscall (getuid)", s, v, "ns/call");
    s = measure_per_call(open_close_work, 16, &v);
    report("open+close (/dev/null)", s, v, "ns/call");

    sb_report_group("Handoff = half a ping-pong round trip");
    sb_report_info("Peers: %s", sb_ipc_peers_desc());
    s = measure_pipe_handoff(false, &v);
    report("Pipe, blocking (incl. wakeup)", s, v, "ns/handoff");
    if (sb_ipc_can_spin()) {
        s = measure_pipe_handoff(true, &v);
        report("Pipe, busy-poll", s, v, "ns/handoff");
        s = measure_shm_handoff(&v);
        report("Shared cache line (threads)", s, v, "ns/handoff");
    } else {
        sb_report_skip("Pipe, busy-poll", "needs 2 CPUs");
        sb_report_skip("Shared cache line (threads)", "needs 2 CPUs");
    }

    sb_report_group("Threads");
    s = measure_per_call(create_join_work, 16, &v);
    report("Thread create+join", s, v / 1000.0, "us");
    return SB_OK;
}

const sb_section sb_section_sys = {
    .name       = "sys",
    .title      = "System Overhead",
    .help       = "  Syscall: getuid (bare kernel trap) and open+close of /dev/null (a file\n"
                  "  path that endpoint-security software hooks; compare the two).\n"
                  "  Handoff: half a one-byte ping-pong round trip. Pipe blocking runs between\n"
                  "  two processes and includes sleep + wakeup; pipe busy-poll spins on\n"
                  "  O_NONBLOCK reads (transport + syscalls only); shared cache line is two\n"
                  "  threads bouncing an atomic (cross-core floor). Peers are pinned to two\n"
                  "  non-SMT-sibling CPUs on Linux, QoS USER_INTERACTIVE on macOS.\n"
                  "  Thread create+join: create, first schedule, exit and join of an empty\n"
                  "  thread. All rows: 250 ms warmup, then a fixed 0.5 s window.\n",
    .run        = sys_run,
    .repeatable = true,
};
