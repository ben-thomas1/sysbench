#include "sys/sys.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"
#include "core/timer.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#define HANDOFF_ITERS 10'000

static sb_status_e full_read(int fd, void *buf, size_t len) {
    char  *p    = buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { return SB_ERR_IO; }
        done += (size_t)n;
    }
    return SB_OK;
}

static sb_status_e full_write(int fd, const void *buf, size_t len) {
    const char *p    = buf;
    size_t      done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) { return SB_ERR_IO; }
        done += (size_t)n;
    }
    return SB_OK;
}

/* --- Syscall: getuid() is a real kernel trap on both platforms (getpid() is
 * cached in user space on macOS). --- */

static u64 getuid_work(void *ctx, u32 tid, u64 chunk) {
    (void)ctx;
    (void)tid;
    for (u64 i = 0; i < chunk; i++) {
        uid_t u = getuid();
        __asm__ volatile("" : : "r"(u));
    }
    return chunk;
}

static sb_status_e measure_syscall(f64 *out_ns) {
    sb_par_cfg cfg = {
        .nthreads  = 1,
        .place     = SB_CORE_ANY,
        .window_ns = 500'000'000ULL,
        .chunk     = 1000,
        .fn        = getuid_work,
    };
    sb_par_result r;
    sb_status_e s = sb_par_run(&cfg, &r);
    if (s != SB_OK) { return s; }
    if (r.units_per_sec <= 0) { return SB_ERR_RANGE; }
    *out_ns = 1e9 / r.units_per_sec;
    return SB_OK;
}

/* --- Pipe handoff: two processes ping-pong one byte through two pipes. Each
 * round trip is two handoffs; includes pipe syscalls, sleep and wakeup. --- */

static void close_pair(int p[2]) {
    close(p[0]);
    close(p[1]);
}

static sb_status_e measure_pipe_handoff(f64 *out_ns) {
    signal(SIGPIPE, SIG_IGN);
    int a[2], b[2];
    if (pipe(a) < 0) { return SB_ERR_SYS; }
    if (pipe(b) < 0) {
        close_pair(a);
        return SB_ERR_SYS;
    }

    char  byte = 'x';
    pid_t pid  = fork();
    if (pid < 0) {
        close_pair(a);
        close_pair(b);
        return SB_ERR_SYS;
    }
    if (pid == 0) {
        close(a[1]);
        close(b[0]);
        for (u32 i = 0; i < HANDOFF_ITERS; i++) {
            if (full_read(a[0], &byte, 1) != SB_OK || full_write(b[1], &byte, 1) != SB_OK) { _exit(1); }
        }
        _exit(0);
    }

    close(a[0]);
    close(b[1]);
    sb_status_e s  = SB_OK;
    u64         t0 = sb_timer_now_ns();
    for (u32 i = 0; i < HANDOFF_ITERS && s == SB_OK; i++) {
        s = full_write(a[1], &byte, 1);
        if (s == SB_OK) { s = full_read(b[0], &byte, 1); }
    }
    u64 t1 = sb_timer_now_ns();
    close(a[1]);
    close(b[0]);

    int   status = 0;
    pid_t w;
    do { w = waitpid(pid, &status, 0); } while (w < 0 && errno == EINTR);
    if (s != SB_OK) { return s; }
    if (w < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) { return SB_ERR_SYS; }
    if (t1 <= t0) { return SB_ERR_RANGE; }
    *out_ns = (f64)(t1 - t0) / (2.0 * HANDOFF_ITERS);
    return SB_OK;
}

/* --- Thread create + join of an empty thread --- */

static void *noop_thread(void *arg) { return arg; }

static sb_status_e create_join(u64 n) {
    for (u64 i = 0; i < n; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, noop_thread, NULL) != 0) { return SB_ERR_SYS; }
        pthread_join(t, NULL);
    }
    return SB_OK;
}

static sb_status_e measure_thread_create(f64 *out_us) {
    u64 n  = 1000;
    u64 t0 = sb_timer_now_ns();
    sb_status_e s = create_join(n);
    if (s != SB_OK) { return s; }
    u64 dt = sb_timer_now_ns() - t0;
    if (dt == 0) { return SB_ERR_RANGE; }

    /* Scale to ~2 s, within [1000, 100000] iterations. */
    n = (u64)(2e9 / ((f64)dt / 1000.0));
    if (n < 1000) { n = 1000; }
    if (n > 100'000) { n = 100'000; }

    t0 = sb_timer_now_ns();
    s  = create_join(n);
    if (s != SB_OK) { return s; }
    dt = sb_timer_now_ns() - t0;
    if (dt == 0) { return SB_ERR_RANGE; }
    *out_us = (f64)dt / (f64)n / 1000.0;
    return SB_OK;
}

static sb_status_e sys_run(void) {
    const sb_platform *p = sb_platform_get();
    sb_report_info("OS: %s (%s)", p->os[0] ? p->os : "unknown", p->os_release);

    f64 v = 0;
    sb_status_e s = measure_syscall(&v);
    if (s == SB_OK) { sb_report_value("Syscall (getuid)", v, "ns/call", SB_KIND_MEASURED); }
    else            { sb_report_error("Syscall (getuid)", s); }

    s = measure_pipe_handoff(&v);
    if (s == SB_OK) { sb_report_value("Pipe process handoff", v, "ns", SB_KIND_MEASURED); }
    else            { sb_report_error("Pipe process handoff", s); }

    s = measure_thread_create(&v);
    if (s == SB_OK) { sb_report_value("Thread create+join", v, "us", SB_KIND_MEASURED); }
    else            { sb_report_error("Thread create+join", s); }
    return SB_OK;
}

const sb_section sb_section_sys = {
    .name       = "sys",
    .title      = "System Overhead",
    .help       = "  getuid call time, pipe process handoff (half a round trip), and\n"
                  "  pthread create+join time. Handoff includes pipe and scheduler costs.\n",
    .run        = sys_run,
    .repeatable = true,
};
