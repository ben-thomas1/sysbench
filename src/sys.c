#include "bench.h"
#include "timer.h"

#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>

#include <sys/utsname.h>

static int full_read(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0)
            return -1;
        done += (size_t)n;
    }
    return 0;
}

static int full_write(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0)
            return -1;
        done += (size_t)n;
    }
    return 0;
}

/* --- Syscall overhead ---
   Use getuid() on both platforms — it's always a real kernel trap.
   getpid() is cached (thread-local) on macOS, so it doesn't measure
   actual syscall overhead. */

static double measure_syscall_ns(void) {
    /* Calibrate */
    size_t iters = 10000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        getuid();
    }
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0) / iters;
    if (ns_per <= 0) return -1;
    iters = (size_t)(200000000.0 / ns_per);
    if (iters < 10000) iters = 10000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        getuid();
    }
    uint64_t t1 = timer_ns();
    return (double)(t1 - t0) / iters;
}

/* --- Context switch (pipe ping-pong between processes) --- */

static double measure_ctx_switch_ns(void) {
    signal(SIGPIPE, SIG_IGN);

    int pipe_a[2], pipe_b[2];
    if (pipe(pipe_a) < 0)
        return -1;
    if (pipe(pipe_b) < 0) {
        close(pipe_a[0]);
        close(pipe_a[1]);
        return -1;
    }

    size_t iters = 10000;
    char byte = 'x';

    pid_t pid = fork();
    if (pid < 0) {
        close(pipe_a[0]);
        close(pipe_a[1]);
        close(pipe_b[0]);
        close(pipe_b[1]);
        return -1;
    }

    if (pid == 0) {
        /* Child: read from pipe_a, write to pipe_b */
        close(pipe_a[1]);
        close(pipe_b[0]);
        for (size_t i = 0; i < iters; i++) {
            if (full_read(pipe_a[0], &byte, 1) < 0 ||
                full_write(pipe_b[1], &byte, 1) < 0) {
                close(pipe_a[0]);
                close(pipe_b[1]);
                _exit(1);
            }
        }
        close(pipe_a[0]);
        close(pipe_b[1]);
        _exit(0);
    }

    /* Parent: write to pipe_a, read from pipe_b */
    close(pipe_a[0]);
    close(pipe_b[1]);

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        if (full_write(pipe_a[1], &byte, 1) < 0 ||
            full_read(pipe_b[0], &byte, 1) < 0) {
            close(pipe_a[1]);
            close(pipe_b[0]);
            waitpid(pid, NULL, 0);
            return -1;
        }
    }
    uint64_t t1 = timer_ns();

    close(pipe_a[1]);
    close(pipe_b[0]);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited < 0) return -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || t1 <= t0)
        return -1;

    /* Each iteration = 2 context switches (parent->child, child->parent) */
    return (double)(t1 - t0) / (iters * 2);
}

/* --- Thread creation --- */

static void *noop_thread(void *arg) {
    (void)arg;
    return NULL;
}

static double measure_thread_create_us(void) {
    /* Calibrate */
    size_t iters = 1000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, noop_thread, NULL) != 0)
            return -1;
        pthread_join(t, NULL);
    }
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0) / iters;
    if (ns_per <= 0) return -1;
    iters = (size_t)(2000000000.0 / ns_per);
    if (iters < 1000) iters = 1000;
    if (iters > 100000) iters = 100000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, noop_thread, NULL) != 0)
            return -1;
        pthread_join(t, NULL);
    }
    uint64_t t1 = timer_ns();
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / iters / 1000.0;
}

/* --- Entry point --- */

void bench_sys(void) {
    struct utsname un = {0};
    if (uname(&un) < 0) perror("uname");

    printf("=== System Overhead ===\n");
    printf("  OS: %s %s\n", un.sysname, un.release);

    printf("%-24s %14s\n", "Test", "Result");
    printf("%-24s %14s\n", "----", "------");

    double sc = measure_syscall_ns();
    if (sc > 0)
        printf("%-24s %10.1f ns/call\n", "Syscall (getuid)", sc);
    fflush(stdout);

    double cs = measure_ctx_switch_ns();
    if (cs > 0)
        printf("%-24s %10.1f ns/handoff\n", "Pipe process handoff", cs);
    fflush(stdout);

    double tc = measure_thread_create_us();
    if (tc > 0)
        printf("%-24s %10.1f us/thread\n", "Thread create+join", tc);
    fflush(stdout);
}
