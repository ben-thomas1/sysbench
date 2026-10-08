#include "sys/ipc.h"
#include "core/platform.h"
#include "core/thread.h"
#include "core/timer.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#if !defined(__APPLE__)
#include <sched.h>
#endif

#define PING_BYTE 'p'
#define STOP_BYTE 'q'
#define RTT_BATCH 8      /* round trips between clock reads */
#define POLL_MASK 0xFFFF /* busy-poll: check the timeout every 64 Ki empty reads */

static struct {
    bool init;
    bool can_spin;
    bool pin;
    u16  cpu[2];
    char desc[96];
} peers;

/* --- Peer placement --- */

#if !defined(__APPLE__)
/* Parse a sysfs CPU list ("0,8" or "0-1,4-5") into out[]; returns the count. */
static u32 parse_cpu_list(const char *s, u32 *out, u32 cap) {
    u32 n = 0;
    while (*s != '\0' && *s != '\n' && n < cap) {
        char *end = NULL;
        unsigned long lo = strtoul(s, &end, 10);
        if (end == s) { break; }
        unsigned long hi = lo;
        s = end;
        if (*s == '-') {
            hi = strtoul(s + 1, &end, 10);
            s  = end;
        }
        for (unsigned long c = lo; c <= hi && n < cap; c++) { out[n++] = (u32)c; }
        if (*s == ',') { s++; }
    }
    return n;
}

static u32 read_siblings(u32 cpu, u32 *out, u32 cap) {
    char path[96];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", cpu);
    FILE *f = fopen(path, "r");
    if (f == NULL) { return 0; }
    char line[256];
    u32  n = 0;
    if (fgets(line, sizeof(line), f) != NULL) { n = parse_cpu_list(line, out, cap); }
    fclose(f);
    return n;
}
#endif

void sb_ipc_peers_init(void) {
    if (peers.init) { return; }
    peers.init = true;
    const sb_platform *p = sb_platform_get();
    peers.can_spin = p->ncpu >= 2;
#if defined(__APPLE__)
    snprintf(peers.desc, sizeof(peers.desc), "QoS USER_INTERACTIVE, placed by the scheduler (no pinning on macOS)");
#else
    const u16 *ids = p->nperf > 0 ? p->perf_ids : p->cpu_ids;
    u32        n   = p->nperf > 0 ? p->nperf : p->ncpu;
    if (n == 0) {
        snprintf(peers.desc, sizeof(peers.desc), "unpinned");
        return;
    }
    peers.pin    = true;
    peers.cpu[0] = ids[0];
    peers.cpu[1] = n > 1 ? ids[1] : ids[0];
    u32  sib[16];
    u32  nsib    = read_siblings(ids[0], sib, SB_ARRAY_LEN(sib));
    bool smt     = false;
    bool found   = false;
    for (u32 i = 1; i < n && !found; i++) {
        bool is_sib = false;
        for (u32 k = 0; k < nsib; k++) {
            if (sib[k] == ids[i]) { is_sib = true; }
        }
        if (!is_sib) {
            peers.cpu[1] = ids[i];
            found        = true;
        }
    }
    if (!found && n > 1) { smt = true; }
    if (n == 1) {
        snprintf(peers.desc, sizeof(peers.desc), "both on CPU %u (only one CPU allowed)", (unsigned)ids[0]);
        return;
    }
    snprintf(peers.desc, sizeof(peers.desc), "CPUs %u and %u (pinned%s)", (unsigned)peers.cpu[0],
             (unsigned)peers.cpu[1], smt ? ", SMT siblings" : "");
#endif
}

const char *sb_ipc_peers_desc(void) { return peers.desc; }

bool sb_ipc_can_spin(void) { return peers.can_spin; }

void sb_ipc_place(u32 side) {
#if defined(__APPLE__)
    (void)side;
    (void)sb_thread_place_self(SB_CORE_PERF, 0);
#else
    if (!peers.pin) { return; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(peers.cpu[side & 1], &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

sb_status_e sb_ipc_set_nonblock(int fd) {
    int fl = fcntl(fd, F_GETFL);
    if (fl < 0 || fcntl(fd, F_SETFL, fl | O_NONBLOCK) < 0) { return SB_ERR_SYS; }
    return SB_OK;
}

/* --- One-byte transfers --- */

/* EAGAIN means "spin again" on an O_NONBLOCK fd (it can't happen on a blocking
 * one). Busy-polling checks the clock rarely so the loop stays tight; a peer
 * that dies or fails closes/shuts down its end, so reads see EOF. */
static sb_status_e read_byte(int fd, char *c, bool spin) {
    u32 polls   = 0;
    u64 t_start = 0;
    for (;;) {
        ssize_t n = read(fd, c, 1);
        if (n == 1) { return SB_OK; }
        if (n == 0) { return SB_ERR_IO; }
        if (errno == EINTR) { continue; }
        if (errno != EAGAIN && errno != EWOULDBLOCK) { return SB_ERR_IO; }
        if (!spin) { return SB_ERR_IO; }
        if ((++polls & POLL_MASK) == 0) {
            u64 t = sb_timer_now_ns();
            if (t_start == 0) { t_start = t; }
            else if (t - t_start > SB_IPC_TIMEOUT_NS) { return SB_ERR_TIMEOUT; }
        }
    }
}

static sb_status_e write_byte(int fd, char c) {
    u64 t_start = 0;
    for (;;) {
        ssize_t n = write(fd, &c, 1);
        if (n == 1) { return SB_OK; }
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            u64 t = sb_timer_now_ns();
            if (t_start == 0) { t_start = t; }
            else if (t - t_start > SB_IPC_TIMEOUT_NS) { return SB_ERR_TIMEOUT; }
            continue;
        }
        return SB_ERR_IO;
    }
}

sb_status_e sb_ipc_echo(int rfd, int wfd, bool spin) {
    for (;;) {
        char        c = 0;
        sb_status_e s = read_byte(rfd, &c, spin);
        if (s != SB_OK) { return s; }
        s = write_byte(wfd, c);
        if (s != SB_OK) { return s; }
        if (c == STOP_BYTE) { return SB_OK; }
    }
}

/* Round trips for at least `dur_ns`, in batches of RTT_BATCH. */
static sb_status_e rtt_loop(int rfd, int wfd, bool spin, u64 dur_ns, u64 *out_n, u64 *out_ns) {
    u64 n  = 0;
    u64 t0 = sb_timer_now_ns();
    u64 t  = t0;
    do {
        for (u32 i = 0; i < RTT_BATCH; i++) {
            char        c = PING_BYTE;
            sb_status_e s = write_byte(wfd, c);
            if (s == SB_OK) { s = read_byte(rfd, &c, spin); }
            if (s != SB_OK) { return s; }
            if (c != PING_BYTE) { return SB_ERR_IO; }
        }
        n += RTT_BATCH;
        t  = sb_timer_now_ns();
    } while (t - t0 < dur_ns);
    *out_n  = n;
    *out_ns = t - t0;
    return SB_OK;
}

sb_status_e sb_ipc_drive(int rfd, int wfd, bool spin, u64 window_ns, f64 *out_rtt_ns) {
    u64 n  = 0;
    u64 ns = 0;
    sb_status_e s = rtt_loop(rfd, wfd, spin, SB_WARMUP_NS, &n, &ns);
    if (s == SB_OK) { s = rtt_loop(rfd, wfd, spin, window_ns, &n, &ns); }
    if (s != SB_OK) {
        (void)shutdown(wfd, SHUT_RDWR); /* sockets: wake the echo with EOF */
        return s;
    }
    char c = STOP_BYTE;
    s = write_byte(wfd, c);
    if (s == SB_OK) { s = read_byte(rfd, &c, spin); }
    if (s != SB_OK) { return s; }
    if (n == 0 || ns == 0) { return SB_ERR_RANGE; }
    *out_rtt_ns = (f64)ns / (f64)n;
    return SB_OK;
}

/* --- Driver + echo on two threads --- */

typedef struct {
    int         rfd;
    int         wfd;
    bool        spin;
    sb_status_e status;
    f64         rtt_ns;
} pp_arg;

static void *echo_thread(void *p) {
    pp_arg *a = p;
    sb_ipc_place(1);
    a->status = sb_ipc_echo(a->rfd, a->wfd, a->spin);
    if (a->status != SB_OK) { (void)shutdown(a->wfd, SHUT_RDWR); }
    return NULL;
}

static void *drive_thread(void *p) {
    pp_arg *a = p;
    sb_ipc_place(0);
    a->status = sb_ipc_drive(a->rfd, a->wfd, a->spin, SB_IPC_WINDOW_NS, &a->rtt_ns);
    return NULL;
}

sb_status_e sb_ipc_pingpong_threads(int a_rfd, int a_wfd, int b_rfd, int b_wfd, bool spin, f64 *out_rtt_ns) {
    pp_arg    da = { .rfd = a_rfd, .wfd = a_wfd, .spin = spin, .status = SB_ERR_SYS };
    pp_arg    ea = { .rfd = b_rfd, .wfd = b_wfd, .spin = spin, .status = SB_ERR_SYS };
    pthread_t te;
    pthread_t td;
    if (pthread_create(&te, NULL, echo_thread, &ea) != 0) { return SB_ERR_SYS; }
    if (pthread_create(&td, NULL, drive_thread, &da) != 0) {
        (void)shutdown(a_wfd, SHUT_RDWR);
        pthread_join(te, NULL);
        return SB_ERR_SYS;
    }
    pthread_join(td, NULL);
    pthread_join(te, NULL);
    if (da.status != SB_OK) { return da.status; }
    if (ea.status != SB_OK) { return ea.status; }
    *out_rtt_ns = da.rtt_ns;
    return SB_OK;
}

/* --- Security software detection --- */

u32 sb_ipc_active_exts(const char *category, char *out, size_t cap) {
    if (cap > 0) { out[0] = '\0'; }
#if defined(__APPLE__)
    if (access("/usr/bin/systemextensionsctl", X_OK) != 0) { return 0; }
    FILE *f = popen("/usr/bin/systemextensionsctl list 2>/dev/null", "r");
    if (f == NULL) { return 0; }
    /* Lines: "--- com.apple.system_extension.<category> (...)" headers, then
     * "enabled\tactive\tteamID\tbundleID (version)\tname\t[state]" rows. */
    char   line[512];
    bool   in_cat = false;
    u32    count  = 0;
    size_t len    = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "--- ", 4) == 0) {
            in_cat = strstr(line, category) != NULL;
            continue;
        }
        if (!in_cat || strstr(line, "[activated enabled]") == NULL) { continue; }
        char *field = line;
        for (u32 i = 0; i < 4 && field != NULL; i++) {
            field = strchr(field, '\t');
            if (field != NULL) { field++; }
        }
        if (field == NULL) { continue; }
        char *end = strchr(field, '\t');
        if (end != NULL) { *end = '\0'; }
        int w = snprintf(out + len, cap - len, "%s%s", count ? ", " : "", field);
        if (w > 0 && (size_t)w < cap - len) { len += (size_t)w; }
        else if (w > 0)                      { len = cap - 1; } /* truncated: stop appending */
        count++;
    }
    pclose(f);
    return count;
#else
    (void)category;
    return 0;
#endif
}
