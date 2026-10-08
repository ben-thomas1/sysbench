/* SSD / storage: sequential and random direct I/O on one temporary file.
 *
 * The file is created with mkstemp under ./build/ and unlinked immediately, so
 * an interrupted run cannot leak it; all tests use the one open descriptor
 * (pread/pwrite are positional and thread-safe). Page caching is bypassed with
 * F_NOCACHE + F_RDAHEAD=0 (macOS) or O_DIRECT (Linux).
 *
 * "N threads" rows are N threads each issuing one synchronous I/O at a time
 * (QD1 per thread), not asynchronous queue depth.
 *
 * Durable sync is F_FULLFSYNC on macOS (plain fsync only reaches the drive's
 * volatile cache there) and fdatasync on Linux (issues a device cache flush). */
#include "disk/disk.h"
#include "core/platform.h"
#include "core/report.h"
#include "core/thread.h"
#include "core/timer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/mount.h>
#include <sys/param.h>
#else
#include <limits.h>
#endif

#define TEST_DIR      "./build"
#define FILE_BYTES    (2ULL << 30)  /* default working set */
#define FREE_MARGIN   (1ULL << 30)  /* keep this much space free on top of the file */
#define SEQ_WRITE_MAX (1ULL << 30)  /* bytes written by the sequential write test */
#define FILL_BLK      (8ULL << 20)
#define SEQ_BLK       (1ULL << 20)
#define BIG_BLK       (8ULL << 20)
#define RND_BLK       4096ULL
#define STAMP_STRIDE  4096ULL       /* every 4 KiB block gets a unique stamp */
#define SMOKE_ENV     "SB_DISK_SMOKE_MIB"

#if defined(__APPLE__)
#define DURABLE_NAME "F_FULLFSYNC"
#else
#define DURABLE_NAME "fdatasync"
#endif

typedef struct {
    u64  file_bytes;
    u64  read_window_ns;
    u64  write_window_ns;
    u64  sync_window_ns;
    u64  read_warmup_ns;
    u64  write_warmup_ns;
    bool smoke;
} disk_cfg;

/* --- Small helpers ------------------------------------------------------- */

/* splitmix64: fast, good enough for offsets and incompressible fill data. */
static inline u64 rng_next(u64 *s) {
    u64 z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Uniform in [0, n) without division (Lemire). */
static inline u64 rng_below(u64 *s, u64 n) {
    return (u64)(((unsigned __int128)rng_next(s) * n) >> 64);
}

static void rng_fill(u64 *s, u8 *buf, u64 len) {
    for (u64 i = 0; i + 8 <= len; i += 8) {
        u64 v = rng_next(s);
        memcpy(buf + i, &v, 8);
    }
}

static u64 io_align(void) {
    u64 page = sb_platform_get()->page_size;
    return page >= 4096 ? page : 4096;
}

static u8 *alloc_io_buf(u64 len) {
    return SB_ALIGNED_ALLOC((size_t)io_align(), (size_t)len);
}

static sb_status_e io_full(int fd, u8 *buf, u64 len, u64 off, bool write) {
    u64 done = 0;
    while (done < len) {
        ssize_t n = write ? pwrite(fd, buf + done, (size_t)(len - done), (off_t)(off + done))
                          : pread(fd, buf + done, (size_t)(len - done), (off_t)(off + done));
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0 && errno == EINVAL) { return SB_ERR_UNSUPPORTED; }  /* O_DIRECT rejected */
        if (n <= 0) { return SB_ERR_IO; }
        done += (u64)n;
    }
    return SB_OK;
}

/* Data reaches stable storage: F_FULLFSYNC on macOS, fdatasync on Linux. */
static sb_status_e sync_durable(int fd) {
    for (;;) {
#if defined(__APPLE__)
        int r = fcntl(fd, F_FULLFSYNC);
#else
        int r = fdatasync(fd);
#endif
        if (r == 0) { return SB_OK; }
        if (errno == EINTR) { continue; }
        return (errno == ENOTSUP || errno == EINVAL) ? SB_ERR_UNSUPPORTED : SB_ERR_IO;
    }
}

#if defined(__APPLE__)
/* Not durable: fsync hands data to the drive; F_BARRIERFSYNC also orders it. */
static sb_status_e sync_fsync(int fd) {
    while (fsync(fd) != 0) {
        if (errno != EINTR) { return SB_ERR_IO; }
    }
    return SB_OK;
}

static sb_status_e sync_barrier(int fd) {
    while (fcntl(fd, F_BARRIERFSYNC) != 0) {
        if (errno == EINTR) { continue; }
        return (errno == ENOTSUP || errno == EINVAL) ? SB_ERR_UNSUPPORTED : SB_ERR_IO;
    }
    return SB_OK;
}
#endif

/* Write the offset into the first bytes of every 4 KiB block, so no two
 * blocks of the file or of successive writes carry identical data. */
static void stamp(u8 *buf, u64 len, u64 off) {
    for (u64 i = 0; i < len; i += STAMP_STRIDE) {
        u64 v = off + i;
        memcpy(buf + i, &v, 8);
    }
}

/* --- Configuration and filesystem facts ---------------------------------- */

static disk_cfg cfg_get(void) {
    disk_cfg c = {
        .file_bytes      = FILE_BYTES,
        .read_window_ns  = 750'000'000ULL,
        .write_window_ns = 1'000'000'000ULL,
        .sync_window_ns  = 500'000'000ULL,
        .read_warmup_ns  = SB_WARMUP_NS,
        .write_warmup_ns = 50'000'000ULL,
        .smoke           = false,
    };
    /* Smoke tests only: tiny file and windows. The numbers are meaningless. */
    const char *env = getenv(SMOKE_ENV);
    if (env != NULL && env[0] != '\0') {
        u64 mib = strtoull(env, NULL, 10);
        if (mib < 16) { mib = 16; }
        if (mib > 2048) { mib = 2048; }
        c.file_bytes      = (mib << 20) / BIG_BLK * BIG_BLK;
        c.read_window_ns  = 100'000'000ULL;
        c.write_window_ns = 100'000'000ULL;
        c.sync_window_ns  = 100'000'000ULL;
        c.read_warmup_ns  = 20'000'000ULL;
        c.write_warmup_ns = 20'000'000ULL;
        c.smoke           = true;
    }
    return c;
}

static void fs_type(const char *dir, char *out, size_t cap) {
    out[0] = '\0';
#if defined(__APPLE__)
    struct statfs sfs;
    if (statfs(dir, &sfs) == 0) { snprintf(out, cap, "%s", sfs.f_fstypename); }
#else
    /* Longest mount-point prefix of the resolved directory in /proc/mounts. */
    char real[PATH_MAX];
    if (realpath(dir, real) == NULL) { return; }
    FILE *f = fopen("/proc/mounts", "r");
    if (f == NULL) { return; }
    char   line[1024];
    size_t best = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        char dev[256], mnt[512], type[64];
        if (sscanf(line, "%255s %511s %63s", dev, mnt, type) != 3) { continue; }
        size_t ml = strlen(mnt);
        bool   root  = strcmp(mnt, "/") == 0;
        bool   match = root || (strncmp(real, mnt, ml) == 0 && (real[ml] == '\0' || real[ml] == '/'));
        if (match && ml >= best) {
            best = ml;
            snprintf(out, cap, "%s", type);
        }
    }
    fclose(f);
#endif
    if (out[0] == '\0') { snprintf(out, cap, "unknown"); }
}

static u64 fs_free_bytes(const char *dir) {
    struct statvfs v;
    if (statvfs(dir, &v) != 0) { return 0; }
    return (u64)v.f_bavail * (u64)v.f_frsize;
}

/* --- Test file ----------------------------------------------------------- */

/* Bypass the OS page cache on `fd`. */
static sb_status_e set_direct(int fd) {
#if defined(__APPLE__)
    if (fcntl(fd, F_NOCACHE, 1) != 0) { return SB_ERR_SYS; }
    (void)fcntl(fd, F_RDAHEAD, 0);
    return SB_OK;
#else
    int fl = fcntl(fd, F_GETFL);
    if (fl < 0) { return SB_ERR_SYS; }
    if (fcntl(fd, F_SETFL, fl | O_DIRECT) != 0) {
        return errno == EINVAL ? SB_ERR_UNSUPPORTED : SB_ERR_SYS;
    }
    return SB_OK;
#endif
}

/* Create, unlink and fill the file with incompressible data, then make it
 * durable so the tests start with a clean drive write cache. Returns the fd. */
static sb_status_e file_create(u64 bytes, u64 *rng, int *out_fd) {
    char path[] = TEST_DIR "/sb-disk-XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0) { return SB_ERR_IO; }
    (void)unlink(path);  /* space is released when fd closes, even on a crash */

    sb_status_e s   = set_direct(fd);
    u8         *buf = NULL;
    if (s != SB_OK) { goto fail; }
    buf = alloc_io_buf(FILL_BLK);
    if (buf == NULL) {
        s = SB_ERR_NOMEM;
        goto fail;
    }
    rng_fill(rng, buf, FILL_BLK);
    for (u64 off = 0; off < bytes; off += FILL_BLK) {
        stamp(buf, FILL_BLK, off);
        s = io_full(fd, buf, FILL_BLK, off, true);
        if (s != SB_OK) { goto fail; }
    }
    s = sync_durable(fd);
    if (s != SB_OK) { goto fail; }
#if !defined(__APPLE__)
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    SB_ALIGNED_FREE(buf);
    *out_fd = fd;
    return SB_OK;

fail:
    SB_ALIGNED_FREE(buf);
    close(fd);
    return s;
}

/* --- Multithreaded direct I/O on sb_par_run ------------------------------ */

typedef struct {
    u64 rng;
    u8 *buf;
} io_thread;

typedef struct {
    int         fd;
    u64         file_bytes;
    u64         blk;
    u64         nblk;
    bool        write;
    bool        random;
    _Atomic u64 cursor;   /* sequential: next byte offset, shared by all threads */
    atomic_int  status;   /* first failure, as sb_status_e */
    u8         *threads;  /* io_thread per thread, cache-line strided */
    size_t      stride;
} io_ctx;

typedef struct {
    u32  nthreads;
    u64  blk;
    bool write;
    bool random;
    u64  chunk;      /* I/Os per sb_par_run callback */
    u64  warmup_ns;
    u64  window_ns;
} io_spec;

static io_thread *thread_at(io_ctx *c, u32 tid) {
    return (io_thread *)(void *)(c->threads + (size_t)tid * c->stride);
}

/* Sequential: threads take the next block from a shared cursor, so together
 * they form one sequential stream with up to N reads in flight. Random:
 * uniform block-aligned offsets over the whole file. */
static u64 io_work(void *p, u32 tid, u64 chunk) {
    io_ctx *c = p;
    if (atomic_load_explicit(&c->status, memory_order_relaxed) != SB_OK) { return 0; }
    io_thread *t = thread_at(c, tid);
    for (u64 i = 0; i < chunk; i++) {
        u64 off;
        if (c->random) {
            off = rng_below(&t->rng, c->nblk) * c->blk;
        } else {
            off = atomic_fetch_add_explicit(&c->cursor, c->blk, memory_order_relaxed) % c->file_bytes;
        }
        if (c->write) { stamp(t->buf, c->blk, off); }
        sb_status_e s = io_full(c->fd, t->buf, c->blk, off, c->write);
        if (s != SB_OK) {
            atomic_store(&c->status, (int)s);
            return i;
        }
    }
    return chunk;
}

/* Runs `spec` for its window; returns I/Os per second summed over threads. */
static sb_status_e par_io(int fd, u64 file_bytes, const io_spec *spec, u64 *rng, f64 *out_ops) {
    u64 line = sb_platform_get()->cache_line;
    if (line < sizeof(io_thread)) { line = 128; }
    size_t stride = (sizeof(io_thread) + (size_t)line - 1) / (size_t)line * (size_t)line;

    io_ctx c = {
        .fd         = fd,
        .file_bytes = file_bytes,
        .blk        = spec->blk,
        .nblk       = file_bytes / spec->blk,
        .write      = spec->write,
        .random     = spec->random,
        .stride     = stride,
    };
    atomic_init(&c.cursor, rng_below(rng, c.nblk) * spec->blk);
    atomic_init(&c.status, (int)SB_OK);

    c.threads = SB_MALLOC((size_t)spec->nthreads * stride);  /* nthreads <= SB_MAX_CPUS */
    if (c.threads == NULL) { return SB_ERR_NOMEM; }
    memset(c.threads, 0, (size_t)spec->nthreads * stride);
    sb_status_e s = SB_OK;
    u32 nbuf = 0;
    for (; nbuf < spec->nthreads; nbuf++) {
        io_thread *t = thread_at(&c, nbuf);
        t->rng = rng_next(rng);
        t->buf = alloc_io_buf(spec->blk);
        if (t->buf == NULL) {
            s = SB_ERR_NOMEM;
            break;
        }
        rng_fill(&t->rng, t->buf, spec->blk);
    }

    if (s == SB_OK) {
        sb_par_cfg pc = {
            .nthreads  = spec->nthreads,
            .place     = SB_CORE_ANY,
            .warmup_ns = spec->warmup_ns,
            .window_ns = spec->window_ns,
            .chunk     = spec->chunk,
            .fn        = io_work,
            .ctx       = &c,
        };
        sb_par_result r;
        s = sb_par_run(&pc, &r);
        if (s == SB_OK) { s = (sb_status_e)atomic_load(&c.status); }
        if (s == SB_OK && r.units_per_sec <= 0) { s = SB_ERR_RANGE; }
        if (s == SB_OK) { *out_ops = r.units_per_sec; }
    }

    for (u32 i = 0; i < nbuf; i++) { SB_ALIGNED_FREE(thread_at(&c, i)->buf); }
    SB_FREE(c.threads);
    return s;
}

/* --- Single-threaded tests ----------------------------------------------- */

/* QD1 1 MiB writes from offset 0, then one durable sync; the sync is timed. */
static sb_status_e seq_write_durable(int fd, u64 bytes, u64 *rng, f64 *out_gbs) {
    u8 *buf = alloc_io_buf(SEQ_BLK);
    if (buf == NULL) { return SB_ERR_NOMEM; }
    rng_fill(rng, buf, SEQ_BLK);

    sb_status_e s  = SB_OK;
    u64         t0 = sb_timer_now_ns();
    for (u64 off = 0; off < bytes && s == SB_OK; off += SEQ_BLK) {
        stamp(buf, SEQ_BLK, off ^ 0x5a5a000000000000ULL);
        s = io_full(fd, buf, SEQ_BLK, off, true);
    }
    if (s == SB_OK) { s = sync_durable(fd); }
    u64 t1 = sb_timer_now_ns();
    SB_ALIGNED_FREE(buf);
    if (s != SB_OK) { return s; }
    if (t1 <= t0) { return SB_ERR_RANGE; }
    *out_gbs = (f64)bytes / (f64)(t1 - t0);
    return SB_OK;
}

typedef sb_status_e (*sync_fn)(int fd);

/* 4 KiB write at a random offset followed by `sync` each, for `window_ns`
 * (at least 10 ops). Returns the mean time per write+sync. */
static sb_status_e sync_latency(int fd, u64 file_bytes, sync_fn sync, u64 window_ns, u64 *rng, f64 *out_us) {
    u8 *buf = alloc_io_buf(RND_BLK);
    if (buf == NULL) { return SB_ERR_NOMEM; }
    rng_fill(rng, buf, RND_BLK);

    u64         nblk = file_bytes / RND_BLK;
    u64         n    = 0;
    sb_status_e s    = SB_OK;
    u64         t0   = sb_timer_now_ns();
    u64         t1   = t0;
    while (s == SB_OK && (n < 10 || t1 - t0 < window_ns)) {
        u64 off = rng_below(rng, nblk) * RND_BLK;
        stamp(buf, RND_BLK, off ^ n);
        s = io_full(fd, buf, RND_BLK, off, true);
        if (s == SB_OK) { s = sync(fd); }
        n++;
        t1 = sb_timer_now_ns();
    }
    SB_ALIGNED_FREE(buf);
    if (s != SB_OK) { return s; }
    *out_us = (f64)(t1 - t0) / (f64)n / 1000.0;
    return SB_OK;
}

/* --- Section ------------------------------------------------------------- */

static void report(const char *test, sb_status_e s, f64 v, const char *unit) {
    if (s == SB_OK)                  { sb_report_value(test, v, unit, SB_KIND_MEASURED); }
    else if (s == SB_ERR_UNSUPPORTED) { sb_report_skip(test, "not supported by this filesystem"); }
    else                             { sb_report_error(test, s); }
}

static void run_sequential(int fd, const disk_cfg *cfg, u64 *rng) {
    sb_report_group("Sequential");
    static const struct { const char *name; u32 nthreads; u64 blk; } reads[] = {
        { "Seq read 1 MiB, 1 thread (QD1)",  1, SEQ_BLK },
        { "Seq read 8 MiB, 1 thread (QD1)",  1, BIG_BLK },
        { "Seq read 1 MiB, 4 threads x QD1", 4, SEQ_BLK },
    };
    for (size_t i = 0; i < SB_ARRAY_LEN(reads); i++) {
        io_spec spec = {
            .nthreads  = reads[i].nthreads,
            .blk       = reads[i].blk,
            .chunk     = 1,
            .warmup_ns = cfg->read_warmup_ns,
            .window_ns = cfg->read_window_ns,
        };
        f64 ops = 0;
        sb_status_e s = par_io(fd, cfg->file_bytes, &spec, rng, &ops);
        report(reads[i].name, s, ops * (f64)reads[i].blk / 1e9, "GB/s");
    }
}

static void run_random_read(int fd, const disk_cfg *cfg, u64 *rng) {
    sb_report_group("Random 4 KiB read (N threads, 1 I/O in flight each)");
    static const struct { const char *name; u32 nthreads; } rows[] = {
        { "Rand 4K read, 1 thread",   1 },
        { "Rand 4K read, 2 threads",  2 },
        { "Rand 4K read, 4 threads",  4 },
        { "Rand 4K read, 8 threads",  8 },
        { "Rand 4K read, 16 threads", 16 },
        { "Rand 4K read, 32 threads", 32 },
    };
    for (size_t i = 0; i < SB_ARRAY_LEN(rows); i++) {
        io_spec spec = {
            .nthreads  = rows[i].nthreads,
            .blk       = RND_BLK,
            .random    = true,
            .chunk     = 8,
            .warmup_ns = cfg->read_warmup_ns,
            .window_ns = cfg->read_window_ns,
        };
        f64 ops = 0;
        sb_status_e s = par_io(fd, cfg->file_bytes, &spec, rng, &ops);
        report(rows[i].name, s, ops, "IOPS");
        if (rows[i].nthreads == 1) {
            report("Rand 4K read QD1 mean latency", s, s == SB_OK ? 1e6 / ops : 0, "us");
        }
    }
}

static void run_writes(int fd, const disk_cfg *cfg, u64 *rng) {
    sb_report_group("Write (durable: " DURABLE_NAME " included in the time)");
    u64 bytes = cfg->file_bytes < SEQ_WRITE_MAX ? cfg->file_bytes : SEQ_WRITE_MAX;
    f64 gbs   = 0;
    sb_status_e s = seq_write_durable(fd, bytes, rng, &gbs);
    report("Seq write 1 MiB QD1, durable", s, gbs, "GB/s");

    /* Random writes for the window, then one durable sync. The sync also flushes
     * the short warmup and the post-window tail, so the rate is slightly low.
     * Individual writes are not durable; per-write durable cost is the sync
     * latency group below. */
    sb_report_info("Rand write+flush: 4 KiB writes for the window, then one durable flush;");
    sb_report_info("  the rate includes the flush. Single writes are not made durable.");
    static const struct { const char *name; u32 nthreads; } rows[] = {
        { "Rand 4K write+flush, 1 thread",   1 },
        { "Rand 4K write+flush, 4 threads",  4 },
        { "Rand 4K write+flush, 16 threads", 16 },
    };
    for (size_t i = 0; i < SB_ARRAY_LEN(rows); i++) {
        io_spec spec = {
            .nthreads  = rows[i].nthreads,
            .blk       = RND_BLK,
            .write     = true,
            .random    = true,
            .chunk     = 4,
            .warmup_ns = cfg->write_warmup_ns,
            .window_ns = cfg->write_window_ns,
        };
        f64 ops = 0;
        s = par_io(fd, cfg->file_bytes, &spec, rng, &ops);
        u64 t0 = sb_timer_now_ns();
        if (s == SB_OK) { s = sync_durable(fd); }
        f64 sync_s = (f64)(sb_timer_now_ns() - t0) / 1e9;
        f64 win_s  = (f64)cfg->write_window_ns / 1e9;
        report(rows[i].name, s, ops * win_s / (win_s + sync_s), "IOPS");
    }
}

static void run_sync_latency(int fd, const disk_cfg *cfg, u64 *rng) {
    sb_report_group("Sync latency (4 KiB write + sync, random offset)");
    static const struct { const char *name; sync_fn fn; } rows[] = {
#if defined(__APPLE__)
        { "4K write+F_FULLFSYNC (durable)", sync_durable },
        { "4K write+barrier (NOT durable)", sync_barrier },
        { "4K write+fsync (NOT durable)",   sync_fsync },
#else
        { "4K write+fdatasync (durable)",   sync_durable },
#endif
    };
    for (size_t i = 0; i < SB_ARRAY_LEN(rows); i++) {
        f64 us = 0;
        sb_status_e s = sync_latency(fd, cfg->file_bytes, rows[i].fn, cfg->sync_window_ns, rng, &us);
        report(rows[i].name, s, us, "us");
    }
}

static sb_status_e disk_run(void) {
    disk_cfg cfg = cfg_get();
    char     size[32];
    char     type[64];

    if (mkdir(TEST_DIR, 0700) != 0 && errno != EEXIST) {
        sb_report_error("Create " TEST_DIR, SB_ERR_IO);
        return SB_OK;
    }
    fs_type(TEST_DIR, type, sizeof(type));
    sb_fmt_size(cfg.file_bytes, size, sizeof(size));
#if defined(__APPLE__)
    sb_report_info("File: %s, unlinked temp file in " TEST_DIR " (%s); F_NOCACHE, read-ahead off", size, type);
    sb_report_info("Durable = F_FULLFSYNC (drive cache flushed). macOS fsync and F_BARRIERFSYNC");
    sb_report_info("only hand data to the drive's volatile cache: NOT durable.");
#else
    sb_report_info("File: %s, unlinked temp file in " TEST_DIR " (%s); O_DIRECT", size, type);
    sb_report_info("Durable = fdatasync on an O_DIRECT fd (device write cache flushed).");
    if (strcmp(type, "tmpfs") == 0 || strcmp(type, "ramfs") == 0) {
        sb_report_info("WARNING: %s is RAM-backed; these are memory speeds, not storage.", type);
    }
#endif
    sb_report_info("\"N threads\" = N threads, each with one synchronous I/O in flight (not async QD).");
    if (cfg.smoke) { sb_report_info("SMOKE MODE (" SMOKE_ENV "): tiny file and windows, numbers are meaningless."); }

    u64 need = cfg.file_bytes + FREE_MARGIN;
    if (fs_free_bytes(TEST_DIR) < need) {
        sb_report_skip("Storage tests", "not enough free space in " TEST_DIR);
        return SB_OK;
    }

    u64 rng = sb_timer_now_ns() ^ ((u64)getpid() << 32);
    int fd  = -1;
    sb_status_e s = file_create(cfg.file_bytes, &rng, &fd);
    if (s == SB_ERR_UNSUPPORTED) {
        sb_report_skip("Storage tests", "direct I/O unsupported on this fs");
        return SB_OK;
    }
    if (s != SB_OK) {
        sb_report_error("Create test file", s);
        return SB_OK;
    }

    /* Order: reads first (on the freshly created file), then writes. */
    run_sequential(fd, &cfg, &rng);
    run_random_read(fd, &cfg, &rng);
    run_writes(fd, &cfg, &rng);
    run_sync_latency(fd, &cfg, &rng);
    close(fd);
    return SB_OK;
}

const sb_section sb_section_disk = {
    .name       = "disk",
    .title      = "SSD / Storage",
    .help       =
        "  One 2 GiB file under ./build/, created fresh each pass and unlinked at once\n"
        "  (no leak if interrupted). F_NOCACHE (macOS) / O_DIRECT (Linux) bypasses the\n"
        "  page cache; drive caches still participate. Reads: sequential 1 MiB and\n"
        "  8 MiB with 1 thread, 1 MiB with 4 threads; random 4 KiB with 1-32 threads.\n"
        "  \"N threads\" means N synchronous I/Os in flight, not async queue depth.\n"
        "  Writes are durable (F_FULLFSYNC / fdatasync, timed): 1 GiB sequential,\n"
        "  random 4 KiB with 1/4/16 threads followed by one flush. Sync latency:\n"
        "  write + sync per 4 KiB; macOS fsync and F_BARRIERFSYNC rows are labelled\n"
        "  NOT durable.\n"
        "  About 3.2 GiB is written per pass; needs 3 GiB free.\n",
    .run        = disk_run,
    .repeatable = true,
};
