#include "bench.h"
#include "sysinfo.h"
#include "timer.h"
#include "platform_io.h"

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <pthread.h>

#define TEST_DIR "./build"
static char test_file[] = TEST_DIR "/bench-disktest-XXXXXX";
#define FILE_SIZE (2ULL * 1024 * 1024 * 1024) /* 2 GiB working set */
#define SEQ_BLOCK (1024 * 1024)           /* 1 MiB */
#define RND_BLOCK 4096                    /* 4 KiB */
#define TARGET_NS 2000000000ULL           /* 2 seconds */

static void fill_random(void *buf, size_t len) {
    uint32_t *p = (uint32_t *)buf;
    for (size_t i = 0; i < len / 4; i++)
        p[i] = (uint32_t)rand();
}

static int ensure_test_dir(void) {
    if (mkdir(TEST_DIR, 0700) == 0 || errno == EEXIST)
        return 0;
    perror("mkdir");
    return -1;
}

static int full_pread(int fd, void *buf, size_t len, off_t off) {
    char *p = (char *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, p + done, len - done, off + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("pread");
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "pread: unexpected EOF\n");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int full_pwrite(int fd, const void *buf, size_t len, off_t off) {
    const char *p = (const char *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, p + done, len - done, off + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("pwrite");
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "pwrite: wrote 0 bytes\n");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int checked_fsync(int fd) {
    while (fsync(fd) < 0) {
        if (errno == EINTR) continue;
        perror("fsync");
        return -1;
    }
    return 0;
}

static int create_test_file(void *buf) {
    int fd = mkstemp(test_file);
    if (fd < 0) { perror("mkstemp"); return -1; }
    if (disable_write_cache(fd) < 0) {
        perror("F_NOCACHE");
        close(fd);
        unlink(test_file);
        return -1;
    }

    size_t written = 0;
    while (written < FILE_SIZE) {
        size_t chunk = SEQ_BLOCK;
        if (chunk > FILE_SIZE - written) chunk = FILE_SIZE - written;
        fill_random(buf, chunk);
        if (full_pwrite(fd, buf, chunk, (off_t)written) < 0) {
            close(fd);
            unlink(test_file);
            return -1;
        }
        written += chunk;
    }
    if (checked_fsync(fd) < 0) { close(fd); unlink(test_file); return -1; }
    drop_page_cache(fd);
    close(fd);
    return 0;
}

static int open_nocache(int flags) {
    int fd = open_direct(test_file, flags);
    if (fd < 0) { perror("open"); return -1; }
    return fd;
}

static double bench_seq_read(void *buf) {
    int fd = open_nocache(O_RDONLY);
    if (fd < 0) return -1;

    /* Single pass after file creation; the drive's internal cache may be warm. */
    uint64_t t0 = timer_ns();
    for (size_t off = 0; off < FILE_SIZE; off += SEQ_BLOCK) {
        if (full_pread(fd, buf, SEQ_BLOCK, (off_t)off) < 0) {
            close(fd);
            return -1;
        }
    }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return -1;
    return (double)FILE_SIZE / ((double)(t1 - t0) / 1e9) / 1e9;
}

static double bench_seq_write(void *buf) {
    int fd = open_nocache(O_RDWR);
    if (fd < 0) return -1;

    fill_random(buf, SEQ_BLOCK);

    /* Calibrate passes */
    uint64_t tc0 = timer_ns();
    for (size_t off = 0; off < FILE_SIZE; off += SEQ_BLOCK) {
        if (full_pwrite(fd, buf, SEQ_BLOCK, (off_t)off) < 0) {
            close(fd);
            return -1;
        }
    }
    if (checked_fsync(fd) < 0) { close(fd); return -1; }
    uint64_t tc1 = timer_ns();
    uint64_t elapsed = tc1 - tc0;
    size_t passes = elapsed > 0 ? (size_t)(TARGET_NS / elapsed) + 1 : 2;
    if (passes < 2) passes = 2;

    uint64_t t0 = timer_ns();
    for (size_t p = 0; p < passes; p++) {
        for (size_t off = 0; off < FILE_SIZE; off += SEQ_BLOCK) {
            if (full_pwrite(fd, buf, SEQ_BLOCK, (off_t)off) < 0) {
                close(fd);
                return -1;
            }
        }
        if (checked_fsync(fd) < 0) { close(fd); return -1; }
    }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return -1;
    return (double)FILE_SIZE * passes / ((double)(t1 - t0) / 1e9) / 1e9;
}

static void bench_rand_read(void *buf, double *out_iops, double *out_lat_us) {
    *out_iops = -1;
    *out_lat_us = -1;
    int fd = open_nocache(O_RDONLY);
    if (fd < 0) return;

    size_t max_off = FILE_SIZE / RND_BLOCK;

    /* Warmup + calibrate */
    size_t ops = 1000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pread(fd, buf, RND_BLOCK, off) < 0) {
            close(fd);
            return;
        }
    }
    uint64_t tc1 = timer_ns();
    double ns_per_op = (double)(tc1 - tc0) / ops;
    if (ns_per_op <= 0) { close(fd); return; }
    ops = (size_t)(TARGET_NS / ns_per_op);
    if (ops < 1000) ops = 1000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pread(fd, buf, RND_BLOCK, off) < 0) {
            close(fd);
            return;
        }
    }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return;
    double elapsed_s = (double)(t1 - t0) / 1e9;
    *out_iops = ops / elapsed_s;
    *out_lat_us = (double)(t1 - t0) / ops / 1000.0;
}

static void bench_rand_write(void *buf, double *out_iops, double *out_lat_us) {
    *out_iops = -1;
    *out_lat_us = -1;
    int fd = open_nocache(O_RDWR);
    if (fd < 0) return;

    fill_random(buf, RND_BLOCK);
    size_t max_off = FILE_SIZE / RND_BLOCK;

    /* Warmup + calibrate */
    size_t ops = 1000;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pwrite(fd, buf, RND_BLOCK, off) < 0) {
            close(fd);
            return;
        }
    }
    if (checked_fsync(fd) < 0) { close(fd); return; }
    uint64_t tc1 = timer_ns();
    double ns_per_op = (double)(tc1 - tc0) / ops;
    if (ns_per_op <= 0) { close(fd); return; }
    ops = (size_t)(TARGET_NS / ns_per_op);
    if (ops < 1000) ops = 1000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pwrite(fd, buf, RND_BLOCK, off) < 0) {
            close(fd);
            return;
        }
    }
    if (checked_fsync(fd) < 0) { close(fd); return; }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return;
    double elapsed_s = (double)(t1 - t0) / 1e9;
    *out_iops = ops / elapsed_s;
    *out_lat_us = (double)(t1 - t0) / ops / 1000.0;
}

static void bench_rand_write_fsync(void *buf, double *out_iops, double *out_lat_us) {
    *out_iops = -1;
    *out_lat_us = -1;
    int fd = open_nocache(O_RDWR);
    if (fd < 0) return;

    fill_random(buf, RND_BLOCK);
    size_t max_off = FILE_SIZE / RND_BLOCK;

    /* Calibrate — fsync per write is slow, start with fewer ops */
    size_t ops = 100;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pwrite(fd, buf, RND_BLOCK, off) < 0 ||
            checked_fsync(fd) < 0) {
            close(fd);
            return;
        }
    }
    uint64_t tc1 = timer_ns();
    double ns_per_op = (double)(tc1 - tc0) / ops;
    if (ns_per_op <= 0) { close(fd); return; }
    ops = (size_t)(TARGET_NS / ns_per_op);
    if (ops < 100) ops = 100;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < ops; i++) {
        off_t off = (off_t)((rand() % max_off) * RND_BLOCK);
        if (full_pwrite(fd, buf, RND_BLOCK, off) < 0 ||
            checked_fsync(fd) < 0) {
            close(fd);
            return;
        }
    }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return;
    double elapsed_s = (double)(t1 - t0) / 1e9;
    *out_iops = ops / elapsed_s;
    *out_lat_us = (double)(t1 - t0) / ops / 1000.0;
}

/* --- QD scaling (multi-threaded random 4K reads) --- */

struct qd_arg {
    size_t ops;
    uint64_t elapsed_ns;
    unsigned int seed;
    int ok;
};

static void *qd_worker(void *arg) {
    struct qd_arg *a = (struct qd_arg *)arg;
    a->ok = 0;
    int fd = open_nocache(O_RDONLY);
    if (fd < 0) { a->elapsed_ns = 0; return NULL; }

    void *buf;
    if (posix_memalign(&buf, 4096, RND_BLOCK) != 0) {
        a->elapsed_ns = 0;
        close(fd);
        return NULL;
    }
    size_t max_off = FILE_SIZE / RND_BLOCK;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < a->ops; i++) {
        off_t off = (off_t)(((size_t)rand_r(&a->seed) % max_off) * RND_BLOCK);
        if (full_pread(fd, buf, RND_BLOCK, off) < 0) {
            free(buf);
            close(fd);
            return NULL;
        }
    }
    uint64_t t1 = timer_ns();

    a->elapsed_ns = t1 - t0;
    a->ok = t1 > t0;
    free(buf);
    close(fd);
    return NULL;
}

static double bench_qd_scaling(void *buf, int nthreads) {
    (void)buf;
    size_t ops_per_thread = 5000;

    struct qd_arg args[16];
    pthread_t threads[16];
    int nt = nthreads > 16 ? 16 : nthreads;

    for (int t = 0; t < nt; t++) {
        args[t].ops = ops_per_thread;
        args[t].seed = (unsigned int)(42 + t);
        args[t].elapsed_ns = 0;
        args[t].ok = 0;
    }
    uint64_t t0 = timer_ns();
    int created = 0;
    for (int t = 0; t < nt; t++) {
        if (pthread_create(&threads[t], NULL, qd_worker, &args[t]) != 0)
            break;
        created++;
    }
    if (created != nt) {
        for (int t = 0; t < created; t++)
            pthread_join(threads[t], NULL);
        return -1;
    }

    int all_ok = 1;
    for (int t = 0; t < nt; t++) {
        pthread_join(threads[t], NULL);
        if (!args[t].ok)
            all_ok = 0;
    }
    uint64_t elapsed = timer_ns() - t0;

    if (!all_ok || elapsed == 0) return -1;
    size_t total_ops = ops_per_thread * nt;
    return (double)total_ops / ((double)elapsed / 1e9);
}

/* --- Fsync latency --- */

static double bench_fsync_latency(void *buf) {
    int fd = open_nocache(O_RDWR);
    if (fd < 0) return -1;

    /* Calibrate */
    size_t iters = 100;
    uint64_t tc0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        if (full_pwrite(fd, buf, RND_BLOCK, 0) < 0 ||
            checked_fsync(fd) < 0) {
            close(fd);
            return -1;
        }
    }
    uint64_t tc1 = timer_ns();
    double ns_per = (double)(tc1 - tc0) / iters;
    if (ns_per <= 0) { close(fd); return -1; }
    iters = (size_t)(2000000000.0 / ns_per);
    if (iters < 100) iters = 100;
    if (iters > 50000) iters = 50000;

    uint64_t t0 = timer_ns();
    for (size_t i = 0; i < iters; i++) {
        if (full_pwrite(fd, buf, RND_BLOCK, 0) < 0 ||
            checked_fsync(fd) < 0) {
            close(fd);
            return -1;
        }
    }
    uint64_t t1 = timer_ns();

    close(fd);
    if (t1 <= t0) return -1;
    return (double)(t1 - t0) / iters / 1000.0;
}

void bench_disk(void) {
    if (ensure_test_dir() < 0) {
        printf("=== SSD / Storage ===\n");
        printf("  Failed to create build directory\n");
        return;
    }

    struct disk_info di;
    query_disk_info(&di, TEST_DIR);

    printf("=== SSD / Storage ===\n");
    if (di.fs_type[0])
        printf("  Filesystem: %s\n", di.fs_type);
    printf("%-22s %14s\n", "Test", "Throughput");
    printf("%-22s %14s\n", "----", "----------");

    void *buf;
    if (posix_memalign(&buf, 4096, SEQ_BLOCK) != 0) {
        printf("  Failed to allocate I/O buffer\n");
        return;
    }

    if (create_test_file(buf) < 0) {
        printf("  Failed to create test file\n");
        free(buf);
        return;
    }

    double sr = bench_seq_read(buf);
    if (sr > 0) printf("%-22s %10.2f GB/s\n", "Seq read", sr);
    fflush(stdout);

    double sw = bench_seq_write(buf);
    if (sw > 0) printf("%-22s %10.2f GB/s\n", "Seq write", sw);
    fflush(stdout);

    double rr_iops, rr_lat, rw_iops, rw_lat;
    bench_rand_read(buf, &rr_iops, &rr_lat);
    if (rr_iops > 0)
        printf("%-22s %7.0fK IOPS  (%.1f \xC2\xB5s)\n",
               "Rand 4K read", rr_iops / 1000.0, rr_lat);
    fflush(stdout);

    bench_rand_write(buf, &rw_iops, &rw_lat);
    if (rw_iops > 0)
        printf("%-22s %7.0fK IOPS  (%.1f \xC2\xB5s)\n",
               "Rand 4K write (burst)", rw_iops / 1000.0, rw_lat);
    fflush(stdout);

    double rws_iops, rws_lat;
    bench_rand_write_fsync(buf, &rws_iops, &rws_lat);
    if (rws_iops > 0)
        printf("%-22s %7.0fK IOPS  (%.1f \xC2\xB5s)\n",
               "Rand 4K write (fsync)", rws_iops / 1000.0, rws_lat);
    fflush(stdout);

    /* --- Queue Depth Scaling --- */
    printf("\n--- Queue Depth Scaling (random 4K read) ---\n");
    printf("%-8s %12s\n", "QD", "IOPS");
    printf("%-8s %12s\n", "--", "----");
    {
        int qds[] = {1, 2, 4, 8, 16};
        for (int q = 0; q < 5; q++) {
            double iops = bench_qd_scaling(buf, qds[q]);
            if (iops > 0)
                printf("%-8d %8.0fK\n", qds[q], iops / 1000.0);
            fflush(stdout);
        }
    }

    /* --- Fsync Latency --- */
    {
        double fsync_us = bench_fsync_latency(buf);
        if (fsync_us > 0) {
            printf("\n%-22s %10.1f \xC2\xB5s/fsync\n", "Fsync latency", fsync_us);
            fflush(stdout);
        }
    }

    free(buf);
    unlink(test_file);
}
