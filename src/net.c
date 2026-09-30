#include "bench.h"
#include "timer.h"

#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <poll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define BW_BLOCK_SIZE  (64 * 1024)  /* 64 KiB */
#define BW_TOTAL_BYTES (2ULL * 1024 * 1024 * 1024)  /* 2 GiB */
#define LAT_ITERS      100000

static int tcp_socket(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct timeval timeout = {.tv_sec = 5};
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    return fd;
}

static int accept_client(int srv) {
    struct pollfd p = {.fd = srv, .events = POLLIN};
    int ready;
    do { ready = poll(&p, 1, 5000); } while (ready < 0 && errno == EINTR);
    if (ready == 0) errno = ETIMEDOUT;
    if (ready <= 0) return -1;
    return accept(srv, NULL, NULL);
}

static int full_read(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = read(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("read");
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "read: unexpected EOF\n");
            return -1;
        }
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
            perror("write");
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "write: wrote 0 bytes\n");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

struct bw_client_arg {
    int port;
    uint64_t total_bytes;
    int ok;
};

static void *bw_client(void *arg) {
    struct bw_client_arg *a = (struct bw_client_arg *)arg;

    int fd = tcp_socket();
    if (fd < 0) { perror("socket"); return NULL; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(a->port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return NULL;
    }

    char *buf = malloc(BW_BLOCK_SIZE);
    if (!buf) {
        close(fd);
        return NULL;
    }
    memset(buf, 0xAB, BW_BLOCK_SIZE);

    uint64_t sent = 0;
    while (sent < a->total_bytes) {
        size_t chunk = BW_BLOCK_SIZE;
        if (chunk > a->total_bytes - sent)
            chunk = (size_t)(a->total_bytes - sent);
        if (full_write(fd, buf, chunk) < 0)
            break;
        sent += chunk;
    }

    a->ok = sent == a->total_bytes;
    free(buf);
    close(fd);
    return NULL;
}

static double measure_loopback_bw_gbps(void) {
    int srv = tcp_socket();
    if (srv < 0) { perror("socket"); return -1; }

    int opt = 1;
    if (setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
        close(srv);
        return -1;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;  /* let OS pick port */
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(srv);
        return -1;
    }

    socklen_t alen = sizeof(addr);
    if (getsockname(srv, (struct sockaddr *)&addr, &alen) < 0) {
        perror("getsockname");
        close(srv);
        return -1;
    }
    int port = ntohs(addr.sin_port);

    if (listen(srv, 1) < 0) {
        perror("listen");
        close(srv);
        return -1;
    }

    struct bw_client_arg carg = { .port = port, .total_bytes = BW_TOTAL_BYTES, .ok = 0 };
    pthread_t client;
    if (pthread_create(&client, NULL, bw_client, &carg) != 0) {
        perror("pthread_create");
        close(srv);
        return -1;
    }

    int conn = accept_client(srv);
    if (conn < 0) {
        perror("accept");
        close(srv);
        pthread_join(client, NULL);
        return -1;
    }

    char *buf = malloc(BW_BLOCK_SIZE);
    if (!buf) {
        close(conn);
        close(srv);
        pthread_join(client, NULL);
        return -1;
    }
    uint64_t total = 0;

    uint64_t t0 = timer_ns();
    while (total < BW_TOTAL_BYTES) {
        size_t chunk = BW_BLOCK_SIZE;
        if (chunk > BW_TOTAL_BYTES - total)
            chunk = (size_t)(BW_TOTAL_BYTES - total);
        if (full_read(conn, buf, chunk) < 0)
            break;
        total += chunk;
    }
    uint64_t t1 = timer_ns();

    free(buf);
    close(conn);
    close(srv);
    pthread_join(client, NULL);

    if (!carg.ok || total != BW_TOTAL_BYTES || t1 <= t0)
        return -1;
    return (double)total / ((double)(t1 - t0) / 1e9) / 1e9;
}

/* --- Loopback latency (TCP ping-pong) --- */

struct lat_client_arg {
    int port;
    int iters;
    int ok;
};

static void *lat_client(void *arg) {
    struct lat_client_arg *a = (struct lat_client_arg *)arg;

    int fd = tcp_socket();
    if (fd < 0) { perror("socket"); return NULL; }

    int opt = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(TCP_NODELAY)");
        close(fd);
        return NULL;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(a->port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return NULL;
    }

    char byte = 'x';
    for (int i = 0; i < a->iters; i++) {
        if (full_write(fd, &byte, 1) < 0 ||
            full_read(fd, &byte, 1) < 0) {
            close(fd);
            return NULL;
        }
    }

    a->ok = 1;
    close(fd);
    return NULL;
}

static double measure_loopback_latency_us(void) {
    int srv = tcp_socket();
    if (srv < 0) { perror("socket"); return -1; }

    int opt = 1;
    if (setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
        close(srv);
        return -1;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(srv);
        return -1;
    }

    socklen_t alen = sizeof(addr);
    if (getsockname(srv, (struct sockaddr *)&addr, &alen) < 0) {
        perror("getsockname");
        close(srv);
        return -1;
    }
    int port = ntohs(addr.sin_port);

    if (listen(srv, 1) < 0) {
        perror("listen");
        close(srv);
        return -1;
    }

    struct lat_client_arg carg = { .port = port, .iters = LAT_ITERS, .ok = 0 };
    pthread_t client;
    if (pthread_create(&client, NULL, lat_client, &carg) != 0) {
        perror("pthread_create");
        close(srv);
        return -1;
    }

    int conn = accept_client(srv);
    if (conn < 0) {
        perror("accept");
        close(srv);
        pthread_join(client, NULL);
        return -1;
    }

    if (setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(TCP_NODELAY)");
        close(conn);
        close(srv);
        pthread_join(client, NULL);
        return -1;
    }

    char byte;
    uint64_t t0 = timer_ns();
    for (int i = 0; i < LAT_ITERS; i++) {
        if (full_read(conn, &byte, 1) < 0 ||
            full_write(conn, &byte, 1) < 0) {
            close(conn);
            close(srv);
            pthread_join(client, NULL);
            return -1;
        }
    }
    uint64_t t1 = timer_ns();

    close(conn);
    close(srv);
    pthread_join(client, NULL);

    if (!carg.ok || t1 <= t0)
        return -1;
    return (double)(t1 - t0) / LAT_ITERS / 1000.0;
}

/* --- Entry point --- */

void bench_net(void) {
    signal(SIGPIPE, SIG_IGN);

    printf("=== Network (TCP loopback \xe2\x80\x94 OS stack) ===\n");
    printf("%-24s %14s\n", "Test", "Result");
    printf("%-24s %14s\n", "----", "------");

    double bw = measure_loopback_bw_gbps();
    if (bw > 0)
        printf("%-24s %10.2f GB/s\n", "TCP throughput", bw);
    else
        printf("%-24s %14s\n", "TCP throughput", "error");
    fflush(stdout);

    double lat = measure_loopback_latency_us();
    if (lat > 0)
        printf("%-24s %10.2f us/RTT\n", "TCP latency", lat);
    else
        printf("%-24s %14s\n", "TCP latency", "error");
    fflush(stdout);
}
