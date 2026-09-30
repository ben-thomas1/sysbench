#pragma once

#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#ifdef __APPLE__

static inline int disable_write_cache(int fd) {
    return fcntl(fd, F_NOCACHE, 1);
}

static inline int open_direct(const char *path, int flags) {
    int fd = open(path, flags, 0600);
    if (fd < 0) return -1;
    if (disable_write_cache(fd) < 0) {
        int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    fcntl(fd, F_RDAHEAD, 0);
    return fd;
}

static inline void drop_page_cache(int fd) {
    (void)fd;  /* macOS uses F_NOCACHE at open time */
}

#else

static inline int disable_write_cache(int fd) {
    (void)fd;  /* Linux uses O_DIRECT at open time */
    return 0;
}

static inline int open_direct(const char *path, int flags) {
    int fd = open(path, flags | O_DIRECT, 0600);
    if (fd < 0) return -1;
    return fd;
}

static inline void drop_page_cache(int fd) {
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
}

#endif
