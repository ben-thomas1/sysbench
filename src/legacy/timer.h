#pragma once

#include <stdint.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <pthread.h>

static double timer_ns_per_tick;
static pthread_once_t timer_once = PTHREAD_ONCE_INIT;

static void timer_init(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    timer_ns_per_tick = (double)info.numer / info.denom;
}

static inline uint64_t timer_ns(void) {
    pthread_once(&timer_once, timer_init);
    return (uint64_t)(mach_absolute_time() * timer_ns_per_tick);
}

#else
#include <time.h>

static inline uint64_t timer_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif
