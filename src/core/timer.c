#include "core/timer.h"

void sb_timer_spin(u64 ns) {
    u64 t0 = sb_timer_now_ns();
    u64 x  = 0;
    while (sb_timer_now_ns() - t0 < ns) {
        for (u32 i = 0; i < 1000; i++) {
            x += i;
            __asm__ volatile("" : "+r"(x));
        }
    }
}
