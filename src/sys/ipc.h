#pragma once

#include "core/status.h"
#include "core/types.h"

/* Ping-pong helpers shared by the sys and net sections.
 *
 * A ping-pong has two peers: the driver (side 0) times round trips, the echo
 * (side 1) returns every byte. Both run on placed threads (or a forked child)
 * so the main thread's affinity and QoS are never changed. */

#define SB_IPC_WINDOW_NS  500'000'000ULL      /* timed window per ping-pong test */
#define SB_IPC_TIMEOUT_NS 5'000'000'000ULL    /* a peer that makes no progress for this long is an error */

/* Choose the two peer CPUs once (Linux: first two allowed CPUs that are not
 * SMT siblings; macOS: QoS only). Call from the main thread before any test. */
void sb_ipc_peers_init(void);
/* Human-readable placement, e.g. "CPUs 0 and 1 (pinned)". */
const char *sb_ipc_peers_desc(void);
/* Busy-poll tests need two CPUs: two spinners on one CPU only progress at
 * scheduler-tick granularity. */
bool sb_ipc_can_spin(void);
/* Place the calling thread as peer `side` (0 or 1). Best effort. */
void sb_ipc_place(u32 side);

sb_status_e sb_ipc_set_nonblock(int fd);

/* Echo side: read one byte, write it back, until the driver sends the stop
 * byte. `spin` busy-polls the (O_NONBLOCK) read fd instead of sleeping. */
sb_status_e sb_ipc_echo(int rfd, int wfd, bool spin);

/* Driver side: warm up for SB_WARMUP_NS, time round trips for `window_ns`,
 * then stop the echo. Returns mean ns per round trip. */
sb_status_e sb_ipc_drive(int rfd, int wfd, bool spin, u64 window_ns, f64 *out_rtt_ns);

/* Driver on endpoint a, echo on endpoint b, each on its own placed thread. */
sb_status_e sb_ipc_pingpong_threads(int a_rfd, int a_wfd, int b_rfd, int b_wfd, bool spin, f64 *out_rtt_ns);

/* macOS: comma-separated names of system extensions in `category` (e.g.
 * "network_extension", "endpoint_security") that are activated and enabled,
 * read from `systemextensionsctl list` (~40 ms). Returns the count; 0 on
 * other platforms or when the tool is unavailable. */
u32 sb_ipc_active_exts(const char *category, char *out, size_t cap);
