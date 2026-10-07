#pragma once

typedef enum {
    SB_OK = 0,
    SB_ERR_NOMEM,        /* allocation failed */
    SB_ERR_INVALID,      /* bad argument or input */
    SB_ERR_RANGE,        /* value out of range / overflow */
    SB_ERR_NOTFOUND,     /* file, device, or resource missing */
    SB_ERR_UNSUPPORTED,  /* not available on this platform or hardware */
    SB_ERR_IO,           /* file or device I/O failed */
    SB_ERR_SYS,          /* other OS call failed (errno-style) */
    SB_ERR_TIMEOUT,      /* operation did not complete in time */
} sb_status_e;

const char *sb_status_str(sb_status_e s);
