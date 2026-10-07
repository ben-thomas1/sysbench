#include "core/status.h"

const char *sb_status_str(sb_status_e s) {
    switch (s) {
    case SB_OK:              return "ok";
    case SB_ERR_NOMEM:       return "out of memory";
    case SB_ERR_INVALID:     return "invalid argument";
    case SB_ERR_RANGE:       return "out of range";
    case SB_ERR_NOTFOUND:    return "not found";
    case SB_ERR_UNSUPPORTED: return "unsupported";
    case SB_ERR_IO:          return "I/O error";
    case SB_ERR_SYS:         return "system call failed";
    case SB_ERR_TIMEOUT:     return "timed out";
    }
    return "unknown error";
}
