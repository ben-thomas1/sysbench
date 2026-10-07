#pragma once
#include <stdlib.h>

static inline void shuffle_indices(size_t *arr, size_t n) {
    if (n < 2) return;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
