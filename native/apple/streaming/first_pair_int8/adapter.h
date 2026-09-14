#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) void* av8_create(size_t K, size_t N, const int8_t* qw_NK, const float* sw_N);
__attribute__((visibility("default"))) int av8_run_q(void*, size_t T, const int8_t* qx_TK, const float* sx_T, float* y_NT, int variant);
__attribute__((visibility("default"))) void av8_destroy(void*);
__attribute__((visibility("default"))) const char* av8_error(void);
#ifdef __cplusplus
}
#endif
