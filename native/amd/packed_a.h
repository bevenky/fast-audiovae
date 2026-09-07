#ifndef NCC_AOCL_PACKED_A_H
#define NCC_AOCL_PACKED_A_H
#include <stdint.h>
#include <stddef.h>
#if defined(__GNUC__)
#define AOCL_API __attribute__((visibility("default")))
#else
#define AOCL_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct ncc_aocl_pack;
AOCL_API int ncc_aocl_cpu_supported(void);
AOCL_API uint32_t ncc_aocl_adapter_abi(void);
struct ncc_aocl_pack* ncc_aocl_create(const float*, int64_t, int64_t);
void ncc_aocl_destroy(struct ncc_aocl_pack*);
size_t ncc_aocl_packed_bytes(const struct ncc_aocl_pack*);
int ncc_aocl_compute(const struct ncc_aocl_pack*, const float*, float*, int64_t, int64_t);
#ifdef __cplusplus
}
#endif
#endif
