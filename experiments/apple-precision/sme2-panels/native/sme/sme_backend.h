/* Exact symmetric INT8 adapter for the separately pinned Arm SME2 kernel. */
#ifndef IPA_SME_BACKEND_H
#define IPA_SME_BACKEND_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Guarded queries: no SME instruction is executed on unsupported hardware. */
int ipa_sme_supported(void);
int ipa_sme_geometry(size_t* svl_bytes, size_t* mr, size_t* nr);
const char* ipa_sme_last_error(void);

/* SIZE0 indicates invalid geometry or overflow. T0 has zero RHS size. */
size_t ipa_sme_lhs_bytes(size_t m, size_t k, size_t svl_bytes);
size_t ipa_sme_rhs_bytes(size_t k, size_t t, size_t svl_bytes);

/* Pack existing signed quantized values and positive FP32 scales unchanged.
 * KP=roundup(K,32), MR=SVL/4, NR=SVL. Block byte layouts:
 * LHS: MR*KP signed bytes, MR int32 zero offsets, MR float weight scales.
 * RHS: NR*KP signed bytes, NR int32 zero sums, NR float activation scales,
 *      NR float negative-zero bias. Out-of-range data/padding is zero.
 * Data address in either block: (k/4)*lanes*4 + lane*4 + k%4.
 * Block strides: MR*(KP+8), NR*(KP+12). Buffers are at least 4-byte aligned.
 * LHS is immutable after construction; RHS belongs to one invocation.
 * Destination must not overlap input bytes or scale arrays.
 */
int ipa_sme_pack_lhs(const int8_t* qw, const float* sw, size_t m, size_t k,
                     void* destination, size_t bytes, size_t svl_bytes);
int ipa_sme_pack_rhs(const int8_t* qx, const float* sx, size_t k, size_t t,
                     void* destination, size_t bytes, size_t svl_bytes);

/* No input quantization, bias or residual is performed here. Exact order:
 * scale=sw*sx; result=float(full_K_signed_int32_dot)*scale.
 * Upstream adds -0.0 and clamps to [-inf,+inf], preserving finite FP32 bits
 * under round-to-nearest. Outputs are checked for nonfinite values afterward.
 * FPCR rounding/flush behavior and streaming vector length are validated.
 * Y is contiguous [M,T]. first must be MR-aligned; last may be a tail.
 * Concurrent disjoint row ranges and immutable shared packs are supported.
 * Successful return0; failure-1 with thread-local last_error. T0 is a no-op.
 */
int ipa_sme_run_rows(size_t m, size_t k, size_t t,
                     const void* lhs, size_t lhs_bytes,
                     const void* rhs, size_t rhs_bytes,
                     float* y, size_t first, size_t last, size_t svl_bytes);
/* Same arithmetic and packed layout, writing a time subregion in global BCT.
 * t is the packed panel length; output_time is the destination row stride.
 * Only [first,last) rows and [output_offset,output_offset+t) are touched. */
int ipa_sme_run_panel(size_t m, size_t k, size_t t,
                     const void* lhs, size_t lhs_bytes,
                     const void* rhs, size_t rhs_bytes,
                     float* y, size_t output_time, size_t output_offset,
                     size_t first, size_t last, size_t svl_bytes);
/* Validate CPU, streaming length and FPCR before quantizing a panel. */
int ipa_sme_validate_runtime(size_t svl_bytes);

#ifdef __cplusplus
}
#endif
#endif
