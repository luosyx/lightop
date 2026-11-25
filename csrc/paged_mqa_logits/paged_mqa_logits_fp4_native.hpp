// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// gfx946 native MXFP4 path for paged_mqa_logits_fp4.
//
// gfx946 (and only gfx946: the feature predicate is hcu-mmop6-insts, which
// gfx938 lacks) has a native E2M1 MMAC, v_mmac_f32_16x16x64_fp4, plus a
// block-scaled form v_mmac_scale_f32_16x16x64_fp4.  Using it removes the whole
// software decode this operator is built around -- no E2M1->E4M3 pivot, no
// BF16 lookup tables, no subnormal shift, no per-tile eligibility ballot.
//
// FRAGMENT LAYOUT  (determinate; from DTK crt/mma_gfx946.hpp load64(), and
// confirmed by codegen: the loads below emit exactly one global_load_dwordx2
// per operand, i.e. no shuffling or unpacking at all.)
//
//   For packed input (two E2M1 codes per byte) with an even ldm in elements:
//       element_base = (lane & 15) * ldm + (lane >> 4) * 16
//       byte_base    = element_base / 2
//                    = (lane & 15) * (ldm / 2) + (lane >> 4) * 8
//       fragment     = the 8 consecutive bytes at byte_base, as int2
//
//   This operator stores 128 channels per row in 64 bytes, so ldm / 2 == 64 is
//   exactly one K row or one Q head, and `lane & 15` is already how this kernel
//   assigns a row (a KV position for K, a head within a tile for Q).  The two
//   K = 64 steps needed to cover 128 channels are therefore the byte ranges
//   [0, 32) and [32, 64) of a row, with lane group g taking 8 bytes at
//   `step * 32 + g * 8`.
//
// CHANNEL <-> SCALE GROUP
//
//   Scales are per 32 channels, so group s covers channels [32 s, 32 s + 32).
//   Lane group g of step t holds channels [64 t + 16 g, +16), hence scale group
//   `2 t + (g >> 1)`.  A single K = 64 MMAC therefore spans two scale groups
//   (lane groups 0,1 take one, lane groups 2,3 the next).  Because an MMAC
//   reduces across lanes, those two scales cannot be factored out of the
//   accumulator afterwards -- this path requires the hardware scale buffer.
//   The K = 32 mixed-format MMAC would map one scale group per step and could
//   use software scaling instead; see kFp4NativeUseHwScale below.
#pragma once

#include <hip/hip_runtime.h>

namespace lightop::deep_gemm {
namespace fp4_native {

#if defined(__gfx946__)
#define LIGHTOP_FP4_NATIVE_DEVICE 1
#else
#define LIGHTOP_FP4_NATIVE_DEVICE 0
#endif

using f32x4 = float __attribute__((ext_vector_type(4)));
using i32x2 = int __attribute__((ext_vector_type(2)));

// Bytes of packed E2M1 per row of 128 channels.
inline constexpr int kRowBytes = 64;
// K = 64 steps needed to cover the 128 channels of one head.
inline constexpr int kSteps = 2;
// Bytes each lane contributes per step.
inline constexpr int kFragBytes = 8;

/// Byte offset of this lane's fragment inside a 64-byte row, for step `step`.
/// `group` is lane / 16; the row index (lane % 16) is applied by the caller,
/// which already knows the row's base address.
__device__ __forceinline__ constexpr int frag_offset(int step, int group) {
    return step * (kRowBytes / kSteps) + group * kFragBytes;
}

/// Scale group (0..3) feeding `group` at `step`.  See the header comment.
__device__ __forceinline__ constexpr int scale_group(int step, int group) {
    return 2 * step + (group >> 1);
}

/// One fragment: 8 packed bytes = 16 E2M1 codes, loaded with no repacking.
__device__ __forceinline__ i32x2 load_fragment(const uint8_t* row_base, int step, int group) {
    return *reinterpret_cast<const i32x2*>(row_base + frag_offset(step, group));
}

/// Load this lane's fragment for `step`, keeping only the channels that belong
/// to scale group `2 * step + half`.  Lane groups outside that scale group
/// contribute zero, so the MMAC's K reduction stays inside one scale group and
/// its result carries a single, uniform block-scale pair.
///
/// This is what makes a correct native-FP4 path possible without the scale
/// buffer: a K = 64 MMAC spans two scale groups, and because an MMAC reduces
/// across lanes those two scales cannot be divided out of the accumulator
/// afterwards.  Masking costs half the lane utilisation (four MMACs per
/// head-tile instead of two) but removes the software decode entirely, and the
/// control flow is identical to the scale-buffer version -- dropping the mask
/// and adding scale_a_offset / scale_b_offset is the only change needed once
/// the scale-buffer addressing has been confirmed on hardware.
__device__ __forceinline__ i32x2 load_fragment_masked(const uint8_t* row_base, int step,
                                                      int group, int half) {
    if ((group >> 1) != half) return i32x2{0, 0};
    return load_fragment(row_base, step, group);
}

/// Scale exponent applied to one masked MMAC's accumulator: the K and Q block
/// scales are UE8M0, so the product is 2^(sk + sq - 254) exactly.
__device__ __forceinline__ float apply_block_scale(float acc, unsigned sk, unsigned sq) {
    return ldexpf(acc, int(sk) + int(sq) - 254);
}

/// Unscaled native FP4 MMAC.  Operands carry raw E2M1 codes; the caller is
/// responsible for any block scaling.
__device__ __forceinline__ f32x4 mmac_fp4(i32x2 a, i32x2 b, f32x4 c) {
#if LIGHTOP_FP4_NATIVE_DEVICE
    return __builtin_hcu_mmac_f32_16x16x64_fp4_lit_lts(a, b, c, false, false);
#else
    (void)a; (void)b;
    __builtin_trap();
    return c;
#endif
}

/// Block-scaled native FP4 MMAC.  `sa` / `sb` select the operands' scale entries
/// in the scale buffer.
///
/// ASSUMPTION (not yet validated on hardware -- there is no gfx946 part on the
/// current nodes): the scale buffer is addressed per lane, m0 supplies the row
/// base for A in its low half and for B in its high half, and these offsets
/// index entries within that row.  This mirrors the only production use of the
/// instruction family available to us (deepgemm gfx946 lowbit GEMM, which pairs
/// `ds_scale_copy_ds2buf op_ctrl0:2` writes of one dword per lane with
/// `scale_a_offset` / `scale_b_offset` immediates).  Everything that depends on
/// that assumption is confined to this function and to store_scales() below, so
/// a hardware probe can correct it in one place.
template <int ScaleA, int ScaleB>
__device__ __forceinline__ f32x4 mmac_fp4_scaled(i32x2 a, i32x2 b, f32x4 c) {
#if LIGHTOP_FP4_NATIVE_DEVICE
    return __builtin_hcu_mmac_scale_f32_16x16x64_fp4_lit_lts(a, b, c, false, false, ScaleA, ScaleB);
#else
    (void)a; (void)b;
    __builtin_trap();
    return c;
#endif
}


// ---------------------------------------------------------------------------
// Path A: 16x16x32 mixed-format MMAC (v_mmac_f32_16x16x32_f8f6f4, fmt = FP4).
//
// Fragment layout, from DTK crt/mma_gfx946.hpp load32():
//     element_base = (lane & 15) * ldm + (lane >> 4) * 8
// so lane (row, g) holds elements [8 g, 8 g + 8) of the 32-element window, one
// code per byte.  Four steps cover 128 channels and step t is exactly scale
// group t, so each MMAC's K reduction already sits inside a single scale group:
// no masking, full lane utilisation, and the block scale still factors out into
// FP32 afterwards.  The cost relative to path B is unpacking the operator's
// two-codes-per-byte storage into the padded container.
// ---------------------------------------------------------------------------

/// Steps of 32 channels needed to cover one head.
inline constexpr int kStepsK32 = 4;
/// Source bytes each lane reads per K=32 step (8 codes, two per byte).
inline constexpr int kFragBytesK32 = 4;

/// Byte offset of this lane's packed source for K=32 step `step`.
__device__ __forceinline__ constexpr int frag_offset_k32(int step, int group) {
    return step * 16 + group * kFragBytesK32;
}

/// Scale group feeding a K=32 step: one step, one group.
__device__ __forceinline__ constexpr int scale_group_k32(int step) { return step; }

/// Expand four packed bytes (eight E2M1 codes, low nibble first) into the
/// padded one-code-per-byte container the mixed-format MMAC expects.
__device__ __forceinline__ i32x2 unpack_fp4x8(unsigned packed) {
    const unsigned even = packed & 0x0f0f0f0fu;          // codes 0, 2, 4, 6
    const unsigned odd = (packed >> 4) & 0x0f0f0f0fu;    // codes 1, 3, 5, 7
    return i32x2{static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x05010400u)),
                 static_cast<int>(__builtin_amdgcn_perm(odd, even, 0x07030602u))};
}

/// Load and expand this lane's K=32 fragment.
__device__ __forceinline__ i32x2 load_fragment_k32(const uint8_t* row_base, int step, int group) {
    return unpack_fp4x8(*reinterpret_cast<const unsigned*>(row_base + frag_offset_k32(step, group)));
}

/// Mixed-format MMAC with the format selector set to FP4 for both operands.
/// The selector value is the one deepgemm's gfx946 lowbit GEMM uses for
/// fp4 x fp4; it is an immediate, so it costs nothing at run time.
inline constexpr int kFmtFp4Fp4 = 0;
__device__ __forceinline__ f32x4 mmac_fp4_k32(i32x2 a, i32x2 b, f32x4 c) {
#if LIGHTOP_FP4_NATIVE_DEVICE
    return __builtin_hcu_mmac_f32_16x16x32_f8f6f4_lit_lts(a, b, c, kFmtFp4Fp4, false, false);
#else
    (void)a; (void)b;
    __builtin_trap();
    return c;
#endif
}

}  // namespace fp4_native
}  // namespace lightop::deep_gemm
