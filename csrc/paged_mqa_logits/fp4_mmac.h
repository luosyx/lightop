// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_runtime.h>

namespace lightop::deep_gemm::fp4_detail {

#if defined(__gfx946__)
using intx2 = int __attribute__((ext_vector_type(2)));
using uintx4 = unsigned __attribute__((ext_vector_type(4)));
using floatx4 = float __attribute__((ext_vector_type(4)));

// The caller holds one 32-element scale group per lane (lane / 16).
// Gather one group across the wave into the first 32 columns of K64; the
// remaining columns are zero. Each participating lane supplies 16 E2M1
// values in two VGPRs, low nibble first. Keep all 64 lanes active at MMAC.
__device__ __forceinline__ intx2 group32_operand(const uintx4& bits, int scale_group) {
    const int lane = threadIdx.x % 64;
    const int source = (lane & 15) + scale_group * 16;
    const unsigned w0 = __shfl(bits[0], source, 64);
    const unsigned w1 = __shfl(bits[1], source, 64);
    const unsigned w2 = __shfl(bits[2], source, 64);
    const unsigned w3 = __shfl(bits[3], source, 64);
    const int group = lane / 16;
    return {static_cast<int>(group < 2 ? (group == 0 ? w0 : w2) : 0u),
            static_cast<int>(group < 2 ? (group == 0 ? w1 : w3) : 0u)};
}

// A is the KV token, B the query head. With lit/lts disabled, C[j] belongs
// to token lane%16 and head 4*j+lane/16, as in the BF16 MQA path. A K64
// instruction cannot share one scale across two independent MX K32 groups:
// accumulate each group's unscaled dot, then restore both UE8M0 exponents
// before adding groups. ReLU and head weighting happen only after K128.
// Callers retain their existing exceptional path outside the normal range.
__device__ __forceinline__ floatx4 mmac_scaled_fp4(
    const uintx4& a, unsigned scale_a, const uintx4& b, unsigned scale_b) {
    const int lane = threadIdx.x % 64;
    floatx4 result = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const intx2 av = group32_operand(a, g);
        const intx2 bv = group32_operand(b, g);
        const floatx4 dot = __builtin_hcu_mmac_f32_16x16x64_fp4_lit_lts(
            av, bv, floatx4{0.f, 0.f, 0.f, 0.f}, false, false);
        const int sa = __shfl(scale_a, (lane & 15) + g * 16, 64);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int sb = __shfl(scale_b, 4 * j + lane / 16 + g * 16, 64);
            result[j] += ldexpf(dot[j], sa + sb - 254);
        }
    }
    return result;
}
#endif

}  // namespace lightop::deep_gemm::fp4_detail
