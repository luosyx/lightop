
// Device compile + codegen regression for the three gfx946 native-FP4 paths.
// Scoring logic matches the host proof in fp4_native_logic_test.cpp.
//   A: 16x16x32 mixed-format, one scale group per step, full lane utilisation
//   B: 16x16x64 masked, one scale group per MMAC, zero unpacking
//   C: 16x16x64 with the hardware scale buffer (addressing pending hardware)
#include "paged_mqa_logits_fp4_native.hpp"
#include <cstdio>

using namespace lightop::deep_gemm::fp4_native;

__device__ __forceinline__ float reduce_heads(const float (*total)[4], const float* w,
                                              int group, int ht) {
    float sum = 0.0f;
    for (int h = 0; h < ht; ++h)
        for (int j = 0; j < 4; ++j)
            sum = fmaf(fmaxf(total[h][j], 0.0f), w[h * 16 + j * 4 + group], sum);
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    return sum;
}

// ---- path A ---------------------------------------------------------------
template <int HT>
__global__ void score_native_k32(const uint8_t* __restrict__ kv, const uint8_t* __restrict__ q,
                                 const unsigned* __restrict__ k_sf,
                                 const unsigned* __restrict__ q_sf,
                                 const float* __restrict__ w, float* __restrict__ out) {
    const int lane = threadIdx.x % 64, row = lane % 16, group = lane / 16;
    const unsigned ksf = k_sf[row];
    float total[HT][4];
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j) total[h][j] = 0.0f;
#pragma unroll
    for (int step = 0; step < kStepsK32; ++step) {
        const int s = scale_group_k32(step);
        const unsigned sk = (ksf >> (s * 8)) & 255u;
        const i32x2 kf = load_fragment_k32(kv + row * kRowBytes, step, group);
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const i32x2 qf = load_fragment_k32(q + (h * 16 + row) * kRowBytes, step, group);
            f32x4 acc = f32x4{0.f, 0.f, 0.f, 0.f};
            acc = mmac_fp4_k32(kf, qf, acc);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned sq = (q_sf[h * 16 + j * 4 + group] >> (s * 8)) & 255u;
                total[h][j] += apply_block_scale(acc[j], sk, sq);
            }
        }
    }
    const float sum = reduce_heads(total, w, group, HT);
    if (group == 0) out[row] = sum;
}

// ---- path B ---------------------------------------------------------------
template <int HT>
__global__ void score_native_k64_masked(const uint8_t* __restrict__ kv,
                                        const uint8_t* __restrict__ q,
                                        const unsigned* __restrict__ k_sf,
                                        const unsigned* __restrict__ q_sf,
                                        const float* __restrict__ w, float* __restrict__ out) {
    const int lane = threadIdx.x % 64, row = lane % 16, group = lane / 16;
    const unsigned ksf = k_sf[row];
    float total[HT][4];
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j) total[h][j] = 0.0f;
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int s = 2 * step + half;
            const unsigned sk = (ksf >> (s * 8)) & 255u;
            const i32x2 kf = load_fragment_masked(kv + row * kRowBytes, step, group, half);
#pragma unroll
            for (int h = 0; h < HT; ++h) {
                const i32x2 qf =
                    load_fragment_masked(q + (h * 16 + row) * kRowBytes, step, group, half);
                f32x4 acc = f32x4{0.f, 0.f, 0.f, 0.f};
                acc = mmac_fp4(kf, qf, acc);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned sq = (q_sf[h * 16 + j * 4 + group] >> (s * 8)) & 255u;
                    total[h][j] += apply_block_scale(acc[j], sk, sq);
                }
            }
        }
    }
    const float sum = reduce_heads(total, w, group, HT);
    if (group == 0) out[row] = sum;
}

// ---- path C ---------------------------------------------------------------
// Two unmasked K=64 MMACs; the block scales come from the scale buffer, so the
// FP32 rescale disappears.  Scale-buffer population is not done here -- the
// addressing is still unconfirmed, see mmac_fp4_scaled().
template <int HT>
__global__ void score_native_k64_hwscale(const uint8_t* __restrict__ kv,
                                         const uint8_t* __restrict__ q,
                                         const float* __restrict__ w, float* __restrict__ out) {
    const int lane = threadIdx.x % 64, row = lane % 16, group = lane / 16;
    float total[HT][4];
#pragma unroll
    for (int h = 0; h < HT; ++h) {
        f32x4 acc = f32x4{0.f, 0.f, 0.f, 0.f};
        acc = mmac_fp4_scaled<0, 1>(load_fragment(kv + row * kRowBytes, 0, group),
                                    load_fragment(q + (h * 16 + row) * kRowBytes, 0, group), acc);
        acc = mmac_fp4_scaled<2, 3>(load_fragment(kv + row * kRowBytes, 1, group),
                                    load_fragment(q + (h * 16 + row) * kRowBytes, 1, group), acc);
#pragma unroll
        for (int j = 0; j < 4; ++j) total[h][j] = acc[j];
    }
    const float sum = reduce_heads(total, w, group, HT);
    if (group == 0) out[row] = sum;
}

#define INST(KERNEL, HT) template __global__ void KERNEL<HT>
INST(score_native_k32, 1)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k32, 2)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k32, 4)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k64_masked, 1)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k64_masked, 2)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k64_masked, 4)(const uint8_t*, const uint8_t*, const unsigned*, const unsigned*, const float*, float*);
INST(score_native_k64_hwscale, 1)(const uint8_t*, const uint8_t*, const float*, float*);
INST(score_native_k64_hwscale, 2)(const uint8_t*, const uint8_t*, const float*, float*);

int main() { printf("NATIVE_DEVICE_COMPILE_OK\n"); return 0; }
