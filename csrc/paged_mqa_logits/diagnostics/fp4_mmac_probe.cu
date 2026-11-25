// gfx946 native FP4 MMAC: end-to-end numerical test at the operator's shapes.
//
// This is the first thing to run once a gfx946 part is available.  It calls
// v_mmac_f32_16x16x64_fp4 directly, so reaching the comparison at all proves
// the native MMAC executed -- there is no decoded path to fall back to.
//
// Covers, against an exact CPU reference:
//   1. fragment layout          (uniform scales, must be bit exact)
//   2. accumulator column map   (both candidate mappings are tried, one must fit)
//   3. per-scale-group masking and the FP32 restore (random scales)
//   4. H = 16 / 32 / 64, i.e. HT = 1 / 2 / 4
//   5. zero / tail fragments    (a row of all-zero codes must give exactly 0)
//
// Build: hipcc -x hip --offload-arch=gfx946 -O2 fp4_mmac_probe.cu -o probe
// Run  : ./probe        (on a non-gfx946 device it reports SKIP_RUN and exits 0)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>

using f32x4 = float __attribute__((ext_vector_type(4)));
using i32x2 = int __attribute__((ext_vector_type(2)));

static constexpr int kRowBytes = 64;    // 128 packed E2M1 channels
static constexpr int kChannels = 128;
static constexpr int kPos = 16;         // KV positions per tile (MMAC M)
static constexpr int kSteps = 2;        // K=64 steps covering 128 channels
static constexpr int kScaleGroups = 4;  // per-32 scale groups per row
static constexpr int kMaxH = 64;

__host__ __device__ static inline float e2m1(unsigned code) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[code & 7u];
    return (code & 8u) ? -m : m;
}
static inline unsigned nibble(const unsigned char* row, int c) {
    return (row[c / 2] >> ((c & 1) * 4)) & 15u;
}

// ---------------------------------------------------------------- device
// Mirrors paged_mqa_logits_fp4.cu's native block: four masked K=64 MMACs per
// head tile, each confined to one scale group, restored on the FP32 partial sum.
template <int HT>
__global__ void score_native(const unsigned char* __restrict__ kv,
                             const unsigned char* __restrict__ q,
                             const unsigned* __restrict__ k_sf,
                             const unsigned* __restrict__ q_sf,
                             const float* __restrict__ w,
                             float* __restrict__ out,
                             int accmap) {
#if defined(__gfx946__)
    const int lane = threadIdx.x % 64;
    const int row = lane % 16, group = lane / 16;
    const unsigned ksf = k_sf[row];

    float dot[HT][4];
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j) dot[h][j] = 0.0f;

#pragma unroll
    for (int sg = 0; sg < kScaleGroups; ++sg) {
        const int step = sg >> 1, half = sg & 1;
        const int keep = ((group >> 1) == half) ? -1 : 0;
        const unsigned sk = (ksf >> (sg * 8)) & 255u;

        const i32x2 kraw =
            *reinterpret_cast<const i32x2*>(kv + row * kRowBytes + step * 32 + group * 8);
        const i32x2 ka{kraw[0] & keep, kraw[1] & keep};

#pragma unroll
        for (int h = 0; h < HT; ++h) {
            const i32x2 qf = *reinterpret_cast<const i32x2*>(
                q + (h * 16 + row) * kRowBytes + step * 32 + group * 8);
            f32x4 acc = f32x4{0.f, 0.f, 0.f, 0.f};
            acc = __builtin_hcu_mmac_f32_16x16x64_fp4_lit_lts(ka, qf, acc, false, false);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // accmap 0: column = j*4 + group   accmap 1: column = group*4 + j
                const int head = accmap == 0 ? h * 16 + j * 4 + group : h * 16 + group * 4 + j;
                const unsigned sq = (q_sf[head] >> (sg * 8)) & 255u;
                dot[h][j] += ldexpf(acc[j], int(sk) + int(sq) - 254);
            }
        }
    }

    float sum = 0.0f;
#pragma unroll
    for (int h = 0; h < HT; ++h)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int head = accmap == 0 ? h * 16 + j * 4 + group : h * 16 + group * 4 + j;
            sum = fmaf(fmaxf(dot[h][j], 0.0f), w[head], sum);
        }
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    if (group == 0) out[row] = sum;
#else
    if (threadIdx.x == 0) out[0] = __builtin_nanf("");
    (void)kv; (void)q; (void)k_sf; (void)q_sf; (void)w; (void)accmap;
#endif
}

// ----------------------------------------------------------------- host
struct Case {
    const char* name;
    bool uniform_scales;
    bool zero_payload;
};

static int run(int H, const Case& c, int accmap, double* worst_out) {
    const int HT = H / 16;
    std::mt19937 rng(0xC0FFEEu + H * 131 + (c.uniform_scales ? 7 : 0) + (c.zero_payload ? 13 : 0));
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(120, 134);

    static unsigned char kpay[kPos][kRowBytes], qpay[kMaxH][kRowBytes];
    static unsigned ksf[kPos], qsf[kMaxH];
    static float weight[kMaxH];

    for (int t = 0; t < kPos; ++t) {
        for (int b = 0; b < kRowBytes; ++b)
            kpay[t][b] = c.zero_payload ? 0 : (unsigned char)byte_dist(rng);
        unsigned packed = 0;
        for (int s = 0; s < kScaleGroups; ++s)
            packed |= unsigned(c.uniform_scales ? 127 : scale_dist(rng)) << (s * 8);
        ksf[t] = packed;
    }
    for (int h = 0; h < H; ++h) {
        for (int b = 0; b < kRowBytes; ++b) qpay[h][b] = (unsigned char)byte_dist(rng);
        unsigned packed = 0;
        for (int s = 0; s < kScaleGroups; ++s)
            packed |= unsigned(c.uniform_scales ? 127 : scale_dist(rng)) << (s * 8);
        qsf[h] = packed;
        weight[h] = float((h % 7) - 3) * 0.25f;
    }

    // exact reference, double accumulation
    static double ref[kPos];
    for (int t = 0; t < kPos; ++t) {
        double acc = 0.0;
        for (int h = 0; h < H; ++h) {
            double d = 0.0;
            for (int ch = 0; ch < kChannels; ++ch) {
                const int s = ch / 32;
                const int ek = int((ksf[t] >> (s * 8)) & 255u) - 127;
                const int eq = int((qsf[h] >> (s * 8)) & 255u) - 127;
                d += double(e2m1(nibble(kpay[t], ch))) * ldexp(1.0, ek)
                   * double(e2m1(nibble(qpay[h], ch))) * ldexp(1.0, eq);
            }
            acc += (d > 0.0 ? d : 0.0) * double(weight[h]);
        }
        ref[t] = acc;
    }

    unsigned char *dk, *dq; unsigned *dks, *dqs; float *dw, *dout;
    hipMalloc(&dk, sizeof(kpay)); hipMalloc(&dq, sizeof(qpay));
    hipMalloc(&dks, sizeof(ksf)); hipMalloc(&dqs, sizeof(qsf));
    hipMalloc(&dw, sizeof(weight)); hipMalloc(&dout, kPos * sizeof(float));
    hipMemcpy(dk, kpay, sizeof(kpay), hipMemcpyHostToDevice);
    hipMemcpy(dq, qpay, sizeof(qpay), hipMemcpyHostToDevice);
    hipMemcpy(dks, ksf, sizeof(ksf), hipMemcpyHostToDevice);
    hipMemcpy(dqs, qsf, sizeof(qsf), hipMemcpyHostToDevice);
    hipMemcpy(dw, weight, sizeof(weight), hipMemcpyHostToDevice);
    hipMemset(dout, 0, kPos * sizeof(float));

    switch (HT) {
        case 1: hipLaunchKernelGGL((score_native<1>), dim3(1), dim3(64), 0, 0, dk, dq, dks, dqs, dw, dout, accmap); break;
        case 2: hipLaunchKernelGGL((score_native<2>), dim3(1), dim3(64), 0, 0, dk, dq, dks, dqs, dw, dout, accmap); break;
        case 4: hipLaunchKernelGGL((score_native<4>), dim3(1), dim3(64), 0, 0, dk, dq, dks, dqs, dw, dout, accmap); break;
        default: return 1;
    }
    const hipError_t st = hipDeviceSynchronize();
    float got[kPos];
    hipMemcpy(got, dout, sizeof(got), hipMemcpyDeviceToHost);
    hipFree(dk); hipFree(dq); hipFree(dks); hipFree(dqs); hipFree(dw); hipFree(dout);
    if (st != hipSuccess) { printf("    launch failed: %s\n", hipGetErrorString(st)); return 1; }

    double worst = 0.0;
    int bad = 0;
    for (int t = 0; t < kPos; ++t) {
        const double d = std::fabs(double(got[t]) - ref[t]);
        const double rel = ref[t] != 0.0 ? d / std::fabs(ref[t]) : d;
        if (rel > worst) worst = rel;
        if (!(rel <= 1e-5)) ++bad;
    }
    *worst_out = worst;
    return bad;
}

int main() {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("no HIP device\n");
        return 0;
    }
    if (std::strncmp(prop.gcnArchName, "gfx946", 6) != 0) {
        printf("SKIP_RUN: device is %s; the native FP4 MMAC needs gfx946.\n", prop.gcnArchName);
        printf("          (compiling this file for gfx946 already checks the build)\n");
        return 0;
    }
    printf("device %s\n", prop.gcnArchName);

    const Case cases[] = {
        {"uniform scales (layout, bit exact)", true, false},
        {"random scales  (mask + restore)   ", false, false},
        {"zero payload   (must be exactly 0)", true, true},
    };

    // Settle the accumulator column mapping once, on the strictest case.
    int accmap = -1;
    for (int m = 0; m < 2 && accmap < 0; ++m) {
        double w = 0.0;
        if (run(32, cases[0], m, &w) == 0 && w == 0.0) accmap = m;
    }
    if (accmap < 0) {
        printf("LAYOUT_MISMATCH: neither accumulator mapping reproduces the reference.\n");
        printf("  Fix the fragment/accumulator assumption before trusting any numbers.\n");
        return 1;
    }
    printf("accumulator column mapping: %s\n",
           accmap == 0 ? "n = j*4 + group  (as assumed in the kernel)" : "n = group*4 + j");

    int failures = 0;
    for (int H : {16, 32, 64}) {
        for (const Case& c : cases) {
            double worst = 0.0;
            const int bad = run(H, c, accmap, &worst);
            const bool exact_required = c.uniform_scales;
            const bool ok = bad == 0 && (!exact_required || worst == 0.0);
            printf("  H=%-2d  %s : %s (worst rel %.3g)\n", H, c.name,
                   ok ? "PASS" : "FAIL", worst);
            if (!ok) ++failures;
        }
    }

    printf("%s\n", failures == 0 ? "NATIVE_FP4_VERIFIED" : "NATIVE_FP4_FAILED");
    return failures != 0;
}
