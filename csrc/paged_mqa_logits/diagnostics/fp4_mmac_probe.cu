// gfx946 native FP4 MMAC fragment-layout probe.
//
// Isolates the fragment mapping of v_mmac_f32_16x16x64_fp4 (no scale), so a
// failure here cannot be confused with a scale-buffer problem.
//
// Layout under test, derived from DTK crt/mma_gfx946.hpp load64() for packed
// __nv_fp4x2_e2m1 input with even ldm:
//   element_base = (lane & 15) * ldm + (lane >> 4) * 16
//   byte_base    = element_base / 2 = (lane & 15) * (ldm/2) + (lane >> 4) * 8
//   fragment     = 8 consecutive bytes at byte_base  (one global_load_dwordx2)
//
// Build:  hipcc -x hip --offload-arch=gfx946 -O2 fp4_mmac_probe.cu -o probe
// Run  :  requires gfx946; on other devices the launch is skipped.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

using f32x4 = float __attribute__((ext_vector_type(4)));
using i32x2 = int __attribute__((ext_vector_type(2)));

static constexpr int kM = 16, kN = 16, kK = 64;
static constexpr int kLdm = kK;          // elements per row
static constexpr int kRowBytes = kLdm / 2;

__host__ __device__ inline float e2m1_to_float(unsigned code) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[code & 7u];
    return (code & 8u) ? -m : m;
}

__global__ void fp4_mmac_k64(const unsigned char* __restrict__ a,
                             const unsigned char* __restrict__ b,
                             float* __restrict__ out) {
#if defined(__gfx946__)
    const int lane = threadIdx.x;
    const int off = (lane & 15) * kRowBytes + (lane >> 4) * 8;
    const i32x2 fa = *reinterpret_cast<const i32x2*>(a + off);
    const i32x2 fb = *reinterpret_cast<const i32x2*>(b + off);
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    acc = __builtin_hcu_mmac_f32_16x16x64_fp4_lit_lts(fa, fb, acc, false, false);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[lane * 4 + j] = acc[j];
#else
    if (threadIdx.x == 0) out[0] = -1.0f;
#endif
}

int main() {
    const size_t bytes = size_t(kM) * kRowBytes;
    unsigned char ha[kM * kRowBytes], hb[kN * kRowBytes];
    for (int r = 0; r < kM; ++r)
        for (int c = 0; c < kRowBytes; ++c) {
            ha[r * kRowBytes + c] = (unsigned char)(((r + c) & 15u) | ((((r * 3 + c * 5 + 7) & 15u)) << 4));
            hb[r * kRowBytes + c] = (unsigned char)(((c + 2 * r) & 15u) | ((((r + c * 7) & 15u)) << 4));
        }

    static float ref[kM][kN];
    for (int m = 0; m < kM; ++m)
        for (int n = 0; n < kN; ++n) {
            double s = 0.0;
            for (int k = 0; k < kK; ++k) {
                const unsigned av = (ha[m * kRowBytes + k / 2] >> ((k & 1) * 4)) & 15u;
                const unsigned bv = (hb[n * kRowBytes + k / 2] >> ((k & 1) * 4)) & 15u;
                s += double(e2m1_to_float(av)) * double(e2m1_to_float(bv));
            }
            ref[m][n] = float(s);
        }

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("no HIP device; reference only\n");
    } else if (strncmp(prop.gcnArchName, "gfx946", 6) != 0) {
        printf("SKIP_RUN: device is %s, probe needs gfx946 (compile-only check passed)\n",
               prop.gcnArchName);
        printf("REF[0][0..3] = %.1f %.1f %.1f %.1f\n", ref[0][0], ref[0][1], ref[0][2], ref[0][3]);
        printf("PROBE_BUILD_OK\n");
        return 0;
    }

    unsigned char *da = nullptr, *db = nullptr; float* dout = nullptr;
    hipMalloc(&da, bytes); hipMalloc(&db, bytes); hipMalloc(&dout, 64 * 4 * sizeof(float));
    hipMemcpy(da, ha, bytes, hipMemcpyHostToDevice);
    hipMemcpy(db, hb, bytes, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(fp4_mmac_k64, dim3(1), dim3(64), 0, 0, da, db, dout);
    if (hipDeviceSynchronize() != hipSuccess) { printf("LAUNCH_FAIL\n"); return 1; }
    float hout[64 * 4];
    hipMemcpy(hout, dout, sizeof(hout), hipMemcpyDeviceToHost);

    // Candidate accumulator mapping: lane -> (row = lane&15), entry j -> column.
    // Report the first mismatch against every plausible column mapping so the
    // real one can be identified from one run.
    int ok_a = 0, ok_b = 0;
    for (int lane = 0; lane < 64; ++lane)
        for (int j = 0; j < 4; ++j) {
            const float got = hout[lane * 4 + j];
            const int row = lane & 15, grp = lane >> 4;
            if (got == ref[row][j * 4 + grp]) ++ok_a;   // n = j*4 + grp
            if (got == ref[row][grp * 4 + j]) ++ok_b;   // n = grp*4 + j
        }
    printf("accumulator mapping matches: n=j*4+grp -> %d/256, n=grp*4+j -> %d/256\n", ok_a, ok_b);
    printf("%s\n", (ok_a == 256 || ok_b == 256) ? "LAYOUT_CONFIRMED" : "LAYOUT_MISMATCH");
    return 0;
}
