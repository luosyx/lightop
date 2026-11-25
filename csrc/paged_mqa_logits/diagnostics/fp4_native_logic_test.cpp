// Host-only proof of the gfx946 native-FP4 scoring algorithm for
// paged_mqa_logits_fp4, with per-scale-group masking and FP32 rescale.
//
// The hardware MMAC is modelled exactly (E2M1 products summed in FP32) over
// the fragment layout documented by DTK crt/mma_gfx946.hpp load64():
//     byte_base = (lane & 15) * (ldm/2) + (lane >> 4) * 8
// Scales are kept in a range where every partial sum is exactly representable,
// so the comparison below demands bit-exact equality.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>

static constexpr int kRowBytes = 64;      // 128 packed E2M1 channels
static constexpr int kChannels = 128;
static constexpr int kPos = 16;           // KV positions in a tile  (MMAC M)
static constexpr int kHeadsTile = 16;     // heads per head-tile     (MMAC N)
static constexpr int kSteps = 2;          // K=64 steps covering 128 channels
static constexpr int kScaleGroups = 4;    // per-32 scale groups per row
static constexpr int kH = 32;             // heads
static constexpr int kHT = kH / 16;

static float e2m1(unsigned code) {
    static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[code & 7u];
    return (code & 8u) ? -m : m;
}
static unsigned nibble(const uint8_t* row, int c) {
    return (row[c / 2] >> ((c & 1) * 4)) & 15u;
}

// ---- layout helpers, mirroring paged_mqa_logits_fp4_native.hpp -------------
static int frag_offset(int step, int group) { return step * (kRowBytes / kSteps) + group * 8; }
static int scale_group(int step, int group) { return 2 * step + (group >> 1); }

struct Frag { uint8_t b[8]; };
static Frag load_frag(const uint8_t* row, int step, int group, int half) {
    Frag f{};
    if ((group >> 1) == half) std::memcpy(f.b, row + frag_offset(step, group), 8);
    return f;  // masked lanes contribute zero
}

// Exact model of v_mmac_f32_16x16x64_fp4: C[m][n] = sum_k A[m][k] * B[n][k].
// acc[lane][j] holds C[lane&15][ j*4 + (lane>>4) ], the accumulator mapping
// this kernel already relies on for the 16x16x32 MMACs.
static void mmac_sim(const Frag A[64], const Frag B[64], float acc[64][4]) {
    float C[16][16];
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            float s = 0.f;
            for (int g = 0; g < 4; ++g)
                for (int b = 0; b < 8; ++b)
                    for (int half = 0; half < 2; ++half) {
                        const unsigned av = (A[g * 16 + m].b[b] >> (half * 4)) & 15u;
                        const unsigned bv = (B[g * 16 + n].b[b] >> (half * 4)) & 15u;
                        s += e2m1(av) * e2m1(bv);
                    }
            C[m][n] = s;
        }
    for (int lane = 0; lane < 64; ++lane) {
        const int m = lane & 15, g = lane >> 4;
        for (int j = 0; j < 4; ++j) acc[lane][j] = C[m][j * 4 + g];
    }
}

// ---- path A: K=32 mixed-format, one scale group per step ------------------
static int frag_offset_k32(int step, int group) { return step * 16 + group * 4; }

struct FragA { uint8_t b[8]; };  // padded: one code per byte
static FragA load_frag_k32(const uint8_t* row, int step, int group) {
    FragA f{};
    const uint8_t* p = row + frag_offset_k32(step, group);
    for (int i = 0; i < 4; ++i) {
        f.b[2 * i] = p[i] & 15u;
        f.b[2 * i + 1] = (p[i] >> 4) & 15u;
    }
    return f;
}

// Exact model of v_mmac_f32_16x16x32_f8f6f4 with fmt = FP4: lane (row, g) holds
// elements [8g, 8g+8) of the 32-element window, one code per byte.
static void mmac_sim_k32(const FragA A[64], const FragA B[64], float acc[64][4]) {
    float C[16][16];
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            float s = 0.f;
            for (int g = 0; g < 4; ++g)
                for (int i = 0; i < 8; ++i)
                    s += e2m1(A[g * 16 + m].b[i]) * e2m1(B[g * 16 + n].b[i]);
            C[m][n] = s;
        }
    for (int lane = 0; lane < 64; ++lane) {
        const int m = lane & 15, g = lane >> 4;
        for (int j = 0; j < 4; ++j) acc[lane][j] = C[m][j * 4 + g];
    }
}

static int run_case(bool uniform_scales) {
    std::mt19937 rng(20261125);
    std::uniform_int_distribution<int> code(0, 255);
    // Scale bytes near the 127 bias keep all partial sums exact.
    std::uniform_int_distribution<int> sdist(120, 134);

    static uint8_t kpay[kPos][kRowBytes], qpay[kH][kRowBytes];
    static uint8_t ksf[kPos][kScaleGroups], qsf[kH][kScaleGroups];
    static float weight[kH];
    for (int t = 0; t < kPos; ++t) {
        for (int b = 0; b < kRowBytes; ++b) kpay[t][b] = uint8_t(code(rng));
        for (int s = 0; s < kScaleGroups; ++s) ksf[t][s] = uint8_t(uniform_scales ? 127 : sdist(rng));
    }
    for (int h = 0; h < kH; ++h) {
        for (int b = 0; b < kRowBytes; ++b) qpay[h][b] = uint8_t(code(rng));
        for (int s = 0; s < kScaleGroups; ++s) qsf[h][s] = uint8_t(uniform_scales ? 127 : sdist(rng));
        weight[h] = float((h % 7) - 3) * 0.25f;
    }

    // ---- reference ---------------------------------------------------------
    static double ref[kPos];
    for (int t = 0; t < kPos; ++t) {
        double acc = 0.0;
        for (int h = 0; h < kH; ++h) {
            double dot = 0.0;
            for (int c = 0; c < kChannels; ++c) {
                const int s = c / 32;
                dot += double(e2m1(nibble(kpay[t], c))) * ldexp(1.0, ksf[t][s] - 127)
                     * double(e2m1(nibble(qpay[h], c))) * ldexp(1.0, qsf[h][s] - 127);
            }
            acc += (dot > 0.0 ? dot : 0.0) * double(weight[h]);
        }
        ref[t] = acc;
    }

    // ---- algorithm under test ---------------------------------------------
    static float total[64][kHT][4];
    std::memset(total, 0, sizeof(total));
    for (int step = 0; step < kSteps; ++step)
        for (int half = 0; half < 2; ++half) {
            const int s = scale_group(step, half ? 2 : 0);  // = 2*step + half
            for (int h = 0; h < kHT; ++h) {
                Frag A[64], B[64];
                for (int lane = 0; lane < 64; ++lane) {
                    const int r = lane & 15, g = lane >> 4;
                    A[lane] = load_frag(kpay[r], step, g, half);
                    B[lane] = load_frag(qpay[h * 16 + r], step, g, half);
                }
                float acc[64][4];
                mmac_sim(A, B, acc);
                for (int lane = 0; lane < 64; ++lane) {
                    const int r = lane & 15, g = lane >> 4;
                    for (int j = 0; j < 4; ++j) {
                        const int head = h * 16 + j * 4 + g;
                        const int e = ksf[r][s] + qsf[head][s] - 254;
                        total[lane][h][j] += ldexpf(acc[lane][j], e);
                    }
                }
            }
        }

    // ---- path A: four K=32 MMACs, one scale group each, no masking --------
    static float totalA[64][kHT][4];
    std::memset(totalA, 0, sizeof(totalA));
    for (int step = 0; step < 4; ++step) {
        const int s = step;  // one step == one scale group
        for (int h = 0; h < kHT; ++h) {
            FragA A[64], B[64];
            for (int lane = 0; lane < 64; ++lane) {
                const int r = lane & 15, g = lane >> 4;
                A[lane] = load_frag_k32(kpay[r], step, g);
                B[lane] = load_frag_k32(qpay[h * 16 + r], step, g);
            }
            float acc[64][4];
            mmac_sim_k32(A, B, acc);
            for (int lane = 0; lane < 64; ++lane) {
                const int r = lane & 15, g = lane >> 4;
                for (int j = 0; j < 4; ++j) {
                    const int head = h * 16 + j * 4 + g;
                    totalA[lane][h][j] +=
                        ldexpf(acc[lane][j], ksf[r][s] + qsf[head][s] - 254);
                }
            }
        }
    }

    static float got[kPos];
    for (int r = 0; r < kPos; ++r) {
        float s = 0.f;
        for (int g = 0; g < 4; ++g)
            for (int h = 0; h < kHT; ++h)
                for (int j = 0; j < 4; ++j) {
                    const float v = total[g * 16 + r][h][j];
                    s += (v > 0.f ? v : 0.f) * weight[h * 16 + j * 4 + g];
                }
        got[r] = s;
    }

    static float gotA[kPos];
    for (int r = 0; r < kPos; ++r) {
        float s = 0.f;
        for (int g = 0; g < 4; ++g)
            for (int h = 0; h < kHT; ++h)
                for (int j = 0; j < 4; ++j) {
                    const float v = totalA[g * 16 + r][h][j];
                    s += (v > 0.f ? v : 0.f) * weight[h * 16 + j * 4 + g];
                }
        gotA[r] = s;
    }

    int bad = 0;
    double worstB = 0.0, worstA = 0.0;
    for (int t = 0; t < kPos; ++t) {
        const double relB = ref[t] != 0.0 ? std::fabs(double(got[t]) - ref[t]) / std::fabs(ref[t]) : 0.0;
        const double relA = ref[t] != 0.0 ? std::fabs(double(gotA[t]) - ref[t]) / std::fabs(ref[t]) : 0.0;
        if (relB > worstB) worstB = relB;
        if (relA > worstA) worstA = relA;
        if (!(relB <= 1e-5)) { if (bad < 4) printf("  B pos %2d rel %.3g\n", t, relB); ++bad; }
        if (!(relA <= 1e-5)) { if (bad < 4) printf("  A pos %2d rel %.3g\n", t, relA); ++bad; }
    }
    printf("[%-14s] path A (K=32) worst rel %.3g | path B (K=64 masked) worst rel %.3g\n",
           uniform_scales ? "uniform scales" : "random scales", worstA, worstB);
    return bad != 0;
}


int main() {
    // Case 1 isolates fragment indexing, masking and the accumulator mapping:
    // with every scale at the 127 bias the rescale step is a no-op.
    // Case 2 additionally exercises per-scale-group recombination.
    const int a = run_case(true);
    const int b = run_case(false);
    printf("%s\n", (a == 0 && b == 0) ? "LOGIC_VERIFIED" : "LOGIC_MISMATCH");
    return (a == 0 && b == 0) ? 0 : 1;
}
