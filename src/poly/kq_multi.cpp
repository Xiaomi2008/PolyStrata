// src/poly/kq_multi.cpp - see the header.  Compiled with AVX2, FMA and F16C (CMakeLists.txt) and called only where
// kq_multi_ok() says the CPU has them.
//
// Each function is the __AVX2__ branch of ggml_vec_dot_qN_K_q8_K (ggml-cpu/arch/x86/quants.c) with the loop over
// tokens inside the loop over blocks: what depends only on the weights (the unpacked quants, the scales) is made
// once a block, and what follows for a token is ggml's code for it, in its order.
#include "kq_multi.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml-cpu.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::poly {
namespace {

constexpr int MAXT = kMultiTokens;

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

// ggml's helpers (ggml-cpu/arch/x86/quants.c), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
alignas(32) const uint8_t k_shuffle_q3k[128] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};
inline __m256i get_scale_shuffle_q3k(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle_q3k + i); }
alignas(32) const uint8_t k_shuffle_k4[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle_k4 + i); }

// ---- Q2_K x Q8_K
void q2k_dot(const block_q2_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].data.data.d), xdmin = h2f(x[i].data.data.dmin);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        __m256i q2[2][4], sc[2][4];
        const uint8_t* q2p = x[i].qs;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*) q2p);
            q2p += 32;
            q2[j][0] = _mm256_and_si256(q2bits, m3);
            q2[j][1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            q2[j][2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            q2[j][3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
            for (int k = 0; k < 4; ++k) sc[j][k] = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(k));
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*) y->bsums));
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 128; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p0 = _mm256_maddubs_epi16(q2[j][0], q8_0);
                __m256i p1 = _mm256_maddubs_epi16(q2[j][1], q8_1);
                __m256i p2 = _mm256_maddubs_epi16(q2[j][2], q8_2);
                __m256i p3 = _mm256_maddubs_epi16(q2[j][3], q8_3);
                p0 = _mm256_madd_epi16(sc[j][0], p0);
                p1 = _mm256_madd_epi16(sc[j][1], p1);
                p2 = _mm256_madd_epi16(sc[j][2], p2);
                p3 = _mm256_madd_epi16(sc[j][3], p3);
                p0 = _mm256_add_epi32(p0, p1);
                p2 = _mm256_add_epi32(p2, p3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- Q3_K x Q8_K
void q3k_dot(const block_q3_K* x, int nb, const void* const* act, int nt, float* res) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    uint32_t aux[3];
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        std::memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32((int) (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
                                          (int) (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
                                          (int) ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
                                          (int) ((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].hmask);
        __m256i q3l[2][4], q3h[2][4], sc[2][4];
        const uint8_t* q3 = x[i].qs;
        int bit = 0;
        const int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*) q3);
            q3 += 32;
            for (int k = 0; k < 4; ++k) {
                q3l[j][k] = k == 0 ? _mm256_and_si256(q3bits, m3)
                                   : _mm256_and_si256(_mm256_srli_epi16(q3bits, 2 * k), m3);
                q3h[j][k] = _mm256_slli_epi16(
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                ++bit;
                sc[j][k] = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + k));
            }
            // ggml keeps `is` at 0: scales[j] already holds the half of the scales for j
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const int8_t* q8 = y->qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 128; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i q8s_0 = _mm256_maddubs_epi16(q3h[j][0], q8_0);
                __m256i q8s_1 = _mm256_maddubs_epi16(q3h[j][1], q8_1);
                __m256i q8s_2 = _mm256_maddubs_epi16(q3h[j][2], q8_2);
                __m256i q8s_3 = _mm256_maddubs_epi16(q3h[j][3], q8_3);
                __m256i p16_0 = _mm256_maddubs_epi16(q3l[j][0], q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q3l[j][1], q8_1);
                __m256i p16_2 = _mm256_maddubs_epi16(q3l[j][2], q8_2);
                __m256i p16_3 = _mm256_maddubs_epi16(q3l[j][3], q8_3);
                p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                p16_3 = _mm256_sub_epi16(p16_3, q8s_3);
                p16_0 = _mm256_madd_epi16(sc[j][0], p16_0);
                p16_1 = _mm256_madd_epi16(sc[j][1], p16_1);
                p16_2 = _mm256_madd_epi16(sc[j][2], p16_2);
                p16_3 = _mm256_madd_epi16(sc[j][3], p16_3);
                p16_0 = _mm256_add_epi32(p16_0, p16_1);
                p16_2 = _mm256_add_epi32(p16_2, p16_3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- Q4_K x Q8_K
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].data.data.d), xdmin = h2f(x[i].data.data.dmin);
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

bool cpu_has_avx2_fma_f16c() {
    int r[4];
#if defined(_MSC_VER)
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1, osxsave = (r[2] >> 27) & 1;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1;
    const bool ymm = osxsave && (_xgetbv(0) & 6) == 6;
    return fma && f16c && avx2 && ymm;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
}

}  // namespace

bool kq_multi_ok(int type) {
    static const bool ok = cpu_has_avx2_fma_f16c() && ggml_cpu_has_avx2() && ggml_cpu_has_fma() &&
                           std::getenv("STRATA_POLY_NO_KQ_MULTI") == nullptr;
    return ok && (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K);
}

void kq_multi_dot(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case GGML_TYPE_Q2_K: q2k_dot((const block_q2_K*) row, n / QK_K, act, nt, res); break;
        case GGML_TYPE_Q3_K: q3k_dot((const block_q3_K*) row, n / QK_K, act, nt, res); break;
        case GGML_TYPE_Q4_K: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        default: break;
    }
}

}  // namespace strata::poly
