// src/poly/kq_multi_test.cpp - kq_multi_dot() against ggml-cpu's vec_dot, token by token, bit for bit: random
// Q2_K, Q3_K and Q4_K rows (quantized by ggml from random floats) against 1 to 8 tokens' Q8_K activations.
#include "kq_multi.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    ggml_cpu_init();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0, checked = 0;
    for (const int type : {(int) GGML_TYPE_Q2_K, (int) GGML_TYPE_Q3_K, (int) GGML_TYPE_Q4_K}) {
        if (!strata::poly::kq_multi_ok(type)) {
            std::printf("type %d: not taken here\n", type);
            return 1;
        }
        const ggml_type_traits_cpu* t = ggml_get_type_traits_cpu((ggml_type) type);
        const ggml_type_traits_cpu* a = ggml_get_type_traits_cpu(t->vec_dot_type);
        for (const int n : {2048, 4096}) {
            for (int rep = 0; rep < 50; ++rep) {
                std::vector<float> w((size_t) n), x((size_t) n);
                for (float& v : w) v = nd(rng) * (rep % 3 == 0 ? 0.01f : 1.f);
                std::vector<uint8_t> row(ggml_row_size((ggml_type) type, n));
                t->from_float(w.data(), row.data(), n);
                const size_t ab = ggml_row_size(t->vec_dot_type, n);
                std::vector<uint8_t> acts(ab * strata::poly::kMultiTokens);
                const void* as[strata::poly::kMultiTokens];
                for (int k = 0; k < strata::poly::kMultiTokens; ++k) {
                    for (float& v : x) v = nd(rng) * (k + 1);
                    a->from_float(x.data(), acts.data() + ab * k, n);
                    as[k] = acts.data() + ab * k;
                }
                for (int nt = 1; nt <= strata::poly::kMultiTokens; ++nt) {
                    float got[strata::poly::kMultiTokens];
                    strata::poly::kq_multi_dot(type, row.data(), n, as, nt, got);
                    for (int k = 0; k < nt; ++k) {
                        float want = 0.f;
                        t->vec_dot(n, &want, 0, row.data(), 0, as[k], 0, 1);
                        ++checked;
                        if (std::memcmp(&want, &got[k], 4) != 0 && bad++ < 10)
                            std::printf("type %d n %d nt %d token %d: %.9g, ggml %.9g\n", type, n, nt, k, got[k], want);
                    }
                }
            }
        }
    }
    std::printf("%d of %d dot products differ from ggml's\n", bad, checked);
    return bad != 0;
}
