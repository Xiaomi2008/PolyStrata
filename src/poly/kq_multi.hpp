// src/poly/kq_multi.hpp - one K-quant row against several tokens' Q8_K activations, bit for bit ggml-cpu's AVX2
// vec_dot for each token.
//
// cpu_experts() takes ggml's dot products a row and a token at a time.  A row that several tokens of a window
// routed to (the token the model wrote and the guesses after it) was unpacked once for each of them, and the
// unpacking of 2- and 3-bit weights is most of a dot product's work: a window of three tokens took 1.8 ms a layer
// of CPU experts where one token took 0.76.  These functions unpack each block once and run every token through
// ggml's own integer chain and float operations in ggml's order, so each result is the one vec_dot gives.
#pragma once

#include <cstdint>

namespace strata::poly {

constexpr int kMultiTokens = 8;   ///< the most tokens one call takes

/// Whether kq_multi_dot() takes rows of this ggml type (Q2_K, Q3_K, Q4_K) here: the CPU has AVX2, FMA and F16C,
/// and ggml-cpu was built with its AVX2 dot products, the ones these match.
bool kq_multi_ok(int ggml_type);

/// res[t] = row . act[t] for t < nt (1 to kMultiTokens); a row of n values, each act[t] n values of Q8_K.
void kq_multi_dot(int ggml_type, const uint8_t* row, int n, const void* const* act, int nt, float* res);

}  // namespace strata::poly
