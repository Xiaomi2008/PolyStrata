// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"
#include "kq_multi.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
// 32 and not 16 (local build): with 16 a task read 10 KB of each expert's down, too little for the hardware
// prefetcher to get going (GLM 5.3 Flash Q2_K, 4 experts: down 31.6 GB/s; with these and pf_rows 2, 45 GB/s)
const int kRowsPerDownTask = std::getenv("STRATA_POLY_DROWS") ? std::atoi(std::getenv("STRATA_POLY_DROWS")) : 32;
constexpr int kSpinsBeforeSleep = 20000;   // about a millisecond of pause instructions

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

// STRATA_POLY_CPU_TIMING=1: the phases of cpu_experts() summed, printed every 4,000 calls (local profiling)
struct Timing {
    bool on = std::getenv("STRATA_POLY_CPU_TIMING") != nullptr;
    double act = 0, gu = 0, hq = 0, down = 0, launch = 0, bytes = 0;
    long long calls = 0, experts = 0, runs = 0;
    double multi_ms = 0, multi_bytes = 0, one_ms = 0, one_bytes = 0;   // calls of several tokens, and of one
    long long multi_calls = 0;
};
Timing timing;
// Each row's dot products prefetch the row pf_rows on, in the task (STRATA_POLY_PF; 0 turns it off)
const int pf_rows = std::getenv("STRATA_POLY_PF") ? std::atoi(std::getenv("STRATA_POLY_PF")) : 2;
inline void prefetch_row(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i += 64) _mm_prefetch((const char*) p + i, _MM_HINT_T0);
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

Workers::Workers(int threads) {
    for (int i = 1; i < threads; ++i) th_.emplace_back([this] { loop(); });
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < kSpinsBeforeSleep) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
    while (users_.load() != 0) _mm_pause();
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd,
                 int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile) {
    init_once();
    const double t0 = timing.on ? now_ms() : 0;
    double t_launch = 0;
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    // ggml's quantizers for the activations are scalar: microseconds a row, which on one thread kept the others
    // waiting, here and before the down products
    uint8_t* actp = act.data();
    const auto act_from = traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    const double t1 = timing.on ? now_ms() : 0;

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const bool multi_gu = kq_multi_ok(gu_type), multi_d = kq_multi_ok(d_type);
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) {
            const double l0 = timing.on ? now_ms() : 0;
            (*meanwhile)();
            if (timing.on) t_launch = now_ms() - l0;
            return;
        }
        t -= first;
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        // several uses of the expert (a window's tokens): each row unpacked once for all of them, the same sums
        const bool multi = i1 - i0 > 1 && multi_gu;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * gu_row;
            if (pf_rows > 0 && r + pf_rows < r1) {
                prefetch_row(gate + (size_t) pf_rows * gu_row, gu_row);
                prefetch_row(up + (size_t) pf_rows * gu_row, gu_row);
            }
            float gm[kMultiTokens], um[kMultiTokens];
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                float g = 0.f, u = 0.f;
                if (multi) {
                    const int k = (i - i0) % kMultiTokens;
                    if (k == 0) {
                        const int m = std::min(kMultiTokens, i1 - i);
                        const void* as[kMultiTokens];
                        for (int q = 0; q < m; ++q) as[q] = actp + (size_t) experts[byp[i + q]].row * a_bytes;
                        kq_multi_dot(gu_type, gate, n_embd, as, m, gm);
                        kq_multi_dot(gu_type, up, n_embd, as, m, um);
                    }
                    g = gm[k], u = um[k];
                } else {
                    tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                    tg->vec_dot(n_embd, &u, 0, up, 0, a, 0, 1);
                }
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    const double t2 = timing.on ? now_ms() : 0;
    uint8_t* hqp = hq.data();
    const auto hq_from = traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });
    const double t3 = timing.on ? now_ms() : 0;

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use
        one.resize((size_t) n);
        float* s = one.data();
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        for (int r = r0; r < r1; ++r) {
            for (int u = 0; u < n_run; ++u) {
                const uint8_t* down = experts[byp[runp[u]]].down + (size_t) r * d_row;
                if (pf_rows > 0 && r + pf_rows < r1) prefetch_row(down + (size_t) pf_rows * d_row, d_row);
                if (multi_d && runp[u + 1] - runp[u] > 1) {
                    for (int i = runp[u]; i < runp[u + 1]; i += kMultiTokens) {
                        const int m = std::min(kMultiTokens, runp[u + 1] - i);
                        const void* as[kMultiTokens];
                        float sm[kMultiTokens];
                        for (int q = 0; q < m; ++q) as[q] = hqp + (size_t) byp[i + q] * h_bytes;
                        kq_multi_dot(d_type, down, n_ff, as, m, sm);
                        for (int q = 0; q < m; ++q) s[byp[i + q]] = sm[q];
                    }
                    continue;
                }
                for (int i = runp[u]; i < runp[u + 1]; ++i)
                    td->vec_dot(n_ff, s + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * s[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
    if (timing.on) {
        const double t4 = now_ms();
        Timing& T = timing;
        T.act += t1 - t0, T.gu += t2 - t1, T.hq += t3 - t2, T.down += t4 - t3, T.launch += t_launch;
        T.bytes += (double) n_run * (2.0 * (double) gu_row * n_ff + (double) d_row * n_embd);
        T.experts += n, T.runs += n_run;
        const double call_bytes = (double) n_run * (2.0 * (double) gu_row * n_ff + (double) d_row * n_embd);
        if (rows > 1) T.multi_ms += t4 - t0, T.multi_bytes += call_bytes, ++T.multi_calls;
        else T.one_ms += t4 - t0, T.one_bytes += call_bytes;
        if (++T.calls % 4000 == 0) {
            const double all = T.act + T.gu + T.hq + T.down;
            std::fprintf(stderr,
                         "cpu timing: %lld calls, %.2f experts/call (%.2f distinct); ms/call act %.3f gate-up %.3f "
                         "(launch task %.3f) hq %.3f down %.3f = %.3f; %.1f GB/s over the call\n",
                         T.calls, (double) T.experts / T.calls, (double) T.runs / T.calls, T.act / T.calls,
                         T.gu / T.calls, T.launch / T.calls, T.hq / T.calls, T.down / T.calls, all / T.calls,
                         T.bytes / all / 1e6);
            std::fprintf(stderr,
                         "cpu timing: calls of one token %.0f ms at %.1f GB/s; %lld calls of several %.0f ms at %.1f "
                         "GB/s\n",
                         T.one_ms, T.one_bytes / std::max(T.one_ms, 1e-9) / 1e6, T.multi_calls, T.multi_ms,
                         T.multi_bytes / std::max(T.multi_ms, 1e-9) / 1e6);
            T = Timing{};
            T.on = true;
        }
    }
}

}  // namespace strata::poly
