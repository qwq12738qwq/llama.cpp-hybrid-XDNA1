// test_async_backend.cpp -- Phase 20G step 8 verification.
//
// Runs an I8 MUL_MAT through the XDNA ggml backend and checks the async
// contract: graph_compute must return before the NPU finishes (dst still
// unwritten), and synchronize / event_wait must complete it with correct values.
//
// Build (from repo root):
//   g++ -O2 -std=c++17 experiments/phase20g/test_async_backend.cpp -o experiments/phase20g/test_async_backend \
//       -I ggml/include -I ggml/src -L build-new/bin \
//       -lggml-xdna -lggml -lggml-base -lggml-cpu -Wl,-rpath,build-new/bin
//
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(void) {
    const int M = 64, K = 1024, N = 64;

    ggml_backend_t backend = ggml_backend_xdna_init();
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 1; }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);

    struct ggml_init_params params = { 256 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, N); // weight [K,N]
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M); // input  [K,M]
    struct ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);
    fprintf(stderr, "[test] dst type=%d (F32=%d)\n", (int) dst->type, (int) GGML_TYPE_F32);

    bool supported = ggml_backend_dev_supports_op(dev, dst);
    fprintf(stderr, "[test] supports_op=%d\n", (int) supported);
    if (!supported) { fprintf(stderr, "op not supported\n"); return 1; }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t b0 = ggml_backend_alloc_buffer(backend, ggml_nbytes(src0));
    ggml_backend_buffer_t b1 = ggml_backend_alloc_buffer(backend, ggml_nbytes(src1));
    ggml_backend_buffer_t bd = ggml_backend_alloc_buffer(backend, ggml_nbytes(dst));
    ggml_backend_tensor_alloc(b0, src0, ggml_backend_buffer_get_base(b0));
    ggml_backend_tensor_alloc(b1, src1, ggml_backend_buffer_get_base(b1));
    ggml_backend_tensor_alloc(bd, dst, ggml_backend_buffer_get_base(bd));

    std::vector<int8_t> w((size_t) K * N), in((size_t) K * M);
    for (int i = 0; i < K * N; i++) w[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (int i = 0; i < K * M; i++) in[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    ggml_backend_tensor_set(src0, w.data(), 0, w.size());
    ggml_backend_tensor_set(src1, in.data(), 0, in.size());

    // CPU reference: out[n + m*N] = sum_k w[k + n*K] * in[k + m*K]
    std::vector<float> ref((size_t) N * M, 0.0f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t) w[k + n * K] * (int32_t) in[k + m * K];
            ref[n + m * N] = (float) acc;
        }

    // warmup (first call includes XRT init / first kernel load)
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute_async(backend, gf);
        ggml_backend_synchronize(backend);
    }

    // ---- async run, measured over several iterations ----
    const int ITERS = 5;
    uint64_t sum_submit = 0, sum_sync = 0;
    bool early_unwritten = true;
    enum ggml_status s = GGML_STATUS_SUCCESS;
    std::vector<float> zero((size_t) N * M, 0.0f);
    for (int it = 0; it < ITERS; it++) {
        ggml_backend_tensor_set(dst, zero.data(), 0, zero.size() * sizeof(float));
        uint64_t t0 = now_ns();
        s = ggml_backend_graph_compute_async(backend, gf);
        uint64_t t1 = now_ns();

        // dst must still be unwritten: output is only copied back on drain.
        std::vector<float> early((size_t) N * M, -12345.0f);
        ggml_backend_tensor_get(dst, early.data(), 0, early.size() * sizeof(float));
        for (float v : early) if (v != 0.0f) { early_unwritten = false; break; }

        ggml_backend_synchronize(backend);
        uint64_t t2 = now_ns();
        sum_submit += (t1 - t0);
        sum_sync += (t2 - t1);
    }
    uint64_t avg_submit = sum_submit / ITERS, avg_sync = sum_sync / ITERS;

    std::vector<float> result((size_t) N * M);
    ggml_backend_tensor_get(dst, result.data(), 0, result.size() * sizeof(float));

    long bad = 0;
    for (size_t i = 0; i < result.size(); i++) if (result[i] != ref[i]) bad++;

    fprintf(stderr, "[test] graph_compute_async avg %llu ns (no NPU wait)\n",
            (unsigned long long) avg_submit);
    fprintf(stderr, "[test] synchronize avg %llu ns (NPU execution)\n",
            (unsigned long long) avg_sync);
    fprintf(stderr, "[test] dst unwritten right after submit: %s\n", early_unwritten ? "yes" : "NO");
    fprintf(stderr, "[test] mismatches %ld / %d\n", bad, N * M);

    bool ok = (s == GGML_STATUS_SUCCESS) && early_unwritten && (bad == 0);

    // ---- event contract: record after submit, synchronize the event to finish ----
    bool ev_unwritten = false;
    long ev_bad = -1;
    {
        ggml_backend_event_t ev = ggml_backend_event_new(dev);
        ggml_backend_tensor_set(dst, zero.data(), 0, zero.size() * sizeof(float));
        ggml_backend_graph_compute_async(backend, gf);
        ggml_backend_event_record(ev, backend);

        std::vector<float> ev_early((size_t) N * M, 0.0f);
        ggml_backend_tensor_get(dst, ev_early.data(), 0, ev_early.size() * sizeof(float));
        ev_unwritten = true;
        for (float v : ev_early) if (v != 0.0f) { ev_unwritten = false; break; }

        ggml_backend_event_synchronize(ev);
        std::vector<float> ev_res((size_t) N * M, 0.0f);
        ggml_backend_tensor_get(dst, ev_res.data(), 0, ev_res.size() * sizeof(float));
        ev_bad = 0;
        for (size_t i = 0; i < ev_res.size(); i++) if (ev_res[i] != ref[i]) ev_bad++;
        fprintf(stderr, "[test] event: unwritten_before=%d mismatches=%ld\n",
                (int) ev_unwritten, ev_bad);
        ggml_backend_event_free(ev);
    }
    ok = ok && ev_unwritten && (ev_bad == 0);

    printf("ASYNC_BACKEND %s mismatches=%ld early_unwritten=%d event_ok=%d avg_submit_ns=%llu avg_sync_ns=%llu\n",
           ok ? "PASS" : "FAIL", bad, (int) early_unwritten, (int)(ev_unwritten && ev_bad == 0),
           (unsigned long long) avg_submit, (unsigned long long) avg_sync);



    ggml_backend_buffer_free(b0);
    ggml_backend_buffer_free(b1);
    ggml_backend_buffer_free(bd);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 2;
}
