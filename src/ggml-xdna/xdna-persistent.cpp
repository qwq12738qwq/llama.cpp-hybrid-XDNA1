#include "xdna-persistent.h"
#include "xdna-buffer.h"
#include <cstring>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <xrt/deprecated/xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

extern std::atomic<int> g_xdna_persistent_execute_count;
extern std::atomic<int> g_xdna_xrt_submit_count;
extern std::atomic<int> g_xdna_xrt_complete_count;

// Phase 20E: per-stage timing so the per-block submit cost is measured, not guessed.
struct xdna_timings {
    std::atomic<uint64_t> ensure_ns{0};
    std::atomic<uint64_t> copyin_ns{0};
    std::atomic<uint64_t> memset_ns{0};
    std::atomic<uint64_t> syncin_ns{0};
    std::atomic<uint64_t> run_new_ns{0};
    std::atomic<uint64_t> wait_ns{0};
    std::atomic<uint64_t> syncout_ns{0};
    std::atomic<uint64_t> accum_ns{0};
    std::atomic<uint64_t> chunks{0};
    std::atomic<uint64_t> ops{0};
};

static xdna_timings g_xdna_timings;

static uint64_t xdna_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Phase 38C: exported steady-clock millisecond helper used by ggml-xdna.cpp
extern "C" double ggml_xdna_now_ms(void) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" void ggml_xdna_reset_timings(void) {
    g_xdna_timings.ensure_ns.store(0);
    g_xdna_timings.copyin_ns.store(0);
    g_xdna_timings.memset_ns.store(0);
    g_xdna_timings.syncin_ns.store(0);
    g_xdna_timings.run_new_ns.store(0);
    g_xdna_timings.wait_ns.store(0);
    g_xdna_timings.syncout_ns.store(0);
    g_xdna_timings.accum_ns.store(0);
    g_xdna_timings.chunks.store(0);
    g_xdna_timings.ops.store(0);
}

extern "C" void ggml_xdna_print_timings(void) {
    const uint64_t c = g_xdna_timings.chunks.load();
    const double ms = 1e6;
    if (c == 0) {
        fprintf(stderr, "[XDNA-TIME] no chunks executed\n");
        return;
    }
    fprintf(stderr, "[XDNA-TIME] ops=%llu chunks=%llu (%.1f chunk/op)\n",
            (unsigned long long)g_xdna_timings.ops.load(), (unsigned long long)c,
            (double)c / (double)g_xdna_timings.ops.load());
    fprintf(stderr, "[XDNA-TIME] per-chunk ms: ensure=%.3f copyin=%.3f memset=%.3f syncin=%.3f runnew=%.3f wait=%.3f syncout=%.3f accum=%.3f\n",
            g_xdna_timings.ensure_ns.load() / ms / c,
            g_xdna_timings.copyin_ns.load() / ms / c,
            g_xdna_timings.memset_ns.load() / ms / c,
            g_xdna_timings.syncin_ns.load() / ms / c,
            g_xdna_timings.run_new_ns.load() / ms / c,
            g_xdna_timings.wait_ns.load() / ms / c,
            g_xdna_timings.syncout_ns.load() / ms / c,
            g_xdna_timings.accum_ns.load() / ms / c);
    const double total_ms = (g_xdna_timings.ensure_ns.load() + g_xdna_timings.copyin_ns.load() +
                             g_xdna_timings.memset_ns.load() + g_xdna_timings.syncin_ns.load() +
                             g_xdna_timings.run_new_ns.load() + g_xdna_timings.wait_ns.load() +
                             g_xdna_timings.syncout_ns.load() + g_xdna_timings.accum_ns.load()) / ms;
    fprintf(stderr, "[XDNA-TIME] total=%.1f ms, %.3f ms per op\n", total_ms,
            total_ms / (double)g_xdna_timings.ops.load());
}

static bool load_instructions(const char * path, std::vector<uint32_t> & out_insts) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "[XDNA] Failed to open insts: %s\n", path);
        return false;
    }
    ifs.seekg(0, std::ios::end);
    size_t isz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out_insts.resize(isz / 4);
    ifs.read(reinterpret_cast<char *>(out_insts.data()), isz);
    ifs.close();
    fprintf(stderr, "[XDNA] Loaded %zu instructions from %s\n", out_insts.size(), path);
    return true;
}

struct ggml_xdna_persistent_engine * ggml_xdna_persistent_engine_create(int m, int k, int n, bool two_layer) {
    auto * engine = new ggml_xdna_persistent_engine();
    engine->m = m;
    engine->k = k;
    engine->n = n;
    engine->ctx = nullptr;
    engine->initialized = false;
    engine->two_layer = two_layer;
    engine->inst_bo = nullptr;
    engine->weight_bo[0] = engine->weight_bo[1] = nullptr;
    engine->input_bo[0] = engine->input_bo[1] = nullptr;
    engine->output_bo[0] = engine->output_bo[1] = nullptr;
    engine->current_buffer = 0;
    engine->pending_run = nullptr;
    engine->pending_buffer = -1;

    fprintf(stderr, "[XDNA] persistent engine create m=%d k=%d n=%d two_layer=%d\n", m, k, n, two_layer);
    return engine;
}

void ggml_xdna_persistent_engine_free(struct ggml_xdna_persistent_engine * engine) {
    if (!engine) return;

    ggml_xdna_synchronize_all();
    if (engine->pending_run) {
        ggml_xdna_wait_run(engine->pending_run, "persistent-engine-free");
        delete engine->pending_run;
        engine->pending_run = nullptr;
    }

    if (engine->weight_bo[0]) delete engine->weight_bo[0];
    if (engine->weight_bo[1]) delete engine->weight_bo[1];
    if (engine->input_bo[0]) delete engine->input_bo[0];
    if (engine->input_bo[1]) delete engine->input_bo[1];
    if (engine->output_bo[0]) delete engine->output_bo[0];
    if (engine->output_bo[1]) delete engine->output_bo[1];
    if (engine->inst_bo) delete engine->inst_bo;

    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    delete xrt_ctx;
    engine->ctx = nullptr;
    engine->initialized = false;
    delete engine;
}

bool ggml_xdna_persistent_engine_init_xrt(
    struct ggml_xdna_persistent_engine * engine,
    const char * xclbin_path,
    const char * inst_path
) {
    if (!engine || engine->initialized) {
        return false;
    }

    auto * xrt_ctx = new struct xrt_context();
    xrt_ctx->xclbin_path = xclbin_path;
    xrt_ctx->inst_path = inst_path;

    try {
        xrt_ctx->device = xrt::device(0);
        xrt_ctx->xclbin = xrt::xclbin(std::string(xclbin_path));
        xrt_ctx->device.register_xclbin(xrt_ctx->xclbin);
        xrt_ctx->uuid = xrt_ctx->xclbin.get_uuid();
        xrt_ctx->hw_ctx = xrt::hw_context(xrt_ctx->device, xrt_ctx->uuid);
        xrt_ctx->kernel = xrt::kernel(xrt_ctx->hw_ctx, "MLIR_AIE");

        uint32_t gid1 = xrt_ctx->kernel.group_id(1);
        uint32_t gid3 = xrt_ctx->kernel.group_id(3);
        fprintf(stderr, "[XDNA] XRT init: group_id(1)=%u group_id(3)=%u\n", gid1, gid3);

        std::vector<uint32_t> insts;
        if (!load_instructions(inst_path, insts)) {
            delete xrt_ctx;
            return false;
        }
        xrt_ctx->inst_bytes = insts.size() * sizeof(uint32_t);

        engine->inst_bo = new xrt::bo(xrt_ctx->device, xrt_ctx->inst_bytes, xrt::bo::flags::cacheable, gid1);
        std::memcpy(engine->inst_bo->map(), insts.data(), xrt_ctx->inst_bytes);
        engine->inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        engine->ctx = xrt_ctx;
        engine->initialized = true;

        fprintf(stderr, "[XDNA] Persistent engine initialized: xclbin=%s insts=%s\n", xclbin_path, inst_path);
        return true;
    } catch (const std::exception & e) {
        fprintf(stderr, "[XDNA] XRT init failed: %s\n", e.what());
        ggml_xdna_session_mark_fatal_if_xrt_error(e.what());
        if (engine->inst_bo) {
            delete engine->inst_bo;
            engine->inst_bo = nullptr;
        }
        delete xrt_ctx;
        return false;
    }
}

static bool ensure_buffers(struct ggml_xdna_persistent_engine * engine) {
    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    if (!xrt_ctx) return false;

    int buf_idx = engine->current_buffer;
    size_t input_bytes = engine->m * engine->k * sizeof(int8_t);
    size_t weight_bytes = engine->k * engine->n * sizeof(int8_t);
    size_t output_bytes = engine->m * engine->n * sizeof(int32_t);
    uint32_t gid3 = xrt_ctx->kernel.group_id(3);

    if (!engine->input_bo[buf_idx]) {
        engine->input_bo[buf_idx] = new xrt::bo(xrt_ctx->device, input_bytes, xrt::bo::flags::host_only, gid3);
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr, "[XDNA-P38C-BO] ALLOC input_bo[%d]=%p size=%zu engine=%p\n",
                    buf_idx, (void*)engine->input_bo[buf_idx], input_bytes, (void*)engine);
        }
        engine->output_bo[buf_idx] = new xrt::bo(xrt_ctx->device, output_bytes, xrt::bo::flags::host_only, gid3);
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr, "[XDNA-P38C-BO] ALLOC output_bo[%d]=%p size=%zu engine=%p\n",
                    buf_idx, (void*)engine->output_bo[buf_idx], output_bytes, (void*)engine);
        }
    }
    if (!engine->weight_bo[0]) {
        engine->weight_bo[0] = new xrt::bo(xrt_ctx->device, weight_bytes, xrt::bo::flags::host_only, gid3);
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr, "[XDNA-P38C-BO] ALLOC weight_bo[0]=%p size=%zu engine=%p\n",
                    (void*)engine->weight_bo[0], weight_bytes, (void*)engine);
        }
    }
    if (!engine->weight_bo[1]) {
        engine->weight_bo[1] = new xrt::bo(xrt_ctx->device, weight_bytes, xrt::bo::flags::host_only, gid3);
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr, "[XDNA-P38C-BO] ALLOC weight_bo[1]=%p size=%zu engine=%p\n",
                    (void*)engine->weight_bo[1], weight_bytes, (void*)engine);
        }
    }
    return true;
}

bool ggml_xdna_persistent_engine_execute(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    const int8_t * weight2,
    int32_t * output
) {
    if (!engine || !engine->initialized) {
        return false;
    }

    g_xdna_persistent_execute_count++;
    g_xdna_xrt_submit_count++;

    if (!ensure_buffers(engine)) return false;

    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    int buf_idx = engine->current_buffer;
    fprintf(stderr, "[XDNA-XRT] SUBMIT M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    size_t input_bytes = engine->m * engine->k * sizeof(int8_t);
    size_t weight_bytes = engine->k * engine->n * sizeof(int8_t);
    size_t output_bytes = engine->m * engine->n * sizeof(int32_t);

    std::memcpy(engine->input_bo[buf_idx]->map(), input, input_bytes);
    std::memcpy(engine->weight_bo[buf_idx]->map(), weight1, weight_bytes);
    std::memset(engine->output_bo[buf_idx]->map(), 0, output_bytes);

    engine->input_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->weight_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[XDNA-XRT] sync done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    xrt::run run(xrt_ctx->kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, *engine->inst_bo);
    run.set_arg(2, static_cast<uint32_t>(xrt_ctx->inst_bytes));
    run.set_arg(3, *engine->input_bo[buf_idx]);
    run.set_arg(4, *engine->weight_bo[buf_idx]);
    run.set_arg(5, *engine->output_bo[buf_idx]);

    run.start();
    fprintf(stderr, "[XDNA-XRT] start done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);
    if (!ggml_xdna_wait_run(&run, "persistent-execute")) return false;
    fprintf(stderr, "[XDNA-XRT] wait done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(output, engine->output_bo[buf_idx]->map(), output_bytes);
    fprintf(stderr, "[XDNA-XRT] output copy done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    engine->current_buffer = 1 - buf_idx;

    g_xdna_xrt_complete_count++;
    fprintf(stderr, "[XDNA-XRT] COMPLETE M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);
    return true;
}

// N-parallel execute: runs kernel once with full problem dimensions
// input:  [m, K]  (row-major)
// weight: [K, n]  (row-major)
bool ggml_xdna_persistent_engine_execute_nparallel(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight,
    int32_t * output
) {
    if (!engine || !engine->initialized) {
        return false;
    }

    g_xdna_persistent_execute_count++;
    g_xdna_xrt_submit_count++;

    if (!ensure_buffers(engine)) return false;

    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    int buf_idx = engine->current_buffer;
    fprintf(stderr, "[XDNA-XRT] SUBMIT_NPARALLEL M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    size_t input_bytes = engine->m * engine->k * sizeof(int8_t);
    size_t weight_bytes = engine->k * engine->n * sizeof(int8_t);
    size_t output_bytes = engine->m * engine->n * sizeof(int32_t);

    std::memcpy(engine->input_bo[buf_idx]->map(), input, input_bytes);
    std::memcpy(engine->weight_bo[buf_idx]->map(), weight, weight_bytes);
    std::memset(engine->output_bo[buf_idx]->map(), 0, output_bytes);

    engine->input_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->weight_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "[XDNA-XRT] sync done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    xrt::run run(xrt_ctx->kernel);
    run.set_arg(0, (uint64_t)3);
    run.set_arg(1, *engine->inst_bo);
    run.set_arg(2, static_cast<uint32_t>(xrt_ctx->inst_bytes));
    run.set_arg(3, *engine->input_bo[buf_idx]);
    run.set_arg(4, *engine->weight_bo[buf_idx]);
    run.set_arg(5, *engine->output_bo[buf_idx]);

    run.start();
    fprintf(stderr, "[XDNA-XRT] start done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);
    if (!ggml_xdna_wait_run(&run, "persistent-execute")) return false;
    fprintf(stderr, "[XDNA-XRT] wait done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(output, engine->output_bo[buf_idx]->map(), output_bytes);
    fprintf(stderr, "[XDNA-XRT] output copy done M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);

    engine->current_buffer = 1 - buf_idx;

    g_xdna_xrt_complete_count++;
    fprintf(stderr, "[XDNA-XRT] COMPLETE_NPARALLEL M=%d K=%d N=%d buf=%d\n", engine->m, engine->k, engine->n, buf_idx);
    return true;
}

bool ggml_xdna_persistent_engine_submit(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    int32_t * output,
    xrt::run ** out_run
) {
    if (!engine || !engine->initialized || !out_run) {
        return false;
    }

    g_xdna_persistent_execute_count++;
    g_xdna_xrt_submit_count++;
    if (!ensure_buffers(engine)) return false;

    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    int buf_idx = engine->current_buffer;

    size_t input_bytes = engine->m * engine->k * sizeof(int8_t);
    size_t weight_bytes = engine->k * engine->n * sizeof(int8_t);
    size_t output_bytes = engine->m * engine->n * sizeof(int32_t);

    fprintf(stderr, "[XDNA-XRT] SUBMIT_ASYNC m=%d k=%d n=%d buf=%d in_b=%zu w_b=%zu out_b=%zu\n",
            engine->m, engine->k, engine->n, buf_idx, input_bytes, weight_bytes, output_bytes);

    if (engine->pending_run) {
        if (!ggml_xdna_persistent_engine_wait(engine, engine->pending_run,
            (int32_t*)engine->output_bo[engine->pending_buffer]->map())) {
            return false;
        }
    }

    std::memcpy(engine->input_bo[buf_idx]->map(), input, input_bytes);
    std::memcpy(engine->weight_bo[buf_idx]->map(), weight1, weight_bytes);
    std::memset(engine->output_bo[buf_idx]->map(), 0, output_bytes);

    engine->input_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->weight_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run * run = nullptr;
    bool started = false;
    try {
        run = new xrt::run(xrt_ctx->kernel);
        run->set_arg(0, (uint64_t)3);
        run->set_arg(1, *engine->inst_bo);
        run->set_arg(2, static_cast<uint32_t>(xrt_ctx->inst_bytes));
        run->set_arg(3, *engine->input_bo[buf_idx]);
        run->set_arg(4, *engine->weight_bo[buf_idx]);
        run->set_arg(5, *engine->output_bo[buf_idx]);

        run->start();
        started = true;
    } catch (const std::exception & ex) {
        if (started) {
            ggml_xdna_cleanup_run(run);
        } else {
            delete run;
        }
        ggml_xdna_session_mark_fatal_if_xrt_error(ex.what());
        return false;
    } catch (...) {
        if (started) {
            ggml_xdna_cleanup_run(run);
        } else {
            delete run;
        }
        ggml_xdna_session_mark_fatal("unknown persistent submit exception");
        return false;
    }

    engine->pending_run = run;
    engine->pending_buffer = buf_idx;
    engine->current_buffer = 1 - buf_idx;

    *out_run = run;
    return true;
}

bool ggml_xdna_persistent_engine_wait(
    struct ggml_xdna_persistent_engine * engine,
    xrt::run * run,
    int32_t * output
) {
    if (!engine || !engine->initialized || !run || !output || run != engine->pending_run) {
        return false;
    }

    auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
    int buf_idx = engine->pending_buffer;
    if (!xrt_ctx || buf_idx < 0 || buf_idx > 1 || !engine->output_bo[buf_idx]) {
        return false;
    }

    int wait_state = 0;
    uint32_t return_code = 0;
    if (!ggml_xdna_wait_run(run, "persistent-engine", &wait_state, &return_code)) {
        delete run;
        engine->pending_run = nullptr;
        engine->pending_buffer = -1;
        return false;
    }
    fprintf(stderr, "[XDNA-XRT] WAIT_ASYNC m=%d k=%d n=%d buf=%d state=%d return_code=%u\n",
            engine->m, engine->k, engine->n, buf_idx, wait_state, return_code);

    try {
        size_t output_bytes = engine->m * engine->n * sizeof(int32_t);
        engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(output, engine->output_bo[buf_idx]->map(), output_bytes);
    } catch (...) {
        delete run;
        engine->pending_run = nullptr;
        engine->pending_buffer = -1;
        ggml_xdna_session_mark_fatal("persistent-engine output synchronization exception");
        return false;
    }

    delete run;
    engine->pending_run = nullptr;
    engine->pending_buffer = -1;

    g_xdna_xrt_complete_count++;
    return true;
}

bool ggml_xdna_persistent_engine_execute_chunked(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    int K_total,
    const int8_t * weight,
    int32_t * output
) {
    if (!engine || !engine->initialized) {
        return false;
    }

    if (K_total % engine->k != 0) {
        fprintf(stderr, "[XDNA] ERROR: K_total (%d) must be divisible by engine->k (%d)\n", K_total, engine->k);
        return false;
    }

    int num_chunks = K_total / engine->k;
    // fprintf(stderr, "[XDNA] Chunked execute: M=%d K_total=%d N=%d chunks=%d chunk_k=%d\n",
    //         engine->m, K_total, engine->n, num_chunks, engine->k);

    size_t output_bytes = engine->m * engine->n * sizeof(int32_t);
    std::memset(output, 0, output_bytes);

    g_xdna_timings.ops.fetch_add(1);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        g_xdna_persistent_execute_count++;
        g_xdna_xrt_submit_count++;
        g_xdna_timings.chunks.fetch_add(1);

        uint64_t t0 = xdna_now_ns();
        if (!ensure_buffers(engine)) return false;
        uint64_t t1 = xdna_now_ns();
        g_xdna_timings.ensure_ns.fetch_add(t1 - t0);

        auto * xrt_ctx = static_cast<struct xrt_context *>(engine->ctx);
        int buf_idx = engine->current_buffer;

        size_t weight_bytes = engine->k * engine->n * sizeof(int8_t);

        const int8_t * weight_chunk = weight + chunk * engine->k * engine->n;

        // Phase 38C-2: BO handle identity per chunk
        if (std::getenv("XDNA_P38C_BO_TRACE")) {
            fprintf(stderr,
                    "[XDNA-P38C-BO] chunk=%d/%d buf=%d "
                    "input_bo=%p weight_bo=%p output_bo=%p "
                    "in_b=%zu w_b=%zu out_b=%zu\n",
                    chunk, num_chunks, buf_idx,
                    (void*)engine->input_bo[buf_idx],
                    (void*)engine->weight_bo[buf_idx],
                    (void*)engine->output_bo[buf_idx],
                    engine->m * engine->k * sizeof(int8_t),
                    weight_bytes,
                    engine->m * engine->n * sizeof(int32_t));
        }

        // input is [M, K_total] row-major. Copy K_tile columns for each M row.
        int8_t* input_bo_map = static_cast<int8_t*>(engine->input_bo[buf_idx]->map());
        for (int m = 0; m < engine->m; m++) {
            const int8_t* src = input + m * K_total + chunk * engine->k;
            std::memcpy(input_bo_map + m * engine->k, src, engine->k);
        }
        std::memcpy(engine->weight_bo[buf_idx]->map(), weight_chunk, weight_bytes);
        uint64_t t2 = xdna_now_ns();
        g_xdna_timings.copyin_ns.fetch_add(t2 - t1);

        std::memset(engine->output_bo[buf_idx]->map(), 0, output_bytes);
        uint64_t t3 = xdna_now_ns();
        g_xdna_timings.memset_ns.fetch_add(t3 - t2);

        engine->input_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        engine->weight_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        uint64_t t4 = xdna_now_ns();
        g_xdna_timings.syncin_ns.fetch_add(t4 - t3);

        xrt::run run(xrt_ctx->kernel);
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, *engine->inst_bo);
        run.set_arg(2, static_cast<uint32_t>(xrt_ctx->inst_bytes));
        run.set_arg(3, *engine->input_bo[buf_idx]);
        run.set_arg(4, *engine->weight_bo[buf_idx]);
        run.set_arg(5, *engine->output_bo[buf_idx]);
        uint64_t t5 = xdna_now_ns();
        g_xdna_timings.run_new_ns.fetch_add(t5 - t4);

        run.start();
        if (!ggml_xdna_wait_run(&run, "persistent-execute")) return false;
        uint64_t t6 = xdna_now_ns();
        g_xdna_timings.wait_ns.fetch_add(t6 - t5);

        if (std::getenv("XDNA_P38C_CHUNK_TIMING")) {
            double wait_ms = (t6 - t5) / 1e6;
            fprintf(stderr,
                    "[XDNA-P38C-CHUNK] chunk=%d/%d buf=%d "
                    "wait=%.3fms accum_wait=%.3fms\n",
                    chunk, num_chunks, buf_idx, wait_ms,
                    g_xdna_timings.wait_ns.load() / 1e6);
        }

        engine->output_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        int32_t * chunk_output = static_cast<int32_t *>(engine->output_bo[buf_idx]->map());
        uint64_t t7 = xdna_now_ns();
        g_xdna_timings.syncout_ns.fetch_add(t7 - t6);

        for (int i = 0; i < engine->m * engine->n; i++) {
            output[i] += chunk_output[i];
        }
        uint64_t t8 = xdna_now_ns();
        g_xdna_timings.accum_ns.fetch_add(t8 - t7);

        engine->current_buffer = 1 - buf_idx;

        g_xdna_xrt_complete_count++;
    }

    return true;
}

// C wrappers for async functions
#ifdef __cplusplus
extern "C" {
#endif

bool ggml_xdna_persistent_engine_init_xrt_c(
    struct ggml_xdna_persistent_engine * engine,
    const char * xclbin_path,
    const char * inst_path
) {
    return ggml_xdna_persistent_engine_init_xrt(engine, xclbin_path, inst_path);
}

bool ggml_xdna_persistent_engine_submit_c(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight1,
    int32_t * output,
    void ** out_run
) {
    xrt::run* run = nullptr;
    bool result = ggml_xdna_persistent_engine_submit(engine, input, weight1, output, &run);
    *out_run = run;
    return result;
}

bool ggml_xdna_persistent_engine_wait_c(
    struct ggml_xdna_persistent_engine * engine,
    void * run,
    int32_t * output
) {
    return ggml_xdna_persistent_engine_wait(engine, static_cast<xrt::run*>(run), output);
}

bool ggml_xdna_persistent_engine_execute_nparallel_c(
    struct ggml_xdna_persistent_engine * engine,
    const int8_t * input,
    const int8_t * weight,
    int32_t * output
) {
    return ggml_xdna_persistent_engine_execute_nparallel(engine, input, weight, output);
}

#ifdef __cplusplus
}
#endif