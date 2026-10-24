#include "xdna-q8-c4-mmul-v2.h"
#include "xdna-buffer.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>

static double q8c4mmulv2_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t q8c4mmulv2_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void q8c4mmulv2_forget_run_locked(ggml_xdna_q8_c4_mmul_v2_engine * e, xrt::run * run) {
    e->active_runs.erase(std::remove(e->active_runs.begin(), e->active_runs.end(), run), e->active_runs.end());
}

static bool q8c4mmulv2_load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "[XDNA-Q8-C4-MMUL-V2] cannot open %s\n", path);
        return false;
    }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

struct ggml_xdna_q8_c4_mmul_v2_engine * ggml_xdna_q8_c4_mmul_v2_create(
    int M, int K, int N, int n_cols,
    int m_tile, int n_tile, int k_tile, int mmul,
    const char * xclbin_path, const char * inst_path, int s12, int pure_int8,
    int c256) {
 
    auto * e = new ggml_xdna_q8_c4_mmul_v2_engine();
    e->M = M; e->K = K; e->N = N;
    e->n_cols = n_cols;
    e->m = m_tile; e->n = n_tile; e->kt = k_tile; e->mmul = mmul;
    e->nb = e->kt / 32;
    e->s12 = s12;
    e->c256 = c256;
 
    int a_chunk, b_chunk;
    if (pure_int8) {
        a_chunk = e->m * e->kt;
        b_chunk = e->kt * e->n;
    } else if (c256) {
        // Phase 62: one scale per K chunk for the A rows and the B columns
        a_chunk = e->m * e->kt + e->m * 4;
        b_chunk = e->kt * e->n + e->n * 4;
    } else if (s12) {
        // pre-broadcast scales: A per (bb,i) 32 floats, B per (bb,j) 16 floats
        a_chunk = e->m * e->kt + e->nb * (e->m / 4) * 32 * 4;
        b_chunk = e->kt * e->n + e->nb * (e->n / 8) * 16 * 4;
    } else {
        a_chunk = e->m * e->kt + e->m * e->nb * 4;
        b_chunk = e->kt * e->n + e->nb * e->n * 4;
    }

    int n_mt = M / e->m;
    int n_nt_c = (N / n_cols) / e->n;
    int n_kc = K / e->kt;

    e->a_bytes = (size_t)n_mt * n_kc * a_chunk;
    e->b_bytes = (size_t)n_cols * n_nt_c * n_kc * b_chunk;
    e->c_bytes = (size_t)M * N * sizeof(float);

    e->device = nullptr; e->xclbin = nullptr; e->hw_ctx = nullptr; e->kernel = nullptr;
    e->inst_bo = nullptr;
    e->a_bo[0] = nullptr; e->a_bo[1] = nullptr;
    e->b_bo[0] = nullptr; e->b_bo[1] = nullptr;
    e->c_bo[0] = nullptr; e->c_bo[1] = nullptr;
    e->current_buffer = 0;

    try {
        e->device = ggml_xdna_shared_xrt_device();
        if (!e->device) {
            fprintf(stderr, "[XDNA] shared XRT device unavailable\n");
            ggml_xdna_q8_c4_mmul_v2_free(e);
            return nullptr;
        }
        e->xclbin = new xrt::xclbin(std::string(xclbin_path));
        e->device->register_xclbin(*e->xclbin);
        e->hw_ctx = new xrt::hw_context(*e->device, e->xclbin->get_uuid());
        e->kernel = new xrt::kernel(*e->hw_ctx, "MLIR_AIE");

        std::vector<uint32_t> insts;
        if (!q8c4mmulv2_load_insts(inst_path, insts)) {
            ggml_xdna_q8_c4_mmul_v2_free(e);
            return nullptr;
        }
        e->inst_bytes = insts.size() * sizeof(uint32_t);

        uint32_t gid1 = e->kernel->group_id(1);
        uint32_t gid3 = e->kernel->group_id(3);

        e->inst_bo = new xrt::bo(*e->device, e->inst_bytes, xrt::bo::flags::cacheable, gid1);
        std::memcpy(e->inst_bo->map(), insts.data(), e->inst_bytes);
        e->inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        e->a_bo[0] = new xrt::bo(*e->device, e->a_bytes, xrt::bo::flags::host_only, gid3);
        e->b_bo[0] = new xrt::bo(*e->device, e->b_bytes, xrt::bo::flags::host_only, gid3);
        e->c_bo[0] = new xrt::bo(*e->device, e->c_bytes, xrt::bo::flags::host_only, gid3);
        e->a_bo[1] = new xrt::bo(*e->device, e->a_bytes, xrt::bo::flags::host_only, gid3);
        e->b_bo[1] = new xrt::bo(*e->device, e->b_bytes, xrt::bo::flags::host_only, gid3);
        e->c_bo[1] = new xrt::bo(*e->device, e->c_bytes, xrt::bo::flags::host_only, gid3);

        fprintf(stderr, "[XDNA-Q8-C4-MMUL-V2] engine ready M=%d K=%d N=%d cols=%d "
                "a=%zu b=%zu c=%zu\n",
                M, K, N, n_cols, e->a_bytes, e->b_bytes, e->c_bytes);
        return e;
    } catch (const std::exception & ex) {
        fprintf(stderr, "[XDNA-Q8-C4-MMUL-V2] init failed: %s\n", ex.what());
        ggml_xdna_session_mark_fatal_if_xrt_error(ex.what());
        ggml_xdna_q8_c4_mmul_v2_free(e);
        return nullptr;
    }
}

bool ggml_xdna_q8_c4_mmul_v2_submit(
    struct ggml_xdna_q8_c4_mmul_v2_engine * e,
    const void * a_packed,
    const void * b_packed,
    xrt::run ** out_run,
    ggml_xdna_xrt_trace * trace) {

    if (!e || !e->kernel || !a_packed || !b_packed || !out_run) return false;

    std::unique_lock<std::mutex> lifecycle_lock(e->lifecycle_mutex);
    if (e->freeing) return false;

    *out_run = nullptr;
    int buf_idx = e->current_buffer;
    double t0 = q8c4mmulv2_ms();
    if (trace) {
        trace->submit_start_ns = q8c4mmulv2_now_ns();
        trace->buf_idx = buf_idx;
        trace->run_objects = 1;
        trace->submissions = 0;
        trace->starts = 0;
        trace->waits = 0;
        trace->k_block_size = e->kt;
        trace->expected_k_blocks = e->K / e->kt;
        trace->actual_k_blocks = e->K / e->kt;
        trace->q8_scale_blocks = e->K / 32;
        trace->k_chunks = e->K / e->kt;
        trace->m_tiles = e->M / e->m;
        trace->n_tiles = (e->N / e->n_cols) / e->n;
        trace->n_cols = e->n_cols;
        trace->inst_bo = e->inst_bo;
        trace->input_bo = e->a_bo[buf_idx];
        trace->weight_bo = e->b_bo[buf_idx];
        trace->output_bo = e->c_bo[buf_idx];
    }
    std::memcpy(e->a_bo[buf_idx]->map(), a_packed, e->a_bytes);
    double ta = q8c4mmulv2_ms();
    std::memcpy(e->b_bo[buf_idx]->map(), b_packed, e->b_bytes);
    double t1 = q8c4mmulv2_ms();
    e->a_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    e->b_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // Phase 61: the C tile is zeroed by the worker before the first K chunk
    // and the kernel accumulates into it, so the host does not have to clear
    // the whole output buffer. Skipping it saves a C_bytes memset and a
    // C_bytes DMA to the device on every GEMM. XDNA_SKIP_C_MEMSET=1 enables
    // the skip (A/B switch).
    static const bool skip_c_memset = getenv("XDNA_SKIP_C_MEMSET") != nullptr;
    if (!skip_c_memset) {
        std::memset(e->c_bo[buf_idx]->map(), 0, e->c_bytes);
        e->c_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    double t2 = q8c4mmulv2_ms();

    xrt::run * run = nullptr;
    double t3 = q8c4mmulv2_ms();
    double t_ctor = t3;
    double t4 = t3;
    double t5 = t3;
    bool started = false;
    try {
        run = new xrt::run(*e->kernel);
        t_ctor = q8c4mmulv2_ms();
        run->set_arg(0, (uint64_t)3);
        run->set_arg(1, *e->inst_bo);
        run->set_arg(2, static_cast<uint32_t>(e->inst_bytes));
        run->set_arg(3, *e->a_bo[buf_idx]);
        run->set_arg(4, *e->b_bo[buf_idx]);
        run->set_arg(5, *e->c_bo[buf_idx]);
        t4 = q8c4mmulv2_ms();
        run->start();
        started = true;
        t5 = q8c4mmulv2_ms();
        if (trace) {
            trace->submit_end_ns = q8c4mmulv2_now_ns();
            trace->run_start_ns = trace->submit_end_ns;
            trace->a_copy_ms = ta - t0;
            trace->b_copy_ms = t1 - ta;
            trace->bo_sync_ms = t2 - t1;
            trace->run_ctor_ms = t_ctor - t3;
            trace->set_arg_ms = t4 - t_ctor;
            trace->run_start_ms = t5 - t4;
            trace->submit_ms = (trace->submit_end_ns - trace->submit_start_ns) / 1e6;
            trace->submissions = 1;
            trace->starts = 1;
        }

        e->current_buffer = 1 - buf_idx;
        e->active_runs.push_back(run);
    } catch (const std::exception & ex) {
        if (started) {
            ggml_xdna_cleanup_run(run);
        } else {
            delete run;
        }
        ggml_xdna_session_mark_fatal(ex.what());
        return false;
    } catch (...) {
        if (started) {
            ggml_xdna_cleanup_run(run);
        } else {
            delete run;
        }
        ggml_xdna_session_mark_fatal("unknown XRT submit exception");
        return false;
    }

    if (std::getenv("XDNA_Q8_MMUL_V2")) {
        fprintf(stderr, "[XDNA-Q8-C4-MMUL-V2] SUBMIT a=%zu b=%zu c=%zu buf=%d "
                "a_copy=%.2fms b_copy=%.2fms sync=%.2fms "
                "run_ctor=%.2fms setarg=%.2fms start=%.2fms\n",
                (size_t)e->a_bytes, (size_t)e->b_bytes, (size_t)e->c_bytes, buf_idx,
                t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t4);
    }

    *out_run = run;
    return true;
}

bool ggml_xdna_q8_c4_mmul_v2_wait(
    struct ggml_xdna_q8_c4_mmul_v2_engine * e,
    xrt::run * run,
    float * out,
    int buf_idx,
    ggml_xdna_xrt_trace * trace) {

    if (!e || !run || !out) return false;
    std::unique_lock<std::mutex> lifecycle_lock(e->lifecycle_mutex);
    if (e->freeing) return false;
    double t0 = q8c4mmulv2_ms();
    double t_wait_start = q8c4mmulv2_ms();
    if (trace) {
        trace->wait_start_ns = q8c4mmulv2_now_ns();
    }

    int wait_state = 0;
    uint32_t return_code = 0;
    if (!ggml_xdna_wait_run(run, "q8-c4-mmul-v2", &wait_state, &return_code)) {
        q8c4mmulv2_forget_run_locked(e, run);
        delete run;
        return false;
    }
    double t1 = q8c4mmulv2_ms();
    double t_wait = t1 - t_wait_start;
    try {
        if (trace) {
            trace->wait_end_ns = q8c4mmulv2_now_ns();
            trace->queue_wait_ms = trace->wait_start_ns > trace->run_start_ns
                ? (trace->wait_start_ns - trace->run_start_ns) / 1e6
                : 0.0;
        }
        e->c_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double t2 = q8c4mmulv2_ms();
        std::memcpy(out, e->c_bo[buf_idx]->map(), e->c_bytes);
        double t3 = q8c4mmulv2_ms();
        if (trace) {
            trace->sync_end_ns = q8c4mmulv2_now_ns();
            trace->copy_end_ns = trace->sync_end_ns;
            trace->run_wait_ms = t_wait;
            trace->sync_from_ms = t2 - t1;
            trace->copyback_ms = t3 - t2;
            trace->total_ms = t3 - t0;
            trace->wait_state = wait_state;
            trace->return_code = return_code;
            trace->waits = 1;
        }

        if (std::getenv("XDNA_Q8_MMUL_V2")) {
            fprintf(stderr, "[XDNA-Q8-C4-MMUL-V2] WAIT buf=%d wait=%.2fms sync_from=%.2fms copyback=%.2fms total=%.2fms\n",
                    buf_idx, t_wait, t2 - t1, t3 - t2, t3 - t0);
        }
    } catch (...) {
        q8c4mmulv2_forget_run_locked(e, run);
        delete run;
        ggml_xdna_session_mark_fatal("q8-c4-mmul-v2 output synchronization exception");
        return false;
    }
    q8c4mmulv2_forget_run_locked(e, run);
    delete run;
    return true;
}

void ggml_xdna_q8_c4_mmul_v2_free(struct ggml_xdna_q8_c4_mmul_v2_engine * e) {
    if (!e) return;
    ggml_xdna_synchronize_all();
    {
        std::unique_lock<std::mutex> lifecycle_lock(e->lifecycle_mutex);
        if (e->freeing) return;
        e->freeing = true;
        std::vector<xrt::run *> runs = std::move(e->active_runs);
        e->active_runs.clear();
        lifecycle_lock.unlock();
        for (xrt::run * run : runs) {
            ggml_xdna_cleanup_run(run);
        }
    }
    delete e->kernel;
    delete e->inst_bo;
    delete e->a_bo[0];
    delete e->a_bo[1];
    delete e->b_bo[0];
    delete e->b_bo[1];
    delete e->c_bo[0];
    delete e->c_bo[1];
    delete e->hw_ctx;
    delete e->xclbin;
    delete e;
}
