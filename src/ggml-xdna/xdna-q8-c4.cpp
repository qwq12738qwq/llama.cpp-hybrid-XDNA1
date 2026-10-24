#include "xdna-q8-c4.h"
#include "xdna-buffer.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>

static double q8c4_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool q8c4_load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "[XDNA-Q8-C4] cannot open %s\n", path);
        return false;
    }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

struct ggml_xdna_q8_c4_engine * ggml_xdna_q8_c4_create(
    int M, int K, int N, const char * xclbin_path, const char * inst_path) {
    
    auto * e = new ggml_xdna_q8_c4_engine();
    e->M = M; e->K = K; e->N = N;
    e->nb = K / 32;
    e->a_bytes = (size_t)M * 32;
    e->b_bytes = (size_t)32 * N;
    e->c_bytes = (size_t)M * N * sizeof(int32_t);
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
            ggml_xdna_q8_c4_free(e);
            return nullptr;
        }
        e->xclbin = new xrt::xclbin(std::string(xclbin_path));
        e->device->register_xclbin(*e->xclbin);
        e->hw_ctx = new xrt::hw_context(*e->device, e->xclbin->get_uuid());
        e->kernel = new xrt::kernel(*e->hw_ctx, "MLIR_AIE");

        std::vector<uint32_t> insts;
        if (!q8c4_load_insts(inst_path, insts)) {
            ggml_xdna_q8_c4_free(e);
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

        fprintf(stderr, "[XDNA-Q8-C4] engine ready M=%d K=%d N=%d nb=%d a=%zu b=%zu c=%zu\n",
                M, K, N, e->nb, e->a_bytes, e->b_bytes, e->c_bytes);
        return e;
    } catch (const std::exception & ex) {
        fprintf(stderr, "[XDNA-Q8-C4] init failed: %s\n", ex.what());
        ggml_xdna_session_mark_fatal_if_xrt_error(ex.what());
        ggml_xdna_q8_c4_free(e);
        return nullptr;
    }
}

bool ggml_xdna_q8_c4_submit(
    struct ggml_xdna_q8_c4_engine * e,
    const int8_t * a_block,
    const int8_t * b_block,
    xrt::run ** out_run) {
    
    if (!e || !e->kernel || !a_block || !b_block || !out_run) return false;
    *out_run = nullptr;

    int buf_idx = e->current_buffer;
    double t0 = q8c4_ms();
    std::memcpy(e->a_bo[buf_idx]->map(), a_block, e->a_bytes);
    std::memcpy(e->b_bo[buf_idx]->map(), b_block, e->b_bytes);
    double t1 = q8c4_ms();
    e->a_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    e->b_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::memset(e->c_bo[buf_idx]->map(), 0, e->c_bytes);
    e->c_bo[buf_idx]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    double t2 = q8c4_ms();

    xrt::run * run = nullptr;
    double t3 = q8c4_ms();
    double t4 = t3;
    double t5 = t3;
    bool started = false;
    try {
        run = new xrt::run(*e->kernel);
        run->set_arg(0, (uint64_t)3);
        run->set_arg(1, *e->inst_bo);
        run->set_arg(2, static_cast<uint32_t>(e->inst_bytes));
        run->set_arg(3, *e->a_bo[buf_idx]);
        run->set_arg(4, *e->b_bo[buf_idx]);
        run->set_arg(5, *e->c_bo[buf_idx]);
        t4 = q8c4_ms();
        run->start();
        started = true;
        t5 = q8c4_ms();
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
        ggml_xdna_session_mark_fatal("unknown q8-c4 submit exception");
        return false;
    }

    *out_run = run;
    e->current_buffer = 1 - buf_idx;
    return true;
}

bool ggml_xdna_q8_c4_wait(
    struct ggml_xdna_q8_c4_engine * e,
    xrt::run * run,
    int32_t * c_out,
    int buf_idx) {
    
    if (!e || !run || !c_out) return false;
    double t0 = q8c4_ms();
    double t_wait_start = q8c4_ms();
    if (!ggml_xdna_wait_run(run, "q8-c4")) {
        delete run;
        return false;
    }
    double t1 = q8c4_ms();
    double t_wait = t1 - t_wait_start;
    try {
        e->c_bo[buf_idx]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double t2 = q8c4_ms();
        std::memcpy(c_out, e->c_bo[buf_idx]->map(), e->c_bytes);
        double t3 = q8c4_ms();

        fprintf(stderr, "[XDNA-Q8-C4] wait=%.2fms sync_from=%.2fms copyback=%.2fms total=%.2fms\n",
                t_wait, t2 - t1, t3 - t2, t3 - t0);
    } catch (...) {
        delete run;
        ggml_xdna_session_mark_fatal("q8-c4 output synchronization exception");
        return false;
    }
    delete run;
    return true;
}

void ggml_xdna_q8_c4_free(struct ggml_xdna_q8_c4_engine * e) {
    if (!e) return;
    ggml_xdna_synchronize_all();
    delete e->inst_bo;
    delete e->a_bo[0];
    delete e->a_bo[1];
    delete e->b_bo[0];
    delete e->b_bo[1];
    delete e->c_bo[0];
    delete e->c_bo[1];
    delete e->kernel;
    delete e->hw_ctx;
    delete e->xclbin;
    // shared device is process-wide; do not delete
    delete e;
}