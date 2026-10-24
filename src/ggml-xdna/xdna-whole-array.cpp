#include "xdna-whole-array.h"
#include "xdna-buffer.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>

static bool wa_load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "[XDNA-WA] cannot open %s\n", path);
        return false;
    }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t) ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

static double wa_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void wa_log_timing(const char * label, double bo_copy_us, double bo_sync_us,
                           double run_new_us, double wait_us, double copyback_us) {
    fprintf(stderr,
            "[XDNA-WA-P38C] %s bo_copy=%.2fms bo_sync=%.2fms run_new=%.2fms "
            "wait=%.2fms copyback=%.2fms total=%.2fms a=%zu b=%zu c=%zu\n",
            label,
            bo_copy_us / 1000.0, bo_sync_us / 1000.0, run_new_us / 1000.0,
            wait_us / 1000.0, copyback_us / 1000.0,
            (bo_copy_us + bo_sync_us + run_new_us + wait_us + copyback_us) / 1000.0,
            (size_t)0, (size_t)0, (size_t)0);
}

ggml_xdna_wa_engine * ggml_xdna_wa_create(int M, int K, int N,
                                          const char * xclbin_path,
                                          const char * inst_path) {
    auto * e = new ggml_xdna_wa_engine();
    e->M = M; e->K = K; e->N = N;
    e->a_bytes = (size_t) M * K;
    e->b_bytes = (size_t) K * N;
    e->c_bytes = (size_t) M * N * sizeof(int32_t);
    e->device = nullptr; e->xclbin = nullptr; e->hw_ctx = nullptr; e->kernel = nullptr;
    e->inst_bo = nullptr; e->a_bo = nullptr; e->b_bo = nullptr; e->c_bo = nullptr;

    try {
        e->device = ggml_xdna_shared_xrt_device();
        if (!e->device) {
            fprintf(stderr, "[XDNA] shared XRT device unavailable\n");
            ggml_xdna_wa_free(e);
            return nullptr;
        }
        e->xclbin = new xrt::xclbin(std::string(xclbin_path));
        e->device->register_xclbin(*e->xclbin);
        e->hw_ctx = new xrt::hw_context(*e->device, e->xclbin->get_uuid());
        e->kernel = new xrt::kernel(*e->hw_ctx, "MLIR_AIE");

        std::vector<uint32_t> insts;
        if (!wa_load_insts(inst_path, insts)) {
            ggml_xdna_wa_free(e);
            return nullptr;
        }
        e->inst_bytes = insts.size() * sizeof(uint32_t);

        uint32_t gid1 = e->kernel->group_id(1);
        uint32_t gid3 = e->kernel->group_id(3);

        e->inst_bo = new xrt::bo(*e->device, e->inst_bytes, xrt::bo::flags::cacheable, gid1);
        std::memcpy(e->inst_bo->map(), insts.data(), e->inst_bytes);
        e->inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        e->a_bo = new xrt::bo(*e->device, e->a_bytes, xrt::bo::flags::host_only, gid3);
        e->b_bo = new xrt::bo(*e->device, e->b_bytes, xrt::bo::flags::host_only, gid3);
        e->c_bo = new xrt::bo(*e->device, e->c_bytes, xrt::bo::flags::host_only, gid3);

        fprintf(stderr, "[XDNA-WA] engine ready M=%d K=%d N=%d a=%zu b=%zu c=%zu\n",
                M, K, N, e->a_bytes, e->b_bytes, e->c_bytes);
        return e;
    } catch (const std::exception & ex) {
        fprintf(stderr, "[XDNA-WA] init failed: %s\n", ex.what());
        ggml_xdna_session_mark_fatal_if_xrt_error(ex.what());
        ggml_xdna_wa_free(e);
        return nullptr;
    }
}

bool ggml_xdna_wa_submit(ggml_xdna_wa_engine * e, const int8_t * a, const int8_t * b,
                         xrt::run ** out_run) {
    if (!e || !e->kernel || !a || !b || !out_run) return false;
    *out_run = nullptr;
    double t0 = wa_ms();
    std::memcpy(e->a_bo->map(), a, e->a_bytes);
    std::memcpy(e->b_bo->map(), b, e->b_bytes);
    double t1 = wa_ms();
    e->a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    e->b_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    double t2 = wa_ms();

    xrt::run * run = nullptr;
    double t3 = wa_ms();
    double t4 = t3;
    double t5 = t3;
    bool started = false;
    try {
        run = new xrt::run(*e->kernel);
        run->set_arg(0, (uint64_t) 3);
        run->set_arg(1, *e->inst_bo);
        run->set_arg(2, static_cast<uint32_t>(e->inst_bytes));
        run->set_arg(3, *e->a_bo);
        run->set_arg(4, *e->b_bo);
        run->set_arg(5, *e->c_bo);
        t4 = wa_ms();
        run->start();
        started = true;
        t5 = wa_ms();
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
        ggml_xdna_session_mark_fatal("unknown whole-array submit exception");
        return false;
    }

    if (std::getenv("XDNA_WA_P38C")) {
        fprintf(stderr,
                "[XDNA-WA-P38C] SUBMIT a=%zu b=%zu c=%zu "
                "bo_copy=%.2fms bo_sync=%.2fms run_ctor=%.2fms setarg=%.2fms start=%.2fms\n",
                (size_t)e->a_bytes, (size_t)e->b_bytes, (size_t)e->c_bytes,
                (t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (t5 - t4));
    }

    *out_run = run;
    return true;
}

bool ggml_xdna_wa_wait(ggml_xdna_wa_engine * e, xrt::run * run, int32_t * c_out) {
    if (!e || !run || !c_out) return false;
    double t0 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    double t_wait_start = wa_ms();
    if (!ggml_xdna_wait_run(run, "whole-array")) {
        delete run;
        return false;
    }
    double t1 = wa_ms();
    double t_wait = t1 - t_wait_start;
    try {
        e->c_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        double t2 = wa_ms();
        double t_copyback = t2 - t1;
        std::memcpy(c_out, e->c_bo->map(), e->c_bytes);
        double t3 = wa_ms();

        if (std::getenv("XDNA_WA_P38C")) {
            double total = wa_ms() - t0;
            fprintf(stderr, "[XDNA-WA-P38C] WAIT wait=%.2fms sync_from=%.2fms copyback=%.2fms "
                    "total=%.2fms c=%zu\n",
                    t_wait, t_copyback / 1000.0 * 1000.0,
                    (t3 - t2) / 1000.0, total / 1000.0,
                    (size_t)e->c_bytes);
        } else if (getenv("XDNA_WA_TIMING")) {
            fprintf(stderr, "[XDNA-WA-T] wait=%.2fms copyback=%.2fms\n",
                    t_wait / 1000.0,
                    (t3 - t2) / 1000.0);
        }
    } catch (...) {
        delete run;
        ggml_xdna_session_mark_fatal("whole-array output synchronization exception");
        return false;
    }
    delete run;
    return true;

}

void ggml_xdna_wa_free(ggml_xdna_wa_engine * e) {
    if (!e) return;
    ggml_xdna_synchronize_all();
    delete e->kernel;
    delete e->inst_bo;
    delete e->a_bo;
    delete e->b_bo;
    delete e->c_bo;
    delete e->hw_ctx;
    delete e->xclbin;
    delete e;
}
