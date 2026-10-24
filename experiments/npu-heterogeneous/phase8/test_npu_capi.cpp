#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include "/workspace/llama.cpp/ggml/src/ggml-xdna/stub/uuid/uuid.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>

extern "C" {
#include <xrt/deprecated/xrt.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <xclbin> <insts> <M> <K> <N>\n", argv[0]);
        return 1;
    }
    const char* xclbin_path = argv[1];
    const char* insts_path = argv[2];
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);
    int N = atoi(argv[5]);

    fprintf(stderr, "[NPU] Opening device 0 (C API)\n");
    xrtDeviceHandle dhdl = xrtDeviceOpen(0);
    if (!dhdl) {
        fprintf(stderr, "Failed to open device\n");
        return 1;
    }
    
    fprintf(stderr, "[NPU] Loading xclbin\n");
    xrtXclbinHandle xclbin_hdl = xrtXclbinLoad(dhdl, xclbin_path);
    if (!xclbin_hdl) {
        fprintf(stderr, "Failed to load xclbin\n");
        return 1;
    }
    
    fprintf(stderr, "[NPU] Creating hw_context\n");
    xrtHwContextHandle hw_ctx = xrtHwContextCreate(dhdl, xclbin_hdl);
    if (!hw_ctx) {
        fprintf(stderr, "Failed to create hw_context\n");
        return 1;
    }
    
    fprintf(stderr, "[NPU] Opening kernel MLIR_AIE\n");
    xrtKernelHandle kernel = xrtKernelOpen(hw_ctx, "MLIR_AIE");
    if (!kernel) {
        fprintf(stderr, "Failed to open kernel\n");
        return 1;
    }
    
    fprintf(stderr, "[NPU] Getting group IDs\n");
    uint32_t gid0 = xrtKernelGroupId(kernel, 0);
    uint32_t gid1 = xrtKernelGroupId(kernel, 1);
    uint32_t gid3 = xrtKernelGroupId(kernel, 3);
    fprintf(stderr, "[NPU] group_id(0)=%u group_id(1)=%u group_id(3)=%u\n", gid0, gid1, gid3);
    
    fprintf(stderr, "[NPU] Loading instructions\n");
    std::ifstream ifs(insts_path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "Failed to open insts: %s\n", insts_path);
        return 1;
    }
    ifs.seekg(0, std::ios::end);
    size_t isz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint32_t> insts(isz / 4);
    ifs.read((char*)insts.data(), isz);
    ifs.close();
    
    fprintf(stderr, "[NPU] Creating inst_bo (size=%zu, group=%u)\n", isz, gid1);
    xrtBufferHandle inst_bo = xrtBOAlloc(dhdl, isz, XRT_BO_FLAGS_CACHEABLE, gid1);
    if (!inst_bo) {
        fprintf(stderr, "Failed to allocate inst_bo\n");
        return 1;
    }
    void* inst_ptr = xrtBOMap(inst_bo);
    std::memcpy(inst_ptr, insts.data(), isz);
    xrtBOSync(inst_bo, XCL_BO_SYNC_BO_TO_DEVICE, isz, 0);
    
    fprintf(stderr, "[NPU] Creating A, B, C BOs\n");
    xrtBufferHandle a_bo = xrtBOAlloc(dhdl, M*K*sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, gid3);
    xrtBufferHandle b_bo = xrtBOAlloc(dhdl, K*N*sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, gid3);
    xrtBufferHandle c_bo = xrtBOAlloc(dhdl, M*N*sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, gid3);
    
    std::mt19937 rng(42);
    std::vector<int8_t> a_data(M*K);
    std::vector<int8_t> b_data(K*N);
    std::vector<int32_t> c_ref(M*N);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (size_t i = 0; i < a_data.size(); ++i) a_data[i] = (int8_t)dist(rng);
    for (size_t i = 0; i < b_data.size(); ++i) b_data[i] = (int8_t)dist(rng);
    
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; ++k) sum += (int32_t)a_data[i*K+k] * (int32_t)b_data[k*N+j];
            c_ref[i*N+j] = sum;
        }
    
    std::memcpy(xrtBOMap(a_bo), a_data.data(), M*K*sizeof(int8_t));
    std::memcpy(xrtBOMap(b_bo), b_data.data(), K*N*sizeof(int8_t));
    std::memset(xrtBOMap(c_bo), 0, M*N*sizeof(int32_t));
    xrtBOSync(a_bo, XCL_BO_SYNC_BO_TO_DEVICE, M*K*sizeof(int8_t), 0);
    xrtBOSync(b_bo, XCL_BO_SYNC_BO_TO_DEVICE, K*N*sizeof(int8_t), 0);
    xrtBOSync(c_bo, XCL_BO_SYNC_BO_TO_DEVICE, M*N*sizeof(int32_t), 0);
    
    fprintf(stderr, "[NPU] Running kernel\n");
    xrtRunHandle run = xrtRunOpen(kernel);
    xrtRunSetArg(run, 3, inst_bo);
    xrtRunSetArg(run, 4, (uint32_t)(insts.size()*4));
    xrtRunSetArg(run, 5, a_bo);
    xrtRunSetArg(run, 6, b_bo);
    xrtRunSetArg(run, 7, c_bo);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = xrtRunStart(run);
    if (rc != 0) {
        fprintf(stderr, "xrtRunStart failed: %d\n", rc);
        return 1;
    }
    rc = xrtRunWait(run);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1-t0).count();
    fprintf(stderr, "[NPU] Kernel completed in %.2f us (wait rc=%d)\n", us, rc);
    
    xrtBOSync(c_bo, XCL_BO_SYNC_BO_FROM_DEVICE, M*N*sizeof(int32_t), 0);
    int32_t* c_out = (int32_t*)xrtBOMap(c_bo);
    
    int mismatches = 0;
    int32_t max_err = 0;
    for (uint32_t i = 0; i < M*N; ++i) {
        int32_t err = std::abs(c_out[i] - c_ref[i]);
        if (err > max_err) max_err = err;
        if (c_out[i] != c_ref[i]) mismatches++;
    }
    
    fprintf(stderr, "[NPU] Mismatches: %d/%d, max_abs_error: %d\n", mismatches, M*N, max_err);
    
    xrtRunClose(run);
    xrtBOFree(inst_bo);
    xrtBOFree(a_bo);
    xrtBOFree(b_bo);
    xrtBOFree(c_bo);
    xrtKernelClose(kernel);
    xrtHwContextFree(hw_ctx);
    xrtXclbinFree(xclbin_hdl);
    xrtDeviceClose(dhdl);
    
    return 0;
}
