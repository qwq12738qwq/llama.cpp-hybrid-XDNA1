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

    fprintf(stderr, "[NPU] Opening device 0\n");
    xrt::device dev(0);
    
    fprintf(stderr, "[NPU] Loading xclbin\n");
    xrt::xclbin xclbin_obj{std::string(xclbin_path)};
    dev.register_xclbin(xclbin_obj);
    
    fprintf(stderr, "[NPU] Creating hw_context\n");
    xrt::hw_context hw_ctx(dev, xclbin_obj.get_uuid());
    
    fprintf(stderr, "[NPU] Opening kernel MLIR_AIE\n");
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    
    fprintf(stderr, "[NPU] group_id(1)=%u group_id(3)=%u group_id(4)=%u group_id(5)=%u\n",
            kernel.group_id(1), kernel.group_id(3), kernel.group_id(4), kernel.group_id(5));
    
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
    
    fprintf(stderr, "[NPU] Creating inst_bo with group_id(1) using hw_ctx\n");
    xrt::bo inst_bo(hw_ctx, isz, xrt::bo::flags::cacheable, kernel.group_id(1));
    fprintf(stderr, "[NPU] inst_bo actual memory_group: %u\n", (unsigned)inst_bo.get_memory_group());
    std::memcpy(inst_bo.map(), insts.data(), isz);
    inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    fprintf(stderr, "[NPU] Creating A, B, C BOs using hw_ctx with group_id(3/4/5)=65536\n");
    xrt::bo a_bo(hw_ctx, M*K*sizeof(int8_t), xrt::bo::flags::host_only, kernel.group_id(3));
    xrt::bo b_bo(hw_ctx, K*N*sizeof(int8_t), xrt::bo::flags::host_only, kernel.group_id(4));
    xrt::bo c_bo(hw_ctx, M*N*sizeof(int32_t), xrt::bo::flags::host_only, kernel.group_id(5));
    fprintf(stderr, "[NPU] a_bo actual memory_group: %u\n", (unsigned)a_bo.get_memory_group());
    fprintf(stderr, "[NPU] b_bo actual memory_group: %u\n", (unsigned)b_bo.get_memory_group());
    fprintf(stderr, "[NPU] c_bo actual memory_group: %u\n", (unsigned)c_bo.get_memory_group());
    
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
    
    std::memcpy(a_bo.map(), a_data.data(), M*K*sizeof(int8_t));
    std::memcpy(b_bo.map(), b_data.data(), K*N*sizeof(int8_t));
    std::memset(c_bo.map(), 0, M*N*sizeof(int32_t));
    a_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    fprintf(stderr, "[NPU] Running kernel\n");
    xrt::run run(kernel);
    run.set_arg(3, inst_bo);
    run.set_arg(4, (uint32_t)(insts.size()*4));
    run.set_arg(5, a_bo);
    run.set_arg(6, b_bo);
    run.set_arg(7, c_bo);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1-t0).count();
    fprintf(stderr, "[NPU] Kernel completed in %.2f us\n", us);
    
    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int32_t* c_out = (int32_t*)c_bo.map();
    
    int mismatches = 0;
    int32_t max_err = 0;
    for (uint32_t i = 0; i < M*N; ++i) {
        int32_t err = std::abs(c_out[i] - c_ref[i]);
        if (err > max_err) max_err = err;
        if (c_out[i] != c_ref[i]) mismatches++;
    }
    
    fprintf(stderr, "[NPU] Mismatches: %d/%d, max_abs_error: %d\n", mismatches, M*N, max_err);
    
    return 0;
}