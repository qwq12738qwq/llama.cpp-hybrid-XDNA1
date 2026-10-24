/**
 * simple_xdna_test.cpp - Simple test of XDNA backend MUL_MAT
 */

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    printf("[DEBUG] Starting simple XDNA test\n");
    fflush(stdout);
    
    ggml_backend_load_all();
    printf("[DEBUG] Backends loaded\n");
    fflush(stdout);

    ggml_backend_t backend_xdna = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    printf("[DEBUG] Looking for backends by type...\n");
    fflush(stdout);
    
    // Try by device type first (most reliable)
    backend_xdna = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, "");
    printf("Found XDNA backend by ACCEL type: %p\n", (void*)backend_xdna);
    fflush(stdout);
    
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, "");
    printf("Found CPU backend by CPU type: %p\n", (void*)backend_cpu);
    fflush(stdout);
    
    // Fallback to backend name
    if (!backend_xdna) {
        printf("[DEBUG] Trying XDNA backend by name...\n"); fflush(stdout);
        backend_xdna = ggml_backend_init_by_name("XDNA", "");
        printf("Found XDNA backend by name: %p\n", (void*)backend_xdna);
        fflush(stdout);
    }
    if (!backend_cpu) {
        printf("[DEBUG] Trying CPU backend by name...\n"); fflush(stdout);
        backend_cpu = ggml_backend_init_by_name("CPU", "");
        printf("Found CPU backend by name: %p\n", (void*)backend_cpu);
        fflush(stdout);
    }

    if (!backend_xdna || !backend_cpu) {
        printf("Missing backends: xdna=%p cpu=%p\n", (void*)backend_xdna, (void*)backend_cpu);
        return 1;
    }

    // Test M=512, K=1024, N=4096
    int M = 512, K = 1024, N = 4096;
    printf("\n=== Testing M=%d K=%d N=%d ===\n", M, K, N);
    fflush(stdout);

    struct ggml_init_params params = { 0 };
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    printf("[DEBUG] Initializing context...\n"); fflush(stdout);
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("[ERROR] Failed to initialize context\n");
        return 1;
    }
    printf("[DEBUG] Context initialized\n"); fflush(stdout);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    printf("[DEBUG] Context created\n"); fflush(stdout);

    // Create tensors: a = [K, N] (weight), b = [K, M] (input), c = [N, M] (output)
    printf("[DEBUG] Creating tensors...\n"); fflush(stdout);
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    printf("[DEBUG] Tensors created: a=[%d,%d], b=[%d,%d], c=[%d,%d]\n",
           (int)a->ne[0], (int)a->ne[1], (int)b->ne[0], (int)b->ne[1], (int)c->ne[0], (int)c->ne[1]);
    fflush(stdout);

    // Fill with test data on host - use F32 for easier debugging
    std::vector<float> weight_data_f32(K * N);
    std::vector<float> input_data_f32(K * M);
    for (size_t i = 0; i < weight_data_f32.size(); i++) weight_data_f32[i] = (float)(i % 256) / 256.0f;
    for (size_t i = 0; i < input_data_f32.size(); i++) input_data_f32[i] = (float)((i * 7) % 256) / 256.0f;
    printf("[DEBUG] Test data generated\n"); fflush(stdout);

    // Test CPU first - use alloc_ctx_tensors
    printf("[DEBUG] Running CPU reference...\n"); fflush(stdout);
    
    // Allocate buffers for CPU
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx, backend_cpu);
    if (!buf_cpu) {
        printf("ERROR: Failed to allocate CPU buffers!\n");
        return 1;
    }
    printf("[DEBUG] CPU buffers allocated\n"); fflush(stdout);

    // Set input data AFTER buffers are allocated
    ggml_backend_tensor_set(a, weight_data_f32.data(), 0, weight_data_f32.size() * sizeof(float));
    ggml_backend_tensor_set(b, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    printf("[DEBUG] CPU buffers allocated and data set\n"); fflush(stdout);

    ggml_build_forward_expand(gf, c);
    auto t1 = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(backend_cpu, gf);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("CPU: %ld ms\n", cpu_ms);
    fflush(stdout);

    if (!c->buffer) {
        printf("CPU output tensor has NO buffer!\n");
    } else {
        printf("CPU output tensor has buffer\n");
    }
    fflush(stdout);

    std::vector<float> cpu_out(ggml_nbytes(c) / sizeof(float));
    if (c->buffer) {
        ggml_backend_tensor_get(c, cpu_out.data(), 0, cpu_out.size() * sizeof(float));
        printf("CPU reference: first 5: %.2f %.2f %.2f %.2f %.2f\n",
               cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], cpu_out[4]);
        fflush(stdout);
    } else {
        printf("CPU output tensor has no buffer, cannot get data\n");
        fflush(stdout);
    }

    // Test XDNA - need to create new context because data was consumed
    printf("[DEBUG] Running XDNA...\n"); fflush(stdout);
    struct ggml_context* ctx2 = ggml_init(params);
    struct ggml_cgraph* gf2 = ggml_new_graph(ctx2);
    struct ggml_tensor* a2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c2 = ggml_mul_mat(ctx2, a2, b2);
    
    // Allocate buffers on XDNA backend FIRST
    ggml_backend_buffer_t buf_xdna = ggml_backend_alloc_ctx_tensors(ctx2, backend_xdna);
    if (!buf_xdna) {
        printf("ERROR: Failed to allocate XDNA buffers!\n");
        return 1;
    }
    printf("[DEBUG] XDNA buffers allocated\n"); fflush(stdout);
    
    // Set input data AFTER buffers are allocated
    ggml_backend_tensor_set(a2, weight_data_f32.data(), 0, weight_data_f32.size() * sizeof(float));
    ggml_backend_tensor_set(b2, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    printf("[DEBUG] XDNA buffers allocated and data set\n"); fflush(stdout);
    
    t1 = std::chrono::high_resolution_clock::now();
    ggml_build_forward_expand(gf2, c2);
    ggml_backend_graph_compute(backend_xdna, gf2);
    t2 = std::chrono::high_resolution_clock::now();
    auto xdna_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("XDNA: %ld ms\n", xdna_ms);
    fflush(stdout);

    if (!c2->buffer) {
        printf("XDNA output tensor has NO buffer!\n");
    } else {
        printf("XDNA output tensor has buffer\n");
    }
    fflush(stdout);

    std::vector<float> xdna_out(ggml_nbytes(c2) / sizeof(float));
    if (c2->buffer) {
        ggml_backend_tensor_get(c2, xdna_out.data(), 0, xdna_out.size() * sizeof(float));
        printf("XDNA output: first 5: %.2f %.2f %.2f %.2f %.2f\n",
               xdna_out[0], xdna_out[1], xdna_out[2], xdna_out[3], xdna_out[4]);
        fflush(stdout);
    } else {
        printf("XDNA output tensor has no buffer\n");
        fflush(stdout);
    }

    // Re-run CPU for comparison (need fresh context)
    printf("[DEBUG] Re-running CPU for comparison...\n"); fflush(stdout);
    struct ggml_context* ctx3 = ggml_init(params);
    struct ggml_cgraph* gf3 = ggml_new_graph(ctx3);
    struct ggml_tensor* a3 = ggml_new_tensor_2d(ctx3, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b3 = ggml_new_tensor_2d(ctx3, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c3 = ggml_mul_mat(ctx3, a3, b3);
    
    // Allocate CPU buffers
    ggml_backend_buffer_t buf_cpu3 = ggml_backend_alloc_ctx_tensors(ctx3, backend_cpu);
    if (!buf_cpu3) {
        printf("ERROR: Failed to allocate CPU buffers for re-run!\n");
        return 1;
    }
    
    ggml_backend_tensor_set(a3, weight_data_f32.data(), 0, weight_data_f32.size() * sizeof(float));
    ggml_backend_tensor_set(b3, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    
    ggml_build_forward_expand(gf3, c3);
    ggml_backend_graph_compute(backend_cpu, gf3);
    
    std::vector<float> cpu_out3(ggml_nbytes(c3) / sizeof(float));
    ggml_backend_tensor_get(c3, cpu_out3.data(), 0, cpu_out3.size() * sizeof(float));
    printf("CPU reference (re-run): first 5: %.2f %.2f %.2f %.2f %.2f\n",
           cpu_out3[0], cpu_out3[1], cpu_out3[2], cpu_out3[3], cpu_out3[4]);
    fflush(stdout);
    
    printf("Speedup CPU->XDNA: %.2fx\n", (double)cpu_ms / xdna_ms);
    fflush(stdout);

    // Cleanup
    ggml_free(ctx);
    ggml_free(ctx2);
    ggml_backend_free(backend_xdna);
    ggml_backend_free(backend_cpu);

    return 0;
}