/**
 * n_dim_split_test.cpp - Test N-dimension MUL_MAT splitting
 * 
 * Tests splitting MUL_MAT along N/output dimension:
 * Original: A[M,K] x B[N,K] -> C[N,M]
 * Split:    A[M,K] x B_vulkan[N_v,K] -> C_vulkan[N_v,M]
 *           A[M,K] x B_cpu[N_c,K]  -> C_cpu[N_c,M]
 * Combine:  concat_N(C_vulkan, C_cpu) -> C[N,M]
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-profile.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <string>

struct BenchEvent {
    uint64_t timestamp_ns;
    std::string backend;
    std::string op;
    int M, K, N;
    size_t size;
    std::string phase;
};

static std::mutex g_event_mutex;
static std::vector<BenchEvent> g_events;
static std::atomic<bool> g_tracing{false};

void profile_cb(const ggml_profile_event_t * e, void *) {
    if (!g_tracing.load()) return;
    BenchEvent ev;
    ev.timestamp_ns = e->timestamp_ns;
    ev.backend = e->backend_name ? e->backend_name : "?";
    ev.op = e->op_name ? e->op_name : "?";
    ev.M = e->M;
    ev.K = e->K;
    ev.N = e->N;
    ev.size = e->tensor_size;
    ev.phase = (e->type == GGML_PROFILE_EVENT_OP_START) ? "start" : "end";
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_events.push_back(ev);
}

void record_event(const char* backend, const char* op, int M, int K, int N, size_t size, const char* phase) {
    if (!g_tracing.load()) return;
    BenchEvent ev;
    ev.timestamp_ns = ggml_time_ns();
    ev.backend = backend;
    ev.op = op;
    ev.M = M; ev.K = K; ev.N = N;
    ev.size = size;
    ev.phase = phase;
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_events.push_back(ev);
}

void clear_events() {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_events.clear();
}

void save_events(const char* filename) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    FILE* f = fopen(filename, "w");
    fprintf(f, "[\n");
    for (size_t i = 0; i < g_events.size(); i++) {
        auto & e = g_events[i];
        fprintf(f, "  {\"ts\":%llu,\"b\":\"%s\",\"op\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,\"sz\":%zu,\"ph\":\"%s\"}%s\n",
                e.timestamp_ns, e.backend.c_str(), e.op.c_str(), e.M, e.K, e.N, e.size, e.phase.c_str(),
                (i+1<g_events.size()) ? "," : "");
    }
    fprintf(f, "]\n");
    fclose(f);
}

int main() {
    printf("[DEBUG] Starting N-dimension split test\n");
    fflush(stdout);
    
    ggml_backend_load_all();
    ggml_profile_set_callback(profile_cb, nullptr);
    ggml_profile_enable(true);
    printf("[DEBUG] Backends loaded and profiling enabled\n");
    fflush(stdout);

    ggml_backend_t backend_vk = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    // Try by name first (most reliable)
    backend_vk = ggml_backend_init_by_name("Vulkan0", "");
    backend_cpu = ggml_backend_init_by_name("CPU", "");
    
    // If not found, try by device type
    if (!backend_vk) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (dev) backend_vk = ggml_backend_dev_init(dev, "");
    }
    if (!backend_cpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev) backend_cpu = ggml_backend_dev_init(dev, "");
    }

    printf("Found backends: VK=%p CPU=%p\n", (void*)backend_vk, (void*)backend_cpu);
    fflush(stdout);

    if (!backend_vk || !backend_cpu) {
        printf("Missing backends - running CPU-only scheduler transformation test\n");
        // Continue with CPU-only test
        backend_cpu = ggml_backend_init_by_name("CPU", "");
        if (!backend_cpu) {
            printf("ERROR: CPU backend not available\n");
            return 1;
        }
        backend_vk = backend_cpu;  // Use CPU as both for testing transformation
    }

    // Test shapes from trace analysis - use N=4096 for FFN up
    int M = 512, K = 1024, N = 4096;
    int n_vulkan = 1024;  // Vulkan handles first 1024 output channels
    int n_cpu = N - n_vulkan;  // CPU handles remaining 3072

    printf("\n=== Testing M=%d K=%d N=%d (Vulkan N=%d, CPU N=%d) ===\n", M, K, N, n_vulkan, n_cpu);
    fflush(stdout);

    struct ggml_init_params params = { 0 };
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // Create tensors for original full MUL_MAT
    // A = [K, N] = weight matrix (Q8_0)
    // B = [K, M] = input matrix (Q8_0 or F32)
    // C = [N, M] = output
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c_full = ggml_mul_mat(ctx, a, b);

    // Fill with test data on host - use F32 for easier debugging
    std::vector<float> weight_data_f32(K * N);
    std::vector<float> input_data_f32(K * M);
    for (size_t i = 0; i < weight_data_f32.size(); i++) weight_data_f32[i] = (float)(i % 256) / 256.0f;
    for (size_t i = 0; i < input_data_f32.size(); i++) input_data_f32[i] = (float)((i * 7) % 256) / 256.0f;
    printf("[DEBUG] Test data generated\n"); fflush(stdout);

    // Test 1: CPU reference for full MUL_MAT
    printf("[TEST 1] CPU reference full MUL_MAT...\n"); fflush(stdout);
    ggml_build_forward_expand(gf, c_full);
    
    // Allocate buffers for the graph
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx, backend_cpu);
    if (!buf_cpu) {
        printf("ERROR: Failed to allocate CPU buffers!\n");
        return 1;
    }
    
    g_tracing.store(true); clear_events();
    auto t1 = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(backend_cpu, gf);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_full_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("CPU full: %ld ms\n", cpu_full_ms);
    g_tracing.store(false);
    save_events("trace_cpu_full.json");

    if (!c_full->buffer) {
        printf("ERROR: CPU output tensor has no buffer!\n");
        fflush(stdout);
        return 1;
    }
    
    std::vector<float> cpu_out(ggml_nbytes(c_full) / sizeof(float));
    ggml_backend_tensor_get(c_full, cpu_out.data(), 0, cpu_out.size() * sizeof(float));
    printf("CPU out[0..4]: %.2f %.2f %.2f %.2f %.2f\n", cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], cpu_out[4]);
    fflush(stdout);

    // Free CPU buffer
    ggml_backend_buffer_free(buf_cpu);

    // Test 2: Vulkan split with ggml views (replaces XDNA test)
    printf("[TEST 2] Vulkan split using ggml views...\n"); fflush(stdout);
    
    struct ggml_context* ctx2 = ggml_init(params);
    struct ggml_cgraph* gf2 = ggml_new_graph(ctx2);
    
    // Original weight tensor
    struct ggml_tensor* a2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K, N);
    struct ggml_tensor* b2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K, M);
    
    // Split weight matrix along N dimension
    struct ggml_tensor* a2_vk = ggml_view_2d(ctx2, a2, K, n_vulkan, a2->nb[1], 0);
    struct ggml_tensor* a2_cpu = ggml_view_2d(ctx2, a2, K, n_cpu, a2->nb[1], n_vulkan * a2->nb[0]);
    
    // MUL_MAT for each slice
    struct ggml_tensor* c2_vk = ggml_mul_mat(ctx2, a2_vk, b2);
    struct ggml_tensor* c2_cpu = ggml_mul_mat(ctx2, a2_cpu, b2);
    
    // Concat along N dimension
    struct ggml_tensor* c2_concat = ggml_concat(ctx2, c2_vk, c2_cpu, 0);
    
    // Build full graph
    gf2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf2, c2_concat);
    
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf2));
    for (int i = 0; i < ggml_graph_n_nodes(gf2); i++) {
        struct ggml_tensor* n = ggml_graph_node(gf2, i);
        printf("  Node %d: op=%s (%s) ne=[%d,%d,%d,%d]\n", i, ggml_op_name(n->op), n->name ? n->name : "unnamed",
               n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    }
    fflush(stdout);
    
    // Build full graph
    gf2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf2, c2_concat);
    
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf2));
    for (int i = 0; i < ggml_graph_n_nodes(gf2); i++) {
        struct ggml_tensor* n = ggml_graph_node(gf2, i);
        printf("  Node %d: op=%s (%s) ne=[%d,%d,%d,%d]\n", i, ggml_op_name(n->op), n->name ? n->name : "unnamed",
               n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    }
    fflush(stdout);
    
    // Allocate buffers for Vulkan using alloc_ctx_tensors
    ggml_backend_buffer_t buf_vk = ggml_backend_alloc_ctx_tensors(ctx2, backend_vk);
    if (!buf_vk) {
        printf("ERROR: Failed to allocate Vulkan buffers!\n");
        return 1;
    }
    printf("[DEBUG] Vulkan buffers allocated\n"); fflush(stdout);
    
    // Set data
    ggml_backend_tensor_set(a2, weight_data_f32.data(), 0, weight_data_f32.size() * sizeof(float));
    ggml_backend_tensor_set(b2, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    printf("[DEBUG] Vulkan buffers allocated and data set\n"); fflush(stdout);
    t1 = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(backend_vk, gf2);  // This will use Vulkan for all
    t2 = std::chrono::high_resolution_clock::now();
    auto vk_split_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("Vulkan split: %ld ms\n", vk_split_ms);
    g_tracing.store(false);
    save_events("trace_vk_split.json");

    if (!c2_concat->buffer) {
        printf("ERROR: Vulkan split output tensor has no buffer!\n");
        return 1;
    }

    // Get result
    std::vector<float> vk_split_out(ggml_nbytes(c2_concat) / sizeof(float));
    ggml_backend_tensor_get(c2_concat, vk_split_out.data(), 0, vk_split_out.size() * sizeof(float));
    printf("Vulkan split out[0..4]: %.2f %.2f %.2f %.2f %.2f\n", 
           vk_split_out[0], vk_split_out[1], vk_split_out[2], vk_split_out[3], vk_split_out[4]);
    fflush(stdout);

    // Compare
    float max_diff = 0;
    for (size_t i = 0; i < cpu_out.size(); i++) {
        float diff = fabs(cpu_out[i] - vk_split_out[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff (Vulkan split vs CPU): %.4f\n", max_diff);
    fflush(stdout);
    fflush(stdout);

    // Now test heterogeneous: Vulkan for first slice, CPU for second
    printf("[TEST 4] Heterogeneous: Vulkan + CPU split...\n"); fflush(stdout);
    
    // Create separate context for Vulkan slice
    struct ggml_context* ctx_vk = ggml_init(params);
    struct ggml_tensor* a4_vk = ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, K, n_vulkan);
    struct ggml_tensor* b4_vk = ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c4_vk = ggml_mul_mat(ctx_vk, a4_vk, b4_vk);
    
    struct ggml_cgraph* gf_vk = ggml_new_graph(ctx_vk);
    ggml_build_forward_expand(gf_vk, c4_vk);
    
    // Allocate Vulkan buffers
    ggml_backend_buffer_t buf_vk4 = ggml_backend_alloc_ctx_tensors(ctx_vk, backend_vk);
    if (!buf_vk4) {
        printf("ERROR: Failed to allocate Vulkan buffers!\n");
        return 1;
    }
    
    // Set Vulkan data
    printf("[DEBUG] Setting Vulkan data, a4_vk->data=%p\n", a4_vk->data); fflush(stdout);
    // Use ggml_backend_tensor_set to properly copy data
    // Create temporary flat arrays for the weight and input slices
    std::vector<float> weight_vk(K * n_vulkan);
    for (int n = 0; n < n_vulkan; n++) {
        for (int k = 0; k < K; k++) {
            weight_vk[k * n_vulkan + n] = weight_data_f32[k * N + n];
        }
    }
    ggml_backend_tensor_set(a4_vk, weight_vk.data(), 0, weight_vk.size() * sizeof(float));
    ggml_backend_tensor_set(b4_vk, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    printf("[DEBUG] Vulkan data set\n"); fflush(stdout);
    
    // Create separate context for CPU slice
    struct ggml_context* ctx_cpu4 = ggml_init(params);
    struct ggml_tensor* a4_cpu = ggml_new_tensor_2d(ctx_cpu4, GGML_TYPE_F32, K, n_cpu);
    struct ggml_tensor* b4_cpu = ggml_new_tensor_2d(ctx_cpu4, GGML_TYPE_F32, K, M);
    struct ggml_tensor* c4_cpu = ggml_mul_mat(ctx_cpu4, a4_cpu, b4_cpu);
    
    struct ggml_cgraph* gf_cpu4 = ggml_new_graph(ctx_cpu4);
    ggml_build_forward_expand(gf_cpu4, c4_cpu);
    
    // Allocate CPU buffers
    ggml_backend_buffer_t buf_cpu4 = ggml_backend_alloc_ctx_tensors(ctx_cpu4, backend_cpu);
    if (!buf_cpu4) {
        printf("ERROR: Failed to allocate CPU buffers!\n");
        return 1;
    }
    
    // Set CPU data
    printf("[DEBUG] Setting CPU data, a4_cpu->data=%p\n", a4_cpu->data); fflush(stdout);
    // Use ggml_backend_tensor_set to properly copy data
    std::vector<float> weight_cpu(K * n_cpu);
    for (int n = 0; n < n_cpu; n++) {
        for (int k = 0; k < K; k++) {
            weight_cpu[k * n_cpu + n] = weight_data_f32[k * N + n_vulkan + n];
        }
    }
    ggml_backend_tensor_set(a4_cpu, weight_cpu.data(), 0, weight_cpu.size() * sizeof(float));
    ggml_backend_tensor_set(b4_cpu, input_data_f32.data(), 0, input_data_f32.size() * sizeof(float));
    printf("[DEBUG] CPU data set\n"); fflush(stdout);
    
    // Run Vulkan slice
    printf("[DEBUG] Running Vulkan slice...\n"); fflush(stdout);
    g_tracing.store(true); clear_events();
    t1 = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(backend_vk, gf_vk);
    
    // Run CPU slice
    struct ggml_context* ctx_cpu2 = ggml_init(params);
    struct ggml_cgraph* gf_cpu2 = ggml_new_graph(ctx_cpu2);
    ggml_build_forward_expand(gf_cpu2, c4_cpu);
    
    // Run CPU slice
    ggml_backend_graph_compute(backend_cpu, gf_cpu2);
    t2 = std::chrono::high_resolution_clock::now();
    auto hetero_seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("Heterogeneous sequential: %ld ms\n", hetero_seq_ms);
    g_tracing.store(false);
    save_events("trace_hetero_seq.json");
    
    if (!c4_vk->buffer) {
        printf("ERROR: Hetero Vulkan slice output has no buffer!\n");
        fflush(stdout);
        return 1;
    }
    if (!c4_cpu->buffer) {
        printf("ERROR: Hetero CPU slice output has no buffer!\n");
        fflush(stdout);
        return 1;
    }
    
    // Now we need to concatenate - this needs to happen on a backend that has both outputs
    // For now, let's do concat on CPU
    struct ggml_context* ctx5 = ggml_init(params);
    struct ggml_tensor* c5_vk = ggml_new_tensor_2d(ctx5, GGML_TYPE_F32, n_vulkan, M);
    struct ggml_tensor* c5_cpu = ggml_new_tensor_2d(ctx5, GGML_TYPE_F32, n_cpu, M);
    struct ggml_tensor* c5_concat = ggml_concat(ctx5, c5_vk, c5_cpu, 0);
    
    // Allocate buffers for concat inputs
    ggml_backend_buffer_t buf5_vk = ggml_backend_alloc_ctx_tensors(ctx5, backend_cpu);
    if (!buf5_vk) {
        printf("ERROR: Failed to allocate concat input buffers!\n");
        fflush(stdout);
        return 1;
    }
    
    struct ggml_cgraph* gf_concat = ggml_new_graph(ctx5);
    ggml_build_forward_expand(gf_concat, c5_concat);
    
    // Copy results to concat inputs
    ggml_backend_tensor_get(c4_vk, c5_vk->data, 0, ggml_nbytes(c5_vk));
    ggml_backend_tensor_get(c4_cpu, c5_cpu->data, 0, ggml_nbytes(c5_cpu));
    
    ggml_backend_graph_compute(backend_cpu, gf_concat);
    
    if (!c5_concat->buffer) {
        printf("ERROR: Hetero concat output has no buffer!\n");
        fflush(stdout);
        return 1;
    }

    std::vector<float> hetero_out(ggml_nbytes(c5_concat) / sizeof(float));
    ggml_backend_tensor_get(c5_concat, hetero_out.data(), 0, hetero_out.size() * sizeof(float));
    
    // Compare
    max_diff = 0;
    for (size_t i = 0; i < cpu_out.size(); i++) {
        float diff = fabs(cpu_out[i] - hetero_out[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff (Hetero sequential vs CPU): %.4f\n", max_diff);
    fflush(stdout);

    // Test concurrent: run Vulkan and CPU in parallel threads
    printf("[TEST 5] Heterogeneous concurrent...\n"); fflush(stdout);
    
    std::atomic<bool> vk_done{false}, cpu_done{false};
    long vk_time = 0, cpu_time = 0;
    
    auto vk_thread = std::thread([&]() {
        auto t = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend_vk, gf_vk);
        vk_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t).count();
        vk_done = true;
    });
    
    auto cpu_thread = std::thread([&]() {
        auto t = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend_cpu, gf_cpu2);
        cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t).count();
        cpu_done = true;
    });
    
    auto t_start = std::chrono::high_resolution_clock::now();
    vk_thread.join();
    cpu_thread.join();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto hetero_concurrent_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    
    printf("Vulkan thread: %ld ms, CPU thread: %ld ms, Total concurrent: %ld ms\n", vk_time, cpu_time, hetero_concurrent_ms);
    fflush(stdout);
    
    if (!c4_vk->buffer) {
        printf("ERROR: Concurrent Vulkan slice output has no buffer!\n");
        fflush(stdout);
        return 1;
    }
    if (!c4_cpu->buffer) {
        printf("ERROR: Concurrent CPU slice output has no buffer!\n");
        fflush(stdout);
        return 1;
    }

    // Concat and verify
    ggml_backend_tensor_get(c4_vk, c5_vk->data, 0, ggml_nbytes(c5_vk));
    ggml_backend_tensor_get(c4_cpu, c5_cpu->data, 0, ggml_nbytes(c5_cpu));
    ggml_backend_graph_compute(backend_cpu, gf_concat);
    
    if (!c5_concat->buffer) {
        printf("ERROR: Concurrent concat output has no buffer!\n");
        fflush(stdout);
        return 1;
    }
    
    ggml_backend_tensor_get(c5_concat, hetero_out.data(), 0, hetero_out.size() * sizeof(float));
    
    max_diff = 0;
    for (size_t i = 0; i < cpu_out.size(); i++) {
        float diff = fabs(cpu_out[i] - hetero_out[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff (Hetero concurrent vs CPU): %.4f\n", max_diff);
    fflush(stdout);

    printf("\n=== SUMMARY ===\n");
    printf("CPU full:         %ld ms\n", cpu_full_ms);
    printf("Vulkan split:     %ld ms\n", vk_split_ms);
    printf("Hetero sequential:%ld ms\n", hetero_seq_ms);
    printf("Hetero concurrent:%ld ms\n", hetero_concurrent_ms);
    printf("Speedup (concurrent vs CPU): %.2fx\n", (double)cpu_full_ms / hetero_concurrent_ms);
    printf("Speedup (concurrent vs Vulkan split): %.2fx\n", (double)vk_split_ms / hetero_concurrent_ms);
    fflush(stdout);

    // Cleanup
    ggml_free(ctx);
    ggml_free(ctx2);
    ggml_free(ctx_vk);
    ggml_free(ctx_cpu4);
    ggml_free(ctx5);
    ggml_backend_free(backend_vk);
    ggml_backend_free(backend_cpu);

    return 0;
}