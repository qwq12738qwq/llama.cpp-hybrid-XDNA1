/**
 * hetero_mul_mat_test.cpp - Direct test of heterogeneous MUL_MAT execution
 * 
 * Tests MUL_MAT on Vulkan vs XDNA vs heterogeneous (split)
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

static void run_mul_mat(ggml_backend_t backend, struct ggml_tensor* a, struct ggml_tensor* b, struct ggml_tensor* c, struct ggml_cgraph* gf) {
    // Build a minimal graph with just this MUL_MAT
    printf("  [DEBUG] Building graph...\n");
    ggml_build_forward_expand(gf, c);
    printf("  [DEBUG] Graph built\n");
    printf("  [DEBUG] Computing on backend...\n");
    ggml_backend_graph_compute(backend, gf);
    printf("  [DEBUG] Compute done\n");
}

int main() {
    printf("[DEBUG] Starting heterogeneous MUL_MAT test\n");
    ggml_profile_set_callback(profile_cb, nullptr);
    ggml_profile_enable(true);
    printf("[DEBUG] Profiling enabled\n");

    // Find backends - use load_all to get them registered
    ggml_backend_load_all();
    printf("[DEBUG] Backends loaded\n");

    ggml_backend_t backend_vk = nullptr;
    ggml_backend_t backend_xdna = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    printf("[DEBUG] Looking for backends...\n");
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char* name = ggml_backend_reg_name(reg);
        printf("[DEBUG] Found backend: %s\n", name);
        if (strcmp(name, "Vulkan") == 0) {
            backend_vk = ggml_backend_init_by_name(name, "");
            printf("Found Vulkan backend\n");
        } else if (strcmp(name, "XDNA") == 0) {
            backend_xdna = ggml_backend_init_by_name(name, "");
            printf("Found XDNA backend\n");
        } else if (strcmp(name, "CPU") == 0) {
            backend_cpu = ggml_backend_init_by_name(name, "");
            printf("Found CPU backend\n");
        }
    }

    if (!backend_vk) {
        fprintf(stderr, "Vulkan backend not found\n");
        return 1;
    }
    if (!backend_xdna) {
        fprintf(stderr, "XDNA backend not found\n");
        return 1;
    }
    if (!backend_cpu) {
        fprintf(stderr, "CPU backend not found\n");
        return 1;
    }

    // Test shapes from trace analysis
    struct {
        int M, K, N;
        const char* name;
    } shapes[] = {
        {512, 1024, 4096, "FFN_up"},
        //{512, 1024, 6144, "FFN_gate_up"},
        //{512, 1024, 2048, "FFN_down"},
        //{448, 1024, 4096, "Attn_K"},
        //{448, 1024, 6144, "Attn_QV"},
    };

    for (auto & shape : shapes) {
        int M = shape.M, K = shape.K, N = shape.N;
        printf("\n=== Testing %s: M=%d K=%d N=%d ===\n", shape.name, M, K, N);
        fflush(stdout);

        struct ggml_init_params params = { 0 };
        params.mem_size = 128 * 1024 * 1024;
        params.mem_buffer = nullptr;
        struct ggml_context* ctx = ggml_init(params);
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        printf("[DEBUG] Context created, mem_size: %zu\n", params.mem_size);
        fflush(stdout);

        // Create tensors: a = [K, N] (weight), b = [K, M] (input), c = [N, M] (output)
        printf("[DEBUG] Creating tensors...\n");
        fflush(stdout);
        struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
        struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
        printf("[DEBUG] Tensors created: a=[%d,%d], b=[%d,%d], c=[%d,%d]\n", 
               a->ne[0], a->ne[1], b->ne[0], b->ne[1], c->ne[0], c->ne[1]);

        // Fill with test data
        std::vector<uint8_t> weight_data(ggml_nbytes(a));
        std::vector<uint8_t> input_data(ggml_nbytes(b));
        for (size_t i = 0; i < weight_data.size(); i++) weight_data[i] = i % 256;
        for (size_t i = 0; i < input_data.size(); i++) input_data[i] = (i * 7) % 256;
        printf("[DEBUG] Test data generated\n");
        fflush(stdout);

        // Allocate on each backend (buffers needed for tensor allocation)
        printf("[DEBUG] Allocating buffers...\n");
        fflush(stdout);
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(a));
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(b));
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(c));

        ggml_backend_alloc_buffer(backend_xdna, ggml_nbytes(a));
        ggml_backend_alloc_buffer(backend_xdna, ggml_nbytes(b));
        ggml_backend_alloc_buffer(backend_xdna, ggml_nbytes(c));

        ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(a));
        ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(b));
        ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(c));
        printf("[DEBUG] Buffers allocated\n");
        fflush(stdout);

        // Copy data to all backends
        printf("[DEBUG] Setting tensor data...\n");
        fflush(stdout);
        ggml_backend_tensor_set(a, weight_data.data(), 0, weight_data.size());
        ggml_backend_tensor_set(b, input_data.data(), 0, input_data.size());
        printf("[DEBUG] Tensor data set\n");
        fflush(stdout);

        // Get CPU reference
        printf("[DEBUG] Running CPU reference...\n");
        fflush(stdout);
        g_tracing.store(true); clear_events();
        run_mul_mat(backend_cpu, a, b, c, gf);
        g_tracing.store(false);
        printf("[DEBUG] CPU reference done\n");
        fflush(stdout);
        std::vector<float> cpu_out(ggml_nbytes(c) / sizeof(float));
        ggml_backend_tensor_get(c, cpu_out.data(), 0, cpu_out.size() * sizeof(float));
        printf("CPU reference: first 5: %.2f %.2f %.2f %.2f %.2f\n",
               cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], cpu_out[4]);

        // Test Vulkan
        g_tracing.store(true); clear_events();
        auto t1 = std::chrono::high_resolution_clock::now();
        run_mul_mat(backend_vk, a, b, c, gf);
        auto t2 = std::chrono::high_resolution_clock::now();
        g_tracing.store(false);
        auto vk_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("Vulkan: %ld ms\n", vk_ms);
        save_events("trace_vk.json");

        // Test XDNA
        g_tracing.store(true); clear_events();
        t1 = std::chrono::high_resolution_clock::now();
        run_mul_mat(backend_xdna, a, b, c, gf);
        t2 = std::chrono::high_resolution_clock::now();
        g_tracing.store(false);
        auto xdna_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("XDNA: %ld ms\n", xdna_ms);
        save_events("trace_xdna.json");

        // Verify XDNA correctness
        std::vector<float> xdna_out(ggml_nbytes(c) / sizeof(float));
        ggml_backend_tensor_get(c, xdna_out.data(), 0, xdna_out.size() * sizeof(float));
        float max_diff = 0;
        for (size_t i = 0; i < cpu_out.size(); i++) {
            float diff = fabs(cpu_out[i] - xdna_out[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max diff (XDNA vs CPU): %.4f\n", max_diff);

        // Test Heterogeneous: Split N dimension
        // Vulkan does N[0:split], XDNA does N[split:N]
        int split = N / 2;
        printf("Heterogeneous split at N=%d\n", split);

        struct ggml_context* ctx2 = ggml_init(params);
        struct ggml_cgraph* gf2 = ggml_new_graph(ctx2);
        struct ggml_tensor* a2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q8_0, K, N);
        struct ggml_tensor* b2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_Q8_0, K, M);
        
        // Split weight matrix along N (columns)
        struct ggml_tensor* a_vk = ggml_view_2d(ctx2, a2, K, split, a2->nb[1], 0);
        struct ggml_tensor* a_xdna = ggml_view_2d(ctx2, a2, K, N - split, a2->nb[1], split * a2->nb[0]);
        
        struct ggml_tensor* c_vk = ggml_mul_mat(ctx2, a_vk, b2);
        struct ggml_tensor* c_xdna = ggml_mul_mat(ctx2, a_xdna, b2);
        
        // Combine results
        struct ggml_tensor* c_combined = ggml_concat(ctx2, c_vk, c_xdna, 0);  // concat along N (dim 0)

        gf2 = ggml_new_graph(ctx2);
        ggml_build_forward_expand(gf2, c_combined);

        // Allocate
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(a2));
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(b2));
        ggml_backend_alloc_buffer(backend_vk, ggml_nbytes(c_combined));
        
        ggml_backend_tensor_set(a2, weight_data.data(), 0, weight_data.size());
        ggml_backend_tensor_set(b2, input_data.data(), 0, input_data.size());

        g_tracing.store(true); clear_events();
        t1 = std::chrono::high_resolution_clock::now();
        
        // Run Vulkan part
        run_mul_mat(backend_vk, a_vk, b2, c_vk, gf2);
        // Run XDNA part  
        run_mul_mat(backend_xdna, a_xdna, b2, c_xdna, gf2);
        
        t2 = std::chrono::high_resolution_clock::now();
        g_tracing.store(false);
        auto hetero_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("Heterogeneous (sequential): %ld ms\n", hetero_ms);
        save_events("trace_hetero_seq.json");

        // Verify hetero
        std::vector<float> hetero_out(ggml_nbytes(c_combined) / sizeof(float));
        ggml_backend_tensor_get(c_combined, hetero_out.data(), 0, hetero_out.size() * sizeof(float));
        max_diff = 0;
        for (size_t i = 0; i < cpu_out.size(); i++) {
            float diff = fabs(cpu_out[i] - hetero_out[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max diff (Hetero vs CPU): %.4f\n", max_diff);

        // Speedup
        printf("Speedup Vulkan->Hetero: %.2fx\n", (double)vk_ms / hetero_ms);
        printf("Speedup XDNA->Hetero: %.2fx\n", (double)xdna_ms / hetero_ms);

        ggml_free(ctx);
        ggml_free(ctx2);
    }

    ggml_backend_free(backend_vk);
    ggml_backend_free(backend_xdna);
    ggml_backend_free(backend_cpu);

    return 0;
}