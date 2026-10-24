/**
 * qwen_heterogeneous_pp.cpp - Heterogeneous Vulkan + XDNA Qwen3.5 PP test
 * 
 * Runs Qwen3.5-0.8B prompt processing with MUL_MAT split along N dimension:
 * Vulkan handles first N_vulkan columns, XDNA handles remaining N_xdna columns
 */

#include "llama.h"
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
#include <unordered_map>

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
static std::string g_current_mode = "A";
static int g_current_context = 0;

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

int main(int argc, char ** argv) {
    std::string model_path = "/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf";
    int n_ctx = 512;
    int n_iterations = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iterations = std::stoi(argv[++i]);
        }
    }
    
    printf("[HETERO] Testing Qwen3.5-0.8B heterogeneous PP: context=%d\n", n_ctx);
    fflush(stdout);
    
    ggml_backend_load_all();
    ggml_profile_set_callback(profile_cb, nullptr);
    ggml_profile_enable(true);
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Test prompt
    std::string prompt = "The quick brown fox jumps over the lazy dog. ";
    while ((int)prompt.size() < n_ctx / 4) {
        prompt += prompt;
    }
    prompt = prompt.substr(0, n_ctx / 4);
    
    for (int iter = 0; iter < n_iterations; iter++) {
        g_current_context = n_ctx;
        clear_events();
        g_tracing.store(true);
        
        std::vector<llama_token> tokens(prompt.size() * 2);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
        tokens.resize(n_tokens);
        
        // Warmup
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        llama_decode(ctx, batch);
        
        // Measured run
        batch = llama_batch_get_one(tokens.data(), n_tokens);
        auto start = std::chrono::high_resolution_clock::now();
        llama_decode(ctx, batch);
        auto end = std::chrono::high_resolution_clock::now();
        
        g_tracing.store(false);
        
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("[HETERO] Iteration %d: %ld ms\n", iter, duration_ms);
        
        // Save trace
        char filename[256];
        snprintf(filename, sizeof(filename), "hetero_trace_ctx%d_iter%d.json", n_ctx, iter);
        save_events(filename);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}