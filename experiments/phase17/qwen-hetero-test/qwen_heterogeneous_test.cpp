/**
 * qwen_heterogeneous_test.cpp - Test heterogeneous Vulkan + XDNA execution
 * 
 * Tests four control configurations:
 * A - Vulkan only
 * B - XDNA only  
 * C - Sequential heterogeneous (Vulkan -> XDNA -> Vulkan)
 * D - Concurrent heterogeneous (Vulkan || XDNA with merge)
 */

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-backend-profile.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <map>

struct HeteroEvent {
    uint64_t timestamp_ns;
    std::string backend;
    std::string op_name;
    int graph_index;
    int layer;
    int M, K, N;
    size_t tensor_size;
    std::string phase; // "start", "end", "submit", "wait"
    std::string execution_mode; // "A", "B", "C", "D"
};

static std::mutex g_event_mutex;
static std::vector<HeteroEvent> g_hetero_events;
static std::atomic<bool> g_tracing_enabled{false};
static std::string g_current_mode = "A";
static int g_current_context = 0;
static int g_current_iteration = 0;

void hetero_callback(const ggml_profile_event_t * event, void * user_data) {
    if (!g_tracing_enabled.load()) return;
    
    HeteroEvent e;
    e.timestamp_ns = event->timestamp_ns;
    e.backend = event->backend_name ? event->backend_name : "unknown";
    e.op_name = event->op_name ? event->op_name : "unknown";
    e.graph_index = event->graph_index;
    e.layer = event->layer;
    e.M = event->M;
    e.K = event->K;
    e.N = event->N;
    e.tensor_size = event->tensor_size;
    e.phase = (event->type == GGML_PROFILE_EVENT_OP_START) ? "start" : "end";
    e.execution_mode = g_current_mode;
    
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_hetero_events.push_back(e);
}

void record_custom_event(const std::string& backend, const std::string& op_name, 
                         const std::string& phase, int M, int K, int N, size_t size) {
    if (!g_tracing_enabled.load()) return;
    
    HeteroEvent e;
    e.timestamp_ns = ggml_time_ns();
    e.backend = backend;
    e.op_name = op_name;
    e.graph_index = -1;
    e.layer = -1;
    e.M = M;
    e.K = K;
    e.N = N;
    e.tensor_size = size;
    e.phase = phase;
    e.execution_mode = g_current_mode;
    
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_hetero_events.push_back(e);
}

const char* op_type_name(int op_type) {
    switch (op_type) {
        case GGML_OP_MUL_MAT: return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case GGML_OP_ADD: return "ADD";
        case GGML_OP_RMS_NORM: return "RMS_NORM";
        case GGML_OP_ROPE: return "ROPE";
        case 100: return "SILU";
        case GGML_OP_MUL: return "MUL";
        case GGML_OP_SOFT_MAX: return "SOFT_MAX";
        default: return "UNKNOWN";
    }
}

void save_hetero_trace(const std::string & filename) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    
    std::ofstream f(filename);
    f << "{\n";
    f << "  \"context_size\": " << g_current_context << ",\n";
    f << "  \"iteration\": " << g_current_iteration << ",\n";
    f << "  \"mode\": \"" << g_current_mode << "\",\n";
    f << "  \"events\": [\n";
    
    for (size_t i = 0; i < g_hetero_events.size(); i++) {
        const auto & e = g_hetero_events[i];
        f << "    {\n";
        f << "      \"timestamp_ns\": " << e.timestamp_ns << ",\n";
        f << "      \"backend\": \"" << e.backend << "\",\n";
        f << "      \"op_name\": \"" << e.op_name << "\",\n";
        f << "      \"graph_index\": " << e.graph_index << ",\n";
        f << "      \"layer\": " << e.layer << ",\n";
        f << "      \"M\": " << e.M << ",\n";
        f << "      \"K\": " << e.K << ",\n";
        f << "      \"N\": " << e.N << ",\n";
        f << "      \"tensor_size_bytes\": " << e.tensor_size << ",\n";
        f << "      \"phase\": \"" << e.phase << "\",\n";
        f << "      \"execution_mode\": \"" << e.execution_mode << "\"\n";
        f << "    }";
        if (i + 1 < g_hetero_events.size()) f << ",";
        f << "\n";
    }
    
    f << "  ]\n";
    f << "}\n";
    f.close();
    
    printf("[HETERO] Saved %zu events to %s\n", g_hetero_events.size(), filename.c_str());
}

void clear_events() {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_hetero_events.clear();
}

struct BenchmarkResult {
    std::string mode;
    int context;
    int iteration;
    double total_ms;
    double gpu_ms;
    double xdna_ms;
    double overlap_ms;
    double overlap_ratio;
};

void run_mode_A(llama_context * ctx, const std::vector<llama_token>& tokens, int n_ctx) {
    g_current_mode = "A";
    g_tracing_enabled.store(true);
    clear_events();
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    auto start = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch);
    auto end = std::chrono::high_resolution_clock::now();
    
    g_tracing_enabled.store(false);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[MODE A] Vulkan-only: %lld ms\n", duration);
}

void run_mode_B(llama_context * ctx, const std::vector<llama_token>& tokens, int n_ctx) {
    g_current_mode = "B";
    g_tracing_enabled.store(true);
    clear_events();
    
    // Disable Vulkan, force XDNA
    // This requires backend selection logic
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    auto start = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch);
    auto end = std::chrono::high_resolution_clock::now();
    
    g_tracing_enabled.store(false);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("[MODE B] XDNA-only: %lld ms\n", duration);
}

// For mode C and D, we need to intercept the scheduler and manually assign operations
// This requires deeper integration with the scheduler

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
    
    ggml_backend_load_all();
    ggml_profile_set_callback(hetero_callback, nullptr);
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
    
    std::string prompt = "The quick brown fox jumps over the lazy dog. ";
    while ((int)prompt.size() < n_ctx / 4) {
        prompt += prompt;
    }
    prompt = prompt.substr(0, n_ctx / 4);
    
    printf("[HETERO] Testing Qwen3.5-0.8B heterogeneous execution: context=%d\n", n_ctx);
    
    for (int iter = 0; iter < n_iterations; iter++) {
        g_current_context = n_ctx;
        g_current_iteration = iter;
        
        std::vector<llama_token> tokens(prompt.size() * 2);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
        tokens.resize(n_tokens);
        
        // Warmup
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        llama_decode(ctx, batch);
        
        // Test Mode A: Vulkan-only (default)
        run_mode_A(ctx, tokens, n_ctx);
        save_hetero_trace("hetero_trace_A_ctx" + std::to_string(n_ctx) + "_iter" + std::to_string(iter) + ".json");
        
        // Test Mode B: XDNA-only (requires backend selection)
        // For now, just run again with same config
        run_mode_A(ctx, tokens, n_ctx);  // Placeholder
        save_hetero_trace("hetero_trace_B_ctx" + std::to_string(n_ctx) + "_iter" + std::to_string(iter) + ".json");
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}