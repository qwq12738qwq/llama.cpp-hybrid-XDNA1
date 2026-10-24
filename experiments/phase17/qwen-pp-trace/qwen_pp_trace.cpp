/**
 * qwen_pp_trace.cpp - Capture Qwen3.5 prompt processing execution trace
 * 
 * This tool runs Qwen3.5-0.8B prompt processing at various context sizes
 * and captures detailed timing traces for each operation.
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

struct TraceEvent {
    uint64_t timestamp_ns;
    std::string backend_name;
    std::string op_name;
    int op_type;
    int graph_index;
    int layer;
    int M, K, N;
    int dtype_a, dtype_b, dtype_c;
    size_t tensor_size;
    bool xdna_compatible;
    std::string event_type; // "start" or "end"
};

static std::mutex g_trace_mutex;
static std::vector<TraceEvent> g_trace_events;
static std::atomic<bool> g_tracing_enabled{false};
static int g_current_context = 0;
static int g_current_iteration = 0;

void profile_callback(const ggml_profile_event_t * event, void * user_data) {
    if (!g_tracing_enabled.load()) return;
    
    TraceEvent trace_event;
    trace_event.timestamp_ns = event->timestamp_ns;
    trace_event.backend_name = event->backend_name ? event->backend_name : "unknown";
    trace_event.op_name = event->op_name ? event->op_name : "unknown";
    trace_event.op_type = event->op_type;
    trace_event.graph_index = event->graph_index;
    trace_event.layer = event->layer;
    trace_event.M = event->M;
    trace_event.K = event->K;
    trace_event.N = event->N;
    trace_event.dtype_a = event->dtype_a;
    trace_event.dtype_b = event->dtype_b;
    trace_event.dtype_c = event->dtype_c;
    trace_event.tensor_size = event->tensor_size;
    trace_event.xdna_compatible = event->xdna_compatible;
    trace_event.event_type = (event->type == GGML_PROFILE_EVENT_OP_START) ? "start" : "end";
    
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    g_trace_events.push_back(trace_event);
}

const char* op_type_name(int op_type) {
    switch (op_type) {
        case GGML_OP_MUL_MAT: return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case GGML_OP_ADD: return "ADD";
        case GGML_OP_RMS_NORM: return "RMS_NORM";
        case GGML_OP_ROPE: return "ROPE";
        case 100: return "SILU"; // GGML_OP_SILU value
        case GGML_OP_MUL: return "MUL";
        case GGML_OP_SOFT_MAX: return "SOFT_MAX";
        case GGML_OP_VIEW: return "VIEW";
        case GGML_OP_PERMUTE: return "PERMUTE";
        case GGML_OP_TRANSPOSE: return "TRANSPOSE";
        case GGML_OP_CPY: return "CPY";
        case GGML_OP_RESHAPE: return "RESHAPE";
        case GGML_OP_CONCAT: return "CONCAT";
        case GGML_OP_GET_ROWS: return "GET_ROWS";
        case GGML_OP_SSM_CONV: return "SSM_CONV";
        case GGML_OP_SSM_SCAN: return "SSM_SCAN";
        default: return "UNKNOWN";
    }
}

const char* dtype_name(int dtype) {
    switch (dtype) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_I16: return "I16";
        default: return "UNKNOWN";
    }
}

void escape_json(std::string & str) {
    std::string result;
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: 
                if (c >= 0 && c < 32) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    str = result;
}

void save_trace_json(const std::string & filename) {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    
    std::ofstream f(filename);
    f << "{\n";
    f << "  \"context_size\": " << g_current_context << ",\n";
    f << "  \"iteration\": " << g_current_iteration << ",\n";
    f << "  \"events\": [\n";
    
    for (size_t i = 0; i < g_trace_events.size(); i++) {
        const auto & e = g_trace_events[i];
        std::string backend = e.backend_name;
        std::string op_name = e.op_name;
        std::string event_type = e.event_type;
        escape_json(backend);
        escape_json(op_name);
        escape_json(event_type);
        
        f << "    {\n";
        f << "      \"timestamp_ns\": " << e.timestamp_ns << ",\n";
        f << "      \"backend\": \"" << backend << "\",\n";
        f << "      \"op_name\": \"" << op_name << "\",\n";
        f << "      \"op_type\": \"" << op_type_name(e.op_type) << "\",\n";
        f << "      \"graph_index\": " << e.graph_index << ",\n";
        f << "      \"layer\": " << e.layer << ",\n";
        f << "      \"M\": " << e.M << ",\n";
        f << "      \"K\": " << e.K << ",\n";
        f << "      \"N\": " << e.N << ",\n";
        f << "      \"dtype_a\": \"" << dtype_name(e.dtype_a) << "\",\n";
        f << "      \"dtype_b\": \"" << dtype_name(e.dtype_b) << "\",\n";
        f << "      \"dtype_c\": \"" << dtype_name(e.dtype_c) << "\",\n";
        f << "      \"tensor_size_bytes\": " << e.tensor_size << ",\n";
        f << "      \"xdna_compatible\": " << (e.xdna_compatible ? "true" : "false") << ",\n";
        f << "      \"event_type\": \"" << event_type << "\"\n";
        f << "    }";
        if (i + 1 < g_trace_events.size()) f << ",";
        f << "\n";
    }
    
    f << "  ]\n";
    f << "}\n";
    f.close();
    
    printf("[TRACE] Saved %zu events to %s\n", g_trace_events.size(), filename.c_str());
}

void clear_trace() {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    g_trace_events.clear();
}

int main(int argc, char ** argv) {
    std::string model_path = "/workspace/llama.cpp/models/Qwen3.5-0.8B-Q8_0.gguf";
    int n_ctx = 512;
    int n_iterations = 3;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iterations = std::stoi(argv[++i]);
        }
    }
    
    // Load backends
    ggml_backend_load_all();
    
    // Enable profiling
    ggml_profile_set_callback(profile_callback, nullptr);
    ggml_profile_enable(true);
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // Offload all possible layers
    
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
    
    // Test prompt - use a simple prompt that fills the context
    std::string prompt = "The quick brown fox jumps over the lazy dog. ";
    while ((int)prompt.size() < n_ctx / 4) {
        prompt += prompt;
    }
    prompt = prompt.substr(0, n_ctx / 4);
    
    printf("[TRACE] Running Qwen3.5-0.8B PP trace: context=%d, iterations=%d\n", n_ctx, n_iterations);
    
    for (int iter = 0; iter < n_iterations; iter++) {
        g_current_context = n_ctx;
        g_current_iteration = iter;
        clear_trace();
        g_tracing_enabled.store(true);
        
        // Tokenize
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
        
        g_tracing_enabled.store(false);
        
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("[TRACE] Iteration %d: %ld ms\n", iter, duration_ms);
        
        // Save trace
        char filename[256];
        snprintf(filename, sizeof(filename), "qwen_pp_trace_ctx%d_iter%d.json", n_ctx, iter);
        save_trace_json(filename);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    
    return 0;
}