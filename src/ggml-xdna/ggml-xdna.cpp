#include "ggml-xdna.h"
#include "xdna-persistent.h"
#include "xdna-kernel-registry.h"
#include "xdna-q8-scale.h"
#include "xdna-q8-c4.h"
#include "xdna-q8-c4-device-scale.h"
#include "xdna-q8-scale-c4.h"
#include "xdna-whole-array.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-profile.h"
#include "ggml-alloc.h"

#include <dlfcn.h>
#include <link.h>

#include <cassert>
#include <cstdio>
#include <sys/stat.h>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <cstdarg>
#include <atomic>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <atomic>

static bool is_xdna_buffer(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return false;
    extern ggml_backend_buffer_type_t ggml_xdna_buffer_type(void);
    ggml_backend_buffer_type_t xdna_buft = ggml_xdna_buffer_type();
    if (!xdna_buft) return false;
    const char * xdna_name = xdna_buft->iface.get_name(xdna_buft);
    const char * buf_name  = tensor->buffer->buft->iface.get_name(tensor->buffer->buft);
    return xdna_name && buf_name && strcmp(xdna_name, buf_name) == 0;
}

// Get direct pointer to tensor data, bypassing ggml_backend_tensor_get when
// the tensor is resident in an XDNA-owned XRT BO.
static const void * get_tensor_data(const struct ggml_tensor * tensor) {
    if (is_xdna_buffer(tensor)) {
        return tensor->data;
    }
    return nullptr;
}

static void * get_tensor_data_mut(struct ggml_tensor * tensor) {
    if (is_xdna_buffer(tensor)) {
        return tensor->data;
    }
    return nullptr;
}

// Phase 20G step 8: an in-flight XDNA submission. graph_compute queues these
// with xrt::run.start() and does not wait; synchronize / event_wait drain them.
struct ggml_xdna_pending_sub {
    ggml_xdna_persistent_engine * engine;
    ggml_xdna_q8_scale_engine * q8_engine;
    ggml_xdna_q8_c4_engine * q8_c4_engine;
    ggml_xdna_q8_c4_ds_engine * q8_c4_ds_engine;
    ggml_xdna_q8_scale_c4_engine * q8_scale_c4_engine;
    ggml_xdna_q8_c4_mmul_v2_engine * q8_c4_mmul_v2_engine;
    ggml_xdna_persistent_engine * requant_engine;
    std::vector<float> requant_sa;
    std::vector<float> requant_sb;
    std::vector<int32_t> verify_ref;
    std::vector<float>  verify_fref;
    int    verify_rows = 0;
    float  verify_amax = 0.0f;
    float  verify_wmax = 0.0f;
    float  verify_smax = 0.0f;
    xrt::run * run;
    struct ggml_tensor * dst;
    int M;
    int N;
    int buf_idx;
    ggml_xdna_wa_engine * wa_engine;
    float wa_scale;
    int K;
    ggml_xdna_xrt_trace xrt_trace;
    int layer;
    std::string operation;
    std::string engine_name;
    std::string kernel_key;
    std::string kernel_source;
    std::string xclbin;
    std::string variant;
    std::string input_shape;
    std::string weight_shape;
    std::string output_shape;
    int n_cols;
    int m_tile;
    int n_tile;
    int k_tile;
};

struct ggml_xdna_backend_context {
    std::mutex engines_mutex;

    // async submission queue
    std::mutex pending_mutex;
    std::vector<ggml_xdna_pending_sub> pending;
    std::atomic<uint64_t> submitted_seq{0};
    std::atomic<uint64_t> completed_seq{0};

    // Persistent mode: disable automatic draining during graph_compute
    bool persistent_mode = false;
    int persistent_pending_count = 0;

    // Cached CPU backend used to flush host buffers after NPU execution.
    ggml_backend_t cpu_backend = nullptr;

    // Shared XRT device for buffer allocations
    xrt::device xrt_dev;
    bool xrt_dev_initialized = false;
    std::mutex xrt_dev_mutex;
};

static std::mutex g_xdna_active_backends_mutex;
static std::vector<ggml_xdna_backend_context *> g_xdna_active_backends;
static bool ggml_xdna_drain_pending(ggml_xdna_backend_context * ctx);

static bool ggml_xdna_glu_supported(const struct ggml_tensor * op);
static bool ggml_xdna_glu_compute(struct ggml_tensor * node);

void ggml_xdna_synchronize_all() {
    std::lock_guard<std::mutex> lock(g_xdna_active_backends_mutex);
    for (ggml_xdna_backend_context * ctx : g_xdna_active_backends) {
        (void) ggml_xdna_drain_pending(ctx);
    }
}

struct ggml_xdna_device_context {
    int device_id;
    ggml_backend_buffer_type buffer_type;
};

// Phase 20G step 8: event marker. On this single-queue device an event only needs
// to remember the submission sequence it was recorded at.
struct ggml_xdna_event {
    ggml_xdna_backend_context * ctx;
    uint64_t seq;
};

// Persistent engine counters (used by xdna-persistent.cpp)
std::atomic<int> g_xdna_persistent_execute_count{0};
std::atomic<int> g_xdna_xrt_submit_count{0};
std::atomic<int> g_xdna_xrt_complete_count{0};

// Phase 20E: XDNA op accounting (criterion D) and per-stage timings owned by
// xdna-persistent.cpp (criterion E: no hidden synchronization).
std::atomic<int> g_xdna_executed_ops{0};
std::atomic<int> g_xdna_missing_kernel_ops{0};
static std::atomic<int> g_xdna_glu_ops{0};
extern "C" void ggml_xdna_print_timings(void);

static const char* GGML_XDNA_BACKEND_NAME = "XDNA";

static void ggml_xdna_log(const char* format, ...) {
    const char* env = std::getenv("GGML_XDNA_DEBUG");
    if (!env || *env == '\0') return;

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

static XDNAKernelKey tensor_to_kernel_key(const struct ggml_tensor* op) {
    XDNAKernelKey key;
    key.op_type = XDNAOpType::MUL_MAT;
    key.quantization = XDNAQuantization::Q8_0;
    
    // Check env var for Q8 engine selection
    const char * q8_engine = std::getenv("XDNA_Q8_ENGINE");
    bool use_c4 = false;
    bool use_q8scale_c4 = false;
    bool use_c4_mmul_v2 = false;
    bool use_c4_mmul_v3 = false;
    bool use_c4_mmul_v4 = false;
    bool use_s12 = false;
    bool use_requant = false;
    bool use_int8_pure_v2 = false;
    bool use_int8_pure_v3 = false;
    if (q8_engine) {
        if (strcasecmp(q8_engine, "c4_ds") == 0 || strcasecmp(q8_engine, "c4") == 0) {
            use_c4 = true;
        } else if (strcasecmp(q8_engine, "scale") == 0) {
            use_q8scale_c4 = true;
        } else if (strcasecmp(q8_engine, "c4_mmul_v2") == 0) {
            use_c4_mmul_v2 = true;
        } else if (strcasecmp(q8_engine, "c4_mmul_v3") == 0) {
            // phase 61: scale preserving, native 8x8 mmul, 16-lane float acc
            use_c4_mmul_v3 = true;
        } else if (strcasecmp(q8_engine, "c4_mmul_v4") == 0) {
            // phase 62: per-K-chunk scale, requantized weights, same C4 host path
            use_c4_mmul_v4 = true;
        } else if (strcasecmp(q8_engine, "s12") == 0) {
            use_s12 = true;
        } else if (strcasecmp(q8_engine, "int8_requant") == 0) {
            use_requant = true;
        } else if (strcasecmp(q8_engine, "int8_requant_v2") == 0) {
            use_int8_pure_v2 = true;
        } else if (strcasecmp(q8_engine, "int8_requant_v3") == 0) {
            use_int8_pure_v3 = true;
        } else if (strcasecmp(q8_engine, "int8_requant_v4") == 0) {
            // Phase 60 v4: native 8x8 mmul, 32x64x512 tile. Same host path as
            // the other pure-int8 engines; the tile shape comes from the engine.
            use_int8_pure_v2 = true;
        }
    }
    key.n_cols = (use_c4 || use_q8scale_c4 || use_c4_mmul_v2 || use_c4_mmul_v3 || use_c4_mmul_v4 || use_s12 || use_requant || use_int8_pure_v2 || use_int8_pure_v3) ? 4 : 1;
    
    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];
    
    key.dtype_a = src0 ? src0->type : GGML_TYPE_Q8_0;
    key.dtype_b = src1 ? src1->type : GGML_TYPE_Q8_0;
    key.dtype_c = op->type;
    
    int K = src0 ? src0->ne[0] : 1024;
    int N = src0 ? src0->ne[1] : 512;
    int M = src1 ? src1->ne[1] : 1;
    
    // Q8_0 and I8 both run the full K in a single submit (k_tile == K).
    // For Q8_0 the Q8_0 scale-aware kernel applies the per-32-block scales
    // inside the device kernel; I8 has no scales.
    bool is_q8_0_weight = (src0 && src0->type == GGML_TYPE_Q8_0);
    bool is_i8_weight = (src0 && src0->type == GGML_TYPE_I8);
    
    if ((is_q8_0_weight || is_i8_weight) && M >= 1) {
        key.m_tile = 32;
        key.k_tile = K;
        key.n_tile = 64;
        key.M = M;
        key.K = K;
        key.N = N;
        return key;
    }
    
    // Non-production shapes: zero key signals unsupported
    key.m_tile = 0;
    key.k_tile = 0;
    key.n_tile = 0;
    key.n_cols = 0;
    return key;
}

static bool is_qwen_pp_shape_allowed(int M, int K, int N) {
    // Phase 37: Qwen3.5-0.8B PP routing.
    // Allow only shapes with proven XDNA advantage at M=512.
    // ffn_gate/ffn_up: [1024, 3584] - 1.41x faster
    // ffn_down: [3584, 1024] - 1.68x faster
    // attn_q: [1024, 4096] - 1.37x faster
    // attn_output excluded because 0.91x at M=512.
    if (K == 1024 && N == 3584 && M >= 64) return true;
    if (K == 3584 && N == 1024 && M >= 64) return true;
    if (K == 1024 && N == 4096 && M >= 64) return true;
    return false;
}

// Phase 47B: the Phase 37 speedups were measured on a different kernel with a
// device-only timing boundary. With the production Q8SCALEV kernel the same
// shapes run at 6.8-9.5 GOPS against ~1900 GOPS on the 780M, so one NPU FFN
// GEMM (435-552 ms) costs more than a whole Vulkan PP512 (414 ms). Keep these
// shapes behind the explicit opt-in instead of routing them by default.
static bool is_phase47b_gated_shape(int K, int N) {
    return (K == 1024 && N == 3584) || (K == 3584 && N == 1024) || (K == 1024 && N == 4096);
}

// Phase 63: a cost driven routing decision.
//
// The allowlist above is a fixed shape list that was measured with an old
// kernel.  The phase 62/63 measurements show that the decision belongs to
// the *cost* of the op on both devices, and that the cost depends on the
// kernel and on the device, so it belongs in a table that can be replaced
// without a rebuild.
//
// The table below holds the measured device times of one GEMM (ms) per shape
// class, as measured in phase 63 (the NPU numbers are the phase 62 c256x2
// kernel, the GPU numbers the phase 62 Vulkan profile).  A shape is offloaded
// only when the NPU is faster than the GPU for it, or when the op is small
// enough that the cross device transfer cannot dominate it.
//
// The table can be replaced from the environment:
//   XDNA_COST_TABLE="K,N,npu_ms,gpu_ms;K,N,npu_ms,gpu_ms;..."
// and the policy can be switched on with XDNA_COST_ROUTE=1.  Without that
// environment variable the historical allowlist behaviour is kept.
struct ggml_xdna_cost_entry {
    int K;
    int N;
    double npu_ms;
    double gpu_ms;
};

static const ggml_xdna_cost_entry g_xdna_cost_table[] = {
    // the FFN gate/up GEMM of Qwen3.5-0.8B (M=512)
    { 1024, 3584, 113.1, 3.55 },
    // the FFN down GEMM (M=512)
    { 3584, 1024, 109.8, 2.70 },
    // the attention output projection (M=512)
    { 1024, 4096, 113.1, 3.55 },
};

static bool ggml_xdna_cost_lookup(int M, int K, int N, double * npu_ms, double * gpu_ms) {
    const char * env = std::getenv("XDNA_COST_TABLE");
    if (env && *env) {
        std::string s(env);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t end = s.find(';', pos);
            if (end == std::string::npos) end = s.size();
            std::string item = s.substr(pos, end - pos);
            pos = end + 1;
            if (item.empty()) continue;
            int k = 0, n = 0;
            double npu = 0, gpu = 0;
            if (std::sscanf(item.c_str(), "%d,%d,%lf,%lf", &k, &n, &npu, &gpu) == 4 &&
                k == K && n == N) {
                *npu_ms = npu;
                *gpu_ms = gpu;
                return true;
            }
        }
        return false;
    }
    for (const auto & e : g_xdna_cost_table) {
        if (e.K == K && e.N == N) {
            *npu_ms = e.npu_ms;
            *gpu_ms = e.gpu_ms;
            return true;
        }
    }
    return false;
}

// The cost driven decision: offload only when the NPU is the faster device
// for that shape, or when the op is so small that the transfer cannot
// dominate it (the transfer of an activation of M x K bytes is about
// M * K / 2.6e9 s on the measured 2.63 GB/s of the NoC).
static bool ggml_xdna_cost_route(int M, int K, int N) {
    double npu_ms = 0, gpu_ms = 0;
    if (!ggml_xdna_cost_lookup(M, K, N, &npu_ms, &gpu_ms)) {
        // an unknown shape is not offloaded: the cost of the NPU path is
        // unknown, and the phase 62 measurements show that the NPU loses by
        // more than an order of magnitude on every large GEMM
        ggml_xdna_log("[XDNA-COST] M=%d K=%d N=%d: no measurement, not routed\n",
                      M, K, N);
        return false;
    }
    const bool offload = npu_ms < gpu_ms;
    ggml_xdna_log("[XDNA-COST] M=%d K=%d N=%d: npu=%.2f ms gpu=%.2f ms -> %s\n",
                  M, K, N, npu_ms, gpu_ms, offload ? "route" : "keep on the GPU");
    return offload;
}

static const char * ggml_xdna_gemm_operation(const struct ggml_tensor * src0);
static bool is_mul_mat_supported(const struct ggml_tensor* op) {
    if (ggml_xdna_session_is_fatal()) {
        return false;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor* src0 = op->src[0];
    const struct ggml_tensor* src1 = op->src[1];

    if (!src0 || !src1) {
        return false;
    }

    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_I8) {
        ggml_xdna_log("[XDNA] Unsupported src0 type=%d\n", (int)src0->type);
        return false;
    }
    if (src1->type != GGML_TYPE_Q8_0 && src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_I8) {
        ggml_xdna_log("[XDNA] Unsupported src1 type=%d\n", (int)src1->type);
        return false;
    }

    int K = src0->ne[0];
    if (K % 32 != 0) {
        return false;
    }

    int M = src1->ne[1];
    int N = src0->ne[1];

    // Phase 37: environment-gated Qwen PP routing.
    const char * qwen_pp = std::getenv("GGML_XDNA_QWEN_PP");
    const char * route_ops = std::getenv("XDNA_ROUTE_OPS");
    bool route_enabled = (qwen_pp && *qwen_pp) || (route_ops && *route_ops);
    ggml_xdna_log("[XDNA-DEBUG] GGML_XDNA_QWEN_PP='%s' XDNA_ROUTE_OPS='%s'\n",
                  qwen_pp ? qwen_pp : "(null)", route_ops ? route_ops : "(null)");
    if (route_enabled) {
        bool qwen_allowed = is_qwen_pp_shape_allowed(M, K, N);
        ggml_xdna_log("[XDNA-PP] M=%d K=%d N=%d allowed=%d\n", M, K, N, qwen_allowed);
        if (!qwen_allowed) {
            ggml_xdna_log("[XDNA-PP] blocked op M=%d K=%d N=%d (not in Phase 37 allowlist)\n", M, K, N);
            return false;
        }
        // Phase 63: the cost driven policy replaces the fixed allowlist when
        // it is switched on.  It decides from the measured device times of
        // the shape, so it follows the kernel and the device instead of a
        // shape list that was measured once.
        const char * cost_route = std::getenv("XDNA_COST_ROUTE");
        if (cost_route && *cost_route && strcmp(cost_route, "0") != 0) {
            if (!ggml_xdna_cost_route(M, K, N)) {
                ggml_xdna_log("[XDNA-COST] op M=%d K=%d N=%d stays on the GPU\n", M, K, N);
                return false;
            }
        }
        if (route_ops && *route_ops) {
            const char * opname = ggml_xdna_gemm_operation(src0);
            char buf[256];
            snprintf(buf, sizeof(buf), "%s", route_ops);
            bool ok = false;
            char * tok = strtok(buf, ", ");
            while (tok) {
                if (strcasecmp(tok, opname) == 0 || strcasecmp(tok, "all") == 0) { ok = true; break; }
                tok = strtok(nullptr, ", ");
            }
            if (!ok) {
                ggml_xdna_log("[XDNA-ROUTE] skipping op=%s (not in XDNA_ROUTE_OPS=%s)\n", opname, route_ops);
                return false;
            }
        }
        ggml_xdna_log("[XDNA-PP] routing op M=%d K=%d N=%d to XDNA\n", M, K, N);
    } else if (is_phase47b_gated_shape(K, N)) {
        ggml_xdna_log("[XDNA-PP] blocked op M=%d K=%d N=%d (set GGML_XDNA_QWEN_PP=1 to opt in)\n", M, K, N);
        return false;
    }

    // Production whitelist: Q8_0/I8 M>=64, K in {1024,2048,4096}, M in {512,448}, N in production set
    // This is the Phase 18 capability model — no arbitrary shape compilation
    if (M < 64) {
        ggml_xdna_log("[XDNA] Unsupported M=%d (need >= 64)\n", M);
        return false;
    }

    // K must be a multiple of 32 (Q8_0 block size).
    // The engine always runs with k_tile=32; host-side chunking handles full K.
    // Any K divisible by 32 is valid for the scaled Q8_0 path.
    if (K % 32 != 0) {
        ggml_xdna_log("[XDNA] Unsupported K=%d (must be multiple of 32)\n", K);
        return false;
    }

    // For Q8_0 weight, M must be tile-aligned (>= 64, engine does M-tile chunking)
    if (src0->type == GGML_TYPE_Q8_0) {
        if (M < 64) {
            ggml_xdna_log("[XDNA] Unsupported M=%d for Q8_0 (need >= 64)\n", M);
            return false;
        }
        // N must be a validated production value
        if (N != 512 && N != 768 && N != 1024 && N != 1536 && N != 2048 &&
            N != 2304 && N != 3072 && N != 3584 && N != 4096 &&
            N != 4608 && N != 5120 && N != 6144) {
            ggml_xdna_log("[XDNA] Unsupported N=%d for Q8_0\n", N);
            return false;
        }
    }

    // Verify key generation produces a valid (non-zero) key
    XDNAKernelKey key = tensor_to_kernel_key(op);
    if (key.m_tile == 0) {
        ggml_xdna_log("[XDNA] Invalid kernel key M=%d K=%d N=%d\n", M, K, N);
        return false;
    }

    // Check if kernel registry has cached artifacts for this shape
    XDNAKernelRegistry& registry = XDNAKernelRegistry::instance();
    if (!registry.is_shape_whitelisted(key)) {
        ggml_xdna_log("[XDNA] No cached kernel for M=%d K=%d N=%d\n", M, K, N);
        return false;
    }

    // supports_op must stay a pure function of the shape: opening the HW
    // engine here makes routing depend on transient device state (later
    // queries fail once contexts are exhausted, so trailing layers silently
    // fall back). The compute path opens on demand and aborts loudly.
    return true;
}

// Quantize F32 [M,K] to int8 for XDNA kernel with per-32-K-block scales.
static void quantize_f32_to_i8_blocked(const float* src, int M, int K,
                                       std::vector<int8_t>& out_i8,
                                       std::vector<float>& out_scale) {
    const int nb = K / 32;
    out_i8.resize((size_t)K * M);
    out_scale.resize((size_t)M * nb);
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            const float* blk = src + (size_t)m * K + (size_t)b * 32;
            float max_abs = 0;
            for (int k = 0; k < 32; k++) max_abs = std::max(max_abs, std::fabs(blk[k]));
            float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            out_scale[(size_t)m * nb + b] = scale;
            for (int k = 0; k < 32; k++) {
                float v = blk[k] / scale;
                int q = (int)std::lroundf(v);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out_i8[(size_t)m * K + (b * 32 + k)] = (int8_t)q;
            }
        }
    }
}

// Phase 62: quantize F32 [M,K] to int8 with one scale per `block` elements
// along K.  The c256 engine needs the activation scale to cover a whole K
// chunk, so that the kernel can apply the scale once per chunk.
static void quantize_f32_to_i8_chunked(const float* src, int M, int K, int block,
                                       std::vector<int8_t>& out_i8,
                                       std::vector<float>& out_scale) {
    const int nb = K / block;
    out_i8.resize((size_t)K * M);
    out_scale.resize((size_t)M * nb);
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            const float* blk = src + (size_t)m * K + (size_t)b * block;
            float max_abs = 0;
            for (int k = 0; k < block; k++) max_abs = std::max(max_abs, std::fabs(blk[k]));
            float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            out_scale[(size_t)m * nb + b] = scale;
            for (int k = 0; k < block; k++) {
                float v = blk[k] / scale;
                int q = (int)std::lroundf(v);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out_i8[(size_t)m * K + (b * block + k)] = (int8_t)q;
            }
        }
    }
}

// Production Q8_0 MUL_MAT on XDNA: dequantize Q8_0 weights/inputs to int8 with
// per-32-K-block scales. Engine is created with K=32 (tile size), so each
// XRT submit processes exactly one Q8_0 block. Host loops over K-blocks,
// applies per-block scales, and accumulates.
static void execute_mul_mat_q8_scaled(XDNACompiledKernel* compiled_kernel,
                                      const struct ggml_tensor* src0,
                                      const struct ggml_tensor* src1,
                                      struct ggml_tensor* dst) {
    if (!compiled_kernel || !compiled_kernel->engine) return;

    ggml_xdna_persistent_engine* engine = compiled_kernel->engine;
    int K = src0->ne[0];
    int N = src0->ne[1];
    int M = src1->ne[1];
    int nb = K / 32;

    if (K % 32 != 0) {
        fprintf(stderr, "[XDNA] scaled path requires K divisible by 32 (K=%d)\n", K);
        return;
    }

    if (engine->k != 32) {
        fprintf(stderr, "[XDNA] scaled path requires engine->k=32, got %d\n", engine->k);
        return;
    }

    bool p38c = std::getenv("XDNA_P38C") != nullptr;
    double t0 = p38c ? ggml_xdna_now_ms() : 0.0;

    ggml_xdna_log("[XDNA-SCALED] Execute: M=%d K=%d N=%d nb=%d engine m=%d k=%d n=%d\n",
                  M, K, N, nb, engine->m, engine->k, engine->n);

    // Phase 38C: record tensor → XDNA buffer → engine BO identity
    if (p38c) {
        fprintf(stderr,
                "[XDNA-P38C-BO-ID] src0(type=%d,is_xdna=%d,data=%p,buf=%p) "
                "src1(type=%d,is_xdna=%d,data=%p,buf=%p) "
                "dst(data=%p,buf=%p) "
                "engine=%p inst_bo=%p weight_bo=%p input_bo[0]=%p input_bo[1]=%p output_bo[0]=%p output_bo[1]=%p\n",
                (int)src0->type, is_xdna_buffer(src0), (void*)src0->data, (void*)src0->buffer,
                (int)src1->type, is_xdna_buffer(src1), (void*)src1->data, (void*)src1->buffer,
                (void*)dst->data, (void*)dst->buffer,
                (void*)engine, (void*)engine->inst_bo, (void*)engine->weight_bo,
                (void*)engine->input_bo[0], (void*)engine->input_bo[1],
                (void*)engine->output_bo[0], (void*)engine->output_bo[1]);
    }

    // weight Q8_0 -> int8 + per-block scales (layout [K, N] = k*N + n)
    double t_w_deq0 = p38c ? ggml_xdna_now_ms() : 0.0;
    const uint8_t * w_q8_ptr = nullptr;
    if (is_xdna_buffer(src0)) {
        w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    }
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    std::vector<int8_t> weight_i8((size_t)K * N);
    std::vector<float> w_scale((size_t)N * nb);
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            size_t block_idx = (size_t)n * nb + b;
            const uint8_t* block = &w_q8_ptr[block_idx * 34];
            w_scale[(size_t)n * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)block);
            for (int kk = 0; kk < 32; kk++) {
                weight_i8[(size_t)(b * 32 + kk) * N + n] = (int8_t)block[2 + kk];
            }
        }
    }
    double t_w_deq1 = p38c ? ggml_xdna_now_ms() : 0.0;

    // input -> int8 + per-block scales (layout [M, K] = m*K + k)
    double t_in_deq0 = p38c ? ggml_xdna_now_ms() : 0.0;
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    const float * in_f32_ptr = nullptr;
    const uint8_t * in_q8_ptr = nullptr;
    if (src1->type == GGML_TYPE_F32) {
        in_f32_ptr = (const float *) get_tensor_data(src1);
        std::vector<float> in_f32_local;
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        quantize_f32_to_i8_blocked(in_f32_ptr, M, K, input_i8, i_scale);
    } else if (src1->type == GGML_TYPE_Q8_0) {
        in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
        std::vector<uint8_t> in_q8_local;
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        input_i8.resize((size_t)K * M);
        i_scale.resize((size_t)M * nb);
        for (int m = 0; m < M; m++) {
            for (int b = 0; b < nb; b++) {
                size_t block_idx = (size_t)m * nb + b;
                const uint8_t* block = &in_q8_ptr[block_idx * 34];
                i_scale[(size_t)m * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)block);
                for (int kk = 0; kk < 32; kk++) {
                    input_i8[(size_t)m * K + (b * 32 + kk)] = (int8_t)block[2 + kk];
                }
            }
        }
    } else {
        fprintf(stderr, "[XDNA] scaled path: unsupported input type %d\n", (int)src1->type);
        return;
    }
    double t_in_deq1 = p38c ? ggml_xdna_now_ms() : 0.0;

    // Output accumulator (float for scaled accumulation)
    std::vector<float> out_acc((size_t)M * N, 0.0f);

    // Temporary buffers for single chunk (K=32)
    std::vector<int8_t> input_chunk((size_t)M * 32);
    std::vector<int8_t> weight_chunk((size_t)32 * N);
    std::vector<int32_t> chunk_output((size_t)M * N);

    double t_xrt_total_wait = 0.0;
    double t_xrt_first_wait = 0.0;
    double t_first_pack0 = 0.0, t_first_pack1 = 0.0;

    // Loop over K-blocks (each 32 elements = 1 Q8_0 block)
    for (int b = 0; b < nb; b++) {
        double t_chunk_pack0 = p38c ? ggml_xdna_now_ms() : 0.0;
        // Extract input chunk [M, 32] from input_i8 [M, K] (strided)
        for (int m = 0; m < M; m++) {
            const int8_t* src = input_i8.data() + (size_t)m * K + b * 32;
            std::memcpy(input_chunk.data() + (size_t)m * 32, src, 32 * sizeof(int8_t));
        }

        // Extract weight chunk [32, N] from weight_i8 [K, N] (contiguous)
        const int8_t* wsrc = weight_i8.data() + (size_t)b * 32 * N;
        std::memcpy(weight_chunk.data(), wsrc, (size_t)32 * N * sizeof(int8_t));
        double t_chunk_pack1 = p38c ? ggml_xdna_now_ms() : 0.0;

        // Execute single chunk (K=32)
        double t_chunk_xrt0 = p38c ? ggml_xdna_now_ms() : 0.0;
        std::fill(chunk_output.begin(), chunk_output.end(), 0);
        if (!ggml_xdna_persistent_engine_execute_chunked(engine, input_chunk.data(), 32, weight_chunk.data(), chunk_output.data())) {
            fprintf(stderr, "[XDNA-SCALED] execute_chunked failed at block %d\n", b);
            return;
        }
        double t_chunk_xrt1 = p38c ? ggml_xdna_now_ms() : 0.0;

        if (p38c) {
            t_xrt_total_wait += (t_chunk_xrt1 - t_chunk_xrt0);
            if (b == 0) {
                t_xrt_first_wait = (t_chunk_xrt1 - t_chunk_xrt0);
                t_first_pack0 = t_chunk_pack0;
                t_first_pack1 = t_chunk_pack1;
            }
            // submit time is included in chunked timing (run_new)
        }

        // Apply per-block scale and accumulate to output
        // chunk_output is [M, N] row-major = scratch[m * N + n]
        // GGML output is [N, M] column-major = out[n * M + m]
        float w_scale_b = 0.0f; // Will be set per-n below
        for (int n = 0; n < N; n++) {
            w_scale_b = w_scale[(size_t)n * nb + b];
            for (int m = 0; m < M; m++) {
                float scale = i_scale[(size_t)m * nb + b] * w_scale_b;
                out_acc[(size_t)n * M + m] += (float)chunk_output[(size_t)m * N + n] * scale;
            }
        }
    }

    // Write final output - direct write for XDNA resident, otherwise use ggml API
    double t_out0 = p38c ? ggml_xdna_now_ms() : 0.0;
    size_t expected_bytes = out_acc.size() * sizeof(float);
    size_t tensor_bytes = ggml_nbytes(dst);
    if (expected_bytes != tensor_bytes) {
        fprintf(stderr, "[XDNA-ERROR] Output size mismatch: expected %zu bytes, tensor has %zu bytes\n", expected_bytes, tensor_bytes);
        fprintf(stderr, "[XDNA-ERROR] dst shape: ne=[%lld,%lld,%lld,%lld], nb=[%zu,%zu,%zu,%zu]\n",
                (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
    float * dst_ptr = (float *) get_tensor_data_mut(dst);
    if (dst_ptr) {
        for (size_t i = 0; i < out_acc.size(); i++) {
            dst_ptr[i] = out_acc[i];
        }
    } else {
        ggml_backend_tensor_set(dst, out_acc.data(), 0, expected_bytes);
    }
    double t_out1 = p38c ? ggml_xdna_now_ms() : 0.0;

    if (p38c) {
        double t_total = t0 > 0 ? (t_out1 - t0) : 0.0;
        fprintf(stderr,
                "[XDNA-P38C-SCALED] M=%d K=%d N=%d nb=%d "
                "w_deq=%.2fms in_deq=%.2fms pack_first_chunk=%.2fms "
                "xrt_total_wait=%.2fms xrt_first_wait=%.2fms "
                "out_copy=%.2fms total=%.2fms\n",
                M, K, N, nb,
                t_w_deq1 - t_w_deq0, t_in_deq1 - t_in_deq0,
                t_first_pack1 - t_first_pack0,
                t_xrt_total_wait / 1000.0, t_xrt_first_wait / 1000.0,
                t_out1 - t_out0, t_total / 1000.0);
    }

    ggml_xdna_log("[XDNA-SCALED] Complete: M=%d K=%d N=%d\n", M, K, N);
}

// Production dispatch: Q8_0/I8 weights use the scaled kernel (single XRT submit
// for full K, host applies Q8_0 per-block scales).
static void execute_mul_mat_kernel(XDNACompiledKernel* compiled_kernel,
                                   const struct ggml_tensor* src0,
                                   const struct ggml_tensor* src1,
                                   struct ggml_tensor* dst) {
    if (!compiled_kernel || !compiled_kernel->engine) {
        return;
    }

    if (src0->type == GGML_TYPE_Q8_0) {
        execute_mul_mat_q8_scaled(compiled_kernel, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_I8) {
        execute_mul_mat_q8_scaled(compiled_kernel, src0, src1, dst);
    } else {
        ggml_xdna_log("[XDNA] dispatch: unsupported weight type %d\n", (int)src0->type);
    }
}

static bool ggml_xdna_gemm_profile_enabled() {
    const char * env = std::getenv("XDNA_GEMM_PROFILE");
    return env && *env;
}

static int ggml_xdna_gemm_layer(const struct ggml_tensor * src0, const struct ggml_tensor * dst) {
    const char * name = src0 && src0->name[0] ? src0->name : dst ? dst->name : "";
    const char * p = strstr(name, "blk.");
    if (!p) {
        return -1;
    }
    int layer = -1;
    sscanf(p + 4, "%d", &layer);
    return layer;
}

static const char * ggml_xdna_gemm_operation(const struct ggml_tensor * src0) {
    const char * name = src0 ? src0->name : "";
    if (strstr(name, "ffn_gate")) {
        return "gate";
    }
    if (strstr(name, "ffn_up")) {
        return "up";
    }
    if (strstr(name, "ffn_down")) {
        return "down";
    }
    return "other";
}

static std::string ggml_xdna_tensor_shape(const struct ggml_tensor * tensor) {
    char shape[128];
    snprintf(shape, sizeof(shape), "[%lld,%lld,%lld,%lld]",
            (long long) tensor->ne[0], (long long) tensor->ne[1],
            (long long) tensor->ne[2], (long long) tensor->ne[3]);
    return shape;
}

static void ggml_xdna_gemm_record_metadata(
        ggml_xdna_pending_sub & submission,
        const struct ggml_tensor * node,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const XDNACompiledKernel * compiled_kernel,
        const XDNAKernelKey & key) {
    submission.layer = ggml_xdna_gemm_layer(src0, node);
    submission.operation = ggml_xdna_gemm_operation(src0);
    submission.engine_name = "q8-c4-mmul-v2";
    char kernel_key[128];
    snprintf(kernel_key, sizeof(kernel_key), "M%d_K%d_N%d_C%d_Q8SCALEV_MMUL", key.M, key.K, key.N, key.n_cols);
    submission.kernel_key = kernel_key;
    submission.kernel_source = "experiments/phase40/gen_q8_c4_mmul_v2.py";
    submission.xclbin = compiled_kernel->spec.xclbin_path;
    submission.variant = "q8-c4-mmul-v2/aie::mmul<4,8,8>/tile=32x64x256";
    submission.input_shape = ggml_xdna_tensor_shape(src1);
    submission.weight_shape = ggml_xdna_tensor_shape(src0);
    submission.output_shape = ggml_xdna_tensor_shape(node);
    submission.n_cols = key.n_cols;
    submission.m_tile = key.m_tile;
    submission.n_tile = key.n_tile;
    submission.k_tile = key.k_tile;
}

static void ggml_xdna_gemm_print(const ggml_xdna_pending_sub & submission) {
    if (!ggml_xdna_gemm_profile_enabled() || submission.xrt_trace.submissions == 0) {
        return;
    }
    const ggml_xdna_xrt_trace & trace = submission.xrt_trace;
    fprintf(stderr,
            "[XDNA-GEMM] layer=%d op=%s M=%d K=%d N=%d "
            "input=%s weight=%s output=%s engine=%s key=%s source=%s xclbin=%s variant=%s "
            "cols=%d m_tile=%d n_tile=%d k_tile=%d m_tiles=%d n_tiles=%d "
            "k_block_size=%d k_blocks_expected=%d k_blocks_actual=%d q8_scale_blocks=%d k_chunks=%d "
            "run_objects=%d submissions=%d starts=%d waits=%d buf=%d "
            "host_prep_ms=%.3f submit_ms=%.3f a_copy_ms=%.3f b_copy_ms=%.3f bo_sync_ms=%.3f "
            "run_ctor_ms=%.3f set_arg_ms=%.3f start_ms=%.3f queue_wait_ms=%.3f run_wait_ms=%.3f "
            "sync_from_ms=%.3f copyback_ms=%.3f total_ms=%.3f "
            "wait_state=%d return_code=%u "
            "submit_ns=%llu run_start_ns=%llu wait_start_ns=%llu wait_end_ns=%llu drain_ns=%llu "
            "inst_bo=%p input_bo=%p weight_bo=%p output_bo=%p bo_reuse=engine-persistent\n",
            submission.layer,
            submission.operation.c_str(),
            submission.M, submission.K, submission.N,
            submission.input_shape.c_str(), submission.weight_shape.c_str(), submission.output_shape.c_str(),
            submission.engine_name.c_str(), submission.kernel_key.c_str(), submission.kernel_source.c_str(),
            submission.xclbin.c_str(), submission.variant.c_str(),
            submission.n_cols, submission.m_tile, submission.n_tile, submission.k_tile,
            trace.m_tiles, trace.n_tiles, trace.k_block_size, trace.expected_k_blocks,
            trace.actual_k_blocks, trace.q8_scale_blocks, trace.k_chunks,
            trace.run_objects, trace.submissions, trace.starts, trace.waits, trace.buf_idx,
            trace.host_prep_ms, trace.submit_ms, trace.a_copy_ms, trace.b_copy_ms, trace.bo_sync_ms,
            trace.run_ctor_ms, trace.set_arg_ms, trace.run_start_ms, trace.queue_wait_ms, trace.run_wait_ms,
            trace.sync_from_ms, trace.copyback_ms, trace.total_ms, trace.wait_state, trace.return_code,
            (unsigned long long) trace.submit_start_ns, (unsigned long long) trace.run_start_ns,
            (unsigned long long) trace.wait_start_ns, (unsigned long long) trace.wait_end_ns,
            (unsigned long long) ggml_time_ns(),
            trace.inst_bo, trace.input_bo, trace.weight_bo, trace.output_bo);
}

static bool ggml_xdna_stats_enabled() {
    return std::getenv("GGML_SCHED_TRACE_STATS") != nullptr;
}

// Phase 48: aggregate NPU metrics, printed as one machine-readable line per
// XDNA graph compute. The timings come from the per-GEMM xrt trace that the
// drain path already fills in.
struct ggml_xdna_stats {
    bool enabled = false;
    int  n_gemm = 0;
    int  n_gate = 0;
    int  n_up = 0;
    int  n_down = 0;
    int  n_other = 0;
    int  M = 0, K = 0, N = 0;
    bool shape_uniform = true;
    double t_host_prep = 0;
    double t_submit = 0;
    double t_run_wait = 0;
    double t_sync_from = 0;
    double t_copyback = 0;
    double t_total = 0;
    double t_gemm = 0;
};

static ggml_xdna_stats g_xdna_stats;

static void ggml_xdna_stats_reset() {
    g_xdna_stats = ggml_xdna_stats();
    g_xdna_stats.enabled = ggml_xdna_stats_enabled();
}

static void ggml_xdna_stats_account(const ggml_xdna_pending_sub & s) {
    g_xdna_stats.n_gemm++;
    if      (s.operation == "gate") g_xdna_stats.n_gate++;
    else if (s.operation == "up")   g_xdna_stats.n_up++;
    else if (s.operation == "down") g_xdna_stats.n_down++;
    else                            g_xdna_stats.n_other++;
    if (g_xdna_stats.M == 0) {
        g_xdna_stats.M = s.M;
        g_xdna_stats.K = s.K;
        g_xdna_stats.N = s.N;
    } else if (g_xdna_stats.M != s.M) {
        g_xdna_stats.shape_uniform = false;
    }
    const ggml_xdna_xrt_trace & t = s.xrt_trace;
    g_xdna_stats.t_host_prep += t.host_prep_ms;
    g_xdna_stats.t_submit    += t.submit_ms;
    g_xdna_stats.t_run_wait  += t.run_wait_ms;
    g_xdna_stats.t_sync_from += t.sync_from_ms;
    g_xdna_stats.t_copyback  += t.copyback_ms;
    g_xdna_stats.t_total     += t.total_ms;
    if (t.wait_end_ns > t.submit_start_ns) {
        g_xdna_stats.t_gemm += (t.wait_end_ns - t.submit_start_ns) / 1e6;
    }
}

static bool ggml_xdna_drain_pending(ggml_xdna_backend_context * ctx) {
    const uint64_t drain_start_ns = ggml_time_ns();
    std::unique_lock<std::mutex> lock(ctx->pending_mutex);
    std::vector<ggml_xdna_pending_sub> subs;
    subs.swap(ctx->pending);
    if (subs.empty() && std::getenv("XDNA_TRACE_DRAIN") == nullptr) {
        return true;
    }
    fprintf(stderr, "[XDNA-TRACE] drain submissions=%zu submitted_seq=%llu completed_seq=%llu\n",
            subs.size(), (unsigned long long) ctx->submitted_seq.load(), (unsigned long long) ctx->completed_seq.load());
    for (size_t i = 0; i < subs.size(); i++) {
        auto & s = subs[i];
        bool ok = true;
        if (s.q8_scale_c4_engine) {
            std::vector<float> f((size_t) s.M * s.N);
            ok = ggml_xdna_q8_scale_c4_wait(s.q8_scale_c4_engine, s.run, f.data(), s.buf_idx);
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (size_t j = 0; j < f.size(); j++) dp[j] = f[j];
                } else {
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
            }
        } else if (s.q8_c4_mmul_v2_engine) {
            if (!s.requant_sa.empty()) {
                // Phase 54 pure INT8: device returns int32, host applies sa*sb.
                std::vector<int32_t> out((size_t) s.M * s.N);
                ok = ggml_xdna_q8_c4_mmul_v2_wait(s.q8_c4_mmul_v2_engine, s.run, (float *) out.data(), s.buf_idx, &s.xrt_trace);
                if (ok) {
                    if (!s.verify_ref.empty()) {
                        int64_t maxdiff = 0;
                        size_t nbad = 0;
                        for (size_t i = 0; i < out.size(); i++) {
                            int64_t d = (int64_t) out[i] - (int64_t) s.verify_ref[i];
                            if (d < 0) d = -d;
                            if (d > maxdiff) maxdiff = d;
                            if (d != 0) nbad++;
                        }
                        fprintf(stderr, "[XDNA-VERIFY] M=%d N=%d K=%d maxdiff=%lld bad=%zu/%zu\n",
                                s.M, s.N, s.K, (long long) maxdiff, nbad, out.size());
                    }
                    if (!s.verify_fref.empty()) {
                        const std::vector<float> & sa = s.requant_sa;
                        const std::vector<float> & sb = s.requant_sb;
                        double fmax = 0.0, fref_max = 0.0;
                        int nfb = 0;
                        for (int m = 0; m < s.verify_rows; m++) {
                            for (int n = 0; n < s.N; n++) {
                                double got = (double) out[(size_t) m * s.N + n] * sa[m] * sb[n];
                                double ref = s.verify_fref[(size_t) m * s.N + n];
                                fref_max = std::max(fref_max, std::fabs(ref));
                                double d = std::fabs(got - ref);
                                fmax = std::max(fmax, d);
                                if (d > 1e-3 * (1.0 + std::fabs(ref))) nfb++;
                            }
                        }
                        fprintf(stderr, "[XDNA-VERIFY-FLOAT] M=%d N=%d K=%d maxdiff=%.6g maxref=%.6g bad=%d/%d amax=%.6g wmax=%.6g smax=%.6g\n",
                                s.M, s.N, s.K, fmax, fref_max, nfb, s.verify_rows * s.N,
                                s.verify_amax, s.verify_wmax, s.verify_smax);
                    }
                    float * dp = (float *) get_tensor_data_mut(s.dst);
                    const std::vector<float> & sa = s.requant_sa;
                    const std::vector<float> & sb = s.requant_sb;
                    if (dp) {
                        for (int m = 0; m < s.M; m++)
                            for (int n = 0; n < s.N; n++)
                                dp[(size_t) m * s.N + n] = (float) out[(size_t) m * s.N + n] * sa[m] * sb[n];
                    } else {
                        std::vector<float> f((size_t) s.M * s.N);
                        for (int m = 0; m < s.M; m++)
                            for (int n = 0; n < s.N; n++)
                                f[(size_t) m * s.N + n] = (float) out[(size_t) m * s.N + n] * sa[m] * sb[n];
                        ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                    }
                }
            } else {
            std::vector<float> f((size_t) s.M * s.N);
            ok = ggml_xdna_q8_c4_mmul_v2_wait(s.q8_c4_mmul_v2_engine, s.run, f.data(), s.buf_idx, &s.xrt_trace);
            if (ok) {
                if (s.q8_c4_mmul_v2_engine->s12) {
                    // s12: f holds linear [i][j][4][8] tiles in (mt,col,nt) group
                    // order; scatter into row-major dst.
                    const int m = 32, n = 64;
                    const int MA = m / 4, NA = n / 8;
                    const int n_cols = s.q8_c4_mmul_v2_engine->n_cols;
                    const int n_mt = s.M / m;
                    const int n_nt_c = (s.N / n_cols) / n;
                    std::vector<float> g((size_t) s.M * s.N);
                    for (int mt = 0; mt < n_mt; mt++)
                        for (int col = 0; col < n_cols; col++)
                            for (int nt = 0; nt < n_nt_c; nt++) {
                                const float * tile = &f[(size_t)(mt * n_cols * n_nt_c + col * n_nt_c + nt) * (m * n)];
                                int ntb = (col * n_nt_c + nt) * n;
                                for (int i = 0; i < MA; i++)
                                    for (int j = 0; j < NA; j++)
                                        for (int rr = 0; rr < 4; rr++) {
                                            const float * src = tile + (i * NA + j) * 32 + rr * 8;
                                            float * dst_row = &g[(size_t)(mt * m + i * 4 + rr) * s.N + ntb + j * 8];
                                            for (int ll = 0; ll < 8; ll++) dst_row[ll] = src[ll];
                                        }
                            }
                    float * dp = (float *) get_tensor_data_mut(s.dst);
                    if (dp) {
                        for (size_t j = 0; j < g.size(); j++) dp[j] = g[j];
                    } else {
                        ggml_backend_tensor_set(s.dst, g.data(), 0, g.size() * sizeof(float));
                    }
                } else {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (size_t j = 0; j < f.size(); j++) dp[j] = f[j];
                } else {
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
                }
                ggml_xdna_gemm_print(s);
            }
            }
        } else if (s.requant_engine) {
            std::vector<int32_t> out((size_t) s.M * s.N);
            ok = ggml_xdna_persistent_engine_wait(s.requant_engine, s.run, out.data());
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (int m = 0; m < s.M; m++)
                        for (int n = 0; n < s.N; n++)
                            dp[(size_t)n * s.M + m] = (float) out[(size_t)m * s.N + n] * s.requant_sa[m] * s.requant_sb[n];
                } else {
                    std::vector<float> f((size_t) s.M * s.N);
                    for (int m = 0; m < s.M; m++)
                        for (int n = 0; n < s.N; n++)
                            f[(size_t)n * s.M + m] = (float) out[(size_t)m * s.N + n] * s.requant_sa[m] * s.requant_sb[n];
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
            }
        } else if (s.q8_c4_engine) {
            std::vector<int32_t> out((size_t) s.M * s.N);
            ok = ggml_xdna_q8_c4_wait(s.q8_c4_engine, s.run, out.data(), s.buf_idx);
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (size_t j = 0; j < out.size(); j++) dp[j] = (float) out[j];
                } else {
                    std::vector<float> f(out.size());
                    for (size_t j = 0; j < out.size(); j++) f[j] = (float) out[j];
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
            }
        } else if (s.q8_engine) {
            std::vector<float> f((size_t) s.M * s.N);
            ok = ggml_xdna_q8_scale_wait(s.q8_engine, s.run, f.data(), s.buf_idx);
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (size_t j = 0; j < f.size(); j++) dp[j] = f[j];
                } else {
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
            }
        } else if (s.wa_engine) {
            std::vector<int32_t> out((size_t) s.M * s.N);
            ok = ggml_xdna_wa_wait(s.wa_engine, s.run, out.data());
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (!dp) {
                    std::vector<float> f(out.size());
                    for (size_t j = 0; j < out.size(); j++) f[j] = (float) out[j] * s.wa_scale;
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                } else {
                    for (size_t j = 0; j < out.size(); j++) dp[j] = (float) out[j] * s.wa_scale;
                }
            }
        } else {
            std::vector<int32_t> out((size_t) s.M * s.N);
            ok = ggml_xdna_persistent_engine_wait(s.engine, s.run, out.data());
            if (ok) {
                float * dp = (float *) get_tensor_data_mut(s.dst);
                if (dp) {
                    for (size_t j = 0; j < out.size(); j++) dp[j] = (float) out[j];
                } else {
                    std::vector<float> f(out.size());
                    for (size_t j = 0; j < out.size(); j++) f[j] = (float) out[j];
                    ggml_backend_tensor_set(s.dst, f.data(), 0, f.size() * sizeof(float));
                }
            }
        }
        if (!ok) {
            for (size_t j = i + 1; j < subs.size(); j++) {
                ggml_xdna_cleanup_run(subs[j].run);
            }
            return false;
        }
        if (g_xdna_stats.enabled) {
            ggml_xdna_stats_account(s);
        }
        ctx->completed_seq.fetch_add(1);
    }
    fprintf(stderr, "[XDNA-TRACE] drain_ms=%.3f\n", (ggml_time_ns() - drain_start_ns) / 1e6);
    return true;
}

// Phase 20G step 8: submit an I8 full-K MUL_MAT with xrt::run.start() and no
// wait. Weight src0 = I8 [K,N]; input src1 = I8 [K,M] (both row-major int8, so
// the contiguous bytes already are the kernel's [K][N] and [M][K] operands).
static bool ggml_xdna_submit_i8_fullk(ggml_xdna_backend_context * ctx,
                                      XDNACompiledKernel * compiled_kernel,
                                      const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1,
                                      struct ggml_tensor * dst) {
    if (!compiled_kernel || !compiled_kernel->engine) return false;
    ggml_xdna_persistent_engine * engine = compiled_kernel->engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];

    if (!ctx->persistent_mode) {
        int pending_count = 0;
        {
            std::lock_guard<std::mutex> lock(ctx->pending_mutex);
            for (auto & s : ctx->pending) if (s.engine == engine) { pending_count++; }
        }
        if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;
    }

    // ggml stores a 2D [K,N] tensor column-major (element [k,n] at k + n*K), but
    // the kernel wants B row-major [K][N] (k*N + n), so transpose the weight.
    // The input [K,M] is already [M][K] row-major (k + m*K == m*K + k).
    const int8_t * w_raw_ptr = (const int8_t *) get_tensor_data(src0);
    const int8_t * in_ptr = (const int8_t *) get_tensor_data(src1);
    std::vector<int8_t> w_raw_local, in_local;
    if (!w_raw_ptr) {
        w_raw_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_raw_local.data(), 0, w_raw_local.size());
        w_raw_ptr = w_raw_local.data();
    }
    if (!in_ptr) {
        in_local.resize(ggml_nbytes(src1));
        ggml_backend_tensor_get(src1, in_local.data(), 0, in_local.size());
        in_ptr = in_local.data();
    }
    std::vector<int8_t> w((size_t) K * N);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            w[(size_t) k * N + n] = w_raw_ptr[(size_t) k + (size_t) n * K];

    xrt::run * run = nullptr;
    int used_buf = engine->current_buffer;
    if (!ggml_xdna_persistent_engine_submit(engine, in_ptr, w.data(), nullptr, &run)) {
        return false;
    }

    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.engine = engine;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 53: int8 requant engine. Reuses the pure int8 (Phase 36) persistent
// kernel, but quantizes Q8_0 weights to per-column int8 and activations to
// per-row int8, applying a single sa*sb scale per output on the host. This
// removes the per-32-block float scale path from the device kernel.
struct ggml_xdna_requant_weight {
    int K = 0, N = 0;
    std::vector<int8_t> B;   // [K, N] row-major
    std::vector<float> sb;    // [N] per-column scale
    // Pre-packed tile-major B for n_cols=4 (saves per-GEMM repack)
    std::vector<int8_t> Bp;  // tile-major packed
    int bp_ncols = 0;
    int bp_ready = 0;
};

static std::mutex g_requant_engine_mutex;
static std::unordered_map<uint64_t, ggml_xdna_persistent_engine *> g_requant_engines;
static std::mutex g_requant_wcache_mutex;
static std::unordered_map<uint64_t, ggml_xdna_requant_weight> g_requant_wcache;

// Phase 61: cache of the tile-major packed weights for the scale preserving
// path. The Q8_0 dequantisation and the tile packing of a weight tensor are
// identical for every GEMM that uses it, so they are done once per tensor and
// per tile shape instead of once per GEMM. The cache is bounded so a large
// model cannot exhaust the host memory.
struct ggml_xdna_c4_weight {
    int K = 0, N = 0;
    int m = 0, n = 0, kc = 0, n_cols = 0, nblk = 0;
    const void * src = nullptr;
    char name[64] = {0};
    std::vector<int8_t> Bp;  // tile-major packed B, all columns
};
static std::mutex                g_c4_wcache_mutex;
static std::unordered_map<uint64_t, ggml_xdna_c4_weight> g_c4_wcache;
static size_t                    g_c4_wcache_bytes = 0;

// Phase 61: a proper hash of the cache key. The identity of a weight tensor
// is its tensor object and its name: src0->data, src0->view_offs and
// get_tensor_data() are identical for every weight of an mmap'ed model, so a
// pointer based key makes the cache return the wrong tensor's weights.
static uint64_t ggml_xdna_c4_wkey(const void * obj, const char * name,
                                  int m, int n, int kc, int n_cols, int nblk) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint64_t vals[5] = {
        (uint64_t)(uintptr_t) obj, (uint64_t) m, (uint64_t) n,
        (uint64_t) kc, (uint64_t) n_cols,
    };
    for (int i = 0; i < 5; i++) {
        h ^= vals[i];
        h *= 0x100000001b3ull;
        h ^= h >> 29;
    }
    for (const char * c = name; c && *c; c++) {
        h ^= (uint64_t)(unsigned char) *c;
        h *= 0x100000001b3ull;
    }
    h ^= (uint64_t) nblk;
    h *= 0x100000001b3ull;
    return h;
}

static size_t ggml_xdna_c4_wcache_limit() {
    const char * env = getenv("XDNA_WCACHE_MB");
    if (env && *env) {
        const long mb = atol(env);
        if (mb > 0) return (size_t) mb << 20;
    }
    return (size_t) 128 << 20;
}

static const char * ggml_xdna_requant_xclbin(int K, int N) {
    // Return path to cached pure-int8 xclbin for this shape.
    static char buf[512];
    // Known cached / compiled locations
    if (K == 1024 && N == 3584) {
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C1/design.xclbin");
        return buf;
    }
    if (K == 3584 && N == 1024) {
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/phase53/kernels/M512_K3584_N1024_C1/design.xclbin");
        return buf;
    }
    if (K == 1024 && N == 4096) {
        // try kernel_cache basic
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N4096_C1/design.xclbin");
        return buf;
    }
    return nullptr;
}

static const char * ggml_xdna_requant_insts(int K, int N) {
    static char buf[512];
    if (K == 1024 && N == 3584) {
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N3584_C1/design.insts.bin");
        return buf;
    }
    if (K == 3584 && N == 1024) {
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/phase53/kernels/M512_K3584_N1024_C1/design.insts.bin");
        return buf;
    }
    if (K == 1024 && N == 4096) {
        snprintf(buf, sizeof(buf), "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M512_K1024_N4096_C1/design.insts.bin");
        return buf;
    }
    return nullptr;
}

static ggml_xdna_persistent_engine * ggml_xdna_requant_engine_get(int M, int K, int N) {
    uint64_t key = ((uint64_t) M << 42) ^ ((uint64_t) K << 21) ^ (uint64_t) N;
    {
        std::lock_guard<std::mutex> lock(g_requant_engine_mutex);
        auto it = g_requant_engines.find(key);
        if (it != g_requant_engines.end()) return it->second;
    }
    const char * xc = ggml_xdna_requant_xclbin(K, N);
    const char * in = ggml_xdna_requant_insts(K, N);
    if (!xc || !in) return nullptr;
    struct stat st;
    if (stat(xc, &st) != 0 || stat(in, &st) != 0) return nullptr;

    // engine chunk size = 64
    ggml_xdna_persistent_engine * e = ggml_xdna_persistent_engine_create(M, 64, N, false);
    if (!e) return nullptr;
    bool ok = ggml_xdna_persistent_engine_init_xrt(e, xc, in);
    if (!ok) { ggml_xdna_persistent_engine_free(e); return nullptr; }
    {
        std::lock_guard<std::mutex> lock(g_requant_engine_mutex);
        g_requant_engines[key] = e;
    }
    return e;
}

static const ggml_xdna_requant_weight * ggml_xdna_requant_pack_weight(const struct ggml_tensor * src0,
                                                                      int m_tile, int n_tile, int k_chunk,
                                                                      int n_cols_in, int mmul) {
    int K = src0->ne[0], N = src0->ne[1];
    // Phase 60: the key includes the tile shape so that packed weights of one
    // engine profile are never reused by another.
    uint64_t key = ((uint64_t)(uintptr_t) src0->data)
                 ^ (((uint64_t) m_tile << 44) | ((uint64_t) n_tile << 32) |
                    ((uint64_t) k_chunk << 20) | ((uint64_t) n_cols_in << 12) | (uint64_t) mmul);
    {
        std::lock_guard<std::mutex> lock(g_requant_wcache_mutex);
        auto it = g_requant_wcache.find(key);
        if (it != g_requant_wcache.end() && it->second.K == K && it->second.N == N) {
            return &it->second;
        }
    }
    // fetch Q8_0 bytes
    const uint8_t * wq = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> wlocal;
    if (!wq) {
        wlocal.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, wlocal.data(), 0, wlocal.size());
        wq = wlocal.data();
    }
    int nb = K / 32;
    // dequant to float per element, then requant per column
    std::vector<float> wfloat((size_t) K * N);
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &wq[((size_t) n * nb + b) * 34];
            float sc = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                wfloat[(size_t)(b * 32 + kk) * N + n] = sc * (int8_t) block[2 + kk];
        }
    // per-column scale
    std::vector<float> sb(N, 0.0f);
    for (int n = 0; n < N; n++) {
        float amax = 1e-9f;
        for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(wfloat[(size_t) k * N + n]));
        sb[n] = amax / 127.0f;
    }
    ggml_xdna_requant_weight w;
    w.K = K; w.N = N;
    w.B.assign((size_t) K * N, 0);
    w.sb = sb;
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            float v = wfloat[(size_t) k * N + n] / sb[n];
            int q = (int) std::lroundf(v);
            q = std::max(-127, std::min(127, q));
            w.B[(size_t) k * N + n] = (int8_t) q;
        }
    // Pre-pack tile-major B for n_cols=4
    {
        const int n_cols = n_cols_in;
        const int kc = k_chunk, nblk = kc / 32;
        if (N % (n_cols * n_tile) == 0 && K % kc == 0) {
            int n_nt_c = (N / n_cols) / n_tile;
            int n_kc = K / kc;
            int b_chunk = kc * n_tile;
            w.Bp.assign((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
            for (int col = 0; col < n_cols; col++)
                for (int nt = 0; nt < n_nt_c; nt++)
                    for (int cc = 0; cc < n_kc; cc++) {
                        int8_t * t = &w.Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                        for (int jn = 0; jn < n_tile / 8; jn++)
                            for (int bb = 0; bb < nblk; bb++)
                                for (int kk = 0; kk < 4; kk++)
                                    for (int rr = 0; rr < 8; rr++)
                                        for (int cc_col = 0; cc_col < 8; cc_col++) {
                                            size_t flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                            size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n_tile + jn * 8 + cc_col;
                                            t[flat_idx] = w.B[src_idx];
                                        }
                    }
            w.bp_ncols = n_cols;
            w.bp_ready = 1;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_requant_wcache_mutex);
        g_requant_wcache[key] = w;
        return &g_requant_wcache[key];
    }
}

static bool ggml_xdna_submit_requant(ggml_xdna_backend_context * ctx,
                                     XDNACompiledKernel * compiled_kernel,
                                     const struct ggml_tensor * src0,
                                     const struct ggml_tensor * src1,
                                     struct ggml_tensor * dst) {
    if (!compiled_kernel || !compiled_kernel->q8_c4_mmul_v2_engine) return false;
    ggml_xdna_q8_c4_mmul_v2_engine * e = compiled_kernel->q8_c4_mmul_v2_engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    // Phase 60: the tile shape is a property of the compiled engine.
    const int m = e->m, n = e->n, kc = e->kt, nblk = e->kt / 32, mmul = e->mmul;
    if (K % kc != 0 || M % m != 0 || N % n != 0) return false;
    int nb = K / 32;
    int n_kc = K / kc;
    int n_cols = e->n_cols;
    int n_mt = M / m;
    int n_nt_c = (N / n_cols) / n;
    int a_chunk = m * kc + m * nblk * 4;
    int b_chunk = kc * n + nblk * n * 4;

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_c4_mmul_v2_engine == e) { pending_count++; }
    }
    if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;

    // weight Q8_0 -> per-column requantized int8 + constant scale per column
    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    std::vector<float> wfloat((size_t) K * N);
    for (int nn2 = 0; nn2 < N; nn2++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
            float sc = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                wfloat[(size_t)(b * 32 + kk) * N + nn2] = sc * (int8_t) block[2 + kk];
        }
    std::vector<float> sb(N, 0.0f);
    for (int nn2 = 0; nn2 < N; nn2++) {
        float amax = 1e-9f;
        for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(wfloat[(size_t) k * N + nn2]));
        sb[nn2] = amax / 127.0f;
    }
    std::vector<int8_t> weight_i8((size_t) K * N);
    for (int nn2 = 0; nn2 < N; nn2++)
        for (int k = 0; k < K; k++) {
            float v = wfloat[(size_t) k * N + nn2] / sb[nn2];
            int q = (int) std::lroundf(v);
            q = std::max(-127, std::min(127, q));
            weight_i8[(size_t) k * N + nn2] = (int8_t) q;
        }
    std::vector<float> w_scale((size_t) nb * N);
    for (int b = 0; b < nb; b++)
        for (int nn2 = 0; nn2 < N; nn2++)
            w_scale[(size_t) b * N + nn2] = sb[nn2];

    // activation -> per-row requantized int8 + constant scale per row
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    std::vector<float> a((size_t) M * K);
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize((size_t) M * K);
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        for (size_t i = 0; i < (size_t) M * K; i++) a[i] = in_f32_ptr[i];
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        int nbk = K / 32;
        for (int m = 0; m < M; m++)
            for (int b = 0; b < nbk; b++) {
                const uint8_t * block = &in_q8_ptr[((size_t) m * nbk + b) * 34];
                float sc = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                for (int kk = 0; kk < 32; kk++)
                    a[(size_t) m * K + b * 32 + kk] = sc * (int8_t) block[2 + kk];
            }
    } else {
        return false;
    }
    std::vector<float> sa(M, 0.0f);
    for (int m = 0; m < M; m++) {
        float amax = 1e-9f;
        for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(a[(size_t) m * K + k]));
        sa[m] = amax / 127.0f;
    }
    std::vector<int8_t> input_i8((size_t) M * K);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float v = a[(size_t) m * K + k] / sa[m];
            int q = (int) std::lroundf(v);
            q = std::max(-127, std::min(127, q));
            input_i8[(size_t) m * K + k] = (int8_t) q;
        }
    std::vector<float> i_scale((size_t) M * nb);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++)
            i_scale[(size_t) m * nb + b] = sa[m];

    // pack A (normal, s12=0)
    std::vector<int8_t> Ap((size_t) n_mt * n_kc * a_chunk, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            for (int i = 0; i < m; i++)
                for (int l = 0; l < kc; l++)
                    t[i * kc + l] = input_i8[(size_t)(mt * m + i) * K + cc * kc + l];
            float * ts = (float *)(t + m * kc);
            for (int i = 0; i < m; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = i_scale[(size_t)(mt * m + i) * nb + cc * nblk + bb];
        }
    // pack B (N-sliced across columns)
    std::vector<int8_t> Bp((size_t) n_cols * n_nt_c * n_kc * b_chunk, 0);
    for (int col = 0; col < n_cols; col++)
        for (int nt = 0; nt < n_nt_c; nt++)
            for (int cc = 0; cc < n_kc; cc++) {
                int8_t * t = &Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                for (int l = 0; l < kc; l++)
                    for (int j = 0; j < n; j++)
                        t[l * n + j] = weight_i8[(size_t)(cc * kc + l) * N + (col * n_nt_c + nt) * n + j];
                float * ts = (float *)(t + kc * n);
                for (int bb = 0; bb < nblk; bb++)
                    for (int j = 0; j < n; j++)
                        ts[bb * n + j] = w_scale[(size_t)(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + j];
            }

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    if (!ggml_xdna_q8_c4_mmul_v2_submit(e, Ap.data(), Bp.data(), &run)) return false;
    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_c4_mmul_v2_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    submission.n_cols = n_cols;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 54: submit a pure-INT8 full-K MUL_MAT (int8_requant_v2).
// Requantizes Q8_0 weights to per-column int8 (cached) and activations to
// per-row int8, packs int8-only tile-major buffers, runs the pure int8 kernel
// (no device-side scales), applies sa*sb on the host in the drain path.
static bool ggml_xdna_submit_int8_pure(ggml_xdna_backend_context * ctx,
                                       XDNACompiledKernel * compiled_kernel,
                                       const struct ggml_tensor * src0,
                                       const struct ggml_tensor * src1,
                                       struct ggml_tensor * dst) {
    if (!compiled_kernel || !compiled_kernel->q8_c4_mmul_v2_engine) return false;
    ggml_xdna_q8_c4_mmul_v2_engine * e = compiled_kernel->q8_c4_mmul_v2_engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    // Phase 60: the tile shape is a property of the compiled engine.
    const int m = e->m, n = e->n, kc = e->kt, nblk = e->kt / 32, mmul = e->mmul;
    if (K % kc != 0 || M % m != 0 || N % n != 0) return false;
    int nb = K / 32;
    int n_kc = K / kc;
    int n_cols = e->n_cols;
    int n_mt = M / m;
    int n_nt_c = (N / n_cols) / n;
    int a_chunk = m * kc;
    int b_chunk = kc * n;

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_c4_mmul_v2_engine == e) { pending_count++; }
    }
    if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;

    // weight: cached per-column requantization
    const ggml_xdna_requant_weight * w = ggml_xdna_requant_pack_weight(src0, m, n, kc, n_cols, 4);
    // w->B is [K,N] int8, w->sb is [N]
    const std::vector<int8_t> & weight_i8 = w->B;
    const std::vector<float> & sb = w->sb;

    // activation -> per-row int8
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    std::vector<float> a((size_t) M * K);
    if (src1->type == GGML_TYPE_F32) {
        {   // Phase 60: always read through the backend
            in_f32_local.resize((size_t) M * K);
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        for (size_t i = 0; i < (size_t) M * K; i++) a[i] = in_f32_ptr[i];
    } else if (src1->type == GGML_TYPE_Q8_0) {
        {   // Phase 60: always read through the backend
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        int nbk = K / 32;
        for (int mm = 0; mm < M; mm++)
            for (int b = 0; b < nbk; b++) {
                const uint8_t * block = &in_q8_ptr[((size_t) mm * nbk + b) * 34];
                float sc = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                for (int kk = 0; kk < 32; kk++)
                    a[(size_t) mm * K + b * 32 + kk] = sc * (int8_t) block[2 + kk];
            }
    } else {
        return false;
    }
    std::vector<float> sa(M, 0.0f);
    for (int mm = 0; mm < M; mm++) {
        float amax = 1e-9f;
        for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(a[(size_t) mm * K + k]));
        sa[mm] = amax / 127.0f;
    }
    std::vector<int8_t> input_i8((size_t) M * K);
    for (int mm = 0; mm < M; mm++)
        for (int k = 0; k < K; k++) {
            float v = a[(size_t) mm * K + k] / sa[mm];
            int q = (int) std::lroundf(v);
            q = std::max(-127, std::min(127, q));
            input_i8[(size_t) mm * K + k] = (int8_t) q;
        }
    const uint64_t hp_t1 = std::getenv("XDNA_HOSTPROF") ? ggml_time_ns() : 0;
    if (std::getenv("XDNA_SRCINFO")) {
        static int g_si = 0;
        if (g_si < 8) {
            g_si++;
            float amx = 0.0f;
            for (int i = 0; i < M * K; i++) amx = std::max(amx, std::fabs(a[i]));
            fprintf(stderr, "[XDNA-SRCINFO] op=%s M=%d K=%d N=%d amax=%.6g type=%d xdna_buf=%d data=%p view_offs=%zu nb0=%zu nb1=%zu nb2=%zu\n",
                    ggml_xdna_gemm_operation(src0), M, K, N, amx, (int) src1->type,
                    (int) is_xdna_buffer(src1), (void *) src1->data, src1->view_offs,
                    src1->nb[0], src1->nb[1], src1->nb[2]);
        }
    }

    // pack A (int8 only, tile-major matching mmul-v2 kernel)
    std::vector<int8_t> Ap((size_t) n_mt * n_kc * a_chunk, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            for (int im = 0; im < m / mmul; im++)
                for (int bb = 0; bb < nblk; bb++)
                    for (int kk = 0; kk < 4; kk++)
                        for (int rr = 0; rr < mmul; rr++)
                            for (int cc_col = 0; cc_col < 8; cc_col++) {
                                size_t flat_idx = ((im * nblk + bb) * 4 + kk) * (mmul * 8) + rr * 8 + cc_col;
                                size_t src_idx = (mt * m + im * mmul + rr) * K + cc * kc + bb * 32 + kk * 8 + cc_col;
                                t[flat_idx] = input_i8[src_idx];
                            }
        }
    // pack B: use pre-packed tile-major Bp from cache if available
    std::vector<int8_t> Bp;
    if (!(w->bp_ready && w->bp_ncols == n_cols)) {
        Bp.assign((size_t) n_cols * n_nt_c * n_kc * b_chunk, 0);
        for (int col = 0; col < n_cols; col++)
            for (int nt = 0; nt < n_nt_c; nt++)
                for (int cc = 0; cc < n_kc; cc++) {
                    int8_t * t = &Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                    for (int jn = 0; jn < n / 8; jn++)
                        for (int bb = 0; bb < nblk; bb++)
                            for (int kk = 0; kk < 4; kk++)
                                for (int rr = 0; rr < 8; rr++)
                                    for (int cc_col = 0; cc_col < 8; cc_col++) {
                                        size_t flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                        size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n + jn * 8 + cc_col;
                                        t[flat_idx] = weight_i8[src_idx];
                                    }
                }
    }

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    const void * bp_ptr = (w->bp_ready && w->bp_ncols == n_cols) ? (const void *)w->Bp.data() : (const void *)Bp.data();

    // Phase 60: host-side preparation breakdown (XDNA_HOSTPROF=1).
    const bool hostprof = std::getenv("XDNA_HOSTPROF") != nullptr;
    const uint64_t hp_t0 = hostprof ? ggml_time_ns() : 0;

    // Phase 60 debug: compute the exact int8 result the device should produce
    // from the same host-side operands, so a device error can be separated
    // from a host-side error. Enabled with XDNA_VERIFY=1.
    std::vector<int32_t> verify_ref;
    std::vector<float>  verify_fref;
    int    verify_rows = 0;
    float  verify_amax = 0.0f;
    float  verify_wmax = 0.0f;
    float  verify_smax = 0.0f;

    if (std::getenv("XDNA_VERIFY")) {
        verify_ref.assign((size_t) M * N, 0);
        for (int mm = 0; mm < M; mm++) {
            for (int kk = 0; kk < K; kk++) {
                const int8_t av = input_i8[(size_t) mm * K + kk];
                if (av == 0) continue;
                const int8_t * wrow = &weight_i8[(size_t) kk * N];
                int32_t * orow = &verify_ref[(size_t) mm * N];
                for (int nn = 0; nn < N; nn++) orow[nn] += (int32_t) av * wrow[nn];
            }
        }
        // float reference from the true float activations and the requantized
        // weights: isolates the activation path from the weight path.
        verify_rows = 4;
        verify_fref.assign((size_t) verify_rows * N, 0.0f);
        for (int mm = 0; mm < M; mm++)
            for (int kk = 0; kk < K; kk++)
                verify_amax = std::max(verify_amax, std::fabs(a[(size_t) mm * K + kk]));
        for (int kk = 0; kk < K; kk++)
            for (int nn = 0; nn < N; nn++)
                verify_wmax = std::max(verify_wmax, std::fabs((float) weight_i8[(size_t) kk * N + nn]));
        for (int nn = 0; nn < N; nn++) verify_smax = std::max(verify_smax, std::fabs(sb[nn]));
        for (int mm = 0; mm < verify_rows; mm++) {
            for (int kk = 0; kk < K; kk++) {
                const float av = a[(size_t) mm * K + kk];
                if (av == 0.0f) continue;
                const int8_t * wrow = &weight_i8[(size_t) kk * N];
                float * orow = &verify_fref[(size_t) mm * N];
                for (int nn = 0; nn < N; nn++) orow[nn] += av * ((float) wrow[nn] * sb[nn]);
            }
        }
    }

    if (!ggml_xdna_q8_c4_mmul_v2_submit(e, Ap.data(), bp_ptr, &run)) return false;
    int buf_idx = used_buf;
    if (hostprof) {
        const uint64_t hp_t2 = ggml_time_ns();
        fprintf(stderr, "[XDNA-HOSTPROF] M=%d K=%d N=%d act_dequant_quant=%.2fms packA_packB=%.2fms xrt_submit=%.2fms host_total=%.2fms\n",
                M, K, N,
                (hp_t1 - hp_t0) / 1e6,
                (hp_t2 - hp_t1) / 1e6,
                (hp_t2 - hp_t2) / 1e6,
                (hp_t2 - hp_t0) / 1e6);
    }

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_c4_mmul_v2_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    submission.n_cols = n_cols;
    submission.requant_sa = sa;
    submission.requant_sb = sb;
    submission.verify_ref = std::move(verify_ref);
    submission.verify_fref = std::move(verify_fref);
    submission.verify_rows = verify_rows;
    submission.verify_amax = verify_amax;
    submission.verify_wmax = verify_wmax;
    submission.verify_smax = verify_smax;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 20G step 5: submit a Q8_0 full-K MUL_MAT to the scale-aware kernel.
// The host dequantizes Q8_0 to int8 + per-32-block f32 scales, packs both
// operands tile-major, then issues ONE xrt::run.start() covering the whole K.
static bool ggml_xdna_submit_q8_scalefullk(ggml_xdna_backend_context * ctx,
                                           XDNACompiledKernel * compiled_kernel,
                                           const struct ggml_tensor * src0,
                                           const struct ggml_tensor * src1,
                                           struct ggml_tensor * dst) {
    if (!compiled_kernel || !compiled_kernel->q8_scale_engine) return false;
    ggml_xdna_q8_scale_engine * e = compiled_kernel->q8_scale_engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    int m = 32, n = 64, kc = 256, nblk = 8;
    if (K % kc != 0 || M % m != 0 || N % n != 0) return false;
    int nb = K / 32;
    int n_kc = K / kc;
    int a_chunk = m * kc + m * nblk * 4;
    int b_chunk = kc * n + nblk * n * 4;

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_engine == e) { pending_count++; }
    }
    if (!ctx->persistent_mode) {
        if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;
    }

    // weight Q8_0 [K,N] -> int8 [K,N] row-major + scale [nb,N]
    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    std::vector<int8_t> weight_i8((size_t) K * N);
    std::vector<float> w_scale((size_t) nb * N);
    for (int nn2 = 0; nn2 < N; nn2++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
            w_scale[(size_t) b * N + nn2] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                weight_i8[(size_t)(b * 32 + kk) * N + nn2] = (int8_t) block[2 + kk];
        }

    // input -> int8 [M,K] row-major + scale [M,nb]
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        quantize_f32_to_i8_blocked(in_f32_ptr, M, K, input_i8, i_scale);
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        input_i8.resize((size_t) K * M);
        i_scale.resize((size_t) M * nb);
        for (int mm2 = 0; mm2 < M; mm2++)
            for (int b = 0; b < nb; b++) {
                const uint8_t * block = &in_q8_ptr[((size_t) mm2 * nb + b) * 34];
                i_scale[(size_t) mm2 * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                for (int kk = 0; kk < 32; kk++)
                    input_i8[(size_t) mm2 * K + b * 32 + kk] = (int8_t) block[2 + kk];
            }
    } else {
        return false;
    }

    // pack A: chunk (mt,cc) = [m,kc] int8 + [m,nblk] f32
    std::vector<int8_t> Ap((size_t)(M / m) * n_kc * a_chunk, 0);
    for (int mt = 0; mt < M / m; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            for (int i = 0; i < m; i++)
                for (int l = 0; l < kc; l++)
                    t[i * kc + l] = input_i8[(size_t)(mt * m + i) * K + cc * kc + l];
            float * ts = (float *)(t + m * kc);
            for (int i = 0; i < m; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = i_scale[(size_t)(mt * m + i) * nb + cc * nblk + bb];
        }

    // pack B: chunk (nt,cc) = [kc,n] int8 + [nblk,n] f32
    std::vector<int8_t> Bp((size_t)(N / n) * n_kc * b_chunk, 0);
    for (int nt = 0; nt < N / n; nt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Bp[(size_t)(nt * n_kc + cc) * b_chunk];
            for (int l = 0; l < kc; l++)
                for (int j = 0; j < n; j++)
                    t[l * n + j] = weight_i8[(size_t)(cc * kc + l) * N + nt * n + j];
                float * ts = (float *)(t + kc * n);
                for (int bb = 0; bb < nblk; bb++)
                    for (int j = 0; j < n; j++)
                        ts[bb * n + j] = w_scale[(size_t)(cc * nblk + bb) * N + nt * n + j];
            }

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    if (!ggml_xdna_q8_scale_submit(e, Ap.data(), Bp.data(), &run)) return false;

    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 40H: submit a Q8_0 full-K MUL_MAT to the Q8SCALEV-C4 engine.
// Same packing as Q8SCALEV-C1 but B and C are N-sliced across n_cols columns.
static bool ggml_xdna_submit_q8_scalefullk_c4(ggml_xdna_backend_context * ctx,
                                               XDNACompiledKernel * compiled_kernel,
                                               const struct ggml_tensor * src0,
                                               const struct ggml_tensor * src1,
                                               struct ggml_tensor * dst) {
    if (!compiled_kernel || !compiled_kernel->q8_scale_c4_engine) return false;
    ggml_xdna_q8_scale_c4_engine * e = compiled_kernel->q8_scale_c4_engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    int m = 32, n = 64, kc = 256, nblk = 8;
    if (K % kc != 0 || M % m != 0 || N % n != 0) return false;
    int nb = K / 32;
    int n_kc = K / kc;
    int a_chunk = m * kc + m * nblk * 4;
    int b_chunk = kc * n + nblk * n * 4;

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_scale_c4_engine == e) { pending_count++; }
    }
    if (!ctx->persistent_mode) {
        if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;
    }

    // weight Q8_0 [K,N] -> int8 [K,N] row-major + scale [nb,N]
    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    std::vector<int8_t> weight_i8((size_t) K * N);
    std::vector<float> w_scale((size_t) nb * N);
    for (int nn2 = 0; nn2 < N; nn2++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
            w_scale[(size_t) b * N + nn2] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                weight_i8[(size_t)(b * 32 + kk) * N + nn2] = (int8_t) block[2 + kk];
        }

    // input -> int8 [M,K] row-major + scale [M,nb]
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        quantize_f32_to_i8_blocked(in_f32_ptr, M, K, input_i8, i_scale);
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        input_i8.resize((size_t) K * M);
        i_scale.resize((size_t) M * nb);
        for (int mm2 = 0; mm2 < M; mm2++)
            for (int b = 0; b < nb; b++) {
                const uint8_t * block = &in_q8_ptr[((size_t) mm2 * nb + b) * 34];
                i_scale[(size_t) mm2 * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                for (int kk = 0; kk < 32; kk++)
                    input_i8[(size_t) mm2 * K + b * 32 + kk] = (int8_t) block[2 + kk];
            }
    } else {
        return false;
    }

    int n_cols = e->n_cols;
    int n_mt = M / m;
    int n_nt_c = (N / n_cols) / n;

    // pack A: same as Q8SCALEV-C1
    std::vector<int8_t> Ap((size_t)n_mt * n_kc * a_chunk, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            for (int i = 0; i < m; i++)
                for (int l = 0; l < kc; l++)
                    t[i * kc + l] = input_i8[(size_t)(mt * m + i) * K + cc * kc + l];
            float * ts = (float *)(t + m * kc);
            for (int i = 0; i < m; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = i_scale[(size_t)(mt * m + i) * nb + cc * nblk + bb];
        }

    // pack B: N-sliced across columns
    std::vector<int8_t> Bp((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
    for (int col = 0; col < n_cols; col++)
        for (int nt = 0; nt < n_nt_c; nt++)
            for (int cc = 0; cc < n_kc; cc++) {
                int8_t * t = &Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                for (int l = 0; l < kc; l++)
                    for (int j = 0; j < n; j++)
                        t[l * n + j] = weight_i8[(size_t)(cc * kc + l) * N + (col * n_nt_c + nt) * n + j];
                float * ts = (float *)(t + kc * n);
                for (int bb = 0; bb < nblk; bb++)
                    for (int j = 0; j < n; j++)
                        ts[bb * n + j] = w_scale[(size_t)(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + j];
            }

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    if (!ggml_xdna_q8_scale_c4_submit(e, Ap.data(), Bp.data(), &run)) return false;

    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_scale_c4_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 61: build the tile-major packed weights of the scale preserving
// path. Layout per (col, nt, k-chunk):
//   int8 tiles of 8 K x 8 N, then (KC/32) x N_TILE f32 column block scales.
static void ggml_xdna_c4_build_weight(
        const struct ggml_tensor * src0, int K, int N,
        int n_cols, int n_nt_c, int n_kc, int nblk, int kc, int n,
        std::vector<int8_t> & Bp) {

    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    const int nb = K / 32;
    std::vector<int8_t> weight_i8((size_t) K * N);
    std::vector<float> w_scale((size_t) nb * N);
    for (int nn2 = 0; nn2 < N; nn2++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
            w_scale[(size_t) b * N + nn2] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                weight_i8[(size_t)(b * 32 + kk) * N + nn2] = (int8_t) block[2 + kk];
        }

    const int b_chunk = kc * n + nblk * n * 4;
    Bp.assign((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
    for (int col = 0; col < n_cols; col++)
        for (int nt = 0; nt < n_nt_c; nt++)
            for (int cc = 0; cc < n_kc; cc++) {
                int8_t * t = &Bp[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                for (int jn = 0; jn < n/8; jn++)
                    for (int bb = 0; bb < nblk; bb++)
                        for (int kk = 0; kk < 4; kk++)
                            for (int rr = 0; rr < 8; rr++)
                                for (int cc_col = 0; cc_col < 8; cc_col++) {
                                    size_t flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                    size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n + jn * 8 + cc_col;
                                    t[flat_idx] = weight_i8[src_idx];
                                }
                float * ts = (float *)(t + kc * n);
                for (int bb = 0; bb < nblk; bb++)
                    for (int j = 0; j < n; j++)
                        ts[bb * n + j] = w_scale[(size_t)(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + j];
            }
}

// Phase 61: build (or fetch from the cache) the tile-major packed weights.
// The reference stays valid because std::unordered_map never invalidates
// references to its elements on insert.
static const std::vector<int8_t> * ggml_xdna_c4_pack_weight(
        const struct ggml_tensor * src0, int K, int N,
        int m, int n, int kc, int n_cols, int n_nt_c, int n_kc, int nblk, int b_chunk) {

    // The identity of the weight tensor is its tensor object and its name:
    // src0->data, src0->view_offs and get_tensor_data() are identical for
    // every weight of an mmap'ed model, so a pointer based key made an
    // earlier version of this cache return the wrong tensor's weights
    // (PPL 6.3e6 instead of 16.44).
    const void * wptr = src0;
    const uint64_t key = ggml_xdna_c4_wkey(wptr, src0->name, m, n, kc, n_cols, nblk);
    {
        std::lock_guard<std::mutex> lock(g_c4_wcache_mutex);
        auto it = g_c4_wcache.find(key);
        if (it != g_c4_wcache.end()) {
            const bool same = it->second.K == K && it->second.N == N &&
                it->second.m == m && it->second.n == n && it->second.kc == kc &&
                it->second.n_cols == n_cols && it->second.nblk == nblk &&
                it->second.src == wptr &&
                strncmp(it->second.name, src0->name, sizeof(it->second.name)) == 0;
            if (getenv("XDNA_WCACHE_DEBUG")) {
                fprintf(stderr, "[XDNA-WCACHE] key=%016llx obj=%p name=%s "
                        "req(K=%d N=%d m=%d n=%d kc=%d) "
                        "hit(K=%d N=%d m=%d n=%d kc=%d name=%s) same=%d\n",
                        (unsigned long long) key, wptr, src0->name, K, N, m, n, kc,
                        it->second.K, it->second.N, it->second.m, it->second.n, it->second.kc,
                        it->second.name, (int) same);
            }
            if (same) {
                return &it->second.Bp;
            }
        }
    }

    std::vector<int8_t> Bp;
    ggml_xdna_c4_build_weight(src0, K, N, n_cols, n_nt_c, n_kc, nblk, kc, n, Bp);

    std::lock_guard<std::mutex> lock(g_c4_wcache_mutex);
    const size_t limit = ggml_xdna_c4_wcache_limit();
    auto it = g_c4_wcache.find(key);
    if (it != g_c4_wcache.end()) {
        // a stale entry with the same key: replace it
        g_c4_wcache_bytes -= it->second.Bp.size();
        g_c4_wcache.erase(it);
    }
    if (g_c4_wcache_bytes + Bp.size() > limit) {
        // evict until the new entry fits
        for (auto it2 = g_c4_wcache.begin(); it2 != g_c4_wcache.end(); ) {
            g_c4_wcache_bytes -= it2->second.Bp.size();
            it2 = g_c4_wcache.erase(it2);
            if (g_c4_wcache_bytes + Bp.size() <= limit) break;
        }
    }
    ggml_xdna_c4_weight w;
    w.K = K; w.N = N;
    w.m = m; w.n = n; w.kc = kc; w.n_cols = n_cols; w.nblk = nblk;
    w.src = wptr;
    snprintf(w.name, sizeof(w.name), "%s", src0->name);
    w.Bp = std::move(Bp);
    g_c4_wcache_bytes += w.Bp.size();
    auto res = g_c4_wcache.emplace(key, std::move(w));
    return &res.first->second.Bp;
}

// Phase 61: the uncached variant, used for the A/B measurement.
static const std::vector<int8_t> * ggml_xdna_c4_pack_weight_uncached(
        const struct ggml_tensor * src0, int K, int N,
        int n_cols, int n_nt_c, int n_kc, int nblk, int kc, int n) {
    static thread_local std::vector<int8_t> Bp;
    ggml_xdna_c4_build_weight(src0, K, N, n_cols, n_nt_c, n_kc, nblk, kc, n, Bp);
    return &Bp;
}

// Phase 62: build the tile-major packed weights for the c256 engine.
//
// The c256 kernel applies one scale per K chunk, so the weights have to be
// requantized to one block per chunk: the Q8_0 values are dequantized, the
// chunk maximum is taken over the 256 elements of the chunk and the values
// are requantized to int8 with that scale.  The int8 tile order is the same
// as the scale-preserving builder, only the scale section changes (one
// float per column instead of one per 32-element block).
static void ggml_xdna_c4_build_weight_c256(
        const struct ggml_tensor * src0, int K, int N,
        int n_cols, int n_nt_c, int n_kc, int kc, int n,
        std::vector<int8_t> & Bp) {

    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    const int nb = K / 32;
    const int nkc = K / kc;
    std::vector<int8_t> weight_i8((size_t) K * N);
    std::vector<float> w_scale((size_t) nkc * N);
    std::vector<float> col((size_t) K);

    for (int nn2 = 0; nn2 < N; nn2++) {
        // dequantize the column
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
            const float s = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
            for (int kk = 0; kk < 32; kk++)
                col[(size_t) b * 32 + kk] = (float) (int8_t) block[2 + kk] * s;
        }
        // requantize to one scale per K chunk
        for (int c = 0; c < nkc; c++) {
            const float * blk = col.data() + (size_t) c * kc;
            float max_abs = 0;
            for (int k = 0; k < kc; k++) max_abs = std::max(max_abs, std::fabs(blk[k]));
            const float s = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
            w_scale[(size_t) c * N + nn2] = s;
            for (int k = 0; k < kc; k++) {
                float v = blk[k] / s;
                int q = (int) std::lroundf(v);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                weight_i8[(size_t) (c * kc + k) * N + nn2] = (int8_t) q;
            }
        }
    }

    const int b_chunk = kc * n + n * 4;
    Bp.assign((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
    for (int col_i = 0; col_i < n_cols; col_i++)
        for (int nt = 0; nt < n_nt_c; nt++)
            for (int cc = 0; cc < n_kc; cc++) {
                int8_t * t = &Bp[(size_t)((col_i * n_nt_c + nt) * n_kc + cc) * b_chunk];
                for (int jn = 0; jn < n/8; jn++)
                    for (int bb = 0; bb < kc/32; bb++)
                        for (int kk = 0; kk < 4; kk++)
                            for (int rr = 0; rr < 8; rr++)
                                for (int cc_col = 0; cc_col < 8; cc_col++) {
                                    size_t flat_idx = ((jn * (kc/32) + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                    size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col_i * n_nt_c + nt) * n + jn * 8 + cc_col;
                                    t[flat_idx] = weight_i8[src_idx];
                                }
                float * ts = (float *)(t + kc * n);
                for (int j = 0; j < n; j++)
                    ts[j] = w_scale[(size_t) cc * N + (col_i * n_nt_c + nt) * n + j];
            }
}

// Phase 40M: Q8_0 full-K GEMM using Q8SCALEV-C4 mmul-v2 (aie::mmul<4,8,8>).
// Same data layout as Q8SCALEV-C4 but uses the faster mmul-based kernel.
static bool ggml_xdna_submit_q8_c4_mmul_v2(ggml_xdna_backend_context * ctx,
                                             XDNACompiledKernel * compiled_kernel,
                                             const struct ggml_tensor * src0,
                                             const struct ggml_tensor * src1,
                                             struct ggml_tensor * dst,
                                             bool profile) {
    if (!compiled_kernel || !compiled_kernel->q8_c4_mmul_v2_engine) return false;
    ggml_xdna_q8_c4_mmul_v2_engine * e = compiled_kernel->q8_c4_mmul_v2_engine;
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    // Phase 60/61: the tile shape is a property of the compiled engine.
    const int m = e->m, n = e->n, kc = e->kt, nblk = e->kt / 32, mmul = e->mmul;
    if (K % kc != 0 || M % m != 0 || N % n != 0) return false;
    int nb = K / 32;
    int n_kc = K / kc;
    int n_cols = e->n_cols;
    int n_mt = M / m;
    int n_nt_c = (N / n_cols) / n;
    int a_chunk = m * kc + m * nblk * 4;
    int b_chunk = kc * n + nblk * n * 4;
    const int s12 = e->s12;
    const int c256 = e->c256;
    if (s12) {
        a_chunk = m * kc + nblk * (m / 4) * 32 * 4;
        b_chunk = kc * n + nblk * (n / 8) * 16 * 4;
    }
    if (c256) {
        // Phase 62: one scale per K chunk
        a_chunk = m * kc + m * 4;
        b_chunk = kc * n + n * 4;
    }
    ggml_xdna_xrt_trace trace = {};
    uint64_t host_prep_start_ns = ggml_time_ns();
    if (profile) {
        trace.host_prep_start_ns = host_prep_start_ns;
    }

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_c4_mmul_v2_engine == e) { pending_count++; }
    }
    if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;

    // Phase 61 host-side breakdown (XDNA_HOSTPROF=1).
    const bool hostprof_c4 = std::getenv("XDNA_HOSTPROF") != nullptr;
    const uint64_t hp0 = hostprof_c4 ? ggml_time_ns() : 0;

    // Phase 61: the packed weights can be cached per tensor and tile shape,
    // so the dequantisation and the tile packing happen once per weight
    // tensor instead of once per GEMM. The cache is opt-in: it needs up to
    // one packed copy per weight tensor (4 MB each for this model) and the
    // measured end-to-end effect was inside the noise, so it is off by
    // default to keep the host memory free.
    std::vector<int8_t> Bp_s12;
    const std::vector<int8_t> * Bp_ptr = nullptr;
    static const bool wcache_on = getenv("XDNA_WCACHE") != nullptr &&
                                  strcmp(getenv("XDNA_WCACHE"), "0") != 0;
    if (c256) {
        // The c256 weights are requantized, so they do not share the cache
        // namespace of the scale preserving weights.
        static thread_local std::vector<int8_t> Bp_c256;
        ggml_xdna_c4_build_weight_c256(src0, K, N, n_cols, n_nt_c, n_kc, kc, n, Bp_c256);
        Bp_ptr = &Bp_c256;
    } else if (wcache_on) {
        Bp_ptr = ggml_xdna_c4_pack_weight(src0, K, N, m, n, kc, n_cols, n_nt_c, n_kc, nblk, b_chunk);
    } else {
        Bp_ptr = ggml_xdna_c4_pack_weight_uncached(src0, K, N, n_cols, n_nt_c, n_kc, nblk, kc, n);
    }
    const uint64_t hp1 = hostprof_c4 ? ggml_time_ns() : 0;

    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        if (c256) {
            quantize_f32_to_i8_chunked(in_f32_ptr, M, K, kc, input_i8, i_scale);
        } else {
            quantize_f32_to_i8_blocked(in_f32_ptr, M, K, input_i8, i_scale);
        }
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        if (c256) {
            // The c256 engine needs one scale per K chunk: dequantize the
            // Q8_0 activation and requantize it to the chunk granularity.
            in_f32_local.resize((size_t) K * M);
            for (int mm2 = 0; mm2 < M; mm2++)
                for (int b = 0; b < nb; b++) {
                    const uint8_t * block = &in_q8_ptr[((size_t) mm2 * nb + b) * 34];
                    const float s = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                    for (int kk = 0; kk < 32; kk++)
                        in_f32_local[(size_t) mm2 * K + b * 32 + kk] = (float) (int8_t) block[2 + kk] * s;
                }
            quantize_f32_to_i8_chunked(in_f32_local.data(), M, K, kc, input_i8, i_scale);
        } else {
            input_i8.resize((size_t) K * M);
            i_scale.resize((size_t) M * nb);
            for (int mm2 = 0; mm2 < M; mm2++)
                for (int b = 0; b < nb; b++) {
                    const uint8_t * block = &in_q8_ptr[((size_t) mm2 * nb + b) * 34];
                    i_scale[(size_t) mm2 * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                    for (int kk = 0; kk < 32; kk++)
                        input_i8[(size_t) mm2 * K + b * 32 + kk] = (int8_t) block[2 + kk];
                }
        }
    } else {
        return false;
    }
    const uint64_t hp2 = hostprof_c4 ? ggml_time_ns() : 0;

    std::vector<int8_t> Ap((size_t)n_mt * n_kc * a_chunk, 0);
    for (int mt = 0; mt < n_mt; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_chunk];
            // A tiles are mmul rows x 8 K (4x8 for the phase 40 kernel, 8x8
            // for the phase 61 native matrix unit)
            for (int im = 0; im < m/mmul; im++)
                for (int bb = 0; bb < nblk; bb++)
                    for (int kk = 0; kk < 4; kk++)
                        for (int rr = 0; rr < mmul; rr++)
                            for (int cc_col = 0; cc_col < 8; cc_col++) {
                                size_t flat_idx = ((im * nblk + bb) * 4 + kk) * (mmul * 8) + rr * 8 + cc_col;
                                size_t src_idx = (mt * m + im * mmul + rr) * K + cc * kc + bb * 32 + kk * 8 + cc_col;
                                t[flat_idx] = input_i8[src_idx];
                            }
            float * ts = (float *)(t + m * kc);
            if (c256) {
                // Phase 62: one scale per row per K chunk
                for (int i = 0; i < m; i++)
                    ts[i] = i_scale[(size_t)(mt * m + i) * n_kc + cc];
            } else if (!s12) {
            for (int i = 0; i < m; i++)
                for (int bb = 0; bb < nblk; bb++)
                    ts[i * nblk + bb] = i_scale[(size_t)(mt * m + i) * nb + cc * nblk + bb];
            } else {
                // s12: per (bb,i) 32 pre-broadcast floats [a(i*4+r) x8]
                for (int bb = 0; bb < nblk; bb++)
                    for (int im = 0; im < m / 4; im++)
                        for (int rr = 0; rr < 4; rr++) {
                            float s = i_scale[(size_t)(mt * m + im * 4 + rr) * nb + cc * nblk + bb];
                            for (int ll = 0; ll < 8; ll++)
                                ts[(bb * (m / 4) + im) * 32 + rr * 8 + ll] = s;
                        }
                }
            }
    const uint64_t hp3 = hostprof_c4 ? ggml_time_ns() : 0;

    if (s12) {
        // s12: per (bb,jn) 16 duplicated floats [bsv, bsv]
        const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
        std::vector<uint8_t> w_q8_local;
        if (!w_q8_ptr) {
            w_q8_local.resize(ggml_nbytes(src0));
            ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
            w_q8_ptr = w_q8_local.data();
        }
        std::vector<int8_t> weight_i8((size_t) K * N);
        std::vector<float> w_scale((size_t) nb * N);
        for (int nn2 = 0; nn2 < N; nn2++)
            for (int b = 0; b < nb; b++) {
                const uint8_t * block = &w_q8_ptr[((size_t) nn2 * nb + b) * 34];
                w_scale[(size_t) b * N + nn2] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
                for (int kk = 0; kk < 32; kk++)
                    weight_i8[(size_t)(b * 32 + kk) * N + nn2] = (int8_t) block[2 + kk];
            }
        Bp_s12.assign((size_t)n_cols * n_nt_c * n_kc * b_chunk, 0);
        for (int col = 0; col < n_cols; col++)
            for (int nt = 0; nt < n_nt_c; nt++)
                for (int cc = 0; cc < n_kc; cc++) {
                    int8_t * t = &Bp_s12[(size_t)((col * n_nt_c + nt) * n_kc + cc) * b_chunk];
                    for (int jn = 0; jn < n/8; jn++)
                        for (int bb = 0; bb < nblk; bb++)
                            for (int kk = 0; kk < 4; kk++)
                                for (int rr = 0; rr < 8; rr++)
                                    for (int cc_col = 0; cc_col < 8; cc_col++) {
                                        size_t flat_idx = ((jn * nblk + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                        size_t src_idx = (cc * kc + bb * 32 + kk * 8 + rr) * N + (col * n_nt_c + nt) * n + jn * 8 + cc_col;
                                        t[flat_idx] = weight_i8[src_idx];
                                    }
                    float * ts = (float *)(t + kc * n);
                    for (int bb = 0; bb < nblk; bb++)
                        for (int jn = 0; jn < n / 8; jn++)
                            for (int ll = 0; ll < 8; ll++) {
                                float s = w_scale[(size_t)(cc * nblk + bb) * N + (col * n_nt_c + nt) * n + jn * 8 + ll];
                                ts[(bb * (n / 8) + jn) * 16 + ll] = s;
                                ts[(bb * (n / 8) + jn) * 16 + 8 + ll] = s;
                            }
                }
        Bp_ptr = &Bp_s12;
    }

    const uint64_t hp4 = hostprof_c4 ? ggml_time_ns() : 0;

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    if (profile) {
        trace.host_prep_end_ns = ggml_time_ns();
        trace.expected_k_blocks = K / kc;
        trace.actual_k_blocks = K / kc;
        trace.q8_scale_blocks = K / 32;
        trace.k_block_size = kc;
        trace.k_chunks = K / kc;
        trace.m_tiles = M / m;
        trace.n_tiles = (N / n_cols) / n;
        trace.n_cols = n_cols;
        trace.host_prep_ms = (trace.host_prep_end_ns - trace.host_prep_start_ns) / 1e6;
    }
    if (!ggml_xdna_q8_c4_mmul_v2_submit(e, Ap.data(), Bp_ptr->data(), &run, profile ? &trace : nullptr)) return false;

    if (hostprof_c4) {
        const uint64_t hp5 = ggml_time_ns();
        fprintf(stderr, "[XDNA-HOSTPROF-C4] M=%d K=%d N=%d w_pack=%.2fms act_quant=%.2fms packA=%.2fms packB=%.2fms xrt_submit=%.2fms host_total=%.2fms\n",
                M, K, N,
                (hp1 - hp0) / 1e6,
                (hp2 - hp1) / 1e6,
                (hp3 - hp2) / 1e6,
                (hp4 - hp3) / 1e6,
                (hp5 - hp4) / 1e6,
                (hp5 - hp0) / 1e6);
    }

    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_c4_mmul_v2_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    if (profile) {
        submission.xrt_trace = trace;
    }
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 39C: Q8_0 full-K GEMM using C4_WA mmul kernel with host-side scale folding.
// For each K-block (32 elements), fold per-block scales into int8 data on host,
// run C4_WA mmul, accumulate float results on host.
static std::mutex g_q8_c4_mutex;
static std::unordered_map<uint64_t, ggml_xdna_q8_c4_engine *> g_q8_c4_engines;

// Phase 22: whole-array fast path engine cache.
static std::mutex g_wa_mutex;
static std::unordered_map<uint64_t, ggml_xdna_wa_engine *> g_wa_engines;

static const char * ggml_xdna_q8_c4_cache_dir(void) {
    const char * env = std::getenv("XDNA_KERNEL_CACHE");
    if (env && *env) return env;
    return "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache";
}

static ggml_xdna_q8_c4_engine * ggml_xdna_q8_c4_get(int M, int K, int N) {
    uint64_t key = ((uint64_t) M << 42) ^ ((uint64_t) K << 21) ^ (uint64_t) N;
    std::lock_guard<std::mutex> lock(g_q8_c4_mutex);
    auto it = g_q8_c4_engines.find(key);
    if (it != g_q8_c4_engines.end()) return it->second;

    char xc[512], ins[512];
    snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_WA/design.xclbin",
             ggml_xdna_q8_c4_cache_dir(), M, K, N);
    snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_WA/design.insts.bin",
             ggml_xdna_q8_c4_cache_dir(), M, K, N);
    ggml_xdna_q8_c4_engine * e = ggml_xdna_q8_c4_create(M, K, N, xc, ins);
    if (e) g_q8_c4_engines[key] = e;
    return e;
}

// Fold Q8_0 per-block scales into int8 data for one K-block.
// Returns re-quantized int8 block and the global scale used.
static void q8_fold_scales_block(
    const uint8_t * q8_ptr, int K, int N, int blk, int stride,
    int8_t * out_int8, float * out_gs) {
    // q8_ptr points to Q8_0 blocks: each block is 34 bytes (2 f16 scale + 32 int8)
    // stride is the number of elements in the N dimension (N for row-major, K for col-major)
    // For weight [K,N] row-major: q8_ptr[blk * 34 + n * stride * 34 + ...]
    
    // Find max absolute value after applying per-block scale
    float amax = 1e-9f;
    for (int n = 0; n < N; n++) {
        const uint8_t * block = &q8_ptr[((size_t)n * (K/32) + blk) * 34];
        float scale = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
        for (int kk = 0; kk < 32; kk++) {
            float val = scale * (int8_t)block[2 + kk];
            if (fabsf(val) > amax) amax = fabsf(val);
        }
    }
    float gs = amax / 127.0f;
    *out_gs = gs;

    for (int n = 0; n < N; n++) {
        const uint8_t * block = &q8_ptr[((size_t)n * (K/32) + blk) * 34];
        float scale = ggml_fp16_to_fp32(*(const ggml_fp16_t *) block);
        for (int kk = 0; kk < 32; kk++) {
            float val = scale * (int8_t)block[2 + kk];
            int q = (int) std::lroundf(val / gs);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out_int8[(size_t)(blk * 32 + kk) * N + n] = (int8_t) q;
        }
    }
}

// Submit Q8_0 full-K MUL_MAT using C4_WA mmul with host-side scale folding.
// One XRT submit per K-block (32 elements). Results accumulated on host.
static bool ggml_xdna_submit_q8_c4(ggml_xdna_backend_context * ctx,
                                   const struct ggml_tensor * src0,
                                   const struct ggml_tensor * src1,
                                   struct ggml_tensor * dst) {
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (K % 32 != 0 || M < 64) return false;

    ggml_xdna_q8_c4_engine * e = ggml_xdna_q8_c4_get(M, K, N);
    if (!e) return false;

    bool need_flush = false;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_c4_engine == e) { need_flush = true; break; }
    }
    if (need_flush && !ggml_xdna_drain_pending(ctx)) return false;

    // weight Q8_0 [K,N] -> per-K-block re-quantized int8
    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    int nb = K / 32;
    std::vector<int8_t> weight_i8((size_t) K * N);
    std::vector<float> w_gs(nb);
    for (int b = 0; b < nb; b++) {
        q8_fold_scales_block(w_q8_ptr, K, N, b, N, &weight_i8[(size_t)b * 32 * N], &w_gs[b]);
    }

    // input -> int8 [M,K] row-major (F32 or Q8_0)
    std::vector<int8_t> input_i8((size_t) K * M);
    std::vector<float> i_gs(nb);
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        // Quantize F32 to int8 per K-block
        for (int b = 0; b < nb; b++) {
            float amax = 1e-9f;
            for (int m = 0; m < M; m++) {
                for (int kk = 0; kk < 32; kk++) {
                    float val = in_f32_ptr[(size_t)m * K + b * 32 + kk];
                    if (fabsf(val) > amax) amax = fabsf(val);
                }
            }
            float gs = amax / 127.0f;
            i_gs[b] = gs;
            for (int m = 0; m < M; m++) {
                for (int kk = 0; kk < 32; kk++) {
                    float val = in_f32_ptr[(size_t)m * K + b * 32 + kk];
                    int q = (int) std::lroundf(val / gs);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    input_i8[(size_t)(b * 32 + kk) * M + m] = (int8_t) q;
                }
            }
        }
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        for (int b = 0; b < nb; b++) {
            q8_fold_scales_block(in_q8_ptr, K, M, b, M, &input_i8[(size_t)b * 32 * M], &i_gs[b]);
        }
    } else {
        return false;
    }

    // Output accumulator (float for scaled accumulation)
    // ggml output is [N, M] column-major = out[n * M + m]
    std::vector<float> out_acc((size_t) M * N, 0.0f);

    // Loop over K-blocks
    for (int b = 0; b < nb; b++) {
        const int8_t * a_block = &weight_i8[(size_t)b * 32 * N];
        const int8_t * b_block = &input_i8[(size_t)b * 32 * M];

        xrt::run * run = nullptr;
        if (!ggml_xdna_q8_c4_submit(e, a_block, b_block, &run)) {
            fprintf(stderr, "[XDNA-Q8-C4] submit failed at block %d\n", b);
            return false;
        }

        // Wait for result
        std::vector<int32_t> chunk_output((size_t) M * N);
        int buf_idx = 1 - e->current_buffer;
        if (!ggml_xdna_q8_c4_wait(e, run, chunk_output.data(), buf_idx)) {
            return false;
        }

        // Apply per-block global scale and accumulate
        // chunk_output is row-major [M, N] = chunk_output[m * N + n]
        // out_acc is column-major [N, M] = out_acc[n * M + m]
        float block_gs = w_gs[b] * i_gs[b];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                out_acc[(size_t)n * M + m] += (float)chunk_output[(size_t)m * N + n] * block_gs;
            }
        }
    }

    // Write final output (ggml output is [N, M] column-major)
    float * dp = (float *) get_tensor_data_mut(dst);
    if (dp) {
        for (size_t i = 0; i < out_acc.size(); i++) dp[i] = out_acc[i];
    } else {
        ggml_backend_tensor_set(dst, out_acc.data(), 0, out_acc.size() * sizeof(float));
    }

    return true;
}

// Phase 40C: Q8_0 full-K GEMM using C4_DEVICE_SCALE kernel.
// The host extracts int8 + per-block scales from Q8_0, packs them tile-major,
// then issues ONE xrt::run.start() covering the whole K. Device applies scales.
static bool ggml_xdna_submit_q8_c4_ds(ggml_xdna_backend_context * ctx,
                                      XDNACompiledKernel * compiled_kernel,
                                      const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1,
                                      struct ggml_tensor * dst) {
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (K % 32 != 0 || M < 64) return false;

    if (!compiled_kernel || !compiled_kernel->q8_c4_ds_engine) return false;
    ggml_xdna_q8_c4_ds_engine * e = compiled_kernel->q8_c4_ds_engine;

    int pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) if (s.q8_c4_ds_engine == e) { pending_count++; }
    }
    if (!ctx->persistent_mode) {
        if (pending_count >= 4 && !ggml_xdna_drain_pending(ctx)) return false;
    }

    const int M_TILE = 64, N_TILE = 64, KC = 256, NBLK = KC / 32;
    int nb = K / 32;
    int n_kc = K / KC;
    int n_mt = M / M_TILE;
    int n_nt = N / N_TILE;

    int a_tile = M_TILE * KC + M_TILE * NBLK * 4;
    int b_tile = KC * N_TILE + NBLK * N_TILE * 4;

    // weight Q8_0 [K,N] -> int8 [K,N] row-major + scale [nb,N]
    const uint8_t * w_q8_ptr = (const uint8_t *) get_tensor_data(src0);
    std::vector<uint8_t> w_q8_local;
    if (!w_q8_ptr) {
        w_q8_local.resize(ggml_nbytes(src0));
        ggml_backend_tensor_get(src0, w_q8_local.data(), 0, w_q8_local.size());
        w_q8_ptr = w_q8_local.data();
    }
    std::vector<int8_t> weight_i8((size_t)K * N);
    std::vector<float> w_scale((size_t)nb * N);
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * block = &w_q8_ptr[((size_t)n * nb + b) * 34];
            w_scale[(size_t)b * N + n] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)block);
            for (int kk = 0; kk < 32; kk++)
                weight_i8[(size_t)(b * 32 + kk) * N + n] = (int8_t)block[2 + kk];
        }

    // input -> int8 [M,K] row-major + scale [M,nb]
    std::vector<int8_t> input_i8;
    std::vector<float> i_scale;
    const float * in_f32_ptr = (const float *) get_tensor_data(src1);
    const uint8_t * in_q8_ptr = (const uint8_t *) get_tensor_data(src1);
    std::vector<float> in_f32_local;
    std::vector<uint8_t> in_q8_local;
    if (src1->type == GGML_TYPE_F32) {
        if (!in_f32_ptr) {
            in_f32_local.resize(ggml_nbytes(src1) / sizeof(float));
            ggml_backend_tensor_get(src1, in_f32_local.data(), 0, in_f32_local.size() * sizeof(float));
            in_f32_ptr = in_f32_local.data();
        }
        input_i8.resize((size_t)K * M);
        i_scale.resize((size_t)M * nb);
        quantize_f32_to_i8_blocked(in_f32_ptr, M, K, input_i8, i_scale);
    } else if (src1->type == GGML_TYPE_Q8_0) {
        if (!in_q8_ptr) {
            in_q8_local.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, in_q8_local.data(), 0, in_q8_local.size());
            in_q8_ptr = in_q8_local.data();
        }
        input_i8.resize((size_t)K * M);
        i_scale.resize((size_t)M * nb);
        for (int m = 0; m < M; m++)
            for (int b = 0; b < nb; b++) {
                const uint8_t * block = &in_q8_ptr[((size_t)m * nb + b) * 34];
                i_scale[(size_t)m * nb + b] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)block);
                for (int kk = 0; kk < 32; kk++)
                    input_i8[(size_t)m * K + b * 32 + kk] = (int8_t)block[2 + kk];
            }
    } else {
        return false;
    }

    // pack A: (n_mt * n_kc) tiles of [M_TILE*KC int8][M_TILE*NBLK f32]
    std::vector<int8_t> Ap((size_t)n_mt * n_kc * a_tile);
    for (int mt = 0; mt < n_mt; mt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Ap[(size_t)(mt * n_kc + cc) * a_tile];
            for (int im = 0; im < M_TILE/4; im++)
                for (int bb = 0; bb < NBLK; bb++)
                    for (int kk = 0; kk < 4; kk++)
                        for (int rr = 0; rr < 4; rr++)
                            for (int cc_col = 0; cc_col < 8; cc_col++) {
                                size_t flat_idx = ((im * NBLK + bb) * 4 + kk) * 32 + rr * 8 + cc_col;
                                size_t src_idx = (mt * M_TILE + im * 4 + rr) * K + cc * KC + bb * 32 + kk * 8 + cc_col;
                                t[flat_idx] = input_i8[src_idx];
                            }
            float * ts = (float *)(t + M_TILE * KC);
            for (int i = 0; i < M_TILE; i++)
                for (int bb = 0; bb < NBLK; bb++)
                    ts[i * NBLK + bb] = i_scale[(size_t)(mt * M_TILE + i) * nb + cc * NBLK + bb];
        }

    // pack B: (n_nt * n_kc) tiles of [KC*N_TILE int8][NBLK*N_TILE f32]
    std::vector<int8_t> Bp((size_t)n_nt * n_kc * b_tile);
    for (int nt = 0; nt < n_nt; nt++)
        for (int cc = 0; cc < n_kc; cc++) {
            int8_t * t = &Bp[(size_t)(nt * n_kc + cc) * b_tile];
            for (int jn = 0; jn < N_TILE/8; jn++)
                for (int bb = 0; bb < NBLK; bb++)
                    for (int kk = 0; kk < 4; kk++)
                        for (int rr = 0; rr < 8; rr++)
                            for (int cc_col = 0; cc_col < 8; cc_col++) {
                                size_t flat_idx = ((jn * NBLK + bb) * 4 + kk) * 64 + rr * 8 + cc_col;
                                size_t src_idx = (cc * KC + bb * 32 + kk * 8 + rr) * N + nt * N_TILE + jn * 8 + cc_col;
                                t[flat_idx] = weight_i8[src_idx];
                            }
            float * ts = (float *)(t + KC * N_TILE);
            for (int bb = 0; bb < NBLK; bb++)
                for (int j = 0; j < N_TILE; j++)
                    ts[bb * N_TILE + j] = w_scale[(size_t)(cc * NBLK + bb) * N + nt * N_TILE + j];
        }

    xrt::run * run = nullptr;
    int used_buf = e->current_buffer;
    if (!ggml_xdna_q8_c4_ds_submit(e, Ap.data(), Bp.data(), &run)) return false;

    int buf_idx = used_buf;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub submission = {};
    submission.q8_c4_ds_engine = e;
    submission.run = run;
    submission.dst = dst;
    submission.M = M;
    submission.N = N;
    submission.K = K;
    submission.buf_idx = buf_idx;
    ctx->pending.push_back(submission);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 22: whole-array fast path. The C1 scale-aware kernel runs one ffn_gate

static const char * ggml_xdna_wa_cache_dir(void) {
    const char * env = std::getenv("XDNA_KERNEL_CACHE");
    if (env && *env) return env;
    return "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache";
}

static ggml_xdna_wa_engine * ggml_xdna_wa_get(int M, int K, int N) {
    uint64_t key = ((uint64_t) M << 42) ^ ((uint64_t) K << 21) ^ (uint64_t) N;
    std::lock_guard<std::mutex> lock(g_wa_mutex);
    auto it = g_wa_engines.find(key);
    if (it != g_wa_engines.end()) return it->second;

    char xc[512], ins[512];
    snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_WA/design.xclbin",
             ggml_xdna_wa_cache_dir(), M, K, N);
    snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_WA/design.insts.bin",
             ggml_xdna_wa_cache_dir(), M, K, N);
    ggml_xdna_wa_engine * e = ggml_xdna_wa_create(M, K, N, xc, ins);
    if (e) g_wa_engines[key] = e;
    return e;
}

// Phase 22: weight residency. Q8_0 -> int8 repack (with per-block scale folded
// in, then globally re-quantized) is O(K*N) and was redone every op. Cache it,
// keyed by the weight's data pointer and the GPU column split.
struct ggml_xdna_wa_weight {
    int K = 0, N = 0, n_gpu = 0;
    float gs = 1.0f;
    std::vector<int8_t> B;  // [K, n1] row-major, n1 = N - n_gpu
};

static std::mutex g_wa_wcache_mutex;
static std::unordered_map<uint64_t, ggml_xdna_wa_weight> g_wa_wcache;

static ggml_xdna_wa_weight ggml_xdna_wa_pack_weight(const struct ggml_tensor * src0, int n_gpu) {
    int K = src0->ne[0], N = src0->ne[1];
    int n1 = N - n_gpu;
    uint64_t key = (uint64_t)(uintptr_t) src0->data ^ ((uint64_t)(n_gpu + 1) << 40);
    {
        std::lock_guard<std::mutex> lock(g_wa_wcache_mutex);
        auto it = g_wa_wcache.find(key);
        if (it != g_wa_wcache.end()) return it->second;
    }

    std::vector<uint8_t> w(ggml_nbytes(src0));
    ggml_backend_tensor_get(src0, w.data(), 0, w.size());
    int nb = K / 32;

    float amax = 1e-9f;
    for (int n = 0; n < n1; n++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * blk = &w[((size_t)(n_gpu + n) * nb + b) * 34];
            float s = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
            for (int kk = 0; kk < 32; kk++) amax = std::max(amax, std::fabs(s * (int8_t) blk[2 + kk]));
        }
    float gs = amax / 127.0f;

    ggml_xdna_wa_weight e;
    e.K = K; e.N = N; e.n_gpu = n_gpu; e.gs = gs;
    e.B.assign((size_t) K * n1, 0);
    for (int n = 0; n < n1; n++)
        for (int b = 0; b < nb; b++) {
            const uint8_t * blk = &w[((size_t)(n_gpu + n) * nb + b) * 34];
            float s = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
            for (int kk = 0; kk < 32; kk++) {
                int q = (int) std::lroundf((s * (int8_t) blk[2 + kk]) / gs);
                q = std::max(-127, std::min(127, q));
                e.B[(size_t)(b * 32 + kk) * n1 + n] = (int8_t) q;
            }
        }

    {
        std::lock_guard<std::mutex> lock(g_wa_wcache_mutex);
        g_wa_wcache[key] = e;
    }
    return e;
}

// A[M,K] i8 x B[K,N] i8 -> C[M,N] i32, then scale to dst (F32 [N,M] layout).
static bool ggml_xdna_wa_execute(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 struct ggml_tensor * dst) {
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (M != 512 || K != 1024) return false;

    bool p38c = std::getenv("XDNA_P38C") != nullptr;
    double t_total0 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_eng0 = p38c ? ggml_xdna_now_ms() : 0.0;
    ggml_xdna_wa_engine * e = ggml_xdna_wa_get(M, K, N);
    if (!e) return false;
    double t_eng1 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_wpack0 = p38c ? ggml_xdna_now_ms() : 0.0;
    ggml_xdna_wa_weight W = ggml_xdna_wa_pack_weight(src0, 0);
    double t_wpack1 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_ain0 = p38c ? ggml_xdna_now_ms() : 0.0;
    std::vector<float> a((size_t) K * M);
    ggml_backend_tensor_get(src1, a.data(), 0, a.size() * sizeof(float));
    float amax = 1e-9f;
    for (float v : a) amax = std::max(amax, std::fabs(v));
    float as = amax / 127.0f;
    std::vector<int8_t> A((size_t) K * M);
    for (size_t i = 0; i < a.size(); i++) {
        int q = (int) std::lroundf(a[i] / as);
        q = std::max(-127, std::min(127, q));
        A[i] = (int8_t) q;
    }
    double t_ain1 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_submit0 = p38c ? ggml_xdna_now_ms() : 0.0;
    xrt::run * run = nullptr;
    if (!ggml_xdna_wa_submit(e, A.data(), W.B.data(), &run)) return false;
    double t_submit1 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_wait0 = p38c ? ggml_xdna_now_ms() : 0.0;
    std::vector<int32_t> C((size_t) M * N);
    if (!ggml_xdna_wa_wait(e, run, C.data())) {
        return false;
    }
    double t_wait1 = p38c ? ggml_xdna_now_ms() : 0.0;

    double t_out0 = p38c ? ggml_xdna_now_ms() : 0.0;
    std::vector<float> out(C.size());
    float scale = as * W.gs;
    for (size_t i = 0; i < C.size(); i++) out[i] = (float) C[i] * scale;
    ggml_backend_tensor_set(dst, out.data(), 0, out.size() * sizeof(float));
    double t_out1 = p38c ? ggml_xdna_now_ms() : 0.0;

    if (p38c) {
        fprintf(stderr,
                "[XDNA-WA-P38C] EXECUTE M=%d K=%d N=%d "
                "engine=%.2fms w_pack=%.2fms a_fetchq=%.2fms "
                "xrt_submit=%.2fms xrt_wait=%.2fms out_copy=%.2fms total=%.2fms "
                "a_bo=%p b_bo=%p c_bo=%p\n",
                M, K, N,
                t_eng1 - t_eng0, t_wpack1 - t_wpack0, t_ain1 - t_ain0,
                t_submit1 - t_submit0, t_wait1 - t_wait0, t_out1 - t_out0,
                t_out1 - t_total0,
                (void*)e->a_bo, (void*)e->b_bo, (void*)e->c_bo);
    }
    return true;
}

// Async whole-op: pack A, submit, queue the run; the merge into dst happens in
// drain_pending (synchronize / event_wait), so CPU packing overlaps GPU work.
static bool ggml_xdna_wa_submit_pending(ggml_xdna_backend_context * ctx,
                                        const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1,
                                        struct ggml_tensor * dst) {
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (M != 512 || K != 1024 || ggml_xdna_session_is_fatal()) return false;
    ggml_xdna_wa_engine * e = ggml_xdna_wa_get(M, K, N);
    if (!e) return false;

    bool need_flush = false;
    {
        std::lock_guard<std::mutex> lock(ctx->pending_mutex);
        for (auto & s : ctx->pending) {
            if (s.wa_engine == e) {
                need_flush = true;
                break;
            }
        }
    }
    if (need_flush) {
        if (!ggml_xdna_drain_pending(ctx)) return false;
        if (ggml_xdna_session_is_fatal()) return false;
    }


    ggml_xdna_wa_weight W = ggml_xdna_wa_pack_weight(src0, 0);

    std::vector<float> a((size_t) K * M);
    ggml_backend_tensor_get(src1, a.data(), 0, a.size() * sizeof(float));
    float amax = 1e-9f;
    for (float v : a) amax = std::max(amax, std::fabs(v));
    float as = amax / 127.0f;
    std::vector<int8_t> A((size_t) K * M);
    for (size_t i = 0; i < a.size(); i++) {
        int q = (int) std::lroundf(a[i] / as);
        q = std::max(-127, std::min(127, q));
        A[i] = (int8_t) q;
    }

    xrt::run * run = nullptr;
    if (!ggml_xdna_wa_submit(e, A.data(), W.B.data(), &run)) return false;

    std::lock_guard<std::mutex> lock(ctx->pending_mutex);
    ggml_xdna_pending_sub s = {};
    s.wa_engine = e;
    s.run = run;
    s.dst = dst;
    s.M = M;
    s.N = N;
    s.wa_scale = as * W.gs;
    ctx->pending.push_back(s);
    ctx->submitted_seq.fetch_add(1);
    return true;
}

// Phase 22: N-split. NPU computes cols [n_gpu, N) on the whole-array kernel while
// the in-process Vulkan backend computes cols [0, n_gpu) concurrently.
static ggml_backend_t g_vk_backend = nullptr;

static ggml_backend_t ggml_xdna_get_vk(void) {
    if (!g_vk_backend) g_vk_backend = ggml_backend_init_by_name("Vulkan0", "");
    if (!g_vk_backend) g_vk_backend = ggml_backend_init_by_name("Vulkan", "");
    return g_vk_backend;
}

// Phase 27: the Vulkan backend is dlopen'd with RTLD_LOCAL, so its symbols are
// not reachable via dlsym(RTLD_DEFAULT). Locate the loaded library instead.
static void * g_vk_lib_handle = nullptr;

static int ggml_xdna_find_vk_lib(struct dl_phdr_info * info, size_t, void *) {
    if (info->dlpi_name && strstr(info->dlpi_name, "ggml-vulkan")) {
        g_vk_lib_handle = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD);
        return g_vk_lib_handle ? 1 : 0;
    }
    return 0;
}

static void * ggml_xdna_vk_sym(const char * name) {
    if (!g_vk_lib_handle) {
        dl_iterate_phdr(ggml_xdna_find_vk_lib, nullptr);
    }
    return g_vk_lib_handle ? dlsym(g_vk_lib_handle, name) : nullptr;
}


static bool ggml_xdna_wa_execute_split(const struct ggml_tensor * src0,
                                       const struct ggml_tensor * src1,
                                       struct ggml_tensor * dst, int n_gpu) {
    int K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (M != 512 || K != 1024) return false;
    int n1 = N - n_gpu;
    if (n1 <= 0) return false;

    ggml_xdna_wa_engine * e = ggml_xdna_wa_get(M, K, n1);
    ggml_backend_t vk = ggml_xdna_get_vk();
    if (!e || !vk) return false;

    bool timing = std::getenv("XDNA_WA_TIMING") != nullptr;
    uint64_t t0 = ggml_time_ns();
    std::vector<float> a((size_t) K * M);
    ggml_backend_tensor_get(src1, a.data(), 0, a.size() * sizeof(float));
    uint64_t t1 = ggml_time_ns();
    int nb = K / 32;
    ggml_xdna_wa_weight W = ggml_xdna_wa_pack_weight(src0, n_gpu);
    uint64_t t2 = ggml_time_ns();

    // Vulkan weight slice: the first n_gpu columns are contiguous Q8_0 blocks.
    size_t w0_bytes = (size_t) n_gpu * nb * 34;
    std::vector<uint8_t> w0(w0_bytes);
    if (w0_bytes) ggml_backend_tensor_get(src0, w0.data(), 0, w0_bytes);
    uint64_t t3 = ggml_time_ns();

    float amax = 1e-9f;
    for (float v : a) amax = std::max(amax, std::fabs(v));
    float as = amax / 127.0f;
    std::vector<int8_t> Ai((size_t) K * M);
    for (size_t i = 0; i < a.size(); i++) {
        int q = (int) std::lroundf(a[i] / as);
        q = std::max(-127, std::min(127, q));
        Ai[i] = (int8_t) q;
    }
    uint64_t t4 = ggml_time_ns();

    xrt::run * run = nullptr;
    if (!ggml_xdna_wa_submit(e, Ai.data(), W.B.data(), &run)) return false;  // NPU starts async
    uint64_t t5 = ggml_time_ns();

    // Vulkan slice [0, n_gpu) while the NPU runs.
    std::vector<float> c0;
    if (n_gpu > 0) {
        uint64_t v0 = ggml_time_ns();
        struct ggml_init_params ip{0};
        ip.mem_size = w0_bytes + (size_t) K * M * 4 + 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, n_gpu);
        struct ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * ct = ggml_mul_mat(ctx, wt, bt);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ct);
        ggml_backend_buffer_t buf = nullptr;
        // Phase 27: optionally place the Vulkan slice output in host-imported
        // system memory, so the readback below becomes a plain host read.
        const char * sysmem_out = std::getenv("XDNA_WA_SYSMEM_OUT");
        ggml_backend_buffer_type_t hb = nullptr;
        if (sysmem_out && *sysmem_out) {
            typedef ggml_backend_buffer_type_t (*host_buft_fn)(size_t);
            static host_buft_fn host_buft = (host_buft_fn) ggml_xdna_vk_sym("ggml_backend_vk_host_output_buffer_type");
            if (host_buft) hb = host_buft(0);
            if (timing) {
                fprintf(stderr, "[XDNA-WA-SYSMEM] sym=%p buft=%p\n", (void *) host_buft, (void *) hb);
            }
        }
        if (hb) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, hb);
        } else {
            buf = ggml_backend_alloc_ctx_tensors(ctx, vk);
        }
        uint64_t v1 = ggml_time_ns();
        ggml_backend_tensor_set(wt, w0.data(), 0, w0.size());
        ggml_backend_tensor_set(bt, a.data(), 0, a.size() * sizeof(float));
        uint64_t v2 = ggml_time_ns();
        ggml_backend_graph_compute(vk, gf);
        uint64_t v3 = ggml_time_ns();
        c0.resize((size_t) n_gpu * M);
        ggml_backend_tensor_get(ct, c0.data(), 0, c0.size() * sizeof(float));
        uint64_t v4 = ggml_time_ns();
        // Phase 25 probe: second read of the same throwaway mini-graph buffer
        if (timing) {
            std::vector<float> c0b(c0.size());
            uint64_t v4b = ggml_time_ns();
            ggml_backend_tensor_get(ct, c0b.data(), 0, c0b.size() * sizeof(float));
            uint64_t v4c = ggml_time_ns();
            fprintf(stderr, "[XDNA-WA-VK2] get1=%.2f get2=%.2f\n", (v4 - v3) / 1e6, (v4c - v4b) / 1e6);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        uint64_t v5 = ggml_time_ns();
        if (timing) {
            fprintf(stderr, "[XDNA-WA-VK] alloc=%.2f set=%.2f compute=%.2f get=%.2f free=%.2f\n",
                    (v1 - v0) / 1e6, (v2 - v1) / 1e6, (v3 - v2) / 1e6,
                    (v4 - v3) / 1e6, (v5 - v4) / 1e6);
        }
    }
    uint64_t t6 = ggml_time_ns();

    std::vector<int32_t> Cn((size_t) M * n1);
    if (!ggml_xdna_wa_wait(e, run, Cn.data())) {
        return false;
    }
    uint64_t t7 = ggml_time_ns();

    // dst is [N, M]: element (n, m) lives at n + m*N.
    std::vector<float> out((size_t) N * M);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < n_gpu; n++) out[(size_t) n + (size_t) m * N] = c0[(size_t) n + (size_t) m * n_gpu];
        for (int n = 0; n < n1; n++)    out[(size_t)(n_gpu + n) + (size_t) m * N] = (float) Cn[(size_t) m * n1 + n] * as * W.gs;
    }
    uint64_t t8 = ggml_time_ns();
    ggml_backend_tensor_set(dst, out.data(), 0, out.size() * sizeof(float));
    uint64_t t9 = ggml_time_ns();

    if (timing) {
        fprintf(stderr, "[XDNA-WA-SPLIT] n_gpu=%d get_a=%.2f pack_w=%.2f get_w0=%.2f quant=%.2f "
                        "submit=%.2f vk=%.2f wait=%.2f merge=%.2f set=%.2f total=%.2f\n",
                n_gpu, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t4 - t3) / 1e6,
                (t5 - t4) / 1e6, (t6 - t5) / 1e6, (t7 - t6) / 1e6, (t8 - t7) / 1e6,
                (t9 - t8) / 1e6, (t9 - t0) / 1e6);
    }
    return true;
}

static const char* ggml_xdna_backend_get_name(ggml_backend_t backend) {
    (void)backend;
    return GGML_XDNA_BACKEND_NAME;
}

static void ggml_xdna_backend_free(ggml_backend_t backend) {
    fprintf(stderr, "[XDNA] backend_free called\n");
    fflush(stderr);
    struct ggml_xdna_backend_context* ctx = static_cast<ggml_xdna_backend_context*>(backend->context);
    if (ctx) {
        {
            std::lock_guard<std::mutex> lock(g_xdna_active_backends_mutex);
            auto it = std::find(g_xdna_active_backends.begin(), g_xdna_active_backends.end(), ctx);
            if (it != g_xdna_active_backends.end()) {
                g_xdna_active_backends.erase(it);
            }
            (void) ggml_xdna_drain_pending(ctx);
        }
        ctx->persistent_mode = false;
        if (ctx->cpu_backend) {
            ggml_backend_free(ctx->cpu_backend);
            ctx->cpu_backend = nullptr;
        }
        delete ctx;
    }
    fprintf(stderr, "[XDNA] XDNA ops executed=%d missing_kernel=%d\n",
            g_xdna_executed_ops.load(), g_xdna_missing_kernel_ops.load());
    ggml_xdna_print_timings();
    fprintf(stderr, "[XDNA] backend_free done\n");
    fflush(stderr);
    delete backend;
}

static void ggml_xdna_backend_synchronize(ggml_backend_t backend) {
    ggml_xdna_backend_context * ctx = (ggml_xdna_backend_context *) backend->context;
    if (ctx) {
        (void) ggml_xdna_drain_pending(ctx);
    }
    // XDNA uses CPU buffer type, so tensors are CPU buffers managed by CPU backend.
    // After NPU execution, synchronize the CPU backend to ensure all CPU buffer
    // modifications are visible to subsequent operations on other backends.
    // The CPU backend is cached: init/free per call dominates otherwise.
    if (ctx && ctx->cpu_backend == nullptr) {
        ctx->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (ctx && ctx->cpu_backend) {
        ggml_backend_synchronize(ctx->cpu_backend);
    }
}

static enum ggml_status ggml_xdna_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    ggml_xdna_backend_context * ctx = (ggml_xdna_backend_context *) backend->context;
    if (ggml_xdna_session_is_fatal()) {
        return GGML_STATUS_ABORTED;
    }

    static int g_xdna_graph_count = 0;
    int graph_id = g_xdna_graph_count++;
    const bool profile = ggml_xdna_gemm_profile_enabled();
    enum ggml_status status = GGML_STATUS_SUCCESS;

    fprintf(stderr, "[XDNA-TRACE] graph_compute START id=%d n_nodes=%d\n", graph_id, cgraph->n_nodes);
    fflush(stderr);

    ggml_xdna_stats_reset();

    XDNAKernelRegistry& registry = XDNAKernelRegistry::instance();

    // Phase 44: enable persistent mode for batched FFN execution.
    // Count MUL_MAT nodes to detect FFN regions.
    int mul_mat_count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) mul_mat_count++;
    }
    if (mul_mat_count > 1) {
        ctx->persistent_mode = true;
        fprintf(stderr, "[XDNA-TRACE] graph=%d persistent_mode ON (mul_mat_count=%d)\n", graph_id, mul_mat_count);
        fflush(stderr);
    }
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        
        if (node->op == GGML_OP_MUL_MAT && node->name && strstr(node->name, "ffn_gate")) {
            if (i + 3 < cgraph->n_nodes) {
                struct ggml_tensor* up_node = cgraph->nodes[i+1];
                struct ggml_tensor* glu_node = cgraph->nodes[i+2];
                struct ggml_tensor* down_node = cgraph->nodes[i+3];
                if (up_node->op == GGML_OP_MUL_MAT && up_node->name && strstr(up_node->name, "ffn_up") &&
                    glu_node->op == GGML_OP_GLU &&
                    down_node->op == GGML_OP_MUL_MAT && down_node->name && strstr(down_node->name, "ffn_down") &&
                    node->src[1] == up_node->src[1]) {
                    // Fused FFN
                    struct ggml_tensor* activation = node->src[1];
                    struct ggml_tensor* w_gate = node->src[0];
                    struct ggml_tensor* w_up = up_node->src[0];
                    struct ggml_tensor* w_down = down_node->src[0];
                    struct ggml_tensor* gate_out = node;
                    struct ggml_tensor* up_out = up_node;
                    struct ggml_tensor* glu_out = glu_node;
                    struct ggml_tensor* down_out = down_node;
                    
                    XDNAKernelKey key_gate = tensor_to_kernel_key(node);
                    XDNACompiledKernel* ck_gate = registry.get_or_create_kernel(key_gate);
                    if (ck_gate && w_gate->type == GGML_TYPE_Q8_0) {
                        ggml_xdna_submit_int8_pure(ctx, ck_gate, w_gate, activation, gate_out);
                    }
                    XDNAKernelKey key_up = tensor_to_kernel_key(up_node);
                    XDNACompiledKernel* ck_up = registry.get_or_create_kernel(key_up);
                    if (ck_up && w_up->type == GGML_TYPE_Q8_0) {
                        ggml_xdna_submit_int8_pure(ctx, ck_up, w_up, activation, up_out);
                    }
                    if (!ggml_xdna_drain_pending(ctx)) {
                        status = GGML_STATUS_ABORTED;
                        goto cleanup;
                    }
                    {
                        float* g = (float*)get_tensor_data_mut(gate_out);
                        float* u = (float*)get_tensor_data_mut(up_out);
                        float* h = (float*)get_tensor_data_mut(glu_out);
                        if (g && u && h) {
                            size_t n = ggml_nelements(glu_out);
                            for (size_t j = 0; j < n; j++) {
                                float x = g[j];
                                h[j] = (x / (1.0f + expf(-x))) * u[j];
                            }
                        }
                    }
                    XDNAKernelKey key_down = tensor_to_kernel_key(down_node);
                    XDNACompiledKernel* ck_down = registry.get_or_create_kernel(key_down);
                    if (ck_down && w_down->type == GGML_TYPE_Q8_0) {
                        ggml_xdna_submit_int8_pure(ctx, ck_down, w_down, glu_out, down_out);
                    }
                    if (!ggml_xdna_drain_pending(ctx)) {
                        status = GGML_STATUS_ABORTED;
                        goto cleanup;
                    }
                    g_xdna_executed_ops.fetch_add(3);
                    fprintf(stderr, "[XDNA-FUSED-FFN] layer fused\n");
                    fflush(stderr);
                    i += 3;
                    continue;
                }
            }
        }
        
        if (node->op == GGML_OP_GLU) {
            const uint64_t t0 = ggml_time_ns();
            // Phase 60: the GLU consumes the gate and up results, which are
            // still pending on the device. Drain first, otherwise the GLU
            // reads the uninitialised host buffers and produces zeros.
            if (!ggml_xdna_drain_pending(ctx)) {
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
            if (!ggml_xdna_glu_compute(node)) {
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
            g_xdna_glu_ops.fetch_add(1);
            if (ggml_profile_is_enabled()) {
                ggml_profile_event_t ev = {};
                ev.type = GGML_PROFILE_EVENT_OP_END;
                ev.timestamp_ns = ggml_time_ns();
                ev.backend_name = GGML_XDNA_BACKEND_NAME;
                ev.op_name = "GLU";
                ev.op_type = GGML_OP_GLU;
                ev.M = node->ne[1];
                ev.K = node->ne[0];
                ev.N = 0;
                ev.tensor_size = ggml_nbytes(node);
                ggml_profile_record_event(&ev);
            }
            fprintf(stderr, "[XDNA-GLU] node=%d nr=%lld nc=%lld ms=%.3f\n", i,
                    (long long) ggml_nrows(node), (long long) node->ne[0], (ggml_time_ns() - t0) / 1e6);
            continue;
        }
        // Phase 55: handle silu and mul on XDNA-resident tensors in-place,
        // keeping the entire FFN on XDNA without V<->X round-trips.
        if (node->op == GGML_OP_UNARY || node->op == GGML_OP_MUL) {
            // Only handle if all tensors are XDNA-resident
            struct ggml_tensor* dst = node;
            bool all_xdna = is_xdna_buffer(dst);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s] && !is_xdna_buffer(node->src[s])) { all_xdna = false; break; }
            }
            if (all_xdna) {
                if (node->op == GGML_OP_UNARY && node->op_params[0] == GGML_UNARY_OP_SILU) {
                    float* d = (float*)get_tensor_data_mut(dst);
                    const float* s = (const float*)get_tensor_data(node->src[0]);
                    size_t n = ggml_nelements(dst);
                    for (size_t i = 0; i < n; i++) {
                        float x = s[i];
                        d[i] = x / (1.0f + expf(-x));
                    }
                    g_xdna_executed_ops.fetch_add(1);
                    continue;
                } else if (node->op == GGML_OP_MUL) {
                    float* d = (float*)get_tensor_data_mut(dst);
                    const float* a = (const float*)get_tensor_data(node->src[0]);
                    const float* b = (const float*)get_tensor_data(node->src[1]);
                    size_t n = ggml_nelements(dst);
                    for (size_t i = 0; i < n; i++) d[i] = a[i] * b[i];
                    g_xdna_executed_ops.fetch_add(1);
                    continue;
                }
            }
            // Fall through to skip (Vulkan handles it)
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT) continue;
        
        struct ggml_tensor* src0 = node->src[0];
        struct ggml_tensor* src1 = node->src[1];
        struct ggml_tensor* dst = node;
        
        if (!src0 || !src1 || !dst) continue;

        // Phase 22: whole-array fast path for the ffn_gate shape.
        if (src0->type == GGML_TYPE_Q8_0) {
            const char * wa = std::getenv("XDNA_WA");
            if (wa && *wa && src0->ne[0] == 1024 && src0->ne[1] == 3584 && src1->ne[1] == 512) {
                const char * ng = std::getenv("XDNA_WA_NGPU");
                int n_gpu = (ng && *ng) ? atoi(ng) : 0;
                const char * ae = std::getenv("XDNA_WA_ASYNC");
                bool use_async = (ae && *ae) && n_gpu == 0;
                uint64_t t0 = ggml_time_ns();
                bool ok;
                if (n_gpu > 0) {
                    ok = ggml_xdna_wa_execute_split(src0, src1, dst, n_gpu);
                } else if (use_async) {
                    ok = ggml_xdna_wa_submit_pending(ctx, src0, src1, dst);
                } else {
                    ok = ggml_xdna_wa_execute(src0, src1, dst);
                }
                if (ok) {
                    g_xdna_executed_ops.fetch_add(1);
                    double ms = (ggml_time_ns() - t0) / 1e6;
                    fprintf(stderr, "[XDNA-WA] ffn_gate M=512 K=1024 N=3584 n_gpu=%d async=%d in %.2fms\n",
                            n_gpu, (int) use_async, ms);
                    fflush(stderr);
                    continue;
                }
            }
        }

        XDNAKernelKey key = tensor_to_kernel_key(node);
        XDNACompiledKernel* compiled_kernel = registry.get_or_create_kernel(key);
        
        if (!compiled_kernel) {
            g_xdna_missing_kernel_ops.fetch_add(1);
            ggml_xdna_log("[XDNA] No kernel available for op %d, falling back to CPU\n", node->op);
            return GGML_STATUS_ABORTED;
        }
        
        // Profiling
        uint64_t start_ns = ggml_time_ns();
        const char * engine_type = "unknown";

        // Phase 40K: XDNA_Q8_ENGINE env var overrides Q8 engine selection.
        // Values: auto (default), c4_ds, c4, c4_mmul_v2, scale, s12 (Phase 50)
        const char * q8_engine_env = getenv("XDNA_Q8_ENGINE");
        enum { Q8_ENGINE_AUTO, Q8_ENGINE_C4_DS, Q8_ENGINE_C4, Q8_ENGINE_C4_MMUL_V2, Q8_ENGINE_C4_MMUL_V3, Q8_ENGINE_SCALE, Q8_ENGINE_S12 } q8_engine = Q8_ENGINE_AUTO;
        if (q8_engine_env) {
            if (strcmp(q8_engine_env, "c4_ds") == 0) q8_engine = Q8_ENGINE_C4_DS;
            else if (strcmp(q8_engine_env, "c4_mmul_v2") == 0) q8_engine = Q8_ENGINE_C4_MMUL_V2;
            else if (strcmp(q8_engine_env, "c4_mmul_v3") == 0) q8_engine = Q8_ENGINE_C4_MMUL_V3;
            else if (strcmp(q8_engine_env, "c4") == 0) q8_engine = Q8_ENGINE_C4;
            else if (strcmp(q8_engine_env, "scale") == 0) q8_engine = Q8_ENGINE_SCALE;
            else if (strcmp(q8_engine_env, "s12") == 0) q8_engine = Q8_ENGINE_S12;
        }

        const char * req_env = getenv("XDNA_Q8_ENGINE");
        bool want_requant = req_env && strcasecmp(req_env, "int8_requant") == 0;
        bool want_int8_pure_v2 = req_env && strcasecmp(req_env, "int8_requant_v2") == 0;
        bool want_int8_pure_v3 = req_env && strcasecmp(req_env, "int8_requant_v3") == 0;
        bool want_int8_pure_v4 = req_env && strcasecmp(req_env, "int8_requant_v4") == 0;

        if ((want_int8_pure_v2 || want_int8_pure_v3 || want_int8_pure_v4) && src0->type == GGML_TYPE_Q8_0) {
            engine_type = want_int8_pure_v4 ? "int8-pure-v4" : (want_int8_pure_v3 ? "int8-pure-v3" : "int8-pure-v2");
            if (!ggml_xdna_submit_int8_pure(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (want_requant && src0->type == GGML_TYPE_Q8_0) {
            engine_type = "int8-requant";
            if (!ggml_xdna_submit_requant(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (src0->type == GGML_TYPE_I8) {
            engine_type = "persistent-i8";
            // I8 has no scales: submit the full K in one xrt::run.start(), no wait.
            if (!ggml_xdna_submit_i8_fullk(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (q8_engine == Q8_ENGINE_SCALE && compiled_kernel->q8_scale_engine) {
            engine_type = "q8scalev";
            // Forced Q8SCALEV
            if (!ggml_xdna_submit_q8_scalefullk(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if ((q8_engine == Q8_ENGINE_C4_MMUL_V2 || q8_engine == Q8_ENGINE_C4_MMUL_V3 ||
                   q8_engine == Q8_ENGINE_S12) && compiled_kernel->q8_c4_mmul_v2_engine) {
            engine_type = q8_engine == Q8_ENGINE_C4_MMUL_V3 ? "q8-c4-mmul-v3" : "q8-c4-mmul-v2";
            // Forced Q8-C4 mmul-v2 (s12 uses the same submit; packing/scatter branch on e->s12)
            if (!ggml_xdna_submit_q8_c4_mmul_v2(ctx, compiled_kernel, src0, src1, dst, profile)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
            // the layer/op classification feeds [XDNA-SUMMARY], so record it
            // whenever either trace is on
            if (profile || ggml_xdna_stats_enabled()) {
                std::lock_guard<std::mutex> lock(ctx->pending_mutex);
                ggml_xdna_gemm_record_metadata(ctx->pending.back(), node, src0, src1, compiled_kernel, key);
            }
        } else if (q8_engine == Q8_ENGINE_C4 && compiled_kernel->q8_scale_c4_engine) {
            engine_type = "q8scalev-c4";
            // Forced Q8SCALEV-C4
            if (!ggml_xdna_submit_q8_scalefullk_c4(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (q8_engine == Q8_ENGINE_C4_DS && compiled_kernel->q8_c4_ds_engine) {
            engine_type = "q8-c4-device-scale";
            // Forced C4 device-scale
            if (!ggml_xdna_submit_q8_c4_ds(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (compiled_kernel->q8_c4_mmul_v2_engine) {
            engine_type = (req_env && strcasecmp(req_env, "c4_mmul_v4") == 0) ? "q8-c4-mmul-v4" : "q8-c4-mmul-v2";
            // Q8-C4 mmul-v2: preferred C4 path (auto)
            if (!ggml_xdna_submit_q8_c4_mmul_v2(ctx, compiled_kernel, src0, src1, dst, profile)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
            // the layer/op classification feeds [XDNA-SUMMARY], so record it
            // whenever either trace is on
            if (profile || ggml_xdna_stats_enabled()) {
                std::lock_guard<std::mutex> lock(ctx->pending_mutex);
                ggml_xdna_gemm_record_metadata(ctx->pending.back(), node, src0, src1, compiled_kernel, key);
            }
        } else if (compiled_kernel->q8_scale_c4_engine) {
            engine_type = "q8scalev-c4";
            // Q8SCALEV-C4: multi-column scale-aware Q8_0 GEMM
            if (!ggml_xdna_submit_q8_scalefullk_c4(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (compiled_kernel->q8_c4_ds_engine) {
            engine_type = "q8-c4-device-scale";
            // Q8-C4: C4 mmul with device-side per-block scales.
            if (!ggml_xdna_submit_q8_c4_ds(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else if (compiled_kernel->q8_c4_engine) {
            engine_type = "q8-c4-wa";
            // Q8-C4: C4_WA mmul with host-side scale folding.
            if (!ggml_xdna_submit_q8_c4(ctx, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        } else {
            engine_type = "q8scalev";
            // Q8_0 scale-aware full-K: one submit, device applies per-block scales.
            if (!ggml_xdna_submit_q8_scalefullk(ctx, compiled_kernel, src0, src1, dst)) {
                g_xdna_missing_kernel_ops.fetch_add(1);
                status = GGML_STATUS_ABORTED;
                goto cleanup;
            }
        }
        g_xdna_executed_ops.fetch_add(1);

        uint64_t end_ns = ggml_time_ns();

        // Record profile event
        if (ggml_profile_is_enabled()) {
            ggml_profile_event_t event = {};
            event.type = GGML_PROFILE_EVENT_OP_END;
            event.timestamp_ns = end_ns;
            event.backend_name = GGML_XDNA_BACKEND_NAME;
            event.op_name = "MUL_MAT";
            event.op_type = GGML_OP_MUL_MAT;
            event.graph_index = i;
            event.layer = -1; // Will be filled by caller if needed
            event.M = src1 ? src1->ne[1] : 0;
            event.K = src0 ? src0->ne[0] : 0;
            event.N = src0 ? src0->ne[1] : 0;
            event.dtype_a = src0 ? src0->type : 0;
            event.dtype_b = src1 ? src1->type : 0;
            event.dtype_c = dst ? dst->type : 0;
            event.tensor_size = ggml_nbytes(dst);
            event.xdna_compatible = true;

            // Also record start event
            ggml_profile_event_t start_event = event;
            start_event.type = GGML_PROFILE_EVENT_OP_START;
            start_event.timestamp_ns = start_ns;
            ggml_profile_record_event(&start_event);

            ggml_profile_record_event(&event);
        }

        double ms = (end_ns - start_ns) / 1e6;
        fprintf(stderr, "[XDNA-TRACE] graph=%d node=%d op=MUL_MAT M=%d K=%d N=%d engine=%s time=%.2fms\n",
                graph_id, i, src1->ne[1], src0->ne[0], src0->ne[1], engine_type, ms);
        fflush(stderr);
    }

    fprintf(stderr, "[XDNA-TRACE] graph_compute END id=%d\n", graph_id);
    fflush(stderr);

cleanup:
    fprintf(stderr, "[XDNA-TRACE] graph=%d draining pending submissions (cleanup)\n", graph_id);
    fflush(stderr);
    if (!ggml_xdna_drain_pending(ctx)) {
        status = GGML_STATUS_ABORTED;
    }
    ctx->persistent_mode = false;
    ctx->persistent_pending_count = 0;
    ggml_xdna_print_counters();
    if (g_xdna_stats.enabled) {
        fprintf(stderr,
                "[XDNA-SUMMARY] graph=%d n_nodes=%d n_gemm=%d gate=%d up=%d down=%d other=%d "
                "M=%d K=%d N=%d shape_uniform=%d "
                "host_prep_ms=%.3f submit_ms=%.3f run_wait_ms=%.3f sync_from_ms=%.3f "
                "copyback_ms=%.3f npu_total_ms=%.3f gemm_wall_ms=%.3f\n",
                graph_id, cgraph->n_nodes,
                g_xdna_stats.n_gemm, g_xdna_stats.n_gate, g_xdna_stats.n_up, g_xdna_stats.n_down,
                g_xdna_stats.n_other, g_xdna_stats.M, g_xdna_stats.K, g_xdna_stats.N,
                g_xdna_stats.shape_uniform ? 1 : 0,
                g_xdna_stats.t_host_prep, g_xdna_stats.t_submit, g_xdna_stats.t_run_wait,
                g_xdna_stats.t_sync_from, g_xdna_stats.t_copyback,
                g_xdna_stats.t_total, g_xdna_stats.t_gemm);
        ggml_xdna_stats_reset();
    }
    return status;
}

static enum ggml_status ggml_xdna_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    return ggml_xdna_backend_graph_compute(backend, cgraph);
}

// Phase 20G step 8: record an event marker at the current submission point.
static void ggml_xdna_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_xdna_backend_context * ctx = (ggml_xdna_backend_context *) backend->context;
    ggml_xdna_event * ev = (ggml_xdna_event *) event->context;
    ev->ctx = ctx;
    ev->seq = ctx->submitted_seq.load();
}

// Phase 20G step 8: wait for an event recorded earlier on this single queue.
static void ggml_xdna_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    (void) backend;
    ggml_xdna_event * ev = (ggml_xdna_event *) event->context;
    if (ev->ctx) {
        ggml_xdna_drain_pending(ev->ctx);
    }
}

static const struct ggml_backend_i ggml_xdna_backend_i = {
    /* .get_name          = */ ggml_xdna_backend_get_name,
    /* .free              = */ ggml_xdna_backend_free,
    /* .set_tensor_async  = */ nullptr,
    /* .get_tensor_async  = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async  = */ nullptr,
    /* .synchronize       = */ ggml_xdna_backend_synchronize,
    /* .graph_plan_create = */ nullptr,
    /* .graph_plan_free   = */ nullptr,
    /* .graph_plan_update = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute     = */ ggml_xdna_backend_graph_compute,
    /* .event_record      = */ ggml_xdna_backend_event_record,
    /* .event_wait        = */ ggml_xdna_backend_event_wait,
    /* .graph_optimize    = */ nullptr,
};

static ggml_guid_t ggml_xdna_guid(void) {
    static ggml_guid guid = { 
        0xbd, 0x78, 0xd8, 0x54, 0xa7, 0xf7, 0xb4, 0x9b, 
        0xf4, 0xb0, 0xfb, 0xa3, 0x47, 0xcd, 0xfd, 0x9a 
    };
    return &guid;
}

static ggml_backend_t ggml_xdna_backend_init(void) {
    fprintf(stderr, "[XDNA] ggml_xdna_backend_init: starting\n");
    fflush(stderr);
    struct ggml_xdna_backend_context* ctx = new ggml_xdna_backend_context();
    {
        std::lock_guard<std::mutex> lock(g_xdna_active_backends_mutex);
        g_xdna_active_backends.push_back(ctx);
    }
    fprintf(stderr, "[XDNA] ggml_xdna_backend_init: context created\n");
    fflush(stderr);
    
    XDNAKernelRegistry& registry = XDNAKernelRegistry::instance();
    fprintf(stderr, "[XDNA] ggml_xdna_backend_init: registry obtained\n");
    fflush(stderr);

    const char * cache_dir_env = std::getenv("XDNA_KERNEL_CACHE");
    if (cache_dir_env && *cache_dir_env) {
        registry.set_cache_directory(cache_dir_env);
    }

    // Production kernels are auto-discovered from the kernel cache at lookup time.
    // compile_via_mlir_aie checks for cached artifacts (with and without tile suffix)
    // before attempting any compilation. Only whitelisted shapes (M in {512,448},
    // K in {1024,2048,4096}, N in validated set) are eligible.

    registry.print_registry_status();
    fprintf(stderr, "[XDNA] ggml_xdna_backend_init: registry printed\n");
    fflush(stderr);
    
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_xdna_guid(),
        /* .iface   = */ ggml_xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };
    
    if (!backend) {
        delete ctx;
        return nullptr;
    }
    
    // Initialize shared XRT device for buffer allocations
    //
    // Phase 62: a hung run can leave the NPU driver in a state where it no
    // longer accepts contexts, and then the device open blocks forever, which
    // also stops the GPU only runs because the backend is registered at
    // startup.  GGML_XDNA_DISABLE=1 keeps the backend registered but never
    // opens the device.
    {
        const char * disable_env = std::getenv("GGML_XDNA_DISABLE");
        if (disable_env && *disable_env && strcmp(disable_env, "0") != 0) {
            fprintf(stderr, "[XDNA] disabled by GGML_XDNA_DISABLE, the device is not opened\n");
        } else {
            std::lock_guard<std::mutex> lock(ctx->xrt_dev_mutex);
            if (!ctx->xrt_dev_initialized) {
                try {
                    ctx->xrt_dev = xrt::device(0);
                    ctx->xrt_dev_initialized = true;
                    fprintf(stderr, "[XDNA] Shared XRT device initialized for buffer type\n");
                } catch (const std::exception & e) {
                    fprintf(stderr, "[XDNA] WARNING: XRT device init failed: %s\n", e.what());
                    ggml_xdna_session_mark_fatal(e.what());
                }
            }
        }
    }
    
    fprintf(stderr, "[XDNA] Backend initialized with kernel registry\n");
    return backend;
}

GGML_BACKEND_API const char* ggml_xdna_device_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "AMD Phoenix XDNA1 NPU";
}

GGML_BACKEND_API const char* ggml_xdna_device_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "AMD XDNA1 NPU (Phoenix) with INT8 GEMM support via kernel registry";
}

// Phase 47B: llama.cpp places model weights on a device in proportion to its
// reported free memory. A large budget makes the loader put a slice of the
// layers in XDNA buffers, and every weight placed there is copied to Vulkan at
// graph-build time even when no op is routed to the NPU.
static bool xdna_npu_ops_enabled() {
    const char * e = std::getenv("GGML_XDNA_QWEN_PP");
    return e && *e;
}

// Phase 48: putting weights on the NPU needs a separate opt-in from routing ops
// to it. With GGML_XDNA_QWEN_PP=1 alone the loader puts 6 of 24 layers in NPU
// buffers, which measured over 4 identical runs: 72 splits, 38 transitions,
// 197 MB of copies and 12 of 72 FFN GEMMs reaching the NPU, for 87 tok/s
// against 1262 tok/s Vulkan-only. The 12 GEMMs that do run cost 426 ms each
// (8.8 GOP/s), so 5.1 s of the 5.8 s is the kernel, not the copies; but the
// placement is what routes those 12 in the first place. GGML_XDNA_WEIGHTS=1
// keeps that configuration reachable for measurement.
static bool xdna_npu_weights_enabled() {
    const char * e = std::getenv("GGML_XDNA_WEIGHTS");
    return e && *e;
}

GGML_BACKEND_API void ggml_xdna_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    (void)dev;
    const size_t budget = xdna_npu_weights_enabled() ? (1ull << 30) : (1ull << 20);
    if (free)  *free  = budget;
    if (total) *total = budget;
}

GGML_BACKEND_API enum ggml_backend_dev_type ggml_xdna_device_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

GGML_BACKEND_API void ggml_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    (void)dev;
    props->name = ggml_xdna_device_get_name(dev);
    props->description = ggml_xdna_device_get_description(dev);
    props->type = ggml_xdna_device_get_type(dev);
    ggml_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    // Phase 20G step 8: the XDNA backend submits asynchronously and exposes events.
    props->caps = {};
    props->caps.async = true;
    props->caps.events = true;
}

GGML_BACKEND_API ggml_backend_t ggml_xdna_device_init(ggml_backend_dev_t dev, const char* params) {
    fprintf(stderr, "[XDNA] ggml_xdna_device_init: called\n");
    (void)dev;
    (void)params;
    return ggml_xdna_backend_init();
}

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    struct ggml_xdna_device_context * ctx = (struct ggml_xdna_device_context *) dev->context;
    return &ctx->buffer_type;
}

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_xdna_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return ggml_backend_cpu_buffer_type();
}

GGML_BACKEND_API ggml_backend_buffer_t ggml_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void* ptr, size_t size, size_t max_tensor_size) {
    (void)dev;
    (void)max_tensor_size;
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

GGML_BACKEND_API bool ggml_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    (void)dev;
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    if (op->op == GGML_OP_GLU) {
        return ggml_xdna_glu_supported(op);
    }
    // Phase 55: support silu and mul on XDNA-resident tensors
    if (op->op == GGML_OP_UNARY && op->op_params[0] == GGML_UNARY_OP_SILU) {
        bool all_xdna = is_xdna_buffer(op);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (op->src[s] && !is_xdna_buffer(op->src[s])) { all_xdna = false; break; }
        }
        return all_xdna;
    }
    if (op->op == GGML_OP_MUL) {
        bool all_xdna = is_xdna_buffer(op);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (op->src[s] && !is_xdna_buffer(op->src[s])) { all_xdna = false; break; }
        }
        return all_xdna;
    }
    bool result = is_mul_mat_supported(op);
    if (op->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor* src0 = op->src[0];
        const struct ggml_tensor* src1 = op->src[1];
        fprintf(stderr, "[XDNA-SUPPORTS] op=MUL_MAT M=%d K=%d N=%d src0_type=%d src1_type=%d result=%d\n",
                src1 ? src1->ne[1] : 0,
                src0 ? src0->ne[0] : 0,
                src0 ? src0->ne[1] : 0,
                src0 ? (int)src0->type : -1,
                src1 ? (int)src1->type : -1,
                result);
        fflush(stderr);
    }
    return result;
}

GGML_BACKEND_API bool ggml_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    ggml_backend_buffer_type_t xdna_buft = ggml_xdna_device_get_buffer_type(dev);
    if (!xdna_buft) return false;
    const char * xdna_name = xdna_buft->iface.get_name(xdna_buft);
    const char * buf_name  = buft->iface.get_name(buft);
    return xdna_name && buf_name && strcmp(xdna_name, buf_name) == 0;
}

// Phase 49: the gated GLU. Gate and up are both produced on the NPU, so running
// this op anywhere else copies both results out and the down GEMM's input back
// in. Keeping it here is what lets one layer's whole FFN be a single split.
static bool ggml_xdna_glu_supported(const struct ggml_tensor * op) {
    if (const char * e = std::getenv("XDNA_GLU"); e && *e == '0') {
        return false;
    }
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    // split (separate gate and up tensors), f32, not the fused single-tensor form
    return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
}

static bool ggml_xdna_glu_compute_fallback(struct ggml_tensor * node);
static bool ggml_xdna_glu_compute(struct ggml_tensor * node) {
    const struct ggml_tensor * gate = node->src[0];
    const struct ggml_tensor * up   = node->src[1];
    const int64_t nc = node->ne[0];
    const int64_t nr = ggml_nrows(node);

    float * g = (float *) get_tensor_data_mut((struct ggml_tensor *) gate);
    float * u = (float *) get_tensor_data_mut((struct ggml_tensor *) up);
    float * d = (float *) get_tensor_data_mut(node);
    std::vector<float> g_local, u_local;
    if (!g) { g_local.resize(ggml_nelements(gate)); ggml_backend_tensor_get(gate, g_local.data(), 0, ggml_nbytes(gate)); g = g_local.data(); }
    if (!u) { u_local.resize(ggml_nelements(up));   ggml_backend_tensor_get(up,   u_local.data(), 0, ggml_nbytes(up));   u = u_local.data(); }
    if (!d) { return ggml_xdna_glu_compute_fallback(node); }

    for (int64_t r = 0; r < nr; r++) {
        const float * gp = g + r * nc;
        const float * up_p = u + r * nc;
        float * dp = d + r * nc;
        for (int64_t c = 0; c < nc; c++) {
            const float x = gp[c];
            dp[c] = (x / (1.0f + expf(-x))) * up_p[c];
        }
    }
    return true;
}

// same computation when the destination is not in an XDNA buffer
static bool ggml_xdna_glu_compute_fallback(struct ggml_tensor * node) {
    const struct ggml_tensor * gate = node->src[0];
    const struct ggml_tensor * up   = node->src[1];
    const int64_t nc = node->ne[0];
    const int64_t nr = ggml_nrows(node);
    const size_t n = (size_t) nc * nr;

    std::vector<float> g(n), u(n), d(n);
    ggml_backend_tensor_get(gate, g.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(up,   u.data(), 0, n * sizeof(float));
    for (int64_t r = 0; r < nr; r++) {
        for (int64_t c = 0; c < nc; c++) {
            const float x = g[r * nc + c];
            d[r * nc + c] = (x / (1.0f + expf(-x))) * u[r * nc + c];
        }
    }
    ggml_backend_tensor_set(node, d.data(), 0, n * sizeof(float));
    return true;
}

// Phase 49: which FFN GEMM this is, from the weight tensor name. Only these
// three ops are routed away from the backend that owns their weights; attention,
// rope, norms and sampling keep the default placement policy.
static const char * ggml_xdna_ffn_kind(const struct ggml_tensor * op) {
    const char * name = (op && op->src[0]) ? op->src[0]->name : "";
    if (strstr(name, "ffn_gate")) return "gate";
    if (strstr(name, "ffn_up"))   return "up";
    if (strstr(name, "ffn_down")) return "down";
    return nullptr;
}

GGML_BACKEND_API bool ggml_xdna_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    // offload_op answers "run this here even though its weights live in another
    // backend's buffer", which is what makes the routing explicit rather than a
    // side effect of weight placement. Restricted to the FFN GEMMs.
    (void) dev;
    if (ggml_xdna_session_is_fatal()) {
        return false;
    }
    return ggml_xdna_ffn_kind(op) != nullptr && is_mul_mat_supported(op);
}


GGML_BACKEND_API ggml_backend_event_t ggml_xdna_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_event_t event = new ggml_backend_event;
    event->device = dev;
    event->context = new ggml_xdna_event{ nullptr, 0 };
    return event;
}

GGML_BACKEND_API void ggml_xdna_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    delete (ggml_xdna_event *) event->context;
    delete event;
}

GGML_BACKEND_API void ggml_xdna_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    ggml_xdna_event * ev = (ggml_xdna_event *) event->context;
    if (ev->ctx) {
        ggml_xdna_drain_pending(ev->ctx);
    }
}

static struct ggml_backend_device_i ggml_xdna_device_i = {
    ggml_xdna_device_get_name,
    ggml_xdna_device_get_description,
    ggml_xdna_device_get_memory,
    ggml_xdna_device_get_type,
    ggml_xdna_device_get_props,
    ggml_xdna_device_init,
    ggml_xdna_device_get_buffer_type,
    ggml_xdna_device_get_host_buffer_type,
    ggml_xdna_device_buffer_from_host_ptr,
    ggml_xdna_device_supports_op,
    ggml_xdna_device_supports_buft,
    ggml_xdna_device_offload_op,
    ggml_xdna_device_event_new,
    ggml_xdna_device_event_free,
    ggml_xdna_device_event_synchronize,
};

static const char* ggml_xdna_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return GGML_XDNA_BACKEND_NAME;
}

static size_t ggml_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    const char* env = std::getenv("GGML_XDNA_DISABLE");
    if (env && *env != '\0') {
        return 0;
    }
    return 1;
}

static ggml_backend_dev_t ggml_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    const char* env = std::getenv("GGML_XDNA_DISABLE");
    if (env && *env != '\0') {
        return nullptr;
    }
    if (index != 0) return nullptr;
    
    static struct ggml_xdna_device_context ctx = {0};
    static struct ggml_backend_device dev = {
        ggml_xdna_device_i,
        nullptr,
        &ctx,
    };
    dev.reg = reg;
    
    // Initialize buffer type on first access
    static bool buffer_type_initialized = false;
    if (!buffer_type_initialized) {
        extern const struct ggml_backend_buffer_type_i * ggml_xdna_get_buffer_type_i(void);
        ctx.buffer_type.iface = *ggml_xdna_get_buffer_type_i();
        ctx.buffer_type.device = &dev;
        ctx.buffer_type.context = nullptr;
        buffer_type_initialized = true;
        fprintf(stderr, "[XDNA] Device buffer type initialized\n");
    }
    
    return &dev;
}

static struct ggml_backend_reg_i ggml_xdna_reg_i = {
    ggml_xdna_reg_get_name,
    ggml_xdna_reg_get_device_count,
    ggml_xdna_reg_get_device,
    nullptr,
};

GGML_BACKEND_API struct ggml_xdna_backend_context * ggml_xdna_get_backend_context(void) {
    static struct ggml_xdna_backend_context * ctx = nullptr;
    if (!ctx) {
        ggml_backend_t backend = ggml_xdna_backend_init();
        if (backend) {
            ctx = (struct ggml_xdna_backend_context *) backend->context;
        }
    }
    return ctx;
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    const char* env = std::getenv("GGML_XDNA_DISABLE");
    if (env && *env != '\0') {
        static struct ggml_backend_reg_i reg_i = {
            ggml_xdna_reg_get_name,
            ggml_xdna_reg_get_device_count,
            ggml_xdna_reg_get_device,
            nullptr,
        };
        static struct ggml_backend_reg reg = {
            GGML_BACKEND_API_VERSION,
            reg_i,
            nullptr,
        };
        return &reg;
    }
    
    static struct ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        ggml_xdna_reg_i,
        nullptr,
    };
    return &reg;
}

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(void) {
    return ggml_xdna_backend_init();
}

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_xdna_backend_get_name;
}

GGML_BACKEND_API ggml_backend_dev_t ggml_backend_xdna_reg_get_device(size_t index) {
    return ggml_xdna_reg_get_device(ggml_backend_xdna_reg(), index);
}

GGML_BACKEND_API size_t ggml_backend_xdna_reg_get_device_count(void) {
    return ggml_xdna_reg_get_device_count(ggml_backend_xdna_reg());
}

GGML_BACKEND_API void ggml_xdna_print_counters(void) {
    fprintf(stderr, "[XDNA] counters: persistent_execute=%lld xrt_submit=%lld xrt_complete=%lld\n",
            (long long)g_xdna_persistent_execute_count.load(),
            (long long)g_xdna_xrt_submit_count.load(),
            (long long)g_xdna_xrt_complete_count.load());
}

GGML_BACKEND_API struct ggml_xdna_counters ggml_xdna_get_counters(void) {
    struct ggml_xdna_counters c = {};
    c.persistent_execute = g_xdna_persistent_execute_count.load();
    c.xrt_submit = g_xdna_xrt_submit_count.load();
    c.xrt_complete = g_xdna_xrt_complete_count.load();
    return c;
}

GGML_BACKEND_API void ggml_xdna_print_counters_delta(const struct ggml_xdna_counters* prev, const struct ggml_xdna_counters* cur) {
    fprintf(stderr, "[XDNA] counters delta: persistent_execute=%lld xrt_submit=%lld xrt_complete=%lld\n",
            (long long)(cur->persistent_execute - prev->persistent_execute),
            (long long)(cur->xrt_submit - prev->xrt_submit),
            (long long)(cur->xrt_complete - prev->xrt_complete));
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)