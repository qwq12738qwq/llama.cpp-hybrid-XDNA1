#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

#include "xdna-persistent.h"
#include "xdna-q8-scale.h"
#include "xdna-q8-c4.h"
#include "xdna-q8-c4-device-scale.h"
#include "xdna-q8-scale-c4.h"

#include "xdna-q8-c4-mmul-v2.h"

#ifdef __cplusplus
#endif

enum class XDNAOpType {
    MUL_MAT,
    MUL_MAT_VEC
};

enum class XDNAQuantization {
    Q8_0,
    NONE
};

struct XDNAKernelKey {
    XDNAOpType op_type;
    XDNAQuantization quantization;
    int m_tile;
    int k_tile;
    int n_tile;
    int n_cols;  // Number of parallel columns for N-parallel
    int dtype_a;
    int dtype_b;
    int dtype_c;
    int M;  // Full M dimension
    int K;  // Full K dimension
    int N;  // Full N dimension

    bool operator==(const XDNAKernelKey& other) const {
        return op_type == other.op_type &&
               quantization == other.quantization &&
               m_tile == other.m_tile &&
               k_tile == other.k_tile &&
               n_tile == other.n_tile &&
               n_cols == other.n_cols &&
               dtype_a == other.dtype_a &&
               dtype_b == other.dtype_b &&
               dtype_c == other.dtype_c &&
               M == other.M &&
               K == other.K &&
               N == other.N;
    }
};

struct XDNAKernelKeyHash {
    size_t operator()(const XDNAKernelKey& key) const noexcept {
        size_t h = 0;
        h = h * 31 + static_cast<int>(key.op_type);
        h = h * 31 + static_cast<int>(key.quantization);
        h = h * 31 + key.m_tile;
        h = h * 31 + key.k_tile;
        h = h * 31 + key.n_tile;
        h = h * 31 + key.n_cols;
        h = h * 31 + key.dtype_a;
        h = h * 31 + key.dtype_b;
        h = h * 31 + key.dtype_c;
        h = h * 31 + key.M;
        h = h * 31 + key.K;
        h = h * 31 + key.N;
        return h;
    }
};

struct XDNAKernelSpec {
    XDNAKernelKey key;
    std::string xclbin_path;
    std::string inst_path;
    std::string pdi_path;
    bool compiled;
    bool compilation_in_progress;
    std::string compilation_error;
    
    XDNAKernelSpec() : compiled(false), compilation_in_progress(false) {}
};

struct XDNACompiledKernel {
    XDNAKernelSpec spec;
    ggml_xdna_persistent_engine* engine;
    ggml_xdna_q8_scale_engine* q8_scale_engine;
    ggml_xdna_q8_c4_engine* q8_c4_engine;
    ggml_xdna_q8_c4_ds_engine* q8_c4_ds_engine;
    ggml_xdna_q8_scale_c4_engine* q8_scale_c4_engine;
    ggml_xdna_q8_c4_mmul_v2_engine* q8_c4_mmul_v2_engine;

    XDNACompiledKernel() : engine(nullptr), q8_scale_engine(nullptr), q8_c4_engine(nullptr), q8_c4_ds_engine(nullptr), q8_scale_c4_engine(nullptr), q8_c4_mmul_v2_engine(nullptr) {}
};

class XDNAKernelRegistry {
public:
    static XDNAKernelRegistry& instance() {
        static XDNAKernelRegistry registry;
        return registry;
    }

    XDNACompiledKernel* get_or_create_kernel(const XDNAKernelKey& key);
    XDNACompiledKernel* find_kernel(const XDNAKernelKey& key);
    
    void set_cache_directory(const std::string& path);
    const std::string& get_cache_directory() const;
    
    bool is_shape_whitelisted(const XDNAKernelKey& key) const;
    void print_registry_status() const;

private:
    XDNAKernelRegistry() {
        const char* env_cache = std::getenv("XDNA_KERNEL_CACHE");
        if (env_cache && *env_cache) {
            cache_dir_ = env_cache;
        } else {
            cache_dir_ = "/Storage/Cache/llama.cpp/experiments/npu-int8-gemm/kernel_cache";
        }
    }
    ~XDNAKernelRegistry();

    XDNACompiledKernel* compile_kernel(const XDNAKernelKey& key);
    bool compile_via_mlir_aie(const XDNAKernelKey& key,
                              std::string& out_xclbin,
                              std::string& out_inst,
                              std::string& out_pdi);
    bool find_cached_artifacts(const XDNAKernelKey& key,
                               std::string& out_xclbin,
                               std::string& out_inst,
                               std::string& out_pdi);
    bool find_cached_q8_scale(const XDNAKernelKey& key,
                              std::string& out_xclbin,
                              std::string& out_inst);
    void free_kernel(XDNACompiledKernel& kernel);
    bool free_one_except(const XDNAKernelKey& keep);

    std::unordered_map<XDNAKernelKey, XDNACompiledKernel, XDNAKernelKeyHash> kernels_;
    std::mutex registry_mutex_;
    std::string cache_dir_;
};

inline void xdna_kernel_registry_set_cache_dir(const char* path) {
    XDNAKernelRegistry::instance().set_cache_directory(path);
}

inline const char* xdna_kernel_registry_get_cache_dir() {
    static std::string empty;
    const auto& dir = XDNAKernelRegistry::instance().get_cache_directory();
    return dir.empty() ? nullptr : dir.c_str();
}

inline void xdna_kernel_registry_print_status() {
    XDNAKernelRegistry::instance().print_registry_status();
}