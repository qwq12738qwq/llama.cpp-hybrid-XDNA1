#include "xdna-kernel-registry.h"
#include "xdna-buffer.h"
#include "ggml-xdna.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

XDNAKernelRegistry::~XDNAKernelRegistry() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    for (auto& [key, kernel] : kernels_) {
        free_kernel(kernel);
    }
    kernels_.clear();
}

void XDNAKernelRegistry::free_kernel(XDNACompiledKernel& kernel) {
    if (kernel.engine) {
        ggml_xdna_persistent_engine_free(kernel.engine);
        kernel.engine = nullptr;
    }
    if (kernel.q8_scale_engine) {
        ggml_xdna_q8_scale_free(kernel.q8_scale_engine);
        kernel.q8_scale_engine = nullptr;
    }
    if (kernel.q8_c4_engine) {
        ggml_xdna_q8_c4_free(kernel.q8_c4_engine);
        kernel.q8_c4_engine = nullptr;
    }
    if (kernel.q8_c4_ds_engine) {
        ggml_xdna_q8_c4_ds_free(kernel.q8_c4_ds_engine);
        kernel.q8_c4_ds_engine = nullptr;
    }
    if (kernel.q8_scale_c4_engine) {
        ggml_xdna_q8_scale_c4_free(kernel.q8_scale_c4_engine);
        kernel.q8_scale_c4_engine = nullptr;
    }
    if (kernel.q8_c4_mmul_v2_engine) {
        ggml_xdna_q8_c4_mmul_v2_free(kernel.q8_c4_mmul_v2_engine);
        kernel.q8_c4_mmul_v2_engine = nullptr;
    }
}

// Free one cached engine to free a HW context slot (driver caps concurrent hw_contexts).
bool XDNAKernelRegistry::free_one_except(const XDNAKernelKey& keep) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    for (auto it = kernels_.begin(); it != kernels_.end(); ++it) {
        if (it->first == keep) {
            continue;
        }
        fprintf(stderr, "[XDNA-REGISTRY] evicting engine M=%d K=%d N=%d to free HW context\n",
                it->first.M, it->first.K, it->first.N);
        free_kernel(it->second);
        kernels_.erase(it);
        return true;
    }
    return false;
}

void XDNAKernelRegistry::set_cache_directory(const std::string& path) {
    cache_dir_ = path;
    struct stat st;
    memset(&st, 0, sizeof(st));
    if (stat(cache_dir_.c_str(), &st) == -1) {
        mkdir(cache_dir_.c_str(), 0777);
    }
}

const std::string& XDNAKernelRegistry::get_cache_directory() const {
    return cache_dir_;
}

bool XDNAKernelRegistry::is_shape_whitelisted(const XDNAKernelKey& key) const {
    // Production whitelist: Phase 18 validated shapes + V2 multi-column
    // M in {64, 128, 256, 512, 448}, K in {1024, 2048, 4096}
    // k_tile: 32 for Q8_0 (per-block scales), 64/128 or full K for I8
    // n_cols in {1, 2, 4} for NPU1
    // N dimension: only accept N values that have cached xclbin/insts.bin
    if (key.m_tile != 64 && key.m_tile != 32) { fprintf(stderr, "[WHITELIST] FAIL m_tile=%d\n", key.m_tile); return false; }
    if (key.n_tile != 64) { fprintf(stderr, "[WHITELIST] FAIL n_tile=%d\n", key.n_tile); return false; }
    // I8 (no scales) is allowed to run the whole K in one submit: k_tile == K.
    if (key.k_tile != 32 && key.k_tile != 64 && key.k_tile != 128 && key.k_tile != key.K) { fprintf(stderr, "[WHITELIST] FAIL k_tile=%d key.K=%d\n", key.k_tile, key.K); return false; }
    // Phase 20D: expanded to C1-C4 for multi-column N-parallel kernels
    // Phoenix XDNA1 exposes 4 AIE columns (npu1_1col through npu1)
    if (key.n_cols < 1 || key.n_cols > 4) { fprintf(stderr, "[WHITELIST] FAIL n_cols=%d\n", key.n_cols); return false; }
    if (key.quantization != XDNAQuantization::Q8_0) return false;
    if (key.op_type != XDNAOpType::MUL_MAT) return false;
    // Phase 20D: accept any K divisible by 32 (engine always runs k_tile=32)
    if (key.K % 32 != 0) return false;
    if (key.M < 64) { fprintf(stderr, "[WHITELIST] FAIL M=%d\n", key.M); return false; }

    // Verify cached artifacts exist for this shape
    std::string xclbin_path, inst_path, pdi_path;
    // Use const_cast to call non-const find_cached_artifacts
    XDNAKernelRegistry* non_const_this = const_cast<XDNAKernelRegistry*>(this);
    if (key.k_tile == key.K && key.dtype_a == GGML_TYPE_Q8_0) {
        // scale-aware Q8_0 full-K: dedicated cache namespace
        if (key.n_cols >= 4) {
            // Phase 61: the scale-preserving kernel on the native 8x8 matrix
            // unit with a 16-lane float accumulator has its own cache space
            const char * q8_engine_env = getenv("XDNA_Q8_ENGINE");
            if (q8_engine_env && strcasecmp(q8_engine_env, "c4_mmul_v3") == 0) {
                char xc3[512], ins3[512];
                snprintf(xc3, sizeof(xc3), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL8/design.xclbin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                snprintf(ins3, sizeof(ins3), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL8/design.insts.bin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                struct stat st3;
                if (stat(xc3, &st3) == 0 && stat(ins3, &st3) == 0) {
                    return true;
                }
                fprintf(stderr, "[WHITELIST] FAIL missing c4_mmul_v3 kernel %s\n", xc3);
                return false;
            }
            if (q8_engine_env && strcasecmp(q8_engine_env, "c4_mmul_v4") == 0) {
                // Phase 62: the per-K-chunk scale kernel
                char xc4[512], ins4[512];
                snprintf(xc4, sizeof(xc4), "%s/M%d_K%d_N%d_C%d_Q8C256_MMUL8/design.xclbin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                snprintf(ins4, sizeof(ins4), "%s/M%d_K%d_N%d_C%d_Q8C256_MMUL8/design.insts.bin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                struct stat st4;
                if (stat(xc4, &st4) == 0 && stat(ins4, &st4) == 0) {
                    return true;
                }
                fprintf(stderr, "[WHITELIST] FAIL missing c4_mmul_v4 kernel %s\n", xc4);
                return false;
            }
            // Phase 40M+: try Q8SCALEV-C4 mmul-v2 first (preferred path)
            char xc[512], ins[512];
            snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL/design.xclbin",
                     cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
            snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL/design.insts.bin",
                     cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
            struct stat st;
            if (stat(xc, &st) == 0 && stat(ins, &st) == 0) {
                return true;
            }
            // Phase 40K+: try Q8SCALEV-C4 software fallback
            snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_Q8SCALEV/design.xclbin",
                     cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
            snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_Q8SCALEV/design.insts.bin",
                     cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
            if (stat(xc, &st) == 0 && stat(ins, &st) == 0) {
                return true;
            }
            // Q8-C4: try C4_DEVICE_SCALE cache
            snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_DEVICE_SCALE/design.xclbin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_DEVICE_SCALE/design.insts.bin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            if (stat(xc, &st) == 0 && stat(ins, &st) == 0) {
                return true;
            }
            // Q8-C4: try C4_WA cache
            snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_WA/design.xclbin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_WA/design.insts.bin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            if (stat(xc, &st) == 0 && stat(ins, &st) == 0) {
                return true;
            }
            return false;
        }
        // scale-aware Q8_0 full-K: dedicated cache namespace
        return non_const_this->find_cached_q8_scale(key, xclbin_path, inst_path);
    }
    if (!non_const_this->find_cached_artifacts(const_cast<XDNAKernelKey&>(key), xclbin_path, inst_path, pdi_path)) {
        return false;
    }
    return true;
}

XDNACompiledKernel* XDNAKernelRegistry::find_kernel(const XDNAKernelKey& key) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
        return &it->second;
    }
    return nullptr;
}

XDNACompiledKernel* XDNAKernelRegistry::get_or_create_kernel(const XDNAKernelKey& key) {
    if (ggml_xdna_session_is_fatal()) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = kernels_.find(key);
        if (it != kernels_.end()) {
            return &it->second;
        }
    }
    XDNACompiledKernel* kernel = compile_kernel(key);
    if (kernel) {
        return kernel;
    }
    return nullptr;
}

bool XDNAKernelRegistry::find_cached_artifacts(const XDNAKernelKey& key,
                                                 std::string& out_xclbin,
                                                 std::string& out_inst,
                                                 std::string& out_pdi) {
    struct stat st;

    // Cache directories use full K (not tile K). The tile parameters are in the
    // tile-suffixed path, but the K component is always the full K dimension.
    int cache_K = key.K;

    // Path 0: V2 kernel cache (experiments/xdna-v2/M{}_K{}_N{}_C{})
    char path_v2[512];
    snprintf(path_v2, sizeof(path_v2),
             "%s/experiments/xdna-v2/M%d_K%d_N%d_C%d",
             cache_dir_.c_str(), key.M, cache_K, key.N, key.n_cols);
    std::string cache_dir_str = cache_dir_;
    size_t pos = cache_dir_str.rfind("/experiments/npu-int8-gemm/kernel_cache");
    std::string workspace_root;
    if (pos != std::string::npos) {
        workspace_root = cache_dir_str.substr(0, pos);
    } else {
        workspace_root = "/Storage/Cache/llama.cpp";
    }
    snprintf(path_v2, sizeof(path_v2),
             "%s/experiments/xdna-v2/M%d_K%d_N%d_C%d",
             workspace_root.c_str(), key.M, cache_K, key.N, key.n_cols);
    std::string v2_xclbin = std::string(path_v2) + "/design.xclbin";
    std::string v2_inst = std::string(path_v2) + "/design.insts.bin";
    if (stat(v2_xclbin.c_str(), &st) == 0 && stat(v2_inst.c_str(), &st) == 0) {
        fprintf(stderr, "[XDNA-REGISTRY] Found V2 kernel: %s\n", path_v2);
        out_xclbin = v2_xclbin;
        out_inst = v2_inst;
        out_pdi = std::string(path_v2) + "/design.pdi";
        return true;
    }

    // Path 1: with tile suffix (legacy kernel cache)
    char path1[512];
    snprintf(path1, sizeof(path1), "%s/M%d_K%d_N%d_C%d_Tm%d_Tk%d_Tn%d",
             cache_dir_.c_str(), key.M, cache_K, key.N, key.n_cols,
             key.m_tile, key.k_tile, key.n_tile);

    out_xclbin = std::string(path1) + "/design.xclbin";
    out_inst = std::string(path1) + "/design.insts.bin";
    out_pdi = std::string(path1) + "/main.pdi";

    if (stat(out_xclbin.c_str(), &st) == 0 && stat(out_inst.c_str(), &st) == 0) {
        fprintf(stderr, "[XDNA-REGISTRY] Found cached kernel (suffixed): %s\n", path1);
        return true;
    }

    // Path 2: without tile suffix (older naming convention) - use full K
    char path2[512];
    snprintf(path2, sizeof(path2), "%s/M%d_K%d_N%d_C%d",
             cache_dir_.c_str(), key.M, cache_K, key.N, key.n_cols);

    std::string xclbin2 = std::string(path2) + "/design.xclbin";
    std::string inst2 = std::string(path2) + "/design.insts.bin";
    std::string pdi2 = std::string(path2) + "/main.pdi";

    if (stat(xclbin2.c_str(), &st) == 0 && stat(inst2.c_str(), &st) == 0) {
        out_xclbin = xclbin2;
        out_inst = inst2;
        out_pdi = pdi2;
        fprintf(stderr, "[XDNA-REGISTRY] Found cached kernel (non-suffixed): %s\n", path2);
        return true;
    }

    fprintf(stderr, "[XDNA-REGISTRY] MISS cache_dir=%s M=%d K=%d N=%d C=%d m_tile=%d k_tile=%d n_tile=%d\n",
            cache_dir_.c_str(), key.M, cache_K, key.N, key.n_cols, key.m_tile, key.k_tile, key.n_tile);

    return false;
}

// Phase 20G step 5: scale-aware Q8_0 full-K kernels live in a dedicated cache
// namespace so they do not collide with the K=32 chunked kernels.
bool XDNAKernelRegistry::find_cached_q8_scale(const XDNAKernelKey& key,
                                              std::string& out_xclbin,
                                              std::string& out_inst) {
    struct stat st;
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/M%d_K%d_N%d_C%d_Q8SCALEV",
             cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
    std::string xclbin = std::string(dir) + "/design.xclbin";
    std::string inst = std::string(dir) + "/design.insts.bin";
    if (stat(xclbin.c_str(), &st) == 0 && stat(inst.c_str(), &st) == 0) {
        out_xclbin = xclbin;
        out_inst = inst;
        return true;
    }
    return false;
}

XDNACompiledKernel* XDNAKernelRegistry::compile_kernel(const XDNAKernelKey& key) {
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
        return &it->second;
    }

    // Fail-closed: only whitelisted shapes may attempt compilation or cache lookup
    if (!is_shape_whitelisted(key)) {
        fprintf(stderr, "[XDNA-REGISTRY] FAIL-CLOSED: shape M=%d K=%d N=%d k_tile=%d not in whitelist\n",
                key.M, key.K, key.N, key.k_tile);
        return nullptr;
    }

    // Phase 20G step 5: scale-aware Q8_0 full-K path (k_tile == K). One submit
    // covers the whole K; the device applies the per-block scales.
    // For n_cols>=4, try kernels in order dictated by XDNA_Q8_ENGINE.
    if (key.k_tile == key.K && key.dtype_a == GGML_TYPE_Q8_0) {
        if (key.n_cols >= 4) {
            const char * q8_engine_env = getenv("XDNA_Q8_ENGINE");
            bool want_c4_ds = (q8_engine_env && strcasecmp(q8_engine_env, "c4_ds") == 0);
            bool want_q8scale_c4 = (q8_engine_env && strcasecmp(q8_engine_env, "c4") == 0);
            bool want_q8scale = (q8_engine_env && strcasecmp(q8_engine_env, "scale") == 0);
            bool want_c4_mmul_v2 = (q8_engine_env && strcasecmp(q8_engine_env, "c4_mmul_v2") == 0);
            bool want_s12 = (q8_engine_env && strcasecmp(q8_engine_env, "s12") == 0);
            bool want_requant = (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant") == 0);
            bool want_int8_pure_v2 = (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant_v2") == 0);
            bool want_int8_pure_v3 = (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant_v3") == 0);
            bool want_int8_pure_v4 = (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant_v4") == 0);
            bool want_c4_mmul_v3 = (q8_engine_env && strcasecmp(q8_engine_env, "c4_mmul_v3") == 0);
            bool want_c4_mmul_v4 = (q8_engine_env && strcasecmp(q8_engine_env, "c4_mmul_v4") == 0);
            // Phase 60: the tile shape lives in this table, not in the host code.
            // profile: name, cache suffix, m_tile, n_tile, k_tile, mmul
            const char * prof_name = "int8_requant_v2";
            const char * prof_suffix = "PURE";
            int prof_m = 32, prof_n = 64, prof_k = 256, prof_mmul = 4;
            if (want_int8_pure_v3) {
                prof_name = "int8_requant_v3"; prof_suffix = "PUREV3";
            } else if (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant_v4") == 0) {
                prof_name = "int8_requant_v4"; prof_suffix = "PUREV4_T32x64K512_M8";
                prof_m = 32; prof_n = 64; prof_k = 512; prof_mmul = 8;
            } else if (want_c4_mmul_v3) {
                // Phase 61: scale preserving, native 8x8 mmul, 16-lane float
                // accumulator, 64x64x256 tile
                prof_name = "c4_mmul_v3"; prof_suffix = "Q8SCALEV_MMUL8";
                prof_m = 64; prof_n = 64; prof_k = 256; prof_mmul = 8;
            } else if (want_c4_mmul_v4) {
                // Phase 62: per-K-chunk scale, int32 accumulation in the chunk
                prof_name = "c4_mmul_v4"; prof_suffix = "Q8C256_MMUL8";
                prof_m = 64; prof_n = 64; prof_k = 256; prof_mmul = 8;
            }
            (void) prof_name;
            bool want_auto = (q8_engine_env == nullptr || strcasecmp(q8_engine_env, "auto") == 0);

            // Phase 50 s12: pre-broadcast scales + 16-lane scale path + linear
            // C drain. Separate cache namespace; production path untouched.
            if (want_s12) {
                char xc12[512], ins12[512];
                snprintf(xc12, sizeof(xc12), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_S12/design.xclbin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                snprintf(ins12, sizeof(ins12), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_S12/design.insts.bin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                struct stat st;
                if (stat(xc12, &st) == 0 && stat(ins12, &st) == 0) {
                    XDNACompiledKernel kernel;
                    kernel.q8_c4_mmul_v2_engine = ggml_xdna_q8_c4_mmul_v2_create(
                        key.M, key.K, key.N, key.n_cols, 32, 64, 256, 4, xc12, ins12, 1);
                    if (kernel.q8_c4_mmul_v2_engine) {
                        kernel.spec.key = key;
                        kernel.spec.xclbin_path = xc12;
                        kernel.spec.inst_path = ins12;
                        kernel.spec.compiled = true;
                        auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                        fprintf(stderr, "[XDNA-REGISTRY] Loaded Q8 s12 kernel: M=%d K=%d N=%d cols=%d\n",
                                key.M, key.K, key.N, key.n_cols);
                        return &insert_it->second;
                    }
                    fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8 s12 engine M=%d K=%d N=%d\n",
                            key.M, key.K, key.N);
                    if (ggml_xdna_session_is_fatal()) {
                        return nullptr;
                    }
                } else {
                    fprintf(stderr, "[XDNA-REGISTRY] Missing Q8 s12 kernel: %s\n", xc12);
                    if (ggml_xdna_session_is_fatal()) {
                        return nullptr;
                    }
                }
            }

            // Phase 40M: Q8SCALEV-C4 mmul v2 is the preferred C4 path unless
            // the user explicitly asked for c4_ds, c4 (old software), or scale.
            std::string qs_xclbin, qs_inst;
            char xc[512], ins[512];

            // Try Q8SCALEV-C4 mmul v2 first (auto or explicit)
            if (want_auto || want_c4_mmul_v2 || want_requant || want_int8_pure_v2 || want_int8_pure_v3 ||
                want_c4_mmul_v3 || want_c4_mmul_v4 ||
                (q8_engine_env && strcasecmp(q8_engine_env, "int8_requant_v4") == 0)) {
                if (want_int8_pure_v3) {
                    snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_PUREV3/design.xclbin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                    snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_PUREV3/design.insts.bin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                } else if (want_int8_pure_v2 || want_int8_pure_v4 || want_c4_mmul_v3 ||
                           want_c4_mmul_v4) {
                    snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_%s/design.xclbin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols, prof_suffix);
                    snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_%s/design.insts.bin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols, prof_suffix);
                } else {
                    snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL/design.xclbin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                    snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_Q8SCALEV_MMUL/design.insts.bin",
                              cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                }
                qs_xclbin = xc; qs_inst = ins;
 
                struct stat st;
                if (stat(qs_xclbin.c_str(), &st) == 0 && stat(qs_inst.c_str(), &st) == 0) {
                    XDNACompiledKernel kernel;
                    kernel.q8_c4_mmul_v2_engine = ggml_xdna_q8_c4_mmul_v2_create(
                        key.M, key.K, key.N, key.n_cols,
                         prof_m, prof_n, prof_k, prof_mmul,
                         qs_xclbin.c_str(), qs_inst.c_str(), 0,
                         (want_int8_pure_v2 || want_int8_pure_v3 || want_int8_pure_v4) ? 1 : 0,
                         want_c4_mmul_v4 ? 1 : 0);
                    if (kernel.q8_c4_mmul_v2_engine) {
                        kernel.spec.key = key;
                        kernel.spec.xclbin_path = qs_xclbin;
                        kernel.spec.inst_path = qs_inst;
                        kernel.spec.compiled = true;
                        auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                        fprintf(stderr, "[XDNA-REGISTRY] Loaded Q8-C4 mmul-v2 kernel: M=%d K=%d N=%d cols=%d\n",
                                key.M, key.K, key.N, key.n_cols);
                        return &insert_it->second;
                    }
                    fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8-C4 mmul-v2 engine M=%d K=%d N=%d\n",
                            key.M, key.K, key.N);
                    if (ggml_xdna_session_is_fatal()) {
                        return nullptr;
                    }
                }
            }

            // Fallback: try old Q8SCALEV-C4 software if explicitly requested
            if (want_q8scale_c4) {
                snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C%d_Q8SCALEV/design.xclbin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C%d_Q8SCALEV/design.insts.bin",
                         cache_dir_.c_str(), key.M, key.K, key.N, key.n_cols);
                qs_xclbin = xc; qs_inst = ins;

                struct stat st;
                if (stat(qs_xclbin.c_str(), &st) == 0 && stat(qs_inst.c_str(), &st) == 0) {
                    XDNACompiledKernel kernel;
                    kernel.q8_scale_c4_engine = ggml_xdna_q8_scale_c4_create(
                        key.M, key.K, key.N, key.n_cols, qs_xclbin.c_str(), qs_inst.c_str());
                    if (kernel.q8_scale_c4_engine) {
                        kernel.spec.key = key;
                        kernel.spec.xclbin_path = qs_xclbin;
                        kernel.spec.inst_path = qs_inst;
                        kernel.spec.compiled = true;
                        auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                        fprintf(stderr, "[XDNA-REGISTRY] Loaded Q8SCALEV-C4 kernel: M=%d K=%d N=%d cols=%d\n",
                                key.M, key.K, key.N, key.n_cols);
                        return &insert_it->second;
                    }
                    fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8SCALEV-C4 engine M=%d K=%d N=%d\n",
                            key.M, key.K, key.N);
                    if (ggml_xdna_session_is_fatal()) {
                        return nullptr;
                    }
                }
            }

            // Fallback: try C4_DEVICE_SCALE if requested or if mmul-v2 failed
            if (want_c4_ds || want_auto) {
                snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_DEVICE_SCALE/design.xclbin",
                         cache_dir_.c_str(), key.M, key.K, key.N);
                snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_DEVICE_SCALE/design.insts.bin",
                         cache_dir_.c_str(), key.M, key.K, key.N);
                qs_xclbin = xc; qs_inst = ins;

                struct stat st;
                if (stat(qs_xclbin.c_str(), &st) == 0 && stat(qs_inst.c_str(), &st) == 0) {
                    XDNACompiledKernel kernel;
                    kernel.q8_c4_ds_engine = ggml_xdna_q8_c4_ds_create(
                        key.M, key.K, key.N, key.n_cols, qs_xclbin.c_str(), qs_inst.c_str());
                    if (kernel.q8_c4_ds_engine) {
                        kernel.spec.key = key;
                        kernel.spec.xclbin_path = qs_xclbin;
                        kernel.spec.inst_path = qs_inst;
                        kernel.spec.compiled = true;
                        auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                        fprintf(stderr, "[XDNA-REGISTRY] Loaded Q8-C4 device-scale kernel: M=%d K=%d N=%d\n",
                                key.M, key.K, key.N);
                        return &insert_it->second;
                    }
                    fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8-C4-DS engine M=%d K=%d N=%d\n",
                            key.M, key.K, key.N);
                    if (ggml_xdna_session_is_fatal()) {
                        return nullptr;
                    }
                }
            }

            // Fallback: try C4_WA
            struct stat st;
            snprintf(xc, sizeof(xc), "%s/M%d_K%d_N%d_C4_WA/design.xclbin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            snprintf(ins, sizeof(ins), "%s/M%d_K%d_N%d_C4_WA/design.insts.bin",
                     cache_dir_.c_str(), key.M, key.K, key.N);
            qs_xclbin = xc; qs_inst = ins;

            if (stat(qs_xclbin.c_str(), &st) == 0 && stat(qs_inst.c_str(), &st) == 0) {
                XDNACompiledKernel kernel;
                kernel.q8_c4_engine = ggml_xdna_q8_c4_create(
                    key.M, key.K, key.N, qs_xclbin.c_str(), qs_inst.c_str());
                if (kernel.q8_c4_engine) {
                    kernel.spec.key = key;
                    kernel.spec.xclbin_path = qs_xclbin;
                    kernel.spec.inst_path = qs_inst;
                    kernel.spec.compiled = true;
                    auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                    fprintf(stderr, "[XDNA-REGISTRY] Loaded Q8-C4 (C4_WA) kernel: M=%d K=%d N=%d\n",
                            key.M, key.K, key.N);
                    return &insert_it->second;
                }
                                fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8-C4 engine M=%d K=%d N=%d\n",
                        key.M, key.K, key.N);
                if (ggml_xdna_session_is_fatal()) {
                    return nullptr;
                }
            }
 else {
                fprintf(stderr, "[XDNA-REGISTRY] No C4 kernel found for Q8-C4: M=%d K=%d N=%d\n",
                        key.M, key.K, key.N);
            }
            return nullptr;
        }
        
        std::string qs_xclbin, qs_inst;
        if (find_cached_q8_scale(key, qs_xclbin, qs_inst)) {
            XDNACompiledKernel kernel;
            kernel.q8_scale_engine = ggml_xdna_q8_scale_create(
                key.M, key.K, key.N, qs_xclbin.c_str(), qs_inst.c_str());
            if (kernel.q8_scale_engine) {
                kernel.spec.key = key;
                kernel.spec.xclbin_path = qs_xclbin;
                kernel.spec.inst_path = qs_inst;
                kernel.spec.compiled = true;
                auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                fprintf(stderr, "[XDNA-REGISTRY] Loaded scale-aware Q8_0 kernel: M=%d K=%d N=%d\n",
                        key.M, key.K, key.N);
                return &insert_it->second;
            }
            fprintf(stderr, "[XDNA-REGISTRY] Failed to init Q8_0 scale engine M=%d K=%d N=%d\n",
                    key.M, key.K, key.N);
        }
        fprintf(stderr, "[XDNA-REGISTRY] No scale-aware Q8_0 cached kernel for M=%d K=%d N=%d\n",
                key.M, key.K, key.N);
        return nullptr;
    }

    // Try to find cached artifacts before compiling
    std::string xclbin_path, inst_path, pdi_path;
    if (find_cached_artifacts(key, xclbin_path, inst_path, pdi_path)) {
        XDNACompiledKernel kernel;
        // Use k_tile for engine's k dimension (tile size), not full K.
        // execute_chunked will handle the full K dimension by chunking.
        int engine_k = (key.k_tile > 0) ? key.k_tile : key.K;
        kernel.engine = ggml_xdna_persistent_engine_create(
            key.M, engine_k, key.N, false);

        if (kernel.engine) {
            bool ok = ggml_xdna_persistent_engine_init_xrt(
                kernel.engine, xclbin_path.c_str(), inst_path.c_str());
            if (ok) {
                kernel.spec.key = key;
                kernel.spec.xclbin_path = xclbin_path;
                kernel.spec.inst_path = inst_path;
                kernel.spec.pdi_path = pdi_path;
                kernel.spec.compiled = true;
                auto [insert_it, inserted] = kernels_.emplace(key, std::move(kernel));
                fprintf(stderr, "[XDNA-REGISTRY] Loaded cached kernel: M=%d K=%d N=%d (engine k=%d)\n",
                        key.M, key.K, key.N, engine_k);
                return &insert_it->second;
            } else {
                ggml_xdna_persistent_engine_free(kernel.engine);
                fprintf(stderr, "[XDNA-REGISTRY] Failed to init XRT for cached kernel\n");
            }
        }
    }

    // No cached artifacts — refuse to compile (fail-closed)
    fprintf(stderr, "[XDNA-REGISTRY] No cached artifacts for M=%d K=%d N=%d, refusing to compile\n",
            key.M, key.K, key.N);
    return nullptr;
}

bool XDNAKernelRegistry::compile_via_mlir_aie(const XDNAKernelKey& key,
                                               std::string& out_xclbin,
                                               std::string& out_inst,
                                               std::string& out_pdi) {
    // Fail-closed: only whitelisted shapes may compile
    if (!is_shape_whitelisted(key)) {
        fprintf(stderr, "[XDNA-REGISTRY] compile_via_mlir_aie: shape not whitelisted M=%d K=%d N=%d\n",
                key.M, key.K, key.N);
        return false;
    }

    // Check for cached artifacts first (no compilation needed)
    if (find_cached_artifacts(key, out_xclbin, out_inst, out_pdi)) {
        fprintf(stderr, "[XDNA-REGISTRY] Reusing cached artifacts at %s\n", out_xclbin.c_str());
        return true;
    }

    fprintf(stderr, "[XDNA-REGISTRY] No cached kernel found for M=%d K=%d N=%d, refusing to compile\n",
            key.M, key.K, key.N);
    return false;
}

void XDNAKernelRegistry::print_registry_status() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(registry_mutex_));
    fprintf(stderr, "\n=== XDNA Kernel Registry Status ===\n");
    fprintf(stderr, "Cache directory: %s\n", cache_dir_.c_str());
    fprintf(stderr, "Compiled kernels (%zu):\n", kernels_.size());
    for (const auto& [key, kernel] : kernels_) {
        fprintf(stderr, "  M=%d K=%d N=%d C=%d engine=%p\n",
                key.M, key.K, key.N, key.n_cols, static_cast<void*>(kernel.engine));
    }
    fprintf(stderr, "===================================\n\n");
}
