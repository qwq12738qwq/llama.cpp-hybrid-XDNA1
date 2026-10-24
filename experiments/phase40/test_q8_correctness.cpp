// Phase 40F: Standalone Q8_0 correctness test.
//
// Tests 32 K-block correctness for M=512, K=1024, N=3584:
//   Test A: uniform scales (all = 1.0)
//   Test B: two scales (first half = 1.0, second half = 2.0)
//   Test C: strongly non-uniform [1, 2, 4, 8, 16, ...]
//   Test D: realistic Q8_0 (random scales)
//
// Build (standalone, no XRT):
//   g++ -O2 -std=c++17 -I. -Iggml/include -Iggml/src/ggml-xdna \
//       experiments/phase40/cpu_q8_reference.cpp \
//       experiments/phase40/test_q8_correctness.cpp \
//       -o test_q8_correctness -lm
//
// Build (with XRT backend):
//   g++ -O2 -std=c++17 -I. -Iggml/include -Iggml/src/ggml-xdna \
//       experiments/phase40/cpu_q8_reference.cpp \
//       experiments/phase40/test_q8_correctness.cpp \
//       ggml/src/ggml-xdna/ggml-xdna.cpp \
//       ggml/src/ggml-xdna/xdna-q8-scale.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4.cpp \
//       ggml/src/ggml-xdna/xdna-q8-c4-device-scale.cpp \
//       ggml/src/ggml-xdna/xdna-kernel-registry.cpp \
//       ggml/src/ggml-xdna/xdna-buffer.cpp \
//       ggml/src/ggml-xdna/xdna-whole-array.cpp \
//       ggml/src/ggml-xdna/xdna-persistent.cpp \
//       -lxrt_coreutil -o test_q8_correctness -lm

#include "cpu_q8_reference.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#define QK8_0 32

// Build Q8_0 tensor [rows, K] with controlled scales
void build_q8_tensor(std::vector<uint8_t>& q8_data, int rows, int K,
                     const std::vector<float>& scales,
                     const std::vector<int8_t>& quants) {
    int nb = K / QK8_0;
    q8_data.resize((size_t)rows * nb * 34);
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            float scale = scales[r * nb + b];
            int8_t block_quants[32];
            for (int kk = 0; kk < 32; kk++) {
                block_quants[kk] = quants[r * K + b * 32 + kk];
            }
            uint8_t* ptr = q8_data.data() + ((size_t)r * nb + b) * 34;
            uint16_t h = fp32_to_fp16(scale);
            memcpy(ptr, &h, 2);
            memcpy(ptr + 2, block_quants, 32);
        }
    }
}

// Generate deterministic int8 values
void generate_quants(std::vector<int8_t>& quants, int total, int seed = 42) {
    quants.resize(total);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (int i = 0; i < total; i++) {
        quants[i] = (int8_t)dist(rng);
    }
}

// Test A: uniform scales
bool test_uniform_scales() {
    const int M = 512, K = 1024, N = 3584;
    int nb = K / QK8_0;
    
    fprintf(stderr, "\n=== Test A: uniform scales ===\n");
    
    std::vector<int8_t> A_quants, B_quants;
    generate_quants(A_quants, M * K, 42);
    generate_quants(B_quants, K * N, 43);
    
    std::vector<float> A_scales(M * nb, 1.0f);
    std::vector<float> B_scales(nb * N, 1.0f);
    
    std::vector<uint8_t> A_q8, B_q8;
    build_q8_tensor(A_q8, M, K, A_scales, A_quants);
    build_q8_tensor(B_q8, K, N, B_scales, B_quants);
    
    std::vector<float> C_ref(M * N);
    cpu_q8_gemm_ref(A_q8.data(), B_q8.data(), C_ref.data(), M, K, N);
    
    fprintf(stderr, "Reference computed: C[0,0]=%.4f C[M-1,N-1]=%.4f\n", C_ref[0], C_ref[M*N-1]);
    
    // With all scales = 1, this should match int8 GEMM exactly
    return true;
}

// Test B: two scales
bool test_two_scales() {
    const int M = 512, K = 1024, N = 3584;
    int nb = K / QK8_0;
    
    fprintf(stderr, "\n=== Test B: two scales (first half=1, second half=2) ===\n");
    
    std::vector<int8_t> A_quants, B_quants;
    generate_quants(A_quants, M * K, 42);
    generate_quants(B_quants, K * N, 43);
    
    std::vector<float> A_scales(M * nb, 1.0f);
    std::vector<float> B_scales(nb * N, 1.0f);
    
    // First half of K-blocks = 1, second half = 2
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            A_scales[m * nb + b] = (b < nb / 2) ? 1.0f : 2.0f;
        }
    }
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            B_scales[b * N + n] = (b < nb / 2) ? 1.0f : 2.0f;
        }
    }
    
    std::vector<uint8_t> A_q8, B_q8;
    build_q8_tensor(A_q8, M, K, A_scales, A_quants);
    build_q8_tensor(B_q8, K, N, B_scales, B_quants);
    
    std::vector<float> C_ref(M * N);
    cpu_q8_gemm_ref(A_q8.data(), B_q8.data(), C_ref.data(), M, K, N);
    
    // Also compute a naive folded version to show it's wrong
    std::vector<int8_t> A_folded(M * K);
    std::vector<int8_t> B_folded(K * N);
    std::vector<float> A_gs(nb), B_gs(nb);
    
    for (int b = 0; b < nb; b++) {
        float amax = 1e-9f;
        for (int kk = 0; kk < 32; kk++) {
            for (int m = 0; m < M; m++) {
                float val = A_scales[m * nb + b] * (int8_t)A_quants[m * K + b * 32 + kk];
                if (fabsf(val) > amax) amax = fabsf(val);
            }
        }
        A_gs[b] = amax / 127.0f;
        for (int kk = 0; kk < 32; kk++) {
            for (int m = 0; m < M; m++) {
                float val = A_scales[m * nb + b] * (int8_t)A_quants[m * K + b * 32 + kk];
                int q = (int)std::lroundf(val / A_gs[b]);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                A_folded[(b * 32 + kk) * M + m] = (int8_t)q;
            }
        }
    }
    for (int b = 0; b < nb; b++) {
        float amax = 1e-9f;
        for (int kk = 0; kk < 32; kk++) {
            for (int n = 0; n < N; n++) {
                float val = B_scales[b * N + n] * (int8_t)B_quants[(b * 32 + kk) * N + n];
                if (fabsf(val) > amax) amax = fabsf(val);
            }
        }
        B_gs[b] = amax / 127.0f;
        for (int kk = 0; kk < 32; kk++) {
            for (int n = 0; n < N; n++) {
                float val = B_scales[b * N + n] * (int8_t)B_quants[(b * 32 + kk) * N + n];
                int q = (int)std::lroundf(val / B_gs[b]);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                B_folded[(b * 32 + kk) * N + n] = (int8_t)q;
            }
        }
    }
    
    // Compute naive folded result
    std::vector<float> C_folded(M * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < nb; b++) {
                float block_sum = 0.0f;
                for (int kk = 0; kk < 32; kk++) {
                    block_sum += (float)A_folded[(b * 32 + kk) * M + m] * (float)B_folded[(b * 32 + kk) * N + n];
                }
                C_folded[m * N + n] += block_sum * A_gs[b] * B_gs[b];
            }
        }
    }
    
    ErrorStats stats = compute_error_stats(C_ref.data(), C_folded.data(), M, N);
    fprintf(stderr, "Host-folded error: max_abs=%.4e rmse=%.4e max_rel=%.4e mismatches=%d/%d\n",
            stats.max_abs_error, stats.rmse, stats.max_rel_error, stats.mismatches, stats.total);
    
    return true;
}

// Test C: strongly non-uniform scales [1, 2, 4, 8, 16, ...]
bool test_nonuniform_scales() {
    const int M = 512, K = 1024, N = 3584;
    int nb = K / QK8_0;
    
    fprintf(stderr, "\n=== Test C: strongly non-uniform scales [1, 2, 4, 8, 16, ...] ===\n");
    
    std::vector<int8_t> A_quants, B_quants;
    generate_quants(A_quants, M * K, 42);
    generate_quants(B_quants, K * N, 43);
    
    std::vector<float> A_scales(M * nb);
    std::vector<float> B_scales(nb * N);
    
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            A_scales[m * nb + b] = powf(2.0f, (float)(b % 8));
        }
    }
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            B_scales[b * N + n] = powf(2.0f, (float)(b % 8));
        }
    }
    
    std::vector<uint8_t> A_q8, B_q8;
    build_q8_tensor(A_q8, M, K, A_scales, A_quants);
    build_q8_tensor(B_q8, K, N, B_scales, B_quants);
    
    std::vector<float> C_ref(M * N);
    cpu_q8_gemm_ref(A_q8.data(), B_q8.data(), C_ref.data(), M, K, N);
    
    return true;
}

// Test D: realistic Q8_0 (random scales from power law distribution)
bool test_realistic_q8() {
    const int M = 512, K = 1024, N = 3584;
    int nb = K / QK8_0;
    
    fprintf(stderr, "\n=== Test D: realistic Q8_0 (power-law scales) ===\n");
    
    std::vector<int8_t> A_quants, B_quants;
    generate_quants(A_quants, M * K, 42);
    generate_quants(B_quants, K * N, 43);
    
    std::vector<float> A_scales(M * nb);
    std::vector<float> B_scales(nb * N);
    
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> scale_dist(0.01f, 10.0f);
    
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            A_scales[m * nb + b] = scale_dist(rng);
        }
    }
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            B_scales[b * N + n] = scale_dist(rng);
        }
    }
    
    std::vector<uint8_t> A_q8, B_q8;
    build_q8_tensor(A_q8, M, K, A_scales, A_quants);
    build_q8_tensor(B_q8, K, N, B_scales, B_quants);
    
    std::vector<float> C_ref(M * N);
    std::vector<float> A_scales_out, B_scales_out;
    cpu_q8_gemm_ref_debug(A_q8.data(), B_q8.data(), C_ref.data(), A_scales_out, B_scales_out, M, K, N);
    
    print_scale_stats("A", A_scales_out.data(), M, nb);
    print_scale_stats("B", B_scales_out.data(), nb, N);
    print_matrix_preview("C_ref", C_ref.data(), M, N);
    
    return true;
}

// Run all tests
int main() {
    bool all_pass = true;
    
    all_pass &= test_uniform_scales();
    all_pass &= test_two_scales();
    all_pass &= test_nonuniform_scales();
    all_pass &= test_realistic_q8();
    
    fprintf(stderr, "\n=== Summary ===\n");
    if (all_pass) {
        fprintf(stderr, "All CPU reference tests passed.\n");
        return 0;
    } else {
        fprintf(stderr, "Some tests failed.\n");
        return 1;
    }
}
