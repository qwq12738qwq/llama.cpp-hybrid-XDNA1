// Phase 40B: CPU reference model for Q8_0 GEMM with per-block scales.
//
// Computes C[m,n] = sum_b scale_A[m,b] * scale_B[b,n] * sum_k A_i8[m,k] * B_i8[k,n]
// where the sum over k is only within each 32-element K-block.

#include <cstdint>
#include "cpu_q8_reference.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

// CPU reference: Q8_0 GEMM
// A_q8: [M, K] row-major Q8_0 blocks
// B_q8: [K, N] row-major Q8_0 blocks
// C:    [M, N] float32 output (row-major)
void cpu_q8_gemm_ref(const uint8_t* A_q8, const uint8_t* B_q8, float* C, int M, int K, int N) {
    int nb = K / QK8_0;

    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int b = 0; b < nb; b++) {
                const uint8_t* a_block_ptr = A_q8 + ((size_t)m * nb + b) * 34;
                float scale_a = fp16_to_fp32(*(const uint16_t*)a_block_ptr);
                
                const uint8_t* b_block_ptr = B_q8 + ((size_t)n * nb + b) * 34;
                float scale_b = fp16_to_fp32(*(const uint16_t*)b_block_ptr);

                double block_sum = 0.0;
                for (int kk = 0; kk < QK8_0; kk++) {
                    int8_t aq = (int8_t)a_block_ptr[2 + kk];
                    int8_t bq = (int8_t)b_block_ptr[2 + kk];
                    block_sum += (double)aq * (double)bq;
                }
                sum += block_sum * (double)scale_a * (double)scale_b;
            }
            C[m * N + n] = (float)sum;
        }
    }
}

// Same but also returns extracted scales for debugging
void cpu_q8_gemm_ref_debug(const uint8_t* A_q8, const uint8_t* B_q8, float* C,
                           std::vector<float>& A_scales, std::vector<float>& B_scales,
                           int M, int K, int N) {
    int nb = K / QK8_0;
    A_scales.assign(M * nb, 0.0f);
    B_scales.assign(nb * N, 0.0f);

    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t* a_block_ptr = A_q8 + ((size_t)m * nb + b) * 34;
            A_scales[m * nb + b] = fp16_to_fp32(*(const uint16_t*)a_block_ptr);
        }
    }
    for (int n = 0; n < N; n++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t* b_block_ptr = B_q8 + ((size_t)n * nb + b) * 34;
            B_scales[b * N + n] = fp16_to_fp32(*(const uint16_t*)b_block_ptr);
        }
    }
    cpu_q8_gemm_ref(A_q8, B_q8, C, M, K, N);
}

// Compute statistics between reference and test results
ErrorStats compute_error_stats(const float* ref, const float* test, int M, int N) {
    ErrorStats stats = {};
    stats.total = M * N;
    double sum_sq = 0.0;
    
    for (int i = 0; i < M * N; i++) {
        float err = std::fabsf(ref[i] - test[i]);
        if (err > stats.max_abs_error) stats.max_abs_error = err;
        sum_sq += (double)err * (double)err;
        
        float ref_abs = std::fabsf(ref[i]);
        if (ref_abs > 1e-6f) {
            float rel = err / ref_abs;
            if (rel > stats.max_rel_error) stats.max_rel_error = rel;
        }
        
        if (err > 1e-3f * std::fmaxf(fabsf(ref[i]), 1.0f)) {
            stats.mismatches++;
        }
    }
    stats.rmse = (float)std::sqrt(sum_sq / (M * N));
    return stats;
}

// Utility: print first few values of a matrix
void print_matrix_preview(const char* name, const float* mat, int M, int N, int count) {
    fprintf(stderr, "%s: ", name);
    for (int i = 0; i < std::min(count, M * N); i++) {
        fprintf(stderr, "%.4f ", mat[i]);
    }
    if (M * N > count) fprintf(stderr, "...");
    fprintf(stderr, "\n");
}

// Utility: print scale statistics
void print_scale_stats(const char* name, const float* scales, int rows, int cols) {
    float min_s = INFINITY, max_s = -INFINITY;
    float sum = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        if (scales[i] < min_s) min_s = scales[i];
        if (scales[i] > max_s) max_s = scales[i];
        sum += scales[i];
    }
    float avg = sum / (rows * cols);
    fprintf(stderr, "%s scales: min=%.4f max=%.4f avg=%.4f\n", name, min_s, max_s, avg);
}
