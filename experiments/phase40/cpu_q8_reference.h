#pragma once

// Phase 40B: CPU reference model for Q8_0 GEMM with per-block scales.

#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>

#define QK8_0 32

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    if (exp == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exp--;
        }
        mantissa &= 0x3ff;
        exp++;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    exp = exp + (127 - 15);
    mantissa = mantissa << 13;
    uint32_t result = (sign << 31) | (exp << 23) | mantissa;
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint16_t result = 0;
    if (bits == 0) return result;
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xff) - 127;
    uint32_t mantissa = bits & 0x7fffff;
    if (exp <= -25) {
        return (sign << 15);
    } else if (exp < 0) {
        mantissa |= 0x800000;
        mantissa >>= (1 - exp);
        result = (sign << 15) | (mantissa >> 13);
        return result;
    } else if (exp >= 16) {
        return (sign << 15) | 0x7c00;
    }
    result = (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
    return result;
}

static inline void set_q8_block(uint8_t* q8_ptr, int blk_idx, float scale, const int8_t* quants) {
    uint16_t h = fp32_to_fp16(scale);
    std::memcpy(q8_ptr + blk_idx * 34, &h, 2);
    std::memcpy(q8_ptr + blk_idx * 34 + 2, quants, 32);
}

struct ErrorStats {
    float max_abs_error;
    float rmse;
    float max_rel_error;
    int mismatches;
    int total;
};

// Compute C[m,n] = sum_b scale_A[m,b] * scale_B[b,n] * sum_k A_i8[m,k] * B_i8[k,n]
// A_q8: [M, K] row-major Q8_0 blocks
// B_q8: [K, N] row-major Q8_0 blocks
// C:    [M, N] float32 output (row-major)
void cpu_q8_gemm_ref(const uint8_t* A_q8, const uint8_t* B_q8, float* C, int M, int K, int N);

// Same but also returns extracted scales for debugging
void cpu_q8_gemm_ref_debug(const uint8_t* A_q8, const uint8_t* B_q8, float* C,
                           std::vector<float>& A_scales, std::vector<float>& B_scales,
                           int M, int K, int N);

// Compute statistics between reference and test results
ErrorStats compute_error_stats(const float* ref, const float* test, int M, int N);

// Utility functions
void print_matrix_preview(const char* name, const float* mat, int M, int N, int count = 5);
void print_scale_stats(const char* name, const float* scales, int rows, int cols);
