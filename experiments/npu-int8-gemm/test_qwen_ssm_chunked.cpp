#include "xdna-persistent.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>

// Q8_0 block: 2-byte half scale + 32 int8 values
static const int QK8_0 = 32;

void fill_q8_0_block(uint8_t* block, int8_t qval) {
    uint16_t scale_half = 0x3C00; // 1.0 in half-precision
    block[0] = scale_half & 0xFF;
    block[1] = (scale_half >> 8) & 0xFF;
    for (int j = 0; j < QK8_0; j++) {
        block[2 + j] = (uint8_t)qval;
    }
}

// CPU reference for int8 matmul: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
void cpu_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    printf("[TEST] Qwen ssm_beta XDNA chunked execution validation\n");

    // Test dimensions matching Qwen ssm_beta: M=1, K=1024, N=16
    const int M = 1;
    const int K_TOTAL = 1024;
    const int N = 16;
    const int K_CHUNK = 32;
    const int NUM_CHUNKS = K_TOTAL / K_CHUNK;  // 32

    printf("[TEST] M=%d K_total=%d N=%d (chunk K=%d, %d chunks)\n", M, K_TOTAL, N, K_CHUNK, NUM_CHUNKS);

    // Create persistent engine with chunk dimensions
    struct ggml_xdna_persistent_engine* engine = ggml_xdna_persistent_engine_create(M, K_CHUNK, N, false);
    if (!engine) {
        printf("[FAIL] Failed to create persistent engine\n");
        return 1;
    }

    // Initialize XRT with Phase 9H xclbin
    const char* xclbin_path = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.xclbin";
    const char* inst_path = "/workspace/llama.cpp/experiments/npu-int8-gemm/phase9h_test_new/design.insts.bin";

    if (!ggml_xdna_persistent_engine_init_xrt(engine, xclbin_path, inst_path)) {
        printf("[FAIL] Failed to init XRT\n");
        ggml_xdna_persistent_engine_free(engine);
        return 1;
    }

    printf("[OK] Persistent engine initialized with chunked kernel\n");

    // Generate test data
    std::vector<int8_t> input(K_TOTAL * M);
    std::vector<int8_t> weight(K_TOTAL * N);
    std::vector<int32_t> output_cpu(M * N);
    std::vector<int32_t> output_xdna(M * N);

    // Pattern 1: All ones
    printf("\n=== Pattern 1: All ones (expected output = %d) ===\n", K_TOTAL);
    std::fill(input.begin(), input.end(), 1);
    std::fill(weight.begin(), weight.end(), 1);
    cpu_int8_gemm(input.data(), weight.data(), output_cpu.data(), M, K_TOTAL, N);

    if (!ggml_xdna_persistent_engine_execute_chunked(engine, input.data(), K_TOTAL, weight.data(), output_xdna.data())) {
        printf("[FAIL] Chunked execute failed\n");
        ggml_xdna_persistent_engine_free(engine);
        return 1;
    }

    bool pass = true;
    for (int n = 0; n < N; n++) {
        if (output_xdna[n] != output_cpu[n]) {
            printf("  Mismatch at %d: cpu=%d xdna=%d\n", n, output_cpu[n], output_xdna[n]);
            pass = false;
        }
    }
    if (pass) printf("  PASS: Exact match!\n");

    // Pattern 2: A[k] = k, B = 1
    printf("\n=== Pattern 2: A[k]=k (0..1023), B=1 ===\n");
    for (int k = 0; k < K_TOTAL; k++) {
        input[k] = (int8_t)(k % 256 - 128);
    }
    std::fill(weight.begin(), weight.end(), 1);
    cpu_int8_gemm(input.data(), weight.data(), output_cpu.data(), M, K_TOTAL, N);

    if (!ggml_xdna_persistent_engine_execute_chunked(engine, input.data(), K_TOTAL, weight.data(), output_xdna.data())) {
        printf("[FAIL] Chunked execute failed\n");
        ggml_xdna_persistent_engine_free(engine);
        return 1;
    }

    pass = true;
    for (int n = 0; n < N; n++) {
        if (output_xdna[n] != output_cpu[n]) {
            printf("  Mismatch at %d: cpu=%d xdna=%d\n", n, output_cpu[n], output_xdna[n]);
            pass = false;
        }
    }
    if (pass) printf("  PASS: Exact match!\n");

    // Pattern 3: A=1, B[k,n] = k*N+n
    printf("\n=== Pattern 3: A=1, B[k,n]=k*N+n ===\n");
    std::fill(input.begin(), input.end(), 1);
    for (int k = 0; k < K_TOTAL; k++) {
        for (int n = 0; n < N; n++) {
            weight[k * N + n] = (int8_t)((k * N + n) % 256 - 128);
        }
    }
    cpu_int8_gemm(input.data(), weight.data(), output_cpu.data(), M, K_TOTAL, N);

    if (!ggml_xdna_persistent_engine_execute_chunked(engine, input.data(), K_TOTAL, weight.data(), output_xdna.data())) {
        printf("[FAIL] Chunked execute failed\n");
        ggml_xdna_persistent_engine_free(engine);
        return 1;
    }

    pass = true;
    for (int n = 0; n < N; n++) {
        if (output_xdna[n] != output_cpu[n]) {
            printf("  Mismatch at %d: cpu=%d xdna=%d\n", n, output_cpu[n], output_xdna[n]);
            pass = false;
        }
    }
    if (pass) printf("  PASS: Exact match!\n");

    // Pattern 4: Random test (5 iterations)
    printf("\n=== Pattern 4: Random (5 iterations) ===\n");
    bool all_pass = true;
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < K_TOTAL * M; i++) {
            input[i] = (int8_t)((rand() % 255) - 127);
        }
        for (int i = 0; i < K_TOTAL * N; i++) {
            weight[i] = (int8_t)((rand() % 255) - 127);
        }
        cpu_int8_gemm(input.data(), weight.data(), output_cpu.data(), M, K_TOTAL, N);

        if (!ggml_xdna_persistent_engine_execute_chunked(engine, input.data(), K_TOTAL, weight.data(), output_xdna.data())) {
            printf("  Iter %d: FAIL - execute failed\n", iter);
            all_pass = false;
            continue;
        }

        pass = true;
        for (int n = 0; n < N; n++) {
            if (output_xdna[n] != output_cpu[n]) {
                printf("  Iter %d: Mismatch at %d cpu=%d xdna=%d\n", iter, n, output_cpu[n], output_xdna[n]);
                pass = false;
            }
        }
        if (pass) {
            printf("  Iter %d: PASS\n", iter);
        } else {
            all_pass = false;
        }
    }

    // Cleanup
    ggml_xdna_persistent_engine_free(engine);

    printf("\n=== SUMMARY ===\n");
    if (all_pass && pass) {
        printf("ALL TESTS PASSED!\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED!\n");
        return 1;
    }
}