// AIE2 Vectorized INT8 GEMM kernel for Phoenix XDNA1 NPU
// Compile: clang++ --target=aie2p-none-unknown-elf -O2 -c matmul_i8_i32.cpp
//
// Uses 64-wide INT8 MAC (vmac) instead of scalar multiply
// Expected: 4-10x improvement over 53 GOPS scalar ceiling

#include <stdint.h>

#define M_TILE 64
#define K_TILE 64
#define N_TILE 64

extern "C"
void matmul_i8_i32(const signed char *__restrict pA,
                   const signed char *__restrict pB,
                   int *__restrict pC)
{
    for (int m = 0; m < M_TILE; m++) {
        const signed char *a_row = pA + m * K_TILE;

        for (int n = 0; n < N_TILE; n += 64) {
            v64acc32 acc = undef_v64acc32();

            for (int k = 0; k < K_TILE; k++) {
                v64int8 va = broadcast_to_v64int8(a_row[k]);
                v64int8 vb = *(v64int8 *)(pB + k * N_TILE + n);
                acc = mac_8x8_8x8(va, vb, acc);
            }

            *(v64acc32 *)(pC + m * N_TILE + n) = acc;
        }
    }
}
