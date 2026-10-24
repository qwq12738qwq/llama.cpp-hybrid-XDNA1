
// Debug kernel that traces acquire order
// Each B tile gets a unique ID: (k_tile * N_tiles + n_tile) * 1000
// The kernel accumulates these IDs to trace order

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"
#include <cstdint>

extern "C" void debug_matmul_i8_i32(
    const int8_t* __restrict a,
    const int8_t* __restrict b,
    int32_t* __restrict c
) {
    // Tile dimensions: 64x64 * 64x64 -> 64x64
    constexpr int M = 64;
    constexpr int K = 64;
    constexpr int N = 64;
    
    // We can't easily track tile index from inside kernel without passing it
    // Instead, we'll encode the tile ID in the B matrix data itself
    // This kernel just does the matmul, the tracing is done via input data
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int kk = 0; kk < K; ++kk) {
                sum += static_cast<int32_t>(a[i * K + kk]) * static_cast<int32_t>(b[kk * N + j]);
            }
            c[i * N + j] += sum;
        }
    }
}
