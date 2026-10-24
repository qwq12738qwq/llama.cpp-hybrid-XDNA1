// Phase 20E: fused int8 MAC + F32 per-block Q8_0 scale, full K inside AIE.
//
// ObjectFifo layout:
//   A-tile: [m, K_total] int8  + [m, nb] float32 scales   (one BD per M-row)
//   B-tile: [k_tile, n] int8   + [n] float32 scale        (one BD per K-block)
//   C-tile: [m, n] float32                                   (one BD per M-row)
//
// The core iterates K/k_tile blocks over B, acquiring a fresh B-tile per block,
// while A (data + scales) stays acquired for the full K.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void fused_mm_scale_fullk(
    const int8_t * __restrict a_packed,   // [m, K_total + nb*4]
    const int8_t * __restrict b_packed,   // one B-tile: [k_tile*n + n*4]
    float * __restrict c,                 // [m, n]
    int32_t m, int32_t K_total, int32_t n)
{
    const int32_t nb = K_total / 32;

    // A data starts at offset 0, A scales at offset m*K_total
    const int8_t * a_data = a_packed;
    const float * a_scales = reinterpret_cast<const float *>(a_packed + m * K_total);

    for (int32_t i = 0; i < m; i++) {
        const float * as_row = a_scales + i * nb;

        for (int32_t j = 0; j < n; j++) {
            float acc = 0.0f;

            // Each b_packed points to a different K-block; the caller passes
            // the correct pointer for the current block.  Here we inline the
            // outer K-loop so the same function can be called once per block
            // with the block offset baked in, OR we can iterate internally
            // if all B blocks are contiguous in memory.
            //
            // For simplicity, this version expects a contiguous B tile:
            //   b_packed: [K_total, n] int8 + [nb, n] float32 scales
            const int8_t * b_data_all = b_packed;
            const float * b_scales_all = reinterpret_cast<const float *>(b_packed + K_total * n);

            for (int32_t bl = 0; bl < nb; bl++) {
                int32_t psum = 0;
                const int8_t * a_blk = a_data + i * K_total + bl * 32;
                const int8_t * b_blk = b_data_all + bl * 32 * n + j;
                for (int32_t l = 0; l < 32; l++) {
                    psum += (int32_t)a_blk[l] * (int32_t)b_blk[l * n];
                }
                acc += as_row[bl] * b_scales_all[bl * n + j] * (float)psum;
            }

            c[i * n + j] += acc;
        }
    }
}
