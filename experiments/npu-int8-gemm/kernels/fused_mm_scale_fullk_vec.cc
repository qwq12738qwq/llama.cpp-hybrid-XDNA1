// Phase 20E: vectorized fused int8 MAC + F32 per-block Q8_0 scale.
//
// Uses aie::mmul for 4x8x8 int8 MAC and 16-wide F32 vector ops for scale
// accumulation.  Processes one [m_tile, n_tile] output tile.
//
// A-tile: [m_tile, K] int8 + [m_tile, nb] float32  (contiguous in DDR)
// B-tile: [K, n_tile] int8 + [nb, n_tile] float32  (contiguous in DDR)
// C-tile: [m_tile, n_tile] float32 (in-place accumulator)

#include <aie_api/aie.hpp>
#include <stdint.h>

// Vectorized MAC: accumulate int32 partial sums for one [m, n] output tile
// over one K-block of 32 elements.
template <int M_TILE, int N_TILE>
static inline void mac_block(const int8_t * __restrict a_row,   // [M_TILE, 32]
                             const int8_t * __restrict b_col,   // [32, N_TILE]
                             int32_t * __restrict acc)           // [M_TILE, N_TILE]
{
    constexpr int r = 4;  // aie2 int8 MAC row dim
    constexpr int s = 8;  // aie2 int8 MAC K dim
    constexpr int t = 8;  // aie2 int8 MAC col dim

    for (int i = 0; i < M_TILE; i += r) {
        for (int j = 0; j < N_TILE; j += t) {
            aie::accum<acc32, r * t> sum;
            sum.from_vector(aie::zeros<int32, r * t>());

            for (int l = 0; l < 32; l += s) {
                aie::vector<int8, r * s> va =
                    aie::load_v<r * s>(a_row + i * 32 + l);
                aie::vector<int8, s * t> vb =
                    aie::load_v<s * t>(b_col + l * N_TILE + j);
                sum.mac(va, vb);
            }

            // Store accumulated int32 to acc
            aie::store_v<r * t>(&acc[i * N_TILE + j],
                                sum.template to_vector<int32>());
        }
    }
}

// Vectorized scale: out[i*n+j] += as[i] * ws[j] * acc[i*n+j]
template <int M_TILE, int N_TILE>
static inline void scale_acc(const int32_t * __restrict acc,
                             const float * __restrict as,
                             const float * __restrict ws,
                             float * __restrict out)
{
    for (int i = 0; i < M_TILE; i++) {
        const aie::vector<float, 16> vas = aie::broadcast<float, 16>(as[i]);
        for (int j = 0; j < N_TILE; j += 16) {
            int count = (N_TILE - j >= 16) ? 16 : N_TILE - j;
            aie::vector<float, 16> vws = aie::load_v<16>(ws + j);
            aie::vector<int32, 16> vacc = aie::load_v<16>(&acc[i * N_TILE + j]);
            // int32 -> float -> multiply by as[i]*ws[j] -> accumulate
            aie::accum<accfloat, 16> facc;
            facc.from_vector(aie::cast_to_float(vacc));
            aie::vector<float, 16> fv = facc.to_vector<float>();
            fv = aie::mul(fv, vws);
            fv = aie::mul(fv, vas);
            // accumulate into output
            aie::vector<float, 16> vout = aie::load_v<16>(out + i * N_TILE + j);
            vout = aie::add(vout, fv);
            aie::store_v<16>(out + i * N_TILE + j, vout);
        }
    }
}

extern "C" void fused_mm_scale_fullk_vec(
    const int8_t * __restrict a_packed,   // [m, K] int8 + [m, nb] float32
    const int8_t * __restrict b_packed,   // [K, n] int8 + [nb, n] float32
    float * __restrict c,                 // [m, n]
    int32_t M_TILE, int32_t K, int32_t N_TILE)
{
    constexpr int MT = 64;
    constexpr int NT = 64;
    constexpr int K_TILE = 32;
    const int32_t nb = K / K_TILE;

    const int8_t * a_data = a_packed;
    const float * a_scales = reinterpret_cast<const float *>(a_packed + M_TILE * K);
    const int8_t * b_data = b_packed;
    const float * b_scales = reinterpret_cast<const float *>(b_packed + K * N_TILE);

    for (int32_t bl = 0; bl < nb; bl++) {
        const int8_t * a_blk = a_data + bl * K_TILE;
        const int8_t * b_blk = b_data + bl * K_TILE * N_TILE;

        // Zero acc on first block, accumulate thereafter
        if (bl == 0) {
            // clear output tile
            for (int i = 0; i < MT * NT; i += 16) {
                aie::store_v<16>(c + i, aie::zeros<float, 16>());
            }
        }

        // MAC one K-block
        // For the vectorized path, we use a static buffer.
        // In practice, this would be an on-chip accumulator.
        int32_t acc[MT * NT] __attribute__((aligned(64))) = {};
        mac_block<MT, NT>(a_blk, b_blk, acc);

        // Scale and accumulate
        scale_acc<MT, NT>(acc, a_scales + bl, b_scales + bl * N_TILE, c);
    }
}
