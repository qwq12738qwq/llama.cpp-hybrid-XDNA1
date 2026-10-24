
#include <cstdint>
extern "C" void gemv_batched_test(
    const float* __restrict A,
    const float* __restrict x,
    float* __restrict y
) {
    for (int n = 0; n < 16; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < 512; ++k) {
            acc += A[k * 16 + n] * x[k];
        }
        y[n] = acc;
    }
}
