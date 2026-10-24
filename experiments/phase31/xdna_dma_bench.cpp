// xdna_dma_bench.cpp -- Phase 31: real XDNA staging DMA bandwidth.
//
// Measures how long it takes to move a KV-sized block into and out of an XDNA
// buffer object (the path the ggml-xdna backend actually uses: host_only BO +
// sync TO/FROM_DEVICE). This bounds how much of an attention KV-range partition
// could ever be handled by the NPU under a given critical-path budget.
//
// Build: g++ -O2 -std=c++17 xdna_dma_bench.cpp -o xdna_dma_bench -lxrt_coreutil
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_uuid.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    xrt::device device(0);
    printf("device=%s\n", device.get_info<xrt::info::device::name>().c_str());

    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    const xrt::memory_group grp = 0;

    std::vector<size_t> sizes = {1u << 20, 4u << 20, 16u << 20, 33u << 20};

    for (size_t sz : sizes) {
        std::vector<uint8_t> src(sz, 0xAB);

        auto run_dir = [&](const char * tag, xrt::bo::flags fl) {
            xrt::bo bo(device, sz, fl, grp);
            uint8_t * p = bo.map<uint8_t *>();
            std::memcpy(p, src.data(), sz);

            std::vector<double> tw, tr;
            for (int i = 0; i < reps; i++) {
                double t0 = now_ms();
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                double t1 = now_ms();
                tw.push_back(t1 - t0);

                t0 = now_ms();
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                t1 = now_ms();
                tr.push_back(t1 - t0);
            }
            double mw = median_ms(tw), mr = median_ms(tr);
            double gbps_w = (sz / 1e9) / (mw / 1e3);
            double gbps_r = (sz / 1e9) / (mr / 1e3);
            printf("DMA,%s,size_mb=%.1f,to_dev_ms=%.4f,to_dev_GBps=%.2f,from_dev_ms=%.4f,from_dev_GBps=%.2f\n",
                   tag, sz / 1048576.0, mw, gbps_w, mr, gbps_r);
            fflush(stdout);
        };

        run_dir("host_only", xrt::bo::flags::host_only);
    }
    return 0;
}
