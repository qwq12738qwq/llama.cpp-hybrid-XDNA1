#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

extern "C" {
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin>\n", argv[0]);
        return 1;
    }
    
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "Failed to open xclbin\n");
        return 1;
    }
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> xclbin_data(sz);
    ifs.read(xclbin_data.data(), sz);
    ifs.close();
    
    fprintf(stderr, "Opening device 0\n");
    auto dhdl = xrtDeviceOpen(0);
    if (!dhdl) {
        fprintf(stderr, "Failed to open device\n");
        return 1;
    }
    
    fprintf(stderr, "Loading xclbin via xrtDeviceLoadXclbin\n");
    int rc = xrtDeviceLoadXclbin(dhdl, reinterpret_cast<const struct axlf*>(xclbin_data.data()));
    fprintf(stderr, "xrtDeviceLoadXclbin returned %d\n", rc);
    
    xrtDeviceClose(dhdl);
    return 0;
}
