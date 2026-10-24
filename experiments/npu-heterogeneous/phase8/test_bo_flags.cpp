#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <cstdio>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <xclbin> <insts>\n", argv[0]);
        return 1;
    }
    xrt::device dev(0);
    xrt::xclbin xclbin_obj{std::string(argv[1])};
    dev.register_xclbin(xclbin_obj);
    xrt::hw_context hw_ctx(dev, xclbin_obj.get_uuid());
    xrt::kernel kernel(hw_ctx, "MLIR_AIE");
    
    uint32_t gid1 = kernel.group_id(1);
    uint32_t gid3 = kernel.group_id(3);
    uint32_t gid4 = kernel.group_id(4);
    uint32_t gid5 = kernel.group_id(5);
    fprintf(stderr, "group_ids: 1=%u, 3=%u, 4=%u, 5=%u\n", gid1, gid3, gid4, gid5);
    
    std::ifstream ifs(argv[2], std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t isz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint32_t> insts(isz / 4);
    ifs.read((char*)insts.data(), isz);
    ifs.close();
    
    // Try cacheable for inst
    fprintf(stderr, "Creating inst_bo with cacheable (0x%08X)\n", (uint32_t)xrt::bo::flags::cacheable);
    xrt::bo inst_bo(dev, isz, xrt::bo::flags::cacheable, gid1);
    fprintf(stderr, "inst_bo memory_group: %u\n", inst_bo.get_memory_group());
    
    // Try host_only for A
    fprintf(stderr, "Creating A_bo with host_only (0x%08X)\n", (uint32_t)xrt::bo::flags::host_only);
    xrt::bo a_bo(dev, 1*1024*sizeof(int8_t), xrt::bo::flags::host_only, gid3);
    fprintf(stderr, "A_bo memory_group: %u\n", a_bo.get_memory_group());
    
    // Try host_only for B
    xrt::bo b_bo(dev, 1024*16*sizeof(int8_t), xrt::bo::flags::host_only, gid4);
    fprintf(stderr, "B_bo memory_group: %u\n", b_bo.get_memory_group());
    
    // Try host_only for C
    xrt::bo c_bo(dev, 1*16*sizeof(int32_t), xrt::bo::flags::host_only, gid5);
    fprintf(stderr, "C_bo memory_group: %u\n", c_bo.get_memory_group());
    
    fprintf(stderr, "All BOs created successfully\n");
    return 0;
}