// partition_proof.cpp -- Phase 20G step 11: Vulkan + XDNA N-partition of one GEMM.
//
// One GEMM  C[M x N] = A[M x K] x B[K x N]  is split along N:
//   N0 columns -> Vulkan int8->int32 GEMM  (native Vulkan compute pipeline)
//   N1 columns -> XDNA native XRT int8 GEMM (xrt::run.start(), no wait)
//   N0 + N1 = N
//
// The concatenated output [C0 | C1] must equal the full CPU reference. The two
// partitions are submitted back to back (XDNA start first, then vkQueueSubmit)
// and waited on separate threads, so the overlap is measured, not assumed.
//
// Build:
//   glslc gemm_i8_i32.comp -o gemm_i8_i32.spv
//   g++ -O2 -std=c++17 partition_proof.cpp -o partition_proof \
//       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub -lxrt_coreutil -lvulkan -pthread
//
#include <vulkan/vulkan.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>

struct PushBlock { uint32_t w, h, k; };

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t sz = (size_t)f.tellg();
    std::string buf(sz, '\0');
    f.seekg(0);
    f.read(&buf[0], sz);
    return buf;
}

static bool load_insts(const char * path, std::vector<uint32_t> & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { fprintf(stderr, "cannot open %s\n", path); return false; }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize(sz / 4);
    ifs.read(reinterpret_cast<char *>(out.data()), sz);
    return true;
}

// ---------------------------------------------------------------------------
// Vulkan int32 GEMM (int8 values held in int32 buffers)
// ---------------------------------------------------------------------------
struct VkI32Gemm {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool dspool = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandPool cmdpool = VK_NULL_HANDLE;
    VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
    VkQueryPool qpool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t q_idx = 0;

    VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE, bufC = VK_NULL_HANDLE;
    VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE, memC = VK_NULL_HANDLE;
    void * mapA = nullptr, * mapB = nullptr, * mapC = nullptr;
    uint32_t W = 0, H = 0, Kd = 0;
    uint32_t gx = 0, gy = 0;

    bool create_device() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ic{};
        ic.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ic.pApplicationInfo = &app;
        if (vkCreateInstance(&ic, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t np = 0;
        vkEnumeratePhysicalDevices(instance, &np, nullptr);
        if (np == 0) return false;
        std::vector<VkPhysicalDevice> phys(np);
        vkEnumeratePhysicalDevices(instance, &np, phys.data());
        VkPhysicalDevice pd = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties props{};
        for (auto & p : phys) {
            vkGetPhysicalDeviceProperties(p, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { pd = p; break; }
        }
        if (pd == VK_NULL_HANDLE) { pd = phys[0]; vkGetPhysicalDeviceProperties(pd, &props); }
        fprintf(stderr, "[vk] %s\n", props.deviceName);

        uint32_t qc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, qp.data());
        for (uint32_t i = 0; i < qc; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { q_idx = i; break; }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo dq{};
        dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dq.queueFamilyIndex = q_idx;
        dq.queueCount = 1;
        dq.pQueuePriorities = &prio;
        VkDeviceCreateInfo dd{};
        dd.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dd.queueCreateInfoCount = 1;
        dd.pQueueCreateInfos = &dq;
        if (vkCreateDevice(pd, &dd, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, q_idx, 0, &queue);

        std::string src = read_file("/workspace/llama.cpp/experiments/phase20g/gemm_i8_i32.spv");
        VkShaderModuleCreateInfo sm{};
        sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = src.size();
        sm.pCode = (const uint32_t *)src.data();
        VkShaderModule mod;
        if (vkCreateShaderModule(device, &sm, nullptr, &mod) != VK_SUCCESS) return false;

        VkDescriptorSetLayoutBinding binds[3]{};
        for (int i = 0; i < 3; i++) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = 3;
        dli.pBindings = binds;
        if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &dsl) != VK_SUCCESS) return false;

        VkDescriptorPoolSize dps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &dps;
        if (vkCreateDescriptorPool(device, &dpi, nullptr, &dspool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = dspool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(device, &dai, &ds) != VK_SUCCESS) return false;

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(PushBlock);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &pli, nullptr, &pl) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = pl;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline) != VK_SUCCESS) return false;

        VkCommandPoolCreateInfo cpi2{};
        cpi2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi2.queueFamilyIndex = q_idx;
        cpi2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &cpi2, nullptr, &cmdpool) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo cba{};
        cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cba.commandPool = cmdpool;
        cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cba.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cba, &cmdbuf) != VK_SUCCESS) return false;

        VkQueryPoolCreateInfo qpi{};
        qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        if (vkCreateQueryPool(device, &qpi, nullptr, &qpool) != VK_SUCCESS) return false;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fi, nullptr, &fence) != VK_SUCCESS) return false;
        return true;
    }

    bool make_buffer(VkDeviceSize size, VkBuffer & buf, VkDeviceMemory & mem, void *& map) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device, buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(vk_physical(), &mp);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
        }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = mt;
        if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, buf, mem, 0);
        vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &map);
        return true;
    }

    VkPhysicalDevice pd_cache = VK_NULL_HANDLE;
    VkPhysicalDevice vk_physical() {
        if (pd_cache) return pd_cache;
        uint32_t np = 0;
        vkEnumeratePhysicalDevices(instance, &np, nullptr);
        std::vector<VkPhysicalDevice> phys(np);
        vkEnumeratePhysicalDevices(instance, &np, phys.data());
        VkPhysicalDeviceProperties props{};
        for (auto & p : phys) {
            vkGetPhysicalDeviceProperties(p, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { pd_cache = p; break; }
        }
        if (!pd_cache) pd_cache = phys[0];
        return pd_cache;
    }

    // Allocate buffers and record the command buffer for C[H x W] = A[H x Kd] x B[Kd x W].
    bool alloc(uint32_t width, uint32_t height, uint32_t kdim) {
        W = width; H = height; Kd = kdim;
        if (!make_buffer((VkDeviceSize)H * Kd * 4, bufA, memA, mapA)) return false;
        if (!make_buffer((VkDeviceSize)Kd * W * 4, bufB, memB, mapB)) return false;
        if (!make_buffer((VkDeviceSize)H * W * 4, bufC, memC, mapC)) return false;

        VkDescriptorBufferInfo bi[3]{};
        bi[0] = { bufA, 0, VK_WHOLE_SIZE };
        bi[1] = { bufB, 0, VK_WHOLE_SIZE };
        bi[2] = { bufC, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < 3; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device, 3, w, 0, nullptr);

        gx = (W + 15) / 16;
        gy = (H + 15) / 16;

        vkResetCommandBuffer(cmdbuf, 0);
        VkCommandBufferBeginInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmdbuf, &cb);
        vkCmdResetQueryPool(cmdbuf, qpool, 0, 2);
        vkCmdWriteTimestamp(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        PushBlock pc{ W, H, Kd };
        vkCmdPushConstants(cmdbuf, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushBlock), &pc);
        vkCmdDispatch(cmdbuf, gx, gy, 1);
        vkCmdWriteTimestamp(cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
        vkEndCommandBuffer(cmdbuf);
        return true;
    }

    // CPU upload of A and B (off the critical path). A/B are int8 values widened to int32.
    void upload(const int8_t * A, const int8_t * B) {
        int32_t * a = (int32_t *)mapA;
        int32_t * b = (int32_t *)mapB;
        for (size_t i = 0; i < (size_t)H * Kd; i++) a[i] = A[i];
        for (size_t i = 0; i < (size_t)Kd * W; i++) b[i] = B[i];
    }

    void submit() {
        vkResetFences(device, 1, &fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdbuf;
        vkQueueSubmit(queue, 1, &si, fence);
    }

    void wait(std::vector<int32_t> & out) {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        uint64_t ts[2];
        vkGetQueryPoolResults(device, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        out.resize((size_t)H * W);
        memcpy(out.data(), mapC, (size_t)H * W * 4);
    }
};

// ---------------------------------------------------------------------------
// Native XRT int8 GEMM
// ---------------------------------------------------------------------------
struct XdnaI8Gemm {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo inst_bo, in_bo, w_bo, out_bo;
    std::vector<uint32_t> insts;
    size_t inst_bytes = 0;
    int M = 0, K = 0, N = 0;

    XdnaI8Gemm() : device(0), kernel(), inst_bo(), in_bo(), w_bo(), out_bo() {}

    bool init(int m, int k, int n, const char * xclbin, const char * insts_p) {
        M = m; K = k; N = n;
        try {
            xrt::xclbin xb{std::string(xclbin)};
            device.register_xclbin(xb);
            xrt::uuid uuid = xb.get_uuid();
            xrt::hw_context ctx(device, uuid);
            kernel = xrt::kernel(ctx, "MLIR_AIE");
            if (!load_insts(insts_p, insts)) return false;
            inst_bytes = insts.size() * sizeof(uint32_t);
            inst_bo = xrt::bo(device, inst_bytes, xrt::bo::flags::cacheable, kernel.group_id(1));
            std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
            inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            in_bo  = xrt::bo(device, (size_t)M * K, xrt::bo::flags::host_only, kernel.group_id(3));
            w_bo   = xrt::bo(device, (size_t)K * N, xrt::bo::flags::host_only, kernel.group_id(3));
            out_bo = xrt::bo(device, (size_t)M * N * sizeof(int32_t), xrt::bo::flags::host_only, kernel.group_id(3));
            return true;
        } catch (const std::exception & e) {
            fprintf(stderr, "[xdna] init failed: %s\n", e.what());
            return false;
        }
    }

    // CPU copy-in (off the critical path)
    void prepare(const int8_t * A, const int8_t * B) {
        std::memcpy(in_bo.map(), A, (size_t)M * K);
        std::memcpy(w_bo.map(), B, (size_t)K * N);
        std::memset(out_bo.map(), 0, (size_t)M * N * sizeof(int32_t));
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void start_run(xrt::run & run) {
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        run.start();
    }

    int wait_run(xrt::run & run, std::vector<int32_t> & out) {
        auto st = run.wait();
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        out.resize((size_t)M * N);
        memcpy(out.data(), out_bo.map(), (size_t)M * N * sizeof(int32_t));
        return (int)st;
    }
};

int main(int argc, char ** argv) {
    int M = 64, K = 1024, N = 256, N0 = 128;
    int iters = 20;
    bool serial = false;
    const char * xclbin = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N128_C1/design.xclbin";
    const char * insts_p = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N128_C1/design.insts.bin";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--n0" && i + 1 < argc) N0 = atoi(argv[++i]);
        else if (a == "--serial") serial = true;
    }
    int N1 = N - N0;
    if (N1 != 128) { fprintf(stderr, "this proof uses XDNA N1=128 (got %d)\n", N1); return 1; }

    // Deterministic operands
    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);

    // CPU reference (int32)
    std::vector<int32_t> ref((size_t)M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            ref[m * N + n] = acc;
        }

    // B column partitions (row-major repack, done on CPU before timing)
    std::vector<int8_t> B0((size_t)K * N0), B1((size_t)K * N1);
    for (int k = 0; k < K; k++) {
        memcpy(&B0[(size_t)k * N0], &B[(size_t)k * N + 0], N0);
        memcpy(&B1[(size_t)k * N1], &B[(size_t)k * N + N0], N1);
    }

    VkI32Gemm vk;
    if (!vk.create_device()) { fprintf(stderr, "vk device failed\n"); return 1; }
    if (!vk.alloc(N0, M, K)) { fprintf(stderr, "vk alloc failed\n"); return 1; }

    XdnaI8Gemm xd;
    if (!xd.init(M, K, N1, xclbin, insts_p)) { fprintf(stderr, "xdna init failed\n"); return 1; }
    fprintf(stderr, "[setup] M=%d K=%d N=%d  vulkan N0=%d  xdna N1=%d\n", M, K, N, N0, N1);

    auto run_once = [&](std::vector<int32_t> & c0, std::vector<int32_t> & c1,
                        uint64_t & xdna_submit, uint64_t & xdna_done,
                        uint64_t & vk_submit, uint64_t & vk_done, int & xstate, bool parallel) {
        xd.prepare(A.data(), B1.data());     // CPU copy-in, off critical path
        vk.upload(A.data(), B0.data());      // CPU upload,  off critical path

        xrt::run r(xd.kernel);
        if (parallel) {
            xd.start_run(r);
            xdna_submit = now_ns();
            vk.submit();
            vk_submit = now_ns();
            std::thread th([&]() { vk.wait(c0); vk_done = now_ns(); });
            xstate = xd.wait_run(r, c1);
            xdna_done = now_ns();
            th.join();
        } else {
            vk.submit();
            vk_submit = now_ns();
            vk.wait(c0);
            vk_done = now_ns();
            xd.start_run(r);
            xdna_submit = now_ns();
            xstate = xd.wait_run(r, c1);
            xdna_done = now_ns();
        }
    };

    // Warmup
    std::vector<int32_t> c0, c1;
    uint64_t a, b, c, d; int st;
    for (int i = 0; i < 3; i++) run_once(c0, c1, a, b, c, d, st, true);
    fprintf(stderr, "warmup done\n");

    // Independent full Vulkan GEMM (N columns) to compare the heterogeneous
    // [C0 | C1] against, as required by step 11.
    std::vector<int32_t> cfull;
    {
        VkI32Gemm vkfull;
        if (!vkfull.create_device()) { fprintf(stderr, "vkfull device failed\n"); return 1; }
        if (!vkfull.alloc(N, M, K)) { fprintf(stderr, "vkfull alloc failed\n"); return 1; }
        vkfull.upload(A.data(), B.data());
        vkfull.submit();
        vkfull.wait(cfull);
        long fbad = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) if (cfull[i] != ref[i]) fbad++;
        fprintf(stderr, "[full-vulkan] mismatches vs CPU %ld / %d\n", fbad, M * N);
    }

    // Correctness: concat [C0 | C1] must equal the CPU reference and the full
    // Vulkan GEMM.
    long bad = 0, bad_full = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t got = (n < N0) ? c0[(size_t)m * N0 + n] : c1[(size_t)m * N1 + (n - N0)];
            if (got != ref[(size_t)m * N + n]) bad++;
            if (got != cfull[(size_t)m * N + n]) bad_full++;
        }
    }
    fprintf(stderr, "[correctness] hetero vs CPU %ld / %d, vs full-vulkan %ld / %d\n",
            bad, M * N, bad_full, M * N);
    if (bad || bad_full) { printf("RESULT FAIL layout-mismatch\n"); return 3; }

    printf("iter,xdna_submit_ns,xdna_done_ns,vk_submit_ns,vk_done_ns,overlap_ns,critical_ns,xdna_dur_ns,vk_dur_ns,xdna_state\n");
    int pass = 0; double sum_crit = 0; long long best = -1e18;
    for (int it = 0; it < iters; it++) {
        uint64_t t0 = now_ns();
        run_once(c0, c1, a, b, c, d, st, !serial);
        uint64_t first = a < c ? a : c, last = b > d ? b : d;
        uint64_t s = a > c ? a : c, e = b < d ? b : d;
        long long overlap = (long long)e - (long long)s;
        uint64_t crit = last - first;
        sum_crit += (double)crit;
        if (overlap > best) best = overlap;
        if (overlap > 0) pass++;
        printf("%d,%llu,%llu,%llu,%llu,%lld,%llu,%llu,%llu,%d\n", it,
               (unsigned long long)(a - t0), (unsigned long long)(b - t0),
               (unsigned long long)(c - t0), (unsigned long long)(d - t0),
               overlap, (unsigned long long)crit,
               (unsigned long long)(b - a), (unsigned long long)(d - c), st);
    }
    fprintf(stderr, "OVERLAP %d/%d, best=%lld ns, mean_critical=%.0f ns\n",
            pass, iters, best, sum_crit / iters);
    bool ok = (bad == 0) && (pass == iters);
    printf("RESULT %s mismatches=%ld overlap_positive=%d/%d best_overlap_ns=%lld mean_critical_ns=%.0f\n",
           ok ? "PASS" : "FAIL", bad, pass, iters, best, sum_crit / iters);
    return ok ? 0 : 2;
}
