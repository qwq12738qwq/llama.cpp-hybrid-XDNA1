// overlap_proof.cpp -- Phase 20G step 9: minimal Vulkan + XDNA async overlap proof.
//
// Single process, two independent execution paths:
//   VK   : Vulkan compute GEMM (float, gemm.spv) submitted on a queue, waited on a fence
//   XDNA : native XRT int8 GEMM via xrt::run.start() (no wait), waited later
//
// Timeline per iteration:
//   xdna.run.start()          (async, returns immediately)
//   vkQueueSubmit             (async, returns immediately)
//   vkWaitForFences           (blocks until Vulkan done)
//   xdna.run.wait()           (blocks until XDNA done)
//
// overlap_ns = min(vk_done, xdna_done) - max(xdna_submit, vk_submit)
// PASS requires overlap_ns > 0 for every measured iteration.
//
// Build:
//   g++ -O2 -std=c++17 overlap_proof.cpp -o overlap_proof \
//       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub -lxrt_coreutil -lvulkan
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
#include <atomic>

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
// Vulkan GEMM (structure derived from experiments/gpu-npu-overlap/gpu_burn.cpp)
// ---------------------------------------------------------------------------
struct VkGemm {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buf_mem = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t q_idx = 0;
    uint32_t gx = 0, gy = 0;
    uint32_t w = 0, h = 0, k = 0;
    VkDeviceSize sz_a = 0, sz_b = 0, sz_c = 0;

    bool init(uint32_t width, uint32_t height, uint32_t kdim, const char * spv_path) {
        w = width; h = height; k = kdim;
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "overlap_proof";
        app_info.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ic_info{};
        ic_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ic_info.pApplicationInfo = &app_info;
        if (vkCreateInstance(&ic_info, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t n_phys = 0;
        vkEnumeratePhysicalDevices(instance, &n_phys, nullptr);
        if (n_phys == 0) return false;
        std::vector<VkPhysicalDevice> phys(n_phys);
        vkEnumeratePhysicalDevices(instance, &n_phys, phys.data());
        VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties dev_props{};
        for (auto & pd : phys) {
            vkGetPhysicalDeviceProperties(pd, &dev_props);
            if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { phys_dev = pd; break; }
        }
        if (phys_dev == VK_NULL_HANDLE) { phys_dev = phys[0]; vkGetPhysicalDeviceProperties(phys_dev, &dev_props); }
        fprintf(stderr, "[vk] %s ts_period=%.2f ns\n", dev_props.deviceName, dev_props.limits.timestampPeriod);

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qcount, qprops.data());
        for (uint32_t i = 0; i < qcount; i++) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { q_idx = i; break; }
        }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo dq_info{};
        dq_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dq_info.queueFamilyIndex = q_idx;
        dq_info.queueCount = 1;
        dq_info.pQueuePriorities = &prio;
        VkDeviceCreateInfo dd_info{};
        dd_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dd_info.queueCreateInfoCount = 1;
        dd_info.pQueueCreateInfos = &dq_info;
        if (vkCreateDevice(phys_dev, &dd_info, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, q_idx, 0, &queue);

        sz_a = (VkDeviceSize)h * k * sizeof(float);
        sz_b = (VkDeviceSize)k * w * sizeof(float);
        sz_c = (VkDeviceSize)h * w * sizeof(float);
        VkDeviceSize total = sz_a + sz_b + sz_c;

        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = total;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &buf_info, nullptr, &buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(device, buffer, &mem_req);
        uint32_t mem_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((mem_req.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mem_type = i; break; }
        }
        if (mem_type == UINT32_MAX) return false;
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_req.size;
        alloc_info.memoryTypeIndex = mem_type;
        if (vkAllocateMemory(device, &alloc_info, nullptr, &buf_mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, buffer, buf_mem, 0);

        void * mapped;
        vkMapMemory(device, buf_mem, 0, VK_WHOLE_SIZE, 0, &mapped);
        float * p = (float *)mapped;
        srand(42);
        for (VkDeviceSize i = 0; i < (VkDeviceSize)h * k; i++) p[i] = (float)(rand() % 100) / 99.0f;
        p += (VkDeviceSize)h * k;
        for (VkDeviceSize i = 0; i < (VkDeviceSize)k * w; i++) p[i] = (float)(rand() % 100) / 99.0f;
        p += (VkDeviceSize)k * w;
        for (VkDeviceSize i = 0; i < (VkDeviceSize)h * w; i++) p[i] = 0.0f;
        vkUnmapMemory(device, buf_mem);

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; i++) {
            bindings[i].binding = (uint32_t)i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dsl_info{};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = 3;
        dsl_info.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &ds_layout) != VK_SUCCESS) return false;

        VkDescriptorPoolSize dp_size{};
        dp_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dp_size.descriptorCount = 3;
        VkDescriptorPoolCreateInfo dp_info{};
        dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_info.maxSets = 1;
        dp_info.poolSizeCount = 1;
        dp_info.pPoolSizes = &dp_size;
        if (vkCreateDescriptorPool(device, &dp_info, nullptr, &ds_pool) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo ds_alloc_info{};
        ds_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc_info.descriptorPool = ds_pool;
        ds_alloc_info.descriptorSetCount = 1;
        ds_alloc_info.pSetLayouts = &ds_layout;
        if (vkAllocateDescriptorSets(device, &ds_alloc_info, &ds) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo buf_infos[3]{};
        buf_infos[0].buffer = buffer; buf_infos[0].offset = 0;           buf_infos[0].range = sz_a;
        buf_infos[1].buffer = buffer; buf_infos[1].offset = sz_a;        buf_infos[1].range = sz_b;
        buf_infos[2].buffer = buffer; buf_infos[2].offset = sz_a + sz_b; buf_infos[2].range = sz_c;
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        std::string shader_src = read_file(spv_path);
        VkShaderModuleCreateInfo sm_info{};
        sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_info.codeSize = shader_src.size();
        sm_info.pCode = (const uint32_t *)shader_src.data();
        VkShaderModule shader_mod;
        if (vkCreateShaderModule(device, &sm_info, nullptr, &shader_mod) != VK_SUCCESS) return false;

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc_range.size = sizeof(PushBlock);
        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 1;
        pl_info.pSetLayouts = &ds_layout;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
        if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo cp_info{};
        cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_info.stage.module = shader_mod;
        cp_info.stage.pName = "main";
        cp_info.layout = pipeline_layout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &pipeline) != VK_SUCCESS) return false;

        VkCommandPoolCreateInfo cp_pool_info{};
        cp_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp_pool_info.queueFamilyIndex = q_idx;
        cp_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &cp_pool_info, nullptr, &cmd_pool) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo cb_alloc{};
        cb_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_alloc.commandPool = cmd_pool;
        cb_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_alloc.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cb_alloc, &cmd_buf) != VK_SUCCESS) return false;

        VkQueryPoolCreateInfo qp_info{};
        qp_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp_info.queryCount = 2;
        if (vkCreateQueryPool(device, &qp_info, nullptr, &query_pool) != VK_SUCCESS) return false;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) return false;

        gx = (w + 15) / 16;
        gy = (h + 15) / 16;
        record_cmd();
        return true;
    }

    void record_cmd() {
        vkResetCommandBuffer(cmd_buf, 0);
        VkCommandBufferBeginInfo cb_begin{};
        cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd_buf, &cb_begin);
        vkCmdResetQueryPool(cmd_buf, query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &ds, 0, nullptr);
        PushBlock pc{ w, h, k };
        vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushBlock), &pc);
        vkCmdDispatch(cmd_buf, gx, gy, 1);
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        vkEndCommandBuffer(cmd_buf);
    }

    // async submit; returns at once
    void submit() {
        vkResetFences(device, 1, &fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        vkQueueSubmit(queue, 1, &si, fence);
    }

    // blocks until done; returns GPU timestamps (raw ticks)
    void wait(uint64_t & ts0, uint64_t & ts1) {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        uint64_t timestamps[2] = {0, 0};
        vkGetQueryPoolResults(device, query_pool, 0, 2, sizeof(uint64_t) * 2, timestamps,
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        ts0 = timestamps[0];
        ts1 = timestamps[1];
    }
};

// ---------------------------------------------------------------------------
// Native XRT int8 GEMM (no IRON runtime)
// ---------------------------------------------------------------------------
struct XdnaGemm {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo inst_bo;
    xrt::bo in_bo;
    xrt::bo w_bo;
    xrt::bo out_bo;
    std::vector<uint32_t> insts;
    size_t inst_bytes = 0;
    int M = 0, K = 0, N = 0;
    std::vector<int8_t> A, B;
    std::vector<int32_t> C;
    bool ok = false;

    XdnaGemm() : device(0), kernel(), inst_bo(), in_bo(), w_bo(), out_bo() {}

    bool init(int m, int k, int n, const char * xclbin_path, const char * insts_path) {
        M = m; K = k; N = n;
        try {
            xrt::xclbin xb{std::string(xclbin_path)};
            device.register_xclbin(xb);
            xrt::uuid uuid = xb.get_uuid();
            xrt::hw_context ctx(device, uuid);
            kernel = xrt::kernel(ctx, "MLIR_AIE");
            uint32_t gid1 = kernel.group_id(1);
            uint32_t gid3 = kernel.group_id(3);
            if (!load_insts(insts_path, insts)) return false;
            inst_bytes = insts.size() * sizeof(uint32_t);
            inst_bo = xrt::bo(device, inst_bytes, xrt::bo::flags::cacheable, gid1);
            std::memcpy(inst_bo.map(), insts.data(), inst_bytes);
            inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            in_bo  = xrt::bo(device, (size_t)M * K, xrt::bo::flags::host_only, gid3);
            w_bo   = xrt::bo(device, (size_t)K * N, xrt::bo::flags::host_only, gid3);
            out_bo = xrt::bo(device, (size_t)M * N * sizeof(int32_t), xrt::bo::flags::host_only, gid3);
            A.resize((size_t)M * K);
            B.resize((size_t)K * N);
            C.assign((size_t)M * N, 0);
            for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i * 7 + 3) % 255 - 127);
            for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i * 11 + 5) % 255 - 127);
            ok = true;
            return true;
        } catch (const std::exception & e) {
            fprintf(stderr, "[xdna] init failed: %s\n", e.what());
            return false;
        }
    }

    // async: copy-in + start, no wait
    void submit() {
        std::memcpy(in_bo.map(), A.data(), (size_t)M * K);
        std::memcpy(w_bo.map(), B.data(), (size_t)K * N);
        std::memset(out_bo.map(), 0, (size_t)M * N * sizeof(int32_t));
        in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // start the kernel (async); returns immediately
    void start_run(xrt::run & run) {
        run.set_arg(0, (uint64_t)3);
        run.set_arg(1, inst_bo);
        run.set_arg(2, (uint32_t)inst_bytes);
        run.set_arg(3, in_bo);
        run.set_arg(4, w_bo);
        run.set_arg(5, out_bo);
        run.start();
    }

    int wait_run(xrt::run & run) {
        auto state = run.wait();
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return (int)state;
    }
};

int main(int argc, char ** argv) {
    const char * spv = "/workspace/llama.cpp/experiments/gpu-npu-overlap/gemm.spv";
    const char * xclbin = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.xclbin";
    const char * insts_path = "/workspace/llama.cpp/experiments/npu-int8-gemm/kernel_cache/M64_K1024_N64/design.insts.bin";
    uint32_t vk_w = 1024, vk_h = 1024, vk_k = 1024;
    int xm = 64, xk = 1024, xn = 64;
    int iters = 20;
    bool serial = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--spv" && i + 1 < argc) spv = argv[++i];
        else if (a == "--xclbin" && i + 1 < argc) xclbin = argv[++i];
        else if (a == "--insts" && i + 1 < argc) insts_path = argv[++i];
        else if (a == "--vk-w" && i + 1 < argc) vk_w = (uint32_t)atoi(argv[++i]);
        else if (a == "--vk-h" && i + 1 < argc) vk_h = (uint32_t)atoi(argv[++i]);
        else if (a == "--vk-k" && i + 1 < argc) vk_k = (uint32_t)atoi(argv[++i]);
        else if (a == "--xm" && i + 1 < argc) xm = atoi(argv[++i]);
        else if (a == "--xk" && i + 1 < argc) xk = atoi(argv[++i]);
        else if (a == "--xn" && i + 1 < argc) xn = atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--serial") serial = true;
    }

    VkGemm vk;
    if (!vk.init(vk_w, vk_h, vk_k, spv)) { fprintf(stderr, "vk init failed\n"); return 1; }

    XdnaGemm xd;
    if (!xd.init(xm, xk, xn, xclbin, insts_path)) { fprintf(stderr, "xdna init failed\n"); return 1; }
    fprintf(stderr, "[xdna] ready M=%d K=%d N=%d\n", xm, xk, xn);

    // warmup
    for (int i = 0; i < 3; i++) {
        xrt::run r(xd.kernel);
        xd.submit();
        xd.start_run(r);
        uint64_t a, b;
        vk.submit();
        vk.wait(a, b);
        xd.wait_run(r);
    }
    fprintf(stderr, "warmup done\n");

    printf("iter,xdna_submit_ns,xdna_done_ns,vk_submit_ns,vk_done_ns,overlap_ns,critical_ns,xdna_dur_ns,vk_dur_ns,xdna_state\n");
    int pass = 0;
    double best_overlap = -1e18;
    double sum_critical = 0;
    for (int it = 0; it < iters; it++) {
        xrt::run r(xd.kernel);

        xd.submit();                     // CPU copy-in (not on critical path)
        uint64_t t0 = now_ns();

        uint64_t xdna_submit = 0, xdna_done = 0;
        uint64_t vk_submit = 0, vk_done = 0;
        int xstate = 0;

        if (serial) {
            // Baseline: Vulkan fully, then XDNA fully. No overlap.
            vk.submit();
            vk_submit = now_ns();
            uint64_t g0, g1;
            vk.wait(g0, g1);
            vk_done = now_ns();

            xd.start_run(r);
            xdna_submit = now_ns();
            xstate = xd.wait_run(r);
            xdna_done = now_ns();
        } else {
            xd.start_run(r);                 // XDNA async start
            xdna_submit = now_ns();

            vk.submit();                     // Vulkan async submit
            vk_submit = now_ns();

            // Wait each path on its own thread so both completion instants are
            // observed independently; the two waits are not serialized.
            uint64_t g0 = 0, g1 = 0;
            std::thread vk_waiter([&]() {
                vk.wait(g0, g1);
                vk_done = now_ns();
            });
            xstate = xd.wait_run(r);         // wait XDNA on the main thread
            xdna_done = now_ns();
            vk_waiter.join();
        }

        uint64_t first_submit = xdna_submit < vk_submit ? xdna_submit : vk_submit;
        uint64_t last_done    = vk_done > xdna_done ? vk_done : xdna_done;
        uint64_t start = xdna_submit > vk_submit ? xdna_submit : vk_submit;
        uint64_t end   = vk_done < xdna_done ? vk_done : xdna_done;
        int64_t overlap = (int64_t)end - (int64_t)start;
        uint64_t critical = last_done - first_submit;
        sum_critical += (double)critical;
        if (overlap > (int64_t)best_overlap) best_overlap = (double)overlap;
        if (overlap > 0) pass++;

        printf("%d,%llu,%llu,%llu,%llu,%lld,%llu,%llu,%llu,%d\n", it,
               (unsigned long long)(xdna_submit - t0),
               (unsigned long long)(xdna_done - t0),
               (unsigned long long)(vk_submit - t0),
               (unsigned long long)(vk_done - t0),
               (long long)overlap,
               (unsigned long long)critical,
               (unsigned long long)(xdna_done - xdna_submit),
               (unsigned long long)(vk_done - vk_submit),
               xstate);
    }

    fprintf(stderr, "OVERLAP %d/%d iters with overlap>0 (best=%lld ns)\n",
            pass, iters, (long long)best_overlap);
    fprintf(stderr, "MODE %s mean_critical_ns=%.0f\n", serial ? "serial" : "parallel",
            sum_critical / iters);
    printf("RESULT %s overlap_positive=%d/%d best_overlap_ns=%lld\n",
           pass == iters ? "PASS" : "FAIL", pass, iters, (long long)best_overlap);
    return pass == iters ? 0 : 2;
}
