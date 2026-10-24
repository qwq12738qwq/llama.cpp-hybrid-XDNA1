// gpu_bench.cpp -- Vulkan compute GEMM benchmark for prefill characterization
//
// Usage: ./gpu_bench --M M --K K --N N [--iters N] [--warmup N]
//
// Output (JSON to stdout):
//   {"status":"OK","op":"GEMM","M":...,"K":...,"N":...,
//    "h2d_ms":...,"compute_ms":...,"d2h_ms":...,"total_ms":...,"gops":...}
//
// Measures:
//   h2d_ms     = host->device staging copy time (median)
//   compute_ms = GPU execution time via Vulkan timestamp query (median)
//   d2h_ms     = device->host staging copy time (median)
//   total_ms   = end-to-end wall time (median)
//   gops       = effective GOPS = 2*M*K*N / compute_ms / 1e9
//
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
    fprintf(stderr, "VK error %d at line %d\n", r, __LINE__); exit(1); } } while(0)

static std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    size_t sz = (size_t)f.tellg();
    std::string buf(sz, '\0');
    f.seekg(0);
    f.read(&buf[0], sz);
    return buf;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct PushBlock { uint32_t w, h, k; };

int main(int argc, char** argv) {
    uint32_t M = 1024, K = 4096, N = 4096;
    int n_warmup = 2, n_iters = 5;
    const char* spv_path = "gemm.spv";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) M = (uint32_t)atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = (uint32_t)atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = (uint32_t)atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) n_warmup = atoi(argv[++i]);
        else if (arg == "--spv" && i + 1 < argc) spv_path = argv[++i];
    }

    // --- Instance ---
    VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    // --- Physical device ---
    uint32_t nph = 0;
    vkEnumeratePhysicalDevices(inst, &nph, nullptr);
    std::vector<VkPhysicalDevice> phs(nph);
    vkEnumeratePhysicalDevices(inst, &nph, phs.data());
    VkPhysicalDevice gpu = phs[0];
    VkPhysicalDeviceProperties pp{};
    vkGetPhysicalDeviceProperties(gpu, &pp);
    fprintf(stderr, "GPU: %s  M=%u K=%u N=%u\n", pp.deviceName, M, K, N);

    // --- Queue family ---
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qps(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, qps.data());
    uint32_t qfam = 0;
    for (uint32_t i = 0; i < nq; i++) {
        if ((qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            (qps[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) { qfam = i; break; }
    }

    // --- Device ---
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqi{}; dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqi.queueFamilyIndex = qfam; dqi.queueCount = 1; dqi.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures pf{};
    VkDeviceCreateInfo dc{}; dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dqi; dc.pEnabledFeatures = &pf;
    VkDevice dev; VK_CHECK(vkCreateDevice(gpu, &dc, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfam, 0, &q);

    // --- Memory types ---
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
    uint32_t dev_mt = UINT32_MAX, host_mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) dev_mt = i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            host_mt = i; break;
        }
    }
    if (host_mt == UINT32_MAX) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                host_mt = i; break;
            }
        }
    }

    // --- Buffers ---
    VkDeviceSize szA = (VkDeviceSize)M * K * 4;
    VkDeviceSize szB = (VkDeviceSize)K * N * 4;
    VkDeviceSize szC = (VkDeviceSize)M * N * 4;

    auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags usage, uint32_t mt) ->
        std::pair<VkBuffer, VkDeviceMemory> {
        VkBufferCreateInfo bc{}; bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bc.size = sz; bc.usage = usage; bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf; VK_CHECK(vkCreateBuffer(dev, &bc, nullptr, &buf));
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkMemoryAllocateInfo ma{}; ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ma.allocationSize = mr.size; ma.memoryTypeIndex = mt;
        VkDeviceMemory mem; VK_CHECK(vkAllocateMemory(dev, &ma, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));
        return {buf, mem};
    };

    VkBufferUsageFlags gpu_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto [a_gpu, ma_gpu] = make_buf(szA, gpu_usage, dev_mt);
    auto [b_gpu, mb_gpu] = make_buf(szB, gpu_usage, dev_mt);
    auto [c_gpu, mc_gpu] = make_buf(szC, gpu_usage, dev_mt);

    auto [a_stg, ma_stg] = make_buf(szA, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mt);
    auto [b_stg, mb_stg] = make_buf(szB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mt);
    auto [c_stg, mc_stg] = make_buf(szC, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_mt);

    // Fill staging buffers
    srand(42);
    { float* p; VK_CHECK(vkMapMemory(dev, ma_stg, 0, szA, 0, (void**)&p));
      for (uint32_t i = 0; i < M * K; i++) p[i] = (float)(rand()) / RAND_MAX - 0.5f;
      vkUnmapMemory(dev, ma_stg); }
    { float* p; VK_CHECK(vkMapMemory(dev, mb_stg, 0, szB, 0, (void**)&p));
      for (uint32_t i = 0; i < K * N; i++) p[i] = (float)(rand()) / RAND_MAX - 0.5f;
      vkUnmapMemory(dev, mb_stg); }

    // --- Shader + pipeline ---
    auto code = read_file(spv_path);
    VkShaderModuleCreateInfo smc{}; smc.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smc.codeSize = code.size(); smc.pCode = (uint32_t*)code.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smc, nullptr, &sm));

    VkPushConstantRange pc{}; pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0; pc.size = 12;
    VkDescriptorSetLayoutBinding b0[3];
    for (int i = 0; i < 3; i++) {
        b0[i].binding = i;
        b0[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b0[i].descriptorCount = 1;
        b0[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsc{}; dsc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsc.bindingCount = 3; dsc.pBindings = b0;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &dsc, nullptr, &dsl));
    VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    uint32_t spec[] = {64, 64, 8};
    VkSpecializationMapEntry me[3];
    for (int i = 0; i < 3; i++) { me[i].constantID = i; me[i].offset = i * 4; me[i].size = 4; }
    VkSpecializationInfo si{}; si.mapEntryCount = 3; si.pMapEntries = me;
    si.dataSize = 12; si.pData = spec;
    VkComputePipelineCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = sm;
    pci.stage.pName = "main"; pci.stage.pSpecializationInfo = &si; pci.stage.pNext = nullptr;
    pci.layout = pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe));

    // --- Command pools + buffers (separate for H2D, GPU, D2H) ---
    VkCommandPoolCreateInfo cpc{}; cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.queueFamilyIndex = qfam; cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool_h2d, pool_gpu, pool_d2h;
    VK_CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &pool_h2d));
    VK_CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &pool_gpu));
    VK_CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &pool_d2h));

    VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    VkCommandBuffer cmd_h2d, cmd_gpu, cmd_d2h;
    cba.commandPool = pool_h2d; VK_CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd_h2d));
    cba.commandPool = pool_gpu; VK_CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd_gpu));
    cba.commandPool = pool_d2h; VK_CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd_d2h));

    // --- Descriptor set ---
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci{}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo dsai{}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[3];
    dbi[0].buffer = a_gpu; dbi[0].offset = 0; dbi[0].range = szA;
    dbi[1].buffer = b_gpu; dbi[1].offset = 0; dbi[1].range = szB;
    dbi[2].buffer = c_gpu; dbi[2].offset = 0; dbi[2].range = szC;
    VkWriteDescriptorSet wds[3] = {};
    for (int i = 0; i < 3; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ds; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 3, wds, 0, nullptr);

    // --- Timestamp query pool ---
    VkQueryPoolCreateInfo qpci{}; qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    VkQueryPool qpool; VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

    // --- Fence ---
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));

    // --- Command buffer recording helpers ---
    uint32_t gx = (M + 63) / 64, gy = (N + 63) / 64;

    auto record_h2d = [&]() {
        vkResetCommandBuffer(cmd_h2d, 0);
        VkCommandBufferBeginInfo cb_begin{}; cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd_h2d, &cb_begin));
        VkBufferCopy cpy[2];
        cpy[0].srcOffset = 0; cpy[0].dstOffset = 0; cpy[0].size = szA;
        cpy[1].srcOffset = 0; cpy[1].dstOffset = 0; cpy[1].size = szB;
        vkCmdCopyBuffer(cmd_h2d, a_stg, a_gpu, 2, cpy);
        VK_CHECK(vkEndCommandBuffer(cmd_h2d));
    };

    auto record_gpu = [&]() {
        vkResetCommandBuffer(cmd_gpu, 0);
        VkCommandBufferBeginInfo cb_begin{}; cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd_gpu, &cb_begin));
        vkCmdResetQueryPool(cmd_gpu, qpool, 0, 2);
        vkCmdWriteTimestamp(cmd_gpu, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
        vkCmdBindPipeline(cmd_gpu, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd_gpu, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        PushBlock pc{ M, K, N };
        vkCmdPushConstants(cmd_gpu, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushBlock), &pc);
        vkCmdDispatch(cmd_gpu, gx, gy, 1);
        vkCmdWriteTimestamp(cmd_gpu, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
        VK_CHECK(vkEndCommandBuffer(cmd_gpu));
    };

    auto record_d2h = [&]() {
        vkResetCommandBuffer(cmd_d2h, 0);
        VkCommandBufferBeginInfo cb_begin{}; cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd_d2h, &cb_begin));
        VkBufferCopy cpy{0, 0, szC};
        vkCmdCopyBuffer(cmd_d2h, c_gpu, c_stg, 1, &cpy);
        VK_CHECK(vkEndCommandBuffer(cmd_d2h));
    };

    record_h2d(); record_gpu(); record_d2h();

    // --- Submit helper ---
    auto submit_and_wait = [&](VkCommandBuffer cb) {
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK_CHECK(vkResetFences(dev, 1, &fence));
        VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    };

    // --- Warmup ---
    for (int i = 0; i < n_warmup; i++) {
        submit_and_wait(cmd_h2d);
        submit_and_wait(cmd_gpu);
        submit_and_wait(cmd_d2h);
    }

    VkPhysicalDeviceProperties pp2{};
    vkGetPhysicalDeviceProperties(gpu, &pp2);
    double ts_period = pp2.limits.timestampPeriod;

    // --- Timed loop ---
    std::vector<double> h2d_times, compute_times, d2h_times, total_times;
    for (int it = 0; it < n_iters; it++) {
        record_h2d(); record_gpu(); record_d2h();

        auto t0 = std::chrono::high_resolution_clock::now();
        submit_and_wait(cmd_h2d);
        auto t1 = std::chrono::high_resolution_clock::now();
        submit_and_wait(cmd_gpu);
        uint64_t ts[2];
        vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), (void*)ts,
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        double compute_us = (double)(ts[1] - ts[0]) * ts_period / 1000.0;
        auto t2 = std::chrono::high_resolution_clock::now();
        submit_and_wait(cmd_d2h);
        auto t3 = std::chrono::high_resolution_clock::now();

        h2d_times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        compute_times.push_back(compute_us);
        d2h_times.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
        total_times.push_back(std::chrono::duration<double, std::micro>(t3 - t0).count());
    }

    double ops = 2.0 * (double)M * (double)K * (double)N;
    double h2d_ms = median(h2d_times) / 1000.0;
    double compute_ms = median(compute_times) / 1000.0;
    double d2h_ms = median(d2h_times) / 1000.0;
    double total_ms = median(total_times) / 1000.0;
    double gops = ops / (compute_ms * 1e-3) / 1e9;

    printf("{\"status\":\"OK\",\"op\":\"GEMM\",\"M\":%u,\"K\":%u,\"N\":%u,"
           "\"h2d_ms\":%.3f,\"compute_ms\":%.3f,\"d2h_ms\":%.3f,"
           "\"total_ms\":%.3f,\"gops\":%.2f,"
           "\"total_ops\":%.0f,\"gmac\":%.2f}\n",
           M, K, N, h2d_ms, compute_ms, d2h_ms, total_ms, gops, ops, ops/1e9);

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyCommandPool(dev, pool_h2d, nullptr);
    vkDestroyCommandPool(dev, pool_gpu, nullptr);
    vkDestroyCommandPool(dev, pool_d2h, nullptr);
    vkDestroyBuffer(dev, a_gpu, nullptr); vkFreeMemory(dev, ma_gpu, nullptr);
    vkDestroyBuffer(dev, b_gpu, nullptr); vkFreeMemory(dev, mb_gpu, nullptr);
    vkDestroyBuffer(dev, c_gpu, nullptr); vkFreeMemory(dev, mc_gpu, nullptr);
    vkDestroyBuffer(dev, a_stg, nullptr); vkFreeMemory(dev, ma_stg, nullptr);
    vkDestroyBuffer(dev, b_stg, nullptr); vkFreeMemory(dev, mb_stg, nullptr);
    vkDestroyBuffer(dev, c_stg, nullptr); vkFreeMemory(dev, mc_stg, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
