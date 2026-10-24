// Phase 8-D Single-Process Concurrent Harness
#include <vulkan/vulkan.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/deprecated/xrt.h>
#include "/workspace/llama.cpp/ggml/src/ggml-xdna/stub/uuid/uuid.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)
struct TimingResult { double gpu_device_us = 0; double npu_device_us = 0; double wall_latency_us = 0; };
struct FrameResources { VkCommandBuffer cmd_buf; VkFence fence; bool in_use = false; };
class ConcurrentHarness {
public:
    VkInstance instance = VK_NULL_HANDLE; VkPhysicalDevice phys_dev = VK_NULL_HANDLE; VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE; uint32_t queue_family = 0; double timestamp_period = 0.0;
    VkBuffer a_buf = VK_NULL_HANDLE, b_buf = VK_NULL_HANDLE, c_buf = VK_NULL_HANDLE;
    VkDeviceMemory a_mem = VK_NULL_HANDLE, b_mem = VK_NULL_HANDLE, c_mem = VK_NULL_HANDLE;
    VkBuffer a_stg = VK_NULL_HANDLE, b_stg = VK_NULL_HANDLE, c_stg = VK_NULL_HANDLE;
    VkDeviceMemory a_stg_mem = VK_NULL_HANDLE, b_stg_mem = VK_NULL_HANDLE, c_stg_mem = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE; VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE; VkDescriptorPool desc_pool = VK_NULL_HANDLE; VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE; std::vector<FrameResources> frames; size_t frame_count = 4;
    xrt::device* xrt_dev = nullptr; xrt::hw_context* hw_ctx = nullptr; xrt::kernel* kernel = nullptr;
    xrt::bo* a_bo = nullptr; xrt::bo* b_bo = nullptr; xrt::bo* c_bo = nullptr; xrt::bo* inst_bo = nullptr;
    std::vector<uint32_t> insts; uint32_t M = 0, K = 0, N = 0; std::mt19937 rng;
    std::vector<int8_t> a_data; std::vector<int8_t> b_data; std::vector<int32_t> c_ref;
    ConcurrentHarness(uint32_t M_, uint32_t K_, uint32_t N_, const char* spv, const char* xclbin, const char* insts_path)
        : M(M_), K(K_), N(N_), rng(42) {
        init_gpu(spv); init_npu(xclbin, insts_path); generate_data(); upload_all();
    }
    void generate_data() {
        a_data.resize(M*K); b_data.resize(K*N); c_ref.resize(M*N);
        std::uniform_int_distribution<int> dist(-128, 127);
        for (size_t i = 0; i < a_data.size(); ++i) a_data[i] = (int8_t)dist(rng);
        for (size_t i = 0; i < b_data.size(); ++i) b_data[i] = (int8_t)dist(rng);
        for (uint32_t i = 0; i < M; ++i) for (uint32_t j = 0; j < N; ++j) {
            int32_t sum = 0; for (uint32_t k = 0; k < K; ++k) sum += (int32_t)a_data[i*K+k] * (int32_t)b_data[k*N+j];
            c_ref[i*N+j] = sum;
        }
    }
    void init_gpu(const char* spv_path) {
        VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
        uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(instance, &n, devs.data());
        phys_dev = devs[0]; VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys_dev, &props);
        fprintf(stderr, "[GPU] %s\n", props.deviceName);
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qn, qprops.data());
        for (uint32_t i = 0; i < qn; ++i) if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qprops[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) { queue_family = i; break; }
        float prio = 1.0f; VkDeviceQueueCreateInfo dqi{}; dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = queue_family; dqi.queueCount = 1; dqi.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures pf{}; VkDeviceCreateInfo dc{}; dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dqi; dc.pEnabledFeatures = &pf;
        VK_CHECK(vkCreateDevice(phys_dev, &dc, nullptr, &device)); vkGetDeviceQueue(device, queue_family, 0, &queue);
        VkPhysicalDeviceProperties pprops; vkGetPhysicalDeviceProperties(phys_dev, &pprops);
        timestamp_period = pprops.limits.timestampPeriod;
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys_dev, &mp);
        uint32_t dev_mem = UINT32_MAX, host_mem = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) dev_mem = i;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { host_mem = i; break; }
        auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags usage, uint32_t mt) {
            VkBufferCreateInfo bc{}; bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size = sz; bc.usage = usage; bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buf; VK_CHECK(vkCreateBuffer(device, &bc, nullptr, &buf));
            VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device, buf, &mr);
            VkMemoryAllocateInfo ma{}; ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ma.allocationSize = mr.size; ma.memoryTypeIndex = mt;
            VkDeviceMemory mem; VK_CHECK(vkAllocateMemory(device, &ma, nullptr, &mem)); VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
            return std::make_pair(buf, mem);
        };
        VkDeviceSize szA = (VkDeviceSize)M*K*4, szB = (VkDeviceSize)K*N*4, szC = (VkDeviceSize)M*N*4;
        auto [a_b, a_m] = make_buf(szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
        a_buf = a_b; a_mem = a_m;
        auto [b_b, b_m] = make_buf(szB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
        b_buf = b_b; b_mem = b_m;
        auto [c_b, c_m] = make_buf(szC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, dev_mem);
        c_buf = c_b; c_mem = c_m;
        auto [a_s, a_sm] = make_buf(szA, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
        a_stg = a_s; a_stg_mem = a_sm;
        auto [b_s, b_sm] = make_buf(szB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
        b_stg = b_s; b_stg_mem = b_sm;
        auto [c_s, c_sm] = make_buf(szC, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_mem);
        c_stg = c_s; c_stg_mem = c_sm;
        FILE* f = fopen(spv_path, "rb"); if (!f) { fprintf(stderr, "Failed SPV: %s\n", spv_path); exit(1); }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<char> code(sz); size_t rd = fread(code.data(), 1, sz, f); fclose(f);
        VkShaderModuleCreateInfo smc{}; smc.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smc.codeSize = sz; smc.pCode = (uint32_t*)code.data();
        VK_CHECK(vkCreateShaderModule(device, &smc, nullptr, &shader));
        VkPushConstantRange pc{}; pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc.offset = 0; pc.size = 12;
        VkDescriptorSetLayoutBinding bindings[3];
        for (int i = 0; i < 3; ++i) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo dslci{}; dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dslci.bindingCount = 3; dslci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &desc_layout));
        VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
        VkComputePipelineCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = shader; pci.stage.pName = "main"; pci.layout = pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));
        VkDescriptorPoolSize pool_sz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo dpci{}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_sz;
        VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool));
        VkDescriptorSetAllocateInfo dsai{}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool = desc_pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &desc_set));
        VkDescriptorBufferInfo dbi[3];
        dbi[0].buffer = a_buf; dbi[0].offset = 0; dbi[0].range = szA;
        dbi[1].buffer = b_buf; dbi[1].offset = 0; dbi[1].range = szB;
        dbi[2].buffer = c_buf; dbi[2].offset = 0; dbi[2].range = szC;
        VkWriteDescriptorSet wds[3] = {};
        for (int i = 0; i < 3; ++i) { wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[i].dstSet = desc_set; wds[i].dstBinding = i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
        vkUpdateDescriptorSets(device, 3, wds, 0, nullptr);
        VkCommandPoolCreateInfo cpc{}; cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpc.queueFamilyIndex = queue_family; cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &cpc, nullptr, &cmd_pool));
        frames.resize(frame_count);
        VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1; cba.commandPool = cmd_pool;
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < frame_count; ++i) {
            VK_CHECK(vkAllocateCommandBuffers(device, &cba, &frames[i].cmd_buf));
            VK_CHECK(vkCreateFence(device, &fci, nullptr, &frames[i].fence));
        }
        fprintf(stderr, "[GPU] Initialized M=%u K=%u N=%u\n", M, K, N);
    }
    void init_npu(const char* xclbin_path, const char* insts_path) {
        xrt_dev = new xrt::device(0);
        xrt::xclbin xclbin_obj{std::string(xclbin_path)};
        xrt_dev->register_xclbin(xclbin_obj);
        hw_ctx = new xrt::hw_context(*xrt_dev, xclbin_obj.get_uuid());
        kernel = new xrt::kernel(*hw_ctx, "MLIR_AIE");
        std::ifstream ifs(insts_path, std::ios::binary);
        if (!ifs) { fprintf(stderr, "Failed insts: %s\n", insts_path); exit(1); }
        ifs.seekg(0, std::ios::end); size_t isz = ifs.tellg(); ifs.seekg(0, std::ios::beg);
        insts.resize(isz / 4); ifs.read((char*)insts.data(), isz); ifs.close();
        inst_bo = new xrt::bo(*xrt_dev, isz, xrt::bo::flags::cacheable, kernel->group_id(1));
        std::memcpy(inst_bo->map(), insts.data(), isz); inst_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        a_bo = new xrt::bo(*xrt_dev, M*K*sizeof(int8_t), xrt::bo::flags::host_only, kernel->group_id(3));
        b_bo = new xrt::bo(*xrt_dev, K*N*sizeof(int8_t), xrt::bo::flags::host_only, kernel->group_id(4));
        c_bo = new xrt::bo(*xrt_dev, M*N*sizeof(int32_t), xrt::bo::flags::host_only, kernel->group_id(5));
        fprintf(stderr, "[NPU] Initialized\n");
    }
    void upload_all() {
        std::memcpy(a_bo->map(), a_data.data(), M*K*sizeof(int8_t));
        std::memcpy(b_bo->map(), b_data.data(), K*N*sizeof(int8_t));
        std::memset(c_bo->map(), 0, M*N*sizeof(int32_t));
        a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        int* p; VK_CHECK(vkMapMemory(device, a_stg_mem, 0, (VkDeviceSize)M*K*4, 0, (void**)&p));
        for (uint32_t i = 0; i < M*K; ++i) p[i] = a_data[i];
        vkUnmapMemory(device, a_stg_mem);
        VK_CHECK(vkMapMemory(device, b_stg_mem, 0, (VkDeviceSize)K*N*4, 0, (void**)&p));
        for (uint32_t i = 0; i < K*N; ++i) p[i] = b_data[i];
        vkUnmapMemory(device, b_stg_mem);
        VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1; cba.commandPool = cmd_pool;
        VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(device, &cba, &cb));
        VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
        VkBufferCopy copies[2] = {{0,0,(VkDeviceSize)M*K*4},{0,0,(VkDeviceSize)K*N*4}};
        vkCmdCopyBuffer(cb, a_stg, a_buf, 1, &copies[0]); vkCmdCopyBuffer(cb, b_stg, b_buf, 1, &copies[1]);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence; VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence)); VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, fence, nullptr); vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
        fprintf(stderr, "[GPU] Data uploaded\n");
    }
    FrameResources* acquire_frame() {
        for (size_t i = 0; i < frame_count; ++i) {
            VkResult r = vkGetFenceStatus(device, frames[i].fence);
            if (r == VK_SUCCESS) {
                vkResetFences(device, 1, &frames[i].fence);
                vkResetCommandBuffer(frames[i].cmd_buf, 0);
                frames[i].in_use = true;
                return &frames[i];
            }
        }
        return nullptr;
    }
    void submit_gpu(FrameResources* fr) {
        VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(fr->cmd_buf, &cbbi));
        vkCmdBindPipeline(fr->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(fr->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
        struct PushBlock { uint32_t M, K, N; } pc{ M, K, N };
        vkCmdPushConstants(fr->cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
        uint32_t gx = (N+15)/16, gy = (M+15)/16;
        vkCmdDispatch(fr->cmd_buf, gx, gy, 1);
        VK_CHECK(vkEndCommandBuffer(fr->cmd_buf));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &fr->cmd_buf;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fr->fence));
    }
    double wait_gpu(FrameResources* fr) {
        VK_CHECK(vkWaitForFences(device, 1, &fr->fence, VK_TRUE, UINT64_MAX));
        return 0.0;
    }
    double run_npu() {
        xrt::run run(*kernel);
        run.set_arg(3, *inst_bo); run.set_arg(4, (uint32_t)(insts.size()*4));
        run.set_arg(5, *a_bo); run.set_arg(6, *b_bo); run.set_arg(7, *c_bo);
        run.start(); run.wait();
        return 0.0;
    }
    std::vector<double> run_gpu_solo(int measured, int warmup) {
        std::vector<double> times;
        for (int i = 0; i < warmup+measured; ++i) {
            FrameResources* fr = acquire_frame();
            if (!fr) { fprintf(stderr, "No free GPU frame\n"); exit(1); }
            auto t0 = std::chrono::high_resolution_clock::now();
            submit_gpu(fr);
            wait_gpu(fr);
            double total = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now()-t0).count();
            if (i >= warmup) times.push_back(total);
            fr->in_use = false;
        }
        return times;
    }
    std::vector<double> run_npu_solo(int measured, int warmup) {
        std::vector<double> times;
        for (int i = 0; i < warmup+measured; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            xrt::run run(*kernel);
            run.set_arg(3, *inst_bo); run.set_arg(4, (uint32_t)(insts.size()*4));
            run.set_arg(5, *a_bo); run.set_arg(6, *b_bo); run.set_arg(7, *c_bo);
            run.start(); run.wait();
            double total = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now()-t0).count();
            if (i >= warmup) times.push_back(total);
        }
        return times;
    }
    std::vector<double> run_sequential(int measured, int warmup) {
        std::vector<double> times;
        for (int i = 0; i < warmup+measured; ++i) {
            FrameResources* fr = acquire_frame();
            auto t0 = std::chrono::high_resolution_clock::now();
            submit_gpu(fr); wait_gpu(fr);
            xrt::run run(*kernel);
            run.set_arg(3, *inst_bo); run.set_arg(4, (uint32_t)(insts.size()*4));
            run.set_arg(5, *a_bo); run.set_arg(6, *b_bo); run.set_arg(7, *c_bo);
            run.start(); run.wait();
            double total = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now()-t0).count();
            if (i >= warmup) times.push_back(total);
            fr->in_use = false;
        }
        return times;
    }
    std::vector<TimingResult> run_concurrent(int measured, int warmup) {
        std::vector<TimingResult> results;
        for (int i = 0; i < warmup+measured; ++i) {
            FrameResources* fr = acquire_frame();
            if (!fr) { fprintf(stderr, "No free GPU frame for concurrent\n"); exit(1); }
            TimingResult tr;
            auto t_start = std::chrono::high_resolution_clock::now();
            submit_gpu(fr);
            xrt::run run(*kernel);
            run.set_arg(3, *inst_bo); run.set_arg(4, (uint32_t)(insts.size()*4));
            run.set_arg(5, *a_bo); run.set_arg(6, *b_bo); run.set_arg(7, *c_bo);
            run.start();
            double gpu_dev = wait_gpu(fr);
            run.wait();
            auto t_end = std::chrono::high_resolution_clock::now();
            tr.gpu_device_us = gpu_dev;
            tr.wall_latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
            if (i >= warmup) results.push_back(tr);
            fr->in_use = false;
        }
        return results;
    }
    static std::map<std::string, double> stats(const std::vector<double>& samples) {
        std::map<std::string, double> s;
        if (samples.empty()) return s;
        std::vector<double> sorted = samples; std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        s["min"] = sorted[0]; s["max"] = sorted.back();
        s["mean"] = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
        s["median"] = sorted[n/2]; s["p95"] = sorted[size_t(n*0.95)]; s["p99"] = sorted[size_t(n*0.99)];
        return s;
    }
    static std::vector<double> extract_wall(const std::vector<TimingResult>& r) { std::vector<double> v; for (const auto& x : r) v.push_back(x.wall_latency_us); return v; }
};
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "Usage: %s <M> <K> <N> <spv> <xclbin> [insts]\n", argv[0]); return 1; }
    uint32_t M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const char* spv = argv[4], *xclbin = argv[5], *insts = argc > 6 ? argv[6] : nullptr;
    if (!insts) { fprintf(stderr, "insts required\n"); return 1; }
    ConcurrentHarness h(M, K, N, spv, xclbin, insts);
    int measured = 100, warmup = 20;
    fprintf(stderr, "\n=== GPU Solo ===\n");
    auto gpu_solo = ConcurrentHarness::stats(h.run_gpu_solo(measured, warmup));
    fprintf(stderr, "\n=== NPU Solo ===\n");
    auto npu_solo = ConcurrentHarness::stats(h.run_npu_solo(measured, warmup));
    fprintf(stderr, "\n=== Sequential ===\n");
    auto seq = ConcurrentHarness::stats(h.run_sequential(measured, warmup));
    fprintf(stderr, "\n=== Concurrent ===\n");
    auto conc_res = h.run_concurrent(measured, warmup);
    auto conc_wall = ConcurrentHarness::stats(ConcurrentHarness::extract_wall(conc_res));
    double speedup = seq.at("median") / conc_wall.at("median");
    double gpu_slowdown = conc_wall.at("median") / gpu_solo.at("median") - 1.0;
    printf("GPU solo median: %.2f us\n", gpu_solo.at("median"));
    printf("NPU solo median: %.2f us\n", npu_solo.at("median"));
    printf("Sequential median: %.2f us\n", seq.at("median"));
    printf("Concurrent median: %.2f us\n", conc_wall.at("median"));
    printf("Speedup: %.3fx\n", speedup);
    printf("GPU slowdown: %.1f%%\n", gpu_slowdown * 100.0);
    std::string verdict = (speedup >= 1.10 && gpu_slowdown <= 0.20) ? "PASS" : (speedup < 1.05 || gpu_slowdown > 0.20) ? "FAIL" : "CONDITIONAL";
    printf("Verdict: %s\n", verdict.c_str());
    return 0;
}
