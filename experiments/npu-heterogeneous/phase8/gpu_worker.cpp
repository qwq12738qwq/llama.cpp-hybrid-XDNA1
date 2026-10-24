#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

struct FrameResources {
    VkCommandBuffer cmd_buf;
    VkFence fence;
    VkQueryPool query_pool;
    bool in_use = false;
};

struct GpuWorker {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;

    VkBuffer a_buf = VK_NULL_HANDLE, b_buf = VK_NULL_HANDLE, c_buf = VK_NULL_HANDLE;
    VkDeviceMemory a_mem = VK_NULL_HANDLE, b_mem = VK_NULL_HANDLE, c_mem = VK_NULL_HANDLE;
    VkBuffer a_stg = VK_NULL_HANDLE, b_stg = VK_NULL_HANDLE, c_stg = VK_NULL_HANDLE;
    VkDeviceMemory a_stg_mem = VK_NULL_HANDLE, b_stg_mem = VK_NULL_HANDLE, c_stg_mem = VK_NULL_HANDLE;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;

    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    std::vector<FrameResources> frames;
    size_t frame_count = 4;

    uint32_t M = 0, K = 0, N = 0;
    double timestamp_period = 0.0;

    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::string> cmd_queue;
    std::map<uint64_t, FrameResources*> pending_waits;
    uint64_t next_iter_id = 1;
    bool quit = false;

    void init(uint32_t M_, uint32_t K_, uint32_t N_, const char* spv_path) {
        M = M_; K = K_; N = N_;
        create_instance();
        pick_physical_device();
        create_device();
        get_timestamp_period();
        create_buffers();
        create_shader(spv_path);
        create_pipeline();
        create_descriptors();
        create_command_pool();
        create_frames();
        record_init_commands();
        fprintf(stderr, "[GPU] Initialized M=%u K=%u N=%u\n", M, K, N);
    }

    void create_instance() {
        VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    }

    void pick_physical_device() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        for (auto d : devs) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                phys_dev = d;
                break;
            }
        }
        if (!phys_dev) phys_dev = devs[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_dev, &props);
        fprintf(stderr, "[GPU] Using device: %s\n", props.deviceName);
    }

    void create_device() {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &n, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(n);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &n, qprops.data());
        for (uint32_t i = 0; i < n; ++i) {
            if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qprops[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
                queue_family = i;
                break;
            }
        }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo dqi{}; dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = queue_family; dqi.queueCount = 1; dqi.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures pf{};
        VkDeviceCreateInfo dc{}; dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dqi; dc.pEnabledFeatures = &pf;
        VK_CHECK(vkCreateDevice(phys_dev, &dc, nullptr, &device));
        vkGetDeviceQueue(device, queue_family, 0, &queue);
    }

    void get_timestamp_period() {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_dev, &props);
        timestamp_period = props.limits.timestampPeriod;
    }

    void create_buffers() {
        VkDeviceSize szA = (VkDeviceSize)M * K * 4;
        VkDeviceSize szB = (VkDeviceSize)K * N * 4;
        VkDeviceSize szC = (VkDeviceSize)M * N * 4;

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mp);
        uint32_t dev_mem = UINT32_MAX, host_mem = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) dev_mem = i;
        }
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                host_mem = i; break;
            }
        }

        auto make_buf = [&](VkDeviceSize sz, VkBufferUsageFlags usage, uint32_t mem_type) {
            VkBufferCreateInfo bc{}; bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bc.size = sz; bc.usage = usage; bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buf; VK_CHECK(vkCreateBuffer(device, &bc, nullptr, &buf));
            VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device, buf, &mr);
            VkMemoryAllocateInfo ma{}; ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ma.allocationSize = mr.size; ma.memoryTypeIndex = mem_type;
            VkDeviceMemory mem; VK_CHECK(vkAllocateMemory(device, &ma, nullptr, &mem));
            VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
            return std::make_pair(buf, mem);
        };

        auto [a_b, a_m] = make_buf(szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
        a_buf = a_b; a_mem = a_m;
        auto [b_b, b_m] = make_buf(szB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
        b_buf = b_b; b_mem = b_m;
        auto [c_b, c_m] = make_buf(szC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, dev_mem);
        c_buf = c_b; c_mem = c_m;

        auto [a_s, a_sm] = make_buf(szA, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
        a_stg = a_s; a_stg_mem = a_sm;
        auto [b_s, b_sm] = make_buf(szB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
        b_stg = b_s; b_stg_mem = b_sm;
        auto [c_s, c_sm] = make_buf(szC, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_mem);
        c_stg = c_s; c_stg_mem = c_sm;

        // Fill staging buffers with deterministic data
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-128, 127);
        {
            int* p; VK_CHECK(vkMapMemory(device, a_stg_mem, 0, szA, 0, (void**)&p));
            for (uint32_t i = 0; i < M * K; ++i) p[i] = dist(rng);
            vkUnmapMemory(device, a_stg_mem);
        }
        {
            int* p; VK_CHECK(vkMapMemory(device, b_stg_mem, 0, szB, 0, (void**)&p));
            for (uint32_t i = 0; i < K * N; ++i) p[i] = dist(rng);
            vkUnmapMemory(device, b_stg_mem);
        }
        {
            int* p; VK_CHECK(vkMapMemory(device, c_stg_mem, 0, szC, 0, (void**)&p));
            for (uint32_t i = 0; i < M * N; ++i) p[i] = 0;
            vkUnmapMemory(device, c_stg_mem);
        }
    }

    void create_shader(const char* spv_path) {
        FILE* f = fopen(spv_path, "rb");
        if (!f) { fprintf(stderr, "Failed to open SPV: %s\n", spv_path); exit(1); }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<char> code(sz);
        fread(code.data(), 1, sz, f); fclose(f);
        VkShaderModuleCreateInfo smc{}; smc.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smc.codeSize = sz; smc.pCode = (uint32_t*)code.data();
        VK_CHECK(vkCreateShaderModule(device, &smc, nullptr, &shader));
    }

    void create_pipeline() {
        VkPushConstantRange pc{}; pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0; pc.size = 12;
        VkDescriptorSetLayoutBinding bindings[3];
        for (int i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{}; dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 3; dslci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &desc_layout));

        VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));

        VkComputePipelineCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = shader;
        pci.stage.pName = "main";
        pci.layout = pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));
    }

    void create_descriptors() {
        VkDescriptorPoolSize pool_sz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo dpci{}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_sz;
        VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool));

        VkDescriptorSetAllocateInfo dsai{}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = desc_pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &desc_set));

        VkDescriptorBufferInfo dbi[3];
        dbi[0].buffer = a_buf; dbi[0].offset = 0; dbi[0].range = (VkDeviceSize)M * K * 4;
        dbi[1].buffer = b_buf; dbi[1].offset = 0; dbi[1].range = (VkDeviceSize)K * N * 4;
        dbi[2].buffer = c_buf; dbi[2].offset = 0; dbi[2].range = (VkDeviceSize)M * N * 4;
        VkWriteDescriptorSet wds[3] = {};
        for (int i = 0; i < 3; ++i) {
            wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[i].dstSet = desc_set; wds[i].dstBinding = i; wds[i].descriptorCount = 1;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(device, 3, wds, 0, nullptr);
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo cpc{}; cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpc.queueFamilyIndex = queue_family; cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &cpc, nullptr, &cmd_pool));
    }

    void create_frames() {
        frames.resize(frame_count);
        VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1; cba.commandPool = cmd_pool;
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkQueryPoolCreateInfo qpci{}; qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
        for (size_t i = 0; i < frame_count; ++i) {
            cba.commandPool = cmd_pool;
            VK_CHECK(vkAllocateCommandBuffers(device, &cba, &frames[i].cmd_buf));
            VK_CHECK(vkCreateFence(device, &fci, nullptr, &frames[i].fence));
            VK_CHECK(vkCreateQueryPool(device, &qpci, nullptr, &frames[i].query_pool));
        }
    }

    void record_init_commands() {
        VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1; cba.commandPool = cmd_pool;
        VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(device, &cba, &cb));
        VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
        VkBufferCopy copies[2];
        copies[0] = {0, 0, (VkDeviceSize)M * K * 4};
        copies[1] = {0, 0, (VkDeviceSize)K * N * 4};
        vkCmdCopyBuffer(cb, a_stg, a_buf, 1, &copies[0]);
        vkCmdCopyBuffer(cb, b_stg, b_buf, 1, &copies[1]);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VkFence fence; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
        fprintf(stderr, "[GPU] Initial H2D copy done\n");
    }

    FrameResources* acquire_frame() {
        for (size_t i = 0; i < frame_count; ++i) {
            VkResult r = vkGetFenceStatus(device, frames[i].fence);
            if (r == VK_SUCCESS) {
                vkResetFences(device, 1, &frames[i].fence);
                vkResetCommandBuffer(frames[i].cmd_buf, 0);
                vkResetQueryPool(device, frames[i].query_pool, 0, 2);
                frames[i].in_use = true;
                return &frames[i];
            }
        }
        return nullptr;
    }

    void submit_compute(FrameResources* fr) {
        VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(fr->cmd_buf, &cbbi));
        vkCmdResetQueryPool(fr->cmd_buf, fr->query_pool, 0, 2);
        vkCmdWriteTimestamp(fr->cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, fr->query_pool, 0);
        vkCmdBindPipeline(fr->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(fr->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
        struct PushBlock { uint32_t M, K, N; } pc{ M, K, N };
        vkCmdPushConstants(fr->cmd_buf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
        uint32_t gx = (N + 15) / 16, gy = (M + 15) / 16;
        vkCmdDispatch(fr->cmd_buf, gx, gy, 1);
        vkCmdWriteTimestamp(fr->cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, fr->query_pool, 1);
        VK_CHECK(vkEndCommandBuffer(fr->cmd_buf));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &fr->cmd_buf;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fr->fence));
    }

    void wait_and_read_results(FrameResources* fr, double& device_us, double& wait_us, int& checksum) {
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        VK_CHECK(vkWaitForFences(device, 1, &fr->fence, VK_TRUE, UINT64_MAX));
        wait_us = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t_wait_start).count();

        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(device, fr->query_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        device_us = (double)(ts[1] - ts[0]) * timestamp_period / 1000.0;

        // Copy C to staging for checksum
        VkCommandBuffer cb = fr->cmd_buf;
        vkResetCommandBuffer(cb, 0);
        VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
        VkBufferCopy cpy{0, 0, (VkDeviceSize)M * N * 4};
        vkCmdCopyBuffer(cb, c_buf, c_stg, 1, &cpy);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VkFence tmp_fence; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &tmp_fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, tmp_fence));
        VK_CHECK(vkWaitForFences(device, 1, &tmp_fence, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, tmp_fence, nullptr);

        int* p; VK_CHECK(vkMapMemory(device, c_stg_mem, 0, (VkDeviceSize)M * N * 4, 0, (void**)&p));
        checksum = 0;
        for (uint32_t i = 0; i < M * N; ++i) checksum ^= p[i];
        vkUnmapMemory(device, c_stg_mem);
    }

    void run_command_loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            // Simple JSON parsing for commands
            if (line.find("\"submit\"") != std::string::npos && line.find("\"cmd\"") != std::string::npos) {
                FrameResources* fr = acquire_frame();
                if (!fr) {
                    printf("{\"error\":\"no free frame\"}\n"); fflush(stdout);
                    continue;
                }
                uint64_t iter = next_iter_id++;
                submit_compute(fr);
                double submit_us = 0; // we don't measure host submit separately here
                pending_waits[iter] = fr;
                printf("{\"event\":\"submitted\",\"iter\":%llu}\n", (unsigned long long)iter);
                fflush(stdout);
            } else if (line.find("\"wait\"") != std::string::npos && line.find("\"cmd\"") != std::string::npos) {
                // extract iter
                size_t pos = line.find("\"iter\":");
                if (pos == std::string::npos) {
                    printf("{\"error\":\"missing iter\"}\n"); fflush(stdout);
                    continue;
                }
                uint64_t iter = strtoull(line.c_str() + pos + 7, nullptr, 10);
                auto it = pending_waits.find(iter);
                if (it == pending_waits.end()) {
                    printf("{\"error\":\"unknown iter\"}\n"); fflush(stdout);
                    continue;
                }
                FrameResources* fr = it->second;
                double device_us, wait_us; int checksum;
                wait_and_read_results(fr, device_us, wait_us, checksum);
                fr->in_use = false;
                pending_waits.erase(it);
                printf("{\"event\":\"completed\",\"iter\":%llu,\"device_us\":%.3f,\"wait_us\":%.3f,\"checksum\":%d}\n",
                    (unsigned long long)iter, device_us, wait_us, checksum);
                fflush(stdout);
            } else if (line.find("\"cmd\":\"quit\"") != std::string::npos) {
                break;
            }
        }
    }

    ~GpuWorker() {
        for (auto& fr : frames) {
            if (fr.fence) vkDestroyFence(device, fr.fence, nullptr);
            if (fr.query_pool) vkDestroyQueryPool(device, fr.query_pool, nullptr);
        }
        if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
        if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
        if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (c_buf) vkDestroyBuffer(device, c_buf, nullptr); if (c_mem) vkFreeMemory(device, c_mem, nullptr);
        if (b_buf) vkDestroyBuffer(device, b_buf, nullptr); if (b_mem) vkFreeMemory(device, b_mem, nullptr);
        if (a_buf) vkDestroyBuffer(device, a_buf, nullptr); if (a_mem) vkFreeMemory(device, a_mem, nullptr);
        if (c_stg) vkDestroyBuffer(device, c_stg, nullptr); if (c_stg_mem) vkFreeMemory(device, c_stg_mem, nullptr);
        if (b_stg) vkDestroyBuffer(device, b_stg, nullptr); if (b_stg_mem) vkFreeMemory(device, b_stg_mem, nullptr);
        if (a_stg) vkDestroyBuffer(device, a_stg, nullptr); if (a_stg_mem) vkFreeMemory(device, a_stg_mem, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <M> <K> <N> <spv_path>\n", argv[0]);
        return 1;
    }
    uint32_t M = atoi(argv[1]);
    uint32_t K = atoi(argv[2]);
    uint32_t N = atoi(argv[3]);
    const char* spv = argv[4];

    GpuWorker worker;
    worker.init(M, K, N, spv);
    worker.run_command_loop();
    return 0;
}