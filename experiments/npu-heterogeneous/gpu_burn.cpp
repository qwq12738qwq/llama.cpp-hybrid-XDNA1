// gpu_burn.cpp -- Vulkan compute GEMM for GPU-NPU overlap experiment
//
// Usage: ./gpu_burn --width W --height H --kdim K [--warmup N] [--iters N]
//                   --ready-fifo FIFO --go-fifo FIFO --result-fifo FIFO [--read-back]
//
// Protocol:
//   1. Initializes Vulkan, allocates buffers, compiles shader, records CmdBuffer.
//   2. Runs warmup iterations (not timed).
//   3. Writes "READY" to ready-fifo.
//   4. Blocks reading from go-fifo until "GO" received.
//   5. Dispatches compute with Vulkan timestamps (vkCmdWriteTimestamp).
//   6. Waits for fence.
//   7. Writes timing results to result-fifo as CSV:
//      wall_start_ns,wall_end_ns,gpu_exec_ns,ts_period,ts_start,ts_end
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
#include <iostream>
#include <sys/stat.h>

static std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    size_t sz = (size_t)f.tellg();
    std::string buf(sz, '\0');
    f.seekg(0);
    f.read(&buf[0], sz);
    return buf;
}

struct PushBlock { uint32_t w, h, k; };

static void write_line(int fd, const std::string& s) {
    write(fd, s.c_str(), s.size());
    write(fd, "\n", 1);
}

static void reinit_cmd(VkCommandBuffer cmd_buf, VkQueryPool query_pool,
                       VkPipeline pipeline, VkPipelineLayout pl,
                       VkDescriptorSet ds, uint32_t gx, uint32_t gy,
                       uint32_t w, uint32_t h, uint32_t k) {
    vkResetCommandBuffer(cmd_buf, 0);
    VkCommandBufferBeginInfo cb_begin{};
    cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buf, &cb_begin);
    vkCmdResetQueryPool(cmd_buf, query_pool, 0, 2);
    vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                            0, 1, &ds, 0, nullptr);
    PushBlock pc{ w, h, k };
    vkCmdPushConstants(cmd_buf, pl, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PushBlock), &pc);
    vkCmdDispatch(cmd_buf, gx, gy, 1);
    vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
    vkEndCommandBuffer(cmd_buf);
}

int main(int argc, char** argv) {
    uint32_t width = 1024, height = 1024, k_dim = 1024;
    int n_warmup = 1, n_iters = 1;
    bool do_read_back = false;
    std::string ready_fifo, go_fifo, result_fifo;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) width = (uint32_t)atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = (uint32_t)atoi(argv[++i]);
        else if (arg == "--kdim" && i + 1 < argc) k_dim = (uint32_t)atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) n_warmup = atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (arg == "--ready-fifo" && i + 1 < argc) ready_fifo = argv[++i];
        else if (arg == "--go-fifo" && i + 1 < argc) go_fifo = argv[++i];
        else if (arg == "--result-fifo" && i + 1 < argc) result_fifo = argv[++i];
        else if (arg == "--read-back") do_read_back = true;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Vulkan instance ---
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "gpu_burn";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "gpu_burn";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> exts;
    VkInstanceCreateInfo ic_info{};
    ic_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic_info.pApplicationInfo = &app_info;
    ic_info.enabledExtensionCount = (uint32_t)exts.size();
    ic_info.ppEnabledExtensionNames = exts.data();

    VkInstance instance;
    if (vkCreateInstance(&ic_info, nullptr, &instance) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed\n"); return 1;
    }

    // --- Physical device (prefer integrated GPU) ---
    uint32_t n_phys;
    vkEnumeratePhysicalDevices(instance, &n_phys, nullptr);
    if (n_phys == 0) { fprintf(stderr, "No physical devices\n"); return 1; }
    std::vector<VkPhysicalDevice> phys(n_phys);
    vkEnumeratePhysicalDevices(instance, &n_phys, phys.data());

    VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties dev_props{};
    for (auto& pd : phys) {
        vkGetPhysicalDeviceProperties(pd, &dev_props);
        if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            phys_dev = pd; break;
        }
    }
    if (phys_dev == VK_NULL_HANDLE) {
        phys_dev = phys[0];
        vkGetPhysicalDeviceProperties(phys_dev, &dev_props);
    }

    float ts_period = dev_props.limits.timestampPeriod;
    fprintf(stderr, "GPU: %s  ts_period=%.2f ns\n", dev_props.deviceName, ts_period);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    // --- Queue family ---
    uint32_t qcount;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qcount, qprops.data());
    uint32_t q_idx = 0;
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { q_idx = i; break; }
    }

    // --- Logical device ---
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq_info{};
    dq_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dq_info.queueFamilyIndex = q_idx;
    dq_info.queueCount = 1;
    dq_info.pQueuePriorities = &prio;

    std::vector<const char*> dev_exts;
    VkDeviceCreateInfo dd_info{};
    dd_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dd_info.queueCreateInfoCount = 1;
    dd_info.pQueueCreateInfos = &dq_info;
    dd_info.enabledExtensionCount = (uint32_t)dev_exts.size();
    dd_info.ppEnabledExtensionNames = dev_exts.data();
    VkDevice device;
    if (vkCreateDevice(phys_dev, &dd_info, nullptr, &device) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDevice failed\n"); return 1;
    }
    VkQueue queue;
    vkGetDeviceQueue(device, q_idx, 0, &queue);

    // --- Buffers ---
    VkDeviceSize sz_a = (VkDeviceSize)height * k_dim * sizeof(float);
    VkDeviceSize sz_b = (VkDeviceSize)k_dim * width * sizeof(float);
    VkDeviceSize sz_c = (VkDeviceSize)height * width * sizeof(float);
    VkDeviceSize total = sz_a + sz_b + sz_c;

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = total;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer;
    if (vkCreateBuffer(device, &buf_info, nullptr, &buffer) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateBuffer failed\n"); return 1;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);
    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mem_type = i; break;
        }
    }
    if (mem_type == UINT32_MAX) {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((mem_req.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                mem_type = i; break;
            }
        }
    }
    if (mem_type == UINT32_MAX) { fprintf(stderr, "No host-visible memory\n"); return 1; }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = mem_type;
    VkDeviceMemory buf_mem;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &buf_mem) != VK_SUCCESS) {
        fprintf(stderr, "vkAllocateMemory failed\n"); return 1;
    }
    vkBindBufferMemory(device, buffer, buf_mem, 0);

    // Fill A and B with deterministic data
    void* mapped;
    vkMapMemory(device, buf_mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    float* p = (float*)mapped;
    srand(42);
    for (VkDeviceSize i = 0; i < height * k_dim; i++) p[i] = (float)(rand() % 100) / 99.0f;
    p += height * k_dim;
    for (VkDeviceSize i = 0; i < k_dim * width; i++) p[i] = (float)(rand() % 100) / 99.0f;
    p += k_dim * width;
    for (VkDeviceSize i = 0; i < height * width; i++) p[i] = 0.0f;
    vkUnmapMemory(device, buf_mem);

    // --- Descriptor set layout ---
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayout ds_layout;
    VkDescriptorSetLayoutCreateInfo dsl_info{};
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 3;
    dsl_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDescriptorSetLayout failed\n"); return 1;
    }

    VkDescriptorPoolSize dp_size{};
    dp_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dp_size.descriptorCount = 3;
    VkDescriptorPool ds_pool;
    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = 1;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &dp_size;
    if (vkCreateDescriptorPool(device, &dp_info, nullptr, &ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDescriptorPool failed\n"); return 1;
    }

    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo ds_alloc_info{};
    ds_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc_info.descriptorPool = ds_pool;
    ds_alloc_info.descriptorSetCount = 1;
    ds_alloc_info.pSetLayouts = &ds_layout;
    if (vkAllocateDescriptorSets(device, &ds_alloc_info, &ds) != VK_SUCCESS) {
        fprintf(stderr, "vkAllocateDescriptorSets failed\n"); return 1;
    }

    VkDescriptorBufferInfo buf_infos[3]{};
    buf_infos[0].buffer = buffer; buf_infos[0].offset = 0;            buf_infos[0].range = sz_a;
    buf_infos[1].buffer = buffer; buf_infos[1].offset = sz_a;         buf_infos[1].range = sz_b;
    buf_infos[2].buffer = buffer; buf_infos[2].offset = sz_a + sz_b;  buf_infos[2].range = sz_c;

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

    // --- Compute pipeline ---
    std::string shader_src = read_file("gemm.spv");
    VkShaderModuleCreateInfo sm_info{};
    sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_info.codeSize = shader_src.size();
    sm_info.pCode = (const uint32_t*)shader_src.data();
    VkShaderModule shader_mod;
    if (vkCreateShaderModule(device, &sm_info, nullptr, &shader_mod) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateShaderModule failed\n"); return 1;
    }

    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &ds_layout;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(PushBlock);
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &pc_range;
    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "vkCreatePipelineLayout failed\n"); return 1;
    }

    VkComputePipelineCreateInfo cp_info{};
    cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_info.stage.module = shader_mod;
    cp_info.stage.pName = "main";
    cp_info.layout = pipeline_layout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &pipeline) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateComputePipelines failed\n"); return 1;
    }

    // --- Command pool / buffer ---
    VkCommandPoolCreateInfo cp_pool_info{};
    cp_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_pool_info.queueFamilyIndex = q_idx;
    cp_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmd_pool;
    if (vkCreateCommandPool(device, &cp_pool_info, nullptr, &cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateCommandPool failed\n"); return 1;
    }
    VkCommandBufferAllocateInfo cb_alloc{};
    cb_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_alloc.commandPool = cmd_pool;
    cb_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd_buf;
    if (vkAllocateCommandBuffers(device, &cb_alloc, &cmd_buf) != VK_SUCCESS) {
        fprintf(stderr, "vkAllocateCommandBuffers failed\n"); return 1;
    }

    // --- Query pool ---
    VkQueryPoolCreateInfo qp_info{};
    qp_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp_info.queryCount = 2;
    VkQueryPool query_pool;
    if (vkCreateQueryPool(device, &qp_info, nullptr, &query_pool) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateQueryPool failed\n"); return 1;
    }

    // --- Fence ---
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateFence failed\n"); return 1;
    }

    uint32_t gx = (width + 15) / 16;
    uint32_t gy = (height + 15) / 16;

    // Initial command buffer recording
    reinit_cmd(cmd_buf, query_pool, pipeline, pipeline_layout, ds, gx, gy, width, height, k_dim);

    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t0).count();
    fprintf(stderr, "Init: %.1f ms  Workload: %ux%ux%u  groups=%ux%u\n",
            init_ms, height, k_dim, width, gx, gy);

    // --- Warmup ---
    for (int i = 0; i < n_warmup; i++) {
        vkResetFences(device, 1, &fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    }
    fprintf(stderr, "Warmup done (%d iters)\n", n_warmup);

    if (do_read_back) {
        if (vkMapMemory(device, buf_mem, sz_a + sz_b, sz_c, 0, &mapped) == VK_SUCCESS) {
            float* c = (float*)mapped;
            float sum = 0;
            for (uint32_t i = 0; i < height * width; i++) sum += c[i];
            vkUnmapMemory(device, buf_mem);
            fprintf(stderr, "Read-back: C[0]=%.6f  sum=%.6f\n", c[0], sum);
        } else {
            fprintf(stderr, "Read-back: vkMapMemory failed\n");
        }
    }

    // --- Signal READY ---
    if (!ready_fifo.empty()) {
        mkfifo(ready_fifo.c_str(), 0666);
        int fd = open(ready_fifo.c_str(), O_WRONLY);
        char buf[256];
        snprintf(buf, sizeof(buf), "READY %.1f", init_ms);
        write_line(fd, buf);
        close(fd);
        fprintf(stderr, "Signaled READY\n");
    }

    // --- Wait for GO ---
    if (!go_fifo.empty()) {
        mkfifo(go_fifo.c_str(), 0666);
        fprintf(stderr, "Waiting for GO...\n");
        int fd = open(go_fifo.c_str(), O_RDONLY);
        char buf[256];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        buf[n > 0 ? n : 0] = '\0';
        close(fd);
        fprintf(stderr, "Received GO: %s\n", buf);
    }

    // --- Execute ---
    reinit_cmd(cmd_buf, query_pool, pipeline, pipeline_layout, ds, gx, gy, width, height, k_dim);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    vkResetFences(device, 1, &fence);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf;
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    // Read GPU timestamps
    uint64_t timestamps[2];
    vkGetQueryPoolResults(device, query_pool, 0, 2,
                          sizeof(uint64_t) * 2, timestamps,
                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    uint64_t gpu_start_raw = timestamps[0];
    uint64_t gpu_end_raw = timestamps[1];
    uint64_t gpu_exec_ns = (gpu_end_raw - gpu_start_raw) * (uint64_t)(ts_period > 0 ? ts_period : 1);

    uint64_t wall_start_ns = (uint64_t)ts_start.tv_sec * 1000000000ULL + (uint64_t)ts_start.tv_nsec;
    uint64_t wall_end_ns = (uint64_t)ts_end.tv_sec * 1000000000ULL + (uint64_t)ts_end.tv_nsec;

    fprintf(stderr, "GPU exec (hw ts): %.3f ms  wall: %.3f ms\n",
            gpu_exec_ns / 1e6, (wall_end_ns - wall_start_ns) / 1e6);

    // Write results as CSV
    std::string result = std::to_string(wall_start_ns) + "," +
                         std::to_string(wall_end_ns) + "," +
                         std::to_string(gpu_exec_ns) + "," +
                         std::to_string(ts_period) + "," +
                         std::to_string(gpu_start_raw) + "," +
                         std::to_string(gpu_end_raw);

    if (!result_fifo.empty()) {
        mkfifo(result_fifo.c_str(), 0666);
        int fd = open(result_fifo.c_str(), O_WRONLY);
        write_line(fd, result);
        close(fd);
    }
    std::cout << result << std::endl;

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyQueryPool(device, query_pool, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(device, ds_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
    vkDestroyShaderModule(device, shader_mod, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, buf_mem, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
