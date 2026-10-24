#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "VK error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)
int main() {
    VkInstance instance;
    VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(instance, &n, devs.data());
    VkPhysicalDevice phys_dev = devs[0];
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qn, qprops.data());
    uint32_t qf = 0; for (uint32_t i = 0; i < qn; ++i) if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    float prio = 1.0f; VkDeviceQueueCreateInfo dqi{}; dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqi.queueFamilyIndex = qf; dqi.queueCount = 1; dqi.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures pf{}; VkDeviceCreateInfo dc{}; dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dqi; dc.pEnabledFeatures = &pf;
    VkDevice device; VK_CHECK(vkCreateDevice(phys_dev, &dc, nullptr, &device));
    VkQueue queue; vkGetDeviceQueue(device, qf, 0, &queue);

    uint32_t M = 1, K = 1024, N = 16;
    VkDeviceSize szA = (VkDeviceSize)M * K * 4;
    VkDeviceSize szB = (VkDeviceSize)K * N * 4;
    VkDeviceSize szC = (VkDeviceSize)M * N * 4;

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

    auto [a_b, a_m] = make_buf(szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
    auto [b_b, b_m] = make_buf(szB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, dev_mem);
    auto [c_b, c_m] = make_buf(szC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, dev_mem);
    auto [a_s, a_sm] = make_buf(szA, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
    auto [b_s, b_sm] = make_buf(szB, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_mem);
    auto [c_s, c_sm] = make_buf(szC, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_mem);

    int* p; VK_CHECK(vkMapMemory(device, a_sm, 0, szA, 0, (void**)&p)); memset(p, 0xAB, szA); vkUnmapMemory(device, a_sm);
    VK_CHECK(vkMapMemory(device, b_sm, 0, szB, 0, (void**)&p)); memset(p, 0xCD, szB); vkUnmapMemory(device, b_sm);
    VK_CHECK(vkMapMemory(device, c_sm, 0, szC, 0, (void**)&p)); memset(p, 0, szC); vkUnmapMemory(device, c_sm);

    FILE* f = fopen("gemm_i8.spv", "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> code(sz); fread(code.data(), 1, sz, f); fclose(f);
    VkShaderModuleCreateInfo smc{}; smc.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smc.codeSize = sz; smc.pCode = (uint32_t*)code.data();
    VkShaderModule shader; VK_CHECK(vkCreateShaderModule(device, &smc, nullptr, &shader));

    VkPushConstantRange pc{}; pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc.offset = 0; pc.size = 12;
    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; ++i) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{}; dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dslci.bindingCount = 3; dslci.pBindings = bindings;
    VkDescriptorSetLayout desc_layout; VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &desc_layout));

    VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VkPipelineLayout pipeline_layout; VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));

    VkComputePipelineCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = shader; pci.stage.pName = "main"; pci.layout = pipeline_layout;
    VkPipeline pipeline; VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));

    VkDescriptorPoolSize pool_sz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci{}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_sz;
    VkDescriptorPool desc_pool; VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool));
    VkDescriptorSetAllocateInfo dsai{}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool = desc_pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &desc_layout;
    VkDescriptorSet desc_set; VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &desc_set));

    VkDescriptorBufferInfo dbi[3];
    dbi[0].buffer = a_b; dbi[0].offset = 0; dbi[0].range = szA;
    dbi[1].buffer = b_b; dbi[1].offset = 0; dbi[1].range = szB;
    dbi[2].buffer = c_b; dbi[2].offset = 0; dbi[2].range = szC;
    VkWriteDescriptorSet wds[3] = {};
    for (int i = 0; i < 3; ++i) { wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[i].dstSet = desc_set; wds[i].dstBinding = i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
    vkUpdateDescriptorSets(device, 3, wds, 0, nullptr);

    VkCommandPoolCreateInfo cpc{}; cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpc.queueFamilyIndex = qf; cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmd_pool; VK_CHECK(vkCreateCommandPool(device, &cpc, nullptr, &cmd_pool));

    VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1; cba.commandPool = cmd_pool;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(device, &cba, &cb));

    VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    VkBufferCopy copies[2] = {{0,0,szA},{0,0,szB}};
    vkCmdCopyBuffer(cb, a_s, a_b, 2, copies);
    vkCmdCopyBuffer(cb, b_s, b_b, 1, &copies[1]);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
    fprintf(stderr, "H2D done\n");

    VkFenceCreateInfo fci2{}; fci2.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence2; VK_CHECK(vkCreateFence(device, &fci2, nullptr, &fence2));
    VkCommandBufferAllocateInfo cba2{}; cba2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba2.commandBufferCount = 1; cba2.commandPool = cmd_pool;
    VkCommandBuffer cb2; VK_CHECK(vkAllocateCommandBuffers(device, &cba2, &cb2));
    VkCommandBufferBeginInfo cbbi2{}; cbbi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb2, &cbbi2));
    vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
    struct PB { uint32_t M, K, N; } pc{M,K,N};
    vkCmdPushConstants(cb2, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
    uint32_t gx = (N+15)/16, gy = (M+15)/16;
    vkCmdDispatch(cb2, gx, gy, 1);
    VK_CHECK(vkEndCommandBuffer(cb2));
    VkSubmitInfo si2{}; si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si2.commandBufferCount = 1; si2.pCommandBuffers = &cb2;
    VK_CHECK(vkQueueSubmit(queue, 1, &si2, fence2));
    VK_CHECK(vkWaitForFences(device, 1, &fence2, VK_TRUE, UINT64_MAX));
    fprintf(stderr, "Compute done\n");

    VkCommandBufferAllocateInfo cba3{}; cba3.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cba3.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba3.commandBufferCount = 1; cba3.commandPool = cmd_pool;
    VkCommandBuffer cb3; VK_CHECK(vkAllocateCommandBuffers(device, &cba3, &cb3));
    VkCommandBufferBeginInfo cbbi3{}; cbbi3.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb3, &cbbi3));
    VkBufferCopy cpy{0,0,szC};
    vkCmdCopyBuffer(cb3, c_buf, c_stg, 1, &cpy);
    VK_CHECK(vkEndCommandBuffer(cb3));
    VkSubmitInfo si3{}; si3.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si3.commandBufferCount = 1; si3.pCommandBuffers = &cb3;
    VkFenceCreateInfo fci3{}; fci3.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence3; VK_CHECK(vkCreateFence(device, &fci3, nullptr, &fence3));
    VK_CHECK(vkQueueSubmit(queue, 1, &si3, fence3));
    VK_CHECK(vkWaitForFences(device, 1, &fence3, VK_TRUE, UINT64_MAX));

    int* presult; VK_CHECK(vkMapMemory(device, c_stg_mem, 0, szC, 0, (void**)&presult));
    fprintf(stderr, "Result[0]=%d\n", presult[0]);
    vkUnmapMemory(device, c_stg_mem);

    vkDestroyFence(device, fence2, nullptr); vkDestroyFence(device, fence3, nullptr);
    vkFreeCommandBuffers(device, cmd_pool, 1, &cb2); vkFreeCommandBuffers(device, cmd_pool, 1, &cb3);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyBuffer(device, c_stg, nullptr); vkFreeMemory(device, c_stg_mem, nullptr);
    vkDestroyBuffer(device, b_stg, nullptr); vkFreeMemory(device, b_stg_mem, nullptr);
    vkDestroyBuffer(device, a_stg, nullptr); vkFreeMemory(device, a_stg_mem, nullptr);
    vkDestroyBuffer(device, c_buf, nullptr); vkFreeMemory(device, c_mem, nullptr);
    vkDestroyBuffer(device, b_buf, nullptr); vkFreeMemory(device, b_mem, nullptr);
    vkDestroyBuffer(device, a_buf, nullptr); vkFreeMemory(device, a_mem, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
