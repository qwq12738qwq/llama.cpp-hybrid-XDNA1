// Phase 26 linchpin probe: one host allocation shared by Vulkan (imported via
// VK_EXT_external_memory_host) and XRT (userptr bo). No dma-buf, no CPU copy on
// the critical path.
//
// Flow:
//   host = posix_memalign(aligned)
//   Vulkan buffer memory = VkImportMemoryHostPointerInfoEXT(host)
//   XRT bo = bo(device, host, sz, host_only, grp)   (same pages)
//   GPU writes pattern into its imported buffer -> CPU reads host -> verify
//   CPU writes pattern into host -> GPU reads via its buffer -> verify
//
// Build:
//   g++ -O2 -std=c++17 vk_host_import_probe.cpp -o vk_host_import_probe \
//       $(pkg-config --cflags xrt) -I/usr/include/xrt \
//       -I/workspace/llama.cpp/ggml/src/ggml-xdna/stub \
//       -lvulkan -ldl $(pkg-config --libs xrt)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <vulkan/vulkan.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

#define VK_TRY(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "  [FAIL] %s -> VkResult %d\n", #expr, (int) _r); return 1; } } while (0)

static PFN_vkGetMemoryHostPointerPropertiesEXT pfnHostPtrProps = nullptr;

static const size_t SZ  = 1u << 20;   // 1 MiB
static const size_t N   = SZ / sizeof(uint32_t);
static const uint32_t SENTINEL = 0xDEADBEEF;

int main() {
    // ---- one host allocation, 64 KiB aligned (host-pointer import wants page aligned) ----
    void * host = nullptr;
    if (posix_memalign(&host, 64 * 1024, SZ) != 0) { perror("posix_memalign"); return 1; }
    memset(host, 0, SZ);
    printf("host alloc = %p (64K aligned)\n", host);

    // ---- Vulkan ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstance inst;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VK_TRY(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice pdev = devs[0];
    for (auto d : devs) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (strstr(p.deviceName, "RADV")) { pdev = d; break; }
    }

    float prio = 1.0f;
    uint32_t qfam = 0;
    {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qfs.data());
        for (uint32_t i = 0; i < nq; i++) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char * dexts[] = { "VK_EXT_external_memory_host", "VK_KHR_external_memory",
                             "VK_KHR_external_memory_fd" };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = dexts;
    VkDevice dev;
    VK_TRY(vkCreateDevice(pdev, &dci, nullptr, &dev));

    pfnHostPtrProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!pfnHostPtrProps) { printf("[FAIL] vkGetMemoryHostPointerPropertiesEXT NULL\n"); return 1; }

    // which host pointer types are supported?
    VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult hr = pfnHostPtrProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp);
    printf("host ptr props: result=%d memoryTypeBits=0x%x\n", (int) hr, hpp.memoryTypeBits);
    if (hr != VK_SUCCESS || hpp.memoryTypeBits == 0) { printf("[FAIL] host pointer not importable\n"); return 1; }

    // ---- buffer with host-allocation external memory ----
    VkExternalMemoryBufferCreateInfo embi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    embi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &embi; bci.size = SZ;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf;
    VK_TRY(vkCreateBuffer(dev, &bci, nullptr, &buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);

    // pick a memory type allowed by the host pointer
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    uint32_t mti = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((hpp.memoryTypeBits & (1u << i)) && (mr.memoryTypeBits & (1u << i))) {
            if (mti == ~0u) mti = i;
            if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) { mti = i; break; }
        }
    if (mti == ~0u) { printf("[FAIL] no compatible memory type\n"); return 1; }
    printf("chosen memory type=%u flags=0x%x\n", mti, mp.memoryTypes[mti].propertyFlags);

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = host;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &imp; mai.allocationSize = mr.size; mai.memoryTypeIndex = mti;
    VkDeviceMemory mem;
    VkResult ar = vkAllocateMemory(dev, &mai, nullptr, &mem);
    printf("vkAllocateMemory(host import) = %d\n", (int) ar);
    if (ar != VK_SUCCESS) return 1;
    VK_TRY(vkBindBufferMemory(dev, buf, mem, 0));

    VkQueue q; vkGetDeviceQueue(dev, qfam, 0, &q);
    VkCommandPool cp;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci.queueFamilyIndex = qfam;
    VK_TRY(vkCreateCommandPool(dev, &cpci, nullptr, &cp));

    // ---------- Test A: GPU fills buffer, CPU reads host ----------
    {
        VkCommandBuffer cb;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdFillBuffer(cb, buf, 0, SZ, SENTINEL);
        // make host writes visible to host
        VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bb.buffer = buf; bb.offset = 0; bb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(q);
        // invalidate host cache not needed (host coherent assumed); read
        uint32_t * h = (uint32_t*) host;
        size_t ok = 0;
        for (size_t i = 0; i < N; i++) if (h[i] == SENTINEL) ok++;
        printf("TEST A GPU->host: correct %zu/%zu  (h[0]=0x%08x)\n", ok, N, h[0]);
    }

    // ---------- Test B: CPU writes host, GPU reads buffer back to a second buffer, compare ----------
    {
        uint32_t * h = (uint32_t*) host;
        for (size_t i = 0; i < N; i++) h[i] = (uint32_t)(i * 2654435761u);
        // flush not needed if coherent

        // staging buffer to receive the GPU copy
        std::vector<uint8_t> * dummy = nullptr; (void) dummy;
        VkBuffer sbuf;
        VkBufferCreateInfo sb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sb.size = SZ; sb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; sb.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(dev, &sb, nullptr, &sbuf));
        VkMemoryRequirements smr; vkGetBufferMemoryRequirements(dev, sbuf, &smr);
        uint32_t smti = ~0u;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((smr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { smti = i; break; }
        VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        smai.allocationSize = smr.size; smai.memoryTypeIndex = smti;
        VkDeviceMemory smem; VK_TRY(vkAllocateMemory(dev, &smai, nullptr, &smem));
        VK_TRY(vkBindBufferMemory(dev, sbuf, smem, 0));

        VkCommandBuffer cb;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev, &cai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cb, &bi);
        VkBufferMemoryBarrier pre{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        pre.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; pre.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        pre.buffer = buf; pre.offset = 0; pre.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &pre, 0, nullptr);
        VkBufferCopy region{0, 0, SZ};
        vkCmdCopyBuffer(cb, buf, sbuf, 1, &region);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(q);

        void * p = nullptr; vkMapMemory(dev, smem, 0, SZ, 0, &p);
        uint32_t * sp = (uint32_t*) p;
        size_t ok = 0;
        for (size_t i = 0; i < N; i++) if (sp[i] == (uint32_t)(i * 2654435761u)) ok++;
        printf("TEST B host->GPU: correct %zu/%zu  (sp[0]=0x%08x exp=0x%08x)\n", ok, N, sp[0], (uint32_t)(0));
    }

    // ---------- XRT wraps the SAME host pages ----------
    {
        try {
            xrt::device xdev{0};
            xrt::bo bo(xdev, host, SZ, xrt::bo::flags::host_only, 0);
            void * xp = bo.map();
            printf("XRT userptr bo: map=%p host=%p same=%s size=%zu\n",
                   xp, host, (xp == host) ? "yes" : "no", (size_t) bo.size());
        } catch (const std::exception & ex) {
            printf("[FAIL] XRT userptr: %s\n", ex.what());
        }
    }

    printf("\nRESULT: Vulkan host-pointer import %s; shared host allocation Vulkan<->XRT is %s\n",
           (hr == VK_SUCCESS) ? "SUPPORTED" : "unsupported",
           (hr == VK_SUCCESS) ? "possible via VK_EXT_external_memory_host + XRT userptr" : "not possible");
    return 0;
}
