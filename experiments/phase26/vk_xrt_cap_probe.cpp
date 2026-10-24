// Phase 26 capability probe: can a Vulkan allocation and an XRT (XDNA) buffer
// share one backing allocation on this stack?
//
// Stages:
//   1. Vulkan device external-memory capabilities (dma_buf, host pointer)
//   2. Vulkan buffer allocated with DMA_BUF export -> vkGetMemoryFdKHR -> fd
//   3. XRT bo export_buffer() -> fd
//   4. XRT import a foreign fd (the Vulkan dma-buf fd) via xrt::bo(device, fd)
//   5. XRT userptr bo over a Vulkan-mapped host pointer (same physical pages)
//
// Build:
//   g++ -O2 -std=c++17 vk_xrt_cap_probe.cpp -o vk_xrt_cap_probe \
//       $(pkg-config --cflags --libs xrt) -lvulkan -ldl
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <dlfcn.h>

#include <vulkan/vulkan.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

#define VK_TRY(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "  [FAIL] %s -> VkResult %d\n", #expr, (int) _r); return false; } } while (0)

static PFN_vkGetMemoryFdKHR             pfnGetMemoryFdKHR = nullptr;
static PFN_vkGetMemoryFdPropertiesKHR   pfnGetMemoryFdPropertiesKHR = nullptr;
static PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR pfnExtBufProps = nullptr;

static VkInstance       g_inst = VK_NULL_HANDLE;
static VkPhysicalDevice g_pdev = VK_NULL_HANDLE;
static VkDevice         g_dev  = VK_NULL_HANDLE;

static bool stage1_vulkan_ext_caps() {
    printf("[stage 1] Vulkan external-memory capabilities\n");

    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> iexts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, iexts.data());

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> iext_names;
    for (auto & e : iexts) {
        if (strcmp(e.extensionName, "VK_KHR_external_memory_capabilities") == 0 ||
            strcmp(e.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
            iext_names.push_back(e.extensionName);
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t) iext_names.size();
    ci.ppEnabledExtensionNames = iext_names.data();
    VK_TRY(vkCreateInstance(&ci, nullptr, &g_inst));

    pfnExtBufProps = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)
        vkGetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(g_inst, &ndev, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (strstr(p.deviceName, "RADV") || strstr(p.deviceName, "AMD")) { g_pdev = d; break; }
    }
    if (!g_pdev) g_pdev = devs[0];

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(g_pdev, &p);
    printf("  device: %s\n", p.deviceName);

    uint32_t ndext = 0;
    vkEnumerateDeviceExtensionProperties(g_pdev, nullptr, &ndext, nullptr);
    std::vector<VkExtensionProperties> dexts(ndext);
    vkEnumerateDeviceExtensionProperties(g_pdev, nullptr, &ndext, dexts.data());
    std::vector<const char*> wanted = {
        "VK_KHR_external_memory", "VK_KHR_external_memory_fd",
        "VK_EXT_external_memory_dma_buf", "VK_EXT_external_memory_host",
    };
    std::vector<const char*> dext_names;
    for (auto & w : wanted) {
        bool found = false;
        for (auto & e : dexts) if (strcmp(e.extensionName, w) == 0) { found = true; break; }
        printf("  ext %-34s : %s\n", w, found ? "YES" : "no");
        if (found) dext_names.push_back(w);
    }

    // Query dma_buf external features for a plain buffer.
    if (pfnExtBufProps) {
        VkPhysicalDeviceExternalBufferInfo bi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkExternalBufferProperties bp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        pfnExtBufProps(g_pdev, &bi, &bp);
        VkExternalMemoryFeatureFlags f = bp.externalMemoryProperties.externalMemoryFeatures;
        printf("  dma_buf features: export=%d import=%d dedicated_only=%d\n",
               !!(f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT),
               !!(f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT),
               !!(f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT));

        VkPhysicalDeviceExternalBufferInfo bi2 = bi;
        bi2.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkExternalBufferProperties bp2{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        pfnExtBufProps(g_pdev, &bi2, &bp2);
        VkExternalMemoryFeatureFlags f2 = bp2.externalMemoryProperties.externalMemoryFeatures;
        printf("  host_ptr features: export=%d import=%d\n",
               !!(f2 & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT),
               !!(f2 & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT));
    } else {
        printf("  vkGetPhysicalDeviceExternalBufferPropertiesKHR: NULL\n");
    }

    // Create device with the external memory extensions.
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    uint32_t qfam = 0;
    {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(g_pdev, &nq, qfs.data());
        for (uint32_t i = 0; i < nq; i++) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t) dext_names.size();
    dci.ppEnabledExtensionNames = dext_names.data();
    VK_TRY(vkCreateDevice(g_pdev, &dci, nullptr, &g_dev));

    pfnGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(g_dev, "vkGetMemoryFdKHR");
    pfnGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(g_dev, "vkGetMemoryFdPropertiesKHR");
    printf("  vkGetMemoryFdKHR=%p vkGetMemoryFdPropertiesKHR=%p\n",
           (void*) pfnGetMemoryFdKHR, (void*) pfnGetMemoryFdPropertiesKHR);
    return true;
}

// Allocate a Vulkan buffer with dma-buf export enabled, bind memory, export fd.
static bool stage2_vulkan_export_fd(int * out_fd) {
    printf("[stage 2] Vulkan buffer -> dma-buf fd\n");
    const VkDeviceSize SZ = 1u << 20;

    VkExternalMemoryBufferCreateInfo embi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    embi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &embi;
    bci.size = SZ;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf;
    VK_TRY(vkCreateBuffer(g_dev, &bci, nullptr, &buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, buf, &mr);

    VkExportMemoryAllocateInfo emai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &emai;
    mai.allocationSize = mr.size;
    // prefer device-local, fall back to any
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_pdev, &mp);
    uint32_t mti = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mti = i; break; }
    if (mti == ~0u)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if (mr.memoryTypeBits & (1u << i)) { mti = i; break; }
    mai.memoryTypeIndex = mti;
    printf("  buffer mem type=%u size=%zu usage storage\n", mti, (size_t) mr.size);

    VkDeviceMemory mem;
    VK_TRY(vkAllocateMemory(g_dev, &mai, nullptr, &mem));

    if (!pfnGetMemoryFdKHR) { printf("  [FAIL] vkGetMemoryFdKHR NULL\n"); return false; }
    VkMemoryGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gfi.memory = mem;
    gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    int fd = -1;
    VkResult r = pfnGetMemoryFdKHR(g_dev, &gfi, &fd);
    if (r != VK_SUCCESS) { printf("  [FAIL] vkGetMemoryFdKHR -> %d\n", (int) r); return false; }
    printf("  exported dma-buf fd = %d\n", fd);
    *out_fd = fd;
    return true;
}

// XRT: allocate a host_only BO and export it.
static bool stage3_xrt_export_fd(int * out_fd) {
    printf("[stage 3] XRT bo export_buffer() -> handle\n");
    try {
        static xrt::device dev{0};
        xrt::bo bo(dev, 1u << 20, xrt::bo::flags::host_only, 0);
        auto h = bo.export_buffer();
        printf("  xrt export handle = %d (sizeof=%zu)\n", (int) h, sizeof(h));
        *out_fd = (int) h;
        return true;
    } catch (const std::exception & ex) {
        printf("  [FAIL] %s\n", ex.what());
        return false;
    }
}

// XRT: import a foreign fd (the Vulkan dma-buf) as an XRT bo.
static bool stage4_xrt_import_fd(int vk_fd) {
    printf("[stage 4] XRT import foreign fd %d\n", vk_fd);
    if (vk_fd < 0) { printf("  [SKIP] no fd\n"); return false; }
    try {
        static xrt::device dev{0};
        xrt::bo bo(dev, (xrt::bo::export_handle) vk_fd);
        auto * p = bo.map();
        printf("  imported OK, mapped=%p size=%zu\n", (void*) p, (size_t) bo.size());
        return true;
    } catch (const std::exception & ex) {
        printf("  [FAIL] %s\n", ex.what());
        return false;
    }
}

// XRT: wrap a Vulkan host-mapped pointer as a userptr BO.
static bool stage5_xrt_userptr() {
    printf("[stage 5] XRT userptr bo over a host pointer\n");
    const size_t SZ = 1u << 20;
    void * host = nullptr;
    if (posix_memalign(&host, 4096, SZ) != 0) { printf("  [FAIL] posix_memalign\n"); return false; }
    memset(host, 0x5a, SZ);
    try {
        static xrt::device dev{0};
        xrt::bo bo(dev, host, SZ, xrt::bo::flags::host_only, 0);
        void * p = bo.map();
        printf("  userptr bo OK map=%p host=%p same_pages=%s size=%zu\n",
               p, host, (p == host) ? "yes" : "no(map differs)", (size_t) bo.size());
        return true;
    } catch (const std::exception & ex) {
        printf("  [FAIL] %s\n", ex.what());
        return false;
    }
}

int main() {
    bool s1 = stage1_vulkan_ext_caps();
    int vk_fd = -1, xrt_fd = -1;
    bool s2 = s1 && stage2_vulkan_export_fd(&vk_fd);
    bool s3 = stage3_xrt_export_fd(&xrt_fd);
    bool s4 = s1 && stage4_xrt_import_fd(vk_fd);
    bool s5 = stage5_xrt_userptr();
    printf("\nSUMMARY: s1=%d s2(vk-export-fd)=%d s3(xrt-export)=%d s4(xrt-import-vkfd)=%d s5(userptr)=%d\n",
           s1, s2, s3, s4, s5);
    return 0;
}
