// Phase 26 readback-tax micro-benchmark (raw Vulkan, no ggml).
//
// Writes 3.67 MB from the GPU into two different allocations and measures the
// time until the CPU can read the result:
//   (A) DEVICE_LOCAL | HOST_VISIBLE buffer, vkMapMemory'd   (today's mini-graph)
//   (B) host-imported buffer (VK_EXT_external_memory_host), plain host memory
//
// Build:
//   g++ -O2 -std=c++17 vk_readback_bench.cpp -o vk_readback_bench -lvulkan -ldl
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include <vulkan/vulkan.h>

#define VK_TRY(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "  [FAIL] %s -> %d\n", #expr, (int) _r); return 1; } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static const size_t SZ = 3670016;  // 3.67 MB, the ffn_gate ct slice size
static const uint32_t PAT = 0x12345678u;

static PFN_vkGetMemoryHostPointerPropertiesEXT pfnHostPtrProps = nullptr;

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VK_TRY(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice pdev = devs[0];
    for (auto d : devs) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p); if (strstr(p.deviceName, "RADV")) { pdev = d; break; } }
    VkPhysicalDeviceProperties dp; vkGetPhysicalDeviceProperties(pdev, &dp);
    printf("device: %s\n", dp.deviceName);

    uint32_t qfam = 0;
    { uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pdev,&nq,nullptr); std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(pdev,&nq,qf.data()); for(uint32_t i=0;i<nq;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfam=i;break;} }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char * dexts[] = { "VK_EXT_external_memory_host" };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=1; dci.ppEnabledExtensionNames=dexts;
    VkDevice dev; VK_TRY(vkCreateDevice(pdev,&dci,nullptr,&dev));
    pfnHostPtrProps = (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(dev,"vkGetMemoryHostPointerPropertiesEXT");

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev,&mp);
    VkQueue q; vkGetDeviceQueue(dev,qfam,0,&q);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci.queueFamilyIndex=qfam;
    VkCommandPool cp; VK_TRY(vkCreateCommandPool(dev,&cpci,nullptr,&cp));

    // ---------- (A) device-local host-visible ----------
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=SZ; bci.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf; VK_TRY(vkCreateBuffer(dev,&bci,nullptr,&buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr);
        uint32_t mti=~0u;
        for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i))){
            VkMemoryPropertyFlags f=mp.memoryTypes[i].propertyFlags;
            if((f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)&&(f&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){mti=i;break;}
        }
        if(mti==~0u) for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)){mti=i;break;}
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize=mr.size; mai.memoryTypeIndex=mti;
        VkDeviceMemory mem; VK_TRY(vkAllocateMemory(dev,&mai,nullptr,&mem));
        VK_TRY(vkBindBufferMemory(dev,buf,mem,0));
        void* p=nullptr; VK_TRY(vkMapMemory(dev,mem,0,SZ,0,&p));
        printf("\n(A) device-local host-visible, type=%u flags=0x%x\n", mti, mp.memoryTypes[mti].propertyFlags);

        for (int it=0; it<5; it++) {
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool=cp; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
            vkAllocateCommandBuffers(dev,&cai,&cb);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cb,&bi);
            vkCmdFillBuffer(cb,buf,0,SZ,PAT);
            VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; bb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            bb.buffer=buf; bb.offset=0; bb.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&bb,0,nullptr);
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;

            double t0=now_ms();
            vkQueueSubmit(q,1,&si,VK_NULL_HANDLE);
            vkQueueWaitIdle(q);
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory=mem; r.offset=0; r.size=VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(dev,1,&r);
            double t1=now_ms();
            volatile uint32_t s=0; uint32_t* u=(uint32_t*)p;
            for(size_t i=0;i<SZ/sizeof(uint32_t);i+=64) s+=u[i];
            double t2=now_ms();
            printf("  it%d submit+wait+invalidate=%.2fms touch=%.2fms (sum=%u)\n", it, t1-t0, t2-t1, (unsigned)s);
        }
        vkUnmapMemory(dev,mem);
    }

    // ---------- (B) host-imported ----------
    if (pfnHostPtrProps) {
        void* host=nullptr; if(posix_memalign(&host,64*1024,SZ)!=0){perror("memalign");return 1;}
        memset(host,0,SZ);
        VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        VkResult hr=pfnHostPtrProps(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,host,&hpp);
        printf("\n(B) host-imported, hostptr result=%d memoryTypeBits=0x%x\n",(int)hr,hpp.memoryTypeBits);
        if(hr==VK_SUCCESS && hpp.memoryTypeBits){
            VkExternalMemoryBufferCreateInfo embi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            embi.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.pNext=&embi; bci.size=SZ; bci.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buf; VK_TRY(vkCreateBuffer(dev,&bci,nullptr,&buf));
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr);
            uint32_t mti=~0u;
            for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((hpp.memoryTypeBits&(1u<<i))&&(mr.memoryTypeBits&(1u<<i))){mti=i;break;}
            VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
            imp.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT; imp.pHostPointer=host;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.pNext=&imp; mai.allocationSize=mr.size; mai.memoryTypeIndex=mti;
            VkDeviceMemory mem; VK_TRY(vkAllocateMemory(dev,&mai,nullptr,&mem));
            VK_TRY(vkBindBufferMemory(dev,buf,mem,0));
            for (int it=0; it<5; it++) {
                VkCommandBuffer cb;
                VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cai.commandPool=cp; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
                vkAllocateCommandBuffers(dev,&cai,&cb);
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(cb,&bi);
                vkCmdFillBuffer(cb,buf,0,SZ,PAT);
                VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                bb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; bb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
                bb.buffer=buf; bb.offset=0; bb.size=VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&bb,0,nullptr);
                vkEndCommandBuffer(cb);
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
                double t0=now_ms();
                vkQueueSubmit(q,1,&si,VK_NULL_HANDLE);
                vkQueueWaitIdle(q);
                double t1=now_ms();
                volatile uint32_t s=0; uint32_t* u=(uint32_t*)host;
                for(size_t i=0;i<SZ/sizeof(uint32_t);i+=64) s+=u[i];
                double t2=now_ms();
                printf("  it%d submit+wait=%.2fms touch=%.2fms (sum=%u)\n", it, t1-t0, t2-t1, (unsigned)s);
            }
        }
    }
    return 0;
}
