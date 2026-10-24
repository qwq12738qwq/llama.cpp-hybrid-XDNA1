// Phase 26: does a *compute shader* write into a host-imported (shared with XRT)
// buffer remove the readback tax that a device-local buffer suffers?
//
// Same 3.67 MB dispatch, two output allocations:
//   (A) DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT   (what ggml's mini-graph uses)
//   (B) host-imported via VK_EXT_external_memory_host (shared with XRT)
// Measures submit+wait and host read time after the shader writes.
//
// Build: glslc sh_write.comp -o sh_write.spv
//        g++ -O2 -std=c++17 vk_shader_host_bench.cpp -o vk_shader_host_bench -lvulkan -ldl
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>

#include <vulkan/vulkan.h>

#define VK_TRY(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "  [FAIL] %s -> %d\n", #expr, (int) _r); return 1; } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static const size_t SZ = 3670016;  // 3.67 MB

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst; VK_TRY(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice pdev = devs[0];
    for (auto d : devs) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p); if (strstr(p.deviceName, "RADV")) { pdev = d; break; } }

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
    PFN_vkGetMemoryHostPointerPropertiesEXT pfnHostPtrProps =
        (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(dev,"vkGetMemoryHostPointerPropertiesEXT");

    // load SPIR-V
    std::ifstream f("sh_write.spv", std::ios::binary | std::ios::ate);
    if (!f) { printf("[FAIL] sh_write.spv missing (run glslc)\n"); return 1; }
    size_t ssz = f.tellg(); f.seekg(0);
    std::vector<uint32_t> spirv(ssz/4); f.read((char*)spirv.data(), ssz);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = ssz; smci.pCode = spirv.data();
    VkShaderModule sm; VK_TRY(vkCreateShaderModule(dev,&smci,nullptr,&sm));

    std::vector<VkDescriptorSetLayoutBinding> binds = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 1; dslci.pBindings = binds.data();
    VkDescriptorSetLayout dsl; VK_TRY(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 0};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl; VK_TRY(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
    (void)pcr;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cpci.layout = pl;
    VkPipeline pipe; VK_TRY(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev,&mp);
    VkQueue q; vkGetDeviceQueue(dev,qfam,0,&q);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex=qfam;
    VkCommandPool cp; VK_TRY(vkCreateCommandPool(dev,&cpi,nullptr,&cp));

    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 8; dpci.poolSizeCount = 1;
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dp; VK_TRY(vkCreateDescriptorPool(dev,&dpci,nullptr,&dp));

    const uint32_t groups = (uint32_t)((SZ/4 + 255)/256);

    auto run_case = [&](const char * label, VkBuffer buf, VkDeviceMemory mem, void * hostp, bool host_imported) -> int {
        printf("\n%s\n", label);
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
        VkDescriptorSet ds; VK_TRY(vkAllocateDescriptorSets(dev,&dsai,&ds));
        VkDescriptorBufferInfo dbi{buf, 0, SZ};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet=ds; wds.dstBinding=0; wds.descriptorCount=1; wds.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.pBufferInfo=&dbi;
        vkUpdateDescriptorSets(dev,1,&wds,0,nullptr);

        for (int it=0; it<5; it++) {
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool=cp; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
            vkAllocateCommandBuffers(dev,&cai,&cb);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cb,&bi);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
            vkCmdDispatch(cb, groups, 1, 1);
            VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; bb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            bb.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bb.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            bb.buffer=buf; bb.offset=0; bb.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,0,nullptr,1,&bb,0,nullptr);
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
            double t0=now_ms();
            vkQueueSubmit(q,1,&si,VK_NULL_HANDLE);
            vkQueueWaitIdle(q);
            if (!host_imported) {
                VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                r.memory=mem; r.offset=0; r.size=VK_WHOLE_SIZE;
                vkInvalidateMappedMemoryRanges(dev,1,&r);
            }
            double t1=now_ms();
            volatile float s=0; float* u=(float*)hostp;
            bool ok=true;
            for (size_t i=0;i<SZ/4;i++) { float v=u[i]; s+=v; if (v!=(float)i*1.5f+0.25f) ok=false; }
            double t2=now_ms();
            printf("  it%d submit+wait+invalidate=%.2fms full_read=%.2fms correct=%d\n", it, t1-t0, t2-t1, (int)ok);
            fflush(stdout);
            vkFreeCommandBuffers(dev, cp, 1, &cb);
        }
        return 0;
    };

    // (A) device-local host-visible coherent
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=SZ; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf; VK_TRY(vkCreateBuffer(dev,&bci,nullptr,&buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr);
        uint32_t mti=~0u;
        for(uint32_t i=0;i<mp.memoryTypeCount;i++) if(mr.memoryTypeBits&(1u<<i)){
            VkMemoryPropertyFlags ff=mp.memoryTypes[i].propertyFlags;
            if((ff&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)&&(ff&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)&&(ff&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){mti=i;break;}
        }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize=mr.size; mai.memoryTypeIndex=mti;
        VkDeviceMemory mem; VK_TRY(vkAllocateMemory(dev,&mai,nullptr,&mem));
        VK_TRY(vkBindBufferMemory(dev,buf,mem,0));
        void* p=nullptr; VK_TRY(vkMapMemory(dev,mem,0,SZ,0,&p));
        char lbl[128]; snprintf(lbl,sizeof(lbl),"(A) DEVICE_LOCAL|HOST_VISIBLE shader write, mem type=%u flags=0x%x",mti,mp.memoryTypes[mti].propertyFlags);
        run_case(lbl, buf, mem, p, false);
        vkUnmapMemory(dev,mem);
    }

    // (B) host-imported
    if (pfnHostPtrProps) {
        void* host=nullptr; if(posix_memalign(&host,64*1024,SZ)!=0){perror("memalign");return 1;}
        memset(host,0,SZ);
        VkMemoryHostPointerPropertiesEXT hpp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        VkResult hr=pfnHostPtrProps(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,host,&hpp);
        if (hr==VK_SUCCESS && hpp.memoryTypeBits) {
            VkExternalMemoryBufferCreateInfo embi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            embi.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.pNext=&embi; bci.size=SZ; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
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
            char lbl[128]; snprintf(lbl,sizeof(lbl),"(B) host-imported shader write, mem type=%u flags=0x%x",mti,mp.memoryTypes[mti].propertyFlags);
            run_case(lbl, buf, mem, host, true);
        } else {
            printf("[FAIL] host pointer not importable\n");
        }
    }
    printf("\nDONE\n");
    return 0;
}
