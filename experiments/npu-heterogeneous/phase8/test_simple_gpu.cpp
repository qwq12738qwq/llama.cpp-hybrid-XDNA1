#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, VK
