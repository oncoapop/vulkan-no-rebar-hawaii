#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define TEST_BUFFER_SIZE 1024

void check_vk(VkResult res, const char *msg) {
    if (res != VK_SUCCESS) {
        fprintf(stderr, "FAIL: %s (error %d)\n", msg, res);
        exit(1);
    }
}

int main() {
    VkInstance instance;
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanStagedUploadTest";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult res = vkCreateInstance(&createInfo, NULL, &instance);
    if (res != VK_SUCCESS) {
        printf("SKIP: Could not create Vulkan instance.\n");
        return 0;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        printf("SKIP: No Vulkan physical devices found.\n");
        return 0;
    }

    VkPhysicalDevice *devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    VkPhysicalDevice hawaiiDevice = VK_NULL_HANDLE;
    uint32_t hawaiiIds[] = {0x67B0, 0x67B1, 0x67B8, 0x67B9, 0x67A0, 0x67A1, 0x67A2, 0x67A8, 0x67A9};
    int numHawaiiIds = sizeof(hawaiiIds) / sizeof(hawaiiIds[0]);

    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        if (deviceProperties.vendorID == 0x1002) {
            for (int j = 0; j < numHawaiiIds; j++) {
                if (deviceProperties.deviceID == hawaiiIds[j]) {
                    hawaiiDevice = devices[i];
                    break;
                }
            }
        }
        if (hawaiiDevice != VK_NULL_HANDLE) break;
    }

    free(devices);

    if (hawaiiDevice == VK_NULL_HANDLE) {
        printf("SKIP: No AMD Hawaii GPU found.\n");
        return 0;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(hawaiiDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties *queueFamilies = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(hawaiiDevice, &queueFamilyCount, queueFamilies);

    int transferQueueFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT || queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT || queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            transferQueueFamily = i;
            break;
        }
    }
    free(queueFamilies);

    if (transferQueueFamily == -1) {
        printf("FAIL: No suitable transfer queue found.\n");
        return 1;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = transferQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    VkDevice device;
    check_vk(vkCreateDevice(hawaiiDevice, &deviceCreateInfo, NULL, &device), "vkCreateDevice");

    VkQueue queue;
    vkGetDeviceQueue(device, transferQueueFamily, 0, &queue);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(hawaiiDevice, &memProperties);

    uint32_t stagingMemTypeIndex = -1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            stagingMemTypeIndex = i;
            break;
        }
    }

    uint32_t deviceLocalMemTypeIndex = -1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            deviceLocalMemTypeIndex = i;
            break;
        }
    }

    if (stagingMemTypeIndex == (uint32_t)-1) {
        printf("FAIL: Could not find HOST_VISIBLE memory type.\n");
        return 1;
    }
    if (deviceLocalMemTypeIndex == (uint32_t)-1) {
        printf("SKIP: Could not find strictly DEVICE_LOCAL (non-HOST_VISIBLE) memory type.\n");
        return 0;
    }

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = TEST_BUFFER_SIZE;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer, deviceBuffer;
    check_vk(vkCreateBuffer(device, &bufferInfo, NULL, &stagingBuffer), "vkCreateBuffer (staging)");
    check_vk(vkCreateBuffer(device, &bufferInfo, NULL, &deviceBuffer), "vkCreateBuffer (device)");

    VkMemoryRequirements stagingMemReq, deviceMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);
    vkGetBufferMemoryRequirements(device, deviceBuffer, &deviceMemReq);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    allocInfo.allocationSize = stagingMemReq.size;
    allocInfo.memoryTypeIndex = stagingMemTypeIndex;
    VkDeviceMemory stagingMemory;
    check_vk(vkAllocateMemory(device, &allocInfo, NULL, &stagingMemory), "vkAllocateMemory (staging)");
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    allocInfo.allocationSize = deviceMemReq.size;
    allocInfo.memoryTypeIndex = deviceLocalMemTypeIndex;
    VkDeviceMemory deviceMemory;
    check_vk(vkAllocateMemory(device, &allocInfo, NULL, &deviceMemory), "vkAllocateMemory (device)");
    vkBindBufferMemory(device, deviceBuffer, deviceMemory, 0);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = transferQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool commandPool;
    check_vk(vkCreateCommandPool(device, &poolInfo, NULL, &commandPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    check_vk(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer), "vkAllocateCommandBuffers");

    uint8_t *data;
    check_vk(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        data[i] = (uint8_t)(i % 256);
    }
    vkUnmapMemory(device, stagingMemory);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = TEST_BUFFER_SIZE;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &copyRegion);

    check_vk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    check_vk(vkCreateFence(device, &fenceInfo, NULL, &fence), "vkCreateFence");

    check_vk(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit (upload)");
    check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    check_vk(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    memset(data, 0, TEST_BUFFER_SIZE);
    vkUnmapMemory(device, stagingMemory);

    check_vk(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
    check_vk(vkResetFences(device, 1, &fence), "vkResetFences");

    check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    vkCmdCopyBuffer(commandBuffer, deviceBuffer, stagingBuffer, 1, &copyRegion);
    check_vk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    check_vk(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit (readback)");
    check_vk(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    check_vk(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    int failed = 0;
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        if (data[i] != (uint8_t)(i % 256)) {
            failed = 1;
            break;
        }
    }
    vkUnmapMemory(device, stagingMemory);

    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyBuffer(device, stagingBuffer, NULL);
    vkDestroyBuffer(device, deviceBuffer, NULL);
    vkFreeMemory(device, stagingMemory, NULL);
    vkFreeMemory(device, deviceMemory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    if (failed) {
        printf("FAIL: Readback validation failed.\n");
        return 1;
    }

    printf("PASS\n");
    return 0;
}
