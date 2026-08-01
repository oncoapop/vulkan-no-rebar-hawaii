#include "../backend_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_BUFFER_SIZE 4096
#define FENCE_TIMEOUT 5000000000ULL // 5 seconds in nanoseconds

#define CHECK_VK(stmt, msg) \
    do { \
        VkResult res = (stmt); \
        if (res != VK_SUCCESS) { \
            printf("FAIL: %s returned %d\n", msg, res); \
            failed = 1; \
            goto cleanup; \
        } \
    } while (0)

int main(void) {
    int failed = 0;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice hawaiiDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkBuffer deviceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProperties;
    VkPhysicalDevice *devices = NULL;

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Colibri Vulkan Hardware Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult initRes = vkCreateInstance(&createInfo, NULL, &instance);
    if (initRes == VK_ERROR_INCOMPATIBLE_DRIVER) {
        printf("SKIP: Incompatible Vulkan driver.\n");
        return 0;
    }
    if (initRes != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance returned %d\n", initRes);
        return 1;
    }

    uint32_t deviceCount = 0;
    VkResult enumRes;
    do {
        enumRes = vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (enumRes == VK_SUCCESS && deviceCount > 0) {
            devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
            if (!devices) {
                printf("FAIL: Out of memory for devices array.\n");
                failed = 1;
                goto cleanup;
            }
            enumRes = vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
            if (enumRes == VK_INCOMPLETE) {
                free(devices);
                devices = NULL;
            }
        }
    } while (enumRes == VK_INCOMPLETE);

    if (enumRes != VK_SUCCESS || deviceCount == 0 || !devices) {
        printf("SKIP: No Vulkan devices found.\n");
        goto cleanup;
    }

    const uint32_t hawaiiIds[] = {0x67B0, 0x67B1, 0x67B8, 0x67B9, 0x67A0, 0x67A1, 0x67A2, 0x67A8, 0x67A9};
    const int numHawaiiIds = sizeof(hawaiiIds) / sizeof(hawaiiIds[0]);

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

    if (hawaiiDevice == VK_NULL_HANDLE) {
        printf("SKIP: No AMD Hawaii GPU found.\n");
        goto cleanup;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(hawaiiDevice, &queueFamilyCount, NULL);
    if (queueFamilyCount == 0) {
        printf("FAIL: No queue families found on the device.\n");
        failed = 1;
        goto cleanup;
    }
    VkQueueFamilyProperties *queueFamilies = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    if (!queueFamilies) {
        printf("FAIL: Out of memory for queue families array.\n");
        failed = 1;
        goto cleanup;
    }
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
        failed = 1;
        goto cleanup;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {0};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = transferQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {0};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    CHECK_VK(vkCreateDevice(hawaiiDevice, &deviceCreateInfo, NULL, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, transferQueueFamily, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(hawaiiDevice, &memProperties);

    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = TEST_BUFFER_SIZE;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VK(vkCreateBuffer(device, &bufferInfo, NULL, &deviceBuffer), "vkCreateBuffer (device)");

    VkMemoryRequirements deviceMemReq;
    vkGetBufferMemoryRequirements(device, deviceBuffer, &deviceMemReq);
    if (deviceMemReq.size == 0 || deviceMemReq.memoryTypeBits == 0) {
        printf("FAIL: Invalid device buffer memory requirements.\n");
        failed = 1;
        goto cleanup;
    }

    uint32_t deviceLocalMemTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((deviceMemReq.memoryTypeBits & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            deviceLocalMemTypeIndex = i;
            break;
        }
    }
    if (deviceLocalMemTypeIndex == (uint32_t)-1) {
        printf("SKIP: Could not find strictly DEVICE_LOCAL (non-HOST_VISIBLE) memory type.\n");
        goto cleanup;
    }

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = deviceMemReq.size;
    allocInfo.memoryTypeIndex = deviceLocalMemTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, NULL, &deviceMemory), "vkAllocateMemory (device)");
    CHECK_VK(vkBindBufferMemory(device, deviceBuffer, deviceMemory, 0), "vkBindBufferMemory (device)");

    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = transferQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK(vkCreateCommandPool(device, &poolInfo, NULL, &commandPool), "vkCreateCommandPool");

    // Prepare upload data
    uint8_t uploadData[TEST_BUFFER_SIZE];
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        uploadData[i] = (uint8_t)(i % 256);
    }

    VulkanUploadOp op = {0};
    VulkanUploadResult uploadRes = vulkan_staged_upload(
        device, queue, commandPool, &memProperties, deviceBuffer,
        TEST_BUFFER_SIZE, uploadData, FENCE_TIMEOUT, &op);

    if (uploadRes == VULKAN_UPLOAD_TIMEOUT) {
        // Wait for it again
        uploadRes = vulkan_upload_finish(&op, FENCE_TIMEOUT);
    }

    if (uploadRes != VULKAN_UPLOAD_SUCCESS) {
        printf("FAIL: vulkan_staged_upload failed or timed out.\n");
        failed = 1;
        // deliberate leak path on timeout
        if (uploadRes == VULKAN_UPLOAD_TIMEOUT) {
            return 1;
        }
        goto cleanup;
    }

    // Now verify the data using a direct readback similar to test_vulkan_staged_upload.c
    // We will do a manual readback here, allocating a temporary readback buffer

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CHECK_VK(vkCreateBuffer(device, &bufferInfo, NULL, &stagingBuffer), "vkCreateBuffer (staging)");
    VkMemoryRequirements stagingMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);

    uint32_t stagingMemTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((stagingMemReq.memoryTypeBits & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            stagingMemTypeIndex = i;
            break;
        }
    }

    if (stagingMemTypeIndex == (uint32_t)-1) {
        printf("FAIL: Could not find HOST_VISIBLE memory type for staging buffer.\n");
        failed = 1;
        goto cleanup_readback;
    }

    allocInfo.allocationSize = stagingMemReq.size;
    allocInfo.memoryTypeIndex = stagingMemTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, NULL, &stagingMemory), "vkAllocateMemory (staging)");
    CHECK_VK(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0), "vkBindBufferMemory (staging)");

    VkCommandBufferAllocateInfo cmdAllocInfo = {0};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VkBufferCopy copyRegion = {0};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = TEST_BUFFER_SIZE;
    vkCmdCopyBuffer(commandBuffer, deviceBuffer, stagingBuffer, 1, &copyRegion);
    CHECK_VK(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VK(vkCreateFence(device, &fenceInfo, NULL, &fence), "vkCreateFence");

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    CHECK_VK(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit");

    VkResult waitRes = vkWaitForFences(device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
    if (waitRes != VK_SUCCESS) {
        printf("FAIL: vkWaitForFences returned %d\n", waitRes);
        failed = 1;
        if (waitRes == VK_TIMEOUT) {
            return 1;
        }
        goto cleanup_readback;
    }

    void *mapped = NULL;
    CHECK_VK(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, &mapped), "vkMapMemory");
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        if (((uint8_t*)mapped)[i] != (uint8_t)(i % 256)) {
            printf("FAIL: Readback validation failed at offset %d.\n", i);
            failed = 1;
            break;
        }
    }
    vkUnmapMemory(device, stagingMemory);

    if (!failed) {
        printf("PASS\n");
    }

cleanup_readback:
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, NULL);
    if (commandBuffer != VK_NULL_HANDLE) vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    if (stagingMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingMemory, NULL);
    if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, NULL);

cleanup:
    if (device != VK_NULL_HANDLE) {
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, NULL);
        }
        if (deviceBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, deviceBuffer, NULL);
        }
        if (deviceMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, deviceMemory, NULL);
        }
        vkDestroyDevice(device, NULL);
    }
    if (devices) free(devices);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, NULL);

    return failed ? 1 : 0;
}
