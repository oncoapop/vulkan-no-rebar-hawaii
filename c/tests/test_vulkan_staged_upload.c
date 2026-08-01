#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define TEST_BUFFER_SIZE 1024
#define FENCE_TIMEOUT 5000000000ULL // 5 seconds in nanoseconds

#define CHECK_VK(call, msg) \
    do { \
        VkResult res = (call); \
        if (res != VK_SUCCESS) { \
            fprintf(stderr, "FAIL: %s (error %d)\n", msg, res); \
            failed = 1; \
            goto cleanup; \
        } \
    } while (0)

#define CHECK_VK_TIMEOUT(call, msg) \
    do { \
        VkResult res = (call); \
        if (res == VK_TIMEOUT) { \
            fprintf(stderr, "FAIL: %s (timed out)\n", msg); \
            failed = 1; \
            goto timeout_cleanup; \
        } else if (res != VK_SUCCESS) { \
            fprintf(stderr, "FAIL: %s (error %d)\n", msg, res); \
            failed = 1; \
            goto cleanup; \
        } \
    } while (0)

int main() {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkBuffer deviceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDevice *devices = NULL;
    int failed = 0;

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
    if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
        printf("SKIP: Could not create Vulkan instance.\n");
        return 0;
    } else if (res != VK_SUCCESS) {
        printf("FAIL: vkCreateInstance failed with error %d.\n", res);
        return 1;
    }

    uint32_t deviceCount = 0;
    do {
        res = vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
            fprintf(stderr, "FAIL: vkEnumeratePhysicalDevices (count) (error %d)\n", res);
            failed = 1;
            goto cleanup;
        }
        if (deviceCount == 0) {
            printf("SKIP: No Vulkan physical devices found.\n");
            goto cleanup;
        }

        if (devices) free(devices);
        devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
        if (!devices) {
            printf("FAIL: Out of memory for physical devices array.\n");
            failed = 1;
            goto cleanup;
        }
        res = vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
        if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
            fprintf(stderr, "FAIL: vkEnumeratePhysicalDevices (devices) (error %d)\n", res);
            failed = 1;
            goto cleanup;
        }
    } while (res == VK_INCOMPLETE);

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
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = transferQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    CHECK_VK(vkCreateDevice(hawaiiDevice, &deviceCreateInfo, NULL, &device), "vkCreateDevice");

    VkQueue queue;
    vkGetDeviceQueue(device, transferQueueFamily, 0, &queue);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(hawaiiDevice, &memProperties);

    // Create buffers before allocating memory to determine memory requirements
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = TEST_BUFFER_SIZE;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    CHECK_VK(vkCreateBuffer(device, &bufferInfo, NULL, &stagingBuffer), "vkCreateBuffer (staging)");
    CHECK_VK(vkCreateBuffer(device, &bufferInfo, NULL, &deviceBuffer), "vkCreateBuffer (device)");

    VkMemoryRequirements stagingMemReq, deviceMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemReq);
    vkGetBufferMemoryRequirements(device, deviceBuffer, &deviceMemReq);

    if (stagingMemReq.size == 0 || stagingMemReq.memoryTypeBits == 0) {
        printf("FAIL: Invalid staging buffer memory requirements.\n");
        failed = 1;
        goto cleanup;
    }
    if (deviceMemReq.size == 0 || deviceMemReq.memoryTypeBits == 0) {
        printf("FAIL: Invalid device buffer memory requirements.\n");
        failed = 1;
        goto cleanup;
    }

    uint32_t stagingMemTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((stagingMemReq.memoryTypeBits & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            stagingMemTypeIndex = i;
            break;
        }
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

    if (stagingMemTypeIndex == (uint32_t)-1) {
        printf("FAIL: Could not find HOST_VISIBLE memory type for staging buffer.\n");
        failed = 1;
        goto cleanup;
    }
    if (deviceLocalMemTypeIndex == (uint32_t)-1) {
        printf("SKIP: Could not find strictly DEVICE_LOCAL (non-HOST_VISIBLE) memory type.\n");
        goto cleanup;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    allocInfo.allocationSize = stagingMemReq.size;
    allocInfo.memoryTypeIndex = stagingMemTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, NULL, &stagingMemory), "vkAllocateMemory (staging)");
    CHECK_VK(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0), "vkBindBufferMemory (staging)");

    allocInfo.allocationSize = deviceMemReq.size;
    allocInfo.memoryTypeIndex = deviceLocalMemTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, NULL, &deviceMemory), "vkAllocateMemory (device)");
    CHECK_VK(vkBindBufferMemory(device, deviceBuffer, deviceMemory, 0), "vkBindBufferMemory (device)");

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = transferQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK(vkCreateCommandPool(device, &poolInfo, NULL, &commandPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer), "vkAllocateCommandBuffers");

    uint8_t *data = NULL;
    CHECK_VK(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        data[i] = (uint8_t)(i % 256);
    }
    vkUnmapMemory(device, stagingMemory);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = TEST_BUFFER_SIZE;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, deviceBuffer, 1, &copyRegion);

    CHECK_VK(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VK(vkCreateFence(device, &fenceInfo, NULL, &fence), "vkCreateFence");

    CHECK_VK(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit (upload)");
    CHECK_VK_TIMEOUT(vkWaitForFences(device, 1, &fence, VK_TRUE, FENCE_TIMEOUT), "vkWaitForFences (upload)");

    CHECK_VK(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    memset(data, 0, TEST_BUFFER_SIZE);
    vkUnmapMemory(device, stagingMemory);

    CHECK_VK(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
    CHECK_VK(vkResetFences(device, 1, &fence), "vkResetFences");

    CHECK_VK(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    vkCmdCopyBuffer(commandBuffer, deviceBuffer, stagingBuffer, 1, &copyRegion);
    CHECK_VK(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    CHECK_VK(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit (readback)");
    CHECK_VK_TIMEOUT(vkWaitForFences(device, 1, &fence, VK_TRUE, FENCE_TIMEOUT), "vkWaitForFences (readback)");

    CHECK_VK(vkMapMemory(device, stagingMemory, 0, TEST_BUFFER_SIZE, 0, (void **)&data), "vkMapMemory");
    for (int i = 0; i < TEST_BUFFER_SIZE; i++) {
        if (data[i] != (uint8_t)(i % 256)) {
            printf("FAIL: Readback validation failed at offset %d.\n", i);
            failed = 1;
            break;
        }
    }
    vkUnmapMemory(device, stagingMemory);

    if (!failed) {
        printf("PASS\n");
    }

cleanup:
    if (device != VK_NULL_HANDLE) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, NULL);
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, NULL);
        }
        if (stagingBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, stagingBuffer, NULL);
        }
        if (deviceBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, deviceBuffer, NULL);
        }
        if (stagingMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, stagingMemory, NULL);
        }
        if (deviceMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, deviceMemory, NULL);
        }
        vkDestroyDevice(device, NULL);
    }
    if (devices) {
        free(devices);
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, NULL);
    }

/*
 * Deliberate leak path: on VK_TIMEOUT, we bypass Vulkan cleanup.
 * GPU work may still be in flight; destroying resources out from under it
 * can crash the driver. Process exit will reclaim host memory safely.
 */
timeout_cleanup:
    return failed ? 1 : 0;
}
