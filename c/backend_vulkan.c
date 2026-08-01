#include "backend_vulkan.h"
#include <string.h>

#define CHECK_VK_CLEANUP(stmt) \
    do { \
        if ((stmt) != VK_SUCCESS) { \
            if (device != VK_NULL_HANDLE) { \
                if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, NULL); \
                if (commandBuffer != VK_NULL_HANDLE) vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer); \
                if (stagingMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingMemory, NULL); \
                if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, NULL); \
            } \
            return VULKAN_UPLOAD_FAILURE; \
        } \
    } while (0)

VulkanUploadResult vulkan_staged_upload(
    VkDevice device,
    VkQueue queue,
    VkCommandPool commandPool,
    const VkPhysicalDeviceMemoryProperties *memProps,
    VkBuffer dstBuffer,
    VkDeviceSize size,
    const void *data,
    uint64_t timeout_ns,
    VulkanUploadOp *outOp
) {
    if (!outOp) {
        return VULKAN_UPLOAD_FAILURE;
    }
    memset(outOp, 0, sizeof(VulkanUploadOp));

    if (!device || !queue || !commandPool || !memProps || !dstBuffer || !data) {
        return VULKAN_UPLOAD_FAILURE;
    }

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    CHECK_VK_CLEANUP(vkCreateBuffer(device, &bufferInfo, NULL, &stagingBuffer));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);

    uint32_t memTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProps->memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1u << i)) &&
            (memProps->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProps->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    if (memTypeIndex == (uint32_t)-1) {
        CHECK_VK_CLEANUP(VK_ERROR_INITIALIZATION_FAILED);
    }

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    CHECK_VK_CLEANUP(vkAllocateMemory(device, &allocInfo, NULL, &stagingMemory));
    CHECK_VK_CLEANUP(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

    void *mapped = NULL;
    CHECK_VK_CLEANUP(vkMapMemory(device, stagingMemory, 0, size, 0, &mapped));
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(device, stagingMemory);

    VkCommandBufferAllocateInfo cmdAllocInfo = {0};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    CHECK_VK_CLEANUP(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    CHECK_VK_CLEANUP(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferCopy copyRegion = {0};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, dstBuffer, 1, &copyRegion);

    CHECK_VK_CLEANUP(vkEndCommandBuffer(commandBuffer));

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VK_CLEANUP(vkCreateFence(device, &fenceInfo, NULL, &fence));

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
        CHECK_VK_CLEANUP(VK_ERROR_INITIALIZATION_FAILED);
    }

    // After this point, resources are in flight.
    VkResult waitRes = vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);
    if (waitRes == VK_SUCCESS) {
        vkDestroyFence(device, fence, NULL);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkFreeMemory(device, stagingMemory, NULL);
        vkDestroyBuffer(device, stagingBuffer, NULL);
        return VULKAN_UPLOAD_SUCCESS;
    } else {
        outOp->device = device;
        outOp->commandPool = commandPool;
        outOp->stagingBuffer = stagingBuffer;
        outOp->stagingMemory = stagingMemory;
        outOp->commandBuffer = commandBuffer;
        outOp->fence = fence;

        if (waitRes == VK_TIMEOUT) {
            return VULKAN_UPLOAD_TIMEOUT;
        } else {
            return VULKAN_UPLOAD_FAILURE;
        }
    }
}

VulkanUploadResult vulkan_upload_finish(
    VulkanUploadOp *op,
    uint64_t timeout_ns
) {
    if (!op || !op->device || !op->fence) {
        return VULKAN_UPLOAD_FAILURE;
    }

    VkResult waitRes = vkWaitForFences(op->device, 1, &op->fence, VK_TRUE, timeout_ns);
    if (waitRes == VK_SUCCESS) {
        vkDestroyFence(op->device, op->fence, NULL);
        vkFreeCommandBuffers(op->device, op->commandPool, 1, &op->commandBuffer);
        vkFreeMemory(op->device, op->stagingMemory, NULL);
        vkDestroyBuffer(op->device, op->stagingBuffer, NULL);
        memset(op, 0, sizeof(VulkanUploadOp));
        return VULKAN_UPLOAD_SUCCESS;
    } else if (waitRes == VK_TIMEOUT) {
        return VULKAN_UPLOAD_TIMEOUT;
    } else {
        return VULKAN_UPLOAD_FAILURE;
    }
}
