#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VULKAN_UPLOAD_SUCCESS = 0,
    VULKAN_UPLOAD_FAILURE = 1,
    VULKAN_UPLOAD_TIMEOUT = 2
} VulkanUploadResult;

/*
 * Retains unreachable Vulkan resources on a timeout or fence failure to prevent
 * driver crashes from destroying in-flight resources. Cleaned up via vulkan_upload_finish().
 */
typedef struct {
    VkDevice device;
    VkCommandPool commandPool;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkCommandBuffer commandBuffer;
    VkFence fence;
} VulkanUploadOp;

/*
 * Uploads data to a STRICTLY DEVICE_LOCAL buffer (which must have VK_BUFFER_USAGE_TRANSFER_DST_BIT).
 * Requires `outOp` to be non-NULL; it will be zeroed immediately.
 * `size` must be nonzero and representable as size_t.
 *
 * Vulkan host synchronization is the caller's responsibility: access to `queue`
 * and allocation/free operations on `commandPool` must be externally synchronized.
 * If this function retains an operation in `outOp`, the caller must not reset or
 * destroy `commandPool` until vulkan_upload_finish() succeeds.
 *
 * After vkQueueSubmit succeeds, temporary resources are NEVER destroyed unless
 * vkWaitForFences returns VK_SUCCESS. On VK_TIMEOUT or any other fence-wait error,
 * the resources are retained in `outOp` and it returns the appropriate error code.
 */
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
);

/*
 * Waits for the fence in `op` with `timeout_ns`.
 * Allocation/free operations on the retained command pool must be externally
 * synchronized by the caller.
 * Only cleans up the resources and zeroes `op` if the fence successfully signals (VK_SUCCESS).
 * Otherwise returns VULKAN_UPLOAD_TIMEOUT or VULKAN_UPLOAD_FAILURE and leaves resources intact.
 */
VulkanUploadResult vulkan_upload_finish(
    VulkanUploadOp *op,
    uint64_t timeout_ns
);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_VULKAN_H */
