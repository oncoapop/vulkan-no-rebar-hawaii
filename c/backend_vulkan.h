#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <vulkan/vulkan.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_VULKAN_VENDOR_ID 0x1002u
#define COLI_VULKAN_DEVICE_ID 0x67b1u
#define COLI_VULKAN_VRAM_RESERVE_BYTES (1024ULL * 1024ULL * 1024ULL)

typedef enum {
    VULKAN_UPLOAD_SUCCESS = 0,
    VULKAN_UPLOAD_FAILURE = 1,
    VULKAN_UPLOAD_TIMEOUT = 2,
    VULKAN_UPLOAD_DEVICE_LOST = 3
} VulkanUploadResult;

/*
 * Retains temporary resources after a submitted operation times out or loses
 * the device. The resources are released only after vulkan_upload_finish()
 * observes VK_SUCCESS. `api` is private dispatch state used by unit tests.
 */
typedef struct {
    VkDevice device;
    VkCommandPool commandPool;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkCommandBuffer commandBuffer;
    VkFence fence;
    const void *api;
    VkResult lastResult;
} VulkanUploadOp;

/*
 * Low-level staged copy into an existing destination buffer. The caller owns
 * the device, queue, command pool, destination, and external synchronization.
 * `dstBuffer` must have VK_BUFFER_USAGE_TRANSFER_DST_BIT and must be large
 * enough for `size` bytes beginning at offset zero.
 * Persistent-context backend internals and Phase 3A tests normally use the
 * guarded tensor transport wrapper instead.
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
 * Waits for a retained operation. Resources are destroyed and `op` is zeroed
 * only after VK_SUCCESS. Timeout, device loss, and other failures retain it.
 * Host access to the retained operation and its fence must be externally
 * synchronized. Allocation/free access to `op->commandPool` must also be
 * externally synchronized. The caller must not reset or destroy the command
 * pool, device, fence, or any retained resource until this function returns
 * VULKAN_UPLOAD_SUCCESS and clears the operation.
 */
VulkanUploadResult vulkan_upload_finish(
    VulkanUploadOp *op,
    uint64_t timeout_ns
);

typedef enum {
    COLI_VULKAN_OK = 0,
    COLI_VULKAN_UNAVAILABLE = 1,
    COLI_VULKAN_INVALID_ARGUMENT = 2,
    COLI_VULKAN_UNSUPPORTED = 3,
    COLI_VULKAN_OUT_OF_MEMORY = 4,
    COLI_VULKAN_LIMIT_EXCEEDED = 5,
    COLI_VULKAN_TIMEOUT = 6,
    COLI_VULKAN_DEVICE_LOST = 7,
    COLI_VULKAN_BUSY = 8,
    COLI_VULKAN_ERROR = 9
} ColiVulkanResult;

typedef enum {
    COLI_VULKAN_TENSOR_ALLOCATED = 0,
    COLI_VULKAN_TENSOR_PENDING = 1,
    COLI_VULKAN_TENSOR_READY = 2,
    COLI_VULKAN_TENSOR_FAILED = 3,
    COLI_VULKAN_TENSOR_DESTROY_PENDING = 4
} ColiVulkanTensorState;

typedef struct ColiVulkanContext ColiVulkanContext;
typedef struct ColiVulkanTensor ColiVulkanTensor;

typedef struct {
    VkDeviceSize weight_offset;
    VkDeviceSize weight_size;
    VkDeviceSize scale_offset;
    VkDeviceSize scale_size;
    VkDeviceSize packed_size;
} ColiVulkanTensorLayout;

#ifdef COLI_VULKAN_TESTING
/* Test-only injectable Vulkan entry points. Production always uses Vulkan. */
typedef struct {
    PFN_vkCreateInstance CreateInstance;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers FreeCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkWaitForFences WaitForFences;
} ColiVulkanApi;
#endif

typedef struct {
    /* Required and nonzero. There is intentionally no implicit budget. */
    uint64_t expert_budget_bytes;
    /* Required bounded wait for uploads when a call does not override it. */
    uint64_t upload_timeout_ns;
#ifdef COLI_VULKAN_TESTING
    const ColiVulkanApi *test_api;
#endif
} ColiVulkanConfig;

typedef struct {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t queue_family_index;
    uint32_t memory_heap_count;
    uint32_t memory_type_count;
    uint32_t max_memory_allocation_count;
    VkDeviceSize min_storage_buffer_offset_alignment;
    uint64_t requested_budget_bytes;
    uint64_t effective_budget_bytes;
    uint64_t committed_bytes;
    uint32_t live_tensors;
    uint32_t live_allocations;
    uint32_t pending_operations;
    int usable;
    int device_lost;
} ColiVulkanContextInfo;

typedef struct {
    ColiVulkanTensorLayout layout;
    VkDeviceSize allocation_size;
    uint32_t memory_type_index;
    uint32_t heap_index;
    VkMemoryPropertyFlags memory_property_flags;
    ColiVulkanTensorState state;
} ColiVulkanTensorInfo;

const char *coli_vulkan_result_string(ColiVulkanResult result);

/* Parses the required VULKAN_EXPERT_MB value. Missing/zero/invalid is rejected. */
ColiVulkanResult coli_vulkan_config_from_env(
    ColiVulkanConfig *config,
    uint64_t upload_timeout_ns
);

/* Overflow-checked layout aligned to minStorageBufferOffsetAlignment. */
ColiVulkanResult coli_vulkan_plan_tensor_layout(
    uint64_t weight_bytes,
    uint64_t scale_bytes,
    VkDeviceSize alignment,
    ColiVulkanTensorLayout *layout
);

/* Pure strict-memory selector used by allocation and unit tests. */
ColiVulkanResult coli_vulkan_select_strict_memory_type(
    const VkPhysicalDeviceMemoryProperties *properties,
    uint32_t memory_type_bits,
    VkDeviceSize allocation_size,
    uint64_t reserve_bytes,
    uint32_t *memory_type_index,
    uint32_t *heap_index,
    uint64_t *heap_usable_bytes
);

/* Creates the one allowed process-wide Vulkan context. */
ColiVulkanResult coli_vulkan_context_create(
    ColiVulkanContext **context,
    const ColiVulkanConfig *config
);

/*
 * Centralized explicit cleanup. A TIMEOUT, ERROR, or DEVICE_LOST result leaves
 * `*context` non-NULL and retains the context and every resource whose safe
 * destruction cannot be proven. The caller may retry after TIMEOUT or ERROR;
 * DEVICE_LOST is terminal and retains unprovable resources until process exit.
 */
ColiVulkanResult coli_vulkan_context_destroy(
    ColiVulkanContext **context,
    uint64_t timeout_ns
);

ColiVulkanResult coli_vulkan_context_get_info(
    const ColiVulkanContext *context,
    ColiVulkanContextInfo *info
);

ColiVulkanResult coli_vulkan_context_get_memory_properties(
    const ColiVulkanContext *context,
    VkPhysicalDeviceMemoryProperties *properties
);

#if defined(COLI_VULKAN_INTERNAL) || defined(COLI_VULKAN_TESTING)
/*
 * Phase 3A raw transport API, visible only to backend internals and tests.
 * It creates one strict-local buffer/allocation, packs already-validated raw
 * weight and scale byte ranges, and uploads through the persistent queue and
 * command pool. It does not validate QT format geometry and MUST NOT be called
 * by the Phase 3B loader. Phase 3B must add a public format-aware entry point
 * that validates fmt, I, O, gs, derived scale count, source lengths, and format
 * eligibility before calling this internal transport function.
 *
 * TIMEOUT, ERROR, or DEVICE_LOST may be returned with `*tensor` non-NULL after
 * submission. The caller must retain that handle, call
 * coli_vulkan_finish_pending() with bounded waits, and eventually call
 * coli_vulkan_tensor_free(); it must not discard or overwrite the handle.
 */
ColiVulkanResult coli_vulkan_tensor_upload(
    ColiVulkanContext *context,
    ColiVulkanTensor **tensor,
    const void *weights,
    uint64_t weight_bytes,
    const void *scales,
    uint64_t scale_bytes,
    uint64_t timeout_ns
);
#endif

/*
 * Exact diagnostic readback of the packed tensor bytes. `output` is valid only
 * when COLI_VULKAN_OK is returned. After TIMEOUT, finish the retained operation
 * with coli_vulkan_finish_pending(), then issue a new readback; the timed-out
 * call does not retain caller-owned output storage.
 */
ColiVulkanResult coli_vulkan_tensor_readback(
    ColiVulkanContext *context,
    ColiVulkanTensor *tensor,
    void *output,
    uint64_t output_bytes,
    uint64_t timeout_ns
);

ColiVulkanResult coli_vulkan_tensor_get_info(
    const ColiVulkanTensor *tensor,
    ColiVulkanTensorInfo *info
);

/*
 * Frees immediately only when no submitted operation can still reference the
 * tensor. BUSY or DEVICE_LOST may be returned after clearing `*tensor`: the
 * caller's ownership is relinquished, but Vulkan destruction is deferred until
 * completion is proven or, after DEVICE_LOST, until process exit.
 */
ColiVulkanResult coli_vulkan_tensor_free(
    ColiVulkanContext *context,
    ColiVulkanTensor **tensor
);

/* Finishes the single retained upload/readback operation with a bounded wait. */
ColiVulkanResult coli_vulkan_finish_pending(
    ColiVulkanContext *context,
    uint64_t timeout_ns
);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_VULKAN_H */
