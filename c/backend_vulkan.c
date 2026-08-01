#define COLI_VULKAN_INTERNAL 1
#include "backend_vulkan.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef COLI_VULKAN_TESTING
typedef ColiVulkanApi VulkanApi;
#else
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
} VulkanApi;
#endif

static const VulkanApi g_real_api = {
    vkCreateInstance,
    vkDestroyInstance,
    vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceQueueFamilyProperties,
    vkCreateDevice,
    vkDestroyDevice,
    vkGetDeviceQueue,
    vkCreateCommandPool,
    vkDestroyCommandPool,
    vkCreateBuffer,
    vkDestroyBuffer,
    vkGetBufferMemoryRequirements,
    vkAllocateMemory,
    vkFreeMemory,
    vkBindBufferMemory,
    vkMapMemory,
    vkUnmapMemory,
    vkAllocateCommandBuffers,
    vkFreeCommandBuffers,
    vkBeginCommandBuffer,
    vkEndCommandBuffer,
    vkCmdCopyBuffer,
    vkCreateFence,
    vkDestroyFence,
    vkQueueSubmit,
    vkWaitForFences
};

typedef enum {
    PENDING_NONE = 0,
    PENDING_UPLOAD = 1,
    PENDING_READBACK = 2
} PendingKind;

struct ColiVulkanTensor {
    ColiVulkanContext *context;
    VkBuffer buffer;
    VkDeviceMemory memory;
    ColiVulkanTensorLayout layout;
    VkDeviceSize allocation_size;
    uint32_t memory_type_index;
    uint32_t heap_index;
    VkMemoryPropertyFlags memory_property_flags;
    ColiVulkanTensorState state;
    int destroy_requested;
    struct ColiVulkanTensor *next;
};

typedef struct {
    PendingKind kind;
    VulkanUploadOp op;
    ColiVulkanTensor *tensor;
} PendingOperation;

struct ColiVulkanContext {
    const VulkanApi *api;
    pthread_mutex_t mutex;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    uint32_t queue_family_index;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint64_t requested_budget_bytes;
    uint64_t effective_budget_bytes;
    uint64_t committed_bytes;
    uint64_t upload_timeout_ns;
    uint32_t live_tensors;
    uint32_t live_allocations;
    uint32_t max_memory_allocation_count;
    int usable;
    int device_lost;
    int degraded;
    int shutting_down;
    ColiVulkanTensor *tensors;
    PendingOperation pending;
};

static pthread_mutex_t g_context_gate = PTHREAD_MUTEX_INITIALIZER;
static int g_context_active;

static int api_complete(const VulkanApi *api) {
    return api && api->CreateInstance && api->DestroyInstance &&
        api->EnumeratePhysicalDevices && api->GetPhysicalDeviceProperties &&
        api->GetPhysicalDeviceMemoryProperties &&
        api->GetPhysicalDeviceQueueFamilyProperties && api->CreateDevice &&
        api->DestroyDevice && api->GetDeviceQueue && api->CreateCommandPool &&
        api->DestroyCommandPool && api->CreateBuffer && api->DestroyBuffer &&
        api->GetBufferMemoryRequirements && api->AllocateMemory &&
        api->FreeMemory && api->BindBufferMemory && api->MapMemory &&
        api->UnmapMemory && api->AllocateCommandBuffers &&
        api->FreeCommandBuffers && api->BeginCommandBuffer &&
        api->EndCommandBuffer && api->CmdCopyBuffer && api->CreateFence &&
        api->DestroyFence && api->QueueSubmit && api->WaitForFences;
}

static const VulkanApi *config_api(const ColiVulkanConfig *config) {
#ifdef COLI_VULKAN_TESTING
    if (config && config->test_api) return (const VulkanApi *)config->test_api;
#else
    (void)config;
#endif
    return &g_real_api;
}

static void operation_cleanup(const VulkanApi *api, VulkanUploadOp *op) {
    if (!api || !op || op->device == VK_NULL_HANDLE) return;
    if (op->fence != VK_NULL_HANDLE)
        api->DestroyFence(op->device, op->fence, NULL);
    if (op->commandBuffer != VK_NULL_HANDLE && op->commandPool != VK_NULL_HANDLE)
        api->FreeCommandBuffers(op->device, op->commandPool, 1, &op->commandBuffer);
    if (op->stagingBuffer != VK_NULL_HANDLE)
        api->DestroyBuffer(op->device, op->stagingBuffer, NULL);
    if (op->stagingMemory != VK_NULL_HANDLE)
        api->FreeMemory(op->device, op->stagingMemory, NULL);
    memset(op, 0, sizeof(*op));
}

static void operation_retain(
    VulkanUploadOp *op,
    const VulkanApi *api,
    VkDevice device,
    VkCommandPool command_pool,
    VkBuffer staging_buffer,
    VkDeviceMemory staging_memory,
    VkCommandBuffer command_buffer,
    VkFence fence,
    VkResult result
) {
    op->device = device;
    op->commandPool = command_pool;
    op->stagingBuffer = staging_buffer;
    op->stagingMemory = staging_memory;
    op->commandBuffer = command_buffer;
    op->fence = fence;
    op->api = api;
    op->lastResult = result;
}

static VulkanUploadResult upload_result_from_vk(VkResult result) {
    if (result == VK_SUCCESS) return VULKAN_UPLOAD_SUCCESS;
    if (result == VK_TIMEOUT) return VULKAN_UPLOAD_TIMEOUT;
    if (result == VK_ERROR_DEVICE_LOST) return VULKAN_UPLOAD_DEVICE_LOST;
    return VULKAN_UPLOAD_FAILURE;
}

static VulkanUploadResult staged_upload_with_api(
    const VulkanApi *api,
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const VkPhysicalDeviceMemoryProperties *memory_properties,
    VkBuffer destination,
    VkDeviceSize size,
    const void *data,
    uint64_t timeout_ns,
    VulkanUploadOp *out_op
) {
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VulkanUploadOp cleanup = {0};
    VkResult result;

    if (!out_op) return VULKAN_UPLOAD_FAILURE;
    memset(out_op, 0, sizeof(*out_op));
    if (!api_complete(api) || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE ||
        command_pool == VK_NULL_HANDLE || !memory_properties ||
        destination == VK_NULL_HANDLE || !data || size == 0 ||
        size > (VkDeviceSize)SIZE_MAX || memory_properties->memoryTypeCount == 0 ||
        memory_properties->memoryTypeCount > VK_MAX_MEMORY_TYPES)
        return VULKAN_UPLOAD_FAILURE;

    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = api->CreateBuffer(device, &buffer_info, NULL, &staging_buffer);
    if (result != VK_SUCCESS) return upload_result_from_vk(result);

    cleanup.device = device;
    cleanup.commandPool = command_pool;
    cleanup.stagingBuffer = staging_buffer;
    cleanup.api = api;

    VkMemoryRequirements requirements = {0};
    api->GetBufferMemoryRequirements(device, staging_buffer, &requirements);
    if (requirements.size < size || requirements.memoryTypeBits == 0) {
        operation_cleanup(api, &cleanup);
        return VULKAN_UPLOAD_FAILURE;
    }

    uint32_t memory_type_index = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties->memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = memory_properties->memoryTypes[i].propertyFlags;
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memory_type_index = i;
            break;
        }
    }
    if (memory_type_index == UINT32_MAX) {
        operation_cleanup(api, &cleanup);
        return VULKAN_UPLOAD_FAILURE;
    }

    VkMemoryAllocateInfo allocation_info = {0};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;
    result = api->AllocateMemory(device, &allocation_info, NULL, &staging_memory);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }
    cleanup.stagingMemory = staging_memory;

    result = api->BindBufferMemory(device, staging_buffer, staging_memory, 0);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }

    void *mapped = NULL;
    result = api->MapMemory(device, staging_memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS || !mapped) {
        operation_cleanup(api, &cleanup);
        return result == VK_SUCCESS ? VULKAN_UPLOAD_FAILURE : upload_result_from_vk(result);
    }
    memcpy(mapped, data, (size_t)size);
    api->UnmapMemory(device, staging_memory);

    VkCommandBufferAllocateInfo command_allocation = {0};
    command_allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_allocation.commandPool = command_pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1;
    result = api->AllocateCommandBuffers(device, &command_allocation, &command_buffer);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }
    cleanup.commandBuffer = command_buffer;

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = api->BeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }

    VkBufferCopy copy = {0};
    copy.size = size;
    api->CmdCopyBuffer(command_buffer, staging_buffer, destination, 1, &copy);
    result = api->EndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }

    VkFenceCreateInfo fence_info = {0};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = api->CreateFence(device, &fence_info, NULL, &fence);
    if (result != VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return upload_result_from_vk(result);
    }
    cleanup.fence = fence;

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    result = api->QueueSubmit(queue, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            operation_retain(out_op, api, device, command_pool, staging_buffer,
                staging_memory, command_buffer, fence, result);
            return VULKAN_UPLOAD_DEVICE_LOST;
        }
        operation_cleanup(api, &cleanup);
        return VULKAN_UPLOAD_FAILURE;
    }

    result = api->WaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);
    if (result == VK_SUCCESS) {
        operation_cleanup(api, &cleanup);
        return VULKAN_UPLOAD_SUCCESS;
    }

    operation_retain(out_op, api, device, command_pool, staging_buffer,
        staging_memory, command_buffer, fence, result);
    return upload_result_from_vk(result);
}

VulkanUploadResult vulkan_staged_upload(
    VkDevice device,
    VkQueue queue,
    VkCommandPool command_pool,
    const VkPhysicalDeviceMemoryProperties *memory_properties,
    VkBuffer destination,
    VkDeviceSize size,
    const void *data,
    uint64_t timeout_ns,
    VulkanUploadOp *out_op
) {
    return staged_upload_with_api(&g_real_api, device, queue, command_pool,
        memory_properties, destination, size, data, timeout_ns, out_op);
}

VulkanUploadResult vulkan_upload_finish(VulkanUploadOp *op, uint64_t timeout_ns) {
    if (!op || op->device == VK_NULL_HANDLE || op->fence == VK_NULL_HANDLE)
        return VULKAN_UPLOAD_FAILURE;
    const VulkanApi *api = op->api ? (const VulkanApi *)op->api : &g_real_api;
    if (!api_complete(api)) return VULKAN_UPLOAD_FAILURE;
    VkResult result = api->WaitForFences(op->device, 1, &op->fence, VK_TRUE, timeout_ns);
    op->lastResult = result;
    if (result == VK_SUCCESS) {
        operation_cleanup(api, op);
        return VULKAN_UPLOAD_SUCCESS;
    }
    return upload_result_from_vk(result);
}

const char *coli_vulkan_result_string(ColiVulkanResult result) {
    switch (result) {
        case COLI_VULKAN_OK: return "OK";
        case COLI_VULKAN_UNAVAILABLE: return "UNAVAILABLE";
        case COLI_VULKAN_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case COLI_VULKAN_UNSUPPORTED: return "UNSUPPORTED";
        case COLI_VULKAN_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case COLI_VULKAN_LIMIT_EXCEEDED: return "LIMIT_EXCEEDED";
        case COLI_VULKAN_TIMEOUT: return "TIMEOUT";
        case COLI_VULKAN_DEVICE_LOST: return "DEVICE_LOST";
        case COLI_VULKAN_BUSY: return "BUSY";
        case COLI_VULKAN_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static ColiVulkanResult result_from_vk(VkResult result) {
    if (result == VK_SUCCESS) return COLI_VULKAN_OK;
    if (result == VK_TIMEOUT) return COLI_VULKAN_TIMEOUT;
    if (result == VK_ERROR_DEVICE_LOST) return COLI_VULKAN_DEVICE_LOST;
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        return COLI_VULKAN_OUT_OF_MEMORY;
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER) return COLI_VULKAN_UNAVAILABLE;
    return COLI_VULKAN_ERROR;
}

ColiVulkanResult coli_vulkan_config_from_env(
    ColiVulkanConfig *config,
    uint64_t upload_timeout_ns
) {
    if (!config || upload_timeout_ns == 0) return COLI_VULKAN_INVALID_ARGUMENT;
    const char *value = getenv("VULKAN_EXPERT_MB");
    if (!value || !*value) return COLI_VULKAN_INVALID_ARGUMENT;
    errno = 0;
    char *end = NULL;
    unsigned long long mb = strtoull(value, &end, 10);
    if (errno || end == value || *end != '\0' || mb == 0 ||
        mb > UINT64_MAX / (1024ULL * 1024ULL))
        return COLI_VULKAN_INVALID_ARGUMENT;
    memset(config, 0, sizeof(*config));
    config->expert_budget_bytes = (uint64_t)mb * 1024ULL * 1024ULL;
    config->upload_timeout_ns = upload_timeout_ns;
    return COLI_VULKAN_OK;
}

static int align_up_u64(uint64_t value, uint64_t alignment, uint64_t *output) {
    if (!output || alignment == 0) return 0;
    uint64_t remainder = value % alignment;
    if (!remainder) {
        *output = value;
        return 1;
    }
    uint64_t addition = alignment - remainder;
    if (value > UINT64_MAX - addition) return 0;
    *output = value + addition;
    return 1;
}

ColiVulkanResult coli_vulkan_plan_tensor_layout(
    uint64_t weight_bytes,
    uint64_t scale_bytes,
    VkDeviceSize alignment,
    ColiVulkanTensorLayout *layout
) {
    if (!layout || weight_bytes == 0 || alignment == 0)
        return COLI_VULKAN_INVALID_ARGUMENT;
    uint64_t a = (uint64_t)alignment;
    uint64_t scale_offset = 0;
    uint64_t end = weight_bytes;
    if (scale_bytes) {
        if (!align_up_u64(weight_bytes, a, &scale_offset) ||
            scale_offset > UINT64_MAX - scale_bytes)
            return COLI_VULKAN_LIMIT_EXCEEDED;
        end = scale_offset + scale_bytes;
    }
    uint64_t packed_size = 0;
    if (!align_up_u64(end, a, &packed_size) || packed_size == 0)
        return COLI_VULKAN_LIMIT_EXCEEDED;
    if ((uint64_t)(VkDeviceSize)weight_bytes != weight_bytes ||
        (uint64_t)(VkDeviceSize)scale_bytes != scale_bytes ||
        (uint64_t)(VkDeviceSize)scale_offset != scale_offset ||
        (uint64_t)(VkDeviceSize)packed_size != packed_size)
        return COLI_VULKAN_LIMIT_EXCEEDED;
    memset(layout, 0, sizeof(*layout));
    layout->weight_offset = 0;
    layout->weight_size = (VkDeviceSize)weight_bytes;
    layout->scale_offset = scale_bytes ? (VkDeviceSize)scale_offset : 0;
    layout->scale_size = (VkDeviceSize)scale_bytes;
    layout->packed_size = (VkDeviceSize)packed_size;
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_select_strict_memory_type(
    const VkPhysicalDeviceMemoryProperties *properties,
    uint32_t memory_type_bits,
    VkDeviceSize allocation_size,
    uint64_t reserve_bytes,
    uint32_t *memory_type_index,
    uint32_t *heap_index,
    uint64_t *heap_usable_bytes
) {
    if (!properties || !memory_type_index || !heap_index || !heap_usable_bytes ||
        properties->memoryTypeCount == 0 ||
        properties->memoryTypeCount > VK_MAX_MEMORY_TYPES ||
        properties->memoryHeapCount == 0 ||
        properties->memoryHeapCount > VK_MAX_MEMORY_HEAPS ||
        memory_type_bits == 0 || allocation_size == 0)
        return COLI_VULKAN_INVALID_ARGUMENT;

    int found_strict = 0;
    for (uint32_t i = 0; i < properties->memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = properties->memoryTypes[i].propertyFlags;
        if (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            continue;
        found_strict = 1;
        if (!(memory_type_bits & (1u << i))) continue;
        uint32_t hi = properties->memoryTypes[i].heapIndex;
        if (hi >= properties->memoryHeapCount) continue;
        const VkMemoryHeap *heap = &properties->memoryHeaps[hi];
        if (!(heap->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ||
            heap->size <= reserve_bytes)
            continue;
        uint64_t usable = (uint64_t)heap->size - reserve_bytes;
        if ((uint64_t)allocation_size > usable) continue;
        *memory_type_index = i;
        *heap_index = hi;
        *heap_usable_bytes = usable;
        return COLI_VULKAN_OK;
    }
    return found_strict ? COLI_VULKAN_LIMIT_EXCEEDED : COLI_VULKAN_UNSUPPORTED;
}

static void context_mark_device_lost(ColiVulkanContext *context) {
    context->device_lost = 1;
    context->usable = 0;
    context->degraded = 1;
}

static void context_clear_gate(void) {
    pthread_mutex_lock(&g_context_gate);
    g_context_active = 0;
    pthread_mutex_unlock(&g_context_gate);
}

static void context_partial_cleanup(ColiVulkanContext *context) {
    if (!context) return;
    if (context->device != VK_NULL_HANDLE && context->command_pool != VK_NULL_HANDLE)
        context->api->DestroyCommandPool(context->device, context->command_pool, NULL);
    if (context->device != VK_NULL_HANDLE)
        context->api->DestroyDevice(context->device, NULL);
    if (context->instance != VK_NULL_HANDLE)
        context->api->DestroyInstance(context->instance, NULL);
    pthread_mutex_destroy(&context->mutex);
    free(context);
}

static ColiVulkanResult enumerate_devices(
    const VulkanApi *api,
    VkInstance instance,
    VkPhysicalDevice **devices_out,
    uint32_t *count_out
) {
    VkPhysicalDevice *devices = NULL;
    for (int attempt = 0; attempt < 8; attempt++) {
        uint32_t count = 0;
        VkResult result = api->EnumeratePhysicalDevices(instance, &count, NULL);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE)
            return result_from_vk(result);
        if (count == 0) return COLI_VULKAN_UNAVAILABLE;
        free(devices);
        devices = calloc((size_t)count, sizeof(*devices));
        if (!devices) return COLI_VULKAN_OUT_OF_MEMORY;
        uint32_t written = count;
        result = api->EnumeratePhysicalDevices(instance, &written, devices);
        if (result == VK_SUCCESS) {
            *devices_out = devices;
            *count_out = written;
            return written ? COLI_VULKAN_OK : COLI_VULKAN_UNAVAILABLE;
        }
        if (result != VK_INCOMPLETE) {
            free(devices);
            return result_from_vk(result);
        }
    }
    free(devices);
    return COLI_VULKAN_ERROR;
}

ColiVulkanResult coli_vulkan_context_create(
    ColiVulkanContext **context_out,
    const ColiVulkanConfig *config
) {
    if (!context_out || *context_out || !config || config->expert_budget_bytes == 0 ||
        config->upload_timeout_ns == 0)
        return COLI_VULKAN_INVALID_ARGUMENT;
    const VulkanApi *api = config_api(config);
    if (!api_complete(api)) return COLI_VULKAN_INVALID_ARGUMENT;

    pthread_mutex_lock(&g_context_gate);
    if (g_context_active) {
        pthread_mutex_unlock(&g_context_gate);
        return COLI_VULKAN_BUSY;
    }
    g_context_active = 1;
    pthread_mutex_unlock(&g_context_gate);

    ColiVulkanContext *context = calloc(1, sizeof(*context));
    if (!context) {
        context_clear_gate();
        return COLI_VULKAN_OUT_OF_MEMORY;
    }
    context->api = api;
    context->requested_budget_bytes = config->expert_budget_bytes;
    context->upload_timeout_ns = config->upload_timeout_ns;
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        free(context);
        context_clear_gate();
        return COLI_VULKAN_ERROR;
    }

    VkApplicationInfo application = {0};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Colibri Vulkan Phase 3A";
    application.applicationVersion = VK_MAKE_VERSION(3, 0, 0);
    application.pEngineName = "Colibri";
    application.engineVersion = VK_MAKE_VERSION(3, 0, 0);
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info = {0};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    VkResult vk_result = api->CreateInstance(&instance_info, NULL, &context->instance);
    if (vk_result != VK_SUCCESS) {
        ColiVulkanResult result = result_from_vk(vk_result);
        context_partial_cleanup(context);
        context_clear_gate();
        return result;
    }

    VkPhysicalDevice *devices = NULL;
    uint32_t device_count = 0;
    ColiVulkanResult result = enumerate_devices(api, context->instance, &devices, &device_count);
    if (result != COLI_VULKAN_OK) {
        context_partial_cleanup(context);
        context_clear_gate();
        return result;
    }

    uint32_t matches = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties properties;
        memset(&properties, 0, sizeof(properties));
        api->GetPhysicalDeviceProperties(devices[i], &properties);
        if (properties.vendorID == COLI_VULKAN_VENDOR_ID &&
            properties.deviceID == COLI_VULKAN_DEVICE_ID) {
            matches++;
            context->physical_device = devices[i];
            context->properties = properties;
        }
    }
    free(devices);
    if (matches == 0) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNAVAILABLE;
    }
    if (matches != 1) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }

    context->max_memory_allocation_count =
        context->properties.limits.maxMemoryAllocationCount;
    if (context->max_memory_allocation_count == 0) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }

    api->GetPhysicalDeviceMemoryProperties(context->physical_device,
        &context->memory_properties);
    if (context->memory_properties.memoryTypeCount == 0 ||
        context->memory_properties.memoryTypeCount > VK_MAX_MEMORY_TYPES ||
        context->memory_properties.memoryHeapCount == 0 ||
        context->memory_properties.memoryHeapCount > VK_MAX_MEMORY_HEAPS) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }

    uint64_t maximum_usable_heap = 0;
    for (uint32_t i = 0; i < context->memory_properties.memoryTypeCount; i++) {
        const VkMemoryType *type = &context->memory_properties.memoryTypes[i];
        if (!(type->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
            type->heapIndex >= context->memory_properties.memoryHeapCount)
            continue;
        const VkMemoryHeap *heap = &context->memory_properties.memoryHeaps[type->heapIndex];
        if (!(heap->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ||
            heap->size <= COLI_VULKAN_VRAM_RESERVE_BYTES)
            continue;
        uint64_t usable = (uint64_t)heap->size - COLI_VULKAN_VRAM_RESERVE_BYTES;
        if (usable > maximum_usable_heap) maximum_usable_heap = usable;
    }
    if (!maximum_usable_heap) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }
    context->effective_budget_bytes = config->expert_budget_bytes < maximum_usable_heap
        ? config->expert_budget_bytes : maximum_usable_heap;

    uint32_t queue_count = 0;
    api->GetPhysicalDeviceQueueFamilyProperties(context->physical_device,
        &queue_count, NULL);
    if (queue_count == 0) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }
    VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
    if (!queues) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_OUT_OF_MEMORY;
    }
    api->GetPhysicalDeviceQueueFamilyProperties(context->physical_device,
        &queue_count, queues);
    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; i++) {
        if (queues[i].queueCount > 0 && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            queue_family = i;
            break;
        }
    }
    free(queues);
    if (queue_family == UINT32_MAX) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_UNSUPPORTED;
    }
    context->queue_family_index = queue_family;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {0};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    vk_result = api->CreateDevice(context->physical_device, &device_info, NULL,
        &context->device);
    if (vk_result != VK_SUCCESS) {
        result = result_from_vk(vk_result);
        context_partial_cleanup(context);
        context_clear_gate();
        return result;
    }
    api->GetDeviceQueue(context->device, queue_family, 0, &context->queue);
    if (context->queue == VK_NULL_HANDLE) {
        context_partial_cleanup(context);
        context_clear_gate();
        return COLI_VULKAN_ERROR;
    }

    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_result = api->CreateCommandPool(context->device, &pool_info, NULL,
        &context->command_pool);
    if (vk_result != VK_SUCCESS) {
        result = result_from_vk(vk_result);
        context_partial_cleanup(context);
        context_clear_gate();
        return result;
    }

    context->usable = 1;
    *context_out = context;
    return COLI_VULKAN_OK;
}

static int tensor_is_live_locked(
    const ColiVulkanContext *context,
    const ColiVulkanTensor *tensor
) {
    for (const ColiVulkanTensor *current = context->tensors; current;
         current = current->next)
        if (current == tensor) return 1;
    return 0;
}

static void tensor_destroy_locked(
    ColiVulkanContext *context,
    ColiVulkanTensor *tensor
) {
    ColiVulkanTensor **link = &context->tensors;
    while (*link && *link != tensor) link = &(*link)->next;
    if (*link == tensor) *link = tensor->next;
    if (tensor->buffer != VK_NULL_HANDLE)
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
    if (tensor->memory != VK_NULL_HANDLE) {
        context->api->FreeMemory(context->device, tensor->memory, NULL);
        if (context->live_allocations) context->live_allocations--;
    }
    if (context->committed_bytes >= (uint64_t)tensor->allocation_size)
        context->committed_bytes -= (uint64_t)tensor->allocation_size;
    else
        context->committed_bytes = 0;
    if (context->live_tensors) context->live_tensors--;
    memset(tensor, 0, sizeof(*tensor));
    free(tensor);
}

static ColiVulkanResult create_destination_locked(
    ColiVulkanContext *context,
    const ColiVulkanTensorLayout *layout,
    ColiVulkanTensor **tensor_out
) {
    ColiVulkanTensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return COLI_VULKAN_OUT_OF_MEMORY;
    tensor->context = context;
    tensor->layout = *layout;
    tensor->state = COLI_VULKAN_TENSOR_ALLOCATED;

    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = layout->packed_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult vk_result = context->api->CreateBuffer(context->device, &buffer_info,
        NULL, &tensor->buffer);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        free(tensor);
        return result_from_vk(vk_result);
    }

    VkMemoryRequirements requirements = {0};
    context->api->GetBufferMemoryRequirements(context->device, tensor->buffer,
        &requirements);
    if (requirements.size < layout->packed_size || requirements.size == 0 ||
        requirements.memoryTypeBits == 0) {
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        free(tensor);
        return COLI_VULKAN_ERROR;
    }

    uint64_t heap_usable = 0;
    ColiVulkanResult result = coli_vulkan_select_strict_memory_type(
        &context->memory_properties, requirements.memoryTypeBits,
        requirements.size, COLI_VULKAN_VRAM_RESERVE_BYTES,
        &tensor->memory_type_index, &tensor->heap_index, &heap_usable);
    if (result != COLI_VULKAN_OK) {
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        free(tensor);
        return result;
    }
    if ((uint64_t)requirements.size > context->effective_budget_bytes ||
        context->committed_bytes > context->effective_budget_bytes -
            (uint64_t)requirements.size ||
        context->committed_bytes > heap_usable - (uint64_t)requirements.size) {
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        free(tensor);
        return COLI_VULKAN_LIMIT_EXCEEDED;
    }
    if (context->live_allocations >= context->max_memory_allocation_count) {
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        free(tensor);
        return COLI_VULKAN_LIMIT_EXCEEDED;
    }

    VkMemoryAllocateInfo allocation_info = {0};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = tensor->memory_type_index;
    vk_result = context->api->AllocateMemory(context->device, &allocation_info,
        NULL, &tensor->memory);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        free(tensor);
        return result_from_vk(vk_result);
    }
    context->live_allocations++;

    vk_result = context->api->BindBufferMemory(context->device, tensor->buffer,
        tensor->memory, 0);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        context->api->FreeMemory(context->device, tensor->memory, NULL);
        context->live_allocations--;
        free(tensor);
        return result_from_vk(vk_result);
    }

    tensor->allocation_size = requirements.size;
    tensor->memory_property_flags =
        context->memory_properties.memoryTypes[tensor->memory_type_index].propertyFlags;
    if (!(tensor->memory_property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        (tensor->memory_property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        context->api->DestroyBuffer(context->device, tensor->buffer, NULL);
        context->api->FreeMemory(context->device, tensor->memory, NULL);
        context->live_allocations--;
        free(tensor);
        return COLI_VULKAN_ERROR;
    }

    context->committed_bytes += (uint64_t)requirements.size;
    context->live_tensors++;
    tensor->next = context->tensors;
    context->tensors = tensor;
    *tensor_out = tensor;
    return COLI_VULKAN_OK;
}

static void register_pending_locked(
    ColiVulkanContext *context,
    PendingKind kind,
    ColiVulkanTensor *tensor,
    const VulkanUploadOp *op
) {
    context->pending.kind = kind;
    context->pending.tensor = tensor;
    context->pending.op = *op;
}

static ColiVulkanResult finish_pending_locked(
    ColiVulkanContext *context,
    uint64_t timeout_ns
) {
    if (context->pending.kind == PENDING_NONE) return COLI_VULKAN_OK;
    VulkanUploadResult upload_result = vulkan_upload_finish(&context->pending.op,
        timeout_ns);
    if (upload_result == VULKAN_UPLOAD_TIMEOUT) return COLI_VULKAN_TIMEOUT;
    if (upload_result == VULKAN_UPLOAD_DEVICE_LOST) {
        context_mark_device_lost(context);
        if (context->pending.tensor)
            context->pending.tensor->state = COLI_VULKAN_TENSOR_FAILED;
        return COLI_VULKAN_DEVICE_LOST;
    }
    if (upload_result != VULKAN_UPLOAD_SUCCESS) {
        context->degraded = 1;
        context->usable = 0;
        if (context->pending.tensor && context->pending.kind == PENDING_UPLOAD)
            context->pending.tensor->state = COLI_VULKAN_TENSOR_FAILED;
        return COLI_VULKAN_ERROR;
    }

    if (context->live_allocations) context->live_allocations--;
    ColiVulkanTensor *tensor = context->pending.tensor;
    PendingKind kind = context->pending.kind;
    memset(&context->pending, 0, sizeof(context->pending));
    if (tensor && kind == PENDING_UPLOAD)
        tensor->state = COLI_VULKAN_TENSOR_READY;
    if (tensor && tensor->destroy_requested)
        tensor_destroy_locked(context, tensor);
    context->degraded = 0;
    if (!context->device_lost && !context->shutting_down) context->usable = 1;
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_tensor_upload(
    ColiVulkanContext *context,
    ColiVulkanTensor **tensor_out,
    const void *weights,
    uint64_t weight_bytes,
    const void *scales,
    uint64_t scale_bytes,
    uint64_t timeout_ns
) {
    if (!context || !tensor_out || *tensor_out || !weights ||
        (scale_bytes && !scales))
        return COLI_VULKAN_INVALID_ARGUMENT;
    VkDeviceSize alignment = context->properties.limits.minStorageBufferOffsetAlignment;
    if (!alignment) alignment = 1;
    ColiVulkanTensorLayout layout;
    ColiVulkanResult result = coli_vulkan_plan_tensor_layout(weight_bytes,
        scale_bytes, alignment, &layout);
    if (result != COLI_VULKAN_OK) return result;
    if (layout.packed_size > (VkDeviceSize)SIZE_MAX)
        return COLI_VULKAN_LIMIT_EXCEEDED;
    uint8_t *packed = calloc(1, (size_t)layout.packed_size);
    if (!packed) return COLI_VULKAN_OUT_OF_MEMORY;
    memcpy(packed + (size_t)layout.weight_offset, weights, (size_t)layout.weight_size);
    if (layout.scale_size)
        memcpy(packed + (size_t)layout.scale_offset, scales, (size_t)layout.scale_size);

    pthread_mutex_lock(&context->mutex);
    if (!context->usable || context->device_lost || context->shutting_down) {
        result = context->device_lost ? COLI_VULKAN_DEVICE_LOST : COLI_VULKAN_BUSY;
        pthread_mutex_unlock(&context->mutex);
        free(packed);
        return result;
    }
    if (context->pending.kind != PENDING_NONE) {
        pthread_mutex_unlock(&context->mutex);
        free(packed);
        return COLI_VULKAN_BUSY;
    }

    ColiVulkanTensor *tensor = NULL;
    result = create_destination_locked(context, &layout, &tensor);
    if (result != COLI_VULKAN_OK) {
        pthread_mutex_unlock(&context->mutex);
        free(packed);
        return result;
    }
    if (context->live_allocations >= context->max_memory_allocation_count) {
        tensor_destroy_locked(context, tensor);
        pthread_mutex_unlock(&context->mutex);
        free(packed);
        return COLI_VULKAN_LIMIT_EXCEEDED;
    }

    /* Reserve one allocation slot for the helper's staging memory. */
    context->live_allocations++;
    VulkanUploadOp op = {0};
    uint64_t wait_ns = timeout_ns ? timeout_ns : context->upload_timeout_ns;
    VulkanUploadResult upload_result = staged_upload_with_api(context->api,
        context->device, context->queue, context->command_pool,
        &context->memory_properties, tensor->buffer, layout.packed_size, packed,
        wait_ns, &op);
    free(packed);

    if (upload_result == VULKAN_UPLOAD_SUCCESS) {
        context->live_allocations--;
        tensor->state = COLI_VULKAN_TENSOR_READY;
        *tensor_out = tensor;
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_OK;
    }
    if (op.device == VK_NULL_HANDLE) {
        context->live_allocations--;
        if (upload_result == VULKAN_UPLOAD_DEVICE_LOST)
            context_mark_device_lost(context);
        tensor_destroy_locked(context, tensor);
        pthread_mutex_unlock(&context->mutex);
        return upload_result == VULKAN_UPLOAD_DEVICE_LOST
            ? COLI_VULKAN_DEVICE_LOST : COLI_VULKAN_ERROR;
    }

    tensor->state = upload_result == VULKAN_UPLOAD_TIMEOUT
        ? COLI_VULKAN_TENSOR_PENDING : COLI_VULKAN_TENSOR_FAILED;
    register_pending_locked(context, PENDING_UPLOAD, tensor, &op);
    *tensor_out = tensor;
    if (upload_result == VULKAN_UPLOAD_DEVICE_LOST) {
        context_mark_device_lost(context);
        result = COLI_VULKAN_DEVICE_LOST;
    } else if (upload_result == VULKAN_UPLOAD_TIMEOUT) {
        result = COLI_VULKAN_TIMEOUT;
    } else {
        context->degraded = 1;
        context->usable = 0;
        result = COLI_VULKAN_ERROR;
    }
    pthread_mutex_unlock(&context->mutex);
    return result;
}

static uint32_t find_host_memory_type(
    const VkPhysicalDeviceMemoryProperties *properties,
    uint32_t memory_type_bits
) {
    for (uint32_t i = 0; i < properties->memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = properties->memoryTypes[i].propertyFlags;
        if ((memory_type_bits & (1u << i)) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            return i;
    }
    return UINT32_MAX;
}

ColiVulkanResult coli_vulkan_tensor_readback(
    ColiVulkanContext *context,
    ColiVulkanTensor *tensor,
    void *output,
    uint64_t output_bytes,
    uint64_t timeout_ns
) {
    if (!context || !tensor || !output ||
        output_bytes < (uint64_t)tensor->layout.packed_size ||
        tensor->layout.packed_size > (VkDeviceSize)SIZE_MAX)
        return COLI_VULKAN_INVALID_ARGUMENT;
    pthread_mutex_lock(&context->mutex);
    if (!tensor_is_live_locked(context, tensor) || tensor->context != context) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_INVALID_ARGUMENT;
    }
    if (!context->usable || context->device_lost || context->shutting_down ||
        tensor->state != COLI_VULKAN_TENSOR_READY) {
        ColiVulkanResult result = context->device_lost
            ? COLI_VULKAN_DEVICE_LOST : COLI_VULKAN_BUSY;
        pthread_mutex_unlock(&context->mutex);
        return result;
    }
    if (context->pending.kind != PENDING_NONE) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_BUSY;
    }

    VulkanUploadOp op = {0};
    op.device = context->device;
    op.commandPool = context->command_pool;
    op.api = context->api;
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = tensor->layout.packed_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult vk_result = context->api->CreateBuffer(context->device, &buffer_info,
        NULL, &op.stagingBuffer);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        pthread_mutex_unlock(&context->mutex);
        return result_from_vk(vk_result);
    }

    VkMemoryRequirements requirements = {0};
    context->api->GetBufferMemoryRequirements(context->device, op.stagingBuffer,
        &requirements);
    uint32_t memory_type = find_host_memory_type(&context->memory_properties,
        requirements.memoryTypeBits);
    if (requirements.size < tensor->layout.packed_size ||
        requirements.memoryTypeBits == 0 || memory_type == UINT32_MAX ||
        context->live_allocations >= context->max_memory_allocation_count) {
        operation_cleanup(context->api, &op);
        pthread_mutex_unlock(&context->mutex);
        return memory_type == UINT32_MAX ? COLI_VULKAN_UNSUPPORTED
            : COLI_VULKAN_LIMIT_EXCEEDED;
    }

    VkMemoryAllocateInfo allocation_info = {0};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type;
    vk_result = context->api->AllocateMemory(context->device, &allocation_info,
        NULL, &op.stagingMemory);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        operation_cleanup(context->api, &op);
        pthread_mutex_unlock(&context->mutex);
        return result_from_vk(vk_result);
    }
    context->live_allocations++;
    vk_result = context->api->BindBufferMemory(context->device, op.stagingBuffer,
        op.stagingMemory, 0);
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        operation_cleanup(context->api, &op);
        context->live_allocations--;
        pthread_mutex_unlock(&context->mutex);
        return result_from_vk(vk_result);
    }

    VkCommandBufferAllocateInfo command_info = {0};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = context->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    vk_result = context->api->AllocateCommandBuffers(context->device,
        &command_info, &op.commandBuffer);
    if (vk_result == VK_SUCCESS) {
        VkCommandBufferBeginInfo begin_info = {0};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_result = context->api->BeginCommandBuffer(op.commandBuffer, &begin_info);
    }
    if (vk_result == VK_SUCCESS) {
        VkBufferCopy copy = {0};
        copy.size = tensor->layout.packed_size;
        context->api->CmdCopyBuffer(op.commandBuffer, tensor->buffer,
            op.stagingBuffer, 1, &copy);
        vk_result = context->api->EndCommandBuffer(op.commandBuffer);
    }
    if (vk_result == VK_SUCCESS) {
        VkFenceCreateInfo fence_info = {0};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk_result = context->api->CreateFence(context->device, &fence_info, NULL,
            &op.fence);
    }
    if (vk_result != VK_SUCCESS) {
        if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
        operation_cleanup(context->api, &op);
        context->live_allocations--;
        pthread_mutex_unlock(&context->mutex);
        return result_from_vk(vk_result);
    }

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &op.commandBuffer;
    vk_result = context->api->QueueSubmit(context->queue, 1, &submit_info, op.fence);
    if (vk_result != VK_SUCCESS && vk_result != VK_ERROR_DEVICE_LOST) {
        operation_cleanup(context->api, &op);
        context->live_allocations--;
        pthread_mutex_unlock(&context->mutex);
        return result_from_vk(vk_result);
    }
    if (vk_result == VK_SUCCESS) {
        uint64_t wait_ns = timeout_ns ? timeout_ns : context->upload_timeout_ns;
        vk_result = context->api->WaitForFences(context->device, 1, &op.fence,
            VK_TRUE, wait_ns);
    }
    op.lastResult = vk_result;
    if (vk_result != VK_SUCCESS) {
        register_pending_locked(context, PENDING_READBACK, tensor, &op);
        if (vk_result == VK_ERROR_DEVICE_LOST) {
            context_mark_device_lost(context);
            pthread_mutex_unlock(&context->mutex);
            return COLI_VULKAN_DEVICE_LOST;
        }
        if (vk_result != VK_TIMEOUT) {
            context->degraded = 1;
            context->usable = 0;
        }
        pthread_mutex_unlock(&context->mutex);
        return vk_result == VK_TIMEOUT ? COLI_VULKAN_TIMEOUT : COLI_VULKAN_ERROR;
    }

    void *mapped = NULL;
    vk_result = context->api->MapMemory(context->device, op.stagingMemory, 0,
        tensor->layout.packed_size, 0, &mapped);
    if (vk_result == VK_SUCCESS && mapped) {
        memcpy(output, mapped, (size_t)tensor->layout.packed_size);
        context->api->UnmapMemory(context->device, op.stagingMemory);
    }
    operation_cleanup(context->api, &op);
    context->live_allocations--;
    if (vk_result == VK_ERROR_DEVICE_LOST) context_mark_device_lost(context);
    pthread_mutex_unlock(&context->mutex);
    if (vk_result != VK_SUCCESS || !mapped)
        return vk_result == VK_SUCCESS ? COLI_VULKAN_ERROR : result_from_vk(vk_result);
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_tensor_get_info(
    const ColiVulkanTensor *tensor,
    ColiVulkanTensorInfo *info
) {
    if (!tensor || !info || !tensor->context) return COLI_VULKAN_INVALID_ARGUMENT;
    ColiVulkanContext *context = tensor->context;
    pthread_mutex_lock(&context->mutex);
    if (!tensor_is_live_locked(context, tensor)) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_INVALID_ARGUMENT;
    }
    info->layout = tensor->layout;
    info->allocation_size = tensor->allocation_size;
    info->memory_type_index = tensor->memory_type_index;
    info->heap_index = tensor->heap_index;
    info->memory_property_flags = tensor->memory_property_flags;
    info->state = tensor->state;
    pthread_mutex_unlock(&context->mutex);
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_tensor_free(
    ColiVulkanContext *context,
    ColiVulkanTensor **tensor_pointer
) {
    if (!context || !tensor_pointer || !*tensor_pointer)
        return COLI_VULKAN_INVALID_ARGUMENT;
    pthread_mutex_lock(&context->mutex);
    ColiVulkanTensor *tensor = *tensor_pointer;
    if (tensor->context != context || !tensor_is_live_locked(context, tensor)) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_INVALID_ARGUMENT;
    }
    if (context->device_lost) {
        tensor->destroy_requested = 1;
        tensor->state = COLI_VULKAN_TENSOR_DESTROY_PENDING;
        *tensor_pointer = NULL;
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_DEVICE_LOST;
    }
    if (context->pending.kind != PENDING_NONE &&
        context->pending.tensor == tensor) {
        tensor->destroy_requested = 1;
        tensor->state = COLI_VULKAN_TENSOR_DESTROY_PENDING;
        *tensor_pointer = NULL;
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_BUSY;
    }
    tensor_destroy_locked(context, tensor);
    *tensor_pointer = NULL;
    pthread_mutex_unlock(&context->mutex);
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_finish_pending(
    ColiVulkanContext *context,
    uint64_t timeout_ns
) {
    if (!context) return COLI_VULKAN_INVALID_ARGUMENT;
    pthread_mutex_lock(&context->mutex);
    ColiVulkanResult result = finish_pending_locked(context, timeout_ns);
    pthread_mutex_unlock(&context->mutex);
    return result;
}

ColiVulkanResult coli_vulkan_context_get_info(
    const ColiVulkanContext *context_const,
    ColiVulkanContextInfo *info
) {
    if (!context_const || !info) return COLI_VULKAN_INVALID_ARGUMENT;
    ColiVulkanContext *context = (ColiVulkanContext *)context_const;
    pthread_mutex_lock(&context->mutex);
    memset(info, 0, sizeof(*info));
    info->vendor_id = context->properties.vendorID;
    info->device_id = context->properties.deviceID;
    info->queue_family_index = context->queue_family_index;
    info->memory_heap_count = context->memory_properties.memoryHeapCount;
    info->memory_type_count = context->memory_properties.memoryTypeCount;
    info->max_memory_allocation_count = context->max_memory_allocation_count;
    info->min_storage_buffer_offset_alignment =
        context->properties.limits.minStorageBufferOffsetAlignment;
    info->requested_budget_bytes = context->requested_budget_bytes;
    info->effective_budget_bytes = context->effective_budget_bytes;
    info->committed_bytes = context->committed_bytes;
    info->live_tensors = context->live_tensors;
    info->live_allocations = context->live_allocations;
    info->pending_operations = context->pending.kind != PENDING_NONE;
    info->usable = context->usable;
    info->device_lost = context->device_lost;
    pthread_mutex_unlock(&context->mutex);
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_context_get_memory_properties(
    const ColiVulkanContext *context_const,
    VkPhysicalDeviceMemoryProperties *properties
) {
    if (!context_const || !properties) return COLI_VULKAN_INVALID_ARGUMENT;
    ColiVulkanContext *context = (ColiVulkanContext *)context_const;
    pthread_mutex_lock(&context->mutex);
    *properties = context->memory_properties;
    pthread_mutex_unlock(&context->mutex);
    return COLI_VULKAN_OK;
}

ColiVulkanResult coli_vulkan_context_destroy(
    ColiVulkanContext **context_pointer,
    uint64_t timeout_ns
) {
    if (!context_pointer || !*context_pointer) return COLI_VULKAN_INVALID_ARGUMENT;
    ColiVulkanContext *context = *context_pointer;
    pthread_mutex_lock(&context->mutex);
    context->shutting_down = 1;
    context->usable = 0;
    if (context->pending.kind != PENDING_NONE) {
        if (context->pending.tensor) {
            context->pending.tensor->destroy_requested = 1;
            context->pending.tensor->state = COLI_VULKAN_TENSOR_DESTROY_PENDING;
        }
        ColiVulkanResult result = finish_pending_locked(context, timeout_ns);
        if (result != COLI_VULKAN_OK) {
            pthread_mutex_unlock(&context->mutex);
            return result;
        }
    }
    if (context->device_lost) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_DEVICE_LOST;
    }
    while (context->tensors) tensor_destroy_locked(context, context->tensors);
    if (context->live_allocations != 0) {
        pthread_mutex_unlock(&context->mutex);
        return COLI_VULKAN_ERROR;
    }
    context->api->DestroyCommandPool(context->device, context->command_pool, NULL);
    context->command_pool = VK_NULL_HANDLE;
    context->api->DestroyDevice(context->device, NULL);
    context->device = VK_NULL_HANDLE;
    context->api->DestroyInstance(context->instance, NULL);
    context->instance = VK_NULL_HANDLE;
    pthread_mutex_unlock(&context->mutex);
    pthread_mutex_destroy(&context->mutex);
    memset(context, 0, sizeof(*context));
    free(context);
    *context_pointer = NULL;
    context_clear_gate();
    return COLI_VULKAN_OK;
}
