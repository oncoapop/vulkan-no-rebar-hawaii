#define COLI_VULKAN_TESTING 1
#include "../backend_vulkan.h"

#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_TIMEOUT_NS 1000000ULL
#define MAX_FAKE_OBJECTS 128
#define HANDLE(type, value) ((type)(uintptr_t)(value))

typedef enum {
    FP_NONE = 0,
    FP_CREATE_INSTANCE,
    FP_ENUMERATE,
    FP_CREATE_DEVICE,
    FP_CREATE_POOL,
    FP_CREATE_BUFFER,
    FP_ALLOCATE_MEMORY,
    FP_BIND_MEMORY,
    FP_MAP_MEMORY,
    FP_ALLOCATE_COMMAND,
    FP_BEGIN_COMMAND,
    FP_END_COMMAND,
    FP_CREATE_FENCE,
    FP_QUEUE_SUBMIT,
    FP_WAIT_FENCE,
    FP_CREATE_SHADER,
    FP_CREATE_DESCRIPTOR_LAYOUT,
    FP_CREATE_PIPELINE_LAYOUT,
    FP_CREATE_PIPELINE,
    FP_CREATE_DESCRIPTOR_POOL,
    FP_ALLOCATE_DESCRIPTOR_SET,
    FP_COUNT
} FaultPoint;

typedef struct {
    VkBuffer handle;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkDeviceMemory memory;
    int live;
} FakeBuffer;

typedef struct {
    VkDeviceMemory handle;
    VkDeviceSize size;
    uint8_t *data;
    int live;
} FakeMemory;

typedef struct {
    uint64_t next_handle;
    uint32_t physical_device_count;
    uint32_t matching_device_count;
    uint32_t max_allocations;
    VkDeviceSize storage_alignment;
    FaultPoint fault_point;
    unsigned fault_call;
    VkResult fault_result;
    unsigned point_calls[FP_COUNT];
    VkResult wait_results[8];
    unsigned wait_result_count;
    unsigned wait_result_index;
    FakeBuffer buffers[MAX_FAKE_OBJECTS];
    FakeMemory memories[MAX_FAKE_OBJECTS];
    unsigned create_instance_count;
    unsigned destroy_instance_count;
    unsigned create_device_count;
    unsigned destroy_device_count;
    unsigned create_pool_count;
    unsigned destroy_pool_count;
    unsigned create_buffer_count;
    unsigned destroy_buffer_count;
    unsigned allocate_memory_count;
    unsigned free_memory_count;
    unsigned allocate_command_buffer_count;
    unsigned free_command_buffer_count;
    unsigned create_fence_count;
    unsigned destroy_fence_count;
    unsigned queue_submit_count;
    unsigned create_shader_count, destroy_shader_count;
    unsigned create_descriptor_layout_count, destroy_descriptor_layout_count;
    unsigned create_pipeline_layout_count, destroy_pipeline_layout_count;
    unsigned create_pipeline_count, destroy_pipeline_count;
    unsigned create_descriptor_pool_count, destroy_descriptor_pool_count;
    unsigned allocate_descriptor_set_count;
    unsigned update_descriptor_count;
    unsigned pipeline_barrier_count;
    unsigned bind_pipeline_count, bind_descriptor_count, push_constant_count;
    unsigned dispatch_count;
    unsigned host_to_compute_barrier_count;
    unsigned compute_to_host_barrier_count;
    VkBuffer descriptor_buffers[3];
    VkDeviceSize descriptor_ranges[3];
    uint32_t push_constants[6];
    volatile unsigned api_active;
    volatile unsigned api_max_active;
} FakeState;

static FakeState g_fake;
static int g_failures;

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __func__, __LINE__, message); \
            g_failures++; \
            return; \
        } \
    } while (0)

#define CHECK_RESULT(expression, expected, message) \
    do { \
        ColiVulkanResult got_ = (expression); \
        if (got_ != (expected)) { \
            fprintf(stderr, "FAIL: %s:%d: %s: got %s expected %s\n", \
                __func__, __LINE__, message, coli_vulkan_result_string(got_), \
                coli_vulkan_result_string(expected)); \
            g_failures++; \
            return; \
        } \
    } while (0)

static uint64_t next_handle(void) {
    return ++g_fake.next_handle + 1000;
}

static VkResult maybe_fault(FaultPoint point) {
    unsigned call = ++g_fake.point_calls[point];
    if (g_fake.fault_point == point && call == g_fake.fault_call)
        return g_fake.fault_result;
    return VK_SUCCESS;
}

static void set_fault(FaultPoint point, unsigned call, VkResult result) {
    g_fake.fault_point = point;
    g_fake.fault_call = call;
    g_fake.fault_result = result;
    memset(g_fake.point_calls, 0, sizeof(g_fake.point_calls));
}

static void fake_check_no_live_resources(const char *where) {
    unsigned live_buffers = 0;
    unsigned live_memories = 0;
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++) {
        live_buffers += g_fake.buffers[i].live != 0;
        live_memories += g_fake.memories[i].live != 0;
    }
    if (live_buffers || live_memories ||
        g_fake.create_instance_count != g_fake.destroy_instance_count ||
        g_fake.create_device_count != g_fake.destroy_device_count ||
        g_fake.create_pool_count != g_fake.destroy_pool_count ||
        g_fake.create_buffer_count != g_fake.destroy_buffer_count ||
        g_fake.allocate_memory_count != g_fake.free_memory_count ||
        g_fake.allocate_command_buffer_count !=
            g_fake.free_command_buffer_count ||
        g_fake.create_fence_count != g_fake.destroy_fence_count) {
        fprintf(stderr,
            "FAIL: %s: leaked fake Vulkan owners: buffers=%u memories=%u "
            "instance=%u/%u device=%u/%u pool=%u/%u buffer=%u/%u memory=%u/%u "
            "commandBuffer=%u/%u fence=%u/%u\n",
            where, live_buffers, live_memories,
            g_fake.create_instance_count, g_fake.destroy_instance_count,
            g_fake.create_device_count, g_fake.destroy_device_count,
            g_fake.create_pool_count, g_fake.destroy_pool_count,
            g_fake.create_buffer_count, g_fake.destroy_buffer_count,
            g_fake.allocate_memory_count, g_fake.free_memory_count,
            g_fake.allocate_command_buffer_count,
            g_fake.free_command_buffer_count,
            g_fake.create_fence_count, g_fake.destroy_fence_count);
        g_failures++;
    }
    if (g_fake.create_shader_count != g_fake.destroy_shader_count ||
        g_fake.create_descriptor_layout_count !=
            g_fake.destroy_descriptor_layout_count ||
        g_fake.create_pipeline_layout_count !=
            g_fake.destroy_pipeline_layout_count ||
        g_fake.create_pipeline_count != g_fake.destroy_pipeline_count ||
        g_fake.create_descriptor_pool_count !=
            g_fake.destroy_descriptor_pool_count) {
        fprintf(stderr,
            "FAIL: %s: leaked fake compute owners: shader=%u/%u "
            "descriptorLayout=%u/%u pipelineLayout=%u/%u pipeline=%u/%u "
            "descriptorPool=%u/%u\n", where,
            g_fake.create_shader_count, g_fake.destroy_shader_count,
            g_fake.create_descriptor_layout_count,
            g_fake.destroy_descriptor_layout_count,
            g_fake.create_pipeline_layout_count,
            g_fake.destroy_pipeline_layout_count,
            g_fake.create_pipeline_count, g_fake.destroy_pipeline_count,
            g_fake.create_descriptor_pool_count,
            g_fake.destroy_descriptor_pool_count);
        g_failures++;
    }
}

static void fake_reset(void) {
    fake_check_no_live_resources("before fake reset");
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++) free(g_fake.memories[i].data);
    memset(&g_fake, 0, sizeof(g_fake));
    g_fake.next_handle = 1000;
    g_fake.physical_device_count = 1;
    g_fake.matching_device_count = 1;
    g_fake.max_allocations = 4096;
    g_fake.storage_alignment = 256;
    g_fake.fault_call = 1;
    g_fake.fault_result = VK_ERROR_INITIALIZATION_FAILED;
}

static FakeBuffer *find_buffer(VkBuffer handle) {
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++)
        if (g_fake.buffers[i].live && g_fake.buffers[i].handle == handle)
            return &g_fake.buffers[i];
    return NULL;
}

static FakeMemory *find_memory(VkDeviceMemory handle) {
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++)
        if (g_fake.memories[i].live && g_fake.memories[i].handle == handle)
            return &g_fake.memories[i];
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateInstance(
    const VkInstanceCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkInstance *instance
) {
    (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_INSTANCE);
    if (result != VK_SUCCESS) return result;
    *instance = HANDLE(VkInstance, next_handle());
    g_fake.create_instance_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks *callbacks
) {
    (void)instance; (void)callbacks;
    g_fake.destroy_instance_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_EnumeratePhysicalDevices(
    VkInstance instance,
    uint32_t *count,
    VkPhysicalDevice *devices
) {
    (void)instance;
    VkResult result = maybe_fault(FP_ENUMERATE);
    if (result != VK_SUCCESS) return result;
    if (!devices) {
        *count = g_fake.physical_device_count;
        return VK_SUCCESS;
    }
    uint32_t write = *count < g_fake.physical_device_count
        ? *count : g_fake.physical_device_count;
    for (uint32_t i = 0; i < write; i++)
        devices[i] = HANDLE(VkPhysicalDevice, 2000u + i);
    *count = write;
    return write < g_fake.physical_device_count ? VK_INCOMPLETE : VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_GetPhysicalDeviceProperties(
    VkPhysicalDevice device,
    VkPhysicalDeviceProperties *properties
) {
    memset(properties, 0, sizeof(*properties));
    uint32_t index = (uint32_t)((uintptr_t)device - 2000u);
    if (index < g_fake.matching_device_count) {
        properties->vendorID = COLI_VULKAN_VENDOR_ID;
        properties->deviceID = COLI_VULKAN_DEVICE_ID;
    } else {
        properties->vendorID = 0x10de;
        properties->deviceID = 0x1234 + index;
    }
    properties->limits.maxMemoryAllocationCount = g_fake.max_allocations;
    properties->limits.minStorageBufferOffsetAlignment = g_fake.storage_alignment;
    properties->limits.maxStorageBufferRange = UINT32_MAX;
    properties->limits.maxPushConstantsSize = 128;
    properties->limits.maxComputeWorkGroupInvocations = 256;
    properties->limits.maxComputeWorkGroupSize[0] = 256;
    properties->limits.maxComputeWorkGroupSize[1] = 256;
    properties->limits.maxComputeWorkGroupSize[2] = 64;
    properties->limits.maxComputeWorkGroupCount[0] = 65535;
    properties->limits.maxComputeWorkGroupCount[1] = 65535;
    properties->limits.maxComputeWorkGroupCount[2] = 65535;
    properties->limits.maxPerStageDescriptorStorageBuffers = 8;
    properties->limits.maxDescriptorSetStorageBuffers = 8;
}

static void fill_memory_properties(VkPhysicalDeviceMemoryProperties *properties) {
    memset(properties, 0, sizeof(*properties));
    properties->memoryHeapCount = 3;
    properties->memoryHeaps[0].size = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    properties->memoryHeaps[1].size = 16ULL * 1024ULL * 1024ULL * 1024ULL;
    properties->memoryHeaps[1].flags = 0;
    properties->memoryHeaps[2].size = 268435456ULL;
    properties->memoryHeaps[2].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    properties->memoryTypeCount = 7;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[1].heapIndex = 1;
    properties->memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    properties->memoryTypes[2].heapIndex = 1;
    properties->memoryTypes[2].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    properties->memoryTypes[3].heapIndex = 2;
    properties->memoryTypes[3].propertyFlags = 0x0007;
    properties->memoryTypes[4].heapIndex = 2;
    properties->memoryTypes[4].propertyFlags = 0x0007;
    properties->memoryTypes[5].heapIndex = 0;
    properties->memoryTypes[5].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[6].heapIndex = 1;
    properties->memoryTypes[6].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

static VKAPI_ATTR void VKAPI_CALL fake_GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice device,
    VkPhysicalDeviceMemoryProperties *properties
) {
    (void)device;
    fill_memory_properties(properties);
}

static VKAPI_ATTR void VKAPI_CALL fake_GetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice device,
    uint32_t *count,
    VkQueueFamilyProperties *properties
) {
    (void)device;
    if (!properties) {
        *count = 2;
        return;
    }
    memset(properties, 0, sizeof(*properties) * (*count < 2 ? *count : 2));
    if (*count > 0) {
        properties[0].queueCount = 1;
        properties[0].queueFlags = VK_QUEUE_TRANSFER_BIT;
    }
    if (*count > 1) {
        properties[1].queueCount = 1;
        properties[1].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    }
    if (*count > 2) *count = 2;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateDevice(
    VkPhysicalDevice physical_device,
    const VkDeviceCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkDevice *device
) {
    (void)physical_device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_DEVICE);
    if (result != VK_SUCCESS) return result;
    *device = HANDLE(VkDevice, next_handle());
    g_fake.create_device_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)callbacks;
    g_fake.destroy_device_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_GetDeviceQueue(
    VkDevice device,
    uint32_t family,
    uint32_t index,
    VkQueue *queue
) {
    (void)device; (void)family; (void)index;
    *queue = HANDLE(VkQueue, next_handle());
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateCommandPool(
    VkDevice device,
    const VkCommandPoolCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkCommandPool *pool
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_POOL);
    if (result != VK_SUCCESS) return result;
    *pool = HANDLE(VkCommandPool, next_handle());
    g_fake.create_pool_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyCommandPool(
    VkDevice device,
    VkCommandPool pool,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)pool; (void)callbacks;
    g_fake.destroy_pool_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateBuffer(
    VkDevice device,
    const VkBufferCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkBuffer *buffer
) {
    (void)device; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_BUFFER);
    if (result != VK_SUCCESS) return result;
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++) if (!g_fake.buffers[i].live) {
        g_fake.buffers[i].live = 1;
        g_fake.buffers[i].handle = HANDLE(VkBuffer, next_handle());
        g_fake.buffers[i].size = info->size;
        g_fake.buffers[i].usage = info->usage;
        *buffer = g_fake.buffers[i].handle;
        g_fake.create_buffer_count++;
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_HOST_MEMORY;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyBuffer(
    VkDevice device,
    VkBuffer buffer,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)callbacks;
    FakeBuffer *record = find_buffer(buffer);
    if (record) record->live = 0;
    g_fake.destroy_buffer_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_GetBufferMemoryRequirements(
    VkDevice device,
    VkBuffer buffer,
    VkMemoryRequirements *requirements
) {
    (void)device;
    FakeBuffer *record = find_buffer(buffer);
    memset(requirements, 0, sizeof(*requirements));
    if (!record) return;
    requirements->alignment = 4096;
    requirements->size = (record->size + 4095) & ~(VkDeviceSize)4095;
    requirements->memoryTypeBits = 0x7f;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_AllocateMemory(
    VkDevice device,
    const VkMemoryAllocateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkDeviceMemory *memory
) {
    (void)device; (void)callbacks;
    VkResult result = maybe_fault(FP_ALLOCATE_MEMORY);
    if (result != VK_SUCCESS) return result;
    for (unsigned i = 0; i < MAX_FAKE_OBJECTS; i++) if (!g_fake.memories[i].live) {
        if (info->allocationSize > SIZE_MAX) return VK_ERROR_OUT_OF_HOST_MEMORY;
        uint8_t *data = calloc(1, (size_t)info->allocationSize);
        if (!data) return VK_ERROR_OUT_OF_HOST_MEMORY;
        g_fake.memories[i].live = 1;
        g_fake.memories[i].handle = HANDLE(VkDeviceMemory, next_handle());
        g_fake.memories[i].size = info->allocationSize;
        g_fake.memories[i].data = data;
        *memory = g_fake.memories[i].handle;
        g_fake.allocate_memory_count++;
        return VK_SUCCESS;
    }
    return VK_ERROR_OUT_OF_HOST_MEMORY;
}

static VKAPI_ATTR void VKAPI_CALL fake_FreeMemory(
    VkDevice device,
    VkDeviceMemory memory,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)callbacks;
    FakeMemory *record = find_memory(memory);
    if (record) {
        free(record->data);
        memset(record, 0, sizeof(*record));
    }
    g_fake.free_memory_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_BindBufferMemory(
    VkDevice device,
    VkBuffer buffer,
    VkDeviceMemory memory,
    VkDeviceSize offset
) {
    (void)device; (void)offset;
    VkResult result = maybe_fault(FP_BIND_MEMORY);
    if (result != VK_SUCCESS) return result;
    FakeBuffer *record = find_buffer(buffer);
    if (!record || !find_memory(memory)) return VK_ERROR_MEMORY_MAP_FAILED;
    record->memory = memory;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_MapMemory(
    VkDevice device,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkMemoryMapFlags flags,
    void **data
) {
    (void)device; (void)flags;
    VkResult result = maybe_fault(FP_MAP_MEMORY);
    if (result != VK_SUCCESS) return result;
    FakeMemory *record = find_memory(memory);
    if (!record || offset > record->size || size > record->size - offset)
        return VK_ERROR_MEMORY_MAP_FAILED;
    *data = record->data + (size_t)offset;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_UnmapMemory(
    VkDevice device,
    VkDeviceMemory memory
) {
    (void)device; (void)memory;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_AllocateCommandBuffers(
    VkDevice device,
    const VkCommandBufferAllocateInfo *info,
    VkCommandBuffer *commands
) {
    (void)device;
    VkResult result = maybe_fault(FP_ALLOCATE_COMMAND);
    if (result != VK_SUCCESS) return result;
    for (uint32_t i = 0; i < info->commandBufferCount; i++)
        commands[i] = HANDLE(VkCommandBuffer, next_handle());
    g_fake.allocate_command_buffer_count += info->commandBufferCount;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_FreeCommandBuffers(
    VkDevice device,
    VkCommandPool pool,
    uint32_t count,
    const VkCommandBuffer *commands
) {
    (void)device; (void)pool; (void)commands;
    g_fake.free_command_buffer_count += count;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_BeginCommandBuffer(
    VkCommandBuffer command,
    const VkCommandBufferBeginInfo *info
) {
    (void)command; (void)info;
    return maybe_fault(FP_BEGIN_COMMAND);
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_EndCommandBuffer(VkCommandBuffer command) {
    (void)command;
    return maybe_fault(FP_END_COMMAND);
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdCopyBuffer(
    VkCommandBuffer command,
    VkBuffer source,
    VkBuffer destination,
    uint32_t region_count,
    const VkBufferCopy *regions
) {
    (void)command;
    FakeBuffer *source_buffer = find_buffer(source);
    FakeBuffer *destination_buffer = find_buffer(destination);
    FakeMemory *source_memory = source_buffer ? find_memory(source_buffer->memory) : NULL;
    FakeMemory *destination_memory = destination_buffer
        ? find_memory(destination_buffer->memory) : NULL;
    if (!source_memory || !destination_memory) return;
    for (uint32_t i = 0; i < region_count; i++) {
        if (regions[i].srcOffset <= source_memory->size &&
            regions[i].size <= source_memory->size - regions[i].srcOffset &&
            regions[i].dstOffset <= destination_memory->size &&
            regions[i].size <= destination_memory->size - regions[i].dstOffset)
            memcpy(destination_memory->data + (size_t)regions[i].dstOffset,
                source_memory->data + (size_t)regions[i].srcOffset,
                (size_t)regions[i].size);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateFence(
    VkDevice device,
    const VkFenceCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkFence *fence
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_FENCE);
    if (result != VK_SUCCESS) return result;
    *fence = HANDLE(VkFence, next_handle());
    g_fake.create_fence_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyFence(
    VkDevice device,
    VkFence fence,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)fence; (void)callbacks;
    g_fake.destroy_fence_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_QueueSubmit(
    VkQueue queue,
    uint32_t submit_count,
    const VkSubmitInfo *submits,
    VkFence fence
) {
    (void)queue; (void)submit_count; (void)submits; (void)fence;
    g_fake.queue_submit_count++;
    return maybe_fault(FP_QUEUE_SUBMIT);
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_WaitForFences(
    VkDevice device,
    uint32_t fence_count,
    const VkFence *fences,
    VkBool32 wait_all,
    uint64_t timeout
) {
    (void)device; (void)fence_count; (void)fences; (void)wait_all; (void)timeout;
    VkResult fault = maybe_fault(FP_WAIT_FENCE);
    if (fault != VK_SUCCESS) return fault;
    if (g_fake.wait_result_index < g_fake.wait_result_count)
        return g_fake.wait_results[g_fake.wait_result_index++];
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdPipelineBarrier(
    VkCommandBuffer command,
    VkPipelineStageFlags source_stage,
    VkPipelineStageFlags destination_stage,
    VkDependencyFlags dependency_flags,
    uint32_t memory_barrier_count,
    const VkMemoryBarrier *memory_barriers,
    uint32_t buffer_barrier_count,
    const VkBufferMemoryBarrier *buffer_barriers,
    uint32_t image_barrier_count,
    const VkImageMemoryBarrier *image_barriers
) {
    (void)command; (void)dependency_flags; (void)memory_barrier_count;
    (void)memory_barriers; (void)image_barrier_count; (void)image_barriers;
    g_fake.pipeline_barrier_count++;
    if (buffer_barrier_count == 1 && buffer_barriers &&
        source_stage == VK_PIPELINE_STAGE_HOST_BIT &&
        destination_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT &&
        buffer_barriers[0].srcAccessMask == VK_ACCESS_HOST_WRITE_BIT &&
        buffer_barriers[0].dstAccessMask == VK_ACCESS_SHADER_READ_BIT)
        g_fake.host_to_compute_barrier_count++;
    if (buffer_barrier_count == 1 && buffer_barriers &&
        source_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT &&
        destination_stage == VK_PIPELINE_STAGE_HOST_BIT &&
        buffer_barriers[0].srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT &&
        buffer_barriers[0].dstAccessMask == VK_ACCESS_HOST_READ_BIT)
        g_fake.compute_to_host_barrier_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdBindPipeline(
    VkCommandBuffer command,
    VkPipelineBindPoint bind_point,
    VkPipeline pipeline
) {
    (void)command; (void)bind_point; (void)pipeline;
    g_fake.bind_pipeline_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdBindDescriptorSets(
    VkCommandBuffer command,
    VkPipelineBindPoint bind_point,
    VkPipelineLayout layout,
    uint32_t first_set,
    uint32_t set_count,
    const VkDescriptorSet *sets,
    uint32_t dynamic_offset_count,
    const uint32_t *dynamic_offsets
) {
    (void)command; (void)bind_point; (void)layout; (void)first_set;
    (void)set_count; (void)sets; (void)dynamic_offset_count;
    (void)dynamic_offsets;
    g_fake.bind_descriptor_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdPushConstants(
    VkCommandBuffer command,
    VkPipelineLayout layout,
    VkShaderStageFlags stage_flags,
    uint32_t offset,
    uint32_t size,
    const void *values
) {
    (void)command; (void)layout; (void)stage_flags;
    if (offset == 0 && size == sizeof(g_fake.push_constants) && values)
        memcpy(g_fake.push_constants, values, size);
    g_fake.push_constant_count++;
}

static VKAPI_ATTR void VKAPI_CALL fake_CmdDispatch(
    VkCommandBuffer command,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z
) {
    (void)command; (void)group_count_x; (void)group_count_y;
    (void)group_count_z;
    g_fake.dispatch_count++;
    FakeBuffer *packed_buffer = find_buffer(g_fake.descriptor_buffers[0]);
    FakeBuffer *input_buffer = find_buffer(g_fake.descriptor_buffers[1]);
    FakeBuffer *output_buffer = find_buffer(g_fake.descriptor_buffers[2]);
    FakeMemory *packed_memory = packed_buffer
        ? find_memory(packed_buffer->memory) : NULL;
    FakeMemory *input_memory = input_buffer
        ? find_memory(input_buffer->memory) : NULL;
    FakeMemory *output_memory = output_buffer
        ? find_memory(output_buffer->memory) : NULL;
    if (!packed_memory || !input_memory || !output_memory) return;
    uint32_t I = g_fake.push_constants[0];
    uint32_t O = g_fake.push_constants[1];
    uint32_t rows = g_fake.push_constants[2];
    uint32_t row_words = g_fake.push_constants[3];
    uint32_t scale_words = g_fake.push_constants[4];
    uint32_t total = g_fake.push_constants[5];
    if ((uint64_t)total != (uint64_t)rows * O) return;
    const uint32_t *packed = (const uint32_t *)packed_memory->data;
    const float *input = (const float *)input_memory->data;
    float *output = (float *)output_memory->data;
    for (uint32_t index = 0; index < total; index++) {
        uint32_t row = index / O, column = index % O;
        float sum = 0;
        for (uint32_t word = 0; word < row_words; word++) {
            uint32_t bits = packed[column * row_words + word];
            for (uint32_t nibble = 0; nibble < 8; nibble++)
                sum += input[(uint64_t)row * I + word * 8 + nibble] *
                    (float)((int)((bits >> (nibble * 4)) & 15u) - 8);
        }
        float scale;
        memcpy(&scale, &packed[scale_words + column], sizeof(scale));
        output[index] = sum * scale;
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkShaderModule *module
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_SHADER);
    if (result != VK_SUCCESS) return result;
    *module = HANDLE(VkShaderModule, next_handle());
    g_fake.create_shader_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyShaderModule(
    VkDevice device,
    VkShaderModule module,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)module; (void)callbacks;
    g_fake.destroy_shader_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkDescriptorSetLayout *layout
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_DESCRIPTOR_LAYOUT);
    if (result != VK_SUCCESS) return result;
    *layout = HANDLE(VkDescriptorSetLayout, next_handle());
    g_fake.create_descriptor_layout_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyDescriptorSetLayout(
    VkDevice device,
    VkDescriptorSetLayout layout,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)layout; (void)callbacks;
    g_fake.destroy_descriptor_layout_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreatePipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkPipelineLayout *layout
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_PIPELINE_LAYOUT);
    if (result != VK_SUCCESS) return result;
    *layout = HANDLE(VkPipelineLayout, next_handle());
    g_fake.create_pipeline_layout_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyPipelineLayout(
    VkDevice device,
    VkPipelineLayout layout,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)layout; (void)callbacks;
    g_fake.destroy_pipeline_layout_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateComputePipelines(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t count,
    const VkComputePipelineCreateInfo *infos,
    const VkAllocationCallbacks *callbacks,
    VkPipeline *pipelines
) {
    (void)device; (void)cache; (void)infos; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_PIPELINE);
    if (result != VK_SUCCESS) return result;
    for (uint32_t i = 0; i < count; i++)
        pipelines[i] = HANDLE(VkPipeline, next_handle());
    g_fake.create_pipeline_count += count;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyPipeline(
    VkDevice device,
    VkPipeline pipeline,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)pipeline; (void)callbacks;
    g_fake.destroy_pipeline_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_CreateDescriptorPool(
    VkDevice device,
    const VkDescriptorPoolCreateInfo *info,
    const VkAllocationCallbacks *callbacks,
    VkDescriptorPool *pool
) {
    (void)device; (void)info; (void)callbacks;
    VkResult result = maybe_fault(FP_CREATE_DESCRIPTOR_POOL);
    if (result != VK_SUCCESS) return result;
    *pool = HANDLE(VkDescriptorPool, next_handle());
    g_fake.create_descriptor_pool_count++;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_DestroyDescriptorPool(
    VkDevice device,
    VkDescriptorPool pool,
    const VkAllocationCallbacks *callbacks
) {
    (void)device; (void)pool; (void)callbacks;
    g_fake.destroy_descriptor_pool_count++;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_AllocateDescriptorSets(
    VkDevice device,
    const VkDescriptorSetAllocateInfo *info,
    VkDescriptorSet *sets
) {
    (void)device;
    VkResult result = maybe_fault(FP_ALLOCATE_DESCRIPTOR_SET);
    if (result != VK_SUCCESS) return result;
    for (uint32_t i = 0; i < info->descriptorSetCount; i++)
        sets[i] = HANDLE(VkDescriptorSet, next_handle());
    g_fake.allocate_descriptor_set_count += info->descriptorSetCount;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_UpdateDescriptorSets(
    VkDevice device,
    uint32_t write_count,
    const VkWriteDescriptorSet *writes,
    uint32_t copy_count,
    const VkCopyDescriptorSet *copies
) {
    (void)device; (void)copy_count; (void)copies;
    unsigned active = __sync_add_and_fetch(&g_fake.api_active, 1);
    unsigned old_max = g_fake.api_max_active;
    while (active > old_max && !__sync_bool_compare_and_swap(
        &g_fake.api_max_active, old_max, active))
        old_max = g_fake.api_max_active;
    sched_yield();
    for (uint32_t i = 0; i < write_count; i++) {
        if (writes[i].dstBinding < 3 && writes[i].pBufferInfo) {
            uint32_t binding = writes[i].dstBinding;
            g_fake.descriptor_buffers[binding] = writes[i].pBufferInfo->buffer;
            g_fake.descriptor_ranges[binding] = writes[i].pBufferInfo->range;
        }
    }
    g_fake.update_descriptor_count++;
    __sync_sub_and_fetch(&g_fake.api_active, 1);
}

static const ColiVulkanApi g_fake_api = {
    .CreateInstance = fake_CreateInstance,
    .DestroyInstance = fake_DestroyInstance,
    .EnumeratePhysicalDevices = fake_EnumeratePhysicalDevices,
    .GetPhysicalDeviceProperties = fake_GetPhysicalDeviceProperties,
    .GetPhysicalDeviceMemoryProperties = fake_GetPhysicalDeviceMemoryProperties,
    .GetPhysicalDeviceQueueFamilyProperties = fake_GetPhysicalDeviceQueueFamilyProperties,
    .CreateDevice = fake_CreateDevice,
    .DestroyDevice = fake_DestroyDevice,
    .GetDeviceQueue = fake_GetDeviceQueue,
    .CreateCommandPool = fake_CreateCommandPool,
    .DestroyCommandPool = fake_DestroyCommandPool,
    .CreateBuffer = fake_CreateBuffer,
    .DestroyBuffer = fake_DestroyBuffer,
    .GetBufferMemoryRequirements = fake_GetBufferMemoryRequirements,
    .AllocateMemory = fake_AllocateMemory,
    .FreeMemory = fake_FreeMemory,
    .BindBufferMemory = fake_BindBufferMemory,
    .MapMemory = fake_MapMemory,
    .UnmapMemory = fake_UnmapMemory,
    .AllocateCommandBuffers = fake_AllocateCommandBuffers,
    .FreeCommandBuffers = fake_FreeCommandBuffers,
    .BeginCommandBuffer = fake_BeginCommandBuffer,
    .EndCommandBuffer = fake_EndCommandBuffer,
    .CmdCopyBuffer = fake_CmdCopyBuffer,
    .CmdPipelineBarrier = fake_CmdPipelineBarrier,
    .CmdBindPipeline = fake_CmdBindPipeline,
    .CmdBindDescriptorSets = fake_CmdBindDescriptorSets,
    .CmdPushConstants = fake_CmdPushConstants,
    .CmdDispatch = fake_CmdDispatch,
    .CreateFence = fake_CreateFence,
    .DestroyFence = fake_DestroyFence,
    .QueueSubmit = fake_QueueSubmit,
    .WaitForFences = fake_WaitForFences,
    .CreateShaderModule = fake_CreateShaderModule,
    .DestroyShaderModule = fake_DestroyShaderModule,
    .CreateDescriptorSetLayout = fake_CreateDescriptorSetLayout,
    .DestroyDescriptorSetLayout = fake_DestroyDescriptorSetLayout,
    .CreatePipelineLayout = fake_CreatePipelineLayout,
    .DestroyPipelineLayout = fake_DestroyPipelineLayout,
    .CreateComputePipelines = fake_CreateComputePipelines,
    .DestroyPipeline = fake_DestroyPipeline,
    .CreateDescriptorPool = fake_CreateDescriptorPool,
    .DestroyDescriptorPool = fake_DestroyDescriptorPool,
    .AllocateDescriptorSets = fake_AllocateDescriptorSets,
    .UpdateDescriptorSets = fake_UpdateDescriptorSets
};

static ColiVulkanConfig fake_config(uint64_t budget) {
    ColiVulkanConfig config;
    memset(&config, 0, sizeof(config));
    config.expert_budget_bytes = budget;
    config.upload_timeout_ns = TEST_TIMEOUT_NS;
    config.test_api = &g_fake_api;
    return config;
}

static void test_config_and_layout(void) {
    ColiVulkanConfig config;
    unsetenv("VULKAN_EXPERT_MB");
    CHECK_RESULT(coli_vulkan_config_from_env(&config, TEST_TIMEOUT_NS),
        COLI_VULKAN_INVALID_ARGUMENT, "missing budget");
    setenv("VULKAN_EXPERT_MB", "0", 1);
    CHECK_RESULT(coli_vulkan_config_from_env(&config, TEST_TIMEOUT_NS),
        COLI_VULKAN_INVALID_ARGUMENT, "zero budget");
    setenv("VULKAN_EXPERT_MB", "64x", 1);
    CHECK_RESULT(coli_vulkan_config_from_env(&config, TEST_TIMEOUT_NS),
        COLI_VULKAN_INVALID_ARGUMENT, "trailing budget text");
    setenv("VULKAN_EXPERT_MB", "64", 1);
    CHECK_RESULT(coli_vulkan_config_from_env(&config, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "explicit budget");
    CHECK(config.expert_budget_bytes == 64ULL * 1024ULL * 1024ULL,
        "64 MiB conversion");

    ColiVulkanTensorLayout layout;
    CHECK_RESULT(coli_vulkan_plan_tensor_layout(4093, 513, 256, &layout),
        COLI_VULKAN_OK, "aligned layout");
    CHECK(layout.weight_offset == 0 && layout.scale_offset == 4096 &&
        layout.packed_size == 4864, "layout offsets");
    CHECK_RESULT(coli_vulkan_plan_tensor_layout(4093, 0, 256, &layout),
        COLI_VULKAN_OK, "weight-only layout");
    CHECK(layout.scale_offset == 0 && layout.packed_size == 4096,
        "weight-only padding");
    CHECK_RESULT(coli_vulkan_plan_tensor_layout(1, 1, 0, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "zero alignment");
    CHECK_RESULT(coli_vulkan_plan_tensor_layout(UINT64_MAX - 3, 8, 256, &layout),
        COLI_VULKAN_LIMIT_EXCEEDED, "layout overflow");
}

static void check_valid_qt(
    uint32_t fmt,
    uint64_t I,
    uint64_t O,
    uint64_t gs,
    uint64_t weight_bytes,
    uint64_t scale_bytes,
    uint64_t scale_count,
    uint64_t effective_group_size,
    uint64_t uploaded_scale_bytes
) {
    uint8_t weight = 0, scale = 0;
    ColiVulkanQTSpec spec = {
        fmt, I, O, gs, &weight, weight_bytes,
        scale_bytes ? &scale : NULL, scale_bytes
    };
    ColiVulkanQTLayout layout;
    ColiVulkanResult result = coli_vulkan_validate_qt_spec(&spec, 256, &layout);
    if (result != COLI_VULKAN_OK || layout.scale_count != scale_count ||
        layout.effective_group_size != effective_group_size ||
        layout.uploaded_scale_bytes != uploaded_scale_bytes ||
        layout.packed.weight_size != weight_bytes ||
        layout.packed.scale_size != uploaded_scale_bytes ||
        layout.packed.weight_offset != 0 ||
        (uploaded_scale_bytes && layout.packed.scale_offset % 256 != 0) ||
        layout.packed.packed_size % 256 != 0) {
        fprintf(stderr, "FAIL: %s:%d: valid fmt=%u I=%" PRIu64
            " O=%" PRIu64 " gs=%" PRIu64 " layout mismatch (%s)\n",
            __func__, __LINE__, fmt, I, O, gs,
            coli_vulkan_result_string(result));
        g_failures++;
    }
}

static void test_qt_format_validation(void) {
    check_valid_qt(0, 17, 3, 0, 204, 0, 0, 0, 0);
    check_valid_qt(0, 16, 2, 0, 128, 0, 0, 0, 0);
    check_valid_qt(1, 17, 3, 0, 51, 12, 3, 0, 12);
    check_valid_qt(1, 16, 2, 0, 32, 8, 2, 0, 8);
    check_valid_qt(2, 17, 3, 0, 27, 12, 3, 0, 12);
    check_valid_qt(2, 16, 2, 0, 16, 8, 2, 0, 8);
    check_valid_qt(3, 17, 3, 0, 15, 12, 3, 0, 12);
    check_valid_qt(3, 16, 2, 0, 8, 8, 2, 0, 8);
    check_valid_qt(4, 33, 3, 16, 51, 36, 9, 16, 36);
    check_valid_qt(4, 64, 2, 32, 64, 16, 4, 32, 16);
    check_valid_qt(5, 65, 3, 0, 144, 24, 6, 64, 24);
    check_valid_qt(5, 64, 2, 0, 48, 8, 2, 64, 8);
    check_valid_qt(6, 257, 3, 0, 588, 4, 0, 256, 0);
    check_valid_qt(6, 256, 2, 0, 196, 4, 0, 256, 0);

    static const uint64_t groups[] = {16, 32, 48, 64, 96, 128, 192, 256};
    uint8_t weight = 0, scale = 0;
    for (size_t i = 0; i < sizeof(groups) / sizeof(groups[0]); i++) {
        uint64_t gs = groups[i], I = 257, O = 2;
        uint64_t wb = O * ((I + 1) / 2);
        uint64_t sc = O * (I / gs + (I % gs != 0));
        ColiVulkanQTSpec spec = {4, I, O, gs, &weight, wb, &scale, sc * 4};
        ColiVulkanQTLayout layout;
        CHECK_RESULT(coli_vulkan_validate_qt_spec(&spec, 256, &layout),
            COLI_VULKAN_OK, "supported grouped-int4 size");
        CHECK(layout.scale_count == sc && layout.effective_group_size == gs,
            "grouped-int4 derived metadata");
    }

    ColiVulkanQTLayout layout;
    ColiVulkanQTSpec bad = {7, 16, 2, 0, &weight, 16, &scale, 8};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_UNSUPPORTED, "unknown format");
    bad = (ColiVulkanQTSpec){4, 33, 3, 0, &weight, 51, &scale, 36};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_UNSUPPORTED, "grouped format requires group size");
    bad.gs = 24;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_UNSUPPORTED, "unsupported group size");
    bad.gs = 48;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_UNSUPPORTED, "group size over I");
    bad = (ColiVulkanQTSpec){2, 17, 3, 1, &weight, 27, &scale, 12};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_UNSUPPORTED, "row format rejects gs");
    bad.gs = 0;
    bad.weight_bytes--;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "short weight source");
    bad.weight_bytes += 2;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "long weight source");
    bad.weight_bytes = 27;
    bad.scale_bytes--;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "short scale source");
    bad.scale_bytes += 2;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "long scale source");
    bad = (ColiVulkanQTSpec){6, 256, 2, 0, &weight, 196, &scale, 3};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "format-6 tag too short");
    bad.scale_bytes = 5;
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "format-6 tag too long");
    bad = (ColiVulkanQTSpec){0, 1, 1, 0, &weight, 4, &scale, 0};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "scale pointer without scale source");
    bad = (ColiVulkanQTSpec){0, UINT64_MAX, 2, 0, &weight, 1, NULL, 0};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 256, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "geometry exceeds CPU int range");
    bad = (ColiVulkanQTSpec){0, 1, 1, 0, &weight, 4, NULL, 0};
    CHECK_RESULT(coli_vulkan_validate_qt_spec(&bad, 0, &layout),
        COLI_VULKAN_INVALID_ARGUMENT, "zero storage alignment");

    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config), COLI_VULKAN_OK,
        "validation counter context");
    unsigned buffers = g_fake.create_buffer_count;
    unsigned allocations = g_fake.allocate_memory_count;
    unsigned commands = g_fake.allocate_command_buffer_count;
    unsigned fences = g_fake.create_fence_count;
    unsigned submits = g_fake.queue_submit_count;
    bad = (ColiVulkanQTSpec){2, 17, 3, 0, &weight, 26, &scale, 12};
    CHECK_RESULT(coli_vulkan_tensor_create_qt(context, &tensor, &bad,
        TEST_TIMEOUT_NS), COLI_VULKAN_INVALID_ARGUMENT,
        "creator rejects before Vulkan work");
    CHECK(!tensor && buffers == g_fake.create_buffer_count &&
        allocations == g_fake.allocate_memory_count &&
        commands == g_fake.allocate_command_buffer_count &&
        fences == g_fake.create_fence_count && submits == g_fake.queue_submit_count,
        "validator rejection allocated or submitted");

    uint8_t weights[51] = {0};
    float scales[9] = {0};
    ColiVulkanQTSpec good = {4, 33, 3, 16, weights, sizeof(weights),
        scales, sizeof(scales)};
    CHECK_RESULT(coli_vulkan_tensor_create_qt(context, &tensor, &good,
        TEST_TIMEOUT_NS), COLI_VULKAN_OK, "validated creator");
    ColiVulkanTensorInfo info;
    CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &info), COLI_VULKAN_OK,
        "validated metadata info");
    CHECK(info.fmt == 4 && info.I == 33 && info.O == 3 && info.gs == 16 &&
        info.scale_count == 9 && info.effective_group_size == 16 &&
        info.source_weight_bytes == sizeof(weights) &&
        info.source_scale_bytes == sizeof(scales) &&
        info.uploaded_scale_bytes == sizeof(scales) && !info.compute_eligible,
        "persisted validated metadata");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "validated tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "validated context destroy");
}

static void test_memory_selector(void) {
    VkPhysicalDeviceMemoryProperties properties;
    fill_memory_properties(&properties);
    uint32_t type = UINT32_MAX, heap = UINT32_MAX;
    uint64_t usable = 0;
    CHECK_RESULT(coli_vulkan_select_strict_memory_type(&properties, 0x7f, 4096,
        COLI_VULKAN_VRAM_RESERVE_BYTES, &type, &heap, &usable),
        COLI_VULKAN_OK, "strict selector");
    CHECK(type == 0 && heap == 0, "strict type and heap");
    CHECK(usable == 7ULL * 1024ULL * 1024ULL * 1024ULL,
        "fixed 1 GiB reserve");
    CHECK_RESULT(coli_vulkan_select_strict_memory_type(&properties,
        (1u << 3) | (1u << 4), 4096, COLI_VULKAN_VRAM_RESERVE_BYTES,
        &type, &heap, &usable), COLI_VULKAN_LIMIT_EXCEEDED,
        "BAR types rejected");
    properties.memoryTypes[0].propertyFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    properties.memoryTypes[5].propertyFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    CHECK_RESULT(coli_vulkan_select_strict_memory_type(&properties, 0x7f, 4096,
        COLI_VULKAN_VRAM_RESERVE_BYTES, &type, &heap, &usable),
        COLI_VULKAN_UNSUPPORTED, "no strict type");
}

static void test_device_selection_and_partial_init(void) {
    ColiVulkanContext *context = NULL;
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);

    fake_reset();
    g_fake.physical_device_count = 0;
    g_fake.matching_device_count = 0;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_UNAVAILABLE, "no device");
    CHECK(!context && g_fake.destroy_instance_count == 1, "no-device cleanup");

    fake_reset();
    g_fake.physical_device_count = 2;
    g_fake.matching_device_count = 2;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_UNSUPPORTED, "multiple exact devices");
    CHECK(!context && g_fake.destroy_instance_count == 1, "multi-device cleanup");

    const FaultPoint init_points[] = {
        FP_CREATE_INSTANCE, FP_ENUMERATE, FP_CREATE_DEVICE, FP_CREATE_POOL
    };
    for (unsigned i = 0; i < sizeof(init_points) / sizeof(init_points[0]); i++) {
        fake_reset();
        set_fault(init_points[i], 1, VK_ERROR_INITIALIZATION_FAILED);
        CHECK_RESULT(coli_vulkan_context_create(&context, &config),
            COLI_VULKAN_ERROR, "injected init failure");
        CHECK(!context, "failed init published context");
        if (init_points[i] == FP_CREATE_POOL)
            CHECK(g_fake.destroy_device_count == 1 &&
                g_fake.destroy_instance_count == 1, "pool failure cleanup");
    }
}

static void test_success_lifecycle(void) {
    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL, *second = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "create");
    CHECK_RESULT(coli_vulkan_context_create(&second, &config),
        COLI_VULKAN_BUSY, "singleton context");
    ColiVulkanContextInfo context_info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &context_info),
        COLI_VULKAN_OK, "context info");
    CHECK(context_info.vendor_id == COLI_VULKAN_VENDOR_ID &&
        context_info.device_id == COLI_VULKAN_DEVICE_ID &&
        context_info.queue_family_index == 1, "device and compute queue");
    CHECK(context_info.max_memory_allocation_count == 4096 &&
        context_info.min_storage_buffer_offset_alignment == 256,
        "queried limits");

    uint8_t weights[37], scales[11];
    for (size_t i = 0; i < sizeof(weights); i++) weights[i] = (uint8_t)(i * 7);
    for (size_t i = 0; i < sizeof(scales); i++) scales[i] = (uint8_t)(i * 19);
    ColiVulkanTensor *tensor = NULL;
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, weights,
        sizeof(weights), scales, sizeof(scales), TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "upload");
    CHECK(tensor != NULL, "upload handle");
    ColiVulkanTensorInfo tensor_info;
    CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &tensor_info),
        COLI_VULKAN_OK, "tensor info");
    CHECK(tensor_info.layout.weight_offset == 0 &&
        tensor_info.layout.scale_offset == 256 &&
        tensor_info.layout.packed_size == 512, "tensor alignment");
    CHECK((tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        !(tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
        "strict device memory");

    uint8_t readback[512];
    memset(readback, 0, sizeof(readback));
    CHECK_RESULT(coli_vulkan_tensor_readback(context, tensor, readback,
        sizeof(readback), TEST_TIMEOUT_NS), COLI_VULKAN_OK, "readback");
    CHECK(!memcmp(readback, weights, sizeof(weights)), "weight readback");
    CHECK(!memcmp(readback + 256, scales, sizeof(scales)), "scale readback");

    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK, "free");
    CHECK(!tensor, "free clears owner");
    CHECK_RESULT(coli_vulkan_context_get_info(context, &context_info),
        COLI_VULKAN_OK, "post-free info");
    CHECK(context_info.committed_bytes == 0 && context_info.live_tensors == 0 &&
        context_info.live_allocations == 0, "zero accounting");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "destroy");
    CHECK(!context, "destroy clears owner");
    CHECK(g_fake.create_instance_count == 1 && g_fake.destroy_instance_count == 1 &&
        g_fake.create_device_count == 1 && g_fake.destroy_device_count == 1 &&
        g_fake.create_pool_count == 1 && g_fake.destroy_pool_count == 1,
        "persistent object lifetime");
}

static void test_allocation_limit_and_budget(void) {
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL;
    uint8_t data[64] = {0};

    fake_reset();
    g_fake.max_allocations = 1;
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "max-allocation context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_LIMIT_EXCEEDED,
        "maxMemoryAllocationCount enforced");
    CHECK(!tensor && g_fake.allocate_memory_count == 1 &&
        g_fake.free_memory_count == 1, "allocation-limit rollback");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "allocation-limit destroy");

    fake_reset();
    config = fake_config(1024);
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "small-budget context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_LIMIT_EXCEEDED,
        "actual allocation exceeds budget");
    CHECK(!tensor, "budget failure published tensor");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "small-budget destroy");
}

static void test_pre_submit_fault_cleanup(void) {
    typedef struct { FaultPoint point; unsigned call; VkResult result; } Case;
    const Case cases[] = {
        {FP_CREATE_BUFFER, 1, VK_ERROR_INITIALIZATION_FAILED},
        {FP_CREATE_BUFFER, 2, VK_ERROR_INITIALIZATION_FAILED},
        {FP_ALLOCATE_MEMORY, 1, VK_ERROR_OUT_OF_DEVICE_MEMORY},
        {FP_ALLOCATE_MEMORY, 2, VK_ERROR_OUT_OF_DEVICE_MEMORY},
        {FP_BIND_MEMORY, 1, VK_ERROR_INITIALIZATION_FAILED},
        {FP_BIND_MEMORY, 2, VK_ERROR_INITIALIZATION_FAILED},
        {FP_MAP_MEMORY, 1, VK_ERROR_MEMORY_MAP_FAILED},
        {FP_ALLOCATE_COMMAND, 1, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_BEGIN_COMMAND, 1, VK_ERROR_INITIALIZATION_FAILED},
        {FP_END_COMMAND, 1, VK_ERROR_INITIALIZATION_FAILED},
        {FP_CREATE_FENCE, 1, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_QUEUE_SUBMIT, 1, VK_ERROR_INITIALIZATION_FAILED}
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fake_reset();
        ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
        ColiVulkanContext *context = NULL;
        ColiVulkanTensor *tensor = NULL;
        uint8_t data[64] = {0};
        CHECK_RESULT(coli_vulkan_context_create(&context, &config),
            COLI_VULKAN_OK, "fault context");
        set_fault(cases[i].point, cases[i].call, cases[i].result);
        ColiVulkanResult result = coli_vulkan_tensor_upload(context, &tensor,
            data, sizeof(data), NULL, 0, TEST_TIMEOUT_NS);
        CHECK(result != COLI_VULKAN_OK && result != COLI_VULKAN_TIMEOUT &&
            result != COLI_VULKAN_DEVICE_LOST, "pre-submit fault result");
        CHECK(!tensor, "pre-submit fault published tensor");
        ColiVulkanContextInfo info;
        CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
            COLI_VULKAN_OK, "fault info");
        CHECK(info.live_tensors == 0 && info.live_allocations == 0 &&
            info.committed_bytes == 0 && info.pending_operations == 0,
            "pre-submit fault cleanup/accounting");
        CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
            COLI_VULKAN_OK, "fault destroy");
    }
}

static void test_timeout_retention(void) {
    fake_reset();
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_TIMEOUT;
    g_fake.wait_results[2] = VK_SUCCESS;
    g_fake.wait_result_count = 3;
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL, *other = NULL;
    uint8_t data[64] = {0};
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "timeout context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_TIMEOUT, "initial timeout");
    CHECK(tensor != NULL, "pending tensor handle");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
        COLI_VULKAN_OK, "timeout info");
    CHECK(info.pending_operations == 1 && info.live_tensors == 1 &&
        info.live_allocations == 2, "retained staging and destination");
    CHECK(g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "timeout retained command buffer and fence");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &other, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_BUSY,
        "new submission rejected while pending");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_BUSY,
        "pending free deferred");
    CHECK(!tensor, "deferred free releases owner pointer");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_TIMEOUT, "finish timeout retains");
    CHECK(g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "repeated timeout retained command buffer and fence");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "eventual finish");
    CHECK(g_fake.allocate_command_buffer_count ==
            g_fake.free_command_buffer_count &&
        g_fake.create_fence_count == g_fake.destroy_fence_count,
        "successful finish released command buffer and fence");
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
        COLI_VULKAN_OK, "finished info");
    CHECK(info.pending_operations == 0 && info.live_tensors == 0 &&
        info.live_allocations == 0 && info.committed_bytes == 0,
        "deferred destroy after finish");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "timeout context destroy");
}

static void test_post_submit_failure_recovery(void) {
    fake_reset();
    g_fake.wait_results[0] = VK_ERROR_UNKNOWN;
    g_fake.wait_results[1] = VK_SUCCESS;
    g_fake.wait_result_count = 2;
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL;
    uint8_t data[64] = {0};
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "post-submit context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_ERROR,
        "post-submit wait failure");
    CHECK(tensor != NULL, "failed submitted tensor retained");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
        COLI_VULKAN_OK, "degraded info");
    CHECK(!info.usable && info.pending_operations == 1,
        "context degraded while completion unknown");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "post-submit eventual success");
    ColiVulkanTensorInfo tensor_info;
    CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &tensor_info),
        COLI_VULKAN_OK, "recovered tensor info");
    CHECK(tensor_info.state == COLI_VULKAN_TENSOR_READY,
        "recovered tensor ready");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "recovered tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "recovered context destroy");
}

static void test_readback_fault_cleanup(void) {
    typedef struct { FaultPoint point; VkResult result; } Case;
    const Case cases[] = {
        {FP_CREATE_BUFFER, VK_ERROR_INITIALIZATION_FAILED},
        {FP_ALLOCATE_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY},
        {FP_BIND_MEMORY, VK_ERROR_INITIALIZATION_FAILED},
        {FP_ALLOCATE_COMMAND, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_BEGIN_COMMAND, VK_ERROR_INITIALIZATION_FAILED},
        {FP_END_COMMAND, VK_ERROR_INITIALIZATION_FAILED},
        {FP_CREATE_FENCE, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_QUEUE_SUBMIT, VK_ERROR_INITIALIZATION_FAILED},
        {FP_MAP_MEMORY, VK_ERROR_MEMORY_MAP_FAILED}
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fake_reset();
        ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
        ColiVulkanContext *context = NULL;
        ColiVulkanTensor *tensor = NULL;
        uint8_t data[64] = {0};
        uint8_t output[256] = {0};
        CHECK_RESULT(coli_vulkan_context_create(&context, &config),
            COLI_VULKAN_OK, "readback fault context");
        CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data,
            sizeof(data), NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_OK,
            "readback fault upload");
        set_fault(cases[i].point, 1, cases[i].result);
        ColiVulkanResult result = coli_vulkan_tensor_readback(context, tensor,
            output, sizeof(output), TEST_TIMEOUT_NS);
        CHECK(result != COLI_VULKAN_OK && result != COLI_VULKAN_TIMEOUT &&
            result != COLI_VULKAN_DEVICE_LOST, "readback pre-submit fault result");
        ColiVulkanContextInfo info;
        CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
            COLI_VULKAN_OK, "readback fault info");
        CHECK(info.live_tensors == 1 && info.live_allocations == 1 &&
            info.committed_bytes != 0 && info.pending_operations == 0,
            "readback fault cleanup/accounting");
        CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
            "readback fault tensor free");
        CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
            COLI_VULKAN_OK, "readback fault destroy");
    }

    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL;
    uint8_t data[64] = {0};
    uint8_t output[256] = {0};
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "readback timeout context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_OK, "readback timeout upload");
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_TIMEOUT;
    g_fake.wait_results[2] = VK_SUCCESS;
    g_fake.wait_result_count = 3;
    CHECK_RESULT(coli_vulkan_tensor_readback(context, tensor, output,
        sizeof(output), TEST_TIMEOUT_NS), COLI_VULKAN_TIMEOUT,
        "readback initial timeout");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "readback timeout info");
    CHECK(info.pending_operations == 1 && info.live_tensors == 1 &&
        info.live_allocations == 2, "readback resources retained");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_BUSY,
        "readback pending free deferred");
    CHECK(!tensor, "readback deferred free releases owner pointer");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_TIMEOUT, "readback finish timeout retains");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "readback eventual finish");
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "readback finished info");
    CHECK(info.pending_operations == 0 && info.live_tensors == 0 &&
        info.live_allocations == 0 && info.committed_bytes == 0,
        "readback deferred cleanup after finish");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "readback timeout destroy");
}

/* Last: DEVICE_LOST deliberately retains the process-wide context until exit. */
static void test_device_lost_retention(void) {
    fake_reset();
    set_fault(FP_WAIT_FENCE, 1, VK_ERROR_DEVICE_LOST);
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    ColiVulkanTensor *tensor = NULL, *other = NULL;
    uint8_t data[64] = {0};
    CHECK_RESULT(coli_vulkan_context_create(&context, &config),
        COLI_VULKAN_OK, "device-lost context");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &tensor, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_DEVICE_LOST,
        "device loss after submit");
    CHECK(tensor != NULL, "device-lost tensor retained");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info),
        COLI_VULKAN_OK, "device-lost info");
    CHECK(info.device_lost && !info.usable && info.pending_operations == 1 &&
        info.live_allocations == 2, "device-lost terminal state");
    CHECK(g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "device loss retained command buffer and fence");
    CHECK_RESULT(coli_vulkan_tensor_upload(context, &other, data, sizeof(data),
        NULL, 0, TEST_TIMEOUT_NS), COLI_VULKAN_DEVICE_LOST,
        "submissions rejected after device loss");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor),
        COLI_VULKAN_DEVICE_LOST, "device-lost free retained");
    CHECK(!tensor, "device-lost owner released");
    set_fault(FP_WAIT_FENCE, 1, VK_ERROR_DEVICE_LOST);
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_DEVICE_LOST, "device-lost finish retained");
    set_fault(FP_WAIT_FENCE, 1, VK_ERROR_DEVICE_LOST);
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_DEVICE_LOST, "device-lost cleanup retained");
    CHECK(context != NULL, "device-lost context must survive until process exit");
    CHECK(g_fake.destroy_pool_count == 0 && g_fake.destroy_device_count == 0 &&
        g_fake.destroy_instance_count == 0,
        "device-lost Vulkan owners were destroyed");
    CHECK(g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "device-lost command buffer or fence was destroyed");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--device-lost") == 0) {
        test_device_lost_retention();
        if (g_failures) {
            fprintf(stderr, "FAIL: %d terminal DEVICE_LOST retention test(s)\n",
                g_failures);
            return 1;
        }
        printf("PASS: terminal DEVICE_LOST retention test\n");
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--device-lost]\n", argv[0]);
        return 2;
    }
    test_config_and_layout();
    test_qt_format_validation();
    test_memory_selector();
    test_device_selection_and_partial_init();
    test_success_lifecycle();
    test_allocation_limit_and_budget();
    test_pre_submit_fault_cleanup();
    test_timeout_retention();
    test_post_submit_failure_recovery();
    test_readback_fault_cleanup();
    fake_check_no_live_resources("ordinary test process exit");
    if (g_failures) {
        fprintf(stderr, "FAIL: %d Vulkan context/fault-injection test(s)\n", g_failures);
        return 1;
    }
    printf("PASS: Vulkan context unit and fault-injection tests\n");
    return 0;
}
