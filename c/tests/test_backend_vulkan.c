#include "../backend_vulkan.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FENCE_TIMEOUT_NS 5000000000ULL
#define TEST_WEIGHT_BYTES 4093u
#define TEST_SCALE_BYTES 513u

static int fail_result(const char *what, ColiVulkanResult result) {
    fprintf(stderr, "FAIL: %s: %s\n", what, coli_vulkan_result_string(result));
    return 1;
}

int main(int argc, char **argv) {
    int strict = 0;
    if (argc == 2 && strcmp(argv[1], "--strict") == 0) {
        strict = 1;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--strict]\n", argv[0]);
        return 2;
    }
    int failed = 0;
    ColiVulkanConfig config;
    ColiVulkanResult result = coli_vulkan_config_from_env(&config, FENCE_TIMEOUT_NS);
    if (result != COLI_VULKAN_OK) {
        fprintf(stderr,
            "FAIL: VULKAN_EXPERT_MB must be explicitly set (hardware tests use 64)\n");
        return 1;
    }
    if (config.expert_budget_bytes != 64ULL * 1024ULL * 1024ULL) {
        fprintf(stderr, "FAIL: hardware test requires VULKAN_EXPERT_MB=64\n");
        return 1;
    }

    ColiVulkanContext *context = NULL;
    result = coli_vulkan_context_create(&context, &config);
    if (result == COLI_VULKAN_UNAVAILABLE) {
        if (strict) {
            fprintf(stderr,
                "FAIL: strict hardware acceptance requires exact AMD R9 390 0x1002/0x67b1\n");
            return 1;
        }
        printf("SKIP: exact AMD R9 390 0x1002/0x67b1 is unavailable\n");
        return 0;
    }
    if (result != COLI_VULKAN_OK) return fail_result("context create", result);

    ColiVulkanContextInfo context_info;
    result = coli_vulkan_context_get_info(context, &context_info);
    if (result != COLI_VULKAN_OK) {
        failed = fail_result("context info", result);
        goto cleanup;
    }
    printf("device vendor=0x%04x device=0x%04x queueFamily=%u\n",
        context_info.vendor_id, context_info.device_id,
        context_info.queue_family_index);
    printf("limits maxMemoryAllocationCount=%u minStorageBufferOffsetAlignment=%" PRIu64 "\n",
        context_info.max_memory_allocation_count,
        (uint64_t)context_info.min_storage_buffer_offset_alignment);
    printf("budget requested=%" PRIu64 " effective=%" PRIu64 " reserve=%" PRIu64 "\n",
        context_info.requested_budget_bytes, context_info.effective_budget_bytes,
        (uint64_t)COLI_VULKAN_VRAM_RESERVE_BYTES);

    if (context_info.vendor_id != COLI_VULKAN_VENDOR_ID ||
        context_info.device_id != COLI_VULKAN_DEVICE_ID) {
        fprintf(stderr, "FAIL: selected device is not exact 0x1002/0x67b1\n");
        failed = 1;
        goto cleanup;
    }
    if (!context_info.max_memory_allocation_count ||
        !context_info.min_storage_buffer_offset_alignment) {
        fprintf(stderr, "FAIL: invalid physical-device limits\n");
        failed = 1;
        goto cleanup;
    }
    if (context_info.requested_budget_bytes != 64ULL * 1024ULL * 1024ULL ||
        context_info.effective_budget_bytes > context_info.requested_budget_bytes) {
        fprintf(stderr, "FAIL: explicit 64 MiB cap was not enforced\n");
        failed = 1;
        goto cleanup;
    }

    VkPhysicalDeviceMemoryProperties memory_properties;
    result = coli_vulkan_context_get_memory_properties(context, &memory_properties);
    if (result != COLI_VULKAN_OK) {
        failed = fail_result("memory properties", result);
        goto cleanup;
    }
    printf("memory heaps=%u types=%u\n", memory_properties.memoryHeapCount,
        memory_properties.memoryTypeCount);
    if (memory_properties.memoryHeapCount != 3 ||
        memory_properties.memoryTypeCount != 7) {
        fprintf(stderr, "FAIL: expected the host capture's three heaps and seven types\n");
        failed = 1;
        goto cleanup;
    }
    if (memory_properties.memoryHeaps[2].size != 268435456ULL) {
        fprintf(stderr, "FAIL: heap 2 is not the 256 MiB BAR\n");
        failed = 1;
        goto cleanup;
    }
    for (uint32_t i = 3; i <= 4; i++) {
        if (memory_properties.memoryTypes[i].heapIndex != 2 ||
            memory_properties.memoryTypes[i].propertyFlags != 0x0007u) {
            fprintf(stderr,
                "FAIL: memory type %u does not match heap 2/propertyFlags 0x0007\n", i);
            failed = 1;
            goto cleanup;
        }
    }

    for (unsigned iteration = 0; iteration < 3; iteration++) {
        uint8_t weights[TEST_WEIGHT_BYTES];
        uint8_t scales[TEST_SCALE_BYTES];
        for (size_t i = 0; i < sizeof(weights); i++)
            weights[i] = (uint8_t)((i * 29u + iteration * 17u) & 0xffu);
        for (size_t i = 0; i < sizeof(scales); i++)
            scales[i] = (uint8_t)((i * 13u + iteration * 31u) & 0xffu);

        ColiVulkanTensor *tensor = NULL;
        result = coli_vulkan_tensor_upload(context, &tensor, weights,
            sizeof(weights), scales, sizeof(scales), FENCE_TIMEOUT_NS);
        if (result == COLI_VULKAN_TIMEOUT && tensor) {
            result = coli_vulkan_finish_pending(context, FENCE_TIMEOUT_NS);
        }
        if (result != COLI_VULKAN_OK || !tensor) {
            failed = fail_result("tensor upload", result);
            goto cleanup;
        }

        ColiVulkanTensorInfo tensor_info;
        result = coli_vulkan_tensor_get_info(tensor, &tensor_info);
        if (result != COLI_VULKAN_OK) {
            failed = fail_result("tensor info", result);
            (void)coli_vulkan_tensor_free(context, &tensor);
            goto cleanup;
        }
        VkDeviceSize alignment = context_info.min_storage_buffer_offset_alignment;
        if (tensor_info.state != COLI_VULKAN_TENSOR_READY ||
            tensor_info.layout.weight_offset % alignment != 0 ||
            tensor_info.layout.scale_offset % alignment != 0 ||
            !(tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            (tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
            tensor_info.heap_index == 2) {
            fprintf(stderr, "FAIL: tensor layout or strict-local placement is invalid\n");
            failed = 1;
            (void)coli_vulkan_tensor_free(context, &tensor);
            goto cleanup;
        }
        printf("tensor[%u] packed=%" PRIu64 " allocation=%" PRIu64
               " weightOffset=%" PRIu64 " scaleOffset=%" PRIu64
               " memoryType=%u heap=%u flags=0x%04x\n",
            iteration, (uint64_t)tensor_info.layout.packed_size,
            (uint64_t)tensor_info.allocation_size,
            (uint64_t)tensor_info.layout.weight_offset,
            (uint64_t)tensor_info.layout.scale_offset,
            tensor_info.memory_type_index, tensor_info.heap_index,
            tensor_info.memory_property_flags);

        uint8_t *readback = malloc((size_t)tensor_info.layout.packed_size);
        if (!readback) {
            fprintf(stderr, "FAIL: readback allocation\n");
            failed = 1;
            (void)coli_vulkan_tensor_free(context, &tensor);
            goto cleanup;
        }
        memset(readback, 0xa5, (size_t)tensor_info.layout.packed_size);
        result = coli_vulkan_tensor_readback(context, tensor, readback,
            (uint64_t)tensor_info.layout.packed_size, FENCE_TIMEOUT_NS);
        if (result == COLI_VULKAN_TIMEOUT) {
            result = coli_vulkan_finish_pending(context, FENCE_TIMEOUT_NS);
            if (result == COLI_VULKAN_OK)
                result = coli_vulkan_tensor_readback(context, tensor, readback,
                    (uint64_t)tensor_info.layout.packed_size, FENCE_TIMEOUT_NS);
        }
        if (result != COLI_VULKAN_OK) {
            failed = fail_result("tensor readback", result);
            free(readback);
            (void)coli_vulkan_tensor_free(context, &tensor);
            goto cleanup;
        }
        if (memcmp(readback + tensor_info.layout.weight_offset, weights,
                sizeof(weights)) != 0 ||
            memcmp(readback + tensor_info.layout.scale_offset, scales,
                sizeof(scales)) != 0) {
            fprintf(stderr, "FAIL: exact packed byte readback mismatch\n");
            failed = 1;
            free(readback);
            (void)coli_vulkan_tensor_free(context, &tensor);
            goto cleanup;
        }
        free(readback);

        result = coli_vulkan_tensor_free(context, &tensor);
        if (result != COLI_VULKAN_OK || tensor) {
            failed = fail_result("tensor free", result);
            goto cleanup;
        }
        result = coli_vulkan_context_get_info(context, &context_info);
        if (result != COLI_VULKAN_OK || context_info.committed_bytes != 0 ||
            context_info.live_tensors != 0 || context_info.live_allocations != 0 ||
            context_info.pending_operations != 0) {
            fprintf(stderr, "FAIL: counters did not return to zero\n");
            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    result = coli_vulkan_context_destroy(&context, FENCE_TIMEOUT_NS);
    if (result != COLI_VULKAN_OK || context) {
        fprintf(stderr, "FAIL: context destroy: %s\n",
            coli_vulkan_result_string(result));
        failed = 1;
    }
    if (!failed) printf("PASS: persistent Vulkan Phase 3A hardware lifecycle\n");
    return failed ? 1 : 0;
}
