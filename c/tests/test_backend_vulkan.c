#include "../backend_vulkan.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FENCE_TIMEOUT_NS 5000000000ULL

typedef struct {
    uint32_t fmt;
    uint64_t I, O, gs;
    uint64_t weight_bytes, scale_bytes, scale_count, uploaded_scale_bytes;
} HardwareCase;

static int fail_result(const char *what, ColiVulkanResult result) {
    fprintf(stderr, "FAIL: %s: %s\n", what, coli_vulkan_result_string(result));
    return 1;
}

static int run_compute_hardware(ColiVulkanContext *context) {
    enum { I = 32, O = 128, MAX_ROWS = 64 };
    ColiVulkanComputeConfig compute = {MAX_ROWS, I, O};
    ColiVulkanResult result = coli_vulkan_compute_prepare(context, &compute);
    if (result != COLI_VULKAN_OK) return fail_result("compute prepare", result);
    ColiVulkanContextInfo context_info;
    result = coli_vulkan_context_get_info(context, &context_info);
    if (result != COLI_VULKAN_OK) return fail_result("compute info", result);
    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!context_info.compute_prepared || context_info.compute_max_rows != 64 ||
        (context_info.compute_input_memory_property_flags & required) != required ||
        (context_info.compute_output_memory_property_flags & required) != required ||
        (context_info.compute_input_memory_property_flags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        (context_info.compute_output_memory_property_flags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        context_info.compute_input_heap_index == 2 ||
        context_info.compute_output_heap_index == 2 ||
        context_info.compute_input_memory_type_index == 3 ||
        context_info.compute_input_memory_type_index == 4 ||
        context_info.compute_output_memory_type_index == 3 ||
        context_info.compute_output_memory_type_index == 4) {
        fprintf(stderr, "FAIL: compute scratch selected BAR/device-local memory\n");
        return 1;
    }
    printf("compute scratch inputType=%u inputHeap=%u inputFlags=0x%04x "
           "outputType=%u outputHeap=%u outputFlags=0x%04x\n",
        context_info.compute_input_memory_type_index,
        context_info.compute_input_heap_index,
        context_info.compute_input_memory_property_flags,
        context_info.compute_output_memory_type_index,
        context_info.compute_output_heap_index,
        context_info.compute_output_memory_property_flags);

    uint8_t weights[O * I / 2];
    float scales[O];
    for (uint32_t o = 0; o < O; o++) {
        scales[o] = 0.001f * (float)(o + 1);
        for (uint32_t i = 0; i < I; i += 2) {
            uint8_t low = (uint8_t)((o + 3 * i) & 15u);
            uint8_t high = (uint8_t)((5 * o + i + 1) & 15u);
            weights[o * (I / 2) + i / 2] =
                (uint8_t)(low | (uint8_t)(high << 4));
        }
    }
    ColiVulkanQTSpec spec = {2, I, O, 0, weights, sizeof(weights), scales,
        sizeof(scales)};
    ColiVulkanTensor *tensor = NULL;
    result = coli_vulkan_tensor_create_qt(context, &tensor, &spec,
        FENCE_TIMEOUT_NS);
    if (result != COLI_VULKAN_OK || !tensor)
        return fail_result("compute tensor upload", result);
    ColiVulkanTensorInfo tensor_info;
    result = coli_vulkan_tensor_get_info(tensor, &tensor_info);
    if (result != COLI_VULKAN_OK || !tensor_info.compute_eligible) {
        (void)coli_vulkan_tensor_free(context, &tensor);
        return fail_result("compute tensor eligibility",
            result == COLI_VULKAN_OK ? COLI_VULKAN_ERROR : result);
    }

    const uint32_t row_cases[] = {1, 4, 12};
    float input[12 * I], actual[12 * O], expected[12 * O];
    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); i++)
        input[i] = (float)((int)(i % 29) - 14) / 19.0f;
    for (size_t c = 0; c < sizeof(row_cases) / sizeof(row_cases[0]); c++) {
        uint32_t rows = row_cases[c];
        for (uint32_t row = 0; row < rows; row++) for (uint32_t o = 0; o < O; o++) {
            float sum = 0;
            for (uint32_t i = 0; i < I; i++) {
                uint8_t byte = weights[o * (I / 2) + i / 2];
                int weight = (int)((i & 1u) ? byte >> 4 : byte & 15u) - 8;
                sum += input[row * I + i] * (float)weight;
            }
            expected[row * O + o] = sum * scales[o];
        }
        memset(actual, 0xa5, sizeof(actual));
        result = coli_vulkan_tensor_matmul_fmt2(context, tensor, input, rows,
            actual, FENCE_TIMEOUT_NS);
        if (result != COLI_VULKAN_OK) {
            if (result == COLI_VULKAN_TIMEOUT)
                (void)coli_vulkan_finish_pending(context, FENCE_TIMEOUT_NS);
            (void)coli_vulkan_tensor_free(context, &tensor);
            return fail_result("compute dispatch", result);
        }
        for (uint32_t i = 0; i < rows * O; i++) {
            float absolute = fabsf(actual[i] - expected[i]);
            float magnitude = fmaxf(fabsf(actual[i]), fabsf(expected[i]));
            if (!isfinite(actual[i]) ||
                absolute > 5e-5f + 5e-4f * magnitude) {
                fprintf(stderr,
                    "FAIL: compute numerical mismatch rows=%u index=%u "
                    "actual=%.9g expected=%.9g\n",
                    rows, i, actual[i], expected[i]);
                (void)coli_vulkan_tensor_free(context, &tensor);
                return 1;
            }
        }
        printf("compute rows=%u exact-format2-within-tolerance\n", rows);
    }
    result = coli_vulkan_tensor_free(context, &tensor);
    if (result != COLI_VULKAN_OK) return fail_result("compute tensor free", result);
    result = coli_vulkan_context_get_info(context, &context_info);
    if (result != COLI_VULKAN_OK) return fail_result("final compute info", result);
    if (context_info.compute_dispatch_recorded != 3 ||
        context_info.compute_submitted != 3 ||
        context_info.compute_completed != 3 ||
        context_info.compute_rows_completed != 17 ||
        context_info.compute_timeouts || context_info.compute_errors ||
        context_info.compute_device_lost || context_info.pending_operations ||
        context_info.live_tensors != 0) {
        fprintf(stderr, "FAIL: strict compute telemetry/lifecycle mismatch\n");
        return 1;
    }
    printf("compute dispatches=3 submitted=3 completed=3 rows=17 pending=0\n");
    return 0;
}

int main(int argc, char **argv) {
    int strict = 0;
    int compute_strict = 0;
    if (argc == 2 && strcmp(argv[1], "--strict") == 0) {
        strict = 1;
    } else if (argc == 2 && strcmp(argv[1], "--compute-strict") == 0) {
        strict = 1;
        compute_strict = 1;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--strict|--compute-strict]\n", argv[0]);
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

    static const HardwareCase cases[] = {
        {0, 17, 3, 0, 204, 0, 0, 0},
        {1, 17, 3, 0, 51, 12, 3, 12},
        {2, 17, 3, 0, 27, 12, 3, 12},
        {3, 17, 3, 0, 15, 12, 3, 12},
        {4, 33, 3, 16, 51, 36, 9, 36},
        {5, 65, 3, 0, 144, 24, 6, 24},
        {6, 257, 3, 0, 588, 4, 0, 0}
    };
    for (unsigned iteration = 0;
         iteration < sizeof(cases) / sizeof(cases[0]); iteration++) {
        const HardwareCase *test = &cases[iteration];
        uint8_t *weights = malloc((size_t)test->weight_bytes);
        uint8_t *scales = test->scale_bytes
            ? malloc((size_t)test->scale_bytes) : NULL;
        if (!weights || (test->scale_bytes && !scales)) {
            fprintf(stderr, "FAIL: format fixture allocation\n");
            free(weights); free(scales); failed = 1; goto cleanup;
        }
        for (size_t i = 0; i < (size_t)test->weight_bytes; i++)
            weights[i] = (uint8_t)((i * 29u + iteration * 17u) & 0xffu);
        for (size_t i = 0; i < (size_t)test->scale_bytes; i++)
            scales[i] = (uint8_t)((i * 13u + iteration * 31u) & 0xffu);

        ColiVulkanTensor *tensor = NULL;
        ColiVulkanQTSpec spec = {
            test->fmt, test->I, test->O, test->gs, weights,
            test->weight_bytes, scales, test->scale_bytes
        };
        result = coli_vulkan_tensor_create_qt(context, &tensor, &spec,
            FENCE_TIMEOUT_NS);
        if (result == COLI_VULKAN_TIMEOUT && tensor) {
            result = coli_vulkan_finish_pending(context, FENCE_TIMEOUT_NS);
        }
        if (result != COLI_VULKAN_OK || !tensor) {
            failed = fail_result("tensor upload", result);
            free(weights); free(scales);
            goto cleanup;
        }

        ColiVulkanTensorInfo tensor_info;
        result = coli_vulkan_tensor_get_info(tensor, &tensor_info);
        if (result != COLI_VULKAN_OK) {
            failed = fail_result("tensor info", result);
            (void)coli_vulkan_tensor_free(context, &tensor);
            free(weights); free(scales);
            goto cleanup;
        }
        VkDeviceSize alignment = context_info.min_storage_buffer_offset_alignment;
        if (tensor_info.state != COLI_VULKAN_TENSOR_READY ||
            tensor_info.fmt != test->fmt || tensor_info.I != test->I ||
            tensor_info.O != test->O || tensor_info.gs != test->gs ||
            tensor_info.scale_count != test->scale_count ||
            tensor_info.source_weight_bytes != test->weight_bytes ||
            tensor_info.source_scale_bytes != test->scale_bytes ||
            tensor_info.uploaded_scale_bytes != test->uploaded_scale_bytes ||
            tensor_info.compute_eligible ||
            tensor_info.layout.weight_offset % alignment != 0 ||
            tensor_info.layout.scale_offset % alignment != 0 ||
            !(tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            (tensor_info.memory_property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
            tensor_info.heap_index == 2) {
            fprintf(stderr, "FAIL: tensor layout or strict-local placement is invalid\n");
            failed = 1;
            (void)coli_vulkan_tensor_free(context, &tensor);
            free(weights); free(scales);
            goto cleanup;
        }
        printf("tensor[fmt=%u] packed=%" PRIu64 " allocation=%" PRIu64
               " weightOffset=%" PRIu64 " scaleOffset=%" PRIu64
               " memoryType=%u heap=%u flags=0x%04x\n",
            test->fmt, (uint64_t)tensor_info.layout.packed_size,
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
            free(weights); free(scales);
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
            free(weights); free(scales);
            goto cleanup;
        }
        if (memcmp(readback + tensor_info.layout.weight_offset, weights,
                (size_t)test->weight_bytes) != 0 ||
            (test->uploaded_scale_bytes &&
             memcmp(readback + tensor_info.layout.scale_offset, scales,
                (size_t)test->uploaded_scale_bytes) != 0)) {
            fprintf(stderr, "FAIL: exact packed byte readback mismatch\n");
            failed = 1;
            free(readback);
            (void)coli_vulkan_tensor_free(context, &tensor);
            free(weights); free(scales);
            goto cleanup;
        }
        free(readback);
        free(weights); free(scales);

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
    if (compute_strict && run_compute_hardware(context)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    result = coli_vulkan_context_destroy(&context, FENCE_TIMEOUT_NS);
    if (result != COLI_VULKAN_OK || context) {
        fprintf(stderr, "FAIL: context destroy: %s\n",
            coli_vulkan_result_string(result));
        failed = 1;
    }
    if (!failed) printf(compute_strict
        ? "PASS: Vulkan Phase 4A strict R9 390 format-2 compute lifecycle\n"
        : "PASS: persistent Vulkan Phase 3B validated QT hardware lifecycle\n");
    return failed ? 1 : 0;
}
