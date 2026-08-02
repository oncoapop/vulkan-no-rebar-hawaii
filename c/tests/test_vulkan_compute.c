/* Phase 4A format-2 compute tests using the shared deterministic fake Vulkan. */
#define main phase3a_context_main_unused
#include "test_vulkan_context.c"
#undef main

#include <math.h>
#include "../vulkan_phase4a_gate.h"

#define COMPUTE_I 32u
#define COMPUTE_O 128u
#define COMPUTE_MAX_ROWS 64u

typedef struct {
    uint8_t weights[COMPUTE_O * COMPUTE_I / 2];
    float scales[COMPUTE_O];
} ComputeFixture;

static void compute_fixture_init(ComputeFixture *fixture) {
    for (uint32_t o = 0; o < COMPUTE_O; o++) {
        fixture->scales[o] = 0.0025f * (float)(o + 1);
        for (uint32_t i = 0; i < COMPUTE_I; i += 2) {
            uint8_t low = (uint8_t)((o + i * 3u) & 15u);
            uint8_t high = (uint8_t)((o * 5u + i + 1u) & 15u);
            fixture->weights[o * (COMPUTE_I / 2) + i / 2] =
                (uint8_t)(low | (uint8_t)(high << 4));
        }
    }
}

static void compute_reference(
    const ComputeFixture *fixture,
    const float *input,
    uint32_t rows,
    float *output
) {
    for (uint32_t row = 0; row < rows; row++) for (uint32_t o = 0; o < COMPUTE_O; o++) {
        float sum = 0;
        for (uint32_t i = 0; i < COMPUTE_I; i++) {
            uint8_t packed = fixture->weights[o * (COMPUTE_I / 2) + i / 2];
            int weight = (int)((i & 1u) ? packed >> 4 : packed & 15u) - 8;
            sum += input[row * COMPUTE_I + i] * (float)weight;
        }
        output[row * COMPUTE_O + o] = sum * fixture->scales[o];
    }
}

static ColiVulkanTensor *create_compute_tensor(
    ColiVulkanContext *context,
    ComputeFixture *fixture
) {
    compute_fixture_init(fixture);
    ColiVulkanQTSpec spec = {
        2, COMPUTE_I, COMPUTE_O, 0,
        fixture->weights, sizeof(fixture->weights),
        fixture->scales, sizeof(fixture->scales)
    };
    ColiVulkanTensor *tensor = NULL;
    ColiVulkanResult result = coli_vulkan_tensor_create_qt(context, &tensor,
        &spec, TEST_TIMEOUT_NS);
    if (result != COLI_VULKAN_OK) {
        fprintf(stderr, "FAIL: %s:%d: compute tensor create: %s\n",
            __func__, __LINE__, coli_vulkan_result_string(result));
        g_failures++;
        return NULL;
    }
    return tensor;
}

static void prepare_compute(ColiVulkanContext *context) {
    ColiVulkanComputeConfig compute = {
        COMPUTE_MAX_ROWS, COMPUTE_I, COMPUTE_O
    };
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &compute),
        COLI_VULKAN_OK, "compute prepare");
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &compute),
        COLI_VULKAN_OK, "identical compute prepare is idempotent");
    compute.output_width--;
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &compute),
        COLI_VULKAN_INVALID_ARGUMENT, "different second prepare rejected");
}

static void check_outputs(
    const float *actual,
    const float *expected,
    size_t count,
    const char *message
) {
    for (size_t i = 0; i < count; i++) if (fabsf(actual[i] - expected[i]) >
        5e-5f + 5e-4f * fmaxf(fabsf(actual[i]), fabsf(expected[i]))) {
        fprintf(stderr, "FAIL: %s:%d: %s at %zu: %.9g != %.9g\n",
            __func__, __LINE__, message, i, actual[i], expected[i]);
        g_failures++;
        return;
    }
}

static void test_phase4a_mtp_runtime_gate(void) {
    CHECK(coli_vulkan_phase4a_mtp_gate(1, "0", 0) ==
        COLI_VULKAN_PHASE4A_GATE_OK,
        "checkpoint MTP tensors rejected despite explicit MTP=0 DRAFT=0");
    CHECK(coli_vulkan_phase4a_mtp_gate(0, "0", 0) ==
        COLI_VULKAN_PHASE4A_GATE_OK,
        "checkpoint without MTP tensors rejected with MTP=0 DRAFT=0");
    CHECK(coli_vulkan_phase4a_mtp_gate(1, NULL, 0) ==
        COLI_VULKAN_PHASE4A_GATE_MTP_SETTING,
        "missing MTP setting did not fail closed");
    CHECK(coli_vulkan_phase4a_mtp_gate(0, "1", 0) ==
        COLI_VULKAN_PHASE4A_GATE_MTP_SETTING,
        "MTP=1 did not fail closed");
    CHECK(coli_vulkan_phase4a_mtp_gate(1, "yes", 0) ==
        COLI_VULKAN_PHASE4A_GATE_MTP_SETTING,
        "malformed MTP setting did not fail closed");
    CHECK(coli_vulkan_phase4a_mtp_gate(1, "0", 1) ==
        COLI_VULKAN_PHASE4A_GATE_DRAFT_ACTIVE &&
        coli_vulkan_phase4a_mtp_gate(0, "0", -1) ==
        COLI_VULKAN_PHASE4A_GATE_DRAFT_ACTIVE,
        "effective nonzero draft passed the MTP runtime gate");
}

static void test_compute_endian_gate(void) {
    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    config.test_force_big_endian = 1;
    ColiVulkanContext *context = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config), COLI_VULKAN_OK,
        "big-endian upload-only context");
    ColiVulkanComputeConfig compute = {
        COMPUTE_MAX_ROWS, COMPUTE_I, COMPUTE_O
    };
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &compute),
        COLI_VULKAN_UNSUPPORTED, "big-endian compute rejected");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "big-endian context info");
    CHECK(!info.compute_prepared && info.live_allocations == 0 &&
        g_fake.create_shader_count == 0 &&
        g_fake.create_descriptor_layout_count == 0 &&
        g_fake.create_pipeline_layout_count == 0 &&
        g_fake.create_pipeline_count == 0 &&
        g_fake.create_descriptor_pool_count == 0 &&
        g_fake.allocate_descriptor_set_count == 0,
        "big-endian gate created a compute object");

    ComputeFixture fixture;
    ColiVulkanTensor *tensor = create_compute_tensor(context, &fixture);
    ColiVulkanTensorInfo tensor_info;
    CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &tensor_info),
        COLI_VULKAN_OK, "big-endian upload-only tensor info");
    CHECK(tensor_info.state == COLI_VULKAN_TENSOR_READY,
        "big-endian gate disabled upload-only residency");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "big-endian upload-only tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "big-endian upload-only context destroy");
}

static void test_compute_success_and_eligibility(void) {
    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config), COLI_VULKAN_OK,
        "compute context");
    prepare_compute(context);
    ComputeFixture fixture;
    ColiVulkanTensor *tensor = create_compute_tensor(context, &fixture);
    ColiVulkanTensorInfo tensor_info;
    CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &tensor_info),
        COLI_VULKAN_OK, "compute tensor info");
    CHECK(tensor_info.compute_eligible, "format-2 tensor not compute eligible");

    enum { ROWS = 4 };
    float input[ROWS * COMPUTE_I];
    float actual[ROWS * COMPUTE_O];
    float expected[ROWS * COMPUTE_O];
    for (size_t i = 0; i < ROWS * COMPUTE_I; i++)
        input[i] = ((float)((int)(i % 23) - 11)) / 17.0f;
    compute_reference(&fixture, input, ROWS, expected);
    memset(actual, 0xa5, sizeof(actual));
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, ROWS,
        actual, TEST_TIMEOUT_NS), COLI_VULKAN_OK, "format-2 dispatch");
    check_outputs(actual, expected, ROWS * COMPUTE_O, "fake compute output");

    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "compute context info");
    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    CHECK(info.compute_prepared && info.compute_max_rows == 64 &&
        (info.compute_input_memory_property_flags & required) == required &&
        !(info.compute_input_memory_property_flags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (info.compute_output_memory_property_flags & required) == required &&
        !(info.compute_output_memory_property_flags &
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        info.compute_input_heap_index == 1 && info.compute_output_heap_index == 1,
        "scratch was not coherent non-device-local system memory");
    CHECK(info.compute_dispatch_recorded == 1 && info.compute_submitted == 1 &&
        info.compute_completed == 1 && info.compute_rows_completed == ROWS &&
        !info.pending_operations, "compute telemetry mismatch");
    CHECK(g_fake.host_to_compute_barrier_count == 1 &&
        g_fake.compute_to_host_barrier_count == 1 &&
        g_fake.dispatch_count == 1, "required barriers or dispatch missing");

    uint8_t other_weights[64] = {0}; float other_scales[2] = {1, 1};
    ColiVulkanQTSpec other_spec = {1, 32, 2, 0, other_weights,
        sizeof(other_weights), other_scales, sizeof(other_scales)};
    ColiVulkanTensor *other = NULL;
    CHECK_RESULT(coli_vulkan_tensor_create_qt(context, &other, &other_spec,
        TEST_TIMEOUT_NS), COLI_VULKAN_OK, "resident non-compute format");
    CHECK_RESULT(coli_vulkan_tensor_get_info(other, &tensor_info),
        COLI_VULKAN_OK, "non-compute tensor info");
    CHECK(!tensor_info.compute_eligible, "format 1 became compute eligible");
    float wrong_output[2];
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, other, input, 1,
        wrong_output, TEST_TIMEOUT_NS), COLI_VULKAN_UNSUPPORTED,
        "non-format-2 compute rejected");
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 0,
        actual, TEST_TIMEOUT_NS), COLI_VULKAN_INVALID_ARGUMENT,
        "zero rows rejected");
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 65,
        actual, TEST_TIMEOUT_NS), COLI_VULKAN_UNSUPPORTED,
        "row capacity rejected");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &other), COLI_VULKAN_OK,
        "other tensor free");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "compute tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "compute context destroy");
}

static void test_noncompute_formats_remain_resident_only(void) {
    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config), COLI_VULKAN_OK,
        "noncompute-format context");
    uint8_t weights[600] = {0};
    float scales[10] = {0};
    for (size_t i = 0; i < 10; i++) scales[i] = 0.25f + (float)i;
    const ColiVulkanQTSpec specs[] = {
        {0, 16, 2, 0, weights, 128, NULL, 0},
        {1, 16, 2, 0, weights, 32, scales, 8},
        {2, 17, 3, 0, weights, 27, scales, 12},
        {3, 16, 2, 0, weights, 8, scales, 8},
        {4, 32, 2, 16, weights, 32, scales, 16},
        {5, 64, 2, 0, weights, 48, scales, 8},
        {6, 256, 2, 0, weights, 196, scales, 4},
    };
    for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); i++) {
        ColiVulkanTensor *tensor = NULL;
        CHECK_RESULT(coli_vulkan_tensor_create_qt(context, &tensor, &specs[i],
            TEST_TIMEOUT_NS), COLI_VULKAN_OK, "resident noncompute format");
        ColiVulkanTensorInfo info;
        CHECK_RESULT(coli_vulkan_tensor_get_info(tensor, &info), COLI_VULKAN_OK,
            "resident noncompute info");
        CHECK(!info.compute_eligible,
            "unsupported format/geometry became compute eligible");
        CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
            "resident noncompute free");
    }
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "noncompute-format context destroy");
}

static void run_prepare_fault(FaultPoint point, unsigned call) {
    fake_reset();
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    ColiVulkanContext *context = NULL;
    CHECK_RESULT(coli_vulkan_context_create(&context, &config), COLI_VULKAN_OK,
        "prepare fault context");
    set_fault(point, call, VK_ERROR_INITIALIZATION_FAILED);
    ColiVulkanComputeConfig compute = {64, COMPUTE_I, COMPUTE_O};
    CHECK(coli_vulkan_compute_prepare(context, &compute) != COLI_VULKAN_OK,
        "prepare fault was ignored");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "prepare fault info");
    CHECK(!info.compute_prepared && info.live_allocations == 0,
        "prepare fault retained ordinary resources");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "prepare fault context destroy");
}

static void test_compute_prepare_faults(void) {
    const FaultPoint points[] = {
        FP_CREATE_SHADER, FP_CREATE_DESCRIPTOR_LAYOUT, FP_CREATE_PIPELINE_LAYOUT,
        FP_CREATE_PIPELINE, FP_CREATE_DESCRIPTOR_POOL, FP_ALLOCATE_DESCRIPTOR_SET,
        FP_CREATE_BUFFER, FP_ALLOCATE_MEMORY, FP_BIND_MEMORY, FP_MAP_MEMORY
    };
    for (size_t i = 0; i < sizeof(points) / sizeof(points[0]); i++) {
        run_prepare_fault(points[i], 1);
        if (points[i] == FP_CREATE_BUFFER || points[i] == FP_ALLOCATE_MEMORY ||
            points[i] == FP_BIND_MEMORY || points[i] == FP_MAP_MEMORY)
            run_prepare_fault(points[i], 2);
    }
}

static void setup_dispatch_case(
    ColiVulkanContext **context,
    ColiVulkanTensor **tensor,
    ComputeFixture *fixture
) {
    ColiVulkanConfig config = fake_config(64ULL * 1024ULL * 1024ULL);
    CHECK_RESULT(coli_vulkan_context_create(context, &config), COLI_VULKAN_OK,
        "dispatch case context");
    prepare_compute(*context);
    *tensor = create_compute_tensor(*context, fixture);
}

static void test_compute_dispatch_faults_and_timeout(void) {
    const FaultPoint faults[] = {
        FP_ALLOCATE_COMMAND, FP_BEGIN_COMMAND, FP_END_COMMAND,
        FP_CREATE_FENCE, FP_QUEUE_SUBMIT
    };
    for (size_t i = 0; i < sizeof(faults) / sizeof(faults[0]); i++) {
        fake_reset();
        ColiVulkanContext *context = NULL; ColiVulkanTensor *tensor = NULL;
        ComputeFixture fixture;
        setup_dispatch_case(&context, &tensor, &fixture);
        set_fault(faults[i], 1, VK_ERROR_INITIALIZATION_FAILED);
        float input[COMPUTE_I] = {0}, output[COMPUTE_O];
        memset(output, 0xa5, sizeof(output));
        CHECK(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1, output,
            TEST_TIMEOUT_NS) != COLI_VULKAN_OK, "dispatch fault ignored");
        ColiVulkanContextInfo info;
        CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
            "dispatch fault info");
        CHECK(!info.pending_operations &&
            g_fake.allocate_command_buffer_count ==
                g_fake.free_command_buffer_count &&
            g_fake.create_fence_count == g_fake.destroy_fence_count,
            "pre-submit fault retained transient owner");
        CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
            "dispatch fault tensor free");
        CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
            COLI_VULKAN_OK, "dispatch fault context destroy");
    }

    fake_reset();
    ColiVulkanContext *context = NULL; ColiVulkanTensor *tensor = NULL;
    ComputeFixture fixture;
    setup_dispatch_case(&context, &tensor, &fixture);
    unsigned commands = g_fake.allocate_command_buffer_count;
    unsigned frees = g_fake.free_command_buffer_count;
    unsigned fences = g_fake.create_fence_count;
    unsigned destroys = g_fake.destroy_fence_count;
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_SUCCESS;
    g_fake.wait_result_count = 2;
    float input[COMPUTE_I] = {0}, output[COMPUTE_O];
    for (uint32_t i = 0; i < COMPUTE_O; i++) output[i] = 12345.0f;
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1,
        output, TEST_TIMEOUT_NS), COLI_VULKAN_TIMEOUT,
        "compute timeout retained");
    ColiVulkanComputeConfig same_compute = {
        COMPUTE_MAX_ROWS, COMPUTE_I, COMPUTE_O
    };
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &same_compute),
        COLI_VULKAN_BUSY, "identical prepare rejected while compute pending");
    CHECK(output[0] == 12345.0f &&
        g_fake.allocate_command_buffer_count == commands + 1 &&
        g_fake.free_command_buffer_count == frees &&
        g_fake.create_fence_count == fences + 1 &&
        g_fake.destroy_fence_count == destroys,
        "timeout exposed output or failed to retain command and fence");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "compute timeout finish");
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &same_compute),
        COLI_VULKAN_OK, "identical prepare restored after bounded finish");
    ColiVulkanContextInfo timeout_info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &timeout_info),
        COLI_VULKAN_OK, "timeout-success telemetry");
    CHECK(timeout_info.compute_timeouts == 1 &&
        timeout_info.compute_completed == 1 &&
        timeout_info.compute_rows_completed == 0 &&
        timeout_info.compute_errors == 0 &&
        timeout_info.compute_device_lost == 0,
        "timeout-success telemetry or invalid-output accounting mismatch");
    CHECK(g_fake.allocate_command_buffer_count ==
            g_fake.free_command_buffer_count &&
        g_fake.create_fence_count == g_fake.destroy_fence_count,
        "timeout finish did not balance transient owners");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "timeout tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "timeout context destroy");

    fake_reset();
    context = NULL; tensor = NULL;
    setup_dispatch_case(&context, &tensor, &fixture);
    commands = g_fake.allocate_command_buffer_count;
    frees = g_fake.free_command_buffer_count;
    fences = g_fake.create_fence_count;
    destroys = g_fake.destroy_fence_count;
    g_fake.wait_results[0] = VK_ERROR_UNKNOWN;
    g_fake.wait_results[1] = VK_SUCCESS;
    g_fake.wait_result_count = 2;
    for (uint32_t i = 0; i < COMPUTE_O; i++) output[i] = 12345.0f;
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1,
        output, TEST_TIMEOUT_NS), COLI_VULKAN_ERROR,
        "compute post-submit error retained");
    CHECK(output[0] == 12345.0f &&
        g_fake.allocate_command_buffer_count == commands + 1 &&
        g_fake.free_command_buffer_count == frees &&
        g_fake.create_fence_count == fences + 1 &&
        g_fake.destroy_fence_count == destroys,
        "post-submit error exposed output or destroyed submitted owners");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "post-submit error eventual finish");
    ColiVulkanContextInfo recovered;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &recovered),
        COLI_VULKAN_OK, "post-submit recovery info");
    CHECK(recovered.usable && !recovered.pending_operations &&
        recovered.compute_submitted == 1 && recovered.compute_completed == 1 &&
        recovered.compute_rows_completed == 0 && recovered.compute_errors == 1,
        "post-submit recovery counters/state");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "post-submit recovery tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "post-submit recovery context destroy");

    fake_reset();
    context = NULL; tensor = NULL;
    setup_dispatch_case(&context, &tensor, &fixture);
    commands = g_fake.allocate_command_buffer_count;
    frees = g_fake.free_command_buffer_count;
    fences = g_fake.create_fence_count;
    destroys = g_fake.destroy_fence_count;
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_ERROR_UNKNOWN;
    g_fake.wait_results[2] = VK_SUCCESS;
    g_fake.wait_result_count = 3;
    for (uint32_t i = 0; i < COMPUTE_O; i++) output[i] = 54321.0f;
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1,
        output, TEST_TIMEOUT_NS), COLI_VULKAN_TIMEOUT,
        "timeout-error-success initial timeout");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_ERROR, "timeout-error-success later error");
    ColiVulkanContextInfo error_info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &error_info),
        COLI_VULKAN_OK, "timeout-error telemetry");
    CHECK(!error_info.usable && error_info.pending_operations == 1 &&
        error_info.compute_timeouts == 1 && error_info.compute_errors == 1 &&
        error_info.compute_completed == 0 &&
        error_info.compute_rows_completed == 0 && output[0] == 54321.0f &&
        g_fake.allocate_command_buffer_count == commands + 1 &&
        g_fake.free_command_buffer_count == frees &&
        g_fake.create_fence_count == fences + 1 &&
        g_fake.destroy_fence_count == destroys,
        "later error telemetry, output validity, or retention mismatch");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "timeout-error-success eventual finish");
    CHECK_RESULT(coli_vulkan_context_get_info(context, &error_info),
        COLI_VULKAN_OK, "timeout-error-success final telemetry");
    CHECK(error_info.usable && !error_info.pending_operations &&
        error_info.compute_timeouts == 1 && error_info.compute_errors == 1 &&
        error_info.compute_completed == 1 &&
        error_info.compute_rows_completed == 0 && output[0] == 54321.0f,
        "timeout-error-success final accounting mismatch");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "timeout-error-success tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "timeout-error-success context destroy");
}

typedef struct {
    ColiVulkanContext *context;
    ColiVulkanTensor *tensor;
    float input[COMPUTE_I];
    float output[COMPUTE_O];
    ColiVulkanResult result;
} ThreadCall;

static void *compute_thread(void *opaque) {
    ThreadCall *call = opaque;
    call->result = coli_vulkan_tensor_matmul_fmt2(call->context, call->tensor,
        call->input, 1, call->output, TEST_TIMEOUT_NS);
    return NULL;
}

static void test_compute_two_thread_serialization(void) {
    fake_reset();
    ColiVulkanContext *context = NULL; ColiVulkanTensor *tensor = NULL;
    ComputeFixture fixture;
    setup_dispatch_case(&context, &tensor, &fixture);
    ThreadCall calls[2]; memset(calls, 0, sizeof(calls));
    pthread_t threads[2];
    for (unsigned i = 0; i < 2; i++) {
        calls[i].context = context; calls[i].tensor = tensor;
        for (uint32_t j = 0; j < COMPUTE_I; j++)
            calls[i].input[j] = (float)(i + j) / 31.0f;
        CHECK(pthread_create(&threads[i], NULL, compute_thread, &calls[i]) == 0,
            "compute thread create");
    }
    for (unsigned i = 0; i < 2; i++)
        CHECK(pthread_join(threads[i], NULL) == 0, "compute thread join");
    CHECK(calls[0].result == COLI_VULKAN_OK &&
        calls[1].result == COLI_VULKAN_OK && g_fake.api_max_active == 1,
        "concurrent callers were not serialized");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "thread telemetry");
    CHECK(info.compute_dispatch_recorded == 2 && info.compute_submitted == 2 &&
        info.compute_completed == 2 && info.compute_rows_completed == 2,
        "race-free concurrent counters");
    CHECK_RESULT(coli_vulkan_tensor_free(context, &tensor), COLI_VULKAN_OK,
        "thread tensor free");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "thread context destroy");
}

/* Terminal process-isolated case: submitted owners intentionally survive. */
static void test_compute_device_lost_retention(void) {
    fake_reset();
    ColiVulkanContext *context = NULL; ColiVulkanTensor *tensor = NULL;
    ComputeFixture fixture;
    setup_dispatch_case(&context, &tensor, &fixture);
    unsigned command_frees = g_fake.free_command_buffer_count;
    unsigned fence_destroys = g_fake.destroy_fence_count;
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_ERROR_DEVICE_LOST;
    g_fake.wait_result_count = 2;
    float input[COMPUTE_I] = {0}, output[COMPUTE_O];
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1,
        output, TEST_TIMEOUT_NS), COLI_VULKAN_TIMEOUT,
        "compute timeout before terminal device loss");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_DEVICE_LOST, "pending compute observed terminal device loss");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "device-lost info");
    CHECK(info.device_lost && !info.usable && info.pending_operations == 1 &&
        info.compute_timeouts == 1 && info.compute_device_lost == 1 &&
        g_fake.free_command_buffer_count == command_frees &&
        g_fake.destroy_fence_count == fence_destroys,
        "device loss destroyed submitted owners");
    CHECK_RESULT(coli_vulkan_finish_pending(context, TEST_TIMEOUT_NS),
        COLI_VULKAN_DEVICE_LOST, "repeated terminal finish rejected");
    CHECK_RESULT(coli_vulkan_context_get_info(context, &info), COLI_VULKAN_OK,
        "repeated device-lost info");
    CHECK(info.compute_device_lost == 1,
        "pending compute device loss was double-counted");
    ColiVulkanComputeConfig same_compute = {
        COMPUTE_MAX_ROWS, COMPUTE_I, COMPUTE_O
    };
    CHECK_RESULT(coli_vulkan_compute_prepare(context, &same_compute),
        COLI_VULKAN_DEVICE_LOST,
        "device-lost context reported successful preparation");
    CHECK_RESULT(coli_vulkan_tensor_matmul_fmt2(context, tensor, input, 1,
        output, TEST_TIMEOUT_NS), COLI_VULKAN_DEVICE_LOST,
        "device-lost context rejected submission");
    CHECK_RESULT(coli_vulkan_context_destroy(&context, TEST_TIMEOUT_NS),
        COLI_VULKAN_DEVICE_LOST, "device-lost destruction retained context");
    CHECK(context != NULL && g_fake.destroy_pool_count == 0 &&
        g_fake.destroy_device_count == 0 && g_fake.destroy_pipeline_count == 0 &&
        g_fake.destroy_descriptor_pool_count == 0 &&
        g_fake.free_command_buffer_count == command_frees &&
        g_fake.destroy_fence_count == fence_destroys,
        "device loss destroyed persistent dependency");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--device-lost") == 0) {
        test_compute_device_lost_retention();
        if (g_failures) return 1;
        printf("PASS: Vulkan compute terminal DEVICE_LOST retention test\n");
        return 0;
    }
    if (argc != 1) return 2;
    test_phase4a_mtp_runtime_gate();
    test_compute_endian_gate();
    test_compute_success_and_eligibility();
    test_noncompute_formats_remain_resident_only();
    test_compute_prepare_faults();
    test_compute_dispatch_faults_and_timeout();
    test_compute_two_thread_serialization();
    fake_check_no_live_resources("compute ordinary test process exit");
    if (g_failures) {
        fprintf(stderr, "FAIL: %d Vulkan compute test(s)\n", g_failures);
        return 1;
    }
    printf("PASS: Vulkan format-2 compute unit/fault/concurrency tests\n");
    return 0;
}
