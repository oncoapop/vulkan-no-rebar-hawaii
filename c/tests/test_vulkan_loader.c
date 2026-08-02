/*
 * Phase 3B loader transaction tests. Reuse the deterministic Phase 3A fake
 * dispatch in this translation unit, then include the real loader seam so the
 * tests exercise the exact QT/ESlot publication and cleanup code used by
 * colibri.c without touching a model or real hardware.
 */
#define main phase3a_context_main_unused
#include "test_vulkan_context.c"
#undef main

#define main colibri_main_unused
#include "../colibri.c"
#undef main

typedef struct {
    uint8_t weights[3][27];
    float scales[3][3];
} HostTriplet;

static void init_triplet(ESlot *slot, HostTriplet *host) {
    memset(slot, 0, sizeof(*slot));
    memset(host, 0, sizeof(*host));
    QT *qt[3] = {&slot->g, &slot->u, &slot->d};
    for (int k = 0; k < 3; k++) {
        for (size_t i = 0; i < sizeof(host->weights[k]); i++)
            host->weights[k][i] = (uint8_t)(k * 41 + (int)i);
        for (size_t i = 0; i < 3; i++) host->scales[k][i] = (float)(k + i + 1);
        qt[k]->fmt = 2;
        qt[k]->I = 17;
        qt[k]->O = 3;
        qt[k]->gs = 0;
        qt[k]->q4 = host->weights[k];
        qt[k]->s = host->scales[k];
        qt_set_source_lengths(qt[k], sizeof(host->weights[k]),
            sizeof(host->scales[k]));
    }
}

static void loader_model_init(Model *model, int slots, uint64_t budget) {
    memset(model, 0, sizeof(*model));
    model->c.n_layers = 0;
    model->c.n_experts = slots;
    model->npin = calloc(1, sizeof(*model->npin));
    model->pin = calloc(1, sizeof(*model->pin));
    CHECK(model->npin && model->pin, "loader model arrays");
    model->npin[0] = slots;
    model->pin[0] = calloc((size_t)slots, sizeof(*model->pin[0]));
    CHECK(model->pin[0], "loader slots");
    model->vulkan_requested = 1;
    model->vulkan_timeout_ns = TEST_TIMEOUT_NS;
    ColiVulkanConfig config = fake_config(budget);
    CHECK_RESULT(coli_vulkan_context_create(&model->vulkan_context, &config),
        COLI_VULKAN_OK, "loader context create");
    g_vulkan_exit_context = model->vulkan_context;
    g_vulkan_exit_timeout_ns = TEST_TIMEOUT_NS;
}

static void loader_model_destroy(Model *model) {
    CHECK_RESULT(model_vulkan_release(model), COLI_VULKAN_OK,
        "loader model release");
    free(model->pin[0]);
    free(model->pin);
    free(model->npin);
    model->pin = NULL;
    model->npin = NULL;
}

static void test_loader_transaction_success(void) {
    fake_reset();
    Model model;
    loader_model_init(&model, 1, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host;
    init_triplet(&model.pin[0][0], &host);
    const void *weight_ptrs[3] = {
        model.pin[0][0].g.q4, model.pin[0][0].u.q4, model.pin[0][0].d.q4
    };
    const void *scale_ptrs[3] = {
        model.pin[0][0].g.s, model.pin[0][0].u.s, model.pin[0][0].d.s
    };
    CHECK_RESULT(expert_vulkan_upload_triplet(&model, &model.pin[0][0]),
        COLI_VULKAN_OK, "complete triplet upload");
    CHECK(model.pin[0][0].g.vulkan && model.pin[0][0].u.vulkan &&
        model.pin[0][0].d.vulkan && model.vulkan_expert_count == 1 &&
        model.vulkan_tensor_count == 3, "complete triplet publication");
    QT *qt[3] = {&model.pin[0][0].g, &model.pin[0][0].u,
        &model.pin[0][0].d};
    for (int k = 0; k < 3; k++) {
        ColiVulkanTensorInfo info;
        CHECK_RESULT(coli_vulkan_tensor_get_info(qt[k]->vulkan, &info),
            COLI_VULKAN_OK, "published tensor info");
        CHECK(info.state == COLI_VULKAN_TENSOR_READY && info.fmt == 2 &&
            info.I == 17 && info.O == 3 && info.source_weight_bytes == 27 &&
            info.source_scale_bytes == 12 && !info.compute_eligible,
            "published format metadata");
        CHECK((const void *)qt[k]->q4 == weight_ptrs[k] &&
            (const void *)qt[k]->s == scale_ptrs[k] &&
            qt[k]->weight_bytes == 27 && qt[k]->scale_bytes == 12,
            "host source retained exactly");
    }
    CHECK(g_fake.allocate_command_buffer_count ==
            g_fake.free_command_buffer_count &&
        g_fake.create_fence_count == g_fake.destroy_fence_count,
        "successful upload temporary owners balanced");
    loader_model_destroy(&model);
    CHECK_RESULT(model_vulkan_release(&model), COLI_VULKAN_OK,
        "repeated model release is idempotent");
}

static void test_loader_validation_precedes_allocation(void) {
    fake_reset();
    Model model;
    loader_model_init(&model, 1, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host;
    init_triplet(&model.pin[0][0], &host);
    model.pin[0][0].d.weight_bytes--;
    unsigned buffers = g_fake.create_buffer_count;
    unsigned allocations = g_fake.allocate_memory_count;
    unsigned commands = g_fake.allocate_command_buffer_count;
    unsigned fences = g_fake.create_fence_count;
    unsigned submits = g_fake.queue_submit_count;
    CHECK_RESULT(expert_vulkan_upload_triplet(&model, &model.pin[0][0]),
        COLI_VULKAN_INVALID_ARGUMENT, "invalid down tensor");
    CHECK(!model.pin[0][0].g.vulkan && !model.pin[0][0].u.vulkan &&
        !model.pin[0][0].d.vulkan, "invalid triplet remained unpublished");
    CHECK(buffers == g_fake.create_buffer_count &&
        allocations == g_fake.allocate_memory_count &&
        commands == g_fake.allocate_command_buffer_count &&
        fences == g_fake.create_fence_count && submits == g_fake.queue_submit_count,
        "triplet validation performed no Vulkan work");
    loader_model_destroy(&model);
}

static void run_loader_fault_case(FaultPoint point, unsigned call, VkResult vk_result) {
    fake_reset();
    Model model;
    loader_model_init(&model, 1, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host;
    init_triplet(&model.pin[0][0], &host);
    set_fault(point, call, vk_result);
    ColiVulkanResult result = expert_vulkan_upload_triplet(&model,
        &model.pin[0][0]);
    CHECK(result != COLI_VULKAN_OK, "injected loader failure reported");
    CHECK(!model.pin[0][0].g.vulkan && !model.pin[0][0].u.vulkan &&
        !model.pin[0][0].d.vulkan && model.vulkan_expert_count == 0 &&
        model.vulkan_tensor_count == 0, "fault published a partial expert");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(model.vulkan_context, &info),
        COLI_VULKAN_OK, "fault context info");
    CHECK(info.live_tensors == 0 && info.live_allocations == 0 &&
        info.committed_bytes == 0 && info.pending_operations == 0,
        "ordinary loader fault cleanup");
    loader_model_destroy(&model);
}

static void test_loader_fault_matrix(void) {
    const struct { FaultPoint point; unsigned calls; VkResult result; } cases[] = {
        {FP_CREATE_BUFFER, 6, VK_ERROR_INITIALIZATION_FAILED},
        {FP_ALLOCATE_MEMORY, 6, VK_ERROR_OUT_OF_DEVICE_MEMORY},
        {FP_BIND_MEMORY, 6, VK_ERROR_INITIALIZATION_FAILED},
        {FP_MAP_MEMORY, 3, VK_ERROR_MEMORY_MAP_FAILED},
        {FP_ALLOCATE_COMMAND, 3, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_BEGIN_COMMAND, 3, VK_ERROR_INITIALIZATION_FAILED},
        {FP_END_COMMAND, 3, VK_ERROR_INITIALIZATION_FAILED},
        {FP_CREATE_FENCE, 3, VK_ERROR_OUT_OF_HOST_MEMORY},
        {FP_QUEUE_SUBMIT, 3, VK_ERROR_INITIALIZATION_FAILED},
        {FP_WAIT_FENCE, 3, VK_ERROR_UNKNOWN}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        for (unsigned call = 1; call <= cases[i].calls; call++)
            run_loader_fault_case(cases[i].point, call, cases[i].result);
}

static void test_loader_timeout_retention(void) {
    fake_reset();
    Model model;
    loader_model_init(&model, 1, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host;
    init_triplet(&model.pin[0][0], &host);
    g_fake.wait_results[0] = VK_TIMEOUT;
    g_fake.wait_results[1] = VK_TIMEOUT;
    g_fake.wait_results[2] = VK_SUCCESS;
    g_fake.wait_result_count = 3;
    CHECK_RESULT(expert_vulkan_upload_triplet(&model, &model.pin[0][0]),
        COLI_VULKAN_TIMEOUT, "loader bounded timeout");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(model.vulkan_context, &info),
        COLI_VULKAN_OK, "loader timeout info");
    CHECK(info.pending_operations == 1 && info.live_tensors == 1 &&
        info.live_allocations == 2 &&
        g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "loader timeout retained submitted owners");
    CHECK_RESULT(coli_vulkan_finish_pending(model.vulkan_context, TEST_TIMEOUT_NS),
        COLI_VULKAN_OK, "loader timeout eventual finish");
    CHECK(g_fake.allocate_command_buffer_count ==
            g_fake.free_command_buffer_count &&
        g_fake.create_fence_count == g_fake.destroy_fence_count,
        "loader timeout finish balanced submitted owners");
    loader_model_destroy(&model);
}

static void test_loader_prefix_limits(void) {
    fake_reset();
    g_fake.max_allocations = 4;
    Model model;
    loader_model_init(&model, 2, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host[2];
    init_triplet(&model.pin[0][0], &host[0]);
    init_triplet(&model.pin[0][1], &host[1]);
    PinRec records[2] = {{0, 0, 20}, {0, 1, 10}};
    int slots[2] = {0, 1};
    pin_vulkan_upload_prefix(&model, records, slots, 2);
    CHECK(!model.vulkan_startup_failed && model.vulkan_expert_count == 1 &&
        model.vulkan_tensor_count == 3 && model.pin[0][0].g.vulkan &&
        !model.pin[0][1].g.vulkan, "allocation boundary kept complete prefix");
    loader_model_destroy(&model);

    fake_reset();
    loader_model_init(&model, 1, 8192);
    init_triplet(&model.pin[0][0], &host[0]);
    PinRec one = {0, 0, 1}; int slot = 0;
    pin_vulkan_upload_prefix(&model, &one, &slot, 1);
    CHECK(model.vulkan_startup_failed && model.vulkan_expert_count == 0 &&
        !model.pin[0][0].g.vulkan && !model.pin[0][0].u.vulkan &&
        !model.pin[0][0].d.vulkan, "zero-expert budget boundary failed closed");
    loader_model_destroy(&model);
}

/* Terminal case: resources are deliberately retained until process exit. */
static void test_loader_device_lost_retention(void) {
    fake_reset();
    Model model;
    loader_model_init(&model, 1, 64ULL * 1024ULL * 1024ULL);
    HostTriplet host;
    init_triplet(&model.pin[0][0], &host);
    set_fault(FP_WAIT_FENCE, 1, VK_ERROR_DEVICE_LOST);
    CHECK_RESULT(expert_vulkan_upload_triplet(&model, &model.pin[0][0]),
        COLI_VULKAN_DEVICE_LOST, "loader terminal device loss");
    ColiVulkanContextInfo info;
    CHECK_RESULT(coli_vulkan_context_get_info(model.vulkan_context, &info),
        COLI_VULKAN_OK, "loader device-lost info");
    CHECK(info.device_lost && !info.usable && info.pending_operations == 1 &&
        info.live_tensors == 1 && info.live_allocations == 2,
        "loader terminal state retained");
    CHECK(g_fake.allocate_command_buffer_count == 1 &&
        g_fake.free_command_buffer_count == 0 &&
        g_fake.create_fence_count == 1 && g_fake.destroy_fence_count == 0,
        "loader device loss destroyed submitted owners");
    ColiVulkanTensor *other = NULL;
    ColiVulkanQTSpec spec = qt_vulkan_spec(&model.pin[0][0].g);
    CHECK_RESULT(coli_vulkan_tensor_create_qt(model.vulkan_context, &other, &spec,
        TEST_TIMEOUT_NS), COLI_VULKAN_DEVICE_LOST,
        "loader rejects submission after device loss");
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--device-lost") == 0) {
        test_loader_device_lost_retention();
        if (g_failures) {
            fprintf(stderr, "FAIL: %d loader DEVICE_LOST test(s)\n", g_failures);
            return 1;
        }
        printf("PASS: Vulkan loader terminal DEVICE_LOST retention test\n");
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--device-lost]\n", argv[0]);
        return 2;
    }
    test_loader_transaction_success();
    test_loader_validation_precedes_allocation();
    test_loader_fault_matrix();
    test_loader_timeout_retention();
    test_loader_prefix_limits();
    fake_check_no_live_resources("loader ordinary test process exit");
    if (g_failures) {
        fprintf(stderr, "FAIL: %d Vulkan loader/fault-injection test(s)\n",
            g_failures);
        return 1;
    }
    printf("PASS: Vulkan loader transaction and fault-injection tests\n");
    return 0;
}
