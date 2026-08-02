#ifndef COLI_VULKAN_PHASE4A_GATE_H
#define COLI_VULKAN_PHASE4A_GATE_H

#include <string.h>

typedef enum {
    COLI_VULKAN_PHASE4A_GATE_OK = 0,
    COLI_VULKAN_PHASE4A_GATE_MTP_SETTING,
    COLI_VULKAN_PHASE4A_GATE_DRAFT_ACTIVE
} ColiVulkanPhase4aGateResult;

/*
 * Phase 4A cares whether MTP execution is explicitly disabled, not whether
 * the checkpoint contains unused MTP tensors. Keep has_mtp in this seam so
 * tests prove that checkpoint metadata cannot change the result.
 */
static inline ColiVulkanPhase4aGateResult coli_vulkan_phase4a_mtp_gate(
    int has_mtp,
    const char *mtp_setting,
    int effective_draft
) {
    (void)has_mtp;
    if (!mtp_setting || strcmp(mtp_setting, "0") != 0)
        return COLI_VULKAN_PHASE4A_GATE_MTP_SETTING;
    if (effective_draft != 0)
        return COLI_VULKAN_PHASE4A_GATE_DRAFT_ACTIVE;
    return COLI_VULKAN_PHASE4A_GATE_OK;
}

#endif
