# Testing Plan for Vulkan Acceleration on AMD R9 390 (Hawaii)

**Verified in Jules:**
*   Source inspection and compilation of the baseline and proposed changes.
*   Implementation of the `feature/vulkan-no-rebar-hawaii` experimental staged upload patch.
*   Non-GPU unit test runs.

**Not verified in Jules (Pending Execution on Damian's Server):**
1. **Phase 1: CPU Baseline**: Document the existing working environment and run a controlled CPU-only test with greedy decoding.
2. **Phase 2: Hardware Capabilities**: Collect `vulkaninfo` and `lspci`. Validate 256MB BAR limit behavior.
3. **Phase 3: Upstream Vulkan Test**: Build the unmodified Vulkan backend from upstream and test correctness and stability on the GPU.
4. **Phase 4 & 5: Evaluate Kernels**: Enable Vulkan offloads sequentially (`COLI_VK_EXPERTS`, `COLI_VK_DENSE`, `COLI_VK_ATTN`) and observe VRAM utilization, performance, and stability.
5. **Phase 6: Analyze BAR Limitation**: Provide evidence of VRAM residency constraints.
6. **Phase 7: Test No-ReBAR Staged Uploads**: Run the experimental staging implementation (`COLI_VK_STAGED=1`) and validate stability, exact token correctness, and performance.
