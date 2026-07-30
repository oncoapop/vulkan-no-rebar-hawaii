# Upstream Vulkan Assessment

**Verified in Jules:**
*   Source inspection of Vulkan backend (`c/backend_vulkan.c`) and shaders.
*   Compilation command (`make glm VK=1`) succeeds.
*   Non-GPU unit tests execute successfully.

**Not verified in Jules (Pending Hardware):**
*   R9 390 Vulkan initialization.
*   RADV Hawaii subgroup behavior.
*   Device-local residency vs system RAM fallback on the 256MB BAR limit.
*   Performance and VRAM use.
*   GLM-5.2 correctness and token-identical output.

*Note: Upstream Vulkan execution and analysis must be performed on Damian's server.*
