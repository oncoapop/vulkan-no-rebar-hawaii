# Risk Register for Damian's Server (R9 390 Hawaii)

1. **Subgroup Size Instability:** The R9 390 uses GCN 2.0 (Hawaii), which might not correctly report or handle the Vulkan 1.2 subgroup sizes required by the compute shaders if the Mesa driver version is too old.
2. **Memory Type Selection Failure:** If the system fails to expose any memory type that is purely `DEVICE_LOCAL` (Type 0 typically), the staged memory path will fail to initialize.
3. **Queue / Sync Timeouts:** Given the age of the architecture, the 5-second timeout on the staging fence (`vkWaitForFences`) might theoretically be breached if the system is under extreme PCIe or compute load.
4. **Driver Stability:** Experimental compute shaders might trigger an AMDGPU driver reset (ring timeout) on Hawaii.
