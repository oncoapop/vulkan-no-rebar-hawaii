# Hardware Capabilities

**GPU:** AMD Radeon R9 390 8 GB, Hawaii (Pending verification on host)
**Driver:** Mesa RADV (Pending verification on host)
**PCIe BAR:** ~256 MB (Resizable BAR disabled/unavailable)

**Pending Verification on Host:**
*   Vulkan API Version
*   Memory Heaps and Types
*   Actual fallback behavior of `HOST_VISIBLE | DEVICE_LOCAL` allocations beyond the 256MB BAR limit.

*Note: Hardware metrics were not collected because the Jules environment lacks the R9 390.*
