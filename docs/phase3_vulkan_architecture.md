# Phase 3 Vulkan Expert-Loading Architecture

Architecture review performed on clean branch
`feature/vulkan-expert-upload-integration-phase3` at
`b67a1cb1cae6448b8076334d35a68e8bc2be48a9`.

The goal is to integrate the validated Vulkan staged-upload helper into the
real expert-loading architecture for one AMD R9 390 with a 256 MB PCIe BAR,
without assuming ReBAR and without changing CPU, CUDA, HIP, or Metal behavior.
Vulkan must remain opt-in.

## Architecture verdict

**GO for a persistent Vulkan context and tensor-lifecycle milestone. NO-GO for
direct expert-compute integration yet.**

The current Vulkan helper is a safe transfer primitive, but it has no persistent
device owner, destination-buffer lifecycle, VRAM accounting, retained-operation
registry, or Vulkan expert-compute consumer. Phase 3 therefore first requires a
persistent context and tensor/buffer lifecycle layer. Uploading expert weights
directly from `colibri.c` before that layer exists would be unsafe. Uploading
weights without a Vulkan compute path would also consume VRAM without changing
inference execution; releasing the host copies would break CPU fallback.

## Constraints

- One AMD R9 390 only.
- No multi-GPU support.
- No ReBAR assumption.
- No per-upload Vulkan instance, device, queue, or command-pool creation.
- Expert destination buffers must be strictly DEVICE_LOCAL and non-HOST_VISIBLE.
- VRAM use must be explicitly bounded.
- CPU, CUDA, HIP, and Metal behavior must remain unchanged.
- Vulkan must require build-time and runtime opt-in.
- Do not touch or run the active GLM-5.2 installation.
- Do not run GLM-5.2.
- Do not modify `main`.
- Do not create a PR, commit, or merge as part of the architecture review.

## 1. Exact current expert-loading call path and relevant functions

### Startup and model indexing

1. `c/colibri.c:6809`, `main()`, parses CPU, CUDA/HIP, Metal, expert-cache,
   pinning, mmap, pipe, and io_uring options.
2. `main()` calls `model_init(&m, snap, cap, ebits, dbits)` at
   `c/colibri.c:7117`.
3. `c/colibri.c:1161`, `model_init()`, calls `st_init_multi()` to index the
   safetensors shards, then loads dense tensors through `qt_load()` and
   `qt_from_disk()`.
4. `c/colibri.c:113`, `QT`, stores tensor format, host weight/scale pointers,
   geometry, and an optional `ColiCudaTensor *cuda`. It has no Vulkan handle.
5. `c/colibri.c:162`, `ESlot`, owns one expert's gate/up/down `QT` values and
   their host `slab`/`fslab` backing.
6. `c/colibri.c:185`, `Model`, owns:
   - `m->ecache`: per-layer resident LRU slots;
   - `m->ws[64]`: reusable working-set slots for misses;
   - `m->pin`: per-layer permanently pinned hot experts.
7. `model_init()` allocates sparse-layer LRU arrays at `c/colibri.c:1222` and
   the associated routing/usage state.

### Startup pinned experts

1. `main()` invokes `pin_load()` at `c/colibri.c:7209` for explicit `PIN`, or
   at `c/colibri.c:7234` for automatic usage-history pinning.
2. `c/colibri.c:6445`, `pin_load()`, ranks `(layer, expert)` records, calculates
   the RAM and optional CUDA VRAM tiers, allocates `m->pin[layer]`, and assigns
   stable slot indices.
3. It loads the selected experts in parallel through `expert_load()` at
   `c/colibri.c:6534-6536` and `c/colibri.c:6600-6603`.
4. Its CUDA-only serial loop at `c/colibri.c:6542-6573`:
   - calculates the gate/up/down logical byte requirement;
   - selects a CUDA device;
   - marks all three tensors eligible;
   - calls `qt_cuda_upload()` for gate, up, and down;
   - accounts actual bytes with `coli_cuda_tensor_bytes()`;
   - optionally calls `expert_host_release()`;
   - rolls back all three handles if any upload fails.
5. This serial post-load loop is the natural eventual Vulkan upload insertion
   point. Disk loading can remain parallel, while Vulkan queue and command-pool
   access stays on the main thread.

### Runtime routed-expert misses

1. `c/colibri.c:2838`, `moe()`, performs sparse routing and builds the unique
   expert set.
2. At `c/colibri.c:3211-3217`, each selected expert resolves in order from:
   - `m->pin[layer]`;
   - `m->ecache[layer]`;
   - `m->ws[nmiss]` on a miss.
3. Misses are loaded through one of three paths:
   - blocking/OpenMP `expert_load()` at `c/colibri.c:3288-3291`;
   - `pipe_dispatch()` and `pipe_worker()`, where the workers call
     `expert_load()` into distinct `ws[]` slots at `c/colibri.c:1995-2013`;
   - Linux `uring_load_add()` followed by `uring_finalize_load()` at
     `c/colibri.c:1793` and `c/colibri.c:1902`.
4. `c/colibri.c:2064`, `pipe_wait()`, finishes each asynchronous load before
   the host tensor is consumed or its working slot is recycled.
5. At `c/colibri.c:3675-3680`, complete `ESlot` structures are swapped from
   `m->ws[]` into the LRU. A future Vulkan handle inside `QT` would move with
   the structure and therefore needs explicit eviction and pending-operation
   semantics.

### Disk-to-host expert loading

`c/colibri.c:1511`, `expert_load_impl()`, constructs the three names
`gate_proj`, `up_proj`, and `down_proj`, then supports these layouts:

- Unquantized fallback: `qt_from_disk()` reads or quantizes each tensor
  separately.
- Quantized mmap: each `QT` points directly into mapped safetensors data.
- Quantized slab: coalesced or individual `pread` calls fill `s->slab`; scale
  reads fill `s->fslab`; `c/colibri.c:1722-1729` publishes the three `QT` views.
- io_uring: `uring_load_add()` submits the equivalent reads and
  `uring_finalize_load()` publishes the same tensor views.

`c/colibri.c:1734`, `expert_load()`, wraps `expert_load_impl()` and records disk
service time. The relevant safetensors functions are:

- `c/st.h:370`, `st_find()`;
- `c/st.h:454`, `st_prefetch()`;
- `c/st.h:461`, `st_prefetch_rep()`;
- `c/st.h:471`, `st_read_f32()`;
- `c/st.h:503`, `st_read_f32_cap()`;
- `c/st.h:515`, `st_nbytes()`;
- `c/st.h:521`, `st_read_raw()`;
- `c/st.h:531`, `st_read_slice_f32()`.

The optimized expert path normally uses `st_find()` metadata and direct or
coalesced reads rather than `st_read_raw()`.

### Compute and fallback

- CPU expert execution calls `expert_gate_up()`, applies SiLU and the up
  projection product, then calls `matmul_qt()` for the down projection.
- `c/colibri.c:596`, `matmul_qt_ex()`, attempts Metal, then CUDA, then the
  format-specific CPU kernels.
- At `c/colibri.c:3500-3667`, CUDA-resident expert triplets are consumed through
  `coli_cuda_expert_mlp()` or grouped CUDA expert APIs.
- At `c/colibri.c:3520` and on grouped failures, CUDA can call
  `expert_host_ensure()` to restore a released host copy before CPU fallback.
- No Vulkan matmul, expert-MLP, descriptor, or activation-transfer API exists.
  A Vulkan upload alone cannot produce expert inference.

### CUDA/HIP lifecycle used as the comparison model

- `c/backend_cuda.h:25` defines the opaque persistent `ColiCudaTensor`.
- `c/backend_cuda.h:28-35` exposes backend initialization, shutdown, device
  queries, and statistics.
- `c/backend_cuda.h:44-50` exposes upload-only tensor creation.
- `c/backend_cuda.h:64-94` exposes fused and grouped expert compute.
- `c/backend_cuda.h:121-127` exposes tensor destruction, accounting, device
  identity, and in-place update.
- `c/backend_cuda.cu:21`, `ColiCudaTensor`, owns device weight/scale allocations
  and format geometry.
- `c/backend_cuda.cu:34`, `DeviceContext`, owns the persistent per-device stream,
  scratch allocations, and accounting.
- `c/backend_cuda.cu:625`, `coli_cuda_init()`, creates the contexts and streams.
- `c/backend_cuda.cu:672`, `coli_cuda_shutdown()`, destroys persistent backend
  scratch and streams.
- `c/backend_cuda.cu:748`, `coli_cuda_tensor_upload()`, allocates persistent
  device weights/scales and copies host data.
- `c/backend_cuda.cu:805`, `coli_cuda_tensor_update()`, refreshes an existing
  allocation.
- `c/backend_cuda.cu:1324`, `coli_cuda_tensor_free()`, releases tensor storage
  and updates counters.
- HIP compiles this same interface through the compatibility layer. The Vulkan
  work must not change it.

### `backend_loader.c`

`c/backend_loader.c` is guarded by `_WIN32` and is solely a dynamic CUDA DLL
shim:

- `c/backend_loader.c:160`, `coli_cuda_load()`, resolves `coli_cuda.dll`.
- `c/backend_loader.c:279`, `coli_cuda_init()`, and `c/backend_loader.c:284`,
  `coli_cuda_shutdown()`, forward context lifecycle.
- `c/backend_loader.c:356`, `coli_cuda_tensor_upload()`,
  `c/backend_loader.c:379`, `coli_cuda_tensor_free()`, and
  `c/backend_loader.c:537`, `coli_cuda_tensor_update()`, forward tensor
  lifecycle.

It is not part of the native Linux Vulkan path and should not be repurposed as a
generic backend loader during Phase 3.

### Current Vulkan helper and Makefile

- `c/backend_vulkan.h:11` defines `VulkanUploadResult`.
- `c/backend_vulkan.h:21` defines `VulkanUploadOp`, which retains staging
  buffer/memory, command buffer, and fence after a post-submit timeout/failure.
- `c/backend_vulkan.c:17`, `vulkan_staged_upload()`, assumes the caller already
  owns the device, queue, command pool, physical memory properties, and
  destination buffer.
- `c/backend_vulkan.c:145`, `vulkan_upload_finish()`, performs a bounded fence
  wait and releases the retained temporary resources only after success.
- `c/Makefile:222-224` discovers Vulkan compiler/linker flags only for tests.
- `c/Makefile:308`, `BUILD_CONFIG`, does not contain a Vulkan build flag.
- `c/Makefile:321-322`, the engine link, includes only CUDA and Metal objects.
- `c/Makefile:576-590` builds `backend_vulkan.o` and the two standalone Vulkan
  hardware tests; `vulkan-test` uses `&&` so a test failure propagates.

There is no centralized model destruction path in `colibri.c`, and existing
`coli_cuda_shutdown()` and `coli_metal_shutdown()` are not called from all
post-initialization returns. Vulkan cannot rely on process-exit-only ownership
because a retained fence operation makes destruction order safety-critical.

## 2. Exact files and functions that would need modification

### Required for the prerequisite lifecycle milestone

| File | Required change |
| --- | --- |
| `c/backend_vulkan.h` | Add opaque `ColiVulkanContext` and `ColiVulkanTensor` types; configuration/status structures; init, bounded shutdown, tensor free/state/bytes APIs; pending-operation polling/finishing; statistics. Preserve the validated low-level upload API. Keep the Phase 3A raw tensor-upload declaration internal/test-only so loader code cannot bypass Phase 3B format validation. |
| `c/backend_vulkan.c` | Add deterministic physical-device and queue selection, the persistent context, strict destination allocation, tensor packing/lifecycle, VRAM accounting, context mutex, retained-operation registry, bounded shutdown, and partial-init cleanup. |
| `c/tests/test_backend_vulkan.c` | Move the real hardware test to the public context/tensor API and verify identity, topology, strict allocation, persistent-object counts, byte-exact readback, accounting, and shutdown. |
| `c/tests/test_vulkan_context.c` (new) | Unit tests for selection, memory filtering, size validation, budgets, tensor state transitions, and deterministic injected Vulkan failures. |
| `c/Makefile` | Add opt-in `VULKAN=1`, `VULKAN_OBJ`, `-DCOLI_VULKAN`, engine link dependencies when enabled, `VULKAN` in `.build-config`, and fail-propagating unit/hardware targets. Default builds must not include or link Vulkan. |
| `c/tools/clean.py` | Add the Vulkan object and generated Unix test binaries. The existing list does not remove them. |

### Required only after the lifecycle milestone passes

| File/function | Required change |
| --- | --- |
| `c/colibri.c`, includes and backend globals | Include `backend_vulkan.h` only under `COLI_VULKAN`; add one-backend runtime state and explicit budget/timeout configuration. |
| `c/colibri.c`, `QT` | Add an independently guarded `ColiVulkanTensor *vulkan` and explicit Vulkan state. Do not reuse CUDA eligibility/failure/device fields. |
| `c/colibri.c`, new `qt_vulkan_reset()` | Request safe tensor destruction; if its upload is pending, defer actual Vulkan destruction through the context registry. |
| `c/colibri.c`, new `qt_vulkan_upload()` | Construct a format specification, call the mandatory backend format validator, and pass only its validated layout/source ranges to the internal persistent tensor-upload transport. Direct loader calls to the raw transport are prohibited. |
| `c/colibri.c`, new `expert_vulkan_upload_triplet()` | Upload gate/up/down transactionally and publish all three only after all become READY. |
| `c/colibri.c`, `main()` | Parse build/runtime opt-in, create the context once before expert pinning, reject unsupported combinations, and guarantee bounded shutdown on every post-init return. |
| `c/colibri.c`, `pin_load()` | After parallel host loads complete, serially place a bounded prefix into Vulkan memory and retain all host copies. |
| `c/colibri.c`, new `model_vulkan_release()` | Walk every Vulkan-owning model slot and release tensor ownership before final context shutdown. |
| `c/colibri.c`, `expert_load_impl()` and `uring_load_add()` | Only when reusable/LRU Vulkan slots are later enabled: invalidate old Vulkan contents safely before changing `eid`. This must not be added for a pin-only milestone. |
| `c/colibri.c`, `repin_pass_limit()` | Deferred initially. A later implementation must use allocate-new/upload/finish/swap/free-old semantics rather than overwriting an in-use buffer. |
| `c/colibri.c`, `moe()` and `matmul_qt_ex()` | Deferred until Vulkan compute exists. Add Vulkan dispatch only with tested CPU fallback and completed upload state. |

### Files that should remain unchanged in the first milestone

- `c/backend_cuda.h`
- `c/backend_cuda.cu`
- `c/backend_loader.c`
- `c/st.h`
- `c/backend_metal.h`
- `c/backend_metal.mm`

No generic backend abstraction or drive-by CUDA/Metal refactor is needed.

## 3. Persistent Vulkan context, queue, command-pool, and synchronization model

One process-wide `ColiVulkanContext` should own:

- one `VkInstance`;
- one selected `VkPhysicalDevice`;
- one `VkDevice`;
- one compute-capable queue-family index;
- one `VkQueue` from that family;
- one command pool tied to that family;
- cached `VkPhysicalDeviceProperties` and
  `VkPhysicalDeviceMemoryProperties`;
- the selected strict-local memory policy;
- configured and committed byte counters;
- live tensor and retained-operation registries;
- one mutex protecting the queue, command pool, registries, and accounting;
- a state such as UNINITIALIZED, READY, DEGRADED, SHUTTING_DOWN, or RETAINED.

### Device selection

For this milestone, enumerate all devices and require exactly one
`vendorID=0x1002`, `deviceID=0x67b1` match. Zero matches is unavailable; more
than one match is unsupported and fails rather than silently enabling
multi-GPU behavior. Device enumeration and selection must be deterministic.

Accepting the entire Hawaii device-ID family is a later approval decision. The
target server and validated hardware are specifically the R9 390 at `0x67b1`.

### Queue selection

Select the lowest-index compute-capable queue family and obtain one queue from
it. Compute-capable queues support transfer operations, so one family can own
both uploads and future expert compute without queue-family ownership transfers.
A transfer-only family should not be chosen now if it would force cross-family
ownership transfers later.

### Command pool

Create one persistent command pool after device creation, using the selected
queue family and flags suitable for transient, individually reset command
buffers. The current helper may continue to allocate one command buffer per
upload, but must never create or destroy the pool per upload.

### External synchronization

The current helper explicitly assigns queue and command-pool synchronization to
the caller. The context mutex satisfies both rules:

- lock before command-buffer allocation, submission, wait, finish, or free;
- serialize all `vkQueueSubmit()` calls;
- serialize command-pool allocation/free/reset operations;
- keep tensor allocation/accounting and retained-operation publication under
  the same lock.

Host expert reads remain parallel. The upload phase starts only after each host
expert triplet is complete and runs serially on the main thread. Holding the
mutex across the bounded fence wait is acceptable for the one-GPU initial
implementation and prevents command-pool races.

Do not use `vkDeviceWaitIdle()`: it has no bounded timeout. Every backend-owned
submission must have a tracked fence. Once every tracked fence has signaled,
the backend can prove its submitted work is complete without an unbounded
device-wide wait.

## 4. Proposed Vulkan tensor/buffer ownership and destruction model

### Tensor representation

Phase 3A deliberately implements a raw byte-transport tensor rather than
claiming that byte counts alone establish a valid `QT`. Its internal tensor
contains:

- owning context pointer and context generation;
- `VkBuffer` and `VkDeviceMemory`;
- logical payload size and actual allocation size;
- weight offset/length and scale offset/length;
- memory type and heap indices plus recorded property flags;
- state: ALLOCATED, PENDING, READY, FAILED, or DESTROY_PENDING;
- a destroy-requested bit.

Format semantics are deferred to Phase 3B because Phase 3A does not inspect or
own `QT`, safetensor metadata, or loader geometry. Adding `fmt`, `I`, `O`, and
`gs` fields in Phase 3A without validating them against the real source tensors
would produce decorative metadata and could falsely bless invalid byte ranges.

Before any Phase 3B loader integration, add a format-aware specification and
validator in `backend_vulkan.h`/`backend_vulkan.c`. The specification must carry
`fmt`, `I`, `O`, `gs`, derived scale count, weight source length, and scale
source length. The validator must compute and overflow-check the exact format
layout and reject unsupported formats or inconsistent source lengths. The only
production tensor-creation entry point exposed to loader code must accept that
validated specification. The Phase 3A raw byte-upload declaration is guarded
by `COLI_VULKAN_INTERNAL` or `COLI_VULKAN_TESTING`; therefore ordinary Phase 3B
loader translation units cannot compile a direct call to it.

Use one destination buffer/allocation per `QT`, with weights and scales packed at
explicit aligned offsets. This mirrors the existing one-handle-per-`QT` CUDA
ownership model while avoiding two Vulkan allocations per tensor. Gate, up,
and down therefore form three tensor allocations per resident expert.

The Phase 3B validator must compute layout by format, not by blindly treating
`qt_bytes()` as a contiguous source range:

- format 0 has float weights and no quantization scales;
- formats 1, 2, and 3 have per-row scales;
- format 4 has `O * ceil(I/gs)` scales;
- format 5 has int3 group geometry but no current GPU kernel;
- format 6 keeps effective scales inside its weight blocks; its `.qs` value is a
  format tag and should not be treated like an ordinary scale array.

Destination buffers should include `VK_BUFFER_USAGE_TRANSFER_DST_BIT` and
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` so the lifecycle does not require replacing
allocations when compute is introduced.

### Ownership

After integration, `QT` is the logical owner of its `ColiVulkanTensor *`. The
context registry tracks allocations, accounting, and safety state, but does not
make READY tensor handles ownerless. A pending registry entry temporarily owns
the retained staging operation and keeps a reference to the destination tensor.

Whole-`ESlot` swaps naturally move tensor handles with the gate/up/down `QT`
values. Copying individual `QT` values without explicit ownership transfer must
remain prohibited.

### Transactional expert publication

An expert is Vulkan-resident only after all three tensors are ready:

1. Create gate, up, and down tensors into local temporary handles.
2. Upload them serially.
3. If all three reach READY, attach all three to the target `ESlot` in one
   publication step.
4. If any fails, destroy all already-complete temporary tensors.
5. If one is pending, mark it for deferred destruction and do not submit more
   work until it is finished.
6. Leave the `ESlot` entirely host-backed and Vulkan-ineligible on rollback.

A partially uploaded expert must never be visible to `moe()`.

### Destruction order

For a tensor with no pending work:

1. Remove it from the live registry.
2. Destroy its `VkBuffer`.
3. Free its `VkDeviceMemory`.
4. Decrement actual committed-byte accounting.
5. Clear the owner handle.

The context may destroy its command pool, device, and instance only after:

- every model-owned tensor has been released;
- every pending operation has completed and cleaned up;
- every backend-owned submission fence has signaled.

The initial integration must keep host expert copies. Do not call or generalize
`expert_host_release()` for Vulkan until Vulkan compute and CPU
fallback/rematerialization are both implemented and tested.

## 5. Tracking and safely finishing timeout-retained `VulkanUploadOp` objects

The context needs a bounded pending registry. Each entry contains:

- the complete `VulkanUploadOp`;
- the destination `ColiVulkanTensor *`;
- submission timestamp or sequence;
- whether the tensor owner requested destruction;
- the last finish result.

### State transitions

- **Pre-submit failure:** clean staging/command/fence resources immediately,
  destroy the unexposed destination tensor, and create no pending entry.
- **Upload success:** clean temporary upload resources and mark the tensor READY.
- **Timeout:** copy the populated `VulkanUploadOp` into the registry and mark the
  tensor PENDING.
- **Post-submit non-timeout fence failure:** retain the populated operation,
  mark the tensor PENDING/FAILED, and mark the context DEGRADED. The operation
  may still be in flight.
- **Free while pending:** mark the tensor DESTROY_PENDING; do not destroy its
  buffer or memory.
- **Finish success:** call `vulkan_upload_finish()` under the context mutex,
  remove the registry entry, and either mark the tensor READY or destroy it if
  destruction was requested.
- **Finish timeout/failure:** leave the registry entry and every referenced
  Vulkan object intact.

Allow at most one retained upload at a time. Once an upload becomes pending,
stop submitting new uploads until it finishes. This prevents an arbitrary
number of staging allocations from accumulating after repeated timeouts.

The helper copies source data into staging memory before submission, so the
original host expert slab may be reused after the helper returns. The retained
staging allocation, not the host expert pointer, owns the in-flight source data.

### Shutdown

`coli_vulkan_shutdown()` should accept or use a finite overall deadline and
retry each pending fence in bounded slices. It must not call
`vkDeviceWaitIdle()`.

If all operations finish, release tensors, command pool, device, and instance in
order. If any operation cannot be proven complete, return an incomplete status
and intentionally retain the context and Vulkan resources for process
reclamation. Destroying a command pool, destination buffer, device, or instance
while such an operation may be pending is forbidden.

## 6. Strict DEVICE_LOCAL memory selection and verification

Destination memory selection occurs only after `vkCreateBuffer()` and
`vkGetBufferMemoryRequirements()` establish `memoryTypeBits` and the actual
allocation size.

For each eligible memory type, require:

```c
(propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
(propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0
```

Then verify:

- the chosen type is allowed by `memoryTypeBits`;
- its `heapIndex` is less than `memoryHeapCount`;
- the referenced heap has `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`;
- `VkMemoryRequirements.size` is nonzero;
- the actual allocation fits the remaining configured budget;
- the recorded selected flags still exclude HOST_VISIBLE.

The server has three heaps and seven memory types. Heap 2 is 268435456 bytes,
the 256 MB PCIe BAR. Memory types 3 and 4 reference heap 2 with
`propertyFlags=0x0007`, which includes DEVICE_LOCAL, HOST_VISIBLE, and
HOST_COHERENT. Those types must be rejected for expert destination buffers.

The selector must distinguish three categories:

1. strict DEVICE_LOCAL and non-HOST_VISIBLE VRAM: valid expert destination;
2. DEVICE_LOCAL and HOST_VISIBLE BAR memory: invalid expert destination;
3. HOST_VISIBLE system memory: staging only.

Never select expert memory by heap size, heap number, or first-match ordering
without checking the buffer's type mask and flags.

### Bounded VRAM accounting

Charge `VkMemoryRequirements.size`, including Vulkan alignment, rather than
logical tensor bytes. Maintain accounting under the context mutex.

The effective cap should be:

```text
min(explicit user cap, strict-local heap size - approved reserve)
```

The initial implementation should require an explicit positive cap such as
`VULKAN_EXPERT_MB`; zero or unset means no expert residency. There should be no
automatic fill-all-VRAM behavior. A triplet must reserve enough budget for all
three actual allocations or roll back without publication.

## 7. Smallest safe Phase 3 implementation milestone

The smallest safe milestone is **Phase 3A: persistent context and Vulkan tensor
lifecycle only**.

It should implement:

- one persistent R9 390 context;
- exact deterministic device selection;
- one persistent queue and command pool;
- strict destination-buffer creation/allocation;
- overflow-checked aligned raw weight/scale transport, exposed only to backend
  internals and Phase 3A tests;
- bounded actual-allocation accounting;
- upload, readback, and tensor destruction;
- one-entry retained-operation registry;
- bounded finish and shutdown behavior;
- deterministic unit tests and the real Hawaii hardware test.

It should not yet:

- add Vulkan fields to `QT`;
- modify `pin_load()` or `moe()`;
- attach Vulkan resources to the model;
- release host copies;
- claim expert inference acceleration.

This milestone proves the lifetime foundation independently before the real
loader creates long-lived objects.

After Phase 3A passes, a separately reviewed Phase 3B may connect the startup
`pin_load()` path using transactional gate/up/down uploads while retaining all
host copies. That would validate real expert-loader residency but would still
not accelerate inference until Vulkan compute exists. Phase 3B is blocked until
the format-aware specification/validator described in section 4 is implemented
and its public API is the sole tensor-creation path available to loader code.

## 8. Features explicitly deferred to later phases

- Vulkan matmul kernels.
- Fused Vulkan gate/up/SiLU/down expert kernels.
- Vulkan dispatch from `moe()` or `matmul_qt_ex()`.
- Activation upload/download and persistent activation buffers.
- Descriptor layouts, descriptor pools, pipelines, shader modules, and pipeline
  caches.
- Releasing host expert slabs and CPU rematerialization after Vulkan failure.
- Uploading streaming misses or ordinary LRU slots.
- Vulkan `REPIN` and in-place tensor refresh.
- Multiple GPUs, device lists, peer access, or cross-device routing.
- Multiple queues or transfer/compute queue-family ownership transfers.
- Dense tensors, attention, shared experts, MTP, router, KV, or residual-pipeline
  acceleration.
- Persistent staging rings, staging suballocation, or chunked transfers.
- Destination-memory arenas, suballocation, compaction, or defragmentation.
- Required use of `VK_EXT_memory_budget`.
- Full Hawaii-family or non-Hawaii device support.
- Windows Vulkan dynamic loading.
- Generic CUDA/HIP/Metal/Vulkan backend-interface refactors.
- GLM-5.2 execution or benchmarking; `benchmarks.glm_5_2` remains exactly
  `NOT_RUN`.

## 9. Unit, hardware, failure-path, and regression tests

### Unit tests

Add deterministic tests for:

- zero, one, and multiple matching physical devices;
- exact `0x1002/0x67b1` acceptance;
- non-AMD and non-Hawaii rejection;
- stable device and queue-family selection;
- the real three-heap/seven-type memory fixture;
- rejection of heap-2 types 3 and 4 with flags `0x0007`;
- distinction among strict VRAM, BAR, and HOST_VISIBLE system memory;
- `memoryTypeBits` filtering;
- missing strict-local memory;
- zero size, null data, null output, and invalid properties;
- `VkDeviceSize` to `size_t` overflow;
- raw byte-layout alignment and arithmetic overflow in Phase 3A;
- in Phase 3B, layout arithmetic, derived scale counts, source lengths, and
  unsupported-format rejection for every quantization format before loader
  integration is permitted;
- actual allocation-size budget accounting;
- three-tensor transactional reservation and rollback;
- context initialization and partial-initialization cleanup;
- repeated init or shutdown calls;
- tensor create, upload, free, and accounting return to zero;
- one instance/device/queue/command-pool creation per context;
- free while pending;
- context behavior after a post-submit failure;
- shutdown with no work, completed work, timeout, and retained work.

Use a small test-only Vulkan dispatch table or equivalent link-time fake so each
failure can be injected deterministically. Real fence timing cannot reliably
cover timeout and post-submit error paths.

### Real R9 390 hardware tests

On the target server:

1. Initialize the public persistent context API.
2. Verify selected vendor/device `0x1002/0x67b1`.
3. Verify three heaps and seven memory types.
4. Verify heap 2 is exactly 268435456 bytes.
5. Verify types 3 and 4 reference heap 2 with flags `0x0007`.
6. Allocate the destination from a strict DEVICE_LOCAL, non-HOST_VISIBLE type.
7. Allocate staging from HOST_VISIBLE memory.
8. Upload deterministic byte patterns at several nonzero sizes.
9. Read each pattern back and compare every byte exactly.
10. Repeat upload/free operations through the same instance, device, queue, and
    command pool.
11. Confirm the committed-byte and tensor counters return to zero.
12. Confirm normal bounded shutdown leaves no retained operation.

The test must use small synthetic buffers and must not load or run GLM-5.2.

### Failure-path tests

Inject failures at every resource transition:

- instance creation;
- physical-device enumeration, including `VK_INCOMPLETE` handling;
- device and command-pool creation;
- destination buffer creation;
- memory-requirement validation;
- strict memory-type selection;
- budget reservation;
- device-memory allocation and binding;
- staging buffer/memory creation and mapping;
- command-buffer allocation, begin, and end;
- fence creation;
- queue submission;
- initial fence timeout;
- post-submit non-timeout fence failure;
- repeated finish timeout;
- eventual finish success;
- destruction requested while pending;
- shutdown deadline expiration.

For every pre-submit failure, assert complete cleanup and accounting rollback.
For every post-submit timeout/failure, assert intentional retention and no
destruction of in-flight resources.

### Regression tests

- Default CPU build and complete `make test`.
- Default engine must not link Vulkan.
- Build-time Vulkan opt-in plus runtime-off behavior.
- CUDA build/tests where CUDA is available.
- HIP build/tests where ROCm is available.
- Metal build/tests on macOS.
- A binary without Vulkan support must give a clear error if runtime Vulkan is
  requested.
- `make vulkan-test` must return nonzero and stop after the first failing test;
  it must never print a false overall PASS.
- No changes to expert selection, host cache behavior, CPU arithmetic, CUDA/HIP
  dispatch, or Metal zero-copy paths while Vulkan is off.
- No GLM-5.2 run; benchmark fields remain exactly `NOT_RUN`.

## 10. PASS, FAIL, and SKIP criteria

### PASS

- All pure and fault-injection unit tests pass.
- The target hardware selects exactly the R9 390 at `0x1002/0x67b1`.
- Hardware topology matches three heaps and seven memory types.
- Heap 2 and BAR types match the validated host capture.
- Every expert destination allocation is DEVICE_LOCAL and non-HOST_VISIBLE.
- Byte-exact readback succeeds for every tested size.
- Context objects are created once and reused for every upload.
- Actual committed bytes never exceed the explicit cap.
- Tensor counters and bytes return to zero after normal destruction.
- Normal shutdown completes within its deadline with no retained operation.
- CPU, CUDA, HIP, and Metal builds and tests remain unchanged when Vulkan is
  disabled.
- Test target failures propagate as nonzero status.

### FAIL

- The wrong device is selected or multiple matching GPUs are silently accepted.
- A BAR or HOST_VISIBLE system type is selected for an expert destination.
- A memory type is selected outside the destination buffer's `memoryTypeBits`.
- Per-upload instance, device, queue, or command-pool creation occurs.
- Any pending command buffer, fence, staging buffer, destination buffer, command
  pool, device, or instance is destroyed before completion is proven.
- Shutdown uses an unbounded device-wide wait.
- Size/accounting arithmetic can overflow or underflow.
- Actual allocations can exceed the configured budget.
- A partial gate/up/down triplet becomes visible.
- Host copies are released before a tested Vulkan compute/fallback path exists.
- Vulkan changes default CPU/CUDA/HIP/Metal behavior.
- A failed test yields a successful Make target.
- GLM-5.2 is run or its benchmark status changes from `NOT_RUN`.

### SKIP

On generic CI or non-target developer machines, the hardware test may SKIP with
an explicit reason when:

- Vulkan headers or libraries are unavailable;
- no Vulkan loader/driver is present;
- no AMD Hawaii device exists.

Unit tests must not SKIP. On the designated R9 390 server, a missing/wrong
device, missing strict-local type, or unexpected memory topology is FAIL, not
SKIP.

## 11. Unresolved architectural decisions requiring approval

### Decision 1: Phase 3 boundary

Approve **Phase 3A as a foundation-only context/tensor-lifecycle milestone**
before modifying `colibri.c`.

Recommendation: approve. Direct expert integration is not yet safe.

### Decision 2: Device policy

Choose between:

- exact `0x1002/0x67b1` enforcement for the validated R9 390; or
- accepting every Hawaii-family ID already listed in the tests.

Recommendation: require exactly one `0x1002/0x67b1` device in Phase 3A. Expand
support only after separate hardware validation.

### Decision 3: VRAM cap and reserve

Approve the runtime configuration and conservative reserve. Proposed policy:

- compile-time `VULKAN=1`;
- runtime `COLI_VULKAN=1`;
- explicit positive `VULKAN_EXPERT_MB`;
- unset/zero expert budget means no expert residency;
- no `auto` or fill-all-VRAM mode.

The exact reserve must be approved before expert pinning. It should leave room
for future pipelines, descriptors, activations, and driver allocations.

### Decision 4: Initial allocation strategy

Choose one allocation per `QT` versus an arena/suballocator.

Recommendation: one buffer/allocation per `QT` for the small bounded validation
milestone. Defer arena design until real allocation counts and compute descriptor
requirements are known.

### Decision 5: Backend coexistence

Approve rejecting simultaneous runtime Vulkan with CUDA/HIP or Metal, and
rejecting `REPIN>0` while Vulkan expert residency is enabled.

Recommendation: reject these combinations in the initial implementation rather
than define ambiguous dispatch and synchronization semantics.

### Decision 6: Phase 3B upload-only integration

Decide whether to connect `pin_load()` before Vulkan expert compute exists.

An upload-only Phase 3B can validate that real expert gate/up/down bytes become
persistent strict-local buffers, but it provides no inference acceleration and
must retain all host copies. It must be described as residency validation, not
as a Vulkan inference backend. It must first add and test the format-aware
`fmt`/`I`/`O`/`gs` specification, derived scale-count validation, exact source
length validation, and format-specific layout computation. Loader code must use
that public validated API; the Phase 3A raw transport remains internal/test-only.

Recommendation: keep this separate from Phase 3A and approve it only if that
validation evidence is specifically desired before compute development.

### Decision 7: Engine shutdown structure

Choose between refactoring post-model-init returns through one explicit cleanup
path or registering a Vulkan-specific exit handler.

Recommendation: add explicit `model_vulkan_release()` followed by bounded
context shutdown through a centralized cleanup path, with an exit handler only
as a last-resort fallback. An exit handler must intentionally retain the
context if completion cannot be proven.

### Decision 8: Future tensor update semantics

Choose whether future Vulkan `REPIN` uses in-place updates or allocate-new
replacement.

Recommendation: allocate new tensors, finish their uploads, atomically swap the
complete triplet, then destroy the old tensors. In-place refresh is unsafe until
all possible compute readers and uploads have explicit completion tracking.

### Decision 9: Compute scope

Approve the rule that no Vulkan handle becomes eligible for `moe()` until a
complete compute contract exists for supported formats, descriptors,
activation transfers, bounded fences, and CPU fallback.

Recommendation: approve. Weight residency by itself is not an execution
backend.

## Final conclusion

A persistent Vulkan context and buffer/tensor lifecycle is a mandatory Phase 3
prerequisite. The validated staged-upload helper should remain the low-level
copy primitive beneath that owner. Direct insertion into `expert_load_impl()`,
`pin_load()`, or `moe()` before context ownership, actual-allocation budgeting,
retained-operation tracking, transactional triplet publication, and bounded
shutdown are implemented would be unsafe.

Proceed only after approval of the decisions in section 11, beginning with the
bounded Phase 3A foundation. No GLM-5.2 benchmark is part of this phase, and
`benchmarks.glm_5_2` must remain exactly `NOT_RUN`.
