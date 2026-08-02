# Phase 4A Vulkan routed-down compute architecture

## Status, base, and conclusion

This is an architecture-only design against merged Phase 3B commit
`eabc956e22e492e3cfeeac84847396be917e7c5f`. No Phase 4A implementation,
shader generation, shader compilation, model generation, or inference was
performed while preparing it.

The smallest safe end-to-end compute milestone is one format-2 (per-row INT4)
routed-expert **down projection** per selected startup-pinned expert. The
existing CPU code continues to calculate gate projection, up projection, SiLU,
and the gate/up product. Vulkan consumes that product and calculates:

```text
hh[nr, hidden] = gg[nr, moe_inter] * down[hidden, moe_inter]^T
```

This is real routed-expert arithmetic in the production `moe()` path. It is a
narrower seam than a complete gate/up/SiLU/down Vulkan MLP, but it proves all
new end-to-end properties required here: routing chooses an expert; that
expert's startup-resident strict-local weight buffer is bound to a descriptor;
live activations reach a compute shader; a real dispatch produces expert
output; the result returns to the CPU residual accumulation; and the engine
generates deterministic tokens.

Gate, up, and down do **not** all need to run through Vulkan for this proof.
Moving all three would require three dispatches or a fused kernel, two weight
inputs, a larger intermediate lifecycle, SiLU parity, and substantially more
rollback and fault surface. Those additions do not prove the first Vulkan
dispatch more convincingly than routed-down alone.

Phase 4A remains a deliberately narrow validation mode:

- exactly one AMD `0x1002/0x67b1` device;
- only main-model sparse experts, all of which must be startup-pinned and
  Vulkan-resident before inference;
- only format 2 and only the down-projection geometry used by `c/glm_tiny`;
- exact-float CPU kernels (`IDOT=0`) for the comparison run;
- at most 64 activation rows per dispatch;
- synchronous, bounded, one-queue submission;
- no CPU down-projection fallback after Vulkan compute is explicitly requested.

Phase 3B upload-only mode and default CPU mode remain independent and unchanged.

## 1. Exact existing call paths and insertion points

### 1.1 Phase 3B startup residency path

The merged startup path is:

```text
c/colibri.c:main()
  -> model_init()
  -> coli_vulkan_context_create()                 [only when COLI_VULKAN=1]
  -> pin_load()
       -> expert_load()                           [parallel host loading]
       -> pin_vulkan_upload_prefix()
            -> expert_vulkan_upload_triplet()
                 -> qt_vulkan_spec()
                 -> coli_vulkan_validate_qt_spec() for gate, up, down
                 -> coli_vulkan_tensor_create_qt() for gate, up, down
                      -> tensor_upload_raw()
                           -> create_destination_locked()
                           -> staged_upload_with_api()
       -> publish the complete gate/up/down triplet
  -> vulkan_finalize_startup()
```

At `eabc956`, the relevant locations are:

- `c/colibri.c:7401`: `model_init(&m, ...)`;
- `c/colibri.c:7403-7424`: persistent context creation and bounded exit-handler
  registration;
- `c/colibri.c:6678`: `pin_load()`;
- `c/colibri.c:6767-6776`: host expert loading followed by the Vulkan prefix
  upload;
- `c/colibri.c:6427-6519`: spec creation, triplet validation, upload, READY
  checks, and transactional publication;
- `c/colibri.c:6545-6584`: final residency accounting and the upload-only READY
  banner;
- `c/backend_vulkan.c:694-890`: instance, exact device, device, queue, and
  command-pool creation;
- `c/backend_vulkan.c:925-1026`: strict-local tensor buffer/allocation creation;
- `c/backend_vulkan.c:1075-1219`: raw upload and validated QT creation.

Phase 4A reuses this context and these resident tensor objects. It must not
create another instance, physical-device selection, logical device, queue, or
command pool.

### 1.2 Current generation-to-expert path

The deterministic oracle/generation path is:

```text
c/colibri.c:main()
  -> generate()
       -> step()                                  [prompt]
            -> layers_forward()
                 -> layers_forward_rows()
                      -> layer_forward_rows()
                           -> moe()                [sparse layers]
       -> spec_decode()
            -> step_all()                         [one row with DRAFT=0]
                 -> layers_forward()
                      -> ... -> moe()
```

Exact merged locations are:

- `c/colibri.c:7630`: `generate()` invocation;
- `c/colibri.c:5081-5087`: `generate()`;
- `c/colibri.c:4494-4528`: `step()`;
- `c/colibri.c:4924-4927`: decode calls `step_all()`;
- `c/colibri.c:4356-4443`: layer traversal;
- `c/colibri.c:4276-4351`: normalization, attention, and sparse `moe()` call;
- `c/colibri.c:2870`: `moe()`.

### 1.3 Current routed-expert calculation

Within `moe()`:

1. Phase A routes each input row and builds `idxs`, weights, and `keff`.
2. Phase B builds the unique expert list.
3. Phase C/D resolves each expert in this order:
   startup pin, LRU cache, then working-set/disk miss
   (`c/colibri.c:3240-3249`).
4. For each resolved expert, it gathers that expert's input rows into `xg`
   (`c/colibri.c:3528-3542`).
5. The ordinary non-CUDA/non-Metal path performs:

```text
expert_gate_up(gg, uu, xg, &e->g, &e->u, nr)
gg[z] = silu(gg[z]) * uu[z]
matmul_qt(hh, gg, &e->d, nr)
out[row,d] += routing_weight * hh[r,d]
```

These operations are at `c/colibri.c:3554-3559`. `expert_gate_up()` is defined
at `c/colibri.c:289-295`, and `matmul_qt_ex()` dispatches CPU formats at
`c/colibri.c:614-651`.

The optional `XEXP=1` block at `c/colibri.c:3445-3511` can bypass the ordinary
per-expert loop. Phase 4A compute mode must reject `XEXP!=0` during startup so
that explicit compute cannot silently take this CPU-only shortcut.

CUDA fallback copies of the same arithmetic exist later in `moe()`, but a
Vulkan build is already mutually exclusive with CUDA/HIP/Metal. Phase 4A does
not edit those guarded paths.

### 1.4 Exact Phase 4A insertion

Add a small `vulkan_expert_down_or_die()` helper under `#ifdef COLI_VULKAN` and
replace only the call at merged line 3557 with:

```c
if (m->vulkan_compute_requested)
    vulkan_expert_down_or_die(m, layer, eid, e, hh, gg, nr);
else
    matmul_qt(hh, gg, &e->d, nr);
```

The helper calls an opaque backend API with `e->d.vulkan`. It never receives a
raw `VkBuffer`, device, queue, descriptor, or command pool. It requires the
handle to be READY, compute-eligible, format 2, strict-local, and owned by the
model's one context.

`matmul_qt()`, `matmul_qt_ex()`, `expert_gate_up()`, routing, LRU, misses,
working-set publication, shared experts, attention, and dense MLPs remain
unchanged. This prevents a general QT call from accidentally dispatching dense
or non-pinned tensors.

## 2. Exact files and functions to modify

### `c/backend_vulkan.h`

Add:

- `ColiVulkanComputeConfig`, with fixed-capacity row/input/output dimensions;
- compute counters in `ColiVulkanContextInfo`;
- `compute_eligible` as a real tensor property rather than the current constant
  zero returned by `coli_vulkan_tensor_get_info()`;
- `coli_vulkan_compute_prepare()`;
- `coli_vulkan_tensor_matmul_fmt2()`;
- explicit output-validity, timeout-retention, thread-safety, and shutdown
  contracts for both APIs;
- fake entry points required for shader modules, descriptor objects, pipelines,
  memory barriers, push constants, descriptor binding, pipeline binding, and
  `vkCmdDispatch` in `ColiVulkanApi`.

Do not expose getters for `VkDevice`, `VkQueue`, `VkBuffer`, `VkDeviceMemory`,
or `VkCommandPool`.

### `c/backend_vulkan.c`

Modify:

- `VulkanApi` and `g_real_api` to contain the new Vulkan entry points;
- `api_complete()` to require them only for a compute-prepared context, so a
  Phase 3B upload-only context retains its existing requirements;
- `PendingKind` / `PendingOperation` to represent a submitted compute operation;
- `ColiVulkanTensor` to persist an actual compute-eligibility bit;
- `ColiVulkanContext` to own one optional compute state and its counters;
- `coli_vulkan_tensor_create_qt()` to mark only the narrowly eligible format-2
  tensors;
- `finish_pending_locked()` to finish a retained compute fence without changing
  an upload tensor's state;
- `coli_vulkan_context_get_info()` to publish exact compute counters;
- `coli_vulkan_context_destroy()` to destroy proven-idle compute objects before
  the command pool/device and to retain everything after an unproven result.

Add private helpers for checked compute geometry, system-host-visible scratch
memory selection, buffer creation/mapping, pipeline creation, command recording,
submitted-operation retention, and reverse-order cleanup.

### `c/colibri.c`

Modify only Vulkan-guarded integration points:

- update the `QT.vulkan` comment from upload-only to resident opaque Vulkan
  tensor ownership;
- extend the Vulkan fields in `Model` with compute-mode state and telemetry;
- parse `COLI_VULKAN_COMPUTE` and `COLI_VULKAN_VALIDATE` in `main()`;
- after the existing persistent context is created, call
  `coli_vulkan_compute_prepare()` only for explicit compute mode;
- strengthen `vulkan_finalize_startup()` for compute-mode complete coverage and
  format/geometry checks;
- add `vulkan_expert_down_or_die()`;
- add the one conditional at the ordinary routed-down call at merged line 3557;
- print the compute READY and final compute-summary telemetry;
- leave `model_vulkan_release()` as the single model cleanup entry, relying on
  context destruction for compute-owned objects.

Do not add Vulkan branches to `matmul_qt()`, `matmul_qt_ex()`, dense MLP,
shared-expert MLP, attention, routing, LRU lookup, working-set lookup, miss
loading, or REPIN.

### Shader files

Add during implementation, not during this design task:

- `c/shaders/vulkan_qt_fmt2_down.comp`: canonical reviewed GLSL source;
- `c/shaders/vulkan_qt_fmt2_down_spv.h`: deterministic checked-in SPIR-V 1.0
  words used by production, with generator version and source SHA-256 in its
  header.

The normal executable must not compile GLSL at runtime and must not depend on a
runtime shader compiler.

### Tests and build files

Modify/add:

- `c/Makefile`: shader verification, strict build, unit, sanitizer, strict
  hardware, and compute smoke targets;
- `c/tests/test_vulkan_compute.c`: pure eligibility/arithmetic plus fake Vulkan
  lifecycle and fault tests;
- `c/tests/test_backend_vulkan.c`: strict real-Hawaii compute cases in addition
  to existing upload/readback coverage;
- `c/tests/test_vulkan_compute_smoke.py`: CPU, upload-only, and compute-enabled
  four-token acceptance;
- `c/tests/test_vulkan_context.c`: pending-kind/context accounting extensions
  where shared ownership invariants belong;
- `c/tools/clean.py`: new test binaries only.

`c/backend_cuda.*`, `c/backend_loader.c`, `c/backend_metal.*`, `c/st.h`, and the
default CPU Makefile path require no behavior change.

## 3. Selected QT format and rationale

### 3.1 Format

Phase 4A compute supports only QT format 2:

```text
weights: O rows, ceil(I/2) bytes per row
value i even: low nibble
value i odd:  high nibble
signed value: int(nibble) - 8        # range -8 through 7
scale: one IEEE binary32 value per output row
y[s,o] = scale[o] * sum(i=0..I-1, x[s,i] * signed_weight[o,i])
```

This exactly matches `matmul_i4()` in `c/quant.h:125-166` and runtime
quantization in `qt_alloc()` / `pack_int4()`.

### 3.2 Why format 2

- The Phase 3B tiny smoke invokes `./colibri 64 4 4`, so startup experts are
  converted to format 2 by `qt_alloc()` (`c/colibri.c:874-895`).
- `c/glm_tiny` uses `hidden=128`, `moe_inter=32`, two sparse layers, eight
  experts, and top-2 routing. A down tensor is therefore `[O=128,I=32]`.
- Format 2 uses integer bit extraction and binary32 arithmetic only. It does not
  require shader `Int8`, 8-bit storage, 16-bit floats, subgroups, atomics, or a
  vendor extension.
- The per-row scale makes a single-output-row invocation independent; no shared
  reduction or cross-invocation synchronization is needed.
- Formats 0, 1, and 3-6 add no value to the first end-to-end proof. Format 4/5/6
  in particular add grouped or in-block scale semantics and should follow only
  after the basic lifecycle is accepted.

### 3.3 Compute eligibility

Residency eligibility remains formats 0 through 6. Compute eligibility is a
separate property and is true only when all of these hold:

- `fmt == 2`, `gs == 0`;
- the Phase 3B validator accepts exact `I`, `O`, weight bytes, scale bytes, and
  packed layout;
- `I` and `O` fit `uint32_t` and are nonzero;
- `I % 8 == 0`, so one shader `uint` contains exactly eight weights and no row
  crosses a word boundary;
- `weight_offset == 0`;
- `weight_size == O * (I / 2)`;
- `scale_offset % 4 == 0` and `scale_size == O * 4`;
- the allocation is strict `DEVICE_LOCAL` and non-`HOST_VISIBLE`;
- `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` was included, as Phase 3B already does;
- all checked products and descriptor ranges fit `size_t`, `VkDeviceSize`,
  `uint32_t`, and `maxStorageBufferRange`.

For the Phase 4A Colibri seam, require exactly `I == m->c.moe_inter` and
`O == m->c.hidden`. Gate and up remain resident but are not bound by this
milestone.

## 4. Shader/kernel arithmetic and data layout

### 4.1 Descriptor layout

One persistent descriptor set has three storage-buffer bindings:

| Binding | Buffer | Access | Layout |
| --- | --- | --- | --- |
| 0 | selected resident packed QT | shader read-only | `uint packed[]` |
| 1 | persistent activation scratch | shader read-only | `float x[]` |
| 2 | persistent result scratch | shader write-only | `float y[]` |

The packed QT descriptor covers exactly `tensor->layout.packed_size`. The scale
payload remains in the same Phase 3B allocation; no scale copy or second expert
allocation is introduced.

### 4.2 Push constants

Use a 24-byte push-constant record of six `uint32_t` values:

```text
I, O, rows, row_words, scale_word_offset, total_outputs
```

All fields are derived and overflow-checked on the host. The backend verifies
`maxPushConstantsSize >= 24`.

### 4.3 Invocation mapping

The shader uses `local_size_x = 64`. One invocation owns one complete output
element:

```text
linear = gl_GlobalInvocationID.x
if linear >= rows * O: return
s = linear / O
o = linear % O
acc = 0.0
for i in 0 .. I-1:
    word = packed[o * row_words + i / 8]
    nibble = (word >> (4 * (i % 8))) & 0xf
    q = int(nibble) - 8
    acc += x[s * I + i] * float(q)
y[linear] = acc * uintBitsToFloat(packed[scale_word_offset + o])
```

There is no atomic operation and no parallel reduction. A given element's
accumulation order is fixed from `i=0` to `I-1`, making repeated executions on
the same driver deterministic. The shader must not use relaxed precision.

Dispatch group count is:

```text
groups_x = ceil((rows * O) / 64)
```

The backend checks `groups_x > 0` and
`groups_x <= maxComputeWorkGroupCount[0]`, as well as a 64-invocation workgroup
against `maxComputeWorkGroupInvocations` and `maxComputeWorkGroupSize[0]`.

### 4.4 Portability boundary

Phase 4A targets the little-endian x86-64 server and the exact R9 390. The
32-bit word view avoids `VK_KHR_8bit_storage`. A non-little-endian build must
fail the compute feature at compile time or preparation time rather than
reinterpret packed bytes differently. Upload-only residency stays portable.

Use Vulkan 1.0 / SPIR-V 1.0 core operations only. Device creation must not add
optional features. Pipeline preparation checks the core storage-buffer,
descriptor, workgroup, push-constant, and range limits before creating objects.

## 5. Activation and result buffer lifecycle

### 5.1 Preparation

When and only when `COLI_VULKAN_COMPUTE=1`, `main()` calls:

```text
coli_vulkan_compute_prepare(context,
    max_rows=64,
    max_input_width=m.c.moe_inter,
    max_output_width=m.c.hidden)
```

This occurs after `model_init()` and persistent context creation, but before
`pin_load()`. Preparing first makes its two memory allocations visible to
`maxMemoryAllocationCount` accounting before the residency prefix is chosen.

The compute state owns two buffers and two allocations for the lifetime of the
context:

- input: `64 * moe_inter * sizeof(float)`;
- output: `64 * hidden * sizeof(float)`.

Every multiplication and alignment is overflow checked. The buffers use
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. Their memory must be
`HOST_VISIBLE | HOST_COHERENT` and explicitly **not** `DEVICE_LOCAL`, so the
256 MiB BAR is not selected for scratch and no ReBAR capacity is assumed.
Failure to find such a compatible memory type is `COLI_VULKAN_UNSUPPORTED` and
fails explicit compute startup.

Both allocations are persistently mapped only after successful creation and
binding. Partial preparation unwinds in reverse order because nothing has been
submitted yet.

### 5.2 Per-dispatch use

For `rows=nr`:

1. Validate `1 <= rows <= 64` and the input/output byte ranges.
2. Copy `rows * I` floats from CPU `gg` to the mapped input buffer.
3. Update the idle descriptor set with the selected tensor and the two scratch
   buffers.
4. Record a host-write-to-compute-read buffer barrier.
5. Bind pipeline, descriptor set, and push constants; record exactly one
   `vkCmdDispatch`.
6. Record a compute-write-to-host-read buffer barrier.
7. Submit and wait on a fence with the configured finite deadline.
8. Only after `VK_SUCCESS`, copy `rows * O` floats from mapped output to `hh`.

The output pointer is valid only on `COLI_VULKAN_OK`. On timeout, error, or
device loss, the caller must treat all of `hh` as uninitialized.

No activation or result allocation occurs per dispatch. No expert host pointer
is freed, and the host gate/up/down copies remain authoritative for validation
and for runs where compute is disabled.

## 6. Pipeline, descriptor, command-buffer, fence, and synchronization lifecycle

### 6.1 Persistent compute objects

The optional context-owned compute state contains:

- shader module;
- descriptor-set layout;
- pipeline layout;
- compute pipeline;
- descriptor pool and one descriptor set;
- input buffer/memory/mapping;
- output buffer/memory/mapping;
- prepared dimensions and readiness flag.

Create these once in `coli_vulkan_compute_prepare()`. No object is lazily
created in `moe()`. Preparation is idempotent only for an identical config; a
different second config returns `COLI_VULKAN_INVALID_ARGUMENT` without changing
the existing state.

### 6.2 Per-dispatch objects

Allocate one primary command buffer and create one fence per dispatch. On
successful fence completion, destroy the fence and free the command buffer
before releasing the context mutex. This matches the already-reviewed Phase 3
ownership model and avoids resetting a possibly pending command buffer.

### 6.3 External synchronization

The existing `ColiVulkanContext.mutex` remains the single synchronization
domain. It covers:

- queue submission external synchronization;
- command-pool allocation/free external synchronization;
- descriptor-set update/use external synchronization;
- mapped scratch ownership;
- pending-operation state;
- context/tensor/compute counters;
- pipeline and scratch destruction.

Phase 4A exposes only a synchronous host API. Two host threads calling compute
serialize on this mutex. There is one queue and no queue-family transfer.

### 6.4 Bounded wait and retained submission

Extend the one-entry pending registry with `PENDING_COMPUTE`. After a successful
`vkQueueSubmit`, any `VK_TIMEOUT`, `VK_ERROR_DEVICE_LOST`, or other fence-wait
failure retains:

- the command buffer;
- its fence;
- the selected tensor reference;
- the context-owned descriptor set, scratch buffers, pipeline, command pool,
  queue, and device.

The compute call returns the mapped result as invalid. It never destroys or
reuses retained objects.

`coli_vulkan_finish_pending()` remains the one bounded completion API. On a
later fence success it destroys the retained fence, frees the command buffer,
clears `PENDING_COMPUTE`, and permits safe shutdown. A caller may not consume
the old output after finishing a timed-out compute; it would have to issue a
new compute. Colibri Phase 4A instead fails the requested run immediately and
uses the bounded exit cleanup.

For a non-device-loss wait error, mark the context degraded/unusable exactly as
the existing transfer path does. For `VK_ERROR_DEVICE_LOST`, mark it terminal,
reject every later submission, and retain all resources whose completion cannot
be proven until process exit.

### 6.5 Destruction order

Normal `model_vulkan_release()` first releases QT handles as it does now. Then
`coli_vulkan_context_destroy()`:

1. sets `shutting_down` and rejects new calls;
2. if an operation is pending, makes one bounded finish attempt;
3. returns with the context pointer intact on TIMEOUT, ERROR, or DEVICE_LOST;
4. after proven completion only, destroys remaining tensors;
5. unmaps and destroys output/input scratch;
6. destroys descriptor pool, pipeline, pipeline layout, descriptor-set layout,
   and shader module in dependency-safe order;
7. destroys the command pool, device, and instance.

There is no `vkDeviceWaitIdle()` or unbounded retry anywhere.

## 7. Runtime gates for upload-only versus compute-enabled Vulkan

### 7.1 Modes

| Build/runtime settings | Mode | Contract |
| --- | --- | --- |
| `VULKAN=0`, no Vulkan env | CPU | Existing behavior, no Vulkan dependency or objects. |
| `VULKAN=1`, `COLI_VULKAN` absent/`0` | CPU in Vulkan-capable binary | Existing behavior, no context. |
| `VULKAN=1`, `COLI_VULKAN=1`, compute absent/`0` | Phase 3B upload-only | Existing READY banner and CPU inference unchanged. |
| Above plus `COLI_VULKAN_COMPUTE=1` | Phase 4A routed-down | All runtime gates below are mandatory. |

`COLI_VULKAN_COMPUTE` and `COLI_VULKAN_VALIDATE` accept only exact `0` or `1`.
Compute without `COLI_VULKAN=1`, or in a binary without `VULKAN=1`, exits 2
with a specific diagnostic before model inference.

### 7.2 Required Phase 4A compute settings

Explicit compute requires:

- existing `VULKAN_EXPERT_MB=<positive integer>` and bounded
  `VULKAN_TIMEOUT_MS`;
- `REPIN=0`;
- `MTP=0` and `DRAFT=0`;
- `XEXP=0`;
- `IDOT=0` so CPU-off and Vulkan-on compare the same exact-float format-2
  operation rather than CPU activation-requantized IDOT;
- a `PIN` input covering every expert ID in every main sparse layer;
- successful Vulkan publication of every one of those complete triplets;
- format-2 down tensors of identical `[hidden, moe_inter]` geometry;
- successful compute preparation and zero pending work before inference.

The complete-residency requirement is intentional. It keeps LRU, working-set,
and disk misses entirely outside Phase 4A and guarantees that any routed expert
has the only legal compute handle. The `c/glm_tiny` fixture satisfies it with 16
experts and 48 tensors.

`COLI_VULKAN_VALIDATE=1` is mandatory for the Phase 4A acceptance command. It
may be optional for manual compute experiments, but an invalid value or use
without compute is rejected.

### 7.3 Mode telemetry

Upload-only mode preserves the existing line exactly:

```text
[VULKAN] READY upload-only: ... compute=CPU host_copies=retained
```

Compute mode prints a different READY line and must not print the upload-only
line:

```text
[VULKAN] READY compute-down-fmt2: vendor=0x1002 device=0x67b1 experts=16 tensors=48 committed=B budget=C pending=0 max_rows=64 gate_up=CPU down=Vulkan host_copies=retained
```

This makes residency and actual compute mode unambiguous.

## 8. Explicit fallback and failure policy

### 8.1 Compute disabled

When `COLI_VULKAN_COMPUTE` is absent or zero, the merged call
`matmul_qt(hh, gg, &e->d, nr)` executes exactly as before. A Vulkan-capable
upload-only run remains CPU arithmetic with resident copies that are not
consulted.

### 8.2 Compute explicitly requested

There is no CPU down fallback. Any of these is fatal to the run:

- selected expert is not from the startup-pinned resident tier;
- its down handle is null, non-READY, non-strict-local, wrong-context,
  non-format-2, or wrong geometry;
- `nr` exceeds the prepared capacity;
- descriptor update, command allocation/recording, pipeline binding,
  submission, or bounded fence wait fails;
- numerical validation fails;
- context accounting becomes inconsistent.

`vulkan_expert_down_or_die()` increments the error or forbidden-fallback
counter as applicable, prints one machine-readable diagnostic naming layer,
expert, rows, and result, then calls `exit(2)`. This is the smallest propagation
change because current `moe()`, layer, step, and generation APIs return no
status, and existing compute paths already treat a wrong routed-expert result
as fatal. The registered Phase 3B `atexit` handler performs bounded cleanup. It
retains rather than destroys unproven resources.

The helper must not call `matmul_qt()` after a Vulkan failure. Users can obtain
the CPU path only by starting a new run with compute disabled.

### 8.3 Startup eligibility failure

Incomplete residency, unsupported format/geometry, insufficient allocation
slots for scratch plus all expert tensors, missing shader support, or pipeline
preparation failure exits 2 before `generate()`. A smaller Phase 3B prefix is
valid for upload-only mode but invalid for Phase 4A compute mode.

## 9. Deterministic tests and PASS, FAIL, and SKIP criteria

### 9.1 Pure unit tests

`c/tests/test_vulkan_compute.c` must test without real hardware:

- format-2 eligibility for the exact tiny down geometry and several small
  multiples of eight;
- rejection of formats 0, 1, and 3-6 for compute while retaining their upload
  eligibility;
- rejection of odd/non-word-aligned `I`, zero rows, rows over 64, mismatched
  geometry, wrong source lengths, wrong scale offset/size, non-READY state, and
  wrong context;
- overflow of rows-by-I, rows-by-O, byte sizes, total outputs, group count, and
  descriptor ranges;
- nibble order and signed conversion for all values `0..15`;
- scale word addressing at aligned offsets;
- deterministic scalar references for zero, alternating extrema, identity-like,
  and seeded pseudo-random weights/activations;
- environment parsing and all compute/upload-only/CPU mode combinations;
- complete-pinned-tier validation and detection of one missing gate/up/down
  handle.

Validator rejection must occur before any fake Vulkan object/call counter
changes.

### 9.2 Fake Vulkan lifecycle and fault tests

Extend the injectable API and counters to cover every new object and call:

- shader module create/destroy;
- descriptor layout/pool/set creation and destruction;
- pipeline layout/pipeline create/destroy;
- input/output buffer and memory allocation/free/map/unmap;
- command buffer allocate/free;
- fence create/destroy;
- descriptor updates, barriers, binds, push constants, dispatch, queue submit,
  and fence wait.

Inject failure at every pre-submit step and assert complete reverse cleanup.
Inject queue-submit failure and assert command/fence cleanup is safe. Inject
post-submit ERROR/TIMEOUT and assert the command buffer and fence remain
retained and every persistent dependency remains live. After a later successful
`coli_vulkan_finish_pending()`, assert command/fence counts balance and normal
destruction reaches zero live resources.

The terminal DEVICE_LOST case must assert:

- context unusable and later submissions rejected;
- command buffer, fence, selected tensor, scratch, descriptors, pipeline,
  command pool, and device deliberately remain undestroyed;
- context destruction returns DEVICE_LOST with a non-null pointer;
- no fake destructor ran for a resource whose completion was unproven.

Add a two-thread serialization test showing that queue submission,
command-pool use, descriptor update, and mapped scratch access never overlap.

### 9.3 Sanitizers

Run all ordinary validation, preparation, dispatch-success, pre-submit failure,
post-submit recoverable failure, timeout-then-finish, and shutdown tests under
ASan, UBSan, and LeakSanitizer with `detect_leaks=1`.

Run only the intentional terminal DEVICE_LOST retention case separately with
`detect_leaks=0`. Any leak in an ordinary path is FAIL.

### 9.4 Strict real R9 390 hardware test

The strict target must:

- require exactly one `0x1002/0x67b1`, never SKIP it;
- retain the Phase 3B three-heap/seven-type and strict expert-memory checks;
- prove input/output scratch uses HOST_VISIBLE/HOST_COHERENT non-DEVICE_LOCAL
  system memory, not heap-2 BAR memory;
- create the pipeline from the production SPIR-V;
- upload deterministic format-2 weights through the validated public creator;
- dispatch rows 1, 4, and 12 at geometries including the tiny down
  `[O=128,I=32]` case;
- compare every output element against the scalar CPU reference;
- verify one `vkCmdDispatch` and one successful queue submission/completion per
  API call;
- finish with zero pending operations, balanced transient command/fence
  resources, and bounded context destruction.

### 9.5 Generation smoke tests

The smoke harness uses three runs of the same `VULKAN=1` binary and byte-identical
temporary copies of `c/glm_tiny`:

1. Vulkan off: CPU reference.
2. `COLI_VULKAN=1`, compute off: independent Phase 3B upload-only regression.
3. Residency plus `COLI_VULKAN_COMPUTE=1` and validation: Phase 4A.

All three use the same 12-token prompt, four continuation positions, all 16
pins, `MTP=0`, `DRAFT=0`, greedy sampling, one OpenMP thread, fixed seed,
`PIPE=0`, `XEXP=0`, and `IDOT=0`.

The harness additionally requires nonzero exits for:

- compute requested without residency;
- missing `VULKAN_EXPERT_MB`;
- `REPIN=1`, `MTP!=0`, `DRAFT!=0`, `XEXP!=0`, or `IDOT!=0`;
- a pin file missing any one of the 16 experts;
- a 64 MiB cap artificially reduced below the complete tier;
- expert bits selecting a non-format-2 down tensor;
- an explicitly supplied missing/incomplete model fixture.

### 9.6 Required commands

Implementation acceptance must run:

```sh
make -C c vulkan-strict-compile-test
make -C c vulkan-unit-test
make -C c vulkan-loader-unit-test
make -C c vulkan-compute-unit-test
make -C c vulkan-sanitize-test
make -C c vulkan-loader-sanitize-test
make -C c vulkan-compute-sanitize-test
make -C c vulkan-device-lost-sanitize-test
make -C c vulkan-compute-device-lost-sanitize-test
make -C c vulkan-hardware-test-strict
make -C c vulkan-compute-hardware-test-strict
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny
make -C c vulkan-compute-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny
make -C c test
git diff --check
```

The strict compile target must prove `-DCOLI_VULKAN -Werror` compiled the real
production compute path and must reject a production export of any raw-buffer
creator/getter. A source/symbol check must also fail if `vkDeviceWaitIdle`
appears in the backend.

### 9.7 PASS

PASS requires all of the following:

- all strict, unit, fault, sanitizer, real-hardware, smoke, and repository
  regression commands exit zero;
- all three smoke runs produce the identical four token IDs;
- the compute run reports positive dispatches and rows, equal submitted and
  completed dispatch counts, zero pending operations, zero CPU down fallbacks,
  zero errors/timeouts/device loss, and finite errors within section 11;
- CPU gate/up row count, Vulkan down row count, and validation-reference down
  row count agree exactly;
- upload-only mode still reports `compute=CPU` and no compute pipeline or
  dispatch;
- compute-disabled CPU behavior is unchanged;
- no ordinary sanitizer leak and no unsafe destructor after submitted work;
- no GLM-5.2 execution or benchmark.

### 9.8 FAIL

FAIL includes any nonzero required command, SKIP in a target-server gate,
missing supplied fixture, output/token mismatch, dispatch count zero, CPU down
fallback count nonzero, counter mismatch, unsupported expert reaching CPU
silently, unbounded wait, `vkDeviceWaitIdle()`, BAR scratch selection, partial
expert coverage, destruction before proven completion, ordinary leak, or any
change to CPU/upload-only/CUDA/HIP/Metal behavior.

### 9.9 SKIP

Only optional developer hardware/smoke targets may SKIP when no model is
specified or the exact device is unavailable. Unit/fault tests never SKIP for
missing Vulkan development files. An explicitly supplied fixture that is
missing is FAIL. Strict R9 390 hardware and target-server generation acceptance
never SKIP.

## 10. Exact telemetry proving dispatch and CPU fallback counts

### 10.1 Backend counters

`ColiVulkanContextInfo` exposes monotonically increasing:

- `compute_dispatch_recorded`: increment immediately after one successful
  `vkCmdDispatch` recording;
- `compute_submitted`: increment only after `vkQueueSubmit == VK_SUCCESS`;
- `compute_completed`: increment only after the dispatch fence returns
  `VK_SUCCESS`;
- `compute_rows_completed`: add `rows` only with a completed result copied to
  the caller;
- `compute_timeouts`, `compute_errors`, `compute_device_lost`.

A failed pre-submit recording may increment `recorded` but not `submitted`.
Acceptance requires `recorded == submitted == completed > 0`.

### 10.2 Colibri counters

The `Model` records:

- `vulkan_down_calls`: successful backend calls used for routed down results;
- `vulkan_down_rows`: sum of their `nr`;
- `vulkan_cpu_gate_up_rows`: rows for which CPU gate/up/SiLU ran before a
  Vulkan down call;
- `vulkan_cpu_reference_down_rows`: validation-only exact CPU down rows;
- `vulkan_cpu_down_fallbacks`: attempts to use CPU down while compute was
  requested;
- `vulkan_validation_failures` and maximum observed errors.

The final machine-readable line is:

```text
[VULKAN] COMPUTE summary: kernel=fmt2-down dispatches=N submitted=N completed=N rows=R cpu_gate_up_rows=R cpu_reference_down_rows=R cpu_down_fallbacks=0 timeouts=0 errors=0 device_lost=0 pending=0 max_abs=A max_rel=B
```

`dispatches` is not inferred from residency. It counts actual recorded and
submitted `vkCmdDispatch` work. `cpu_down_fallbacks` is not the ordinary CPU
mode count; it counts only a forbidden fallback after explicit compute was
enabled. This distinction prevents a resident-but-never-dispatched run from
passing.

## 11. Numerical correctness criteria

### 11.1 Reference calculation

The mathematical reference is the exact-float format-2 CPU path, not the IDOT
activation-requantized optimization. The smoke test requires `IDOT=0`.

In `COLI_VULKAN_VALIDATE=1`, for every routed down call:

1. calculate `reference` with `matmul_qt_ex(reference, gg, &e->d, nr, 0)`;
2. calculate `hh` with Vulkan;
3. compare all `nr * hidden` elements;
4. use the Vulkan `hh`, not the CPU reference, for weighted residual
   accumulation.

Thus validation does not turn generation into CPU output with decorative GPU
work.

### 11.2 Fixed tolerance

Every compared value must be finite. For each element, accept only if:

```text
abs(vulkan - cpu) <= 5e-5 + 5e-4 * max(abs(cpu), abs(vulkan))
```

Record both:

```text
max_abs = max(abs(vulkan - cpu))
max_rel = max(abs(vulkan - cpu) / max(1e-6, abs(cpu), abs(vulkan)))
```

The absolute term handles values near zero; the relative term permits expected
binary32 FMA/reduction differences while remaining small enough to catch
nibble order, row stride, scale offset, and descriptor-range defects. Unit and
hardware tests include deliberately wrong nibble/scale controls that must fail
this threshold.

The final end-to-end criterion is stricter at the observable boundary: all four
token IDs from CPU, upload-only, and compute-enabled runs must match exactly.

## 12. Four-token `c/glm_tiny` acceptance command

The exact top-level acceptance command is:

```sh
make -C c vulkan-compute-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny
```

The compute child run constructed by the harness is equivalent to:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY \
  COLI_VULKAN=1 COLI_VULKAN_COMPUTE=1 COLI_VULKAN_VALIDATE=1 \
  VULKAN_EXPERT_MB=64 VULKAN_TIMEOUT_MS=5000 \
  SNAP="$ON_MODEL" REF="$REF4" PIN="$PIN_FILE" \
  PIN_GB=0.01 PIN_FILL=0 AUTOPIN=0 REPIN=0 \
  MTP=0 DRAFT=0 IDOT=0 XEXP=0 PIPE=0 \
  TEMP=0 SEED=1 PROF=1 OMP_NUM_THREADS=1 COLI_NO_OMP_TUNE=1 \
  ./c/colibri 64 4 4
```

`REF4` contains the checked-in prompt and exactly four continuation positions.
`PIN_FILE` contains experts 0 through 7 for sparse layers 3 and 4. `ON_MODEL`
is a temporary copy so `.coli_usage` cannot modify the source fixture.

The harness must perform CPU-off and Phase 3B upload-only companion runs with
the same effective model, prompt, quantization, routing, `IDOT=0`, and
determinism settings. It does not download or regenerate a model.

## 13. Work explicitly deferred beyond Phase 4A

- Vulkan gate projection, up projection, SiLU, gate/up product, or fused full
  expert MLP.
- Formats 0, 1, and 3-6 compute kernels.
- CPU IDOT-equivalent activation quantization on Vulkan.
- Shared-expert, dense MLP, attention, router, embedding, lm-head, KV-cache,
  residual, or sampling Vulkan compute.
- General dispatch from `matmul_qt()` or `matmul_qt_ex()`.
- LRU experts, working-set experts, inference-time upload, misses, promotion,
  eviction, or REPIN.
- Partial Vulkan tiers with CPU fallback.
- MTP/draft compute or speculative batches.
- More than 64 rows, serving, multiplexed batch decode, scoring, or full-model
  execution.
- Host-copy release.
- Multiple queues, asynchronous overlap, multiple in-flight operations,
  timeline semaphores, descriptor indexing, queue-family transfers, or
  multi-GPU.
- Device-local activation/result scratch, BAR use, ReBAR assumptions, staging
  rings, or scratch suballocation.
- Tensor replacement/update; future updates remain allocate-new, finish,
  atomic-swap, then free-old.
- Pipeline cache persistence, specialization variants, autotuning, subgroup or
  wave-specific kernels, and performance claims.
- Runtime GLSL compilation.
- GLM-5.2 execution or benchmarking; its benchmark remains `NOT_RUN`.

## 14. Unresolved decisions requiring approval

Implementation should not begin until these Phase 4A decisions are approved.

### Decision 1: routed-down-only boundary

Approve format-2 routed `down_proj` as the only Vulkan arithmetic in Phase 4A,
with CPU gate/up/SiLU and Vulkan results used in the residual.

Recommendation: approve. It is the smallest seam that proves real routed
expert dispatch end to end. Requiring all three projections would multiply
lifecycle and numerical risks without strengthening the first-dispatch proof.

### Decision 2: complete startup residency in compute mode

Approve requiring every main sparse-layer expert to be startup-pinned and have
a complete resident triplet before compute mode starts. A bounded Phase 3B
prefix remains valid only in upload-only mode.

Recommendation: approve for Phase 4A. It removes all miss/LRU/fallback behavior
from the milestone and the 16-expert tiny fixture fits the 64 MiB test cap.

### Decision 3: checked-in SPIR-V artifact

Approve a reviewed GLSL source plus a deterministic checked-in SPIR-V header.
Production consumes the header directly; a separate maintainer target rebuilds
and byte-compares it with a pinned `glslangValidator` version. No runtime shader
compiler or ordinary build dependency is added.

Recommendation: approve. This makes deployed builds reproducible on the target
server without introducing runtime compiler loading. The generated header must
be reviewed as a generated artifact and tied to its source hash.

### Decision 4: fail-closed deep compute error

Approve `vulkan_expert_down_or_die()` calling `exit(2)` after a compute failure,
using the existing bounded `atexit` cleanup, instead of changing the return type
of `moe()`, every layer/step function, and all serve/generation callers in this
first milestone.

Recommendation: approve for Phase 4A. Silent CPU fallback is forbidden, and a
broad status-plumbing refactor would be larger and harder to validate than the
compute seam. A later phase can introduce structured inference error returns.

### Decision 5: exact-float comparison mode and fixed capacity

Approve requiring `IDOT=0`, `MTP=0`, `DRAFT=0`, `XEXP=0`, a 64-row persistent
scratch capacity, and the element tolerance:

```text
5e-5 absolute + 5e-4 relative
```

for Phase 4A compute acceptance.

Recommendation: approve. These are explicit milestone boundaries, not hidden
fallbacks. They make CPU and Vulkan compare the same arithmetic and bound every
scratch allocation and dispatch.

With decisions 1 through 5 approved, no further architectural decision is
needed before implementing and validating Phase 4A. Performance expansion and
general model support remain separate later phases.
