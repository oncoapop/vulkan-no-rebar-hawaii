# Phase 3B Vulkan expert-loader architecture

Status: design only; no Phase 3B source implementation has been made.

Design base: commit `13f96097671bfeb054c5bfd1afd13bd6dbbfa087`
(`Add persistent Vulkan context and tensor lifecycle (#5)`) on
`feature/vulkan-expert-upload-integration-phase3`.

Phase 3B is an upload-only integration milestone. It connects real startup
expert loading to the persistent Phase 3A Vulkan context and tensor lifecycle,
but it does not execute any inference operation with Vulkan. CPU inference and
all existing CPU, CUDA, HIP, and Metal behavior remain unchanged when Vulkan is
not explicitly enabled.

The design is safe to implement only with the format-aware public creation API,
transactional triplet publication, retained host storage, fail-closed runtime
enablement, and bounded cleanup specified below.

## 1. Exact existing call paths and functions

All line numbers in this section describe the design base above, before the
proposed edits move them.

### 1.1 Build and backend selection

1. `c/Makefile:131-191` defines `CUDA`, `CUDA_DLL`, `VULKAN`, `HIP`, and the
   first Vulkan coexistence check.
2. `c/Makefile:228-235` discovers Vulkan development files and defines the
   Vulkan object, flags, and sanitizer flags.
3. `c/Makefile:306-328` makes `VULKAN=1` Linux-only, rejects CUDA, CUDA DLL,
   HIP, and Metal combinations, adds `-DCOLI_VULKAN`, links Vulkan, and adds
   `backend_vulkan.o`.
4. `c/Makefile:356-357` links `colibri.c` with the selected backend objects.
5. The default is still `VULKAN=0`. At the current Phase 3A head,
   `c/colibri.c` does not include `backend_vulkan.h` and makes no Vulkan call.

CUDA and HIP share the `COLI_CUDA` host API. `c/backend_cuda.h` declares the
opaque `ColiCudaTensor`, initialization, upload, compute, update, and destruction
functions. On Windows, `c/backend_loader.c` resolves the CUDA DLL ABI. Neither
file participates in the Linux Vulkan path and neither should change in Phase
3B.

### 1.2 Model initialization and resident tensor loading

1. `c/colibri.c:6809`, `main()`, parses process configuration. Backend runtime
   parsing currently covers CUDA/HIP at `c/colibri.c:7035-7100` and Metal at
   `c/colibri.c:7101-7114`.
2. `main()` constructs `Model m` and calls `model_init()` at
   `c/colibri.c:7117`.
3. `c/colibri.c:1161`, `model_init()`, zero-initializes `Model`, calls
   `load_cfg()`, and calls `st_init_multi()` at `c/colibri.c:1165`.
4. `c/st.h:248`, `st_init_multi()`, scans safetensors shards, validates and
   indexes each tensor, and retains file descriptors and `st_tensor` metadata.
   The loader subsequently uses `st_find()`, `st_has()`, `st_nbytes()`,
   `st_read_raw()`, and `st_read_f32_cap()` from `c/st.h`.
5. `model_init()` loads resident non-routed tensors with `qt_load()` at
   `c/colibri.c:1116`, which calls `qt_from_disk()` at `c/colibri.c:1080`.
6. `qt_from_disk()` either:

   - finds the `.qs` sidecar, obtains its exact byte lengths with
     `st_nbytes()`, calls `qt_resolve_fmt()`, allocates the format-specific host
     storage, and reads the weight and scale bytes; or
   - reads a full-precision tensor and uses `qt_alloc()` plus `qt_fill()` to
     create the requested runtime-quantized host representation.

7. Routed experts are not loaded by `model_init()`. Sparse layers receive their
   LRU arrays, usage state, and pin arrays at `c/colibri.c:1222-1228`; routed
   expert bytes remain on disk until startup pinning or an inference-time miss.

### 1.3 Current `QT` and expert ownership

`c/colibri.c:113-119` defines `QT`. It owns or views:

- `fmt`, `O`, `I`, and `gs`;
- exactly one weight pointer among `qf`, `q8`, and `q4`;
- `s` for ordinary scales, grouped scales, or the format-6 `.qs` tag;
- an independently guarded CUDA pointer when `COLI_CUDA` is compiled;
- the existing CUDA eligibility, failure, and device fields.

`c/colibri.c:162-168` defines `ESlot`, which contains the gate (`g`), up (`u`),
and down (`d`) `QT` values plus their host slab or mmap ownership. `Model.pin`
at `c/colibri.c:200` owns startup-pinned `ESlot` arrays. `Model.ecache` owns
ordinary LRU slots, and `Model.ws[64]` owns transient load slots.

### 1.4 Exact routed-expert disk-loading path

`c/colibri.c:1739`, `expert_load()`, times and calls
`expert_load_impl()` at `c/colibri.c:1511`.

`expert_load_impl()`:

1. constructs the exact gate/up/down safetensor names at
   `c/colibri.c:1521-1523`;
2. for an unquantized model, calls `qt_from_disk()` for gate, up, and down at
   `c/colibri.c:1524-1532`;
3. otherwise finds the three weights and three `.qs` tensors at
   `c/colibri.c:1534-1543`;
4. under `COLI_MMAP=1`, validates each layout with `qt_resolve_fmt()` and
   publishes `QT` views into mapped data at `c/colibri.c:1553-1595`;
5. otherwise allocates or reuses `slab` and `fslab`, reads all three weight and
   scale ranges, calls `qt_resolve_fmt()` for each, and publishes the three
   host views at `c/colibri.c:1722-1730`;
6. sets `s->eid` only after the host triplet is complete.

The optional Linux io_uring path uses `uring_load_add()` at
`c/colibri.c:1793`, then `uring_finalize_load()` at `c/colibri.c:1902`. The
finalizer independently calls `qt_resolve_fmt()` and publishes the three `QT`
views at `c/colibri.c:1912-1925`.

Phase 3B must not upload from `expert_load_impl()` or either asynchronous miss
path. It must consume only completely loaded startup pin slots after the
parallel host-load barrier in `pin_load()`.

### 1.5 Exact startup pin path

1. Explicit startup pinning begins in `main()` at `c/colibri.c:7187-7211`.
   `PIN=<file>` uses that file; `PIN=auto` first looks for `.coli_usage`, then
   `stats.txt`.
2. Automatic history pinning can call the same `pin_load()` at
   `c/colibri.c:7222-7235` when no explicit `PIN` is present.
3. `c/colibri.c:6445`, `pin_load()`, parses and deduplicates ranked
   `(layer, expert, count)` records, sorts them with `pin_rec_cmp()`, derives
   the host pin count from `PIN_GB`, allocates `m->pin[layer]`, and records
   `m->npin[layer]`.
4. `pin_arena_bind()` may prepare per-layer host arenas on Linux.
5. `pin_load()` calls `expert_load()` in an OpenMP loop at
   `c/colibri.c:6534-6536`. A CUDA-only prefix/suffix split can perform a second
   host-load loop at `c/colibri.c:6600-6604`.
6. The current CUDA path serially uploads gate/up/down at
   `c/colibri.c:6542-6573`, can release host copies, and reports its tier.
7. `pin_wire()` runs at `c/colibri.c:6608`, after loading and placement.

For a Vulkan build, CUDA/HIP/Metal are excluded. The Phase 3B insertion point is
a new serial Vulkan placement loop after the one OpenMP host-load loop has
completed and before `pin_wire()`. The Vulkan loop follows the already sorted
`PinRec` order and touches only `m->pin` slots created by this invocation.

### 1.6 Exact inference path that remains CPU

The generation modes call `generate()`, `run_replay()`, `run_text()`, or the
serve/scoring functions. They ultimately call `step()`, `layers_forward()`,
`layer_forward()`, and `moe()`.

`c/colibri.c:2838`, `moe()`:

1. computes routing and usage state;
2. builds the unique expert set;
3. resolves each expert from `m->pin`, then `m->ecache`, then `m->ws` at
   `c/colibri.c:3211-3217`;
4. loads misses through the blocking, pthread pipe, or io_uring paths;
5. executes routed experts through the existing Metal, CUDA/HIP, or CPU paths;
6. on the ordinary CPU path, calls `expert_gate_up()` and `matmul_qt()` at
   `c/colibri.c:3522-3525`;
7. may promote complete miss slots into the LRU by whole-`ESlot` swaps at
   `c/colibri.c:3675-3680`.

`c/colibri.c:596`, `matmul_qt_ex()`, tries Metal, then CUDA/HIP, and otherwise
dispatches the existing format-specific CPU kernels at
`c/colibri.c:615-630`.

Phase 3B makes no change to `moe()`, `expert_gate_up()`, `matmul_qt()`, or
`matmul_qt_ex()`. A Vulkan pointer in a pinned `QT` is residency metadata only.
All expert arithmetic still reads the retained host pointers.

### 1.7 Current Phase 3A Vulkan path

`c/backend_vulkan.c` already supplies:

- one process-wide `ColiVulkanContext` with one instance, exact R9 390 physical
  device, logical device, queue, command pool, mutex, budget, tensor registry,
  and one retained pending operation;
- exact-device enforcement in `coli_vulkan_context_create()` at
  `c/backend_vulkan.c:569-766`;
- strict `DEVICE_LOCAL` and non-`HOST_VISIBLE` destination allocation in
  `create_destination_locked()` at `c/backend_vulkan.c:800-901`;
- aligned raw packing and upload in the guarded
  `coli_vulkan_tensor_upload()` at `c/backend_vulkan.c:950-1046`;
- `coli_vulkan_finish_pending()`, tensor inspection, tensor destruction, exact
  readback, and bounded `coli_vulkan_context_destroy()`.

The raw upload declaration at `c/backend_vulkan.h:230-254` explicitly says it
does not validate `QT` geometry and must not be called by Phase 3B loader code.
That remains a merge blocker until the format-aware API below exists.

### 1.8 Current termination path

There is no centralized model destructor. After `model_init()`, `main()` returns
directly from scoring, serve, prompt generation, replay, teacher forcing,
oracle rejection, and normal oracle generation at `c/colibri.c:7244-7341`.
Those returns currently rely on process reclamation for model allocations.

Phase 3B introduces one post-model cleanup label for Vulkan ownership. It does
not attempt a broad CPU-model memory refactor.

## 2. Exact files and functions to modify

### 2.1 Production files

| File | Exact Phase 3B change |
| --- | --- |
| `c/backend_vulkan.h` | Add `ColiVulkanQTSpec` and validated-layout metadata; declare a pure `coli_vulkan_validate_qt_spec()` and the sole public production creator `coli_vulkan_tensor_create_qt()`; extend tensor inspection with format geometry. Keep the raw byte creator internal/test-only. Document format-6 tag handling and timeout ownership. |
| `c/backend_vulkan.c` | Implement checked format formulas and source validation; store validated format metadata in `ColiVulkanTensor`; implement `coli_vulkan_tensor_create_qt()` as validation followed by the existing internal transport; ensure rejected specs allocate and submit nothing. Do not change persistent-context, strict-memory, queue, command-pool, or device-loss behavior. |
| `c/colibri.c` | Include `backend_vulkan.h` under `COLI_VULKAN`; extend `QT` with exact host source lengths and an independently guarded Vulkan tensor pointer; extend `Model` with the one context, timeout, and residency counters; set exact lengths in `qt_alloc()`, `qt_from_disk()`, both direct `expert_load_impl()` publication paths, and `uring_finalize_load()`; add the helpers listed below; add the serial transactional pass to `pin_load()`; add runtime parsing/context creation in `main()`; route every post-model return through bounded cleanup; add truthful upload-only telemetry. |
| `c/Makefile` | Update the Phase 3A comments to Phase 3B semantics; ensure `colibri` depends on `backend_vulkan.h`; add strict format/loader unit and sanitizer targets; make the aggregate Vulkan target sequential and fail-fast; add the opt-in smoke target without putting model execution in default `make test`. |

The new `c/colibri.c` helper functions are:

- `qt_set_source_lengths()` — records actual weight and scale source lengths
  at every construction/publication seam;
- `qt_vulkan_spec()` — converts one fully formed `QT` to a
  `ColiVulkanQTSpec` without deriving or repairing lengths;
- `qt_vulkan_release()` — relinquishes one `QT` tensor handle safely;
- `expert_vulkan_upload_triplet()` — creates three local tensors,
  verifies READY/strict-local state, and publishes or rolls back atomically;
- `pin_vulkan_upload_prefix()` — serially applies the transaction to the
  sorted startup pins while respecting actual budget and allocation limits;
- `model_vulkan_release()` — walks every pinned Vulkan-owning `QT`, requests
  tensor destruction, finishes any retained operation with a finite wait, and
  then destroys the context;
- `vulkan_exit_cleanup()` — last-resort bounded process-exit cleanup for
  legacy fatal paths, disabled after normal cleanup succeeds.

Existing functions modified in place are:

- `qt_alloc()`;
- `qt_from_disk()`;
- the mmap and slab publication loops in `expert_load_impl()`;
- `uring_finalize_load()` only to preserve correct source-length metadata for
  ordinary loader consistency, not to upload;
- `pin_load()`;
- `prof_config()` only to identify `CPU + Vulkan residency (upload-only)`;
- `main()`.

`qt_resolve_fmt()` remains the CPU loader's untrusted-container validation.
CPU-only builds cannot depend on Vulkan, so that security check must remain.
The Vulkan backend performs an independent second validation at the trust
boundary before any device allocation.

### 2.2 Tests

| File | Exact Phase 3B change |
| --- | --- |
| `c/tests/test_vulkan_context.c` | Add pure format validation, overflow, no-allocation-on-rejection, validated metadata, and fault-injection cases. Preserve all Phase 3A ownership counters and device-loss tests. |
| `c/tests/test_backend_vulkan.c` | Replace direct raw-upload use with `coli_vulkan_tensor_create_qt()`; exercise representative real layouts, strict-local allocation, exact readback, and format metadata on the R9 390. |
| `c/tests/test_vulkan_loader.c` (new) | Include the loader seam with fake Vulkan, construct real `QT` triplets, and test transactional publication, host retention, rollback, timeouts, terminal device loss, cleanup, and silent-enable prevention without a model. |
| `c/tests/test_vulkan_loader_smoke.py` (new) | Implement the explicitly invoked four-token on/off comparison described in section 8. It may use only an already present caller-supplied tiny fixture and must never download a model. |

### 2.3 Files explicitly not modified

- `c/backend_cuda.h`
- `c/backend_cuda.cu`
- `c/backend_loader.c`
- `c/backend_metal.h`
- `c/backend_metal.mm`
- `c/st.h`
- `c/uring.h`
- any active Colibri or GLM-5.2 installation

The existing CUDA/HIP/Metal fields in `QT` are not renamed, reordered for
sharing, or reused by Vulkan. No general backend abstraction is introduced.

## 3. Supported `QT` formats and validation formulas

### 3.1 Public specification and sole production creator

Add this logical contract to `c/backend_vulkan.h`:

```c
typedef struct {
    uint32_t fmt;
    uint64_t I;
    uint64_t O;
    uint64_t gs;
    const void *weights;
    uint64_t weight_bytes;
    const void *scales;
    uint64_t scale_bytes;
} ColiVulkanQTSpec;

typedef struct {
    uint64_t scale_count;
    uint64_t effective_group_size;
    uint64_t uploaded_scale_bytes;
    ColiVulkanTensorLayout packed;
} ColiVulkanQTLayout;

ColiVulkanResult coli_vulkan_validate_qt_spec(
    const ColiVulkanQTSpec *spec,
    VkDeviceSize min_storage_buffer_offset_alignment,
    ColiVulkanQTLayout *layout);

ColiVulkanResult coli_vulkan_tensor_create_qt(
    ColiVulkanContext *context,
    ColiVulkanTensor **tensor,
    const ColiVulkanQTSpec *spec,
    uint64_t timeout_ns);
```

`coli_vulkan_tensor_create_qt()` is the only tensor-creation declaration visible
to a normal production translation unit. It must call
`coli_vulkan_validate_qt_spec()` before locking the context for allocation. It
then calls a private raw transport function inside `backend_vulkan.c`.

The existing raw `coli_vulkan_tensor_upload()` remains available only in a
`COLI_VULKAN_TESTING` build, or is converted to a private implementation plus a
test-only wrapper. A production `colibri.c` must not define an internal macro to
reach it. A symbol inspection test should confirm that a production
`backend_vulkan.o` does not export a raw creation entry point.

### 3.2 Arithmetic rules

All formulas use unsigned checked helpers. Define:

```text
ceil_div(x, d) = x / d + (x % d != 0)
```

This form avoids computing `x + d - 1`. Every multiplication and byte
conversion is checked before it is performed. Validation requires:

- `I > 0` and `O > 0`;
- `I <= INT_MAX` and `O <= INT_MAX`, matching `QT` and CPU kernels;
- the exact format-specific `gs` convention below;
- `weights != NULL` and exact `weight_bytes`;
- `scales != NULL` exactly when a scale source or format tag is required;
- exact `scale_bytes`, not a lower bound;
- every derived count and byte length to fit `uint64_t`, `size_t`, and
  `VkDeviceSize`;
- alignment to be nonzero;
- the resulting packed offsets and size to pass the existing
  `coli_vulkan_plan_tensor_layout()` checks.

The backend assumes and compile-time asserts a four-byte IEEE-compatible
`float`, matching the safetensor `.qs` and current CPU representation.

### 3.3 Exact eligible formats

All currently loadable CPU `QT` formats 0 through 6 are eligible for Phase 3B
residency. This does not make any format compute-eligible; Vulkan compute
eligibility is false for every Phase 3B tensor.

| `fmt` | Meaning | Required `gs` in current `QT` | Exact weight bytes | Derived ordinary scale count | Exact scale source bytes | Bytes uploaded as scales |
| --- | --- | ---: | --- | --- | --- | --- |
| 0 | F32 | 0 | `O * I * 4` | 0 | 0 | 0 |
| 1 | INT8 per row | 0 | `O * I` | `O` | `O * 4` | `O * 4` |
| 2 | INT4 per row | 0 | `O * ceil_div(I, 2)` | `O` | `O * 4` | `O * 4` |
| 3 | INT2 per row | 0 | `O * ceil_div(I, 4)` | `O` | `O * 4` | `O * 4` |
| 4 | grouped INT4 | one of the supported values below | `O * ceil_div(I, 2)` | `O * ceil_div(I, gs)` | `scale_count * 4` | same |
| 5 | INT3, fixed group 64 | 0 | `O * ceil_div(I, 64) * 24` | `O * ceil_div(I, 64)` | `scale_count * 4` | same |
| 6 | E8/IQ3, fixed block 256 | 0 | `O * ceil_div(I, 256) * 98` | 0 | 4-byte `.qs` format tag | 0 |

Format 4 accepts exactly the group sizes already recognized by
`detect_group_size()`:

```text
16, 32, 48, 64, 96, 128, 192, 256
```

It also requires `gs <= I`. Formats 5 and 6 keep `QT.gs == 0` because their
group/block geometry is part of the format; validated metadata records effective
group sizes 64 and 256 respectively.

Format 6 is deliberately asymmetric. `qt_resolve_fmt()` recognizes it only
when the safetensor `.qs` source is exactly four bytes. Phase 3B therefore
requires and validates those four source bytes, but does not place them in the
ordinary scale range. `fmt` is the device metadata tag; the effective scales are
already inside the 98-byte weight blocks. The packed Vulkan layout has
`scale_size == 0` for format 6.

Any other `fmt`, group convention, source length, null-pointer combination, or
arithmetic result is `COLI_VULKAN_UNSUPPORTED`,
`COLI_VULKAN_INVALID_ARGUMENT`, or `COLI_VULKAN_LIMIT_EXCEEDED` before any
Vulkan allocation or queue submission.

### 3.4 Recording real source lengths in `QT`

Add two backend-neutral fields to `QT`:

```c
uint64_t weight_bytes;
uint64_t scale_bytes;
```

They record the actual host source allocation/view lengths, not a recomputation
at upload time. They are set as follows:

- `qt_alloc()` stores the actual runtime allocation lengths;
- `qt_from_disk()` stores the exact `st_nbytes()` values;
- the mmap, slab, and io_uring expert publication loops copy
  `tw[k]->nbytes` and `tq[k]->nbytes` into the corresponding `QT`;
- format 0 records zero scale bytes;
- format 6 records the actual four-byte `.qs` tag even though the validated
  uploaded scale payload is zero.

`qt_vulkan_spec()` chooses the weight pointer by format, passes the stored
lengths unchanged, and passes `s` for ordinary scales or the format-6 tag. It
does not call `qt_bytes()` and does not manufacture an expected source length.
This lets the backend independently compare actual loader metadata against the
format formula.

## 4. Ownership and rollback lifecycle

### 4.1 Persistent owners

`Model` owns exactly one `ColiVulkanContext *`. `main()` creates it once after
`model_init()` has succeeded and before `mirror_setup()` and either startup
`pin_load()` call. There is no per-expert instance, device, queue, or command
pool creation.

Each `QT` independently owns at most one `ColiVulkanTensor *`:

```c
#ifdef COLI_VULKAN
    ColiVulkanTensor *vulkan;
#endif
```

The field is separate from `cuda`, `cuda_eligible`, `cuda_failed`, and
`cuda_device`. A non-null Phase 3B Vulkan handle means validated persistent
residency only. It must never be interpreted as permission to dispatch compute.

Only `m->pin[layer][slot].{g,u,d}` can own Vulkan handles in Phase 3B. Dense,
shared, MTP-only non-pin, LRU, and working-set tensors remain host-only.

### 4.2 Startup upload ordering

Host expert loading stays parallel. Vulkan placement is serial on the main
thread after the OpenMP load loop has joined. This preserves the Phase 3A rule
that queue, command-pool allocation/free, retained-operation finishing, and
fence access are serialized by the context mutex.

The placement follows sorted `PinRec` order. It attempts a complete expert at a
time until the next triplet cannot fit the actual Vulkan budget or allocation
limit. A budget/limit boundary after at least one published expert ends the
prefix successfully. If the first complete triplet cannot be published,
explicit Vulkan enablement fails; zero residency is never reported as success.

### 4.3 Transactional gate/up/down publication

`expert_vulkan_upload_triplet()` uses local handles, never the destination
`QT.vulkan` fields, during construction:

1. Verify that all three destination fields are null.
2. Convert gate, up, and down to independent `ColiVulkanQTSpec` values.
3. Validate all three specs before allocating the first tensor.
4. Create gate, then up, then down with `coli_vulkan_tensor_create_qt()`.
5. After each successful return, call `coli_vulkan_tensor_get_info()` and
   require:

   - `state == COLI_VULKAN_TENSOR_READY`;
   - format geometry and derived counts match the validated spec;
   - `DEVICE_LOCAL` is set;
   - `HOST_VISIBLE` is clear.

6. Re-read `ColiVulkanContextInfo` and require the context to remain usable and
   not device-lost.
7. Only after all three checks succeed, assign the three local handles to
   `s->g.vulkan`, `s->u.vulkan`, and `s->d.vulkan` in one main-thread
   publication block and increment expert/tensor counters.

No inference thread exists during startup placement. “Atomic publication” here
means no partial state is exposed: all local work finishes before the three
owner fields are written. It does not add C atomics or a runtime reader.

### 4.4 Rollback

On any failure before publication:

1. Do not write any `QT.vulkan` field.
2. Call `coli_vulkan_tensor_free()` for each completed local tensor in reverse
   order.
3. If the failing creation returned a non-null tensor after TIMEOUT, ERROR, or
   DEVICE_LOST, call `coli_vulkan_tensor_free()` on it too. BUSY or DEVICE_LOST
   may clear the local pointer while transferring deferred destruction to the
   context registry.
4. If a pending operation exists and the context is not device-lost, call
   `coli_vulkan_finish_pending()` with the configured finite wait. Do not submit
   a later tensor or expert until it succeeds.
5. If finishing times out or fails, abort Vulkan startup and retain the context
   for the bounded shutdown path. If it reports DEVICE_LOST, mark startup
   terminal and submit no further work.
6. Leave all host pointers, slabs, mmap views, pin membership, and `eid`
   unchanged.

A validation failure is fatal to an explicitly requested Vulkan startup, not a
silent per-tensor CPU fallback. CPU-only behavior is obtained by leaving
Vulkan disabled, not by accepting a broken Vulkan-on run.

### 4.5 Host ownership

Every host copy is retained for the complete Phase 3B process lifetime.
Vulkan placement must not call `expert_host_release()`, `qt_unwire_mmap()`, or
any CUDA release-host logic. The existing CPU kernels continue to read
`qf`/`q8`/`q4` and `s`.

This also keeps rollback trivial: a failed Vulkan transaction leaves a complete
ordinary CPU pin. Because explicit Vulkan enablement is fail-closed, that pin
is used only during cleanup, not to disguise the failed request as a successful
Vulkan run.

### 4.6 Budget and allocation-count boundary

One expert consumes three persistent destination allocations. A staged upload
temporarily consumes one additional allocation. `pin_vulkan_upload_prefix()`
uses `ColiVulkanContextInfo.max_memory_allocation_count`, live allocations, and
committed bytes as an advisory precheck, while the backend remains authoritative
using actual `VkMemoryRequirements.size`.

There is no partial-expert reservation API in Phase 3A, so actual triplet
atomicity is implemented by allocate/upload then rollback. A
`COLI_VULKAN_LIMIT_EXCEEDED` after a nonempty prefix is a normal prefix boundary;
the same result for the first expert is startup failure. Every other create or
finish error is startup failure regardless of prefix length.

## 5. CLI and runtime configuration

### 5.1 Build-time gate

```text
make -C c colibri VULKAN=1
```

`VULKAN=1` remains opt-in and Linux-only. `c/Makefile` continues to reject
simultaneous `CUDA=1`, `CUDA_DLL=1`, `HIP=1`, or `METAL=1` at parse time.
The default `make -C c colibri` remains CPU behavior with no Vulkan dependency.

### 5.2 Runtime gate

Phase 3B uses:

| Setting | Contract |
| --- | --- |
| `COLI_VULKAN=1` | Explicitly requests Phase 3B Vulkan residency. No other value enables it. |
| `VULKAN_EXPERT_MB=<positive integer>` | Mandatory only when Vulkan is requested. Binary MiB, with no default or `auto`. Parsed by the existing Phase 3A configuration function. |
| `VULKAN_TIMEOUT_MS=<integer>` | Optional bounded upload, finish, and shutdown wait. Default 5000 ms. Accepted range 1 through 60000 ms; invalid values fail startup. |
| `PIN=<stats file or auto>` | Existing startup pin source. Vulkan uploads only experts that `pin_load()` actually materializes. |
| `PIN_GB=<value>` | Existing host pin budget. It remains independent of the Vulkan cap because host copies are retained. |
| `REPIN=0` | Required while Vulkan residency is enabled. |

If `COLI_VULKAN=1` is used with a binary built without `VULKAN=1`, `main()`
prints a specific rebuild instruction and exits 2. A Vulkan-capable binary with
no runtime request behaves as before and ignores Vulkan budget/timeout settings.

### 5.3 Rejected combinations

An explicit Vulkan request exits 2 before context creation when any of these is
true:

- CUDA/HIP is requested through `COLI_CUDA=1`, `COLI_GPU`, or `COLI_GPUS`;
- Metal is requested through `COLI_METAL=1`;
- `REPIN` resolves to a positive value;
- `VULKAN_EXPERT_MB` is absent, zero, malformed, or overflowing;
- `VULKAN_TIMEOUT_MS` is malformed or out of range.

HIP uses the CUDA host API and runtime switch, so `COLI_CUDA=1` covers both
compiled CUDA and compiled HIP backends. Build-time exclusion prevents a Vulkan
binary from containing either implementation.

### 5.4 Fail-closed enablement and telemetry

When Vulkan is requested, all of the following are required before inference:

- context creation succeeds for exactly one `0x1002/0x67b1` device;
- at least one startup `pin_load()` invocation materializes an expert;
- at least one complete expert triplet is published;
- context info reports three live tensors per Vulkan expert and zero pending
  operations;
- every published tensor is READY and strict-local.

Otherwise startup exits nonzero. It must not continue as an unlabelled CPU run.

After successful pinning, emit one machine-parseable line on stderr:

```text
[VULKAN] READY upload-only: vendor=0x1002 device=0x67b1 experts=N tensors=3N committed=B budget=C pending=0 compute=CPU host_copies=retained
```

`prof_config()` reports `CPU + Vulkan residency (upload-only)`, never
“Vulkan backend.” These messages are an acceptance contract and allow the
smoke test to reject silent failure to enable Vulkan.

## 6. Cleanup and failure behavior

### 6.1 Centralized normal release

After `Model m` is initialized, all normal `main()` modes assign an exit status
and jump to one `model_cleanup` label. This replaces the direct post-model
returns in score, serve, text, replay, teacher-forcing, oracle rejection, and
ordinary oracle generation paths.

`model_vulkan_release()` performs:

1. Walk `m->pin[0..n_layers]` and call `qt_vulkan_release()` for every gate,
   up, and down handle.
2. If a free reports BUSY, retain no stale caller pointer, finish the context's
   pending operation with the finite timeout, and continue only after success.
3. Treat DEVICE_LOST as terminal; do not destroy resources whose completion
   cannot be proven.
4. Require the context to report zero pending operations before destruction.
5. Call `coli_vulkan_context_destroy(&m->vulkan_context, timeout_ns)` once.
6. Clear the exit-handler registration only after the context pointer becomes
   null.

No call to `vkDeviceWaitIdle()` is added. All waits remain fence-based and
bounded.

The centralized Phase 3B release owns only Vulkan additions. Existing host
model allocations, file descriptors, CUDA/HIP shutdown behavior, Metal
shutdown behavior, and process-reclamation assumptions are not broadly
refactored in this milestone.

### 6.2 Last-resort exit cleanup

Existing loader code contains legacy `exit()` calls. After context creation,
register one process-local fallback that knows the single context. Normal
cleanup disables it. On an unexpected `exit()` it attempts the same finite
context finish/destroy operation.

If that bounded attempt returns TIMEOUT, ERROR, or DEVICE_LOST, it prints one
diagnostic and leaves the context/resources intact for process reclamation. It
must not loop without a deadline or destroy a command pool, device, fence,
buffer, or memory whose completion is unproven.

### 6.3 Result-specific startup behavior

| Result | Phase 3B behavior |
| --- | --- |
| `COLI_VULKAN_OK` | Require READY metadata; continue transaction. |
| `COLI_VULKAN_INVALID_ARGUMENT` / `UNSUPPORTED` | Roll back completed local tensors; fail requested startup. No allocation should exist for validator rejection. |
| `COLI_VULKAN_OUT_OF_MEMORY` | Roll back; fail requested startup. Do not silently reduce the tier. |
| `COLI_VULKAN_LIMIT_EXCEEDED` | Roll back the current triplet. End a nonempty prefix successfully; fail if no triplet was published. |
| `COLI_VULKAN_TIMEOUT` | Retain the returned handle/context ownership, request deferred free, make one bounded finish attempt, then fail startup if it is still incomplete. |
| `COLI_VULKAN_ERROR` | Treat a non-null returned tensor as potentially submitted; retain/free-defer/finish exactly as documented; fail startup. |
| `COLI_VULKAN_DEVICE_LOST` | Mark terminal, reject all later submissions, clear caller handles only through deferred-free semantics, retain unprovable resources until process exit, and fail startup. |
| `COLI_VULKAN_BUSY` | No parallel submit is legal in Phase 3B; treat it as an invariant failure after bounded pending cleanup. |

### 6.4 Inference-time behavior

Once successful startup reaches inference, Phase 3B submits no more Vulkan
work. No readback occurs in production. A Vulkan device fault cannot be
generated by Colibri compute because there is no Vulkan compute. Tensor handles
remain owned until centralized shutdown.

CPU inference output is therefore governed only by the retained host bytes and
existing CPU kernels. With Vulkan disabled, none of the new context, pin upload,
telemetry, or cleanup branches executes.

## 7. Exact tests and PASS, FAIL, and SKIP criteria

### 7.1 Format and arithmetic unit tests

`c/tests/test_vulkan_context.c` must cover:

- valid odd and even dimensions for every format 0 through 6;
- exact derived weight bytes, scale counts, source scale bytes, uploaded scale
  bytes, effective group size, aligned offsets, and packed size;
- every accepted format-4 group size and rejected zero, unlisted, over-`I`,
  and mismatched group sizes;
- the required `gs == 0` convention for formats 0, 1, 2, 3, 5, and 6;
- format-6 four-byte source tag with zero uploaded scale payload;
- unknown formats;
- zero dimensions and null pointers;
- every exact-length-minus-one and exact-length-plus-one case;
- checked `ceil_div`, multiplication, alignment, `size_t`, and `VkDeviceSize`
  overflow boundaries;
- rejection before `CreateBuffer`, `AllocateMemory`, command-buffer allocation,
  fence creation, or queue submission counters change;
- persisted format metadata returned by `coli_vulkan_tensor_get_info()`.

Unit tests must not SKIP when Vulkan development headers/libraries are missing;
the target fails clearly, preserving the Phase 3A acceptance rule.

### 7.2 Transaction and ownership tests

`c/tests/test_vulkan_loader.c` uses fake Vulkan and real `QT`/`ESlot` geometry to
assert:

- gate/up/down become visible only after all three report READY;
- success publishes exactly three handles and balances temporary command
  buffers and fences;
- all weight and scale host pointers remain unchanged and readable;
- validation failure in gate, up, or down publishes none and submits nothing;
- allocation, staging, begin, end, fence-create, submit, wait, and post-submit
  failure at each of the three positions rolls back every completed tensor;
- TIMEOUT retains the command buffer, fence, staging allocation, and destination
  until `coli_vulkan_finish_pending()` proves completion;
- DEVICE_LOST deliberately does not destroy unprovable command/fence/buffer
  resources and rejects later submissions;
- budget and `maxMemoryAllocationCount` boundaries never publish a partial
  expert;
- a nonempty prefix can stop at a limit, while a zero-expert result fails the
  enablement contract;
- repeated release is rejected safely and normal release reaches zero live
  resources;
- shutdown TIMEOUT/ERROR/DEVICE_LOST leaves the context pointer intact.

Ordinary cases run under ASan, UBSan, and LeakSanitizer with leak detection on.
The intentional terminal DEVICE_LOST retention case runs separately with leak
detection off, as in Phase 3A.

### 7.3 Real R9 390 hardware tests

The strict hardware target runs headless in the child environment and must:

- select exactly one `0x1002/0x67b1` device;
- report the expected three heaps and seven memory types;
- reject the 256 MiB BAR types 3 and 4 (`propertyFlags=0x0007`) for expert
  destinations;
- create only strict `DEVICE_LOCAL`, non-`HOST_VISIBLE` tensors;
- use the public validated creator for formats 0 through 6 with small bounded
  fixtures;
- verify aligned layout metadata and exact readback of uploaded weight and
  ordinary scale payloads;
- verify format-6 tag bytes were validated but not uploaded as scales;
- verify actual committed-byte and allocation-count accounting;
- finish with zero pending work and bounded clean shutdown.

Absence of the exact R9 390 is FAIL in strict mode, not SKIP.

### 7.4 Build and regression commands

The Phase 3B implementation acceptance commands are:

```sh
make -C c vulkan-strict-compile-test
make -C c vulkan-unit-test
make -C c vulkan-loader-unit-test
make -C c vulkan-sanitize-test
make -C c vulkan-loader-sanitize-test
make -C c vulkan-device-lost-sanitize-test
make -C c vulkan-hardware-test-strict
make -C c -j vulkan-test
make -C c test
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny
git diff --check
```

`vulkan-test` must explicitly invoke its sub-makes sequentially with `&&`, so
`make -j` cannot reorder them or turn a failed child into a green target.

Also run negative configuration checks and require nonzero status for:

```sh
make -C c colibri VULKAN=1 CUDA=1
make -C c colibri VULKAN=1 HIP=1
make -C c colibri VULKAN=1 METAL=1
env COLI_VULKAN=1 SNAP=/nonexistent ./c/colibri 1 4 4
```

The final command must fail before model access when the binary lacks Vulkan or
required Vulkan settings; a focused unit harness should be used instead of the
literal nonexistent model where ordering makes that assertion clearer.

### 7.5 PASS criteria

Phase 3B passes only when:

- strict `-Werror` CPU and `VULKAN=1` builds succeed;
- all format, loader, fault, sanitizer, and ordinary repository tests pass;
- no ordinary sanitizer test leaks;
- the intentional DEVICE_LOST case retains exactly the documented resources;
- strict hardware acceptance passes on the real R9 390;
- the four-token smoke test passes with a qualifying tiny model;
- Vulkan-on reports positive, internally consistent residency and no pending
  operation;
- the same Vulkan-built binary with Vulkan off preserves default CPU behavior;
- Vulkan-on and Vulkan-off produce the same four generated token IDs;
- existing CPU profiling proves routed expert rows executed on CPU and zero GPU
  compute time was used;
- `git diff --check` is clean;
- GLM-5.2 remains `NOT_RUN`.

### 7.6 FAIL criteria

Any of these is FAIL:

- validator rejection after an allocation or submission;
- accepted mismatched source length, bad group size, overflow, or unknown format;
- partial gate/up/down publication;
- loss of a host pointer or host-copy release;
- unbounded wait or `vkDeviceWaitIdle()`;
- destruction before submitted work is proven complete;
- accepted BAR or other `HOST_VISIBLE` expert memory;
- budget or allocation-count overrun;
- Vulkan enablement that continues with zero resident tensors;
- requested Vulkan silently falling back to an unlabelled CPU-only run;
- any Vulkan branch in `moe()` or `matmul_qt_ex()`;
- any output mismatch in the deterministic smoke comparison;
- any change in default CPU, CUDA, HIP, or Metal regression behavior;
- any GLM-5.2 execution.

### 7.7 SKIP criteria

- Non-strict hardware tests may SKIP when Vulkan development files or the exact
  R9 390 are unavailable, with a precise reason.
- The smoke target may SKIP in ordinary developer or CI environments only when
  the caller-supplied tiny model is absent. It must not download or generate a
  model implicitly.
- SKIP is not acceptable for the target-server Phase 3B merge gate. The tiny
  fixture decision in section 10 must be resolved and the smoke test must PASS.
- Unit/fault tests never SKIP because of missing Vulkan development files.
- The strict R9 390 target never SKIPs an absent or wrong device.

## 8. Generation smoke-test command and required model

### 8.1 Local model inventory result

No small, already-materialized, Colibri-compatible MoE checkpoint is currently
available on this server.

The inventory found:

- `/srv/claw-storage/models/huggingface/hub/models--mateogrgic--GLM-5.2-colibri-int4-with-int8-mtp/snapshots/3cc8db99b1b13fc79325d987ba3c1c430766b3b8`, which is compatible but is the
  prohibited full GLM-5.2 model and must not be used;
- Mixtral and Ternary Bonsai GGUF files, which are not compatible with
  `c/colibri.c`'s `glm_moe_dsa` safetensors loader;
- the checked-in definition and oracle for the intended tiny fixture:
  `c/tools/make_glm_oracle.py` and `c/ref_glm.json`, but no current
  `c/glm_tiny/config.json` or `c/glm_tiny/model.safetensors` artifact.

The required smoke model is therefore `c/glm_tiny`: the small random-weight
GLM-MoE-DSA fixture defined by `c/tools/make_glm_oracle.py` (hidden 128, five
layers, two sparse layers, eight routed experts, top-2). It is about 0.6 MiB in
the existing test documentation and exercises the same `model_init()`,
`expert_load()`, `pin_load()`, `moe()`, and CPU `QT` paths as the real engine.

It must be supplied or regenerated locally under separate approval before the
Phase 3B acceptance run. The smoke harness itself must not download or generate
it. The full GLM-5.2 checkpoint is never an acceptable substitute.

### 8.2 Exact top-level command

After the fixture is present, the exact acceptance command is:

```sh
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny
```

The target first builds `c/colibri` with `VULKAN=1`; it never invokes a model
download.

### 8.3 Deterministic four-token harness

`c/tests/test_vulkan_loader_smoke.py` creates a temporary test directory outside
the source model, copies the tiny fixture so `.coli_usage` cannot modify the
source, and creates:

- a truncated oracle whose `prompt_ids` are the checked-in 12-token prompt and
  whose `full_ids` contain exactly the first four continuation tokens;
- a deterministic pin file listing every expert 0 through 7 for sparse layers 3
  and 4, with stable descending counts.

Pinning all 16 routed experts makes every routed CPU expert used during the
smoke test also a Vulkan-resident expert. It expects exactly 16 published
experts and 48 Vulkan tensors.

The harness runs the same Vulkan-built binary twice. The effective commands are:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY -u COLI_VULKAN -u VULKAN_EXPERT_MB \
  SNAP="$OFF_MODEL" REF="$REF4" PIN="$PIN_FILE" PIN_GB=0.01 PIN_FILL=0 \
  AUTOPIN=0 REPIN=0 MTP=0 DRAFT=0 TEMP=0 SEED=1 PROF=1 PIPE=0 \
  OMP_NUM_THREADS=1 COLI_NO_OMP_TUNE=1 \
  ./c/colibri 64 4 4

env -u DISPLAY -u WAYLAND_DISPLAY \
  COLI_VULKAN=1 VULKAN_EXPERT_MB=64 VULKAN_TIMEOUT_MS=5000 \
  SNAP="$ON_MODEL" REF="$REF4" PIN="$PIN_FILE" PIN_GB=0.01 PIN_FILL=0 \
  AUTOPIN=0 REPIN=0 MTP=0 DRAFT=0 TEMP=0 SEED=1 PROF=1 PIPE=0 \
  OMP_NUM_THREADS=1 COLI_NO_OMP_TUNE=1 \
  ./c/colibri 64 4 4
```

`OFF_MODEL` and `ON_MODEL` are byte-identical temporary copies. `REF4` and
`PIN_FILE` are temporary files created by the harness. The positional `4 4`
causes the unquantized tiny expert tensors to use the real runtime INT4 `QT`
path. The truncated `full_ids` makes `generate()` request exactly four new
tokens; `TEMP=0`, `SEED=1`, `DRAFT=0`, and one OpenMP thread make the comparison
deterministic.

The harness passes only if:

1. both processes exit zero and generate exactly four token IDs;
2. the four token IDs are identical;
3. Vulkan-on contains exactly one READY banner with vendor/device
   `0x1002/0x67b1`, 16 experts, 48 tensors, positive committed bytes, zero
   pending operations, `compute=CPU`, and `host_copies=retained`;
4. Vulkan-off contains no Vulkan READY banner;
5. Vulkan-on profiling contains positive routed CPU rows and zero routed GPU
   critical time;
6. startup reports all 16 host pins, so the CPU rows necessarily use host copies
   of Vulkan-resident experts;
7. removing `VULKAN_EXPERT_MB`, setting `REPIN=1`, or supplying a pin file with
   no valid experts makes the Vulkan-on run exit nonzero, proving that silent
   failure to enable Vulkan is detected.

No GLM-5.2 model, active installation, inference, or benchmark is involved.

## 9. Work explicitly deferred beyond Phase 3B

- Any Vulkan shader, pipeline, descriptor, or compute dispatch.
- Any branch in `moe()`, `expert_gate_up()`, `matmul_qt()`, or
  `matmul_qt_ex()` based on a Vulkan handle.
- Activation upload, download, staging rings, persistent activation buffers,
  or fused gate/up/SiLU/down execution.
- Releasing host expert weights or scales.
- CPU rematerialization after Vulkan compute failure.
- Uploading inference-time misses, `Model.ws`, or `Model.ecache` slots.
- Vulkan-aware LRU eviction or promotion.
- Live Vulkan `REPIN`.
- Any tensor update API. A later update must allocate a new complete triplet,
  finish it, atomically swap it, then free the old triplet.
- MTP expert Vulkan residency beyond experts explicitly present in startup
  `pin_load()` input; no separate MTP acceleration.
- Dense, shared-expert, attention, router, embedding, lm-head, KV-cache, or
  residual Vulkan residency or compute.
- Multi-GPU, multiple R9 390 devices, Hawaii-family IDs other than
  `0x67b1`, device lists, peer access, and cross-device routing.
- ReBAR assumptions or use of the 256 MiB BAR for expert destinations.
- Multiple queues, queue-family ownership transfers, or concurrent Vulkan
  submission.
- Destination arenas, suballocation, compaction, defragmentation, or allocation
  pooling.
- Persistent/chunked staging optimization.
- Required `VK_EXT_memory_budget` support.
- Windows Vulkan loading.
- A generic CPU/CUDA/HIP/Metal/Vulkan tensor abstraction.
- Broad cleanup of all pre-existing CPU model allocations and worker lifetime.
- Full-model generation or benchmarking.
- GLM-5.2 execution; `benchmarks.glm_5_2` remains exactly `NOT_RUN`.

## 10. Unresolved decisions requiring approval

The following decisions must be approved before implementation. Approval of
this design means accepting decisions 1 through 5 as written.

### Decision 1: Missing tiny smoke fixture — blocking

There is no small materialized compatible MoE model on this server. Approve one
of these non-GLM-5.2-full options before the acceptance run:

1. provide a trusted existing `c/glm_tiny` artifact locally; or
2. separately approve local generation with the checked-in
   `c/tools/make_glm_oracle.py`, then preserve the generated fixture outside Git
   or in the already established ignored test location.

Recommendation: generate or provide `c/glm_tiny`; do not download a different
model and do not use the installed full GLM-5.2 checkpoint. Until this is
resolved, Phase 3B design can proceed but its merge acceptance cannot PASS.

### Decision 2: Residency format set and format-6 tag semantics

Approve formats 0 through 6 for upload-only residency, with compute eligibility
false for all of them. Approve validating format 6's exact four-byte `.qs`
source tag while uploading zero ordinary scale bytes.

Recommendation: approve. The upload layer stores opaque validated bytes and
does not need a compute kernel; excluding formats 5 or 6 would make real loader
eligibility depend on a compute feature that Phase 3B deliberately lacks.

### Decision 3: Exact source lengths in `QT`

Approve adding backend-neutral `weight_bytes` and `scale_bytes` fields to `QT`
and setting them at every construction/publication seam.

Recommendation: approve. Recomputing expected values in `qt_vulkan_spec()`
would compare the validator against its own calculation and would not validate
the actual source allocation/view length.

### Decision 4: Fail-closed runtime policy and timeout

Approve `COLI_VULKAN=1` as the explicit runtime gate,
`VULKAN_EXPERT_MB` as mandatory with no default, and
`VULKAN_TIMEOUT_MS` as optional with a 5000 ms default and a 1–60000 ms range.
Approve startup failure rather than silent CPU fallback when explicit Vulkan
initialization, validation, upload, or zero-residency checks fail.

Recommendation: approve. CPU behavior remains available by disabling Vulkan;
a Vulkan-on validation run must not claim success when it did nothing.

### Decision 5: Bounded sorted-prefix placement

Approve uploading the largest complete prefix of the existing `pin_load()`
ranking that fits actual budget and allocation limits. A limit after at least
one complete expert is a successful bounded tier; failure to publish the first
expert is a startup error. Any non-limit backend error fails startup.

Recommendation: approve. This preserves deterministic ranking, never publishes
a partial expert, and respects the explicit cap without inventing a second
expert-selection policy.

No decision is requested for Vulkan compute, host-copy release, live updates,
or multi-GPU behavior; all remain explicitly deferred.
