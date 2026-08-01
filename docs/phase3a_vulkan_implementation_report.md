# Phase 3A Vulkan Foundation Implementation Report

## 1. Result and scope

Phase 3A is implemented and passes compilation, unit/fault-injection,
repository regression, and real AMD R9 390 hardware tests.

- Branch: `feature/vulkan-expert-upload-integration-phase3`
- Base HEAD before these uncommitted changes:
  `b67a1cb1cae6448b8076334d35a68e8bc2be48a9`
- Scope: persistent Vulkan context plus strict-local tensor allocation, staged
  upload, diagnostic readback, synchronization, accounting, and destruction.
- `c/colibri.c` is unchanged. There is no loader integration and no Vulkan
  dispatch from `moe()`.
- The implementation remains opt-in through `VULKAN=1`.
- GLM-5.2 was not run, modified, or accessed.
- Nothing has been committed, pushed, or merged.

This is foundation-only. Phase 3B upload-only expert-loader integration is not
part of this change.

## 2. Approved decisions implemented

1. Phase 3A does not modify `c/colibri.c`.
2. Context creation accepts exactly one physical device whose IDs are AMD
   `vendorID=0x1002` and Hawaii `deviceID=0x67b1`. Zero matches returns
   `UNAVAILABLE`; more than one exact match returns `UNSUPPORTED`.
3. `VULKAN_EXPERT_MB` is mandatory and has no default. Missing, empty, zero,
   malformed, or overflowing values are rejected. The fixed VRAM reserve is
   1024 MiB. Hardware and fake-device tests explicitly use 64 MiB.
4. Each `ColiVulkanTensor` owns one `VkBuffer` and one `VkDeviceMemory`
   allocation. Phase 3B will create one such tensor per QT.
5. The Makefile rejects `VULKAN=1` combined with any of `CUDA=1`,
   `CUDA_DLL=1`, `HIP=1`, or `METAL=1`.
6. No Phase 3B loader work is included.
7. Cleanup is centralized through explicit tensor and context destruction,
   with one cleanup path for every pre-submit failure and retained-operation
   handling for every post-submit failure.
8. No in-place tensor update API was added. A future update must allocate a
   new tensor, finish its upload, atomically swap the owning reference, and
   only then free the old tensor.
9. No Vulkan compute or `moe()` dispatch exists.

## 3. Files changed

### `c/backend_vulkan.h`

Defines the Phase 3A public foundation API and its ownership contracts:

- opaque `ColiVulkanContext` and `ColiVulkanTensor` owners;
- explicit `ColiVulkanConfig`;
- result and tensor-state enums;
- checked tensor layout and strict-memory selection helpers;
- context creation, inspection, and destruction;
- diagnostic readback, tensor inspection/destruction, and explicit retained
  operation handling;
- bounded finishing of retained operations;
- test-only injectable Vulkan dispatch under `COLI_VULKAN_TESTING`;
- exact Hawaii IDs and the fixed 1024 MiB reserve.

The existing low-level staged-upload interface remains available and now
distinguishes `VK_ERROR_DEVICE_LOST` from timeout and generic failure. Its
contract now requires a sufficiently large destination created with
`VK_BUFFER_USAGE_TRANSFER_DST_BIT`.

The raw byte-oriented `coli_vulkan_tensor_upload()` declaration is visible only
under `COLI_VULKAN_INTERNAL` or `COLI_VULKAN_TESTING`. Its comments pin the
non-NULL tensor behavior after post-submit timeout/error/device loss. The
readback, deferred tensor-free, and context-destroy pointer/resource contracts
are also explicit in this header.

### `c/backend_vulkan.c`

Implements the persistent context, tensor lifecycle, checked allocation and
layout logic, staged transfer/readback, synchronization, resource accounting,
timeout retention, degraded recovery, and terminal device-loss behavior.

### `c/tests/test_vulkan_context.c`

Adds a Vulkan-dispatch fake for deterministic unit and fault-injection testing
without a physical GPU. It covers configuration, layout arithmetic, memory
selection, device selection, partial initialization, lifecycle, limits,
budgeting, upload/readback error paths, timeouts, recovery, and device loss.

### `c/tests/test_backend_vulkan.c`

Reworks the hardware test around the persistent Phase 3A API. It validates the
host memory topology and performs three strict-local allocation, staged upload,
exact readback, and destruction cycles.

### `c/Makefile`

Adds the opt-in Vulkan object/link path, incompatible-backend rejection, the
unit/fault and split sanitizer targets, non-strict developer hardware target,
strict target-server hardware target, and fail-fast aggregate target. Missing
Vulkan development support is a hard failure for unit and strict acceptance.
The hardware child processes are headless and receive the explicit test budget.
The existing Phase 1 staged-upload hardware test remains in both hardware
targets. `vulkan-test` has an explicit recipe that runs the unit target followed
by strict hardware with `&&`, so `make -j` cannot start them concurrently.

### `c/tools/clean.py`

Adds the Phase 3A objects and test executables to repository cleanup.

`c/colibri.c` has no diff.

## 4. Persistent context and synchronization

`coli_vulkan_context_create()` creates one process-wide context containing:

- one `VkInstance`;
- the unique exact `0x1002/0x67b1` physical device;
- one logical `VkDevice`;
- queue 0 from the lowest-index compute-capable queue family;
- one persistent transient/resettable `VkCommandPool`;
- cached physical-device properties and memory properties;
- a tensor registry, resource counters, budget counters, and one retained
  operation slot.

A process-wide gate prevents a second live Vulkan context. The gate remains
closed after terminal device loss because the lost context and its unprovable
resources must remain alive until process exit.

One context mutex provides Vulkan external synchronization for the queue and
command pool. Allocation, command-buffer allocation/free, recording, submit,
fence waits, retained-operation finishing, tensor destruction, and context
destruction all occur while holding that mutex. Phase 3A allows only one
outstanding retained operation, so a pending operation rejects new submissions
with `BUSY`.

Every successful submission is followed by a bounded fence wait. No command
buffer, fence, staging resource, tensor destination, command pool, device, or
instance is destroyed until its submitted work is known complete. Context
destruction first finishes the retained operation with the caller's bounded
timeout, then destroys tensors and persistent owners. It does not use an
unbounded wait.

## 5. Tensor ownership and layout

Each successful internal/test-only `coli_vulkan_tensor_upload()` call returns
an opaque raw-transport tensor that is registered with its context and owns:

- one storage/transfer `VkBuffer`;
- one strict-local `VkDeviceMemory` allocation;
- packed weight and scale layout metadata;
- the allocation size and selected memory type/heap;
- a lifecycle state.

The caller owns the tensor handle until `coli_vulkan_tensor_free()` or
successful context destruction. A free request during a retained operation
clears the caller's handle but defers physical destruction until the fence is
proven complete. Successful context destruction invalidates every tensor handle
owned by that context.

`coli_vulkan_plan_tensor_layout()` uses overflow-checked unsigned arithmetic.
The weight offset is zero. The scale offset is rounded up to
`minStorageBufferOffsetAlignment`, and the final packed size is rounded up to
the same alignment. Conversion to both `VkDeviceSize` and `size_t` is checked
before allocation or copying. Zero weight size and zero alignment are rejected.

The Phase 3A API creates one allocation per tensor/QT rather than suballocating.
No mutation or replacement operation exists, which preserves the approved
future allocate-new, finish, atomic-swap, free-old sequence.

Phase 3A does not claim that raw `weight_bytes` and `scale_bytes` constitute a
valid `QT`. Format-aware `fmt`, `I`, `O`, `gs`, derived scale-count, source
length, and format-eligibility validation belongs in Phase 3B because that is
the first phase that sees real loader/QT metadata. Adding unvalidated metadata
to the transport object in Phase 3A would not establish correctness.

This deferral cannot be bypassed by ordinary Phase 3B loader code: the raw
upload declaration is hidden unless backend-internal or test compilation is
explicitly enabled. Before loader integration, Phase 3B must add a public
format specification and validator that computes format-specific layout with
overflow checks and validates all source lengths. That validated entry point
must be the only production tensor-creation API visible to the loader. The
architecture document now records this as a Phase 3B merge blocker.

## 6. Strict memory, reserve, budget, and allocation limits

Destination selection requires all of the following:

- the buffer memory requirements include the candidate type;
- the type contains `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`;
- the type does not contain `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`;
- its heap contains `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`;
- the heap is larger than the fixed 1024 MiB reserve;
- the actual allocation requirement fits above that reserve.

This explicitly excludes the R9 390's 256 MiB BAR types 3 and 4, whose flags
are `0x0007` (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`). It also excludes
HOST_VISIBLE system memory. The hardware run selected type 0 on heap 0 with
flags `0x0001`, confirming strict DEVICE_LOCAL, non-HOST_VISIBLE VRAM.

The effective expert budget is the smaller of the explicit
`VULKAN_EXPERT_MB` value and strict-local heap size minus the fixed reserve.
Committed bytes are charged using `VkMemoryRequirements.size`, not requested
buffer size. The budget is checked with subtraction-based overflow-safe
arithmetic before allocation.

`maxMemoryAllocationCount` is read from
`VkPhysicalDeviceProperties.limits.maxMemoryAllocationCount`. A zero limit is
rejected. The context tracks every destination allocation plus every transient
or retained staging allocation and checks the queried limit before each
`vkAllocateMemory`. The unit suite pins the boundary where one destination
allocation leaves no slot for staging.

## 7. Upload, timeout, and error lifecycle

Upload data is packed in host memory with zeroed alignment padding, copied to a
HOST_VISIBLE and HOST_COHERENT staging allocation, and transferred with
`vkCmdCopyBuffer` into the strict-local tensor. The persistent context owns the
queue and command pool; only staging resources and one command buffer/fence are
per operation.

Pre-submit failures clean up all objects created by that attempt and roll back
all counters. Generic errors after a successful submit cannot prove completion,
so the operation is retained, the context becomes degraded/unusable, and new
submissions are rejected. `coli_vulkan_finish_pending()` can restore the
context after a later successful fence wait.

Timeout retains the fence, command buffer, staging buffer/allocation, and the
referenced destination tensor. Repeated bounded waits may return `TIMEOUT`
without freeing anything. Only a later `VK_SUCCESS` releases the retained
operation, updates the tensor state, applies a deferred free, and permits new
submissions.

Diagnostic readback uses the same external synchronization, allocation-count
enforcement, bounded fence rule, retention model, and centralized cleanup. A
timed-out readback must first be finished; because the caller's output is not
owned by the retained operation, the caller then starts a new readback to copy
the bytes.

## 8. Terminal `VK_ERROR_DEVICE_LOST` behavior

`VK_ERROR_DEVICE_LOST` is a terminal state for the context:

- `device_lost=1`, `usable=0`, and degraded state are recorded;
- all later upload and readback submissions are rejected;
- a submitted operation whose completion is unprovable remains retained;
- tensor free becomes a deferred logical release rather than Vulkan resource
  destruction;
- context destruction returns `DEVICE_LOST` and leaves the context pointer,
  command pool, device, instance, tensors, staging objects, command buffer, and
  fence intact;
- the process-wide context gate remains held.

The intentional process-lifetime retention avoids freeing resources that the
driver could still reference. Errors occurring before any submission clean up
objects from that unsubmitted attempt; there is no in-flight work in that case.

## 9. Unit and fault-injection coverage

The fake Vulkan implementation asserts:

- missing, zero, malformed, and explicit 64 MiB budget handling;
- aligned weight/scale offsets, final padding, zero alignment rejection, and
  integer overflow rejection;
- strict-local selection, fixed reserve, and BAR rejection;
- zero, one, and multiple exact Hawaii device matches;
- partial initialization cleanup for instance enumeration, device creation,
  and command-pool creation failures;
- single persistent context and deterministic compute queue selection;
- upload/readback byte identity and zero final accounting;
- actual allocation-size budget enforcement;
- `maxMemoryAllocationCount` enforcement and rollback;
- upload failure cleanup for destination/staging buffer creation, destination/
  staging allocation, binding, mapping, command-buffer allocation/begin/end,
  fence creation, and queue submission;
- readback failure cleanup for staging buffer creation/allocation/binding,
  command-buffer allocation/begin/end, fence creation, queue submission,
  mapping, and accounting;
- upload and readback timeout retention, repeated timeout, eventual completion,
  and deferred tensor destruction;
- recovery after a generic post-submit wait error;
- terminal device loss, submission rejection, and deliberate non-destruction
  of unprovable resources.

The ordinary tests and fault injections run in one process under ASan, UBSan,
and LeakSanitizer with `detect_leaks=1`; they pass without leaks. The intentional
terminal `DEVICE_LOST` case runs as a separate process using the same sanitized
binary with only leak detection disabled. Address and undefined-behavior checks
remain enabled in both processes. This split ensures an accidental leak in any
ordinary cleanup path fails acceptance. The fake Vulkan layer also checks live
owner counts before every reset and at ordinary-process exit, so resetting the
fake cannot conceal a missing Vulkan destroy/free call.

## 10. Real R9 390 evidence

Command:

```text
make -C c vulkan-hardware-test-strict
```

Exit status: `0`

Exact acceptance output:

```text
make: Entering directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
make tests/test_vulkan_staged_upload && \
	env -u DISPLAY -u WAYLAND_DISPLAY ./tests/test_vulkan_staged_upload && \
	make tests/test_backend_vulkan && \
	env -u DISPLAY -u WAYLAND_DISPLAY VULKAN_EXPERT_MB=64 \
	./tests/test_backend_vulkan --strict
make[1]: Entering directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
make[1]: 'tests/test_vulkan_staged_upload' is up to date.
make[1]: Leaving directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
PASS
make[1]: Entering directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
make[1]: 'tests/test_backend_vulkan' is up to date.
make[1]: Leaving directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
device vendor=0x1002 device=0x67b1 queueFamily=0
limits maxMemoryAllocationCount=4294967295 minStorageBufferOffsetAlignment=4
budget requested=67108864 effective=67108864 reserve=1073741824
memory heaps=3 types=7
tensor[0] packed=4612 allocation=4624 weightOffset=0 scaleOffset=4096 memoryType=0 heap=0 flags=0x0001
tensor[1] packed=4612 allocation=4624 weightOffset=0 scaleOffset=4096 memoryType=0 heap=0 flags=0x0001
tensor[2] packed=4612 allocation=4624 weightOffset=0 scaleOffset=4096 memoryType=0 heap=0 flags=0x0001
PASS: persistent Vulkan Phase 3A hardware lifecycle
make: Leaving directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
```

The hardware test additionally asserted three heaps, seven memory types, heap 2
size 268435456 bytes, and types 3 and 4 on heap 2 with flags `0x0007`. It ran
three complete allocation/upload/exact-readback/free cycles, and all context
counters returned to zero after each cycle.

Strict-mode negative validation inside the restricted command sandbox saw only
software `llvmpipe` and returned exit 1 with
`FAIL: strict hardware acceptance requires exact AMD R9 390 0x1002/0x67b1`.
The acceptance run above was executed headlessly with access to the real R9 390
and returned exit 0.

## 11. Commands and exit statuses

All acceptance commands below were run without GLM-5.2.

| Command | Status | Result |
| --- | ---: | --- |
| `gcc -std=gnu11 -Wall -Wextra -Werror -pthread $(pkg-config --cflags vulkan) -fsyntax-only c/backend_vulkan.c` | 0 | Production backend warning-clean syntax check |
| `gcc -std=gnu11 -Wall -Wextra -Werror -pthread -DCOLI_VULKAN_TESTING $(pkg-config --cflags vulkan) c/tests/test_vulkan_context.c c/backend_vulkan.c -o /tmp/test_vulkan_context_werror $(pkg-config --libs vulkan)` | 0 | Unit/fault warning-clean compile |
| `gcc -std=gnu11 -Wall -Wextra -Werror -pthread -DCOLI_VULKAN_TESTING $(pkg-config --cflags vulkan) c/tests/test_backend_vulkan.c c/backend_vulkan.c -o /tmp/test_backend_vulkan_werror $(pkg-config --libs vulkan)` | 0 | Hardware-test warning-clean compile |
| `make -C c vulkan-unit-test` | 0 | Ordinary unit/fault PASS, then separate terminal DEVICE_LOST PASS |
| `make -C c vulkan-unit-test PKG_CONFIG=false` | 2 | Expected explicit FAIL for missing Vulkan development support |
| `make -C c vulkan-sanitize-test` | 0 | Ordinary ASan/UBSan tests PASS with leak detection enabled |
| `make -C c vulkan-device-lost-sanitize-test` | 0 | Isolated terminal-retention PASS with leak detection disabled |
| `env -u DISPLAY -u WAYLAND_DISPLAY VULKAN_EXPERT_MB=64 ./c/tests/test_backend_vulkan --strict` inside the restricted sandbox | 1 | Expected strict failure without the R9 390 |
| `make -C c vulkan-hardware-test-strict` | 0 | Phase 1 staging PASS and strict Phase 3A real Hawaii PASS |
| `make -C c -j vulkan-test` | 0 | Unit processes completed before strict hardware began |
| `make -C c -j vulkan-test PKG_CONFIG=false` | 2 | Expected fail-fast unit failure; hardware submake was not invoked |
| `make -C c colibri VULKAN=1` | 0 | Opt-in engine links the persistent Vulkan foundation |
| `make -C c colibri VULKAN=0` | 0 | Default CPU engine still compiles |
| `make -C c -n colibri VULKAN=1 CUDA=1` | 2 | Expected Makefile rejection |
| `make -C c -n colibri VULKAN=1 CUDA_DLL=1` | 2 | Expected Makefile rejection |
| `make -C c -n colibri VULKAN=1 HIP=1` | 2 | Expected Makefile rejection |
| `make -C c -n colibri VULKAN=1 METAL=1` | 2 | Expected Makefile rejection |
| `make -C c test` | 0 | Full combined regression PASS; Python: 243 tests, 32 skips |
| `git diff --check` | 0 | No whitespace errors in tracked diff |

The four expected Makefile failures all emitted:

```text
VULKAN=1 cannot be combined with CUDA=1, CUDA_DLL=1, HIP=1, or METAL=1
```

The combined repository regression ended with:

```text
Ran 243 tests in 75.775s

OK (skipped=32)
make: Leaving directory '/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c'
```

## 12. PASS, FAIL, and SKIP behavior

Phase 3A hardware PASS requires:

- the exact device and captured heap/type topology;
- the explicit 64 MiB test cap and fixed 1024 MiB reserve;
- valid queried allocation/alignment limits;
- strict DEVICE_LOCAL non-HOST_VISIBLE placement outside heap 2;
- aligned packed offsets;
- exact weight and scale byte readback for every iteration;
- safe destruction and zero final accounting.

FAIL is returned for configuration errors, multiple matching devices, topology
or placement mismatch, allocation/budget limit violation, Vulkan operation
failure, byte mismatch, unsafe cleanup state, or nonzero final accounting.

Unit tests never SKIP: missing Vulkan development support is FAIL. The generic
developer hardware target may SKIP when support or the exact R9 390 is absent.
The strict target-server hardware target never accepts that device absence as a
skip; it returns FAIL. A multiple-R9-390 host fails context selection as
unsupported. The real strict acceptance run produced PASS.

## 13. Deferred work and remaining risks

Explicitly deferred to Phase 3B or later:

- changes to `c/colibri.c`, `backend_loader.c`, `st.h`, or expert-loading call
  paths;
- construction of one Vulkan tensor for each real model QT;
- format-aware `fmt`, `I`, `O`, `gs`, derived scale-count, exact source-length,
  overflow, and format-eligibility validation;
- ownership wiring between loader/model/tensor metadata;
- allocate-new/finish/atomic-swap/free-old updates;
- eviction, LRU policy, or asynchronous multi-operation scheduling;
- Vulkan expert compute kernels and any dispatch from `moe()`;
- CPU fallback for Vulkan-resident experts;
- multi-GPU support;
- general AMD or non-Hawaii device support;
- ReBAR-dependent behavior.

The intentional device-loss retention consumes its objects until process exit.
This is the approved safety behavior, not a recoverable context path. Phase 3B
must treat `DEVICE_LOST` as terminal and must not attempt to rebuild or swap in
new Vulkan tensors in the same process.

Diagnostic readback is a test/verification mechanism, not a production expert
data path. A readback timeout requires finishing the retained operation and
issuing a fresh readback because the context does not retain caller-owned output
memory.

Phase 3B must not call the Phase 3A raw upload transport directly. It cannot
merge until the format-aware public tensor-creation contract is implemented and
tested for every eligible QT format; unsupported formats must fail before any
Vulkan allocation or submission.

## 14. Final readiness statement

Phase 3A meets the approved foundation milestone and its mandatory safety
requirements. Compilation, unit tests, upload/readback fault injection,
sanitizers, the repository regressions, and the real R9 390 hardware lifecycle
all pass. Phase 3B should not begin until this uncommitted diff and report have
been reviewed and explicitly approved for commit.
