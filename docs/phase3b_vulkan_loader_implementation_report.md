# Phase 3B Vulkan upload-only expert-loader implementation report

## 1. Scope and result

Phase 3B is implemented on branch
`feature/vulkan-expert-upload-integration-phase3` from base commit
`13f96097671bfeb054c5bfd1afd13bd6dbbfa087`.

Result: PASS for the approved upload-only loader milestone.

The implementation adds format-aware validation and upload-only Vulkan
residency for startup-pinned Colibri experts. Inference, routed-expert
arithmetic, attention, cache misses, LRU movement, working-set movement, and
REPIN behavior remain CPU/CUDA/HIP/Metal code exactly as before. Vulkan does
not provide a compute path and is never consulted by `moe()`, `matmul_qt()`,
`matmul_qt_ex()`, or `expert_gate_up()`.

GLM-5.2 was not run. Its benchmark status remains `NOT_RUN`.

## 2. Files and implementation seams

### `c/backend_vulkan.h`

- Adds `ColiVulkanQTSpec`, carrying exact `fmt`, `I`, `O`, `gs`, host pointers,
  `weight_bytes`, and `scale_bytes`.
- Adds `ColiVulkanQTLayout`, carrying derived scale cardinality, effective
  group size, uploaded scale bytes, and aligned packed layout.
- Extends `ColiVulkanTensorInfo` with the persisted format and source metadata.
- Exposes the pure `coli_vulkan_validate_qt_spec()` validator.
- Exposes `coli_vulkan_tensor_create_qt()` as the sole production creator.
- Restricts `coli_vulkan_tensor_upload()` to `COLI_VULKAN_TESTING` builds.
- Documents retained ownership on timeout/error/device loss and the existing
  bounded finish/free/destroy contracts.

### `c/backend_vulkan.c`

- Implements overflow-checked format validation for formats 0 through 6.
- Performs all format validation before any Vulkan allocation or submission.
- Keeps the raw transport in the private `tensor_upload_raw()` function.
- Persists the validated format, geometry, source lengths, derived scale
  metadata, strict memory placement, and compute-ineligible state on every
  created tensor.
- Continues to use the Phase 3A persistent context, serialized queue and
  command pool, finite fence waits, retained pending operation, budget,
  allocation-count, and terminal device-loss machinery.

### `c/colibri.c`

- Adds backend-neutral exact `weight_bytes` and `scale_bytes` fields to `QT`.
- Sets those lengths in `qt_alloc()`, `qt_from_disk()`, the expert mmap path,
  the expert slab/pread path, and `uring_finalize_load()`.
- Whole-`QT` and whole-`ESlot` copies/swaps preserve those fields naturally.
  Existing reset paths do not replace geometry or source metadata; subsequent
  loader publication writes exact lengths again.
- Adds a Vulkan handle to `QT` without changing CUDA/HIP/Metal handle fields.
- Creates one persistent context in `main()` only after model initialization.
- Hooks Vulkan upload only into the completed startup `pin_load()` population.
- Validates gate/up/down as a complete triplet before allocation, uploads to
  local handles, verifies all three READY and strict-local, and publishes the
  three handles in one startup-thread transaction.
- Retains every host pointer and source allocation.
- Uploads the largest complete sorted pin prefix allowed by the explicit VRAM
  budget and `maxMemoryAllocationCount`; a zero-expert tier fails closed.
- Adds centralized bounded and idempotent `model_vulkan_release()` cleanup.
- Registers a bounded process-exit fallback. Timeout, error, or device loss
  never triggers destruction whose completion cannot be proven.
- Prints one machine-checkable READY banner only after complete residency
  verification, and labels execution as CPU plus upload-only Vulkan residency.

### `c/Makefile`

- Preserves the Phase 1 staged-upload hardware test.
- Adds strict production CPU/Vulkan compilation and a symbol check proving the
  raw creator is absent from the production backend object.
- Supplies `-Werror` through the separate additive `EXTRA_WARNINGS` variable,
  so recursive strict builds retain the Makefile's backend-specific flags.
- Runs a preprocessor macro probe with the exact resolved Vulkan strict-build
  `CFLAGS`; the visible command and probe require both `-Werror` and
  `-DCOLI_VULKAN` before the raw production-symbol check can pass.
- Adds loader unit, fault, sanitizer, terminal device-loss, hardware, and
  generation-smoke targets.
- Keeps `vulkan-test` explicitly sequential and fail-fast under `make -j`.
- Rejects `VULKAN=1` combined with CUDA, CUDA DLL, HIP, or Metal at make time.

### Tests

- `c/tests/test_vulkan_context.c`: all format formulas, odd dimensions,
  supported grouped sizes, source-size mismatches, tags, invalid formats,
  overflow/range checks, validation-before-allocation, and metadata persistence.
- `c/tests/test_vulkan_loader.c`: the real Colibri publication seam with fake
  Vulkan success, rollback, fault matrix, timeout retention and completion,
  allocation/budget prefix limits, idempotent cleanup, and terminal device loss.
- `c/tests/test_backend_vulkan.c`: real R9 390 creation/readback for formats
  0 through 6 through the validated public creator.
- `c/tests/test_vulkan_loader_smoke.py`: deterministic four-token Vulkan-off/on
  comparison, positive CPU-routed rows, zero GPU compute, exact residency
  accounting, retained host pins, and fail-closed negative cases. An explicitly
  supplied incomplete/missing fixture is a hard failure; only an invocation
  with no model specified may SKIP.
- `c/tools/clean.py`: removes the two new loader test binaries.

## 3. Format validation contract

All counts use checked unsigned arithmetic, require positive CPU-representable
`I` and `O`, and require exact host pointers and source lengths.

- Format 0: FP32, `weight_bytes = O * I * 4`, no scale source, `gs = 0`.
- Format 1: INT8 row, `weight_bytes = O * I`, `scale_count = O`,
  `scale_bytes = O * 4`, `gs = 0`.
- Format 2: packed INT4 row, `weight_bytes = O * ceil(I / 2)`,
  `scale_count = O`, `scale_bytes = O * 4`, `gs = 0`.
- Format 3: packed INT2 row, `weight_bytes = O * ceil(I / 4)`,
  `scale_count = O`, `scale_bytes = O * 4`, `gs = 0`.
- Format 4: grouped INT4, format-2 weight bytes,
  `scale_count = O * ceil(I / gs)`, `scale_bytes = scale_count * 4`;
  `gs` must be one of 16, 32, 48, 64, 96, 128, 192, or 256 and cannot exceed
  `I`.
- Format 5: INT3-g64, `weight_bytes = O * ceil(I / 64) * 24`,
  `scale_count = O * ceil(I / 64)`, `scale_bytes = scale_count * 4`, and
  effective group size 64.
- Format 6: E8/IQ3, `weight_bytes = O * ceil(I / 256) * 98`; its `.qs` source
  must exist and be exactly four bytes, but ordinary uploaded scale bytes are
  zero. Effective group size is 256.

Weight and uploaded-scale offsets are packed using the physical device's
`minStorageBufferOffsetAlignment`; all offset and packed-size arithmetic is
overflow checked. The backend records `compute_eligible = 0` for every format.

## 4. Runtime and ownership behavior

Vulkan is off by default. A production run requests it only with exact
`COLI_VULKAN=1` and an explicit positive integer `VULKAN_EXPERT_MB`.
`VULKAN_TIMEOUT_MS` defaults to 5000 and accepts only 1 through 60000.

The Vulkan build rejects runtime CUDA/GPU or Metal settings and requires
`REPIN=0`. The makefile rejects Vulkan combined with CUDA, CUDA DLL, HIP, or
Metal builds.

Only experts selected by startup `PIN`/`pin_load()` are candidates. Each
expert owns three independent strict `DEVICE_LOCAL`, non-`HOST_VISIBLE`
Vulkan buffers/allocations. Host `QT` storage remains authoritative for CPU
inference and is retained for the process lifetime.

The gate/up/down publication sequence is:

1. Build three exact specs from loader metadata.
2. Validate all three without Vulkan work.
3. Create and finish three local tensor handles sequentially.
4. Verify READY state, exact metadata, strict-local flags, context usability,
   and zero pending work.
5. Publish all three handles together and update expert/tensor counts.

Any failure before step 5 publishes nothing. Safe resources are released in
reverse order. Submitted resources remain context-owned until a bounded finish
proves completion. Device loss marks the context unusable, rejects later
submissions, and deliberately retains resources whose completion cannot be
proved until process exit.

## 5. Tiny smoke fixture

The model was generated locally from random weights using only the checked-in
generator. No model was downloaded. Generator dependencies were installed in
an isolated `/tmp/coli-phase3b-generator-venv`; no repository dependency or
active Colibri/GLM installation was changed.

Generation command, run from `c/`:

```sh
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/coli-phase3b-generator-venv/bin/python tools/make_glm_oracle.py
```

Generator runtime versions:

- `transformers=5.14.1`
- `torch=2.13.0+cpu`
- `safetensors=0.8.0`

Generated/verified files:

- `c/glm_tiny/config.json`: 1,566 bytes,
  SHA-256 `482e9dd3275a2c625caf51023bae0c2c3003948c68c1b33ede39eb6981f3ee84`.
- `c/glm_tiny/model.safetensors`: 2,461,840 bytes,
  SHA-256 `24515afee76156b0556930fefdc01ba613a133b1e7b89da16f67448e6c97e799`.
- `c/ref_glm.json`: 391 bytes,
  SHA-256 `6b07bf6c1d4a840ccc585b74122dddc6dc0e4ebd0e59866f757115a0ae9a7188`.

The regenerated `c/ref_glm.json` is byte-identical to the tracked file.
`c/glm_tiny/` remains gitignored and is not included in the review patch.

The four-token smoke uses the same Vulkan-built binary with Vulkan off and on,
runtime INT4 expert/dense conversion, two byte-identical temporary model
copies, one thread, greedy sampling, fixed seed, and the first four continuation
positions. Both modes generated exactly:

```text
197 197 197 197
```

The smoke comparison intentionally tests behavioral equivalence between the
same runtime-quantized CPU inference path with and without Vulkan residency;
the BF16 generator oracle is not used as an INT4 quality assertion.

Vulkan-on verified exactly 16 startup-pinned experts, 48 published tensors,
positive committed bytes, a 64 MiB effective cap, zero pending work, retained
host copies, positive routed CPU rows, and zero routed GPU critical time.

## 6. Test commands and exit statuses

All final acceptance commands below returned exit status 0.

The final hardware and generation-smoke acceptance evidence is exclusively
from the fresh reruns performed after the R9 390 was confirmed clear: the
Ollama service was inactive, no `llama-server` process existed, and baseline
VRAM use was approximately 4 MiB. Hardware or smoke results obtained before
that exclusive-card confirmation are deliberately excluded from final
acceptance evidence.

```text
make -C c vulkan-strict-compile-test                         0
make -C c vulkan-unit-test                                  0
make -C c vulkan-loader-unit-test                           0
make -C c vulkan-sanitize-test                              0
make -C c vulkan-loader-sanitize-test                       0
make -C c vulkan-device-lost-sanitize-test                  0
make -C c vulkan-hardware-test-strict                       0
make -C c -j vulkan-test                                    0
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny  0
make -C c test                                              0
git diff --check                                            0
```

The final consolidated acceptance-test correction rerun returned:

```text
make -C c vulkan-strict-compile-test                         0
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny  0
make -C c vulkan-loader-smoke \
  VULKAN_SMOKE_MODEL=/tmp/phase3b-explicitly-missing-model  2 (expected failure)
git diff --check                                            0
```

The strict-build log shows `-Werror -DCOLI_VULKAN` on the actual production
`backend_vulkan.c` and `colibri.c` compile commands. Its preprocessor probe
also verifies `#define COLI_VULKAN 1`. The production object continues to pass
the check that `coli_vulkan_tensor_upload` is not globally exported.

For smoke fixture policy, a direct developer invocation with no `--model` and
no `VULKAN_SMOKE_MODEL` returned 0 with
`SKIP: no Vulkan smoke fixture was specified`. The acceptance Makefile target
with an explicitly nonexistent path returned 2 and printed a hard `FAIL`.

Ordinary sanitizer tests used
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`. The two intentional terminal DEVICE_LOST
retention cases ran separately with `detect_leaks=0`.

The full regression's final run was outside the restricted sandbox because 42
HTTP/socket tests cannot create loopback sockets inside it. The unrestricted
run passed all C tests and 243 Python tests with 32 expected skips.

## 7. Real R9 390 evidence

The final exclusive-card strict hardware rerun reported:

```text
device vendor=0x1002 device=0x67b1 queueFamily=0
limits maxMemoryAllocationCount=4294967295 minStorageBufferOffsetAlignment=4
budget requested=67108864 effective=67108864 reserve=1073741824
memory heaps=3 types=7
```

Formats 0 through 6 all allocated with memory type 0, heap 0, flags `0x0001`
(`DEVICE_LOCAL`, not `HOST_VISIBLE`), used aligned packed layouts, and passed
exact readback. The final hardware line was:

```text
PASS: persistent Vulkan Phase 3B validated QT hardware lifecycle
```

The legacy staged-upload test also printed `PASS`. The same sequence passed
under `make -j vulkan-test`, proving the explicit child-make chain remains
sequential and fail-fast.

The final exclusive-card smoke rerun also returned status 0 and printed:

```text
PASS: Phase 3B four-token Vulkan-on/off CPU-path smoke test
tokens: 197 197 197 197
residency: 16 experts, 48 tensors, host copies retained, compute CPU
```

## 8. Negative and fail-closed checks

The following expected failures returned nonzero status 2:

```text
make -C c colibri VULKAN=1 CUDA=1
make -C c colibri VULKAN=1 HIP=1
make -C c colibri VULKAN=1 METAL=1
env COLI_VULKAN=1 SNAP=/nonexistent ./c/colibri 1 4 4   # CPU build
env COLI_VULKAN=1 SNAP=/nonexistent ./c/colibri 1 4 4   # Vulkan build, no cap
```

The messages identify the backend conflict, CPU-only build, or missing explicit
`VULKAN_EXPERT_MB` before model access. The smoke harness additionally requires
nonzero exit for `REPIN=1` and for a pin file that yields zero valid resident
experts.

The fake loader matrix covers allocation, binding, mapping, command allocation,
command begin/end, fence creation, queue submission, and fence wait failures at
every gate/up/down occurrence. No ordinary failure leaves a live fake resource
or partially published expert.

## 9. Compatibility and deferred work

- Default CPU behavior is unchanged when `COLI_VULKAN` is absent or zero.
- Vulkan remains compile-time and runtime opt-in.
- CUDA, HIP, and Metal data members and behavior are preserved.
- No host expert copy is released.
- No Vulkan compute object, shader, pipeline, descriptor, dispatch, activation
  transfer, LRU policy, miss upload, live REPIN, update/swap, or multi-GPU path
  is introduced.
- Allocation remains one buffer/allocation per QT.
- Only one exact `0x1002/0x67b1` device is accepted by the persistent context.
- Later compute integration must not repurpose these upload-only handles until
  a validated CPU fallback and complete compute lifecycle are designed.

## 10. Repository state and artifacts

No commit, push, pull request, merge, or rebase was performed. The generated
model is ignored and excluded from source review. Build products were cleaned
after acceptance. The source/test review patch is
`docs/phase3b_code_review.patch`.
