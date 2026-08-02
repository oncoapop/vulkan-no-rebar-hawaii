# Phase 4A Vulkan routed-down compute implementation report

## Status and scope

Implementation and the requested acceptance matrix are complete on merged Phase 3B base
`eabc956e22e492e3cfeeac84847396be917e7c5f`. Nothing has been committed,
pushed, merged, or submitted as a pull request. GLM-5.2 was not run. No model,
compiler, package, or other dependency was downloaded or installed.

The implementation is limited to the approved format-2, startup-resident,
routed down projection. Gate, up, SiLU, and the gate/up product remain CPU
operations. The Vulkan result is written into `hh` and is the value consumed by
the existing weighted residual accumulation. The validation reference is held
in separate storage and is never substituted for the Vulkan output.

## Shader compiler gate and reproducible artifact

The required pre-edit compiler gate ran before source modification and passed.

Command:

```text
command -v glslangValidator
```

Exit status: `0`

Output:

```text
/usr/bin/glslangValidator
```

Command:

```text
glslangValidator --version
```

Exit status: `0`

Complete output:

```text
Glslang Version: 11:15.1.0
ESSL Version: OpenGL ES GLSL 3.20 glslang Khronos. 15.1.0
GLSL Version: 4.60 glslang Khronos. 15.1.0
SPIR-V Version 0x00010600, Revision 1
GLSL.std.450 Version 100, Revision 1
Khronos Tool ID 8
SPIR-V Generator Version 11
GL_KHR_vulkan_glsl version 100
ARB_GL_gl_spirv version 100
```

Exact generation commands:

```text
glslangValidator -V --target-env vulkan1.0 -S comp -o /tmp/vulkan_qt_fmt2_down.spv c/shaders/vulkan_qt_fmt2_down.comp
python3 c/tools/spv_to_header.py /tmp/vulkan_qt_fmt2_down.spv c/shaders/vulkan_qt_fmt2_down_spv.h --source c/shaders/vulkan_qt_fmt2_down.comp --compiler 'glslangValidator 11:15.1.0; target vulkan1.0'
```

Hashes and artifact size:

```text
GLSL SHA-256  e558544f904249ec276dac7eadd7a1682d25b000e65d82dbd4ea43766da51f00
SPIR-V SHA-256 2823e6c93ef3cd3007f08dc4563532a52ae07ee0c8805a51951e74369c1282dd
SPIR-V size 3928 bytes
```

`make -C c vulkan-shader-verify` pins the complete first version line,
regenerates SPIR-V 1.0 and the header in a temporary directory, and
byte-compares the generated header with the production artifact. The
production process consumes only the checked-in header and never compiles GLSL
at runtime.

## Implementation summary

- Added a context-owned optional compute shader module, descriptor layout,
  pipeline layout, compute pipeline, descriptor pool/set, and two persistently
  mapped scratch buffers.
- Scratch selection requires `HOST_VISIBLE | HOST_COHERENT`, rejects
  `DEVICE_LOCAL`, and rejects a device-local heap, excluding the 256 MiB BAR.
- Added format-2 eligibility derived from validated metadata and strict-local
  allocation state.
- Added an opaque synchronous format-2 tensor matmul API. One context mutex
  covers mapped scratch, descriptors, queue submission, command-pool access,
  pending work, and counters.
- Added explicit host-write to shader-read and shader-write to host-read buffer
  barriers.
- Each dispatch owns one command buffer and fence. Submitted timeout/error or
  device loss retains them, the tensor, descriptor/pipeline/scratch, command
  pool, queue, and device until bounded completion is proven. Device loss is
  terminal and no later finish/destruction attempt releases unprovable owners.
- Added race-free recorded/submitted/completed/row/error counters and fake
  two-thread serialization coverage.
- Added exact compute runtime gates, complete startup-tier validation, a
  distinct compute READY banner, fatal no-fallback behavior, exact CPU
  reference validation, and a final machine-readable summary.
- Preserved the default CPU path and independent Phase 3B upload-only path.

No `vkDeviceWaitIdle()` call exists.

## Validation log

This table is append-only evidence for this working-tree review. A strict
hardware result obtained inside the device-isolating sandbox is explicitly not
acceptance evidence.

| Command | Exit | Evidence |
| --- | ---: | --- |
| `make -C c -B backend_vulkan.o VULKAN=1 EXTRA_WARNINGS=-Werror` | 0 | Production backend compiled with `-DCOLI_VULKAN -Werror`. |
| `make -C c vulkan-compute-unit-test EXTRA_WARNINGS=-Werror` | 0 | Fake format-2 arithmetic, lifecycle, faults, barriers, timeout recovery, and two-thread serialization passed. |
| `make -C c -B colibri VULKAN=1 EXTRA_WARNINGS=-Werror` | 0 | Production Colibri compute integration compiled with strict warnings. |
| `make -C c -B tests/test_vulkan_loader EXTRA_WARNINGS=-Werror && ./c/tests/test_vulkan_loader` | 0 | Existing Phase 3B loader transaction/fault suite passed. |
| `./c/tests/test_vulkan_context` | 0 | Existing context/fault suite passed. |
| `./c/tests/test_vulkan_compute --device-lost` | 0 | Terminal compute device loss retained transient and persistent dependencies. |
| `make -C c vulkan-compute-hardware-test-strict` (sandboxed) | 2 | NOT acceptance: sandbox exposed only llvmpipe and hid `/dev/dri`. |
| `make -C c vulkan-compute-hardware-test-strict` (host device access) | 0 | Exact `0x1002/0x67b1`; scratch type 2, heap 1, flags `0x0006`; rows 1/4/12; dispatches/submitted/completed `3/3/3`, rows 17, pending 0. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (first harness revision) | 2 | Product runs passed; harness expected a narrower REPIN diagnostic and was corrected. Not acceptance evidence. |
| same compute smoke command (corrected harness, host device access) | 0 | CPU/upload-only/compute tokens all `197 197 197 197`; 21/21/21 dispatches, 60 rows, zero fallback/timeouts/errors/device loss/pending; max abs `7.4505806e-08`, max rel `0.00207181503`. |
| `make -C c vulkan-strict-compile-test` | 0 | CPU and Vulkan production builds passed `-Werror`; macro probe proved `COLI_VULKAN`; raw upload symbol absent; `vkDeviceWaitIdle` absent; shader header regenerated byte-identically. |
| `make -C c vulkan-unit-test` | 0 | Ordinary context/fault tests and isolated terminal device-loss test passed. |
| `make -C c vulkan-loader-unit-test` | 0 | Loader transaction/fault tests and isolated terminal device-loss test passed. |
| `make -C c vulkan-compute-unit-test` | 0 | Compute arithmetic, lifecycle, fault, timeout, barrier, ownership, counter, and concurrent-caller tests passed. |
| `make -C c vulkan-sanitize-test` (sandboxed) | 2 | NOT acceptance: LeakSanitizer terminated before tests because it cannot operate under the sandbox ptrace environment. |
| `make -C c vulkan-loader-sanitize-test` (sandboxed) | not accepted | Build began, but the sandboxed run did not produce a completed acceptance result. |
| `make -C c vulkan-compute-sanitize-test` (sandboxed) | 2 | NOT acceptance: LeakSanitizer terminated before tests because it cannot operate under the sandbox ptrace environment. |
| `make -C c vulkan-sanitize-test` (outside ptrace sandbox) | 0 | Ordinary context/fault paths passed ASan, UBSan, and LeakSanitizer with `detect_leaks=1`. |
| `make -C c vulkan-loader-sanitize-test` (outside ptrace sandbox) | 0 | Ordinary loader transaction/fault paths passed ASan, UBSan, and LeakSanitizer with `detect_leaks=1`. |
| `make -C c vulkan-compute-sanitize-test` (outside ptrace sandbox) | 0 | Ordinary compute prepare/dispatch/fault/timeout/concurrency paths passed ASan, UBSan, and LeakSanitizer with `detect_leaks=1`. |
| `make -C c vulkan-device-lost-sanitize-test` | 0 | Isolated context and loader terminal device-loss retention passed ASan/UBSan with leak detection deliberately disabled. |
| `make -C c vulkan-compute-device-lost-sanitize-test` | 0 | Isolated compute terminal device-loss retention passed ASan/UBSan with leak detection deliberately disabled; command/fence/pipeline/scratch/context destruction stayed deferred. |
| `make -C c vulkan-hardware-test-strict` (host device access, final source) | 0 | Existing staged copy and Phase 3B seven-format upload/readback lifecycle passed on exact R9 390. |
| `make -C c vulkan-compute-hardware-test-strict` (host device access, final source) | 0 | Exact R9 390; scratch type 2/heap 1/flags `0x0006`; row cases 1/4/12; dispatch/submission/completion `3/3/3`; 17 rows; zero pending. |
| `make -C c vulkan-loader-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` | 0 | Independent Phase 3B upload-only regression passed; tokens `197 197 197 197`, 16 experts/48 tensors, CPU compute, host copies retained. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (final source) | 0 | CPU/upload-only/compute tokens all `197 197 197 197`; 21 recorded/submitted/completed dispatches, 60 rows, zero fallback/timeouts/errors/device loss/pending. All compute-mode negative configuration cases passed. |
| `make -C c -B tests/test_vulkan_compute EXTRA_WARNINGS=-Werror && ./c/tests/test_vulkan_compute` (after expanded post-submit/non-format coverage) | 0 | Strict fake compute tests passed, including post-submit ERROR retention/recovery and formats 0/1/3-6 plus unaligned format 2 remaining residency-only. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/tmp/phase4a-explicitly-missing-model` | 2 (expected nonzero) | Strict supplied-fixture handling failed closed before inference: `explicit Vulkan smoke fixture is incomplete or missing`. |
| `make -C c -B tests/test_vulkan_compute_sanitize && env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./c/tests/test_vulkan_compute_sanitize` (expanded final suite) | 0 | Expanded ordinary compute suite passed ASan/UBSan/LeakSanitizer with leak detection enabled. |
| `make -C c vulkan-compute-device-lost-sanitize-test` (expanded final source) | 0 | Final isolated terminal compute device-loss retention case passed with leak detection disabled. |
| `make -C c test` (sandboxed) | 2 | NOT acceptance: all C tests completed (the io_uring case skipped for sandbox permissions), but 42 Python tests failed when the sandbox denied local socket creation/bind with `PermissionError: [Errno 1] Operation not permitted`; 222 tests otherwise passed and 32 skipped. The suite is rerun below with normal local-loopback access. |
| `make -C c test` (normal local-loopback access) | 0 | Full C regression passed; Python suite ran 243 tests in 74.251 seconds with 32 expected skips and no failures. |
| `make -C c vulkan-hardware-test-strict` (final acceptance rerun) | 0 | Exact `0x1002/0x67b1`; staged copy plus all seven Phase 3B QT upload/readback cases passed on the exclusive R9 390. |
| `make -C c vulkan-compute-hardware-test-strict` (final acceptance rerun) | 0 | Exact `0x1002/0x67b1`; scratch type 2/heap 1/flags `0x0006`; format-2 rows 1/4/12 passed; dispatches/submitted/completed `3/3/3`, rows 17, pending 0. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (expanded final harness) | 0 | CPU/upload-only/compute tokens `197 197 197 197`; dispatches 21, rows 60, CPU down fallbacks 0, max abs `7.4505806e-08`, max rel `0.00207181503`; the complete negative effective-state matrix returned nonzero as required. |
| `make -C c vulkan-strict-compile-test` (final rerun) | 0 | CPU and Vulkan production paths rebuilt with `-Werror`; actual `COLI_VULKAN` macro probe, raw-symbol check, no-`vkDeviceWaitIdle` check, and deterministic shader byte comparison all passed. |
| `make -C c vulkan-loader-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (final rerun) | 0 | Independent upload-only mode produced `197 197 197 197`; 16 experts/48 tensors resident, all host copies retained, arithmetic remained on CPU. |
| `make -C c vulkan-compute-unit-test` (final rerun) | 0 | Final fake format-2 compute unit/fault/concurrency suite passed after the source audit. |
| `git diff --check` plus review-patch reverse/whitespace validation | 0 | Tracked diff is whitespace-clean; the 12-file review patch reverses cleanly against the working tree and all new source/test/shader files are free of trailing whitespace. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (final budget-pressure harness) | 0 | All positive modes and negative gates passed; a temporary locally cloned 20-sparse-layer fixture made the complete expert tier exceed 1 MiB, and valid `VULKAN_EXPERT_MB=1` failed closed for partial residency before inference. The temporary fixture was deleted by the harness. |

## Real R9 390 evidence

The strict real-hardware run selected:

```text
vendor=0x1002 device=0x67b1 queueFamily=0
memory heaps=3 types=7
heap 2=268435456 bytes
memory types 3 and 4: heap 2, propertyFlags=0x0007
```

Persistent activation and result scratch both selected:

```text
memory type 2
heap 1
propertyFlags=0x0006 (HOST_VISIBLE | HOST_COHERENT)
```

They are neither `DEVICE_LOCAL` nor on heap 2 and therefore do not consume the
256 MiB PCIe BAR.

## Four-token evidence

All modes used the same prompt, oracle, expert pins, `IDOT=0`, `XEXP=0`,
`MTP=0`, `DRAFT=0`, greedy sampling, seed, and one OpenMP thread.

```text
CPU:         197 197 197 197
upload-only: 197 197 197 197
compute:     197 197 197 197
```

Compute telemetry:

```text
dispatches=21 submitted=21 completed=21
rows=60 cpu_gate_up_rows=60 cpu_reference_down_rows=60
cpu_down_fallbacks=0 timeouts=0 errors=0 device_lost=0 pending=0
max_abs=7.4505806e-08 max_rel=0.00207181503
```

The relative maximum occurs near zero; every element passed the approved
combined bound `abs <= 5e-5 + 5e-4 * max(abs(cpu), abs(vulkan))`.

## Negative compute-mode acceptance

The generation harness required nonzero status for compute without residency,
validation without compute, invalid compute/validation booleans, missing or
zero `VULKAN_EXPERT_MB`, `REPIN=1`, `MTP=1`, `DRAFT=1`, `XEXP=1`, `IDOT=1`,
one omitted startup expert, and non-format-2 expert tensors. The separate
explicitly missing model invocation also returned 2 before inference.

The production budget contract deliberately accepts only positive integer
MiB, while the ordinary tiny fixture's complete 48-tensor tier is smaller than
1 MiB. To test a valid reduced cap without changing production parsing, the
harness constructs a temporary model by cloning the final checked-in sparse
layer's safetensors bytes into 20 sparse layers. With all 160 experts pinned,
the complete tier exceeds 1 MiB; `VULKAN_EXPERT_MB=1` returned nonzero for
partial startup residency before inference. The temporary model is removed
when the harness exits. Byte-precision fake loader budget/rollback tests remain
in place as independent coverage.

## Final acceptance

PASS. Every required acceptance target has a valid exit-0 host result, every
required negative case returned nonzero, `git diff --check`
passed, and the shader verification regenerated a byte-identical header. The
only nonzero positive-suite attempts in the evidence table were explicitly
invalid sandbox runs superseded by host-access passes. GLM-5.2 remained
`NOT_RUN`.

The generated review patch contains 12 files, 2,500 insertions, and 76
deletions. It includes every source, shader, generated-header, Makefile, and
test change, and excludes `c/glm_tiny`, ordinary build products, and this
implementation report.

## Final code-review corrections (2026-08-01)

The final review corrections are narrowly limited to the Phase 4A compute
backend and its acceptance tests. Production upload-only behavior and the
shader artifact are unchanged.

- `coli_vulkan_compute_prepare()` now tests the host byte order while holding
  the context mutex and returns `COLI_VULKAN_UNSUPPORTED` before creating any
  compute object on a non-little-endian host. A `COLI_VULKAN_TESTING`-only
  configuration flag deterministically exercises this path and proves that
  format-2 upload-only tensor creation remains available.
- Prepare-state checks now give terminal device loss, shutdown/unusable state,
  and retained pending work precedence over the healthy-idle idempotent fast
  path. Tests cover identical healthy prepare, BUSY while compute is retained,
  recovery to idempotent OK after a successful bounded finish, and permanent
  DEVICE_LOST after terminal loss.
- A compute operation whose original wait timed out now records a device-loss
  or other error first observed by `coli_vulkan_finish_pending()` exactly once.
  TIMEOUT-to-SUCCESS still increments completed operations without incrementing
  completed rows, because the old output remains invalid. Fake tests cover
  TIMEOUT-to-SUCCESS, TIMEOUT-to-ERROR-to-SUCCESS, and process-isolated
  TIMEOUT-to-DEVICE_LOST retention. The existing immediate post-submit-error
  recovery test proves the original error is not counted twice.
- The generation harness removes inherited temperature, sampling, routing,
  budget, pinning, quantization, fused-pair, and absorption overrides; it then
  explicitly sets deterministic Phase 4A values, including `COLI_TEMP=0` and
  `TEMP=0`. It derives the four expected continuation IDs from the checked-in
  oracle and checks CPU, upload-only, and compute modes independently.
- The smoke test retains 4-bit expert tensors (format 2) and uses the fixture's
  oracle-compatible 8-bit dense path. The previous 4-bit dense invocation
  produced `197 197 197 197`, revealing that equality between modes alone was
  not an oracle check; the checked-in reference continuation is
  `207 187 119 103`.

Correction-stage results recorded immediately after execution:

| Command | Exit | Evidence |
| --- | ---: | --- |
| `python3 -m py_compile c/tests/test_vulkan_compute_smoke.py && make -C c -B tests/test_vulkan_compute EXTRA_WARNINGS=-Werror && ./c/tests/test_vulkan_compute` | 0 | Smoke script compiled; strict fake compute unit/fault/concurrency/endian tests passed. |
| `./c/tests/test_vulkan_compute --device-lost` | 0 | Isolated TIMEOUT-to-DEVICE_LOST case retained unproven owners and counted device loss once. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` (first oracle-check revision) | 2 | Not acceptance: correctly exposed the old 4-bit dense output `197 197 197 197` versus oracle `207 187 119 103`; corrected to the oracle-compatible 8-bit dense fixture path. |
| same smoke command (second harness revision) | 2 | Not acceptance: exposed an accidentally omitted `PROF=1` harness setting; restored before acceptance. |
| same smoke command (corrected harness) | 0 | Oracle, CPU, upload-only, and compute IDs were all `207 187 119 103`; dispatches 20, rows 60, CPU down fallbacks 0, max abs `4.47034836e-08`, max rel `0.00147122931`. |
| hostile parent environment with `COLI_TEMP=9`, `TEMP=9`, and all requested routing/output overrides, followed by the same smoke target | 0 | The sanitized child environment produced the same four oracle IDs and the same 20-dispatch/60-row telemetry. |

Final required-suite results (recorded as each command completed):

| Command | Exit | Evidence |
| --- | ---: | --- |
| `make -C c vulkan-strict-compile-test` | 0 | CPU and Vulkan production builds used `-Werror`; the Vulkan compile used `-DCOLI_VULKAN`; macro, raw-symbol, no-`vkDeviceWaitIdle`, and shader byte-comparison checks passed. |
| `make -C c vulkan-unit-test` | 0 | Context/fault suite and isolated terminal device-loss retention passed. |
| `make -C c vulkan-loader-unit-test` | 0 | Loader transaction/fault suite and isolated terminal device-loss retention passed. |
| `make -C c vulkan-compute-unit-test` | 0 | Corrected endian, state-ordering, finish telemetry, ownership, arithmetic, and concurrency suite passed. |
| `make -C c vulkan-sanitize-test` | 0 | Ordinary context/fault paths passed ASan, UBSan, and LeakSanitizer with leak detection enabled. |
| `make -C c vulkan-loader-sanitize-test` | 0 | Ordinary loader transaction/fault paths passed ASan, UBSan, and LeakSanitizer with leak detection enabled. |
| `make -C c vulkan-compute-sanitize-test` | 0 | Corrected ordinary compute paths passed ASan, UBSan, and LeakSanitizer with leak detection enabled. |
| `make -C c vulkan-device-lost-sanitize-test` | 0 | Isolated context/loader terminal-loss retention passed ASan/UBSan with leak detection deliberately disabled. |
| `make -C c vulkan-compute-device-lost-sanitize-test` | 0 | Isolated TIMEOUT-to-DEVICE_LOST compute retention passed ASan/UBSan with leak detection deliberately disabled. |
| `make -C c vulkan-hardware-test-strict` | 0 | Exact R9 390 (`0x1002/0x67b1`) passed staged upload plus all seven Phase 3B tensor lifecycles. |
| `make -C c vulkan-compute-hardware-test-strict` | 0 | Exact R9 390 used scratch type 2, heap 1, flags `0x0006`; 1/4/12-row cases passed with dispatch/submitted/completed `3/3/3`, 17 rows, pending 0. |
| `make -C c vulkan-loader-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` | 0 | Independent Phase 3B upload-only mode passed with 16 experts, 48 tensors, retained host copies, and CPU arithmetic. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` | 0 | Final hermetic oracle check passed: oracle/CPU/upload-only/compute were each `207 187 119 103`; dispatches 20, rows 60, fallbacks 0. |
| `make -C c test` | 0 | Full C suite passed; Python ran 243 tests in 76.033 seconds with 32 expected skips and no failures. |
| `git diff --check` | 0 | The complete tracked working-tree diff is whitespace-clean. |

### Correction acceptance

PASS. All requested code-review corrections are implemented without changing
the production shader, upload-only residency, or non-Vulkan behavior. Every
required command has an exit-0 final result. The ordinary sanitizer runs used
leak detection; only the explicitly process-isolated terminal device-loss
retention targets disabled it. No terminal loss destroyed resources whose
completion was unproven.

The regenerated `docs/phase4a_code_review.patch` contains exactly 12 source,
test, shader, generated-header, and build files and is 3,224 lines long. It
excludes `c/glm_tiny`, binaries, object files, the architecture document, and
this implementation report. `git apply --reverse --check
--whitespace=error-all docs/phase4a_code_review.patch` returned 0 against the
final working tree, and independent `git diff --no-index --check` validation
passed for every new source/test/shader file.

## Final MTP runtime-gate correction (2026-08-01)

The Phase 4A compute gate now requires the explicit string `MTP=0` and rejects
a missing value, `MTP=1`, malformed text such as `MTP=yes`, or any other value
with the existing MTP diagnostic. Checkpoint `has_mtp` metadata is deliberately
not a rejection condition: a checkpoint may contain unused MTP tensors. The
separate effective-draft check remains mandatory, so any nonzero effective
draft still fails with the existing DRAFT diagnostic. Normal non-Vulkan MTP
loading, auto-selection, and execution behavior are unchanged.

The pure `coli_vulkan_phase4a_mtp_gate()` helper makes that policy directly
testable without loading a model. Its deterministic unit cases prove:

- `has_mtp=1`, `MTP=0`, effective `DRAFT=0` passes;
- `has_mtp=0`, `MTP=0`, effective `DRAFT=0` passes;
- missing `MTP`, `MTP=1`, and `MTP=yes` fail the strict MTP setting gate;
- positive and negative nonzero effective draft values fail with `MTP=0`,
  independently of checkpoint `has_mtp`.

The generation harness additionally executes missing-`MTP` and malformed-
`MTP=yes` negative cases and requires the production MTP diagnostic.

| Command | Exit | Evidence |
| --- | ---: | --- |
| `make -C c vulkan-strict-compile-test` | 0 | CPU and Vulkan builds passed `-Werror`; the Vulkan path compiled with `-DCOLI_VULKAN`, and all strict artifact/symbol checks passed. |
| `make -C c vulkan-compute-unit-test` | 0 | The new six-case MTP/checkpoint/draft matrix and the existing compute fault/concurrency suite passed. |
| `make -C c vulkan-compute-sanitize-test` | 0 | The corrected compute suite passed ASan, UBSan, and LeakSanitizer with leak detection enabled. |
| `make -C c vulkan-compute-hardware-test-strict` | 0 | Exact `0x1002/0x67b1`; scratch type 2/heap 1/flags `0x0006`; dispatch/submitted/completed `3/3/3`, rows 17, pending 0. |
| `make -C c vulkan-compute-smoke VULKAN_SMOKE_MODEL=/home/openclaw/projects/vulkan-no-rebar-hawaii-codex/c/glm_tiny` | 0 | Missing, `1`, and malformed MTP plus nonzero-draft cases failed closed; oracle/CPU/upload-only/compute IDs each remained `207 187 119 103`; dispatches 20, rows 60, fallbacks 0. |
| `make -C c test` | 0 | Full C regression passed; Python ran 243 tests in 75.298 seconds with 32 expected skips and no failures. |
| `git diff --check` plus new-header whitespace check | 0 | Tracked changes and the new gate header are whitespace-clean. |

The regenerated review patch now contains 13 source/test/shader/build files and
is 3,313 lines long. It includes the new runtime-gate header, excludes the
implementation report and generated model/build artifacts, and passes
`git apply --reverse --check --whitespace=error-all` against the final tree.
