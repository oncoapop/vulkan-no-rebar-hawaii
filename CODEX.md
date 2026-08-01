# CODEX.md — Vulkan No-ReBAR Hawaii Development SOP

## 1. Project objective

Develop a safe, opt-in Vulkan backend for Colibrì that can use one AMD Radeon
R9 390 8 GB GPU with a 256 MB PCIe BAR and no ReBAR.

The immediate architecture uses:

- HOST_VISIBLE staging memory;
- strict DEVICE_LOCAL, non-HOST_VISIBLE destination memory;
- `vkCmdCopyBuffer`;
- persistent Vulkan context and resource ownership;
- finite fence timeouts;
- safe retention of resources that may still be in flight.

The active CPU-streaming GLM-5.2 installation is operational and must not be
modified, rebuilt, replaced, benchmarked, or used for experimental tests.

## 2. Fixed hardware and scope

Assume:

- GPU: one AMD Radeon R9 390;
- PCI vendor/device: `0x1002/0x67b1`;
- VRAM: 8 GB;
- PCIe BAR: 256 MB;
- Vulkan driver: Mesa RADV;
- no ReBAR;
- no ROCm/KFD dependency;
- no multi-GPU support in the current phases.

Do not add multi-GPU, automatic device distribution, ReBAR assumptions, or
unbounded VRAM allocation unless explicitly approved.

## 3. Repository safety

Repository:

`/home/openclaw/projects/vulkan-no-rebar-hawaii-codex`

Rules:

1. Never modify `main` directly.
2. Work only on the currently approved phase branch or a dedicated PR branch.
3. Before modifying anything, run:

   ```bash
   git status --short --branch
Stop if the working tree contains unexpected modifications.
Never merge a pull request automatically.
Never force-push except when explicitly instructed.

When force-pushing after an approved rebase, use:

git push --force-with-lease
Do not delete the persistent phase branch after merging a Jules task branch.
Delete only the temporary Jules-generated head branch after an approved
squash merge.
Use an isolated Git worktree for rebases, conflict resolution, or risky PR
review whenever practical.
4. Phase workflow

### Each phase follows this sequence.

# A. Architecture

Use maximum reasoning effort for:

backend architecture;
Vulkan synchronization;
memory ownership;
resource lifetime;
timeout and failure paths;
persistent-context design;
branch divergence or difficult conflict resolution.

During architecture work:

inspect the real codebase;
identify exact files and function names;
trace the complete call path;
do not modify files;
do not commit;
do not push;
do not open or merge a PR;
stop after producing the design.

# B. Architecture handoff

Long architecture responses must be written to a Markdown file rather than
returned only in the terminal.

Default location:

docs/<phase-name>_architecture.md

After writing the file, report:

exact path;
line count;
git status --short.

Do not commit the architecture document unless explicitly instructed.

This requirement prevents terminal or screenshot truncation and allows the
complete design to be reviewed externally.

# C. Implementation

Implementation may begin only after the architecture is approved.

Implementation must:

remain within the approved files and scope;
preserve CPU, CUDA, HIP, and Metal behaviour;
keep Vulkan opt-in;
avoid unrelated refactoring;
include exact failure handling;
include tests for the new behaviour;
stop before merge.

# D. Review

Use routine high reasoning for ordinary implementation review.

Return to maximum reasoning when the changes involve:

queue or command-pool synchronization;
submitted command-buffer lifetime;
fences, timeouts, or device loss;
resource destruction ordering;
persistent buffer ownership;
allocator design;
concurrency;
cross-thread access;
rebasing a divergent branch;
changes that could hang or reset the GPU.

# 5. Jules workflow

Jules is primarily an implementation agent. Do not rely on Jules alone to
design subtle Vulkan synchronization or lifetime APIs.

Preferred workflow:

Codex produces the complete architecture.
Save the architecture to a Markdown file.
Review and approve the design.
Give Jules a complete, fixed implementation specification.
Jules implements and creates a PR.
Retarget the Jules PR to the correct persistent phase branch.
Codex reviews and fixes the PR branch.
Run real hardware tests.
Stop before merge.
The user performs the squash merge.

Avoid repeatedly changing Jules requirements one item at a time. Consolidate
the complete API, ownership model, timeout behaviour, tests, and scope before
implementation begins.

# 6. Pull-request discipline

For a Jules-generated PR:

base: persistent phase branch;
head: Jules-generated task branch.

Before review, verify:

exact base branch;
exact head branch;
changed files;
commit count;
mergeability;
whether the branches diverged.

When resolving conflicts:

rebase the PR head onto the persistent phase branch;
preserve previously validated phase work;
resolve conflicts in an isolated worktree;
push only the PR head branch;
do not alter the base branch;
use --force-with-lease, never unrestricted --force.

After approval:

use squash merge;
delete only the Jules-generated task branch;
retain the persistent phase branch.

# 7. Vulkan safety requirements
Persistent context

Real integration must not create and destroy a Vulkan instance, physical
device, logical device, queue, or command pool for every upload.

Use a persistent context with explicit initialization and shutdown.

Destination memory

Expert destination buffers must be:

compatible with the buffer’s memoryTypeBits;
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
not VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
created with VK_BUFFER_USAGE_TRANSFER_DST_BIT;
at least as large as the requested upload.
Staging memory

A staging-memory type must satisfy both:

the staging buffer’s memoryTypeBits;
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.

Never select a memory type based only on property flags.

Submitted-work lifetime

After vkQueueSubmit succeeds:

do not destroy the fence;
do not free the command buffer;
do not destroy the staging buffer;
do not free staging memory;
do not reset or destroy the command pool;
do not destroy the logical device;

unless the associated fence has returned VK_SUCCESS, or the device has been
handled through an explicitly approved device-loss shutdown path.

VK_TIMEOUT means the work may still be in flight.

Any retained timeout operation must remain reachable and must be finished by a
function that:

waits again using a finite timeout;
leaves all resources intact on another timeout;
cleans up only after VK_SUCCESS;
zeros the operation structure after successful cleanup.

Fence-wait errors must not silently trigger destruction of potentially pending
resources.

External synchronization

Document and enforce Vulkan external-synchronization requirements for:

queues;
command pools;
command-buffer allocation and freeing;
shared context mutation;
shutdown.

Do not assume these objects are safe for concurrent unsynchronized use.

# 8. VRAM policy

VRAM use must be bounded and explicit.

Until another policy is approved:

do not allocate experts without a configured cap;
do not infer that all nominal VRAM is available;
preserve a reserve for driver and working buffers;
reject allocations that exceed the approved budget;
do not silently fall back to unsafe oversubscription.

Arena allocation, suballocation, eviction, cache replacement, and multi-buffer
streaming are separate design decisions and must not be added casually.

# 9. Testing requirements
Compilation

Compile changed C files with strict warnings, including -Werror, where the
project toolchain permits it.

A successful build must have:

exit status 0;
no compiler warnings;
no unexpected generated files committed.
Makefile correctness

Test targets must fail fast.

Do not join tests with shell constructs that allow an earlier failure to be
hidden by a later successful command.

A failing test must cause the Make target to return a non-zero status.

Hardware test

Run the standalone Vulkan hardware tests on the real R9 390 before approval.

Current expected command:

make -C c -B vulkan-test

Required result:

Phase 1 staged upload: PASS;
Phase 2 backend helper: PASS;
overall exit status: 0.

A hardware absence or missing Vulkan development environment may produce an
explicit SKIP, but a missing strict DEVICE_LOCAL memory type must never be
reported as a false PASS.

Failure paths

Review and, where safely feasible, test:

invalid arguments;
zero-sized transfers;
VkDeviceSize to size_t overflow;
memory-type selection failure;
allocation failure;
map failure;
command-buffer allocation and recording failure;
queue submission failure;
fence timeout;
fence-wait error;
repeated finish timeout;
shutdown with retained operations;
Makefile fail-fast behaviour.

Do not deliberately hang or reset the production GPU merely to induce a
timeout unless a safe fault-injection mechanism has been designed and approved.

Existing runtime

Do not run the full 372 GB GLM-5.2 model during unit, helper, or architecture
phases.

A full-model run requires separate explicit approval and a written test plan.

# 10. System-risk precautions

Before running a real Vulkan hardware test:

keep the workload small;
stop other GPU workloads;
use finite timeouts;
save unrelated work;
preferably retain another terminal or SSH session;
avoid running the active GLM-5.2 service simultaneously.

A driver reset is possible even when the test is small. Never describe a GPU
test as risk-free.

# 11. Output and reporting standard

At the end of each task, report:

branch and commit SHA;
files changed;
exact commands run;
exit status of each command;
compiler warnings, if any;
hardware PASS/FAIL/SKIP output;
tests not performed;
assumptions not verified;
remaining untracked files;
whether anything was committed or pushed;
confirmation that the active GLM-5.2 installation was untouched.

Do not report a test as passed when it was only inspected statically.

Distinguish clearly between:

executed and passed;
executed and failed;
skipped;
reviewed by control-flow analysis only.

# 12. Long-output SOP

When a response is likely to exceed approximately 800–1,000 words, or contains
architecture, audit findings, a large diff review, or many exact function
references:

write the full result to a .md file under docs/;
provide only a concise terminal summary;
report the file path and line count;
leave the file uncommitted unless instructed;
do not rely on screenshots as the sole record.

# 13. Current completed milestones

Validated milestones:

Phase 0: Vulkan preflight and Hawaii memory-topology detection.
Phase 1: standalone HOST_VISIBLE to strict DEVICE_LOCAL staged-copy test.
Phase 2: reusable staged-upload helper with finite timeout and retained
operation lifecycle.

Current development branch:

feature/vulkan-expert-upload-integration-phase3

Phase 3 must first determine and implement the smallest safe persistent Vulkan
context and tensor-lifecycle milestone before attempting inference integration.

# 14. Stop conditions

Stop and request review before:

changing the approved architecture;
touching main;
merging a PR;
running GLM-5.2;
introducing multi-GPU support;
adding automatic VRAM policies;
de stroying resources after an unsignalled fence;
destroying a command pool or device with pending work;
modifying the active installation;
force-pushing without --force-with-lease;
expanding the phase beyond its approved milestone.
