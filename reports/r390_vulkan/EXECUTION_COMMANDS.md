# Commands for Damian's Server

**Build:**
```bash
git switch feature/vulkan-no-rebar-hawaii
cd c
make clean
make glm VK=1
make tests/test_vk_staged VK=1
```

**Isolated Smoke Test:**
```bash
cd c
./tests/test_vk_staged
```
*Expected: SUCCESS indicating pure memory picking passes, and actual byte-level staged-copy test passes on the R9 390.*

**VRAM-residency & Correctness Comparison:**
```bash
# CPU Baseline
./colibri --model /path/to/model -t 18 --temp 0 --draft 0 -p "Explain quantum computing."

# Staged Vulkan Offload
COLI_VK_STAGED=1 COLI_VK_EXPERTS=320 ./colibri --model /path/to/model -t 18 --temp 0 --draft 0 -p "Explain quantum computing."
```
*Verify TTFT, token generation rate, VRAM utilization, and ensure the generated token text is exactly identical.*

**Rollback:**
```bash
git checkout dev
cd c
make clean
make glm VK=1
```
