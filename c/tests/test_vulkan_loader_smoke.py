#!/usr/bin/env python3
"""Deterministic four-token Phase 3B Vulkan-residency smoke test."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


READY_RE = re.compile(
    r"^\[VULKAN\] READY upload-only: vendor=0x1002 device=0x67b1 "
    r"experts=(\d+) tensors=(\d+) committed=(\d+) budget=(\d+) pending=0 "
    r"compute=CPU host_copies=retained$",
    re.MULTILINE,
)
TOKENS_RE = re.compile(r"^GLM C engine\s+:\s+([0-9 ]+)$", re.MULTILINE)
CPU_PROFILE_RE = re.compile(
    r"P0-EXEC: routed CPU [0-9.]+s / [0-9.]+ GB/s \((\d+) row\) \| "
    r"routed GPU critical ([0-9.]+)s"
)
PIN_RE = re.compile(r"^\[PIN\] placement: 0 VRAM \+ 16 RAM expert ", re.MULTILINE)


def fail(message, result=None):
    print(f"FAIL: {message}", file=sys.stderr)
    if result is not None:
        print(f"command: {' '.join(result.args)}", file=sys.stderr)
        print(f"exit: {result.returncode}", file=sys.stderr)
        print("--- stdout ---", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("--- stderr ---", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    raise SystemExit(1)


def run(binary, model, ref, pin, vulkan, extra=None, timeout=120):
    env = os.environ.copy()
    for key in (
        "DISPLAY", "WAYLAND_DISPLAY", "COLI_VULKAN", "VULKAN_EXPERT_MB",
        "VULKAN_TIMEOUT_MS", "COLI_CUDA", "COLI_GPU", "COLI_GPUS",
        "COLI_METAL", "CUDA_DENSE", "CUDA_EXPERT_GB",
    ):
        env.pop(key, None)
    env.update({
        "SNAP": str(model),
        "REF": str(ref),
        "PIN": str(pin),
        "PIN_GB": "0.01",
        "PIN_FILL": "0",
        "AUTOPIN": "0",
        "REPIN": "0",
        "MTP": "0",
        "DRAFT": "0",
        "TEMP": "0",
        "SEED": "1",
        "PROF": "1",
        "PIPE": "0",
        "MLOCK": "0",
        "OMP_NUM_THREADS": "1",
        "COLI_NO_OMP_TUNE": "1",
    })
    if vulkan:
        env.update({
            "COLI_VULKAN": "1",
            "VULKAN_EXPERT_MB": "64",
            "VULKAN_TIMEOUT_MS": "5000",
        })
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    command = [str(binary), "64", "4", "4"]
    return subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False,
    )


def tokens(result, label):
    match = TOKENS_RE.search(result.stdout)
    if not match:
        fail(f"{label} did not print generated token IDs", result)
    ids = [int(value) for value in match.group(1).split()]
    if len(ids) != 4:
        fail(f"{label} generated {len(ids)} tokens instead of four", result)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    model_text = args.model
    model_supplied = args.model is not None
    if not model_supplied and "VULKAN_SMOKE_MODEL" in os.environ:
        model_text = os.environ["VULKAN_SMOKE_MODEL"]
        model_supplied = True
    if not model_supplied:
        print("SKIP: no Vulkan smoke fixture was specified")
        return 0
    if not model_text:
        fail("an explicitly supplied Vulkan smoke fixture path is empty")
    model = Path(model_text).resolve()
    binary = Path(args.binary).resolve()
    source_ref = Path(args.ref).resolve()
    if not (model / "config.json").is_file() or not (
        model / "model.safetensors"
    ).is_file():
        fail(f"explicit Vulkan smoke fixture is incomplete or missing: {model}")
    if not binary.is_file() or not source_ref.is_file():
        fail("smoke binary or oracle is missing")

    with source_ref.open("r", encoding="utf-8") as handle:
        oracle = json.load(handle)
    prompt = oracle.get("prompt_ids")
    full = oracle.get("full_ids")
    if not isinstance(prompt, list) or not isinstance(full, list) or len(full) < len(prompt) + 4:
        fail("oracle does not contain a prompt plus four continuation tokens")

    with tempfile.TemporaryDirectory(prefix="coli-phase3b-") as tmp_text:
        tmp = Path(tmp_text)
        off_model = tmp / "off-model"
        on_model = tmp / "on-model"
        shutil.copytree(model, off_model)
        shutil.copytree(model, on_model)
        ref4 = tmp / "ref4.json"
        ref4.write_text(json.dumps({
            "prompt_ids": prompt,
            "full_ids": full[: len(prompt) + 4],
        }), encoding="utf-8")
        pin = tmp / "pins.txt"
        pin.write_text("".join(
            f"{layer} {expert} {1000 - rank}\n"
            for rank, (layer, expert) in enumerate(
                (layer, expert) for layer in (3, 4) for expert in range(8)
            )
        ), encoding="utf-8")

        off = run(binary, off_model, ref4, pin, False)
        if off.returncode != 0:
            fail("Vulkan-off CPU-path run failed", off)
        on = run(binary, on_model, ref4, pin, True)
        if on.returncode != 0:
            fail("Vulkan-on upload-only run failed", on)

        off_ids = tokens(off, "Vulkan-off")
        on_ids = tokens(on, "Vulkan-on")
        if on_ids != off_ids:
            fail(f"Vulkan-on tokens {on_ids} differ from Vulkan-off {off_ids}", on)
        off_log = off.stdout + off.stderr
        on_log = on.stdout + on.stderr
        if READY_RE.search(off_log):
            fail("Vulkan-off run printed a Vulkan READY banner", off)
        ready = READY_RE.findall(on_log)
        if len(ready) != 1:
            fail(f"Vulkan-on run printed {len(ready)} valid READY banners", on)
        experts, tensors, committed, budget = map(int, ready[0])
        if experts != 16 or tensors != 48 or committed <= 0 or budget != 64 * 1024 * 1024:
            fail("Vulkan residency accounting is not 16 experts/48 tensors/64 MiB", on)
        if not PIN_RE.search(on_log):
            fail("Vulkan-on run did not retain all 16 host-pinned experts", on)
        profile = CPU_PROFILE_RE.search(on_log)
        if not profile or int(profile.group(1)) <= 0 or float(profile.group(2)) != 0.0:
            fail("profiling did not prove positive routed CPU rows and zero GPU compute", on)
        if "backend CPU + Vulkan residency (upload-only)" not in on_log:
            fail("profiling did not label the upload-only CPU compute path", on)

        no_cap = run(binary, on_model, ref4, pin, True,
            {"VULKAN_EXPERT_MB": None})
        if no_cap.returncode == 0 or "requires an explicit positive integer" not in no_cap.stderr:
            fail("missing VULKAN_EXPERT_MB did not fail closed", no_cap)
        repin = run(binary, on_model, ref4, pin, True, {"REPIN": "1"})
        if repin.returncode == 0 or "requires REPIN=0" not in repin.stderr:
            fail("REPIN=1 did not fail closed", repin)
        empty_pin = tmp / "no-valid-experts.txt"
        empty_pin.write_text("0 0 1\n", encoding="utf-8")
        no_residency = run(binary, on_model, ref4, empty_pin, True)
        if no_residency.returncode == 0 or "produced no valid expert tier" not in no_residency.stderr:
            fail("zero-residency Vulkan request did not fail closed", no_residency)

    print("PASS: Phase 3B four-token Vulkan-on/off CPU-path smoke test")
    print("tokens:", " ".join(map(str, on_ids)))
    print("residency: 16 experts, 48 tensors, host copies retained, compute CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
