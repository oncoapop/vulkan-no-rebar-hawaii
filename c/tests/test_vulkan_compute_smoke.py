#!/usr/bin/env python3
"""Deterministic four-token Phase 4A CPU/upload/compute acceptance."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile


TOKENS_RE = re.compile(r"^GLM C engine\s+:\s+([0-9 ]+)$", re.MULTILINE)
UPLOAD_READY_RE = re.compile(
    r"^\[VULKAN\] READY upload-only: vendor=0x1002 device=0x67b1 "
    r"experts=16 tensors=48 committed=(\d+) budget=(\d+) pending=0 "
    r"compute=CPU host_copies=retained$", re.MULTILINE)
COMPUTE_READY_RE = re.compile(
    r"^\[VULKAN\] READY compute-down-fmt2: vendor=0x1002 device=0x67b1 "
    r"experts=16 tensors=48 committed=(\d+) budget=(\d+) pending=0 "
    r"max_rows=64 gate_up=CPU down=Vulkan host_copies=retained$", re.MULTILINE)
SUMMARY_RE = re.compile(
    r"^\[VULKAN\] COMPUTE summary: kernel=fmt2-down dispatches=(\d+) "
    r"submitted=(\d+) completed=(\d+) rows=(\d+) cpu_gate_up_rows=(\d+) "
    r"cpu_reference_down_rows=(\d+) cpu_down_fallbacks=(\d+) timeouts=(\d+) "
    r"errors=(\d+) device_lost=(\d+) pending=(\d+) max_abs=([0-9.eE+-]+) "
    r"max_rel=([0-9.eE+-]+)$", re.MULTILINE)
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


def run(binary, model, ref, pin, mode, extra=None, expert_bits=4, timeout=120):
    env = os.environ.copy()
    for key in (
        "DISPLAY", "WAYLAND_DISPLAY", "COLI_VULKAN", "COLI_VULKAN_COMPUTE",
        "COLI_VULKAN_VALIDATE", "VULKAN_EXPERT_MB", "VULKAN_TIMEOUT_MS",
        "COLI_CUDA", "COLI_GPU", "COLI_GPUS", "COLI_METAL", "CUDA_DENSE",
        "CUDA_EXPERT_GB", "MTP", "DRAFT", "IDOT", "XEXP", "REPIN",
        "COLI_TEMP", "TEMP", "NUCLEUS", "TOPK", "TOPP", "CACHE_ROUTE",
        "ROUTE_J", "ROUTE_M", "ROUTE_P", "ROUTE_ALPHA", "EXPERT_BUDGET",
        "EXPERT_BUDGET_EXPERIMENTAL", "SPEC_PIN", "I4S",
        "COLI_NO_FUSED_PAIR", "ABSORB",
    ):
        env.pop(key, None)
    env.update({
        "SNAP": str(model), "REF": str(ref), "PIN": str(pin),
        "PIN_GB": "0.01", "PIN_FILL": "0", "AUTOPIN": "0",
        "REPIN": "0", "MTP": "0", "DRAFT": "0", "IDOT": "0",
        "XEXP": "0", "COLI_TEMP": "0", "TEMP": "0", "SEED": "1",
        "PROF": "1",
        "PIPE": "0", "MLOCK": "0", "OMP_NUM_THREADS": "1",
        "COLI_NO_OMP_TUNE": "1",
    })
    if mode in ("upload", "compute"):
        env.update({
            "COLI_VULKAN": "1", "VULKAN_EXPERT_MB": "64",
            "VULKAN_TIMEOUT_MS": "5000",
        })
    if mode == "compute":
        env.update({
            "COLI_VULKAN_COMPUTE": "1", "COLI_VULKAN_VALIDATE": "1",
        })
    if extra:
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    # Expert bits stay at 4 (QT format 2); dense 8-bit matches the checked-in
    # framework oracle while preserving the routed-down compute milestone.
    command = [str(binary), "64", str(expert_bits), "8"]
    return subprocess.run(command, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        check=False)


def token_ids(result, label):
    match = TOKENS_RE.search(result.stdout)
    if not match:
        fail(f"{label} did not print generated token IDs", result)
    ids = [int(value) for value in match.group(1).split()]
    if len(ids) != 4:
        fail(f"{label} generated {len(ids)} tokens instead of four", result)
    return ids


def require_negative(label, result, diagnostic):
    if result.returncode == 0 or diagnostic not in result.stderr:
        fail(f"negative case {label} did not fail closed with {diagnostic!r}",
             result)


def make_budget_pressure_model(source, destination, last_layer=22):
    """Clone the final sparse layer locally until its expert tier exceeds 1 MiB."""
    shutil.copytree(source, destination)
    model_path = destination / "model.safetensors"
    raw = model_path.read_bytes()
    if len(raw) < 8:
        fail("budget-pressure source safetensors is truncated")
    header_size = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_size
    if data_start > len(raw):
        fail("budget-pressure source safetensors header is truncated")
    header = json.loads(raw[8:data_start].decode("utf-8"))
    data = raw[data_start:]
    prefix = "model.layers.4."
    template = [(name, metadata) for name, metadata in header.items()
                if name.startswith(prefix)]
    if not template:
        fail("budget-pressure source has no sparse layer 4")
    payload = bytearray(data)
    for layer in range(5, last_layer + 1):
        for name, metadata in template:
            offsets = metadata.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                fail(f"invalid safetensors offsets for {name}")
            start, end = offsets
            if not (isinstance(start, int) and isinstance(end, int) and
                    0 <= start <= end <= len(data)):
                fail(f"out-of-range safetensors offsets for {name}")
            clone = dict(metadata)
            clone["data_offsets"] = [len(payload), len(payload) + end - start]
            header[name.replace(prefix, f"model.layers.{layer}.", 1)] = clone
            payload.extend(data[start:end])
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    model_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer_count = last_layer + 1
    config["num_hidden_layers"] = layer_count
    config["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (layer_count - 3)
    for field in ("layer_types", "indexer_types"):
        values = config.get(field)
        if not isinstance(values, list) or not values:
            fail(f"budget-pressure config lacks {field}")
        config[field] = [values[min(index, len(values) - 1)]
                         for index in range(layer_count)]
    config_path.write_text(json.dumps(config), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    supplied = args.model is not None or "VULKAN_SMOKE_MODEL" in os.environ
    model_text = args.model if args.model is not None else os.environ.get(
        "VULKAN_SMOKE_MODEL")
    if not supplied:
        print("SKIP: no Vulkan compute smoke fixture was specified")
        return 0
    if not model_text:
        fail("an explicitly supplied Vulkan smoke fixture path is empty")
    model = Path(model_text).resolve()
    binary = Path(args.binary).resolve()
    source_ref = Path(args.ref).resolve()
    if not (model / "config.json").is_file() or not (
        model / "model.safetensors").is_file():
        fail(f"explicit Vulkan smoke fixture is incomplete or missing: {model}")
    if not binary.is_file() or not source_ref.is_file():
        fail("smoke binary or oracle is missing")

    oracle = json.loads(source_ref.read_text(encoding="utf-8"))
    prompt = oracle.get("prompt_ids")
    full = oracle.get("full_ids")
    if not isinstance(prompt, list) or not isinstance(full, list) or len(
            full) < len(prompt) + 4:
        fail("oracle does not contain a prompt plus four continuation tokens")
    expected = full[len(prompt):len(prompt) + 4]
    if len(expected) != 4 or not all(isinstance(token, int) for token in expected):
        fail("oracle continuation does not contain exactly four integer IDs")

    with tempfile.TemporaryDirectory(prefix="coli-phase4a-") as tmp_text:
        tmp = Path(tmp_text)
        models = {}
        for mode in ("cpu", "upload", "compute"):
            models[mode] = tmp / f"{mode}-model"
            shutil.copytree(model, models[mode])
        ref4 = tmp / "ref4.json"
        ref4.write_text(json.dumps({"prompt_ids": prompt,
            "full_ids": prompt + expected}), encoding="utf-8")
        pin = tmp / "pins.txt"
        pin.write_text("".join(
            f"{layer} {expert} {1000 - rank}\n"
            for rank, (layer, expert) in enumerate(
                (layer, expert) for layer in (3, 4) for expert in range(8))),
            encoding="utf-8")

        results = {mode: run(binary, models[mode], ref4, pin, mode)
                   for mode in ("cpu", "upload", "compute")}
        for mode, result in results.items():
            if result.returncode != 0:
                fail(f"{mode} four-token run failed", result)
        ids = {mode: token_ids(result, mode) for mode, result in results.items()}
        for mode in ("cpu", "upload", "compute"):
            if ids[mode] != expected:
                fail(f"{mode} tokens {ids[mode]} do not match oracle "
                     f"{expected}", results[mode])

        logs = {mode: result.stdout + result.stderr
                for mode, result in results.items()}
        if UPLOAD_READY_RE.search(logs["cpu"]) or COMPUTE_READY_RE.search(
                logs["cpu"]):
            fail("CPU run printed a Vulkan READY banner", results["cpu"])
        upload_ready = UPLOAD_READY_RE.findall(logs["upload"])
        if len(upload_ready) != 1 or COMPUTE_READY_RE.search(logs["upload"]):
            fail("upload-only run did not prove independent Phase 3B mode",
                 results["upload"])
        compute_ready = COMPUTE_READY_RE.findall(logs["compute"])
        if len(compute_ready) != 1 or UPLOAD_READY_RE.search(logs["compute"]):
            fail("compute run did not prove Phase 4A mode", results["compute"])
        if int(upload_ready[0][0]) <= 0 or int(upload_ready[0][1]) != 64 << 20:
            fail("upload-only residency accounting mismatch", results["upload"])
        if int(compute_ready[0][0]) <= 0 or int(compute_ready[0][1]) != 64 << 20:
            fail("compute residency accounting mismatch", results["compute"])
        if not PIN_RE.search(logs["upload"]) or not PIN_RE.search(logs["compute"]):
            fail("host startup tier was not retained", results["compute"])
        if "backend CPU + Vulkan residency (upload-only)" not in logs["upload"]:
            fail("upload-only profile label is missing", results["upload"])
        if "backend CPU gate/up + Vulkan routed down (fmt2)" not in logs["compute"]:
            fail("compute profile label is missing", results["compute"])

        summaries = SUMMARY_RE.findall(logs["compute"])
        if len(summaries) != 1:
            fail("compute run did not print exactly one valid summary",
                 results["compute"])
        values = summaries[0]
        recorded, submitted, completed, rows, gate_rows, ref_rows = map(
            int, values[:6])
        fallback, timeouts, errors, lost, pending = map(int, values[6:11])
        max_abs, max_rel = map(float, values[11:13])
        if not (recorded == submitted == completed > 0 and
                rows == gate_rows == ref_rows > 0 and
                fallback == timeouts == errors == lost == pending == 0 and
                max_abs >= 0 and max_rel >= 0):
            fail("compute dispatch/counter proof is inconsistent",
                 results["compute"])

        require_negative("compute-without-residency", run(binary,
            models["compute"], ref4, pin, "cpu", {
                "COLI_VULKAN_COMPUTE": "1", "COLI_VULKAN_VALIDATE": "1"}),
            "requires COLI_VULKAN=1")
        require_negative("validate-without-compute", run(binary,
            models["compute"], ref4, pin, "upload", {
                "COLI_VULKAN_VALIDATE": "1"}),
            "requires COLI_VULKAN_COMPUTE=1")
        require_negative("invalid-compute-value", run(binary,
            models["compute"], ref4, pin, "upload", {
                "COLI_VULKAN_COMPUTE": "yes"}),
            "must be exactly 0 or 1")
        require_negative("invalid-validate-value", run(binary,
            models["compute"], ref4, pin, "upload", {
                "COLI_VULKAN_VALIDATE": "yes"}),
            "must be exactly 0 or 1")
        require_negative("missing-cap", run(binary, models["compute"], ref4,
            pin, "compute", {"VULKAN_EXPERT_MB": None}),
            "requires an explicit positive integer")
        require_negative("missing-MTP", run(binary, models["compute"], ref4,
            pin, "compute", {"MTP": None}),
            "requires MTP effectively disabled with MTP=0")
        require_negative("malformed-MTP", run(binary, models["compute"], ref4,
            pin, "compute", {"MTP": "yes"}),
            "requires MTP effectively disabled with MTP=0")
        for key, diagnostic in (
            ("REPIN", "requires REPIN=0"),
            ("MTP", "requires MTP effectively disabled"),
            ("DRAFT", "requires effective DRAFT=0"),
            ("XEXP", "requires effective XEXP=0"),
            ("IDOT", "requires effective IDOT=0"),
        ):
            require_negative(f"{key}=1", run(binary, models["compute"], ref4,
                pin, "compute", {key: "1"}), diagnostic)
        incomplete = tmp / "incomplete-pins.txt"
        incomplete.write_text("".join(
            f"{layer} {expert} 1\n" for layer in (3, 4)
            for expert in range(8) if (layer, expert) != (4, 7)),
            encoding="utf-8")
        require_negative("incomplete-tier", run(binary, models["compute"],
            ref4, incomplete, "compute"), "requires every startup expert")
        budget_model = tmp / "budget-pressure-model"
        make_budget_pressure_model(model, budget_model)
        budget_pins = tmp / "budget-pressure-pins.txt"
        budget_pins.write_text("".join(
            f"{layer} {expert} 1\n" for layer in range(3, 23)
            for expert in range(8)), encoding="utf-8")
        require_negative("below-complete-tier-budget", run(binary,
            budget_model, ref4, budget_pins, "compute",
            {"VULKAN_EXPERT_MB": "1"}),
            "compute startup tier has a missing/duplicate expert")
        require_negative("non-format-2", run(binary, models["compute"], ref4,
            pin, "compute", expert_bits=8),
            "compute down tensor format/geometry is unsupported")
        require_negative("nonpositive-cap", run(binary, models["compute"],
            ref4, pin, "compute", {"VULKAN_EXPERT_MB": "0"}),
            "requires an explicit positive integer")

    print("PASS: Phase 4A four-token CPU/upload-only/Vulkan-compute smoke test")
    print("oracle:", " ".join(map(str, expected)))
    print("cpu:", " ".join(map(str, ids["cpu"])))
    print("upload-only:", " ".join(map(str, ids["upload"])))
    print("compute:", " ".join(map(str, ids["compute"])))
    print(f"dispatches={recorded} rows={rows} cpu_fallbacks=0 "
          f"max_abs={max_abs:.9g} max_rel={max_rel:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
