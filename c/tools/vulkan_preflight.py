import json
import subprocess
import sys
import re

def parse_vulkaninfo(output):
    """
    Parses plain text vulkaninfo output when --json is not used.
    Returns a list of device dictionaries.
    """
    devices = []
    current_device = None
    in_memory = False

    for line in output.splitlines():
        line = line.rstrip()

        # New Device Block
        dev_match = re.match(r'^Device Properties and Extensions:', line)
        if dev_match:
            if current_device is not None:
                devices.append(current_device)
            current_device = {
                "deviceName": "",
                "vendorID": 0,
                "deviceID": 0,
                "driverVersion": 0,
                "memoryHeaps": [],
                "memoryTypes": []
            }
            continue

        if current_device is not None:
            name_m = re.search(r'deviceName\s*=\s*(.+)', line)
            if name_m:
                current_device["deviceName"] = name_m.group(1).strip()

            vid_m = re.search(r'vendorID\s*=\s*0x([0-9a-fA-F]+)', line)
            if vid_m:
                current_device["vendorID"] = int(vid_m.group(1), 16)

            did_m = re.search(r'deviceID\s*=\s*0x([0-9a-fA-F]+)', line)
            if did_m:
                current_device["deviceID"] = int(did_m.group(1), 16)

            drv_m = re.search(r'driverVersion\s*=\s*(.+)', line)
            if drv_m:
                try:
                    current_device["driverVersion"] = int(drv_m.group(1).strip(), 16)
                except ValueError:
                    try:
                        current_device["driverVersion"] = int(drv_m.group(1).strip())
                    except ValueError:
                        current_device["driverVersion"] = drv_m.group(1).strip()

            if re.match(r'^VkPhysicalDeviceMemoryProperties:', line):
                in_memory = True
                continue

            if in_memory:
                heap_m = re.match(r'^\s*memoryHeaps\[(\d+)\]:', line)
                if heap_m:
                    idx = int(heap_m.group(1))
                    while len(current_device["memoryHeaps"]) <= idx:
                        current_device["memoryHeaps"].append({"size": 0, "flags": 0})

                size_m = re.search(r'size\s*=\s*(\d+)', line)
                if size_m and len(current_device["memoryHeaps"]) > 0:
                    current_device["memoryHeaps"][-1]["size"] = int(size_m.group(1))

                type_m = re.match(r'^\s*memoryTypes\[(\d+)\]:', line)
                if type_m:
                    idx = int(type_m.group(1))
                    while len(current_device["memoryTypes"]) <= idx:
                        current_device["memoryTypes"].append({"heapIndex": 0, "propertyFlags": 0})

                hi_m = re.search(r'heapIndex\s*=\s*(\d+)', line)
                if hi_m and len(current_device["memoryTypes"]) > 0:
                    current_device["memoryTypes"][-1]["heapIndex"] = int(hi_m.group(1))

                flags_m = re.search(r'propertyFlags\s*=\s*0x([0-9a-fA-F]+)', line)
                if flags_m and len(current_device["memoryTypes"]) > 0:
                    current_device["memoryTypes"][-1]["propertyFlags"] = int(flags_m.group(1), 16)

    if current_device is not None:
        devices.append(current_device)

    return devices

def detect_vulkan_hawaii():
    try:
        # Run vulkaninfo
        # Use full dump because --summary might not have memory info
        result = subprocess.run(["vulkaninfo"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR_PREFLIGHT")
            return

        output = result.stdout

        if not output.strip() or "ERROR_INITIALIZATION_FAILED" in output:
            print("SKIP_NO_VULKAN_DEVICE")
            return

        devices = parse_vulkaninfo(output)

        if not devices:
            if "Cannot create Vulkan instance" in output or "No devices found" in output:
                 print("SKIP_NO_VULKAN_DEVICE")
            else:
                 print("ERROR_PREFLIGHT")
            return

        hawaii_devices = []
        hawaii_ids = [0x67B0, 0x67B1, 0x67B8, 0x67B9, 0x67A0, 0x67A1, 0x67A2, 0x67A8, 0x67A9]

        for d in devices:
            if d.get("vendorID") == 0x1002 and d.get("deviceID") in hawaii_ids:
                hawaii_devices.append(d)

        if not hawaii_devices:
            print("SKIP_UNSUPPORTED_DEVICE")
            return

        # Select one Hawaii device
        selected = hawaii_devices[0]
        selected["benchmarks"] = {
            "glm_5_2": "NOT_RUN"
        }

        print("READY_FOR_HARDWARE_TEST")
        # Ensure we only output exactly what is needed for the selected device
        print(json.dumps(selected, indent=2))

    except FileNotFoundError:
        print("ERROR_PREFLIGHT")
    except Exception as e:
        print("ERROR_PREFLIGHT")

if __name__ == "__main__":
    detect_vulkan_hawaii()
