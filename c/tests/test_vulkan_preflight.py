import unittest
import json
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock

# Ensure we can import the tool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tools')))
import vulkan_preflight

class TestVulkanPreflight(unittest.TestCase):
    @patch('subprocess.run')
    @patch('builtins.print')
    def test_vulkaninfo_missing(self, mock_print, mock_run):
        mock_run.side_effect = FileNotFoundError
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("ERROR_PREFLIGHT")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_command_execution_failure(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("ERROR_PREFLIGHT")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_malformed_output(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is some random garbage output that makes no sense"
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("ERROR_PREFLIGHT")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_no_physical_devices(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Cannot create Vulkan instance."
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("SKIP_NO_VULKAN_DEVICE")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_unsupported_gpu(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131 (4202627)
	driverVersion     = 436.2.0 (1828716544)
	vendorID          = 0x10de
	deviceID          = 0x1eb1
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = GeForce RTX 2060
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("SKIP_UNSUPPORTED_DEVICE")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_one_hawaii_gpu(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131 (4202627)
	driverVersion     = 436.2.0 (1828716544)
	vendorID          = 0x1002
	deviceID          = 0x67b1
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon R9 390 Series

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 8589934592 (0x200000000) (8.00 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 1
	memoryTypes[0]:
		heapIndex     = 0
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_any_call("READY_FOR_HARDWARE_TEST")
        printed_json = mock_print.call_args[0][0]
        data = json.loads(printed_json)
        self.assertEqual(data["selected_device"]["deviceName"], "AMD Radeon R9 390 Series")
        self.assertEqual(data["benchmarks"]["glm_5_2"], "NOT_RUN")
        self.assertEqual(data["hawaii_device_count"], 1)
        self.assertEqual(len(data["hawaii_devices"]), 1)

        # Verify memory properties
        heap = data["selected_device"]["memoryHeaps"][0]
        self.assertEqual(heap["size"], 8589934592)
        self.assertEqual(heap["flags"], 1) # MEMORY_HEAP_DEVICE_LOCAL_BIT

        mem_type = data["selected_device"]["memoryTypes"][0]
        self.assertEqual(mem_type["heapIndex"], 0)
        self.assertEqual(mem_type["propertyFlags"], 1)

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_two_hawaii_gpus(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131 (4202627)
	driverVersion     = 436.2.0 (1828716544)
	vendorID          = 0x1002
	deviceID          = 0x67b1
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon R9 390 Series GPU0

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 8589934592 (0x200000000) (8.00 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 1
	memoryTypes[0]:
		heapIndex     = 0
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT

Device Properties and Extensions:
=================================
GPU1:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131 (4202627)
	driverVersion     = 436.2.0 (1828716544)
	vendorID          = 0x1002
	deviceID          = 0x67b1
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon R9 390 Series GPU1

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 4294967296 (0x100000000) (4.00 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 1
	memoryTypes[0]:
		heapIndex     = 0
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_any_call("READY_FOR_HARDWARE_TEST")
        printed_json = mock_print.call_args[0][0]
        data = json.loads(printed_json)

        # Assertions
        self.assertEqual(data["hawaii_device_count"], 2)
        self.assertEqual(len(data["hawaii_devices"]), 2)

        device_names = [d["deviceName"] for d in data["hawaii_devices"]]
        self.assertTrue("AMD Radeon R9 390 Series GPU0" in device_names)
        self.assertTrue("AMD Radeon R9 390 Series GPU1" in device_names)

        # Verify exactly one selected_device
        self.assertTrue(isinstance(data["selected_device"], dict))
        self.assertEqual(data["selected_device"]["deviceName"], "AMD Radeon R9 390 Series GPU0")

        # Determinism and correct memory for selected
        self.assertEqual(data["selected_device"]["memoryHeaps"][0]["size"], 8589934592)

        # Verify benchmark fields
        self.assertEqual(data["benchmarks"]["glm_5_2"], "NOT_RUN")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_hawaii_missing_memory(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131 (4202627)
	driverVersion     = 436.2.0 (1828716544)
	vendorID          = 0x1002
	deviceID          = 0x67b1
	deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName        = AMD Radeon R9 390 Series
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("ERROR_PREFLIGHT")

if __name__ == '__main__':
    unittest.main()
