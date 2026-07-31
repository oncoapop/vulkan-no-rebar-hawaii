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
    def test_timeout(self, mock_print, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="vulkaninfo", timeout=15.0)
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_called_with("ERROR_PREFLIGHT")

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_environment_headless(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Cannot create Vulkan instance."
        mock_run.return_value = mock_result

        with patch.dict(os.environ, {'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-0', 'OTHER': '1'}):
            vulkan_preflight.detect_vulkan_hawaii()

            # Verify the env passed to subprocess.run does not have DISPLAY or WAYLAND_DISPLAY
            env_used = mock_run.call_args[1]['env']
            self.assertNotIn('DISPLAY', env_used)
            self.assertNotIn('WAYLAND_DISPLAY', env_used)
            self.assertIn('OTHER', env_used)
            mock_print.assert_called_with("SKIP_NO_VULKAN_DEVICE")

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

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_realistic_regression_output(self, mock_print, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131
	driverVersion     = 1234
	vendorID          = 0x10de
	deviceID          = 0x1eb1
	deviceName        = GeForce RTX 2060

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 1
	memoryHeaps[0]:
		size   = 100
		flags: count = 0
memoryTypes: count = 1
	memoryTypes[0]:
		heapIndex     = 0
		propertyFlags = 0x0001

VkPhysicalDeviceLimits:
----------------------
	maxImageDimension1D                      = 16384
	size                                     = 9999999

Device Properties and Extensions:
=================================
GPU1:
VkPhysicalDeviceProperties:
---------------------------
	apiVersion        = 1.2.131
	driverVersion     = 1234
	vendorID          = 0x1002
	deviceID          = 0x67b1
	deviceName        = AMD Radeon R9 390 Series

VkPhysicalDeviceMemoryProperties:
=================================
memoryHeaps: count = 3
	memoryHeaps[0]:
		size   = 8589934592 (0x200000000) (8.00 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
	memoryHeaps[1]:
		size   = 268435456 (0x10000000) (256.00 MiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
	memoryHeaps[2]:
		size   = 8589934592 (0x200000000) (8.00 GiB)
		flags: count = 0
memoryTypes: count = 7
	memoryTypes[0]:
		heapIndex     = 0
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	memoryTypes[1]:
		heapIndex     = 1
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	memoryTypes[2]:
		heapIndex     = 2
		propertyFlags = 0x0006: count = 2
			MEMORY_PROPERTY_HOST_VISIBLE_BIT
			MEMORY_PROPERTY_HOST_COHERENT_BIT
	memoryTypes[3]:
		heapIndex     = 2
		propertyFlags = 0x000e: count = 3
			MEMORY_PROPERTY_HOST_VISIBLE_BIT
			MEMORY_PROPERTY_HOST_COHERENT_BIT
			MEMORY_PROPERTY_HOST_CACHED_BIT
	memoryTypes[4]:
		heapIndex     = 0
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	memoryTypes[5]:
		heapIndex     = 1
		propertyFlags = 0x0001: count = 1
			MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	memoryTypes[6]:
		heapIndex     = 2
		propertyFlags = 0x000e: count = 3
			MEMORY_PROPERTY_HOST_VISIBLE_BIT
			MEMORY_PROPERTY_HOST_COHERENT_BIT
			MEMORY_PROPERTY_HOST_CACHED_BIT

VkPhysicalDeviceFeatures:
-------------------------
	size                                     = 9999999
	heapIndex                                = 99
	propertyFlags                            = 0xffff
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_any_call("READY_FOR_HARDWARE_TEST")
        printed_json = mock_print.call_args[0][0]
        data = json.loads(printed_json)

        self.assertEqual(data["hawaii_device_count"], 1)
        self.assertEqual(data["selected_device"]["deviceName"], "AMD Radeon R9 390 Series")

        # Verify heaps
        heaps = data["selected_device"]["memoryHeaps"]
        self.assertEqual(len(heaps), 3)
        self.assertEqual(heaps[0]["size"], 8589934592)
        self.assertEqual(heaps[0]["flags"], 1) # DEVICE_LOCAL
        self.assertEqual(heaps[1]["size"], 268435456)
        self.assertEqual(heaps[1]["flags"], 1) # DEVICE_LOCAL
        self.assertEqual(heaps[2]["size"], 8589934592)
        self.assertEqual(heaps[2]["flags"], 0) # host memory

        # Verify types
        types = data["selected_device"]["memoryTypes"]
        self.assertEqual(len(types), 7)
        self.assertEqual(types[2]["heapIndex"], 2)
        self.assertEqual(types[2]["propertyFlags"], 0x0006)
        self.assertEqual(types[6]["heapIndex"], 2)
        self.assertEqual(types[6]["propertyFlags"], 0x000e)

if __name__ == '__main__':
    unittest.main()
