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
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_any_call("READY_FOR_HARDWARE_TEST")
        printed_json = mock_print.call_args[0][0]
        self.assertTrue("AMD Radeon R9 390 Series" in printed_json)
        self.assertTrue("NOT_RUN" in printed_json)

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
"""
        mock_run.return_value = mock_result
        vulkan_preflight.detect_vulkan_hawaii()
        mock_print.assert_any_call("READY_FOR_HARDWARE_TEST")
        printed_json = mock_print.call_args[0][0]
        data = json.loads(printed_json)
        # Verify exactly one is selected
        self.assertEqual(data["deviceName"], "AMD Radeon R9 390 Series GPU0")
        self.assertEqual(data["benchmarks"]["glm_5_2"], "NOT_RUN")

if __name__ == '__main__':
    unittest.main()
