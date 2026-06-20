"""Tests for runtime device selection."""

import unittest
from unittest.mock import patch

from retroagi.core import select_device


class TestDeviceSelection(unittest.TestCase):
    def test_auto_prefers_cuda_over_mps(self):
        with (
            patch("retroagi.core.devices.torch.cuda.is_available", return_value=True),
            patch("retroagi.core.devices.is_mps_available", return_value=True),
        ):
            self.assertEqual(select_device("auto").type, "cuda")

    def test_auto_uses_mps_when_cuda_is_unavailable(self):
        with (
            patch("retroagi.core.devices.torch.cuda.is_available", return_value=False),
            patch("retroagi.core.devices.is_mps_available", return_value=True),
        ):
            self.assertEqual(select_device("auto").type, "mps")

    def test_auto_falls_back_to_cpu(self):
        with (
            patch("retroagi.core.devices.torch.cuda.is_available", return_value=False),
            patch("retroagi.core.devices.is_mps_available", return_value=False),
        ):
            self.assertEqual(select_device("auto").type, "cpu")

    def test_apple_silicon_alias_resolves_to_mps(self):
        with (
            patch("retroagi.core.devices.is_mps_built", return_value=True),
            patch("retroagi.core.devices.is_mps_available", return_value=True),
        ):
            self.assertEqual(select_device("apple-silicon").type, "mps")

    def test_rejects_unavailable_cuda(self):
        with patch("retroagi.core.devices.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_device("cuda")

    def test_rejects_unavailable_mps(self):
        with (
            patch("retroagi.core.devices.is_mps_built", return_value=True),
            patch("retroagi.core.devices.is_mps_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon MPS was requested"):
                select_device("mps")


if __name__ == "__main__":
    unittest.main()
