"""Tests for the bootstrap_perception script."""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import torch
import numpy as np

# Mock external heavy dependencies that might not be installed
sys.modules['cv2'] = MagicMock()
sys.modules['retro'] = MagicMock()

# Ensure scripts can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scripts import bootstrap_perception

class TestBootstrapPerception(unittest.TestCase):
    """Test suite for the bootstrap_perception script."""

    @patch('scripts.bootstrap_perception.retro.make')
    @patch('scripts.bootstrap_perception.models.segmentation.deeplabv3_resnet50')
    @patch('scripts.bootstrap_perception.torch.save')
    @patch('scripts.bootstrap_perception.os.makedirs')
    @patch('scripts.bootstrap_perception.os.path.exists')
    @patch('scripts.bootstrap_perception.torch.load')
    def test_bootstrap_main_loop(self, mock_torch_load, mock_exists, mock_makedirs, mock_save, mock_deeplab, mock_retro_make):
        """
        Test the main loop of bootstrap_perception with mocked environment and tutor.
        """
        # 1. Mock Environment
        mock_env = MagicMock()
        mock_obs = np.zeros((240, 256, 3), dtype=np.uint8)
        mock_env.reset.return_value = mock_obs
        mock_env.step.return_value = (mock_obs, 0, False, {})
        mock_env.action_space.sample.return_value = 0
        mock_retro_make.return_value = mock_env

        # 2. Mock Tutor Model
        mock_tutor = MagicMock()
        mock_tutor.to.return_value = mock_tutor
        # Mock output of tutor segmentation mask
        mock_out = torch.randn(6, 240, 256)
        mock_tutor.return_value = {'out': [mock_out]}
        mock_deeplab.return_value = mock_tutor

        # 3. Mock file paths to pretend the tutor model exists
        mock_exists.return_value = True

        # Mock cv2.resize since it's used to resize the mask
        sys.modules['cv2'].resize = lambda src, dsize, **kwargs: np.zeros((dsize[1], dsize[0], src.shape[2]) if len(src.shape) == 3 else (dsize[1], dsize[0]), dtype=src.dtype)

        # 4. Patch sys.argv to run only 2 steps
        test_args = ['bootstrap_perception.py', '--steps', '2']
        with patch.object(sys, 'argv', test_args):
            # Run the script
            bootstrap_perception.main()
            success = True

        self.assertTrue(success, "bootstrap_perception.main() failed with an exception.")
        
        # Verify that torch.save was called (saving the weights at the end)
        self.assertTrue(mock_save.called)

if __name__ == '__main__':
    unittest.main()
