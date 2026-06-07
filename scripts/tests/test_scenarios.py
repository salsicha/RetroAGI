"""Tests for the Mario scenario environments."""
import unittest
from pathlib import Path

from retroagi.stages.block_smb import SCENARIOS_DIR, MarioScenarioEnv

class TestMarioScenarios(unittest.TestCase):
    """Test suite for the different Mario level configurations."""

    def setUp(self):
        self.env = MarioScenarioEnv()
        self.scenarios_dir = Path(SCENARIOS_DIR)

    def _test_scenario(self, filename, expected_mario, expected_platforms, expected_coins, expected_goal):
        """Helper method to load a scenario and validate its instantiation."""
        filepath = self.scenarios_dir / filename
        self.assertTrue(filepath.exists(), f"Scenario file not found: {filepath}")
        
        scenario_config = MarioScenarioEnv.load_scenario_from_json(filepath)
        obs, _ = self.env.reset(scenario=scenario_config)
        
        # Basic validation of the observation shape (H, W, 3)
        self.assertEqual(obs.shape, (self.env.height, self.env.width, 3))
        
        # Validate state variables
        self.assertEqual(self.env.mario['x'], expected_mario[0])
        self.assertEqual(self.env.mario['y'], expected_mario[1])
        self.assertEqual(len(self.env.platforms), expected_platforms)
        self.assertEqual(len(self.env.coins), expected_coins)
        
        if expected_goal:
            self.assertIsNotNone(self.env.goal)
        else:
            self.assertIsNone(self.env.goal)
            
        # Perform one step to ensure the physics and rendering logic don't crash
        _, _, _, _, _ = self.env.step(0) # NOOP action

    def test_level_1_flat(self):
        self._test_scenario('level_1_flat.json', expected_mario=[20, 200], expected_platforms=1, expected_coins=1, expected_goal=True)

    def test_level_2_gap(self):
        self._test_scenario('level_2_gap.json', expected_mario=[20, 200], expected_platforms=2, expected_coins=1, expected_goal=True)

    def test_level_3_stairs(self):
        self._test_scenario('level_3_stairs.json', expected_mario=[20, 200], expected_platforms=4, expected_coins=2, expected_goal=True)

    def test_level_4_platforms(self):
        self._test_scenario('level_4_platforms.json', expected_mario=[20, 100], expected_platforms=4, expected_coins=2, expected_goal=True)

if __name__ == '__main__':
    unittest.main()
