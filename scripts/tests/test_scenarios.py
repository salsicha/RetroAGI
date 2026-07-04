"""Tests for the Mario scenario environments."""
import unittest
from pathlib import Path

from retroagi.stages.block_smb import SCENARIOS_DIR, MarioScenarioEnv

class TestMarioScenarios(unittest.TestCase):
    """Test suite for the different Mario level configurations."""

    def setUp(self):
        self.env = MarioScenarioEnv()
        self.scenarios_dir = Path(SCENARIOS_DIR)

    def _test_scenario(
        self,
        filename,
        expected_mario,
        expected_platforms,
        expected_coins,
        expected_goal,
        expected_enemies=0,
        expected_moving_platforms=0,
    ):
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
        self.assertEqual(len(self.env.enemies), expected_enemies)
        self.assertEqual(
            sum(1 for platform in self.env.platforms if platform['moving']),
            expected_moving_platforms,
        )
        
        if expected_goal:
            self.assertIsNotNone(self.env.goal)
        else:
            self.assertIsNone(self.env.goal)
            
        # Perform one step to ensure the physics and rendering logic don't crash
        _, _, _, _, _ = self.env.step(0) # NOOP action

    def test_level_1_flat(self):
        self._test_scenario(
            'level_1_flat.json',
            expected_mario=[20, 200],
            expected_platforms=1,
            expected_coins=1,
            expected_goal=True,
        )

    def test_level_2_gap(self):
        self._test_scenario(
            'level_2_gap.json',
            expected_mario=[20, 200],
            expected_platforms=2,
            expected_coins=1,
            expected_goal=True,
        )

    def test_level_3_stairs(self):
        self._test_scenario(
            'level_3_stairs.json',
            expected_mario=[20, 200],
            expected_platforms=4,
            expected_coins=2,
            expected_goal=True,
        )

    def test_level_4_platforms(self):
        self._test_scenario(
            'level_4_platforms.json',
            expected_mario=[20, 100],
            expected_platforms=4,
            expected_coins=2,
            expected_goal=True,
        )

    def test_level_5_enemy_hop(self):
        self._test_scenario(
            'level_5_enemy_hop.json',
            expected_mario=[20, 200],
            expected_platforms=1,
            expected_coins=1,
            expected_goal=True,
            expected_enemies=1,
        )

    def test_level_6_enemy_patrol(self):
        self._test_scenario(
            'level_6_enemy_patrol.json',
            expected_mario=[20, 200],
            expected_platforms=1,
            expected_coins=2,
            expected_goal=True,
            expected_enemies=2,
        )

    def test_level_7_moving_bridge(self):
        self._test_scenario(
            'level_7_moving_bridge.json',
            expected_mario=[20, 200],
            expected_platforms=3,
            expected_coins=1,
            expected_goal=True,
            expected_moving_platforms=1,
        )

    def test_level_8_enemy_gap(self):
        self._test_scenario(
            'level_8_enemy_gap.json',
            expected_mario=[20, 200],
            expected_platforms=2,
            expected_coins=2,
            expected_goal=True,
            expected_enemies=1,
        )

    def test_level_9_enemy_stomp(self):
        self._test_scenario(
            'level_9_enemy_stomp.json',
            expected_mario=[20, 200],
            expected_platforms=1,
            expected_coins=1,
            expected_goal=True,
            expected_enemies=1,
        )

    def test_level_10_left_retreat(self):
        self._test_scenario(
            'level_10_left_retreat.json',
            expected_mario=[190, 200],
            expected_platforms=1,
            expected_coins=1,
            expected_goal=True,
        )

    def test_level_11_left_jump_recovery(self):
        self._test_scenario(
            'level_11_left_jump_recovery.json',
            expected_mario=[205, 200],
            expected_platforms=2,
            expected_coins=1,
            expected_goal=True,
        )

    def test_level_12_wait_bridge(self):
        self._test_scenario(
            'level_12_wait_bridge.json',
            expected_mario=[20, 200],
            expected_platforms=3,
            expected_coins=1,
            expected_goal=True,
            expected_moving_platforms=1,
        )

    def test_level_13_variable_pits(self):
        self._test_scenario(
            'level_13_variable_pits.json',
            expected_mario=[20, 200],
            expected_platforms=4,
            expected_coins=4,
            expected_goal=True,
        )
        platforms = [
            platform['rect']
            for platform in self.env.platforms
        ]
        gaps = [
            next_platform.left - platform.right
            for platform, next_platform in zip(platforms, platforms[1:])
        ]
        self.assertEqual(len(gaps), 3)
        self.assertGreater(len(set(gaps)), 1)
        self.assertGreaterEqual(min(gaps), 36)

    def test_level_14_under_enemy_platform(self):
        self._test_scenario(
            'level_14_under_enemy_platform.json',
            expected_mario=[20, 200],
            expected_platforms=2,
            expected_coins=3,
            expected_goal=True,
            expected_enemies=6,
        )
        overhead_platform = self.env.platforms[1]['rect']
        enemy_positions = [
            (enemy['x'], enemy['y'])
            for enemy in self.env.enemies
            if not enemy['dead']
        ]

        self.assertLess(overhead_platform.top, self.env.mario['y'])
        self.assertGreaterEqual(len(enemy_positions), 6)
        for _, enemy_y in enemy_positions:
            self.assertLessEqual(enemy_y + 14, overhead_platform.top + 1)

    def test_level_15_wait_long_bridge(self):
        self._test_scenario(
            'level_15_wait_long_bridge.json',
            expected_mario=[20, 200],
            expected_platforms=3,
            expected_coins=2,
            expected_goal=True,
            expected_moving_platforms=1,
        )
        bridge = next(platform for platform in self.env.platforms if platform['moving'])
        start_platform = self.env.platforms[0]['rect']

        self.assertGreater(bridge['move_max'], bridge['move_min'])
        self.assertGreater(bridge['rect'].left - start_platform.right, 60)
        self.assertEqual(bridge['move_min'], 88)

    def test_level_16_wait_enemy_gate(self):
        self._test_scenario(
            'level_16_wait_enemy_gate.json',
            expected_mario=[20, 200],
            expected_platforms=1,
            expected_coins=3,
            expected_goal=True,
            expected_enemies=1,
        )
        enemy = self.env.enemies[0]

        self.assertGreater(enemy['patrol_max'], enemy['patrol_min'])
        self.assertGreater(enemy['speed'], 0.0)
        self.assertGreater(enemy['x'], self.env.mario['x'])

if __name__ == '__main__':
    unittest.main()
