"""Focused physics tests for the Block SMB environment."""

import unittest

from retroagi.stages.block_smb import BlockSMBRewardConfig, MarioScenarioEnv


class TestBlockSMBPhysics(unittest.TestCase):
    def make_env(self, scenario):
        env = MarioScenarioEnv()
        env.reset(scenario=scenario, seed=123)
        return env

    def test_vertical_and_horizontal_platform_collisions_resolve(self):
        env = self.make_env(
            {
                "mario": [27, 204],
                "platforms": [[0, 220, 256, 20], [40, 180, 10, 40]],
            }
        )
        try:
            env.mario["on_ground"] = False
            _, _, terminated, truncated, _ = env.step(1)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(env.mario["x"], 26)
            self.assertEqual(env.mario["vx"], 0)
            self.assertEqual(env.mario["y"] + env.mario["h"], 220)
            self.assertTrue(env.mario["on_ground"])
            self.assertEqual(env.mario["vy"], 0)
        finally:
            env.close()

    def test_gap_fall_eventually_terminates_with_death_penalty(self):
        env = self.make_env(
            {
                "mario": [110, 180],
                "platforms": [[0, 220, 80, 20], [170, 220, 86, 20]],
                "world_width": 256,
            }
        )
        try:
            terminated = truncated = False
            total_reward = 0.0
            for _ in range(80):
                _, reward, terminated, truncated, _ = env.step(0)
                total_reward += reward
                if terminated or truncated:
                    break

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertGreater(env.mario["y"], env.height)
            self.assertLess(total_reward, -1.0)
        finally:
            env.close()

    def test_moving_platform_carries_mario(self):
        env = self.make_env(
            {
                "mario": [50, 184],
                "platforms": [
                    {"x": 40, "y": 200, "w": 80, "h": 10, "moving": [40, 80, 2]}
                ],
            }
        )
        try:
            start_x = env.mario["x"]
            _, _, terminated, truncated, _ = env.step(0)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(env.mario["on_ground"])
            self.assertEqual(env.platforms[0]["delta_x"], 2)
            self.assertEqual(env.mario["x"], start_x + 2)
        finally:
            env.close()

    def test_coin_collection_marks_coin_and_increases_score(self):
        env = self.make_env(
            {
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
                "coins": [[22, 204, 10, 10]],
            }
        )
        try:
            _, reward, terminated, truncated, _ = env.step(0)

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(env.coins[0]["collected"])
            self.assertEqual(env.score, 10)
            self.assertGreaterEqual(reward, 9.0)
        finally:
            env.close()

    def test_enemy_side_collision_terminates_and_stomp_kills_enemy(self):
        side_env = self.make_env(
            {
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
                "enemies": [[22, 206, 22, 22, 0]],
            }
        )
        try:
            _, reward, terminated, truncated, _ = side_env.step(0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertLess(reward, -9.0)
        finally:
            side_env.close()

        stomp_env = self.make_env(
            {
                "mario": [20, 190],
                "platforms": [[0, 220, 256, 20]],
                "enemies": [[22, 206, 22, 22, 0]],
            }
        )
        try:
            stomp_env.mario["vy"] = 8.0
            _, reward, terminated, truncated, _ = stomp_env.step(0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(stomp_env.enemies[0]["dead"])
            self.assertEqual(stomp_env.score, 5)
            self.assertGreater(reward, 4.0)
            self.assertLess(stomp_env.mario["vy"], 0)
        finally:
            stomp_env.close()

    def test_goal_collision_terminates_with_goal_reward(self):
        env = self.make_env(
            {
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
                "goal": [22, 204, 12, 16],
            }
        )
        try:
            _, reward, terminated, truncated, _ = env.step(0)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertGreaterEqual(reward, 49.0)
        finally:
            env.close()

    def test_reset_restores_scenario_state_and_seeded_procedural_generation(self):
        scenario = MarioScenarioEnv.generate_scenario(seed=77)
        env = self.make_env(scenario)
        try:
            env.step(1)
            env.step(1)
            self.assertNotEqual(env.steps, 0)

            _, info = env.reset(scenario=scenario, seed=77)
            self.assertEqual(env.steps, 0)
            self.assertEqual(env.score, 0)
            self.assertEqual(env.camera_x, 0.0)
            self.assertEqual(env.mario["x"], scenario["mario"][0])
            self.assertEqual(env.mario["y"], scenario["mario"][1])
            self.assertEqual(info["max_x_reached"], 0.0)

            regenerated = MarioScenarioEnv.generate_scenario(seed=77)
            self.assertEqual(scenario, regenerated)
        finally:
            env.close()

    def test_reward_terms_sum_to_transition_reward(self):
        env = self.make_env(
            {
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
                "coins": [[22, 204, 10, 10]],
            }
        )
        try:
            _, reward, _, _, info = env.step(0)
            terms = info["reward_terms"]

            self.assertEqual(
                set(terms),
                {
                    "progress",
                    "coin",
                    "enemy_stomp",
                    "goal",
                    "fall_death",
                    "enemy_hit",
                    "step_penalty",
                },
            )
            self.assertEqual(terms["coin"], env.reward_config.coin)
            self.assertEqual(terms["step_penalty"], env.reward_config.step_penalty)
            self.assertAlmostEqual(sum(terms.values()), reward)
            self.assertAlmostEqual(info["reward_total"], reward)
            self.assertEqual(info["reward_config"]["coin"], env.reward_config.coin)
        finally:
            env.close()

    def test_reward_config_tunes_environment_reward_terms(self):
        reward_config = BlockSMBRewardConfig(
            progress_scale=0.0,
            coin=2.0,
            enemy_stomp=1.0,
            goal=5.0,
            fall_death=-3.0,
            enemy_hit=-4.0,
            step_penalty=-0.5,
        )
        env = MarioScenarioEnv(reward_config=reward_config)
        env.reset(
            scenario={
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
                "coins": [[22, 204, 10, 10]],
            },
            seed=123,
        )
        try:
            _, reward, _, _, info = env.step(0)

            self.assertEqual(info["reward_terms"]["progress"], 0.0)
            self.assertEqual(info["reward_terms"]["coin"], 2.0)
            self.assertEqual(info["reward_terms"]["step_penalty"], -0.5)
            self.assertAlmostEqual(reward, 1.5)
        finally:
            env.close()

    def test_timeout_sets_truncation_without_termination(self):
        env = self.make_env(
            {
                "mario": [20, 204],
                "platforms": [[0, 220, 256, 20]],
            }
        )
        try:
            env.max_steps = 1
            _, _, terminated, truncated, _ = env.step(0)

            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(env.steps, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
