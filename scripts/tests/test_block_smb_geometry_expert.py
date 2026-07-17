"""Tests for the state-conditional Block SMB geometry expert."""

import json
import unittest
from pathlib import Path

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import (
    BlockSMBGeometryExpert,
    evaluate_block_smb_geometry_expert,
    restore_env_state,
    snapshot_env_state,
)

SCENARIO_DIR = Path("retroagi/stages/block_smb/scenarios")


def load_scenario(name: str) -> dict:
    return json.loads((SCENARIO_DIR / name).read_text(encoding="utf-8"))


def run_expert_episode(scenario: dict, *, max_steps: int = 400) -> dict:
    env = MarioScenarioEnv()
    expert = BlockSMBGeometryExpert()
    try:
        env.reset(scenario=dict(scenario))
        env.max_steps = max_steps
        for step in range(max_steps):
            action = expert.action(env)
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated:
                died = bool(info.get("death"))
                return {"goal_reached": not died, "died": died, "steps": step + 1}
            if truncated:
                break
        return {"goal_reached": False, "died": False, "steps": max_steps}
    finally:
        env.close()


class TestGeometryExpertScenarios(unittest.TestCase):
    def test_solves_every_fixed_scenario_closed_loop(self):
        scenarios = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(SCENARIO_DIR.glob("*.json"))
        }
        summary = evaluate_block_smb_geometry_expert(scenarios)

        failures = {
            name: result
            for name, result in summary["scenarios"].items()
            if not result["goal_reached"]
        }
        self.assertFalse(failures, f"geometry expert failed scenarios: {failures}")
        self.assertEqual(summary["success_rate"], 1.0)

    def test_solves_gap_scenario_from_diverged_states(self):
        # This is the exact failure mode of the time-indexed DAgger teacher: a
        # student that lags the script timeline must still get labels that
        # solve the level from where it actually is.
        scenario = load_scenario("level_2_gap.json")
        for start_x in (20, 50, 70, 82):
            with self.subTest(start_x=start_x):
                diverged = dict(scenario)
                diverged["mario"] = [start_x, scenario["mario"][1]]
                result = run_expert_episode(diverged)
                self.assertTrue(result["goal_reached"], result)

    def test_expert_is_deterministic(self):
        scenario = load_scenario("level_4_platforms.json")
        actions_by_run = []
        for _ in range(2):
            env = MarioScenarioEnv()
            expert = BlockSMBGeometryExpert()
            actions = []
            try:
                env.reset(scenario=dict(scenario))
                for _step in range(60):
                    action = expert.action(env)
                    actions.append(action)
                    _obs, _reward, terminated, truncated, _info = env.step(action)
                    if terminated or truncated:
                        break
            finally:
                env.close()
            actions_by_run.append(actions)
        self.assertEqual(actions_by_run[0], actions_by_run[1])


class TestGeometryExpertStateHygiene(unittest.TestCase):
    def test_planning_does_not_mutate_env_state(self):
        scenario = load_scenario("level_7_moving_bridge.json")
        env = MarioScenarioEnv()
        expert = BlockSMBGeometryExpert()
        try:
            env.reset(scenario=dict(scenario))
            for _ in range(3):
                env.step(1)
            before = snapshot_env_state(env)
            expert.plan(env)
            after = snapshot_env_state(env)
        finally:
            env.close()

        self.assertEqual(before["mario"], after["mario"])
        self.assertEqual(before["camera_x"], after["camera_x"])
        self.assertEqual(before["steps"], after["steps"])
        self.assertEqual(
            [plat["rect"] for plat in before["platforms"]],
            [plat["rect"] for plat in after["platforms"]],
        )
        self.assertEqual(before["enemies"], after["enemies"])
        # render must be restored to the real implementation
        self.assertNotIn("render", env.__dict__)
        frame = env.render()
        self.assertEqual(frame.shape, (env.height, env.width, 3))

    def test_snapshot_restore_round_trip_preserves_trajectory(self):
        scenario = load_scenario("level_3_stairs.json")
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario))
            for _ in range(5):
                env.step(1)
            snapshot = snapshot_env_state(env)

            trajectory_a = []
            for _ in range(20):
                env.step(2)
                trajectory_a.append((env.mario["x"], env.mario["y"], env.mario["vy"]))

            restore_env_state(env, snapshot)
            trajectory_b = []
            for _ in range(20):
                env.step(2)
                trajectory_b.append((env.mario["x"], env.mario["y"], env.mario["vy"]))
        finally:
            env.close()

        self.assertEqual(trajectory_a, trajectory_b)


if __name__ == "__main__":
    unittest.main()
