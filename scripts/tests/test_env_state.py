"""Tests for the state-conditional Block SMB geometry expert."""

import json
import unittest
from pathlib import Path

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state

SCENARIO_DIR = Path("retroagi/stages/block_smb/scenarios")


def load_scenario(name: str) -> dict:
    return json.loads((SCENARIO_DIR / name).read_text(encoding="utf-8"))


class TestGeometryExpertStateHygiene(unittest.TestCase):
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
