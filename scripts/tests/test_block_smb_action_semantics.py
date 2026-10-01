"""Action semantics of Block SMB, which moves Mario with the NES's motion rules.

Block SMB is the transfer source for the real-emulator Full SMB stage, so its
action-to-motion contract must match the NES: tap-jump fires once, variable
jump height, air control, direction mapping, and none of the forgiveness
mechanics (coyote time, jump buffering, rebound on a held jump).
"""

import unittest

from retroagi.core import SMBAction
from retroagi.stages.block_smb.env import MarioScenarioEnv

FLAT_SCENARIO = {
    "world_width": 256,
    "mario": [40, 204],
    "platforms": [[0, 220, 256, 20]],
}

AIRBORNE_SCENARIO = {
    "world_width": 256,
    "mario": [40, 60],
    "platforms": [[0, 220, 256, 20]],
}


def make_env(scenario: dict) -> MarioScenarioEnv:
    env = MarioScenarioEnv()
    env.reset(scenario=dict(scenario))
    return env


def settle(env: MarioScenarioEnv, frames: int = 30) -> None:
    for _ in range(frames):
        env.step(0)


class TestNESActionSemantics(unittest.TestCase):
    def test_action_ids_match_shared_smb_action_vocabulary(self):
        self.assertEqual(int(SMBAction.NOOP), 0)
        self.assertEqual(int(SMBAction.RIGHT), 1)
        self.assertEqual(int(SMBAction.RIGHT_JUMP), 2)
        self.assertEqual(int(SMBAction.LEFT), 3)
        self.assertEqual(int(SMBAction.LEFT_JUMP), 4)
        self.assertEqual(int(SMBAction.JUMP), 5)

    def test_direction_actions_move_the_matching_direction(self):
        env = make_env(FLAT_SCENARIO)
        try:
            settle(env)
            x0 = env.mario["x"]
            for _ in range(10):
                env.step(int(SMBAction.RIGHT))
            self.assertGreater(env.mario["x"], x0)
            x1 = env.mario["x"]
            for _ in range(30):
                env.step(int(SMBAction.LEFT))
            self.assertLess(env.mario["x"], x1)
        finally:
            env.close()

    def test_tap_jump_fires_once(self):
        env = make_env(FLAT_SCENARIO)
        try:
            settle(env)
            env.step(int(SMBAction.JUMP))
            self.assertFalse(env.mario["on_ground"])
            landings = 0
            for _ in range(80):
                was_on_ground = env.mario["on_ground"]
                env.step(int(SMBAction.NOOP))
                if not was_on_ground and env.mario["on_ground"]:
                    landings += 1
            self.assertEqual(landings, 1)
            self.assertTrue(env.mario["on_ground"])
        finally:
            env.close()

    def test_early_release_cuts_jump_height(self):
        def apex_height(hold_frames: int) -> float:
            env = make_env(FLAT_SCENARIO)
            try:
                settle(env)
                start_y = env.mario["y"]
                for _ in range(hold_frames):
                    env.step(int(SMBAction.JUMP))
                min_y = env.mario["y"]
                for _ in range(80):
                    env.step(int(SMBAction.NOOP))
                    min_y = min(min_y, env.mario["y"])
                    if env.mario["on_ground"]:
                        break
                return start_y - min_y
            finally:
                env.close()

        self.assertLess(apex_height(2), apex_height(20))

    def test_held_jump_does_not_jump_again_on_landing(self):
        env = make_env(AIRBORNE_SCENARIO)
        try:
            landed = False
            for _ in range(160):
                env.step(int(SMBAction.JUMP))  # hold jump the whole time
                landed = landed or env.mario["on_ground"]
                if landed:
                    self.assertTrue(env.mario["on_ground"])
            self.assertTrue(landed)
        finally:
            env.close()

    def test_no_jump_after_walking_off_a_ledge(self):
        env = make_env({**FLAT_SCENARIO, "platforms": [[0, 220, 80, 20]]})
        try:
            settle(env, 20)
            while env.mario["on_ground"]:
                env.step(int(SMBAction.RIGHT))
            y = env.mario["y"]
            env.step(int(SMBAction.RIGHT_JUMP))
            self.assertGreaterEqual(env.mario["y"], y)
        finally:
            env.close()

    def test_press_before_landing_is_not_buffered(self):
        env = make_env(AIRBORNE_SCENARIO)
        try:
            while env.mario["y"] < 190:
                env.step(int(SMBAction.NOOP))
            self.assertFalse(env.mario["on_ground"])
            env.step(int(SMBAction.JUMP))
            for _ in range(10):
                env.step(int(SMBAction.NOOP))
            self.assertTrue(env.mario["on_ground"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
