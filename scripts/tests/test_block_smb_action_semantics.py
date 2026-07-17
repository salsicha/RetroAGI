"""Action-semantics parity tests between Block SMB and real SMB.

Block SMB is the transfer source for the real-emulator Full SMB stage, so its
action-to-motion contract matters beyond the toy env itself. These tests pin
two things:

1. Parity behaviors that DO match real SMB (tap-jump fires once, variable jump
   height, air control, direction mapping) so they cannot silently drift.
2. Every KNOWN divergence from real SMB, empirically. If a divergence test
   fails, the physics changed: update KNOWN_REAL_SMB_DIVERGENCES and re-tune
   the scripted curriculum before shipping — the teachers are calibrated
   against these exact behaviors.
"""

import unittest

from retroagi.core import SMBAction
from retroagi.stages.block_smb.env import (
    JUMP_BUFFER_FRAMES,
    KNOWN_REAL_SMB_DIVERGENCES,
    MarioScenarioEnv,
)

FLAT_SCENARIO = {
    "world_width": 256,
    "mario": [40, 200],
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


class TestRealSMBParityBehaviors(unittest.TestCase):
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
            for _ in range(20):
                env.step(int(SMBAction.LEFT))
            self.assertLess(env.mario["x"], x1)
        finally:
            env.close()

    def test_tap_jump_fires_once_and_does_not_rejump(self):
        # Parity with real SMB: press A, release well before landing — one
        # jump only. The release lets the jump buffer decay, so no rebound.
        env = make_env(FLAT_SCENARIO)
        try:
            settle(env)
            env.step(int(SMBAction.JUMP))
            self.assertLess(env.mario["vy"], 0.0)
            airborne_frames = 0
            landings = 0
            for _ in range(80):
                was_on_ground = env.mario["on_ground"]
                env.step(int(SMBAction.NOOP))
                if not env.mario["on_ground"]:
                    airborne_frames += 1
                if not was_on_ground and env.mario["on_ground"]:
                    landings += 1
            self.assertEqual(landings, 1)
            self.assertTrue(env.mario["on_ground"])
            self.assertEqual(env.mario["vy"], 0.0)
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
                for _ in range(60):
                    env.step(int(SMBAction.NOOP))
                    min_y = min(min_y, env.mario["y"])
                    if env.mario["on_ground"]:
                        break
                return start_y - min_y
            finally:
                env.close()

        self.assertLess(apex_height(2), apex_height(14))

    def test_no_new_jump_while_airborne(self):
        env = make_env(FLAT_SCENARIO)
        try:
            settle(env)
            env.step(int(SMBAction.JUMP))
            for _ in range(3):
                env.step(int(SMBAction.NOOP))
            self.assertFalse(env.mario["on_ground"])
            vy_before = env.mario["vy"]
            env.step(int(SMBAction.JUMP))  # fresh mid-air press
            # Gravity keeps integrating; the press must not restart the jump.
            self.assertGreater(env.mario["vy"], vy_before - 1.0)
            self.assertGreater(env.mario["vy"], MarioScenarioEnv().jump_power)
        finally:
            env.close()


class TestKnownDivergencesFromRealSMB(unittest.TestCase):
    def registry(self, name: str) -> dict:
        for entry in KNOWN_REAL_SMB_DIVERGENCES:
            if entry["name"] == name:
                return entry
        self.fail(
            f"divergence {name!r} is exercised by the env but missing from "
            "KNOWN_REAL_SMB_DIVERGENCES — document it or remove the behavior"
        )

    def test_registry_entries_are_complete(self):
        for entry in KNOWN_REAL_SMB_DIVERGENCES:
            for key in ("name", "block_behavior", "real_behavior", "reason"):
                self.assertIn(key, entry)
                self.assertTrue(str(entry[key]).strip())

    def test_held_jump_rebounds_on_landing(self):
        # DIVERGES from real SMB (holding A does not re-jump). If this test
        # fails, the jump buffer semantics changed: re-tune every scripted
        # teacher and Monte Carlo oracle before shipping.
        self.registry("held_jump_rebounds_on_landing")
        env = make_env(AIRBORNE_SCENARIO)
        try:
            rebounded = False
            for _ in range(120):
                was_on_ground = env.mario["on_ground"]
                env.step(int(SMBAction.JUMP))  # hold jump the whole time
                if was_on_ground is False and env.mario["vy"] < 0 and env.mario["y"] < 200:
                    if env.steps > 10:
                        rebounded = True
                        break
            self.assertTrue(
                rebounded,
                "held jump no longer rebounds on landing — update "
                "KNOWN_REAL_SMB_DIVERGENCES and re-tune the curriculum",
            )
        finally:
            env.close()

    def test_coyote_time_allows_late_jump(self):
        self.registry("coyote_time")
        scenario = {
            "world_width": 256,
            "mario": [40, 200],
            "platforms": [[0, 220, 80, 20]],  # ledge ends at x=80
        }
        env = make_env(scenario)
        try:
            settle(env, 20)
            # Walk off the ledge.
            while env.mario["on_ground"]:
                env.step(int(SMBAction.RIGHT))
            # Press jump within the coyote window: it must still fire.
            env.step(int(SMBAction.RIGHT_JUMP))
            self.assertLess(env.mario["vy"], -5.0)
        finally:
            env.close()

    def test_jump_buffer_fires_recent_press_on_landing(self):
        self.registry("jump_buffer")
        env = make_env(AIRBORNE_SCENARIO)
        try:
            # Fall until close to the ground, then tap jump before contact.
            for _ in range(200):
                env.step(int(SMBAction.NOOP))
                if env.mario["y"] >= 190:
                    break
            self.assertFalse(env.mario["on_ground"])
            env.step(int(SMBAction.JUMP))  # buffered press
            launched = False
            for _ in range(JUMP_BUFFER_FRAMES + 2):
                env.step(int(SMBAction.NOOP))
                if env.mario["vy"] < -5.0:
                    launched = True
                    break
            self.assertTrue(launched, "buffered jump press no longer fires on landing")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
