"""Contract tests pinning vision labels against their source of truth.

The Block SMB simulator's drawing must type exactly the pixels it draws
(MarioScenarioEnv.render_labels against MarioScenarioEnv.render). The Full SMB
labels read from game memory are pinned in test_full_smb_pixel_labels.py.
"""

import unittest

import numpy as np

from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.stages.block_smb import env as block_env
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario


def _block_scenario(**extra):
    scenario = {
        "world_width": 512,
        "mario": [20, 208],
        "platforms": [
            [0, 220, 512, 20],
            [100, 150, 64, 10],
            [200, 180, 30, 40],
            {"x": 240, "y": 150, "w": 40, "h": 10, "moving": [240, 320, 1.0]},
            [340, 196, 32, 24],
        ],
        "coins": [[120, 120, 10, 10]],
        "enemies": [
            [150, 206, 140, 190, 0.8],
            {"kind": "piranha_plant", "x": 209, "pipe_top": 180, "phase": 30},
        ],
        "goal": [480, 200, 16, 20],
    }
    scenario.update(extra)
    return scenario


def _frames(scenario, actions=(1, 1, 2, 2, 1, 4, 0), steps=60):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        yield env, env.render(), env.render_labels()
        for step in range(steps):
            _obs, _reward, terminated, truncated, _info = env.step(actions[step % len(actions)])
            yield env, env.render(), env.render_labels()
            if terminated or truncated:
                return
    finally:
        env.close()


class TestBlockSMBDrawingContract(unittest.TestCase):
    """render_labels() types exactly the pixels render() draws."""

    def test_labels_are_the_drawn_shapes_types(self):
        sky = np.array(block_env.SKY)
        mario_colours = {block_env.MARIO, block_env.MARIO_SKIDDING, block_env.EYE}
        types_seen = set()
        for env, frame, labels in _frames(_block_scenario(platform_kinds=None)):
            self.assertEqual(labels.shape, (240, 256))
            self.assertEqual(labels.dtype, np.uint8)
            types_seen.update(np.unique(labels).tolist())
            is_sky = (frame == sky).all(-1)
            np.testing.assert_array_equal(is_sky, labels == TYPE_ID["background"])
            mario = {tuple(c) for c in frame[labels == TYPE_ID["mario"]].tolist()}
            self.assertLessEqual(mario, mario_colours)
            # Mario's eye is drawn inside his 10-pixel body.
            m = env.mario
            left, top = int(m["x"]) - int(env.camera_x), int(m["y"])
            rows, columns = np.nonzero(labels == TYPE_ID["mario"])
            if rows.size:
                self.assertGreaterEqual(columns.min(), max(left, 0))
                self.assertLess(columns.max(), left + m["w"])
                self.assertLess(rows.max(), top + m["h"])
        for name in ("background", "mario", "ground", "brick", "coin", "enemy"):
            self.assertIn(TYPE_ID[name], types_seen, name)
        self.assertIn(TYPE_ID["moving_platform"], types_seen)

    def test_eyes_belong_to_their_owner(self):
        white = np.array(block_env.EYE)
        for env, frame, labels in _frames(_block_scenario()):
            eyes = labels[(frame == white).all(-1)]
            self.assertTrue(np.isin(eyes, (TYPE_ID["mario"], TYPE_ID["enemy"])).all())
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=_block_scenario())
            frame, labels = env.render(), env.render_labels()
            goomba = env.enemy_screen_rects()[0]
            window = (slice(goomba.top, goomba.bottom), slice(goomba.left, goomba.right))
            eye = (frame[window] == white).all(-1)
            self.assertTrue(eye.any())
            self.assertTrue((labels[window][eye] == TYPE_ID["enemy"]).all())
        finally:
            env.close()

    def test_the_finish_marker_is_never_drawn(self):
        with_goal = list(_frames(_block_scenario(), steps=10))
        without = _block_scenario()
        del without["goal"]
        without_goal = list(_frames(without, steps=10))
        for (_, frame_a, labels_a), (_, frame_b, labels_b) in zip(with_goal, without_goal):
            np.testing.assert_array_equal(frame_a, frame_b)
            np.testing.assert_array_equal(labels_a, labels_b)
        self.assertFalse(hasattr(MarioScenarioEnv(), "render_goal"))

    def test_untagged_platforms_follow_the_default_rule(self):
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=_block_scenario())
            # floor; floating row; raised block on the floor; lift; step on the floor
            self.assertEqual(
                env.platform_kinds,
                ["ground", "brick_row", "ground", "moving_platform", "ground"],
            )
            labels = env.render_labels()
            row = labels[150:160, 100:164]
            self.assertTrue(np.isin(row, (TYPE_ID["brick"], TYPE_ID["question_block"])).all())
            # Question cells sit at fixed world positions: the cell at x=112.
            self.assertTrue((labels[150:160, 116:128] == TYPE_ID["question_block"]).all())
            self.assertTrue((labels[150:160, 100:112] == TYPE_ID["brick"]).all())
        finally:
            env.close()

    def test_tagged_platform_kinds_are_drawn_and_never_change_play(self):
        tagged = _block_scenario(platform_kinds=["ground", "question_block", "pipe", None, "brick"])
        positions = {}
        for name, scenario in (("plain", _block_scenario()), ("tagged", tagged)):
            positions[name] = [
                (env.mario["x"], env.mario["y"], tuple(e["x"] for e in env.enemies))
                for env, _frame, _labels in _frames(scenario)
            ]
        self.assertEqual(positions["plain"], positions["tagged"])
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=tagged)
            labels = env.render_labels()
            self.assertTrue((labels[150:160, 100:164] == TYPE_ID["question_block"]).all())
            self.assertTrue((labels[184:220, 200:209] == TYPE_ID["pipe"]).all())
            self.assertTrue((labels[196:220, 340:372] == TYPE_ID["brick"]).all())
            with self.assertRaisesRegex(ValueError, "one kind per platform"):
                env.reset(scenario=_block_scenario(platform_kinds=["pipe"]))
            with self.assertRaisesRegex(ValueError, "platform kind"):
                env.reset(scenario=_block_scenario(platform_kinds=["lava"] * 5))
        finally:
            env.close()

    def test_pipe_families_tag_their_pipes(self):
        expected = {
            "tall_pipe_jump": ["ground", "pipe"],
            "pipe_mount": ["ground", "pipe"],
            "piranha_avoidance": ["ground", "pipe"],
        }
        for family, kinds in expected.items():
            sample = sample_block_smb_monte_carlo_scenario(
                split="train", seed=3, sample_index=0, family=family, difficulty="easy"
            )
            self.assertEqual(sample.scenario["platform_kinds"], kinds, family)
            self.assertEqual(len(sample.scenario["platforms"]), len(kinds), family)
        # Composed layouts tag one pipe per pipe or plant section.
        for family in ("chained_obstacles", "full_smb_opening_proxy", "tactics_obstacle_sequence"):
            sample = sample_block_smb_monte_carlo_scenario(
                split="train", seed=3, sample_index=0, family=family, difficulty="easy"
            )
            kinds = sample.scenario["platform_kinds"]
            pipes = sum(s in ("pipe", "plant") for s in sample.parameters["sections"])
            self.assertEqual(kinds.count("pipe"), pipes, family)
            self.assertEqual(len(sample.scenario["platforms"]), len(kinds), family)


if __name__ == "__main__":
    unittest.main()
