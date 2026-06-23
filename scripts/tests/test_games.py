"""Tests for game profile contracts."""

import unittest

from retroagi.core import (
    SMB_ACTIONS,
    SMB_GAME_SPEC,
    ActionSpec,
    GameSpec,
    assert_stage_ladder,
    game_names,
    get_game_spec,
    validate_game_spec,
)


class TestGameSpec(unittest.TestCase):
    def test_smb_game_spec_declares_progressive_resolution_contract(self):
        spec = get_game_spec("smb")

        self.assertIs(spec, SMB_GAME_SPEC)
        self.assertIn("smb", game_names())
        self.assertEqual(spec.family, "super_mario_bros")
        self.assertEqual(spec.action_count, len(SMB_ACTIONS))
        self.assertEqual(
            spec.stage_names,
            ("synthetic", "block", "full_asset_mock", "full"),
        )
        assert_stage_ladder(spec, ("synthetic", "block", "full_asset_mock", "full"))

        self.assertIn("full_smb_emulator_rgb_frames", spec.observation_sources)
        self.assertIn("mario", spec.semantic_classes)
        self.assertIn("progress", spec.signal_schema)
        self.assertIn("goal", spec.reward_terms)
        self.assertEqual(spec.emulator_backend, "stable-retro")
        self.assertTrue(any(asset.name == "smb_sprites" for asset in spec.asset_requirements))
        self.assertIn("rom", spec.licensing)

    def test_smb_actions_preserve_stable_ids_and_button_metadata(self):
        spec = SMB_GAME_SPEC
        ids = [action.stable_id for action in spec.action_space]

        self.assertEqual(ids, list(range(len(SMB_ACTIONS))))
        self.assertEqual(spec.action("right_jump").buttons, ("RIGHT", "A"))
        self.assertEqual(spec.action(5).name, "jump")
        with self.assertRaisesRegex(KeyError, "unknown action"):
            spec.action("dash")

    def test_game_spec_validation_rejects_sparse_action_ids(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(
                    ActionSpec("noop", 0),
                    ActionSpec("jump", 2),
                ),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
            )

    def test_validate_game_spec_round_trips_profile(self):
        copy = validate_game_spec(SMB_GAME_SPEC)

        self.assertEqual(copy, SMB_GAME_SPEC)
        with self.assertRaisesRegex(ValueError, "stage ladder"):
            assert_stage_ladder(copy, ("synthetic", "full"))


if __name__ == "__main__":
    unittest.main()
