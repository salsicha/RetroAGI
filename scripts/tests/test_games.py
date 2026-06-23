"""Tests for game profile contracts."""

import unittest

import numpy as np

from retroagi.core import (
    GAME_SPECS,
    GAME_PLUGIN_REGISTRY,
    SMB_ACTIONS,
    SMB_BLOCK_GAME_SPEC,
    SMB_GAME_SPEC,
    SMB_GAME_PLUGIN,
    SMB_REWARD_SCHEMA,
    SMB_SYNTHETIC_DATA_SPECS,
    SMB_TASK_SCHEMA,
    ActionSpec,
    BlockGameSpec,
    GamePluginRegistry,
    GamePluginSpec,
    GameSpec,
    GameTaskSchema,
    GameTaskSpec,
    RewardConfigSchema,
    RewardTermSpec,
    SyntheticDataSpec,
    SyntheticSplitSpec,
    TaskSuccessThreshold,
    assert_stage_ladder,
    game_names,
    game_plugin_names,
    get_game_plugin,
    get_game_spec,
    validate_game_spec,
)
from retroagi.stages.synthetic_1d.train import (
    SyntheticSplitSeeds,
    SyntheticSplitSizes,
    generate_dataset_splits,
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
        self.assertIs(spec.reward_schema, SMB_REWARD_SCHEMA)
        self.assertIs(spec.task_schema, SMB_TASK_SCHEMA)
        self.assertEqual(spec.synthetic_data, SMB_SYNTHETIC_DATA_SPECS)
        self.assertIs(spec.block_game, SMB_BLOCK_GAME_SPEC)

    def test_smb_game_plugin_loads_profile_and_components_by_name(self):
        plugin = get_game_plugin("smb")

        self.assertIs(plugin, SMB_GAME_PLUGIN)
        self.assertIs(GAME_PLUGIN_REGISTRY.get("smb"), plugin)
        self.assertEqual(game_plugin_names(), ("smb",))
        self.assertIs(plugin.game, SMB_GAME_SPEC)
        self.assertIs(GAME_PLUGIN_REGISTRY.game("smb"), SMB_GAME_SPEC)
        self.assertEqual(
            plugin.stage_adapter("block"),
            "retroagi.stages.block_smb.adapter.BlockSMBStage",
        )
        self.assertEqual(
            GAME_PLUGIN_REGISTRY.stage_adapter("smb", "full"),
            "retroagi.stages.full_smb.adapter.FullSMBStage",
        )
        self.assertEqual(
            plugin.vision_encoder("block"),
            "retroagi.stages.block_smb.vision.BlockVisionTransformer",
        )
        self.assertEqual(
            GAME_PLUGIN_REGISTRY.vision_encoder("smb", "full_asset_mock"),
            "retroagi.stages.full_smb.vision.FullSMBVisionTransformer",
        )
        self.assertEqual(
            plugin.asset_pipeline("perception"),
            "scripts.vit.generate_dataset",
        )
        self.assertEqual(
            GAME_PLUGIN_REGISTRY.asset_pipeline("smb", "assets"),
            "scripts.vit.extract_sprites",
        )

        reward_config = plugin.reward_config({"progress": 0.07})
        self.assertEqual(reward_config["progress"], 0.07)
        self.assertEqual(reward_config["goal"], 50.0)
        threshold = GAME_PLUGIN_REGISTRY.success_threshold("smb", "level_1_flat.json")
        self.assertIs(threshold, SMB_GAME_SPEC.task("level_1_flat.json").success_threshold)

    def test_smb_actions_preserve_stable_ids_and_button_metadata(self):
        spec = SMB_GAME_SPEC
        ids = [action.stable_id for action in spec.action_space]

        self.assertEqual(ids, list(range(len(SMB_ACTIONS))))
        self.assertEqual(spec.action("right_jump").buttons, ("RIGHT", "A"))
        self.assertTrue(spec.action("noop").release_all)
        self.assertTrue(spec.action("noop").is_noop)
        self.assertEqual(spec.action(5).name, "jump")
        with self.assertRaisesRegex(KeyError, "unknown action"):
            spec.action("dash")

    def test_registered_games_map_policy_actions_to_backend_controls(self):
        for game in GAME_SPECS.values():
            with self.subTest(game=game.name):
                for action in game.action_space:
                    if action.kind != "continuous":
                        self.assertEqual(
                            game.action_backend_id(action.stable_id),
                            action.backend_id,
                        )
                    self.assertIs(game.action(action.name), action)

    def test_smb_button_mapping_uses_profile_metadata_without_button_order(self):
        spec = SMB_GAME_SPEC
        buttons = ("A", "RIGHT", "B", "LEFT", "START")
        permuted_buttons = ("START", "LEFT", "B", "RIGHT", "A")

        mapped = spec.action_button_vector("right_jump", buttons)
        permuted = spec.action_button_vector("right_jump", permuted_buttons)
        np.testing.assert_array_equal(mapped, np.array([1, 1, 0, 0, 0], dtype=np.int8))
        np.testing.assert_array_equal(
            permuted,
            np.array([0, 0, 0, 1, 1], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            spec.action_button_vector("noop", permuted_buttons),
            np.zeros(len(permuted_buttons), dtype=np.int8),
        )

    def test_smb_reward_schema_owns_reward_defaults_and_validation(self):
        defaults = SMB_GAME_SPEC.reward_config()

        self.assertEqual(defaults["progress"], 0.05)
        self.assertEqual(defaults["coin"], 10.0)
        self.assertEqual(defaults["goal"], 50.0)
        self.assertEqual(defaults["fall_death"], -10.0)
        self.assertEqual(set(defaults), set(SMB_GAME_SPEC.reward_terms))
        self.assertEqual(SMB_REWARD_SCHEMA.term("coin").signal, "collectibles.coins")

        tuned = SMB_GAME_SPEC.reward_config({"progress": 0.08, "fall_death": -12.0})
        self.assertEqual(tuned["progress"], 0.08)
        self.assertEqual(tuned["fall_death"], -12.0)
        self.assertEqual(tuned["coin"], defaults["coin"])

        with self.assertRaisesRegex(ValueError, "does not define terms"):
            SMB_GAME_SPEC.reward_config({"trainer_bonus": 1.0})
        with self.assertRaisesRegex(ValueError, "positive reward term"):
            SMB_GAME_SPEC.reward_config({"coin": -1.0})
        with self.assertRaisesRegex(ValueError, "negative reward term"):
            SMB_GAME_SPEC.reward_config({"enemy_hit": 1.0})

    def test_smb_task_schema_declares_fixed_generated_and_curriculum_tasks(self):
        schema = SMB_GAME_SPEC.task_schema
        self.assertIs(schema, SMB_TASK_SCHEMA)
        self.assertIsNotNone(schema)
        assert schema is not None

        fixed_names = tuple(task.name for task in schema.fixed_tasks)
        self.assertEqual(
            fixed_names,
            (
                "level_1_flat.json",
                "level_2_gap.json",
                "level_3_stairs.json",
                "level_4_platforms.json",
            ),
        )
        self.assertEqual(tuple(task.name for task in SMB_GAME_SPEC.fixed_tasks), fixed_names)
        self.assertEqual(
            tuple(task.name for task in schema.curriculum),
            fixed_names + ("generated_block_smb",),
        )
        self.assertEqual(
            schema.reset_seeds()["level_1_flat.json"],
            SMB_GAME_SPEC.task("level_1_flat.json").reset_seed,
        )

        gap = SMB_GAME_SPEC.task("level_2_gap.json")
        self.assertEqual(gap.stage_name, "block_smb")
        self.assertEqual(gap.task_type, "fixed")
        self.assertTrue(gap.source.endswith("level_2_gap.json"))
        self.assertEqual(gap.curriculum_stage, 2)
        self.assertIsNotNone(gap.success_threshold)
        assert gap.success_threshold is not None
        self.assertEqual(gap.success_threshold.min_success_rate, 1.0)
        self.assertEqual(gap.success_threshold.max_steps, 200)

        generated = schema.task("generated_block_smb")
        self.assertEqual(generated.task_type, "procedural")
        self.assertEqual(generated.generation_seed, 50_000)
        self.assertIn("enemy_density", generated.generation_config)

    def test_smb_synthetic_data_spec_drives_pixel_free_concept_splits(self):
        spec = SMB_GAME_SPEC.synthetic_data_spec("synthetic_1d_concept")

        self.assertIs(spec, SMB_SYNTHETIC_DATA_SPECS[0])
        self.assertEqual(spec.stage_name, "synthetic_1d")
        self.assertTrue(spec.metadata["pixel_free"])
        self.assertTrue(spec.metadata["emulator_free"])
        self.assertEqual(
            spec.split_sizes(),
            {"train": 1_000, "validation": 200, "test": 200},
        )
        self.assertEqual(
            spec.split_seeds(),
            {"train": 10_001, "validation": 20_001, "test": 30_001},
        )
        self.assertEqual(spec.split("train").seed, 10_001)
        self.assertEqual(spec.shape_contract["tensors"], ("xa", "ya", "xb", "yb", "xc", "yc"))

        splits = generate_dataset_splits(
            sizes=SyntheticSplitSizes(
                train=spec.split("train").size,
                validation=spec.split("validation").size,
                test=spec.split("test").size,
            ),
            seeds=SyntheticSplitSeeds(
                train=spec.split("train").seed,
                validation=spec.split("validation").seed,
                test=spec.split("test").seed,
            ),
        )

        self.assertEqual(splits.train[0].shape[0], 1_000)
        self.assertEqual(splits.validation[0].shape[0], 200)
        self.assertEqual(splits.test[0].shape[0], 200)

    def test_smb_block_game_spec_declares_reusable_mid_fidelity_simulator(self):
        spec = SMB_GAME_SPEC.block_game_spec()

        self.assertIs(spec, SMB_BLOCK_GAME_SPEC)
        self.assertEqual(spec.stage_name, "block_smb")
        self.assertTrue(spec.metadata["fast_reset"])
        self.assertTrue(spec.metadata["exact_semantic_labels"])
        self.assertTrue(spec.metadata["procedural_scenarios"])
        self.assertIn("state_vec", spec.observation_kind)
        self.assertIn("mario_position", spec.symbolic_state)
        self.assertIn("moving_platform", spec.semantic_classes)
        self.assertEqual(
            spec.exact_label_source("semantics"),
            "BlockVisionTransformer.semantic_targets",
        )
        self.assertEqual(
            spec.fixed_scenario_names,
            (
                "level_1_flat.json",
                "level_2_gap.json",
                "level_3_stairs.json",
                "level_4_platforms.json",
            ),
        )
        self.assertTrue(
            spec.fixed_scenario_source("level_2_gap.json").endswith(
                "level_2_gap.json"
            )
        )
        self.assertEqual(
            tuple(task.name for task in SMB_GAME_SPEC.fixed_tasks),
            spec.fixed_scenario_names,
        )
        self.assertEqual(
            spec.procedural_scenario_generator,
            SMB_GAME_SPEC.task("generated_block_smb").source,
        )

    def test_game_spec_validation_rejects_mismatched_reward_schema(self):
        with self.assertRaisesRegex(ValueError, "reward schema is for"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                reward_schema=RewardConfigSchema(
                    game_name="other",
                    terms=(
                        RewardTermSpec(
                            "progress",
                            1.0,
                            "positive",
                            "progress",
                            "unit progress reward",
                        ),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "missing described terms"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward", "goal": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                reward_schema=RewardConfigSchema(
                    game_name="bad",
                    terms=(
                        RewardTermSpec(
                            "progress",
                            1.0,
                            "positive",
                            "progress",
                            "unit progress reward",
                        ),
                    ),
                ),
            )

    def test_game_spec_validation_rejects_mismatched_task_schema(self):
        with self.assertRaisesRegex(ValueError, "task schema is for"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                task_schema=GameTaskSchema(
                    game_name="other",
                    tasks=(
                        GameTaskSpec(
                            name="fixed",
                            stage_name="block_smb",
                            task_type="fixed",
                            source="fixed.json",
                            reset_seed=1,
                            curriculum_stage=1,
                            success_threshold=TaskSuccessThreshold(
                                1.0,
                                0.0,
                                1,
                                10,
                                "unit threshold",
                            ),
                        ),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "unknown stages"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                task_schema=GameTaskSchema(
                    game_name="bad",
                    tasks=(
                        GameTaskSpec(
                            name="fixed",
                            stage_name="missing_stage",
                            task_type="fixed",
                            source="fixed.json",
                            reset_seed=1,
                            curriculum_stage=1,
                            success_threshold=TaskSuccessThreshold(
                                1.0,
                                0.0,
                                1,
                                10,
                                "unit threshold",
                            ),
                        ),
                    ),
                ),
            )

    def test_game_spec_validation_rejects_invalid_synthetic_data_specs(self):
        with self.assertRaisesRegex(ValueError, "synthetic data .* is for"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                synthetic_data=(
                    SyntheticDataSpec(
                        game_name="other",
                        name="concept",
                        stage_name="synthetic_1d",
                        observation_kind="state",
                        target_kind="target",
                        generator="unit.generate",
                        splits=(SyntheticSplitSpec("train", 1, 1),),
                        shape_contract={"x": (1,)},
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "unknown stage"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                synthetic_data=(
                    SyntheticDataSpec(
                        game_name="bad",
                        name="concept",
                        stage_name="missing_stage",
                        observation_kind="state",
                        target_kind="target",
                        generator="unit.generate",
                        splits=(SyntheticSplitSpec("train", 1, 1),),
                        shape_contract={"x": (1,)},
                    ),
                ),
            )

    def test_game_spec_validation_rejects_invalid_block_game_specs(self):
        with self.assertRaisesRegex(ValueError, "block game spec .* is for"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                block_game=BlockGameSpec(
                    game_name="other",
                    name="block",
                    stage_name="block_smb",
                    adapter="unit.Adapter",
                    environment="unit.Env",
                    physics="unit physics",
                    observation_kind="rgb plus state",
                    symbolic_state=("position",),
                    semantic_classes=("background",),
                    exact_label_sources={"semantics": "unit.labels"},
                    fixed_scenarios={"fixed.json": "fixed.json"},
                    procedural_scenario_generator="unit.generate",
                    reset_modes=("fixed_scenario",),
                ),
            )

        with self.assertRaisesRegex(ValueError, "unknown stage"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=SMB_GAME_SPEC.stage_ladder,
                emulator_backend="mock",
                licensing={"assets": "none"},
                block_game=BlockGameSpec(
                    game_name="bad",
                    name="block",
                    stage_name="missing_stage",
                    adapter="unit.Adapter",
                    environment="unit.Env",
                    physics="unit physics",
                    observation_kind="rgb plus state",
                    symbolic_state=("position",),
                    semantic_classes=("background",),
                    exact_label_sources={"semantics": "unit.labels"},
                    fixed_scenarios={"fixed.json": "fixed.json"},
                    procedural_scenario_generator="unit.generate",
                    reset_modes=("fixed_scenario",),
                ),
            )

    def test_game_plugin_registry_rejects_invalid_plugins(self):
        with self.assertRaisesRegex(ValueError, "must match game profile"):
            GamePluginSpec(
                name="other",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
            )

        with self.assertRaisesRegex(ValueError, "unknown names"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"missing": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
            )

        with self.assertRaisesRegex(ValueError, "unknown fixed tasks"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
                success_thresholds={
                    "missing.json": TaskSuccessThreshold(1.0, 0.0, 1, 10, "unit")
                },
            )

        with self.assertRaisesRegex(ValueError, "names must be unique"):
            GamePluginRegistry((SMB_GAME_PLUGIN, SMB_GAME_PLUGIN))

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
