"""Tests for game profile contracts."""

import unittest

import numpy as np

from retroagi.core import (
    BACKEND_PROVIDER_KINDS,
    GAME_SPECS,
    GAME_PLUGIN_REGISTRY,
    PONG_ACTION_SPECS,
    PONG_BLOCK_GAME_SPEC,
    PONG_GAME_SPEC,
    PONG_GAME_PLUGIN,
    PONG_REWARD_SCHEMA,
    PONG_SYNTHETIC_DATA_SPECS,
    PONG_TASK_SCHEMA,
    SMB_ACTIONS,
    SMB_BLOCK_GAME_SPEC,
    SMB_GAME_SPEC,
    SMB_GAME_PLUGIN,
    SMB_REWARD_SCHEMA,
    SMB_SYNTHETIC_DATA_SPECS,
    SMB_TASK_SCHEMA,
    ActionSpec,
    AssetChecklistItem,
    AssetRequirement,
    BackendCapabilitySpec,
    BlockGameSpec,
    GamePluginRegistry,
    GamePluginSpec,
    GamePromotionGateSpec,
    GameBackendSpec,
    GameSpec,
    GameTaskSchema,
    GameTaskSpec,
    OPTIONAL_STAGE_NAMES,
    PERCEPTION_DATASET_SOURCE_KINDS,
    PerceptionDatasetSourceSpec,
    PerceptionPipelineSpec,
    STANDARD_STAGE_NAMES,
    RewardConfigSchema,
    RewardTermSpec,
    SemanticVocabularySpec,
    StageLadderEntry,
    SyntheticDataSpec,
    SyntheticSplitSpec,
    TaskSuccessThreshold,
    assert_stage_ladder,
    game_names,
    game_plugin_names,
    get_game_plugin,
    get_game_spec,
    is_standard_stage_name,
    normalize_stage_name,
    resolve_game_stage,
    stage_name_choices,
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
        self.assertEqual(
            tuple(item.name for item in spec.asset_checklist),
            (
                "smb_sprites_source_license",
                "smb_rom_local_only",
                "smb_generated_data_provenance",
            ),
        )
        self.assertEqual(
            spec.asset_checklist_item("smb_generated_data_provenance").target,
            "generated_data",
        )
        self.assertIn("rom", spec.licensing)
        self.assertIs(spec.reward_schema, SMB_REWARD_SCHEMA)
        self.assertIs(spec.task_schema, SMB_TASK_SCHEMA)
        self.assertEqual(spec.synthetic_data, SMB_SYNTHETIC_DATA_SPECS)
        self.assertIs(spec.block_game, SMB_BLOCK_GAME_SPEC)
        self.assertEqual(BACKEND_PROVIDER_KINDS[0], "stable_retro")
        backend = spec.backend_spec()
        self.assertEqual(backend.name, "stable-retro")
        self.assertEqual(backend.provider_kind, "stable_retro")
        self.assertEqual(backend.entrypoint, "retro.make")
        self.assertTrue(backend.capabilities.reset_seed)
        self.assertTrue(backend.capabilities.save_load_state)
        self.assertTrue(backend.capabilities.legacy_gym_step_api)
        self.assertEqual(backend.metadata["game"], "SuperMarioBros-Nes")

    def test_pong_game_spec_declares_non_smb_three_rung_profile(self):
        spec = get_game_spec("pong")

        self.assertIs(spec, PONG_GAME_SPEC)
        self.assertIn("pong", game_names())
        self.assertEqual(spec.family, "pong")
        self.assertEqual(spec.action_space, PONG_ACTION_SPECS)
        self.assertEqual(spec.action_count, 3)
        self.assertEqual(spec.action("up").backend_id, 2)
        self.assertEqual(spec.action("down").buttons, ("DOWN",))
        self.assertEqual(spec.stage_names, ("synthetic", "block", "full"))
        assert_stage_ladder(spec, ("synthetic", "block", "full"))

        self.assertIn("pong_block_raster_frames", spec.observation_sources)
        self.assertIn("ball", spec.semantic_classes)
        self.assertIn("score_delta", spec.signal_schema)
        self.assertIn("rally_hit", spec.reward_terms)
        self.assertEqual(spec.emulator_backend, "gymnasium-pong")
        backend = spec.backend_spec()
        self.assertEqual(backend.provider_kind, "gymnasium")
        self.assertEqual(backend.entrypoint, "gymnasium.make")
        self.assertEqual(backend.metadata["env_id"], "ALE/Pong-v5")
        self.assertIs(spec.reward_schema, PONG_REWARD_SCHEMA)
        self.assertIs(spec.task_schema, PONG_TASK_SCHEMA)
        self.assertEqual(spec.synthetic_data, PONG_SYNTHETIC_DATA_SPECS)
        self.assertIs(spec.block_game, PONG_BLOCK_GAME_SPEC)
        self.assertEqual(
            tuple(task.name for task in spec.fixed_tasks),
            ("centered_return", "angled_return"),
        )
        self.assertEqual(
            spec.asset_checklist_item("pong_generated_data_provenance").target,
            "generated_data",
        )

    def test_stage_resolution_uses_game_neutral_names_and_legacy_aliases(self):
        self.assertEqual(
            STANDARD_STAGE_NAMES,
            (
                "synthetic",
                "block",
                "full",
                "symbolic",
                "tile",
                "sprite",
                "emulator",
                "full_asset_mock",
            ),
        )
        self.assertIn("sprite", OPTIONAL_STAGE_NAMES)
        self.assertIn("block-smb", stage_name_choices())
        self.assertEqual(normalize_stage_name("synthetic_1d"), "synthetic")
        self.assertEqual(normalize_stage_name("block-smb"), "block")
        self.assertEqual(normalize_stage_name("full_asset_mock"), "full_asset_mock")
        self.assertTrue(is_standard_stage_name("emulator"))
        self.assertFalse(is_standard_stage_name("missing"))

        resolved = resolve_game_stage(SMB_GAME_SPEC, "block_smb")
        self.assertEqual(resolved.name, "block")
        self.assertEqual(resolved.stage_spec_name, "block_smb")
        self.assertEqual(SMB_GAME_PLUGIN.resolve_stage("full-smb").name, "full")
        self.assertEqual(
            SMB_GAME_PLUGIN.stage_adapter("full_smb"),
            "retroagi.stages.full_smb.adapter.FullSMBStage",
        )

    def test_smb_game_plugin_loads_profile_and_components_by_name(self):
        plugin = get_game_plugin("smb")

        self.assertIs(plugin, SMB_GAME_PLUGIN)
        self.assertIs(GAME_PLUGIN_REGISTRY.get("smb"), plugin)
        self.assertEqual(game_plugin_names(), ("pong", "smb"))
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
        block_perception = plugin.perception_pipeline("block")
        full_asset_perception = GAME_PLUGIN_REGISTRY.perception_pipeline(
            "smb", "full_asset_mock"
        )
        self.assertEqual(block_perception.stage_name, "block")
        self.assertEqual(
            block_perception.checkpoint_path,
            "data/block_vit/block_vit.pth",
        )
        self.assertEqual(block_perception.semantic_vocabulary.class_index("mario"), 1)
        self.assertEqual(
            block_perception.asset_extraction,
            (
                "retroagi.stages.block_smb.vision.BlockVisionTransformer."
                "semantic_targets"
            ),
        )
        self.assertEqual(
            block_perception.synthetic_frame_composition,
            "retroagi.stages.block_smb.env.MarioScenarioEnv",
        )
        self.assertEqual(block_perception.diagnostic_thresholds["min_accuracy"], 0.95)
        self.assertTrue(block_perception.supports_source_kind("emulator_state"))
        self.assertFalse(block_perception.supports_source_kind("asset_synthetic"))
        self.assertEqual(
            block_perception.dataset_sources[0].label_source,
            (
                "MarioScenarioEnv symbolic state and "
                "BlockVisionTransformer palette targets"
            ),
        )
        self.assertEqual(full_asset_perception.stage_name, "full_asset_mock")
        self.assertEqual(
            full_asset_perception.semantic_classes,
            SMB_GAME_SPEC.semantic_classes,
        )
        self.assertEqual(
            full_asset_perception.asset_extraction,
            "scripts.vit.extract_sprites",
        )
        self.assertEqual(
            full_asset_perception.synthetic_frame_composition,
            "scripts.vit.generate_dataset",
        )
        self.assertEqual(
            full_asset_perception.checkpoint_path,
            "data/vit/full_smb_vit.pth",
        )
        self.assertEqual(
            full_asset_perception.diagnostic_thresholds["min_class_coverage"],
            13.0,
        )
        self.assertIn(
            "data/vit/val.npz",
            full_asset_perception.to_manifest()["dataset_artifacts"],
        )
        self.assertTrue(full_asset_perception.supports_source_kind("asset_synthetic"))
        self.assertEqual(
            full_asset_perception.to_manifest()["dataset_sources"][0]["source_kind"],
            "asset_synthetic",
        )

        reward_config = plugin.reward_config({"progress": 0.07})
        self.assertEqual(reward_config["progress"], 0.07)
        self.assertEqual(reward_config["goal"], 50.0)
        threshold = GAME_PLUGIN_REGISTRY.success_threshold("smb", "level_1_flat.json")
        self.assertIs(threshold, SMB_GAME_SPEC.task("level_1_flat.json").success_threshold)
        block_gate = plugin.promotion_gate("block-smb-smoke")
        self.assertIs(GAME_PLUGIN_REGISTRY.promotion_gate("smb", "block-smb-smoke"), block_gate)
        assert block_gate is not None
        self.assertEqual(block_gate.failure_reason, "SMB block smoke gate failed")
        self.assertEqual(block_gate.metric_gates[0].metric, "eval_success_rate")
        self.assertEqual(block_gate.metric_gates[0].threshold_key, "success_rate_threshold")
        self.assertEqual(
            [gate.field for gate in block_gate.artifact_gates],
            ["summary_path", "checkpoint_path", "log_path"],
        )

    def test_pong_game_plugin_registers_profile_components(self):
        plugin = get_game_plugin("pong")

        self.assertIs(plugin, PONG_GAME_PLUGIN)
        self.assertIs(GAME_PLUGIN_REGISTRY.get("pong"), plugin)
        self.assertIs(plugin.game, PONG_GAME_SPEC)
        self.assertIs(GAME_PLUGIN_REGISTRY.game("pong"), PONG_GAME_SPEC)
        self.assertEqual(
            plugin.stage_adapter("synthetic"),
            "retroagi.stages.synthetic_1d.train",
        )
        self.assertEqual(
            GAME_PLUGIN_REGISTRY.stage_adapter("pong", "block"),
            "retroagi.stages.pong_block.adapter.PongBlockStage",
        )
        self.assertEqual(
            GAME_PLUGIN_REGISTRY.stage_adapter("pong", "full"),
            "retroagi.stages.pong_full.adapter.PongFullStage",
        )
        self.assertEqual(
            plugin.vision_encoder("full"),
            "retroagi.stages.pong_full.vision.PongFullVisionEncoder",
        )
        block_perception = plugin.perception_pipeline("block")
        self.assertEqual(block_perception.game_name, "pong")
        self.assertEqual(block_perception.stage_name, "block")
        self.assertEqual(block_perception.semantic_vocabulary.class_index("ball"), 3)
        self.assertTrue(block_perception.supports_source_kind("emulator_state"))
        self.assertFalse(block_perception.supports_source_kind("asset_synthetic"))
        self.assertIsNone(block_perception.asset_extraction)
        self.assertEqual(
            block_perception.to_manifest()["dataset_sources"][0]["source_kind"],
            "emulator_state",
        )
        full_perception = GAME_PLUGIN_REGISTRY.perception_pipeline("pong", "full")
        self.assertEqual(full_perception.stage_name, "full")
        self.assertTrue(full_perception.supports_source_kind("self_supervised"))
        self.assertIsNone(full_perception.asset_extraction)
        self.assertIsNotNone(plugin.success_threshold("centered_return"))
        self.assertIsNotNone(plugin.promotion_gate("synthetic-concept"))

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
                "level_5_enemy_hop.json",
                "level_6_enemy_patrol.json",
                "level_7_moving_bridge.json",
                "level_8_enemy_gap.json",
                "level_9_enemy_stomp.json",
                "level_10_left_retreat.json",
                "level_11_left_jump_recovery.json",
                "level_12_wait_bridge.json",
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
                "level_5_enemy_hop.json",
                "level_6_enemy_patrol.json",
                "level_7_moving_bridge.json",
                "level_8_enemy_gap.json",
                "level_9_enemy_stomp.json",
                "level_10_left_retreat.json",
                "level_11_left_jump_recovery.json",
                "level_12_wait_bridge.json",
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

    def test_smb_asset_checklist_covers_assets_and_generated_data(self):
        spec = SMB_GAME_SPEC
        required_targets = {
            item.target for item in spec.asset_checklist if item.required
        }

        self.assertIn("smb_sprites", required_targets)
        self.assertIn("smb_rom", required_targets)
        self.assertIn("generated_data", required_targets)
        sprites = spec.asset_checklist_item("smb_sprites_source_license")
        self.assertEqual(sprites.stage_names, ("full_asset_mock",))
        self.assertIn("license_or_terms_summary", sprites.evidence)
        self.assertIn("redistribution", sprites.policy)
        rom = spec.asset_checklist_item("smb_rom_local_only")
        self.assertEqual(rom.stage_names, ("full",))
        self.assertIn("gitignore_or_artifact_exclusion", rom.evidence)
        self.assertIn("not be committed", rom.policy)
        generated = spec.asset_checklist_item("smb_generated_data_provenance")
        self.assertEqual(
            generated.stage_names,
            ("synthetic", "block", "full_asset_mock"),
        )
        self.assertIn("split_seeds", generated.evidence)
        self.assertIn("source asset provenance", generated.policy)
        with self.assertRaisesRegex(KeyError, "unknown asset checklist item"):
            spec.asset_checklist_item("missing")

    def test_perception_pipeline_supports_non_asset_dataset_sources(self):
        self.assertEqual(
            PERCEPTION_DATASET_SOURCE_KINDS,
            (
                "asset_synthetic",
                "self_supervised",
                "emulator_state",
                "manual_labels",
            ),
        )
        vocabulary = SemanticVocabularySpec(
            name="unit",
            classes=("background", "actor"),
            background_class="background",
        )

        pipeline = PerceptionPipelineSpec(
            game_name="smb",
            name="full",
            stage_name="full",
            semantic_vocabulary=vocabulary,
            vision_encoder="unit.Vision",
            asset_extraction=None,
            synthetic_frame_composition=None,
            checkpoint_path="data/unit/full_vit.pth",
            diagnostic_thresholds={"min_accuracy": 0.5},
            dataset_sources=(
                PerceptionDatasetSourceSpec(
                    name="contrastive_rollouts",
                    source_kind="self_supervised",
                    stage_names=("full",),
                    observation_source="emulator RGB frame pairs",
                    label_source="temporal contrastive objective",
                    entrypoint="unit.self_supervised",
                    dataset_artifacts=("data/unit/contrastive.npz",),
                ),
                PerceptionDatasetSourceSpec(
                    name="state_snapshots",
                    source_kind="emulator_state",
                    stage_names=("full",),
                    observation_source="emulator RGB frames",
                    label_source="backend object-state snapshots",
                    entrypoint="unit.state_labels",
                    dataset_artifacts=("data/unit/state_labels.npz",),
                ),
                PerceptionDatasetSourceSpec(
                    name="human_labels",
                    source_kind="manual_labels",
                    stage_names=("full",),
                    observation_source="sampled emulator frames",
                    label_source="manual semantic masks",
                    entrypoint="unit.manual_labels",
                    dataset_artifacts=("data/unit/manual_labels.jsonl",),
                ),
            ),
        )

        self.assertEqual(
            pipeline.source_kinds,
            ("self_supervised", "emulator_state", "manual_labels"),
        )
        self.assertTrue(pipeline.supports_source_kind("manual_labels"))
        manifest = pipeline.to_manifest()
        self.assertIsNone(manifest["asset_extraction"])
        self.assertEqual(
            [source["source_kind"] for source in manifest["dataset_sources"]],
            ["self_supervised", "emulator_state", "manual_labels"],
        )

    def test_game_spec_validation_rejects_mismatched_backend_spec(self):
        with self.assertRaisesRegex(ValueError, "must match emulator_backend"):
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
                backend=GameBackendSpec(
                    name="other",
                    provider_kind="custom",
                    entrypoint="unit.make",
                    observation_api="obs",
                    action_api="action",
                    capabilities=BackendCapabilitySpec(
                        reset_seed=False,
                        save_load_state=False,
                        frame_step=True,
                        action_repeat=False,
                        render=False,
                        headless=True,
                    ),
                ),
                licensing={"assets": "none"},
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

    def test_game_spec_validation_rejects_non_standard_stage_names(self):
        with self.assertRaisesRegex(ValueError, "non-standard names"):
            GameSpec(
                name="bad",
                family="unit",
                action_space=(ActionSpec("noop", 0),),
                observation_sources=("state",),
                semantic_classes=("background",),
                signal_schema={"progress": "progress"},
                reward_terms={"progress": "reward"},
                stage_ladder=(
                    StageLadderEntry("synthetic", "synthetic_1d", "concept"),
                    StageLadderEntry("weird", "weird_stage", "unsupported"),
                    StageLadderEntry("full", "full_stage", "final"),
                ),
                emulator_backend="mock",
                licensing={"assets": "none"},
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
        unit_vocab = SemanticVocabularySpec(
            name="unit",
            classes=("background",),
            background_class="background",
        )

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

        with self.assertRaisesRegex(ValueError, "promotion gate keys"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
                promotion_gates={
                    "block-smb-smoke": GamePromotionGateSpec("synthetic-concept")
                },
            )

        with self.assertRaisesRegex(ValueError, "unknown stage"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"full": "unit.Adapter"},
                vision_encoders={"full": "unit.Vision"},
                perception_pipelines={
                    "full": PerceptionPipelineSpec(
                        game_name="smb",
                        name="full",
                        stage_name="full",
                        semantic_vocabulary=unit_vocab,
                        vision_encoder="unit.Vision",
                        asset_extraction=None,
                        synthetic_frame_composition=None,
                        checkpoint_path="data/unit.pth",
                        diagnostic_thresholds={"accuracy": 1.0},
                        dataset_sources=(
                            PerceptionDatasetSourceSpec(
                                name="bad_stage",
                                source_kind="manual_labels",
                                stage_names=("missing",),
                                observation_source="frames",
                                label_source="manual masks",
                            ),
                        ),
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "perception pipeline keys"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
                perception_pipelines={
                    "other": PerceptionPipelineSpec(
                        game_name="smb",
                        name="block",
                        stage_name="block",
                        semantic_vocabulary=unit_vocab,
                        vision_encoder="unit.Vision",
                        asset_extraction="unit.assets",
                        synthetic_frame_composition="unit.compose",
                        checkpoint_path="data/unit.pth",
                        diagnostic_thresholds={"accuracy": 1.0},
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "is for"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
                perception_pipelines={
                    "block": PerceptionPipelineSpec(
                        game_name="other",
                        name="block",
                        stage_name="block",
                        semantic_vocabulary=unit_vocab,
                        vision_encoder="unit.Vision",
                        asset_extraction="unit.assets",
                        synthetic_frame_composition="unit.compose",
                        checkpoint_path="data/unit.pth",
                        diagnostic_thresholds={"accuracy": 1.0},
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "unknown stage"):
            GamePluginSpec(
                name="smb",
                game=SMB_GAME_SPEC,
                stage_adapters={"block": "unit.Adapter"},
                vision_encoders={"block": "unit.Vision"},
                perception_pipelines={
                    "block": PerceptionPipelineSpec(
                        game_name="smb",
                        name="block",
                        stage_name="missing",
                        semantic_vocabulary=unit_vocab,
                        vision_encoder="unit.Vision",
                        asset_extraction="unit.assets",
                        synthetic_frame_composition="unit.compose",
                        checkpoint_path="data/unit.pth",
                        diagnostic_thresholds={"accuracy": 1.0},
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "finite numbers"):
            PerceptionPipelineSpec(
                game_name="smb",
                name="block",
                stage_name="block",
                semantic_vocabulary=unit_vocab,
                vision_encoder="unit.Vision",
                asset_extraction="unit.assets",
                synthetic_frame_composition="unit.compose",
                checkpoint_path="data/unit.pth",
                diagnostic_thresholds={"accuracy": float("inf")},
            )

        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            PerceptionDatasetSourceSpec(
                name="bad_kind",
                source_kind="sprites_only",
                stage_names=("block",),
                observation_source="frames",
                label_source="labels",
            )

        with self.assertRaisesRegex(ValueError, "asset-synthetic sources"):
            PerceptionPipelineSpec(
                game_name="smb",
                name="full",
                stage_name="full",
                semantic_vocabulary=unit_vocab,
                vision_encoder="unit.Vision",
                asset_extraction=None,
                synthetic_frame_composition=None,
                checkpoint_path="data/unit.pth",
                diagnostic_thresholds={"accuracy": 1.0},
                dataset_sources=(
                    PerceptionDatasetSourceSpec(
                        name="asset_mock",
                        source_kind="asset_synthetic",
                        stage_names=("full",),
                        observation_source="sprites",
                        label_source="synthetic masks",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "dataset source names"):
            PerceptionPipelineSpec(
                game_name="smb",
                name="full",
                stage_name="full",
                semantic_vocabulary=unit_vocab,
                vision_encoder="unit.Vision",
                asset_extraction=None,
                synthetic_frame_composition=None,
                checkpoint_path="data/unit.pth",
                diagnostic_thresholds={"accuracy": 1.0},
                dataset_sources=(
                    PerceptionDatasetSourceSpec(
                        name="dup",
                        source_kind="manual_labels",
                        stage_names=("full",),
                        observation_source="frames",
                        label_source="manual masks",
                    ),
                    PerceptionDatasetSourceSpec(
                        name="dup",
                        source_kind="emulator_state",
                        stage_names=("full",),
                        observation_source="frames",
                        label_source="state labels",
                    ),
                ),
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

    def test_game_spec_validation_rejects_invalid_asset_checklists(self):
        base = dict(
            name="bad",
            family="unit",
            action_space=(ActionSpec("noop", 0),),
            observation_sources=("state",),
            semantic_classes=("background",),
            signal_schema={"progress": "progress"},
            reward_terms={"progress": "reward"},
            stage_ladder=SMB_GAME_SPEC.stage_ladder,
            emulator_backend="mock",
            licensing={
                "assets": "unit assets",
                "generated_data": "unit generated data",
            },
        )

        with self.assertRaisesRegex(ValueError, "must define asset_checklist"):
            GameSpec(
                **base,
                synthetic_data=(
                    SyntheticDataSpec(
                        game_name="bad",
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

        with self.assertRaisesRegex(ValueError, "unknown target"):
            GameSpec(
                **base,
                asset_checklist=(
                    AssetChecklistItem(
                        name="missing",
                        target="missing",
                        stage_names=("synthetic",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "unknown stages"):
            GameSpec(
                **base,
                asset_checklist=(
                    AssetChecklistItem(
                        name="bad_stage",
                        target="assets",
                        stage_names=("missing",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "must cover generated data"):
            GameSpec(
                **base,
                synthetic_data=(
                    SyntheticDataSpec(
                        game_name="bad",
                        name="concept",
                        stage_name="synthetic_1d",
                        observation_kind="state",
                        target_kind="target",
                        generator="unit.generate",
                        splits=(SyntheticSplitSpec("train", 1, 1),),
                        shape_contract={"x": (1,)},
                    ),
                ),
                asset_checklist=(
                    AssetChecklistItem(
                        name="assets_only",
                        target="assets",
                        stage_names=("synthetic",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "must cover required assets"):
            GameSpec(
                **base,
                asset_requirements=(
                    AssetRequirement(
                        name="unit_asset",
                        required=True,
                        local_path="assets/unit",
                        provenance="unit source",
                        license_notes="unit license",
                    ),
                ),
                asset_checklist=(
                    AssetChecklistItem(
                        name="generic_assets",
                        target="assets",
                        stage_names=("synthetic",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "item names must be unique"):
            GameSpec(
                **base,
                asset_checklist=(
                    AssetChecklistItem(
                        name="dup",
                        target="assets",
                        stage_names=("synthetic",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                    AssetChecklistItem(
                        name="dup",
                        target="assets",
                        stage_names=("synthetic",),
                        evidence=("source",),
                        policy="unit policy",
                    ),
                ),
            )

    def test_validate_game_spec_round_trips_profile(self):
        copy = validate_game_spec(SMB_GAME_SPEC)

        self.assertEqual(copy, SMB_GAME_SPEC)
        with self.assertRaisesRegex(ValueError, "stage ladder"):
            assert_stage_ladder(copy, ("synthetic", "full"))


if __name__ == "__main__":
    unittest.main()
