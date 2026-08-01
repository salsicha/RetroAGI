"""Tests for Block SMB Monte Carlo scenario sampling."""

import unittest

from retroagi.stages.block_smb import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    BLOCK_SMB_MC_SCHEMA_VERSION,
    DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    block_smb_monte_carlo_family_specs,
    block_smb_monte_carlo_metadata,
    block_smb_monte_carlo_oracle_actions,
    sample_block_smb_monte_carlo_parameter_sweep,
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
    stable_block_smb_monte_carlo_seed,
    validate_block_smb_monte_carlo_oracle,
)


class TestBlockSMBMonteCarlo(unittest.TestCase):
    def test_family_specs_cover_supported_distribution(self):
        specs = block_smb_monte_carlo_family_specs()

        self.assertEqual(set(specs), set(BLOCK_SMB_MC_FAMILIES))
        for family, spec in specs.items():
            self.assertEqual(spec.schema_version, BLOCK_SMB_MC_SCHEMA_VERSION)
            self.assertEqual(spec.distribution_id, DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID)
            self.assertEqual(spec.family, family)
            self.assertTrue(spec.constraints["requires_oracle_reachability"])
            self.assertEqual(spec.oracle["kind"], "scripted_action_sequence")

    def test_stable_seed_uses_replay_tuple(self):
        seed_a = stable_block_smb_monte_carlo_seed(
            DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
            "validation",
            123,
            5,
        )
        seed_b = stable_block_smb_monte_carlo_seed(
            DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
            "validation",
            123,
            5,
        )
        seed_c = stable_block_smb_monte_carlo_seed(
            DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
            "validation",
            123,
            6,
        )

        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)

    def test_sample_contains_replay_metadata_and_oracle_actions(self):
        sample = sample_block_smb_monte_carlo_scenario(
            split="validation",
            seed=123,
            sample_index=0,
            family="single_gap",
        )

        self.assertEqual(sample.schema_version, BLOCK_SMB_MC_SCHEMA_VERSION)
        self.assertEqual(sample.family, "single_gap")
        self.assertEqual(sample.split, "validation")
        self.assertTrue(sample.reachability["reachable"])
        self.assertEqual(sample.scenario_id, "block_smb_mc_v1.validation.123.000000.single_gap")
        metadata = block_smb_monte_carlo_metadata(sample.scenario)
        self.assertEqual(metadata["scenario_id"], sample.scenario_id)
        self.assertEqual(metadata["parameters"]["difficulty_bin"], sample.difficulty_bin)
        actions = block_smb_monte_carlo_oracle_actions(sample.scenario, max_steps=32)
        self.assertEqual(len(actions), 32)
        self.assertTrue(any(action == 2 for action in actions))

    def test_split_is_deterministic_and_family_balanced_by_default(self):
        sample_set_a = sample_block_smb_monte_carlo_split(
            split="train",
            seed=9,
            sample_count=len(BLOCK_SMB_MC_FAMILIES),
        )
        sample_set_b = sample_block_smb_monte_carlo_split(
            split="train",
            seed=9,
            sample_count=len(BLOCK_SMB_MC_FAMILIES),
        )

        self.assertEqual(sample_set_a.manifest(), sample_set_b.manifest())
        self.assertEqual(sample_set_a.sample_count, len(BLOCK_SMB_MC_FAMILIES))
        self.assertEqual(
            sample_set_a.manifest()["coverage"]["family_counts"],
            {family: 1 for family in BLOCK_SMB_MC_FAMILIES},
        )
        self.assertFalse(sample_set_a.manifest()["coverage"]["missing_families"])

    def test_weighted_split_limits_sampled_families(self):
        sample_set = sample_block_smb_monte_carlo_split(
            split="train",
            seed=3,
            sample_count=4,
            family_weights={"flat_run": 1.0},
        )

        self.assertEqual(
            sample_set.manifest()["coverage"]["family_counts"],
            {"flat_run": 4},
        )

    def test_mixed_section_chains_multiple_obstacles_and_enemies(self):
        sample = sample_block_smb_monte_carlo_scenario(
            split="validation",
            seed=123,
            sample_index=0,
            family="mixed_section",
            difficulty="hard",
        )

        self.assertEqual(sample.family, "mixed_section")
        self.assertGreaterEqual(sample.scenario["world_width"], 512)
        self.assertGreaterEqual(len(sample.scenario.get("platforms", [])), 3)
        self.assertGreaterEqual(len(sample.scenario.get("enemies", [])), 2)
        self.assertEqual(sample.parameters["section_count"], 5)
        self.assertIn("single_gap", sample.parameters["families"])
        self.assertTrue(sample.reachability["reachable"])
        actions = sample.oracle["actions"]
        self.assertGreaterEqual(actions.count(2), 60)
        self.assertGreaterEqual(actions.count(1), 100)
        reachability = validate_block_smb_monte_carlo_oracle(
            sample.scenario,
            actions,
        )
        self.assertTrue(reachability["reachable"])

    def test_tall_pipe_jump_is_taller_than_other_pipes_and_reachable(self):
        self.assertIn("tall_pipe_jump", BLOCK_SMB_MC_FAMILIES)
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
            for seed in range(6):
                sample = sample_block_smb_monte_carlo_scenario(
                    split="validation",
                    seed=seed,
                    sample_index=0,
                    family="tall_pipe_jump",
                    difficulty=difficulty,
                )
                self.assertEqual(sample.family, "tall_pipe_jump")
                pipe_height = int(sample.parameters["pipe_height"])
                # Taller than the other single-pipe families' typical max (~60)
                # yet under the ~68px jump-height ceiling so the oracle clears it.
                self.assertGreaterEqual(pipe_height, 56)
                self.assertLessEqual(pipe_height, 68)
                self.assertTrue(sample.reachability["reachable"])
                reachability = validate_block_smb_monte_carlo_oracle(
                    sample.scenario,
                    sample.oracle["actions"],
                )
                self.assertTrue(reachability["reachable"])
        # Hard tall pipes must exceed the tallest full_smb_opening_proxy pipe.
        hard = sample_block_smb_monte_carlo_scenario(
            split="stress",
            seed=7,
            sample_index=0,
            family="tall_pipe_jump",
            difficulty="hard",
        )
        self.assertGreaterEqual(int(hard.parameters["pipe_height"]), 62)

    def test_stomp_oracle_dying_into_enemy_is_not_reachable(self):
        # Regression: under goal_on_stomp the goal rect rides the enemy, so
        # the validator's positional fallback would count the frame where
        # Mario dies against the enemy (overlapping the proxy rect) as a
        # reached goal. Only credited stomps may validate as reachable.
        scenario = {
            "world_width": 340,
            "mario": [40, 200],
            "platforms": [[0, 220, 340, 20]],
            "enemies": [[96, 206, 96, 96, 0.0]],
            "coins": [],
            "goal": [94, 186, 16, 20],
            "goal_on_stomp": True,
        }
        walk_into_enemy = [1] * 200
        result = validate_block_smb_monte_carlo_oracle(scenario, walk_into_enemy)
        self.assertFalse(result["reachable"])
        stomp = [2] * 8 + [1] * 192
        result = validate_block_smb_monte_carlo_oracle(scenario, stomp)
        self.assertTrue(result["reachable"])

    def test_jump_suite_families_isolate_b_level(self):
        # Every B-level isolation family hands the A-level decision to the
        # rollout and opts into goal-distance shaping plus the jump-energy
        # regulator, with reachable oracles at every difficulty.
        for family in ("pit_leap", "stomp_mount", "platform_hop"):
            for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
                for seed in range(3):
                    sample = sample_block_smb_monte_carlo_scenario(
                        split="validation",
                        seed=seed,
                        sample_index=0,
                        family=family,
                        difficulty=difficulty,
                    )
                    self.assertTrue(sample.reachability["reachable"], (family, difficulty, seed))
                    self.assertEqual(sample.parameters["a_level_action"], 2)
                    self.assertEqual(sample.scenario["reward_goal_distance_shaping"], 2.0)
                    self.assertEqual(sample.scenario["reward_energy_jump"], -0.15)

    def test_stomp_mount_targets_the_enemy_itself(self):
        # B-level stomp teaching: success is landing ON the enemy
        # (goal_on_stomp), distance bands are disjoint and increasing, and
        # the upper tiers patrol so the interception point moves in flight.
        distances = {}
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
            for seed in range(4):
                sample = sample_block_smb_monte_carlo_scenario(
                    split="validation",
                    seed=seed,
                    sample_index=0,
                    family="stomp_mount",
                    difficulty=difficulty,
                )
                self.assertTrue(sample.reachability["reachable"], (difficulty, seed))
                scenario = sample.scenario
                parameters = sample.parameters
                self.assertTrue(scenario["goal_on_stomp"])
                self.assertEqual(parameters["a_level_action"], 2)
                enemy = scenario["enemies"][0]
                if difficulty == "easy":
                    self.assertEqual(parameters["enemy_speed"], 0.0)
                    self.assertEqual(enemy[2], enemy[3])
                else:
                    self.assertGreater(parameters["enemy_speed"], 0.0)
                    self.assertLess(enemy[2], enemy[3])
                distances.setdefault(difficulty, []).append(
                    int(parameters["enemy_distance"])
                )
        self.assertLess(max(distances["easy"]), min(distances["medium"]))
        self.assertLess(max(distances["medium"]), min(distances["hard"]))

    def test_pipe_mount_isolates_b_level_with_shaping_and_given_action(self):
        self.assertIn("pipe_mount", BLOCK_SMB_MC_FAMILIES)
        heights = {}
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
            for seed in range(4):
                sample = sample_block_smb_monte_carlo_scenario(
                    split="validation",
                    seed=seed,
                    sample_index=0,
                    family="pipe_mount",
                    difficulty=difficulty,
                )
                self.assertTrue(sample.reachability["reachable"])
                parameters = sample.parameters
                # The A-level decision is given to the rollout.
                self.assertEqual(parameters["a_level_action"], 2)
                # Dense vertical-progress gradient is enabled per scenario.
                self.assertEqual(
                    sample.scenario["reward_goal_distance_shaping"], 2.0
                )
                # The goal sits on the pipe top.
                pipe_height = int(parameters["pipe_height"])
                goal = sample.scenario["goal"]
                self.assertEqual(goal[1], 220 - pipe_height - 20)
                heights.setdefault(difficulty, []).append(pipe_height)
        # Difficulty bins produce disjoint, increasing height bands so the
        # height -> duration mapping is identifiable.
        self.assertLess(max(heights["easy"]), min(heights["medium"]))
        self.assertLess(max(heights["medium"]), min(heights["hard"]))

    def test_parameter_sweep_covers_every_family_and_difficulty(self):
        sample_set = sample_block_smb_monte_carlo_parameter_sweep(
            split="validation",
            seed=42,
            repeats_per_difficulty=1,
        )
        manifest = sample_set.manifest()

        self.assertEqual(
            sample_set.sample_count,
            len(BLOCK_SMB_MC_FAMILIES) * len(BLOCK_SMB_MC_DIFFICULTY_BINS),
        )
        self.assertEqual(
            manifest["coverage"]["family_counts"],
            {family: len(BLOCK_SMB_MC_DIFFICULTY_BINS) for family in BLOCK_SMB_MC_FAMILIES},
        )
        self.assertFalse(manifest["coverage"]["missing_families"])
        for family in BLOCK_SMB_MC_FAMILIES:
            for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
                self.assertEqual(
                    manifest["coverage"]["difficulty_bin_counts"][f"{family}:{difficulty}"],
                    1,
                )
        self.assertTrue(all(sample.reachability["reachable"] for sample in sample_set.samples))
        self.assertTrue(all(sample.parameters["parameter_sweep"] for sample in sample_set.samples))


if __name__ == "__main__":
    unittest.main()
