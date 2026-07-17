"""Second-game smoke coverage for the multi-game profile contracts."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from retroagi import experiments
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    PONG_GAME_SPEC,
    AgentWorldModelCritic,
    PongSignalExtractor,
    StageSpec,
    build_architecture,
    get_game_plugin,
)


def _write_synthetic_summary(argv):
    output_path = Path(argv[argv.index("--output") + 1])
    checkpoint_path = Path(argv[argv.index("--checkpoint") + 1])
    summary = {
        "config": {
            "seed": int(argv[argv.index("--seed") + 1]),
            "device": argv[argv.index("--device") + 1],
            "checkpoint_path": str(checkpoint_path),
            "architecture_name": BASELINE_ARCHITECTURE_NAME,
            "architecture_config": {"hidden_dim": 8},
        },
        "metrics": {"controller_mse": 0.25},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary), encoding="utf-8")
    return 0


class TestSecondGameSmoke(unittest.TestCase):
    def test_pong_profile_exercises_registry_actions_signals_and_manifest(self):
        plugin = get_game_plugin("pong")
        game = plugin.game
        stage = StageSpec(
            name=game.block_game_spec().stage_name,
            observation_kind="pong block smoke",
            action_kind="pong discrete actions",
            seq_len_a=4,
            ratio_ab=2,
            ratio_bc=2,
            vocab_size=8,
            action_space_name=game.name,
            action_count=game.action_count,
            action_names=tuple(action.name for action in game.action_space),
        )

        model = build_architecture(
            BASELINE_ARCHITECTURE_NAME,
            stage,
            {"hidden_dim": 8},
        )
        self.assertIsInstance(model, AgentWorldModelCritic)
        self.assertEqual(plugin.signal_extractor("block"), "retroagi.core.PongSignalExtractor")

        self.assertIs(game, PONG_GAME_SPEC)
        self.assertEqual(game.action_backend_id("up"), 2)
        self.assertEqual(game.action_backend_id("down"), 5)
        np.testing.assert_array_equal(
            game.action_button_vector("up", ("DOWN", "UP")),
            np.asarray([0, 1], dtype=np.int8),
        )

        extractor = PongSignalExtractor()
        signals = extractor.extract(
            {
                "ball_position": (0.75, 0.4),
                "ball_velocity": (-0.2, 0.1),
                "paddle_y": 0.35,
                "opponent_y": 0.6,
                "score_delta": 1,
                "rally_length": 12,
                "rally_hit": True,
                "termination_reason": "point_scored",
            },
            terminated=True,
            truncated=False,
        )
        self.assertEqual(extractor.game_name, "pong")
        self.assertEqual(signals.position, (0.75, 0.4))
        self.assertEqual(signals.ball_velocity, (-0.2, 0.1))
        self.assertEqual(signals.score, 1)
        self.assertEqual(signals.progress, 12.0)
        self.assertTrue(signals.completion)
        self.assertFalse(signals.death)
        self.assertTrue(signals.objectives["rally_hit"])

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"
            artifacts_dir = Path(tmpdir) / "artifacts"
            with patch(
                "retroagi.stages.synthetic_1d.cli.main",
                side_effect=_write_synthetic_summary,
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    exit_code = experiments.main(
                        [
                            "--game",
                            "pong",
                            "--stage",
                            "synthetic",
                            "--output",
                            str(output),
                            "--artifacts-dir",
                            str(artifacts_dir),
                            "--device",
                            "cpu",
                            "--architecture",
                            "baseline",
                            "--architecture-config",
                            "hidden_dim=8",
                            "--gate",
                            "synthetic:controller_mse<=1.0",
                        ]
                    )
                self.assertIn('"name": "pong"', stream.getvalue())
                manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["game"]["name"], "pong")
        self.assertEqual(manifest["game"]["backend"]["contract"]["provider_kind"], "gymnasium")
        self.assertEqual(
            [stage["name"] for stage in manifest["game"]["stage_ladder"]],
            ["synthetic", "block", "full"],
        )
        self.assertEqual(manifest["game"]["content_identifiers"], [])
        self.assertEqual(manifest["game"]["asset_checklist"][0]["target"], "generated_data")
        synthetic_stage = manifest["stages"][0]["game_stage"]
        self.assertEqual(synthetic_stage["name"], "synthetic")
        self.assertEqual(synthetic_stage["stage_adapter"], "retroagi.stages.synthetic_1d.train")
        self.assertEqual(synthetic_stage["vision_encoder"], "retroagi.core.LinearVisionEncoder")
        self.assertTrue(manifest["stages"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
