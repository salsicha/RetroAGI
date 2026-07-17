"""Headless random-agent smoke runner for the Full SMB stage."""

import argparse
import os
import sys

# Add the project root directory to the Python path
# This allows package imports to work when this file is executed directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retroagi.stages.full_smb import (  # noqa: E402
    FullSMBSmokeConfig,
    FullSMBStage,
    run_headless_random_agent_smoke,
)


def main(num_steps=200, seed=0, render=False, encode_observations=False):
    stage = FullSMBStage()
    try:
        result = run_headless_random_agent_smoke(
            stage,
            FullSMBSmokeConfig(
                steps=num_steps,
                seed=seed,
                render=render,
                encode_observations=encode_observations,
            ),
        )
        print(
            "Full SMB smoke: "
            f"steps={result.executed_steps} "
            f"resets={result.resets} "
            f"episodes={result.completed_episodes} "
            f"reward={result.total_reward:.3f}"
        )
        return result
    finally:
        stage.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--encode-observations", action="store_true")
    args = parser.parse_args()

    try:
        main(
            num_steps=args.steps,
            seed=args.seed,
            render=args.render,
            encode_observations=args.encode_observations,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
