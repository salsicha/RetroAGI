"""
Entry point for the RetroAGI agent.
Sets up the Super Mario Bros retro environment and runs the main loop.
"""
import os
import random
import sys

import retro

# Add the project root directory to the Python path
# This allows package imports to work when this file is executed directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retroagi.core import SMB_ACTIONS, full_smb_action  # noqa: E402


def main(num_steps=200):
    # Initialize the Super Mario Bros environment
    env = retro.make(game='SuperMarioBros-Nes')

    obs = env.reset()

    count = num_steps
    while True:
        count -= 1
        if count < 0:
            break

        # Explore using the same named action vocabulary as Block SMB.
        action = random.choice(SMB_ACTIONS)
        obs, rew, done, term, info = env.step(full_smb_action(action, env.buttons))

        env.render()

        if done:
            env.reset()

    env.close()


if __name__ == "__main__":
    print(f"Starting RetroAGI from {project_root}...")

    # Allow overriding the number of steps: python run.py --steps 500
    steps = 200
    if len(sys.argv) > 2 and sys.argv[1] == '--steps':
        steps = int(sys.argv[2])

    try:
        main(num_steps=steps)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
