"""
Entry point for the RetroAGI agent.
Sets up the Super Mario Bros retro environment and runs the main loop.
"""
import sys
import os

import retro

# Add the project root directory to the Python path
# This allows imports like 'from src.models ...' to work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def main(num_steps=200):
    # Initialize the Super Mario Bros environment
    env = retro.make(game='SuperMarioBros-Nes')

    obs = env.reset()

    count = num_steps
    while True:
        count -= 1
        if count < 0:
            break

        # Take a random action to explore the environment
        obs, rew, done, term, info = env.step(env.action_space.sample())

        env.render()

        if done:
            obs = env.reset()

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
