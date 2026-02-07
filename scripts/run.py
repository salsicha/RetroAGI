"""
Entry point for the RetroAGI agent.
Sets up the environment and executes the main agent loop.
"""
import sys
import os

# Add the project root directory to the Python path
# This allows imports like 'from src.models ...' to work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main import main

if __name__ == "__main__":
    print(f"Starting RetroAGI from {project_root}...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()