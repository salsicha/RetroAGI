"""
Data Generator Script
CLI entry point to generate synthetic training data for RetroAGI.
"""
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_gen.recorder import DataGenerator

def main():
    print("Starting Synthetic Data Generation Pipeline...")
    
    # Configuration
    game_name = 'SuperMarioBros-Nes'
    output_path = os.path.join(os.getcwd(), 'data', 'synthetic')
    
    generator = DataGenerator(game=game_name, output_dir=output_path)
    
    # Run generation
    # Keep it small for the "CLI" context, user can edit for massive scale
    generator.run(num_episodes=3, max_steps=200)
    
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
