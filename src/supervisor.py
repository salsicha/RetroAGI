import torch
import os

class Supervisor:
    """
    The Supervisor handles the online training of the lobes, calculates
    prediction errors based on game rewards/penalties, and manages serialization.
    """
    def __init__(self, lobes, checkpoint_dir='data/checkpoints/'):
        self.lobes = lobes
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def update(self, state, action, next_state, reward, info):
        """
        Updates models based on the prediction error (Surprise).
        In Predictive Coding, learning is driven by the difference between 
        predicted state and actual outcome.
        """
        # Penalty for dying or hitting enemies
        life_lost = info.get('life', 2) < 2 # Simplified check
        
        # Supervisor creates a training signal based on objectives
        # Positive: Coins, Progress (x_pos)
        # Negative: Time loss, Death, Enemy collision
        training_signal = {
            'reward': reward,
            'x_pos': info.get('x_pos'),
            'coins': info.get('coins'),
            'collision': life_lost,
            'actual_next_state': next_state
        }

        # Iterate through lobes to perform online weight updates
        for lobe in self.lobes:
            # Each lobe implements its own 'learn' method using pyhgf/backprop
            # to minimize surprise/prediction error
            lobe.learn(training_signal)

    def checkpoint(self):
        """Serializes model weights to disk."""
        print(f"Serializing models to {self.checkpoint_dir}...")
        for lobe in self.lobes:
            lobe_name = lobe.__class__.__name__
            path = os.path.join(self.checkpoint_dir, f"{lobe_name}.pth")
            lobe.save(path)

    def set_planning_mode(self, value):
        """Updates the Prefrontal Lobe's mode (0: Speedrun, 1: Coins)"""
        for lobe in self.lobes:
            if hasattr(lobe, 'planning_mode'):
                lobe.planning_mode = value
                print(f"Planning mode set to: {'Speedrun' if value == 0 else 'Max Coins' if value == 1 else value}")