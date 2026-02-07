"""
Synthetic Data Synthesizer
Generates semantic labels and targets from raw game state data.
"""
import torch
import numpy as np
import cv2

class DataSynthesizer:
    def __init__(self, screen_width=256, screen_height=256):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def generate_temporal_label(self, current_info, prev_info):
        """
        Generates a text description of the event based on state change.
        """
        if prev_info is None:
            return "mario starts level"
        
        events = []
        
        # Movement
        dx = current_info.get('xscrollLo', 0) - prev_info.get('xscrollLo', 0)
        # Handle overflow if needed, but simplistic check:
        if dx > 0:
            events.append("moves right")
        elif dx < 0:
            events.append("moves left")
            
        # Jump (requires vertical velocity or y pos monitoring, assume generic for now if y changes)
        # Note: 'y' is often 0 at top or bottom depending on game. 
        # Let's assume some vertical change implies jumping/falling.
        # But 'info' dict depends on integration. 
        # We'll stick to basic scroll for "movement".
        
        # Resources
        if current_info.get('coins', 0) > prev_info.get('coins', 0):
            events.append("collects coin")
            
        if current_info.get('score', 0) > prev_info.get('score', 0):
            events.append("gains score")
            
        if not events:
            return "mario waits"
            
        return "mario " + " and ".join(events)

    def generate_parietal_target(self, current_info):
        """
        Generates a 2D saliency map/objective map.
        For synthetic data, we assume the objective is always to the right.
        """
        # Create a Gaussian blob on the right side of the screen
        target_map = np.zeros((32, 32), dtype=np.float32)
        
        # Center of objective (Right side, middle height)
        # In a real scenario, this would track enemies or coins.
        # Synthetically: "Go Right"
        center_x, center_y = 28, 16 
        sigma = 4.0
        
        y_grid, x_grid = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
        gaussian = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
        
        return gaussian

    def generate_frontal_label(self, current_info):
        """
        Generates a long-term goal description.
        """
        # Very simple state machine
        if current_info.get('time', 0) < 50: # Urgency?
             return "hurry up finish level"
        
        if current_info.get('coins', 0) < 10:
             return "collect more coins"
             
        return "reach end of level"
