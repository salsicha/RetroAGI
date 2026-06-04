"""
Semantic Segmentation module for MarioScenarioEnv.
Segments frames perfectly by mapping deterministic RGB values to class labels.
"""
import numpy as np
import pygame
import sys
import os

# Ensure we can import the environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mario_scenario_env import MarioScenarioEnv

# Define the exact RGB colors used in the environment and map them to class IDs
COLORS_TO_CLASSES = {
    (107, 140, 255): 0, # Background (Sky Blue)
    (255, 0, 0): 1,     # Mario (Red)
    (139, 69, 19): 2,   # Platforms (Brown)
    (255, 215, 0): 3,   # Coins (Gold)
    (0, 255, 0): 4      # Goal (Green)
}

# Distinct colors for visual debugging of the segmented mask
VISUALIZATION_COLORS = {
    0: (0, 0, 0),       # Class 0: Black
    1: (255, 50, 50),   # Class 1: Bright Red
    2: (100, 100, 255), # Class 2: Bright Blue
    3: (255, 255, 0),   # Class 3: Bright Yellow
    4: (0, 255, 255)    # Class 4: Cyan
}

def segment_frame(rgb_array):
    """
    Takes an RGB array (H, W, 3) and returns a 2D class mask (H, W).
    """
    # Initialize the mask with zeros (default to Background class)
    mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
    
    for color, class_id in COLORS_TO_CLASSES.items():
        # Find all pixels that exactly match the RGB color
        matches = np.all(rgb_array == color, axis=-1)
        mask[matches] = class_id
        
    return mask

def mask_to_rgb(mask):
    """
    Converts a 2D class mask (H, W) back into a colored RGB array (H, W, 3) 
    for visual debugging purposes.
    """
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in VISUALIZATION_COLORS.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

if __name__ == "__main__":
    # Initialize the environment
    env = MarioScenarioEnv()
    obs = env.reset()
    
    # Setup a display that is exactly twice as wide to show side-by-side
    pygame.init()
    display = pygame.display.set_mode((env.width * 2, env.height))
    pygame.display.set_caption("Left: Original RGB | Right: Segmented Mask")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Random agent
        action = np.random.choice([0, 1, 1, 2, 2, 5])
        obs, reward, done, info = env.step(action)
        
        # 1. Perform semantic segmentation
        class_mask = segment_frame(obs)
        
        # 2. Convert mask to high-contrast RGB for visualizing
        vis_rgb = mask_to_rgb(class_mask)
        
        # Blit original array to the left side
        surface_orig = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surface_orig, (0, 0))
        
        # Blit segmented array to the right side
        surface_seg = pygame.surfarray.make_surface(np.transpose(vis_rgb, (1, 0, 2)))
        display.blit(surface_seg, (env.width, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
        if done:
            env.reset()
            
    pygame.quit()