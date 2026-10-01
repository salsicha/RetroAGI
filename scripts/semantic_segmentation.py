"""
Watch MarioScenarioEnv beside its exact per-pixel labels.

Labels come from MarioScenarioEnv.render_labels(), which draws the frame's own
shapes with each shape's pixel type (retroagi.core.smb_pixel_types) instead of
its colour, so they are exact by construction.
"""

import os
import sys

import numpy as np
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retroagi.core.smb_pixel_types import PIXEL_TYPES
from retroagi.stages.block_smb.env import MarioScenarioEnv

# High-contrast colour per pixel type, in PIXEL_TYPES order.
VISUALIZATION_COLORS = np.array(
    [
        (0, 0, 0),  # background
        (255, 50, 50),  # mario
        (139, 69, 19),  # ground
        (200, 76, 12),  # brick
        (252, 160, 68),  # question_block
        (0, 200, 0),  # pipe
        (255, 255, 0),  # coin
        (255, 0, 255),  # enemy
        (150, 150, 150),  # moving_platform
    ],
    dtype=np.uint8,
)
assert len(VISUALIZATION_COLORS) == len(PIXEL_TYPES)


def labels_to_rgb(labels):
    """Colour an (H, W) label image for viewing."""
    return VISUALIZATION_COLORS[labels]


if __name__ == "__main__":
    env = MarioScenarioEnv()
    obs, info = env.reset(scenario=MarioScenarioEnv.generate_scenario(seed=0))

    pygame.init()
    display = pygame.display.set_mode((env.width * 2, env.height))
    pygame.display.set_caption("Left: frame | Right: per-pixel types")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.random.choice([0, 1, 1, 2, 2, 5])
        obs, reward, terminated, truncated, info = env.step(action)
        labels = labels_to_rgb(env.render_labels())

        display.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0, 0))
        display.blit(pygame.surfarray.make_surface(np.transpose(labels, (1, 0, 2))), (env.width, 0))
        pygame.display.flip()
        clock.tick(30)

        if terminated or truncated:
            obs, info = env.reset(scenario=MarioScenarioEnv.generate_scenario(seed=env.steps))

    pygame.quit()
