"""Shared stomp collision geometry and hindsight duration coaching."""

from typing import Any

import pygame


def stomp_collision_geometry(mario: pygame.Rect, enemy: pygame.Rect, vy: float) -> dict[str, Any]:
    """Use the same integer rectangles and approach test as the physics engine."""
    previous_bottom = mario.bottom - vy
    contact_window = vy > 0 and mario.bottom > enemy.top and previous_bottom <= enemy.centery
    gap = (
        float(mario.left - enemy.right)
        if mario.left >= enemy.right
        else float(mario.right - enemy.left)
        if mario.right <= enemy.left
        else 0.0
    )
    return {
        "mario_rect": list(mario),
        "enemy_rect": list(enemy),
        "vertical_velocity": float(vy),
        "contact_window": bool(contact_window),
        "horizontal_gap": gap,
        "stomp": bool(mario.colliderect(enemy) and vy > 0 and previous_bottom <= enemy.centery),
    }
