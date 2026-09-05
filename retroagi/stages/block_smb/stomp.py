"""Shared stomp collision geometry and hindsight duration coaching."""

from typing import Any, Mapping, Sequence

import pygame


def stomp_collision_geometry(mario: pygame.Rect, enemy: pygame.Rect, vy: float) -> dict[str, Any]:
    """Use the same integer rectangles and approach test as the physics engine."""
    previous_bottom = mario.bottom - vy
    contact_window = vy > 0 and mario.bottom > enemy.top and previous_bottom <= enemy.centery
    gap = (
        float(mario.left - enemy.right)
        if mario.left >= enemy.right
        else float(mario.right - enemy.left) if mario.right <= enemy.left else 0.0
    )
    return {
        "mario_rect": list(mario),
        "enemy_rect": list(enemy),
        "vertical_velocity": float(vy),
        "contact_window": bool(contact_window),
        "horizontal_gap": gap,
        "stomp": bool(mario.colliderect(enemy) and vy > 0 and previous_bottom <= enemy.centery),
    }


def stomp_coaching_target(
    records: Sequence[Mapping[str, Any]],
    held: int,
) -> tuple[float | None, str]:
    """Compare both bodies at the same descending contact time, not proxy centers.

    Re-evaluate recorded collision geometry so a credited goal alone cannot
    anchor a physically invalid contact. A real stomp anchors the hold even
    when the centers are outside the narrower goal proxy. Misses take a
    one-bin step toward contact; unfinished arcs receive no invented target.
    """
    geometry = [
        stomp_collision_geometry(
            pygame.Rect(r["mario_rect"]), pygame.Rect(r["enemy_rect"]), r["vertical_velocity"]
        )
        for r in records
    ]
    if any(g["stomp"] for g in geometry):
        return float(min(16, max(1, held))), "success"
    candidates = [g for g in geometry if g["contact_window"]]
    if not candidates:
        return None, "no_contact"
    closest = min(candidates, key=lambda g: abs(g["horizontal_gap"]))
    # Touching edges is not overlap: disambiguate the zero-gap boundary.
    overshoot = closest["mario_rect"][0] >= closest["enemy_rect"][0] + closest["enemy_rect"][2]
    direction = -1 if overshoot else 1
    return float(min(16, max(1, held + direction))), "overshoot" if overshoot else "undershoot"
