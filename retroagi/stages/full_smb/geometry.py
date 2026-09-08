"""Observable NES collision geometry for the shared SMB policy contract.

Addresses and metatile layout are documented in SMB's disassembly:
https://github.com/MitchellSternke/SuperMarioBros-C/blob/master/docs/smbdis.asm
Only the currently loaded, visible collision buffer is used; no level route,
future emulator rollout, or hidden patrol boundary is supplied to the policy.
"""

from types import SimpleNamespace

import numpy as np
import pygame
import torch

from retroagi.core.interfaces import VisionOutput
from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM, skill_goal_encoding
from retroagi.core.smb_geometry import geometry_features
from retroagi.stages.block_smb.local_traversal import local_objective

# Offsets relative to each object's sprite anchor, read through live box control.
# Derived from BoundBoxCtrlData; right/bottom represent collision boundaries.
BOXES = (
    (2, 8, 14, 32),
    (3, 20, 13, 32),
    (2, 20, 14, 32),
    (2, 9, 14, 21),
    (0, 0, 24, 6),
    (0, 0, 32, 13),
    (0, 0, 48, 13),
    (0, 0, 8, 8),
    (6, 4, 10, 8),
    (3, 14, 13, 20),
    (0, 2, 16, 21),
    (4, 4, 12, 28),
)
NON_SOLID = {0, 0x26, 0x5F, 0x60, 0xC2, 0xC3}
SEMANTIC_MAP = (0, 2, 2, 2, 2, 3, 5, 5, 1, 0, 0, 0, 0)


def remap_full_vision(vision, *, visual_tokens="native_unaligned"):
    """Sum probabilities by meaning; never reinterpret native class IDs."""
    if vision.semantic_logits.shape[1] != len(SEMANTIC_MAP):
        raise ValueError("Expected the 13-class Full SMB vision vocabulary")
    probabilities = vision.semantic_logits.float().softmax(1)
    mapped = probabilities.new_zeros((probabilities.shape[0], 7, *probabilities.shape[2:]))
    for original, canonical in enumerate(SEMANTIC_MAP):
        mapped[:, canonical] += probabilities[:, original]
    logits = mapped.clamp_min(1e-12).log()
    return VisionOutput(
        position=vision.position,
        semantic_logits=logits,
        semantic_ids=logits.argmax(1),
        tokens=(
            torch.zeros_like(vision.tokens) if visual_tokens == "zero_ablation" else vision.tokens
        ),
        support_logits=vision.support_logits,
        support_ids=vision.support_ids,
        metadata={**(vision.metadata or {}), "semantic_mapping": SEMANTIC_MAP},
    )


def _signed(value):
    return int(value) if value < 128 else int(value) - 256


def _box(ram, slot, scroll):
    index = int(ram[0x499 + slot])
    if index >= len(BOXES):
        raise ValueError(f"Unsupported NES collision-box index {index}")
    left, top, right, bottom = BOXES[index]
    x = int(ram[0x6D + slot]) * 256 + int(ram[0x86 + slot])
    y = (int(ram[0xB5 + slot]) - 1) * 256 + int(ram[0xCE + slot])
    return pygame.Rect(x + left - scroll, y + top, right - left, bottom - top)


def visible_tiles(ram, scroll):
    """Decode the two interleaved 16x13 metatile pages in world coordinates."""
    tiles = []
    for world_column in range(scroll // 16, (scroll + 255) // 16 + 1):
        page, column = divmod(world_column, 16)
        base = 0x500 + (page % 2) * 0xD0
        for row in range(13):
            value = int(ram[base + row * 16 + column])
            if value:
                tiles.append(
                    (pygame.Rect(world_column * 16 - scroll, 32 + row * 16, 16, 16), value)
                )
    return tiles


def _platforms(tiles):
    # Merge contiguous cells in each horizontal row. This preserves exact
    # collision surfaces while avoiding spurious one-tile landing objectives.
    by_row = {}
    for rect, value in tiles:
        if value not in NON_SOLID:
            by_row.setdefault(rect.top, []).append(rect)
    rectangles = []
    for top, row in sorted(by_row.items()):
        for rect in sorted(row, key=lambda r: r.left):
            if rectangles and rectangles[-1].top == top and rectangles[-1].right == rect.left:
                rectangles[-1].width += rect.width
            else:
                rectangles.append(rect.copy())
    merged = []
    for rect in rectangles:
        above = next(
            (
                p
                for p in merged
                if p.left == rect.left and p.right == rect.right and p.bottom == rect.top
            ),
            None,
        )
        if above is not None:
            above.height += rect.height
        else:
            merged.append(rect.copy())
    return [{"rect": r, "moving": False} for r in merged]


class NESGeometry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_enemies = {}
        self.previous_platforms = {}
        self.bouncing = False
        self.last_frame = None
        self.cached = None
        self.target = None
        self.frames = 0

    def observe(self, ram, *, frame, terminated=False, truncated=False):
        if self.last_frame == frame:
            return self.cached
        ram = np.asarray(ram, dtype=np.uint8)
        if ram.ndim != 1 or len(ram) < 0x800:
            raise ValueError("Shared SMB runtime requires validated NES RAM")
        scroll = int(ram[0x71A]) * 256 + int(ram[0x71C])
        box = _box(ram, 0, scroll)
        world_x = box.x + scroll
        # Fractional horizontal speed is 1/16 px/frame in the NES engine.
        vx = _signed(ram[0x57]) / 16.0
        vy = float(_signed(ram[0x9F]))
        grounded = int(ram[0x1D]) == 0 and int(ram[0xB5]) == 1
        tiles = visible_tiles(ram, scroll)
        platforms = _platforms(tiles)
        mario = dict(
            x=float(box.x),
            y=float(box.y),
            w=box.w,
            h=box.h,
            vx=vx,
            vy=vy,
            on_ground=grounded,
            facing=1 if ram[0x33] == 1 else -1,
            skidding=bool(vx and (vx > 0) != (ram[0x33] == 1)),
            coyote_frames=0,
            jump_buffer=0,
            _platform=None,
        )
        support = [
            p
            for p in platforms
            if p["rect"].left < box.right
            and p["rect"].right > box.left
            and abs(p["rect"].top - box.bottom) <= 2
        ]
        if grounded and support:
            mario["_platform"] = min(support, key=lambda p: abs(p["rect"].top - box.bottom))
        enemies = []
        unavailable = ["coyote", "jump_buffer"]  # NES has neither simulator mechanic.
        unsupported_objects = []
        next_enemies = {}
        next_platforms = {}
        stomped = False
        for slot in range(6):
            flag, kind, state = int(ram[0x0F + slot]), int(ram[0x16 + slot]), int(ram[0x1E + slot])
            if slot in self.previous_enemies and state & 0x20 and vy < 0:
                stomped = True
            if not flag or flag & 0x80 or state & 0x20:
                continue
            if 0x24 <= kind <= 0x2C:
                rect = _box(ram, slot + 1, scroll)
                absolute = rect.left + scroll
                previous = self.previous_platforms.get(slot)
                pvx = absolute - previous[0] if previous else 0.0
                next_platforms[slot] = (absolute, rect.top)
                platforms.append(
                    dict(
                        rect=rect,
                        moving=True,
                        move_x=float(rect.left),
                        move_speed=abs(pvx),
                        move_dir=1 if pvx >= 0 else -1,
                        move_min=float(rect.left),
                        move_max=float(rect.left),
                    )
                )
                unavailable.extend(("bridge_min", "bridge_max"))
                if previous is None:
                    unavailable.append("bridge_vx")
                continue
            if kind in (0x2E, 0x30):  # power-up and flagpole marker, not collision hazards
                continue
            if kind > 0x15:
                unsupported_objects.append(kind)
                continue
            enemy = _box(ram, slot + 1, scroll)
            if enemy.right < 0 or enemy.left > 256:
                continue
            absolute = enemy.x + scroll
            old = self.previous_enemies.get(slot)
            # Differencing world coordinates avoids camera-motion contamination.
            evx = (
                (absolute - old[1]) if old and old[0] == kind else _signed(ram[0x58 + slot]) / 16.0
            )
            if abs(evx) > 8:
                evx = _signed(ram[0x58 + slot]) / 16.0
            next_enemies[slot] = (kind, absolute)
            # Bounds are unavailable, not inferred from a single position.
            # Use the documented neutral value plus availability diagnostics;
            # this domain difference must be qualified by emulator adaptation.
            enemies.append(
                dict(
                    x=enemy.x,
                    y=enemy.y,
                    w=enemy.w,
                    h=enemy.h,
                    dead=False,
                    speed=abs(evx),
                    direction=1 if evx >= 0 else -1,
                    patrol_min=enemy.x,
                    patrol_max=enemy.x,
                    slot=slot,
                    kind=kind,
                )
            )
        if enemies:
            unavailable.extend(("enemy_patrol_min", "enemy_patrol_max"))
        coins = [{"rect": r, "collected": False} for r, v in tiles if v in (0xC2, 0xC3)]
        scene = SimpleNamespace(
            mario=mario,
            platforms=platforms,
            enemies=enemies,
            coins=coins,
            goal=pygame.Rect(240, 188, 16, 20),
            world_width=256,
            height=240,
            max_walk_speed=3.0,
            max_fall_speed=8.0,
            steps=self.frames,
            _terrain_left=False,
            _goal_credited=False,
        )
        if grounded and mario["_platform"] is None:
            mario["_platform"] = next(
                (
                    p
                    for p in platforms
                    if p.get("moving")
                    and p["rect"].left < box.right
                    and p["rect"].right > box.left
                    and abs(p["rect"].top - box.bottom) <= 3
                ),
                None,
            )
        objective = local_objective(scene)
        moving = [p["rect"] for p in platforms if p.get("moving") and p["rect"].right > box.left]
        if objective.kind == "gap" and moving:
            from retroagi.stages.block_smb.local_traversal import LocalObjective

            platform = min(moving, key=lambda r: abs(r.left - box.right))
            if platform.left < objective.left:
                objective = LocalObjective("gap", platform.left, platform.right, platform.top)
        if not grounded and self.target is not None:
            kind, left, right, top = self.target
            from retroagi.stages.block_smb.local_traversal import LocalObjective

            objective = LocalObjective(kind, left - scroll, right - scroll, top)
        elif grounded:
            self.target = (
                objective.kind,
                objective.left + scroll,
                objective.right + scroll,
                objective.top,
            )
        skill = {
            "gap": "clear_gap",
            "mount": "mount_platform",
            "enemy": "enemy_clear",
            "retreat": "retreat_recover",
        }.get(objective.kind)
        if grounded:
            self.bouncing = False
        elif stomped:
            self.bouncing = True
        if self.bouncing:
            skill = None
        features = geometry_features(
            scene,
            death=bool(ram[0x0E] in (6, 11) or (ram[0xB5] >= 2 and ram[0x0E] == 8)),
            terminated=terminated,
            truncated=truncated,
        )
        result = dict(
            scene=scene,
            features=features,
            skill_goal=(
                skill_goal_encoding(skill) if skill else torch.zeros(1, SKILL_GOAL_ENCODING_DIM)
            ),
            objective=objective,
            support="ground" if grounded else "air",
            enemy_contact=stomped,
            bouncing=self.bouncing,
            unavailable_features=unavailable,
            unsupported_objects=unsupported_objects,
            world_x=world_x,
            scroll=scroll,
            player_box=list(box),
            frame=frame,
            physics_profile="nes_smb",
            requires_domain_qualification=True,
        )
        self.previous = (world_x, box.y)
        self.previous_enemies = next_enemies
        self.previous_platforms = next_platforms
        self.frames += 1
        self.last_frame, self.cached = frame, result
        return result
