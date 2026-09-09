"""Shared scene estimation from canonical segmentation, without simulator/RAM inputs.

Unknown quantities have explicit availability slots. This provider is qualified
separately from oracle geometry; its existence does not certify pixel-based play.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pygame
import torch

from retroagi.core.smb_geometry import geometry_features
from retroagi.core.smb_objectives import objective_goal, observable_objective
from retroagi.stages.block_smb.local_traversal import LocalObjective


def component_boxes(mask, minimum_area=1):
    rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    binary = pygame.mask.from_threshold(surface, (255, 255, 255), (1, 1, 1, 255))
    if minimum_area <= 1:
        return binary.get_bounding_rects()
    return [part.get_bounding_rects()[0] for part in binary.connected_components(minimum_area)]


def terrain_rectangles(mask):
    """Linear scan decomposition; exact runs merge across adjacent rows."""
    active = {}
    finished = []
    for y, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        current = {}
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            key = (int(left), int(right))
            rect = active.pop(key, None)
            if rect is None:
                rect = pygame.Rect(key[0], y, key[1] - key[0], 1)
            else:
                rect.h += 1
            current[key] = rect
        finished.extend(active.values())
        active = current
    finished.extend(active.values())
    # A one-pixel classification fringe is not a collision platform. Physical
    # surfaces need a coherent horizontal run and depth before supporting Mario.
    return [{"rect": r, "moving": False} for r in finished if r.w >= 4 and r.h >= 3]


MOTION_TTL = 4


def update_motion_tracks(tracks, boxes, player_center, *, frame, camera_delta, camera_known):
    """Associate visible bodies, retaining bounded estimates across occlusion.

    Hidden tracks aid association only: they never become collision surfaces.
    Ages distinguish measured velocity from a retained estimate. Relative edge
    displacement cancels the camera and uses a common unclipped landmark.
    """
    tracks = [t for t in tracks if frame - t["frame"] <= MOTION_TTL]
    for t in tracks:
        t["camera_sum"] += camera_delta
        t["camera_valid"] &= camera_known
    unmatched = list(tracks)
    updated, estimates = [], {}
    for rect in boxes:
        candidates = []
        for t in unmatched:
            prior = t["rect"]
            edge = (
                "right"
                if rect.left == 0 or prior.left == 0
                else ("left" if rect.right == 256 or prior.right == 256 else "centerx")
            )
            delta = getattr(rect, edge) - getattr(prior, edge)
            elapsed = frame - t["frame"]
            relative = (
                (delta - (player_center - t["player"])) / elapsed
                if (player_center is not None and t["player"] is not None)
                else None
            )
            distance = abs(relative if relative is not None else delta / elapsed)
            if abs(rect.y - prior.y) <= 8 and distance <= 10:
                candidates.append((distance, len(candidates), t, delta, relative))
        if candidates:
            _, _, t, delta, relative = min(candidates, key=lambda c: c[:2])
            unmatched.remove(t)
            elapsed = frame - t["frame"]
            if t["camera_valid"]:
                measured = (delta + t["camera_sum"]) / elapsed
                if frame - t["world_frame"] > MOTION_TTL or (
                    t["world"] and measured * sum(t["world"]) < 0
                ):
                    t["world"].clear()
                t["world"].append(measured)
                t["world_frame"] = frame
            if relative is not None:
                if t["relative"] and relative * sum(t["relative"]) < 0:
                    t["relative"].clear()
                t["relative"].append(relative)
                t["relative_frame"] = frame
        else:
            t = dict(
                world=deque(maxlen=4),
                relative=deque(maxlen=4),
                world_frame=-100,
                relative_frame=-100,
            )
        world_age, relative_age = frame - t["world_frame"], frame - t["relative_frame"]
        estimates[tuple(rect)] = dict(
            vx=float(np.mean(t["world"])) if t["world"] and world_age <= MOTION_TTL else None,
            relative_vx=(
                float(np.mean(t["relative"]))
                if t["relative"] and relative_age <= MOTION_TTL
                else None
            ),
            motion_age=world_age,
            relative_age=relative_age,
        )
        t.update(
            rect=rect.copy(), player=player_center, frame=frame, camera_sum=0, camera_valid=True
        )
        updated.append(t)
    return estimates, updated + unmatched


class PerceivedSMBScene:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.terrain = None
        self.dynamic = None
        self.camera_delta = 0
        self.velocity_history = deque(maxlen=4)
        self.object_history = {5: {}, 6: {}}
        self.stomp_target = None
        self.stomp_complete = False
        self.recent_stomp_contact = -10
        self.previous_bottom = None
        self.previous_grounded = False
        self.scroll = 0
        self.frames = 0
        self.target = None
        self.bouncing = False
        self.previous_vy = 0.0
        self.objects = {5: [], 6: []}
        self.relative_enemy_history = {}
        self.motion_tracks = {5: [], 6: []}
        self.previous_center = None

    def observe(
        self, vision, *, terminated=False, truncated=False, objective_kind=None, goal_direction=1
    ):
        if vision.semantic_logits.shape[0] != 1 or vision.semantic_logits.shape[1] != 7:
            raise ValueError("Scene tracking requires one canonical seven-class observation")
        labels = (
            torch.nn.functional.interpolate(
                vision.semantic_logits.float(), (240, 256), mode="bilinear", align_corners=False
            )
            .argmax(1)[0]
            .cpu()
            .numpy()
        )
        terrain = labels == 2
        # Moving bodies occlude static terrain. Exclude both their old and new
        # silhouettes (with a small boundary margin) from camera registration.
        dynamic = np.isin(labels, (1, 5, 6))
        dynamic = (
            torch.nn.functional.max_pool2d(
                torch.from_numpy(dynamic.astype(np.float32))[None, None], 7, 1, 3
            )[0, 0]
            .numpy()
            .astype(bool)
        )
        camera_known = False
        camera_delta = 0
        if self.terrain is not None:
            errors = []
            edge = np.abs(np.diff(self.terrain.astype(np.int8), axis=1, prepend=0))
            edge = np.maximum.reduce([np.roll(edge, n, axis=1) for n in range(-7, 8)])[:, 8:-8] > 0
            for shift in range(-7, 8):
                current = terrain[:, 8 - shift : 248 - shift]
                valid = ~(self.dynamic[:, 8:-8] | dynamic[:, 8 - shift : 248 - shift])
                # Compare boundary neighborhoods; blank sky must not overwhelm
                # the few stable landmarks that identify camera translation.
                valid &= edge
                if valid.sum() >= 12:
                    errors.append(
                        (float(np.mean((self.terrain[:, 8:-8] != current)[valid])), shift)
                    )
            ordered = sorted(errors)
            if len(ordered) > 1 and ordered[0][0] < 0.1 and ordered[1][0] > ordered[0][0] + 0.005:
                camera_delta = ordered[0][1]
                camera_known = True
            else:
                # Translation of featureless floor is unobservable. Keep the
                # last estimate, explicitly marked unavailable, rather than
                # invent a sudden zero world velocity during scrolling.
                camera_delta = self.camera_delta
        self.camera_delta = camera_delta
        self.scroll += camera_delta
        boxes = component_boxes(labels == 1, minimum_area=24)
        visible = bool(boxes)
        box = max(boxes, key=lambda b: b.w * b.h) if boxes else pygame.Rect(0, 0, 0, 0)
        # Canonical dense training labels describe the player collision body.
        # A teacher emitting sprite outlines must be calibrated before this lane
        # is qualified; do not apply arbitrary engine-only corrections here.
        old = self.previous
        vx = (box.x - old[0] + camera_delta) if old and visible and old[2] else 0.0
        vy = (box.bottom - self.previous_bottom) if old and visible and old[2] else 0.0
        if old and visible and old[2]:
            self.velocity_history.append(vx)
            vx = float(np.mean(self.velocity_history))
        else:
            self.velocity_history.clear()
        velocity_known = bool(old and visible and old[2] and camera_known)
        old_enemies = list(self.objects[5])
        velocities = {}
        relative_velocities = {}
        estimates = {}
        for cls in (5, 6):
            current = sorted(
                component_boxes(labels == cls, minimum_area=12 if cls == 5 else 32),
                key=lambda b: b.w * b.h,
                reverse=True,
            )[:6]
            estimates[cls], self.motion_tracks[cls] = update_motion_tracks(
                self.motion_tracks[cls],
                current,
                box.centerx if visible else None,
                frame=self.frames,
                camera_delta=camera_delta,
                camera_known=camera_known,
            )
            velocities[cls] = {
                key: value["vx"] for key, value in estimates[cls].items() if value["vx"] is not None
            }
            self.objects[cls] = current
        relative_velocities = {key: value["relative_vx"] for key, value in estimates[5].items()}
        motion_memory = {}
        for cls in (5, 6):
            candidates = [
                t
                for t in self.motion_tracks[cls]
                if t["relative"] and self.frames - t["relative_frame"] <= MOTION_TTL
            ]
            track = min(
                candidates,
                key=lambda t: abs(t["rect"].centerx - (t["player"] or box.centerx)),
                default=None,
            )
            if track is not None:
                motion_memory[cls] = dict(
                    relative_vx=float(np.mean(track["relative"])),
                    relative_age=self.frames - track["relative_frame"],
                )
        platforms = terrain_rectangles(terrain)
        fragmented = len(platforms) > 128
        if fragmented:
            # Unqualified/noisy segmentation must not create quadratic
            # obstacle searches or be reported as reliable support geometry.
            platforms = []
        moving = [
            {
                "rect": b,
                "moving": True,
                "move_x": float(b.x),
                "move_speed": 0.0,
                "move_dir": 1,
                "move_min": float(b.x),
                "move_max": float(b.x),
            }
            for b in self.objects[6]
        ]
        for p in moving:
            speed = velocities[6].get(tuple(p["rect"]), 0.0)
            p.update(move_speed=abs(speed), move_dir=1 if speed >= 0 else -1)
            p.update(estimates[6][tuple(p["rect"])])
        platforms.extend(moving)
        supports = [
            p
            for p in platforms
            if box.right > p["rect"].left
            and box.left < p["rect"].right
            and abs(box.bottom - p["rect"].top) <= 2
        ]
        # Proximity alone reports ground one frame before a descending body
        # actually collides. Require stable feet, except the initial contact.
        grounded = bool(
            visible
            and supports
            and (
                (old is None and any(box.bottom == p["rect"].top for p in supports))
                or (
                    old is not None
                    and old[2]
                    and (vy == 0 or (self.previous_grounded and box.y == old[1]))
                )
            )
        )
        if (
            visible
            and old
            and vy > 0
            and any(
                abs(box.bottom - e.top) <= 10
                and box.right > e.left - camera_delta
                and box.left < e.right - camera_delta
                for e in old_enemies
            )
        ):
            self.recent_stomp_contact = self.frames
        if grounded:
            self.bouncing = False
        elif (
            visible
            and old
            and self.previous_vy > 0
            and vy < -1
            and self.frames - self.recent_stomp_contact <= 2
        ):
            # The enemy disappears on contact, one observation before the
            # upward bounce is visible. Retain that observed contact briefly.
            self.bouncing = True
            if objective_kind == "stomp":
                self.stomp_complete = True
                self.stomp_target = None
        self.previous_vy = vy
        support = max(supports, key=lambda p: bool(p.get("moving"))) if grounded else None
        if support is not None and support.get("moving"):
            # The locomotion slot excludes passive platform carry, matching
            # its meaning in the collision provider and NES motion model.
            vx -= support["move_speed"] * support["move_dir"]
            velocity_known &= support.get("motion_age", 5) == 0
            if support.get("relative_vx") is not None:
                # Relative body/platform displacement directly measures active
                # locomotion, even when no shore can register camera motion.
                vx = -support["relative_vx"]
                velocity_known = support.get("relative_age", 5) == 0
        self.previous_grounded = grounded
        mario = dict(
            x=float(box.x),
            # Once contact is confirmed, align the body with its estimated
            # surface. Pixel fringes must not change a supported gap to finish.
            y=float(support["rect"].top - box.h if grounded else box.y),
            w=box.w,
            h=box.h,
            vx=vx,
            vy=vy,
            on_ground=grounded,
            facing=1 if vx >= 0 else -1,
            skidding=False,
            coyote_frames=0,
            jump_buffer=0,
            _platform=support,
        )
        enemies = [
            dict(
                x=b.x,
                y=b.y,
                w=b.w,
                h=b.h,
                dead=False,
                speed=0.0,
                direction=1,
                patrol_min=b.x,
                patrol_max=b.x,
            )
            for b in self.objects[5]
        ]
        for e in enemies:
            speed = velocities[5].get((e["x"], e["y"], e["w"], e["h"]), 0.0)
            e.update(speed=abs(speed), direction=1 if speed >= 0 else -1)
            key = (e["x"], e["y"], e["w"], e["h"])
            e["relative_vx"] = relative_velocities.get(key)
            e.update(estimates[5][key])

        coins = [dict(rect=b, collected=False) for b in component_boxes(labels == 3)]
        goals = component_boxes(labels == 4)
        scene = SimpleNamespace(
            mario=mario,
            platforms=platforms,
            enemies=enemies,
            coins=coins,
            goal=(
                max(goals, key=lambda b: b.w * b.h)
                if goals
                else pygame.Rect(0 if goal_direction < 0 else 240, 188, 16, 20)
            ),
            world_width=256,
            height=240,
            max_walk_speed=3.0,
            max_fall_speed=8.0,
            steps=self.frames,
            _terrain_left=goal_direction < 0,
            _goal_credited=False,
        )
        objective = (
            observable_objective(
                scene, objective_kind=None if self.stomp_complete else objective_kind
            )
            if visible
            else LocalObjective("finish", 240, 256, 208)
        )
        if objective_kind == "stomp" and not self.stomp_complete:
            if objective.kind == "stomp":
                target_enemy = enemies[objective.enemy_index]
                self.stomp_target = (
                    objective.left - box.x,
                    objective.right - box.x,
                    objective.top,
                    target_enemy.get("relative_vx"),
                    0,
                )
            elif self.stomp_target is not None and visible:
                left, right, top, relative_vx, age = self.stomp_target
                # Brief disappearance is not completion. Propagate relative
                # motion briefly; after that retain a recovery direction with
                # explicitly stale geometry instead of extrapolating forever.
                delta = relative_vx if relative_vx is not None and age < 4 else 0.0
                left, right = left + delta, right + delta
                self.stomp_target = (left, right, top, relative_vx, age + 1)
                objective = LocalObjective(
                    "stomp",
                    box.x + left,
                    box.x + right,
                    top,
                    direction=1 if (left + right) / 2 >= box.w / 2 else -1,
                )
        if (
            not grounded
            and self.target is not None
            and objective.kind != "stomp"
            and not (self.target[0] == "stomp" and self.stomp_complete)
        ):
            kind, left, right, top = self.target[:4]
            direction = self.target[4] if len(self.target) > 4 else goal_direction
            objective = LocalObjective(
                kind, left - self.scroll, right - self.scroll, top, direction=direction
            )
        elif grounded:
            self.target = (
                objective.kind,
                objective.left + self.scroll,
                objective.right + self.scroll,
                objective.top,
                objective.direction,
            )
        # Explicit goals are provided by task configuration, not hidden state.
        features = geometry_features(scene, terminated=terminated, truncated=truncated)
        features["motion_vec"][[1, 2, 6, 7]] = 0
        self.previous_center = box.centerx if visible else None
        self.previous = (box.x, box.y, visible)
        self.terrain = terrain.copy()
        self.dynamic = dynamic
        self.previous_bottom = box.bottom
        self.frames += 1
        return dict(
            scene=scene,
            features=features,
            objective=objective,
            skill_goal=objective_goal(objective, bouncing=self.bouncing),
            support="ground" if grounded else "air",
            enemy_contact=False,
            bouncing=self.bouncing,
            availability=[
                int(visible),
                int(visible and not fragmented),
                int(velocity_known),
                int(bool(enemies) and all(e.get("motion_age", 5) == 0 for e in enemies)),
                0,
                int(bool(moving) and all(p.get("motion_age", 5) == 0 for p in moving)),
                0,
                0,
            ],
            unavailable_features=[
                "enemy_vx",
                "enemy_patrol_bounds",
                "platform_vx",
                "platform_bounds",
                "power_state",
            ],
            unsupported_objects=["fragmented_terrain"] if fragmented else [],
            world_x=box.x + self.scroll,
            scroll=self.scroll,
            player_box=list(box),
            frame=self.frames - 1,
            observation_provider="perceived",
            motion_memory=motion_memory,
        )
