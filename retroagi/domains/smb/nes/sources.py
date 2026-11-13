"""The controller's state in the real game (framework/interfaces.
StateSource), from game memory or from vision.

MemorySource builds the state from game memory each frame: Mario's body and
motion (physics.NESPlayerMotion.from_ram), the solids from the exact labels
(pixel_labels), the enemies from the game's slots with velocities averaged
over their last SPEED_FRAMES frames (a Goomba walks half a pixel a frame,
as 0 and 1 by turns: one frame's move says 0 or 1). VisionSource builds it
from what the Full SMB vision shows in the screen and the motion estimator
estimates (perception/state.perceived), and from nothing else: the
difference between the two is what perception costs the controller. Both
give None while nothing solid is known yet.
"""

from typing import Optional

from retroagi.domains.smb.controller import State
from retroagi.domains.smb.perception.estimator import MotionEstimator
from retroagi.domains.smb.perception.scene_labels import scene_from_labels
from retroagi.domains.smb.perception.state import (
    HOP_SAMPLES,
    MEASURED,
    WALK,
    ItemTracker,
    hopper_speed,
    perceived,
    solids_of,
    thrown_box,
)
from retroagi.domains.smb.physics import NESPlayerMotion
from retroagi.domains.smb.planner import Enemy, floor_behind, screen_edge

from .pixel_labels import (
    ENEMY_KIND_CLASSES,
    ENEMY_KIND_TYPES,
    PLAYER_FLOAT_STATE,
    Unexplained,
    label_frame,
)
from .truth import hammers, true_frame

# The solids are labelled again once the camera has scrolled this far
# (pixels) or this many frames have passed (blocks bumped, broken): they
# stay still otherwise, and a label costs 17 ms.
LABEL_SCROLL = 8
LABEL_FRAMES = 16
# An enemy's velocity: its move over at most this many frames, averaged.
SPEED_FRAMES = 16
# A slot's enemy moving more than this in a frame is another enemy.
JUMP = 16
# The places of an enemy kept as its track (frames).
TRACK_FRAMES = 32


# The planner's kind of enemy (smb_planner.ENEMY_KINDS) by the game's enemy
# code (pixel_labels.ENEMY_KIND_TYPES): a Hammer Bro, a Piranha Plant, a
# Koopa Paratroopa, a Bullet Bill; any other walker walks, anything else
# flies. A Hammer Bro's hammers (loose items, not enemies) are thrown things.
SLOT_KINDS = {0x03: "pacer", 0x05: "walker", 0x0D: "plant", 0x0E: "hopper", 0x33: "flier"}


class MemorySource:
    """The controller's state from game memory."""

    def __init__(self, level: str):
        self.level = level
        self.previous_state = None
        self.solids: list = []
        self.labelled_at = (None, 0)  # (camera, frame) of the last label
        self.seen: dict = {}  # slot -> [(frame, x, y)], its recent places
        self.tracks: dict = {}  # slot -> [(x, y)], its places each frame since it was first seen
        self.frame = 0

    def reset(self):
        self.previous_state, self.solids, self.seen, self.tracks, self.frame = None, [], {}, {}, 0
        self.labelled_at = (None, 0)

    def acted(self, action: int) -> None:
        pass

    def read(self, world) -> Optional[State]:
        game = world.game
        ram = game.env.get_ram()
        truth = true_frame(ram, game.start)
        now = game.env.em.get_state()
        camera, then = self.labelled_at
        due = (
            not self.solids
            or camera is None
            or abs(truth["camera"] - camera) >= LABEL_SCROLL
            or self.frame - then >= LABEL_FRAMES
        )
        labelling = self.previous_state is not None and due
        if labelling:
            try:
                labelled = label_frame(game.raw, self.previous_state, now, family=self.level)
            except Unexplained:
                labelling = False
        self.previous_state = now
        self.frame += 1
        enemies, items = [], []
        for k, slot in enumerate(truth["slots"]):
            box = slot["box"]
            item = ENEMY_KIND_TYPES.get(slot["kind"]) == "power_up"
            if not (slot["active"] and (slot["enemy"] or item) and box and not slot["defeated"] and box[2] > box[0]):
                self.seen.pop(k, None)
                self.tracks.pop(k, None)
                continue
            places = self.seen.setdefault(k, [])
            track = self.tracks.setdefault(k, [])
            if places and abs(box[0] - places[-1][1]) > JUMP:
                places.clear()
                track.clear()
            places.append((self.frame, box[0], box[1]))
            del places[:-SPEED_FRAMES]
            track.append((float(box[0]), float(box[1])))
            del track[:-TRACK_FRAMES]
            seen = {"track": tuple(track), "type": slot["kind"]}
            first = places[0]
            frames = self.frame - first[0]
            if frames >= MEASURED:
                vx, vy = (box[0] - first[1]) / frames, (box[1] - first[2]) / frames
                if SLOT_KINDS.get(slot["kind"]) == "hopper":
                    ys = [y for _, _, y in places[-HOP_SAMPLES - 1:]]
                    vy = hopper_speed([b - a for a, b in zip(ys, ys[1:])])
            elif item:
                vx, vy = ItemTracker.MUSHROOM_SPEED, 0.0
            elif ENEMY_KIND_CLASSES.get(slot["kind"]) == "walker":
                vx, vy = (-WALK if box[0] > truth["mario"][0] else WALK), 0.0
            else:
                vx, vy = 0.0, 0.0  # a plant or anything else: still until measured
            if item:
                items.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), stompable=False, **seen))
                continue
            kind = SLOT_KINDS.get(slot["kind"], "walker" if ENEMY_KIND_CLASSES.get(slot["kind"]) == "walker" else "flier")
            if kind == "plant":
                # A plant is seen while any of it shows above its pipe: the
                # game draws it (its drawn box, None while it is down) and
                # its head is over the pipe's top, some 28 frames before
                # the box the game collides with, 14 below its head, clears
                # the pipe (shown only by that box, a plant coming up was
                # known 2 frames before it reached a Mario landing there;
                # vision sees its head). Its pipe not among the solids yet
                # (they are labelled every few frames), it is not seen.
                # Its track starts again when it shows.
                under = [s for s in self.solids if s.x0 <= (box[0] + box[2]) / 2 <= s.x1 and s.bottom > box[3]]
                drawn = slot["drawn"]
                if not under or drawn is None or all(s.top <= drawn[1] for s in under):
                    self.tracks.pop(k, None)
                    continue
            enemies.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), slot["stompable"], kind=kind,
                                 **seen))
        thrown = hammers(ram, truth["camera"])
        for slot in range(9):
            key = ("misc", slot)
            if slot not in thrown:
                self.seen.pop(key, None)
                self.tracks.pop(key, None)
                continue
            box = thrown_box(thrown[slot])
            places, track = self.seen.setdefault(key, []), self.tracks.setdefault(key, [])
            places.append((self.frame, box[0], box[1]))
            del places[:-SPEED_FRAMES]
            track.append((float(box[0]), float(box[1])))
            del track[:-TRACK_FRAMES]
            first, frames = places[0], self.frame - places[0][0]
            vx, vy = ((box[0] - first[1]) / frames, (box[1] - first[2]) / frames) if frames >= MEASURED else (0.0, 0.0)
            enemies.append(Enemy(tuple(float(v) for v in box), vx, vy, key, stompable=False, kind="thrown",
                                 track=tuple(track), type="hammer"))
        if labelling:
            # The enemies drawn in the labelled frame hide the floor under them.
            self.solids = floor_behind(solids_of(scene_from_labels(labelled.scene), truth["camera"]), enemies)
            self.labelled_at = (truth["camera"], self.frame - 1)
        if not self.solids:
            return None  # nothing labelled yet
        return State(
            body=tuple(float(v) for v in truth["mario"]),
            motion=NESPlayerMotion.from_ram(ram, top=truth["mario"][1]),
            grounded=int(ram[PLAYER_FLOAT_STATE]) == 0,
            solids=list(self.solids) + [screen_edge(truth["camera"])],
            enemies=enemies,
            items=items,
        )


class VisionSource:
    """The controller's state from the Full SMB vision and the motion
    estimator; the button pressed is told back (acted)."""

    def __init__(self, vision):
        self.vision = vision
        self.reset()

    def reset(self) -> None:
        self.feedback, self.scene, self.items = MotionEstimator(), None, ItemTracker()

    def read(self, world) -> Optional[State]:
        import torch

        with torch.no_grad():
            self.scene = self.vision.scene([world.screen])[0]
        self.feedback.observe(self.scene)
        camera = self.feedback.camera_position
        self.items.observe([(b[0] + camera, b[1], b[2] + camera, b[3]) for b in self.scene.power_ups])
        state = perceived(self.feedback, self.scene, self.items)
        if state is None or all(s.kind == "edge" for s in state.solids):
            return None  # Mario or anything solid not seen yet
        return state

    def acted(self, action: int) -> None:
        if self.scene is not None:
            self.feedback.executed(action, self.scene)
