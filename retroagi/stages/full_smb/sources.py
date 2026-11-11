"""The controller's state in the real game, from game memory or from
vision, and the player the obstacle bench runs.

TruthState builds the state from game memory each frame: Mario's body and
motion (smb_physics.NESPlayerMotion.from_ram), the solids from the exact
labels (pixel_labels), the enemies from the game's slots with velocities
averaged over their last SPEED_FRAMES frames (a Goomba walks half a pixel a
frame, as 0 and 1 by turns: one frame's move says 0 or 1). PerceivedState
builds it from what the Full SMB vision shows and the motion estimator
estimates (smb_perceived.perceived): the difference is what perception
costs the controller. ScriptedPlayer plays a chooser through the controller,
deciding only while Mario stands (a plan made in the air would start from
a standing jump).
"""

from typing import Optional

from retroagi.core.smb_controller import Controller, State
from retroagi.core.smb_perceived import (
    HOP_SAMPLES,
    MEASURED,
    WALK,
    ItemTracker,
    hopper_speed,
    perceived,
    solids_of,
)
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import Enemy, floor_behind, screen_edge
from retroagi.core.smb_scene_labels import scene_from_labels

from .pixel_labels import (
    ENEMY_KIND_CLASSES,
    ENEMY_KIND_TYPES,
    PLAYER_FLOAT_STATE,
    Unexplained,
    label_frame,
)
from .scripted_chooser import ScriptedChooser
from .truth import true_frame

# The solids are labelled again once the camera has scrolled this far
# (pixels) or this many frames have passed (blocks bumped, broken): they
# stay still otherwise, and a label costs 17 ms.
LABEL_SCROLL = 8
LABEL_FRAMES = 16
# An enemy's velocity: its move over at most this many frames, averaged.
SPEED_FRAMES = 16
# A slot's enemy moving more than this in a frame is another enemy.
JUMP = 16


# The planner's kind of enemy (smb_planner.ENEMY_KINDS) by the game's enemy
# code (pixel_labels.ENEMY_KIND_TYPES): a Piranha Plant, a Koopa Paratroopa,
# a Bullet Bill; any other walker walks, anything else flies.
SLOT_KINDS = {0x0D: "plant", 0x0E: "hopper", 0x33: "flier"}


class TruthState:
    """The controller's state from game memory."""

    def __init__(self, level: str):
        self.level = level
        self.previous_state = None
        self.solids: list = []
        self.labelled_at = (None, 0)  # (camera, frame) of the last label
        self.seen: dict = {}  # slot -> [(frame, x, y)], its recent places
        self.frame = 0

    def reset(self):
        self.previous_state, self.solids, self.seen, self.frame = None, [], {}, 0
        self.labelled_at = (None, 0)

    def read(self, game) -> State:
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
                continue
            places = self.seen.setdefault(k, [])
            if places and abs(box[0] - places[-1][1]) > JUMP:
                places.clear()
            places.append((self.frame, box[0], box[1]))
            del places[:-SPEED_FRAMES]
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
                items.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), stompable=False))
                continue
            kind = SLOT_KINDS.get(slot["kind"], "walker" if ENEMY_KIND_CLASSES.get(slot["kind"]) == "walker" else "flier")
            if kind == "plant" and any(s.x0 <= (box[0] + box[2]) / 2 <= s.x1 and s.top <= box[1] for s in self.solids):
                continue  # down in its pipe: not seen
            enemies.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), slot["stompable"], kind=kind))
        if labelling:
            # The enemies drawn in the labelled frame hide the floor under them.
            self.solids = floor_behind(solids_of(scene_from_labels(labelled.scene), truth["camera"]), enemies)
            self.labelled_at = (truth["camera"], self.frame - 1)
        return State(
            body=tuple(float(v) for v in truth["mario"]),
            motion=NESPlayerMotion.from_ram(ram, top=truth["mario"][1]),
            grounded=int(ram[PLAYER_FLOAT_STATE]) == 0,
            solids=list(self.solids) + [screen_edge(truth["camera"])],
            enemies=enemies,
            items=items,
        )


class PerceivedState:
    """The controller's state from the Full SMB vision and the executor's
    estimates; the button pressed is told back (pressed)."""

    def __init__(self, vision):
        self.vision = vision

    def reset(self):
        from retroagi.core.smb_estimator import MotionEstimator

        self.feedback, self.scene, self.items = MotionEstimator(), None, ItemTracker()

    def read(self, screen) -> Optional[State]:
        import torch

        with torch.no_grad():
            self.scene = self.vision.scene([screen])[0]
        self.feedback.observe(self.scene)
        camera = self.feedback.camera_position
        self.items.observe([(b[0] + camera, b[1], b[2] + camera, b[3]) for b in self.scene.power_ups])
        return perceived(self.feedback, self.scene, self.items)

    def pressed(self, button: int) -> None:
        if self.scene is not None:
            self.feedback.executed(button, self.scene)


class ScriptedPlayer:
    """A chooser (``make_chooser(mode)``: the scripted chooser by default,
    or core/outcome_chooser's) and the controller as an obstacle bench
    player, on game memory, or on perception with ``vision`` (a Full SMB
    vision model)."""

    def __init__(self, level: str = "Level1-1", vision=None, make_chooser=None):
        self.source = TruthState(level) if vision is None else PerceivedState(vision)
        self.controller = Controller()
        self.make_chooser = make_chooser or ScriptedChooser

    def reset(self, seed: int, mode: str) -> None:
        self.source.reset()
        self.chooser = self.make_chooser(mode)
        self.controller = Controller()
        self.last_event, self.last_choice = None, None

    def act(self, screen, game) -> int:
        if isinstance(self.source, TruthState):
            return self._act(self.source.read(game))
        button = self._act(self.source.read(screen))
        self.source.pressed(button)
        return button

    def _act(self, state: Optional[State]) -> int:
        if state is None or all(s.kind == "edge" for s in state.solids):
            return 0  # nothing labelled yet
        if (self.controller.choice is None or self.controller.event is not None) and not (
            state.grounded and state.ready
        ):
            return 0  # decisions are made standing, Mario's speed known
        if self.controller.choice is None or self.controller.event is not None:
            self.last_event = self.controller.event
            choice = self.chooser.decide(state, self.last_event, self.last_choice)
            self.controller.start(choice, state)
            self.last_choice = choice
            if self.controller.event is not None:  # infeasible at once: decide again next frame
                return 0
        return self.controller.press(state)
