"""A hand-written chooser that reads game memory, to test the controller
alone in the real game (docs/choose-and-control-plan.md phase 1; user,
2026-11-07). If a correct chooser and the controller cannot pass an
obstacle, nothing learned on top of them will.

TruthState builds the controller's state from game memory each frame:
Mario's body and motion (smb_physics.NESPlayerMotion.from_ram), the solids
from the exact labels (pixel_labels), the enemies from the game's slots with
velocities from their last move. ScriptedChooser decides at each of the
controller's events, looking ahead from Mario:
- an enemy ahead within ENEMY_AHEAD, or coming from behind within
  BEHIND: stomp it (max points) or pass it (speed run), passing when a
  stomp is infeasible, waiting a moment when neither is feasible;
- a raised thing ahead within RAISED_AHEAD (a pipe, a step): land on it;
- the floor ending within PIT_AHEAD at a pit: land beyond it;
- standing on something raised whose end is near: land on what follows;
- else run on toward the next of these, keeping speed into a jump.
A jump found infeasible is followed by a run closer to its target, or, at
a thing Mario touches, a step back (STEP_BACK) for room.
ScriptedPlayer plays it through the controller for the obstacle bench.
"""

from typing import Optional

from retroagi.core.smb_controller import Choice, Controller, State
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import Enemy, Goal, solids_of
from retroagi.core.smb_scene_labels import scene_from_labels

from .pixel_labels import PLAYER_FLOAT_STATE, Unexplained, label_frame
from .truth import true_frame

ENEMY_AHEAD = 112
# Enemies coming at Mario from behind this close are met too (pixels).
BEHIND = 40
# A step back from a thing too close to jump onto (pixels).
STEP_BACK = 24
RAISED_AHEAD = 96
PIT_AHEAD = 48
# How far before a target an infeasible jump's run stops (pixels).
CLOSER = 40
# The longest stretch a plain run covers before deciding again (pixels).
RUN_STRETCH = 96
# A raised thing: its top this far above Mario's feet or more.
RAISE = 4
# Enemies whose feet are this close to Mario's feet's height are on his way.
SAME_LEVEL = 24
# Labels are made again every this many frames (the solids are still).
LABEL_EVERY = 4


class TruthState:
    """The controller's state from game memory."""

    def __init__(self, level: str):
        self.level = level
        self.previous_state = None
        self.solids: list = []
        self.last_boxes: dict = {}
        self.frame = 0

    def reset(self):
        self.previous_state, self.solids, self.last_boxes, self.frame = None, [], {}, 0

    def read(self, game) -> State:
        ram = game.env.get_ram()
        truth = true_frame(ram, game.start)
        now = game.env.em.get_state()
        if self.previous_state is not None and (self.frame % LABEL_EVERY == 0 or not self.solids):
            try:
                labelled = label_frame(game.raw, self.previous_state, now, family=self.level)
                self.solids = solids_of(scene_from_labels(labelled.scene), truth["camera"])
            except Unexplained:
                pass
        self.previous_state = now
        self.frame += 1
        enemies = []
        for k, slot in enumerate(truth["slots"]):
            box = slot["box"]
            if not (slot["active"] and slot["enemy"] and box and not slot["defeated"] and box[2] > box[0]):
                self.last_boxes.pop(k, None)
                continue
            last = self.last_boxes.get(k)
            vx, vy = (box[0] - last[0], box[1] - last[1]) if last else (0.0, 0.0)
            self.last_boxes[k] = box
            enemies.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), slot["stompable"]))
        return State(
            body=tuple(float(v) for v in truth["mario"]),
            motion=NESPlayerMotion.from_ram(ram),
            grounded=int(ram[PLAYER_FLOAT_STATE]) == 0,
            solids=list(self.solids),
            enemies=enemies,
        )


def _middle(box) -> float:
    return (box[0] + box[2]) / 2


class ScriptedChooser:
    """Decides the next choice from the state (module notes)."""

    def __init__(self, mode: str):
        self.mode = mode

    def decide(self, state: State, last_event: Optional[str], last: Optional[Choice]) -> Choice:
        body = state.body
        mx, feet = _middle(body), body[3]
        if last_event == "infeasible" and last is not None and last.goal is not None:
            if last.goal.kind == "stomp":
                enemy = next((e for e in state.enemies if e.ref == last.goal.ref), None)
                if enemy is not None:
                    return Choice("jump", Goal("pass", enemy.ref, x1=enemy.box[2] + 16))
            if last.goal.kind == "pass":
                return Choice("wait", frames=8)
            target = last.goal.x0
            if target - mx > CLOSER:
                return Choice("run", x=target - CLOSER, keep_speed=True)
            return Choice("run", x=mx - STEP_BACK)
        ahead = [
            e for e in state.enemies
            if abs(e.box[3] - feet) < SAME_LEVEL
            and (
                0 < _middle(e.box) - mx < ENEMY_AHEAD
                or (-BEHIND < _middle(e.box) - mx <= 0 and e.vx > 0)
            )
        ]
        if ahead:
            enemy = min(ahead, key=lambda e: abs(_middle(e.box) - mx))
            if self.mode == "max_points" and enemy.stompable:
                return Choice("jump", Goal("stomp", enemy.ref))
            if _middle(enemy.box) < mx:  # from behind: over it backward
                return Choice("jump", Goal("pass", enemy.ref, x1=enemy.box[0] - 16, side=-1))
            return Choice("jump", Goal("pass", enemy.ref, x1=enemy.box[2] + 16))
        standing = [s for s in state.solids if s.x0 <= mx <= s.x1 and abs(s.top - feet) <= 2]
        under = standing[0] if standing else None
        raised = sorted(
            (s for s in state.solids
             if s.kind in ("pipe", "ground") and s.top < feet - RAISE and -4 < s.x0 - mx < RAISED_AHEAD),
            key=lambda s: s.x0,
        )
        if raised:
            s = raised[0]
            return Choice("jump", Goal("land", s.ref, s.x0, s.x1))
        if under is not None and under.x1 - mx < PIT_AHEAD:
            beyond = sorted((s for s in state.solids if s.x0 >= under.x1 - 2 and s.kind != "block"),
                            key=lambda s: s.x0)
            if beyond:
                s = beyond[0]
                prefer = "near" if s.top >= feet - RAISE else "middle"
                return Choice("jump", Goal("land", s.ref, s.x0, s.x1, prefer=prefer))
        targets = [s.x0 for s in state.solids if s.x0 > mx + 8 and s.kind in ("pipe", "ground") and s.top < feet - RAISE]
        if under is not None:
            targets.append(under.x1)
        target = min([t for t in targets if t > mx + 8] + [mx + RUN_STRETCH])
        return Choice("run", x=min(target - CLOSER, mx + RUN_STRETCH) if target - CLOSER > mx + 8 else mx + 16,
                      keep_speed=True)


class ScriptedPlayer:
    """The scripted chooser and the controller as an obstacle bench player."""

    def __init__(self, level: str = "Level1-1"):
        self.source = TruthState(level)
        self.controller = Controller()

    def reset(self, seed: int, mode: str) -> None:
        self.source.reset()
        self.chooser = ScriptedChooser(mode)
        self.controller = Controller()
        self.last_event, self.last_choice = None, None

    def act(self, screen, game) -> int:
        state = self.source.read(game)
        if not state.solids:
            return 0
        if self.controller.choice is None or self.controller.event is not None:
            self.last_event = self.controller.event
            choice = self.chooser.decide(state, self.last_event, self.last_choice)
            self.controller.start(choice, state)
            self.last_choice = choice
            if self.controller.event is not None:  # infeasible at once: decide again next frame
                return 0
        return self.controller.press(state)
