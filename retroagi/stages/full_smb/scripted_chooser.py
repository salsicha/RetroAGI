"""A hand-written chooser that reads game memory, to test the controller
alone in the real game (docs/choose-and-control-plan.md phase 1; user,
2026-11-07). If a correct chooser and the controller cannot pass an
obstacle, nothing learned on top of them will.

TruthState builds the controller's state from game memory each frame:
Mario's body and motion (smb_physics.NESPlayerMotion.from_ram), the solids
from the exact labels (pixel_labels), the enemies from the game's slots with
velocities averaged over their last SPEED_FRAMES frames (a Goomba walks half
a pixel a frame, as 0 and 1 by turns: one frame's move says 0 or 1). ScriptedChooser decides at each of the
controller's events, looking ahead from Mario:
- an enemy ahead within ENEMY_AHEAD, or coming from behind within
  BEHIND: stomp it (max points) or pass it (speed run), passing when a
  stomp is infeasible, waiting a moment when neither is feasible;
- a raised thing ahead within RAISED_AHEAD (a pipe, a step): land on it;
- the floor ending within PIT_AHEAD at a pit: land beyond it, across at
  the same height within ACROSS before down into what lies between;
- standing on something raised whose end is near: land on what follows;
- else run on toward the next of these, keeping speed into a jump.
A jump found infeasible is followed by a run closer to its target, or, at
a thing Mario touches, a step back (STEP_BACK) for room.
ScriptedPlayer plays it through the controller for the obstacle bench,
deciding only while Mario stands (a plan made in the air would start from
a standing jump), on game memory (TruthState) or, with ``vision``, on what
the Full SMB vision shows and the executor estimates (smb_controller.
perceived): the difference is what perception costs the controller.
"""

from typing import Optional

from retroagi.core.smb_controller import MEASURED, WALK, Choice, Controller, State, perceived
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
# Where the floor ends, a landing at about the same height this near is
# taken before dropping down (pixels).
ACROSS = 64
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
# An enemy's velocity: its move over at most this many frames, averaged.
SPEED_FRAMES = 16
# A slot's enemy moving more than this in a frame is another enemy.
JUMP = 16


class TruthState:
    """The controller's state from game memory."""

    def __init__(self, level: str):
        self.level = level
        self.previous_state = None
        self.solids: list = []
        self.seen: dict = {}  # slot -> [(frame, x, y)], its recent places
        self.frame = 0

    def reset(self):
        self.previous_state, self.solids, self.seen, self.frame = None, [], {}, 0

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
            else:
                vx, vy = (-WALK if box[0] > truth["mario"][0] else WALK), 0.0
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
                    side = 1 if _middle(enemy.box) >= mx else -1
                    return Choice("jump", Goal("pass", enemy.ref, side=side))
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
            # Beyond it: ahead of Mario over it forward, from behind backward.
            return Choice("jump", Goal("pass", enemy.ref, side=1 if _middle(enemy.box) >= mx else -1))
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
            # Across at the same height (from one staircase to the next)
            # before down into what lies between.
            level = [s for s in beyond if s.top <= feet + RAISE and s.x0 - under.x1 <= ACROSS]
            if beyond:
                s = level[0] if level else beyond[0]
                prefer = "near" if s.top >= feet - RAISE else "middle"
                return Choice("jump", Goal("land", s.ref, s.x0, s.x1, prefer=prefer))
        targets = [s.x0 for s in state.solids if s.x0 > mx + 8 and s.kind in ("pipe", "ground") and s.top < feet - RAISE]
        if under is not None:
            targets.append(under.x1)
        target = min([t for t in targets if t > mx + 8] + [mx + RUN_STRETCH])
        return Choice("run", x=min(target - CLOSER, mx + RUN_STRETCH) if target - CLOSER > mx + 8 else mx + 16,
                      keep_speed=True)


class PerceivedState:
    """The controller's state from the Full SMB vision and the executor's
    estimates; the button pressed is told back (pressed)."""

    def __init__(self, vision):
        self.vision = vision

    def reset(self):
        from retroagi.core.smb_spatial_feedback import SpatialFeedback

        self.feedback, self.scene = SpatialFeedback(), None

    def read(self, screen) -> Optional[State]:
        import torch

        with torch.no_grad():
            self.scene = self.vision.scene([screen])[0]
        self.feedback.observe(self.scene)
        return perceived(self.feedback, self.scene)

    def pressed(self, button: int) -> None:
        if self.scene is not None:
            self.feedback.executed(button, self.scene)


class ScriptedPlayer:
    """The scripted chooser and the controller as an obstacle bench player,
    on game memory, or on perception with ``vision`` (a Full SMB vision
    model)."""

    def __init__(self, level: str = "Level1-1", vision=None):
        self.source = TruthState(level) if vision is None else PerceivedState(vision)
        self.controller = Controller()

    def reset(self, seed: int, mode: str) -> None:
        self.source.reset()
        self.chooser = ScriptedChooser(mode)
        self.controller = Controller()
        self.last_event, self.last_choice = None, None

    def act(self, screen, game) -> int:
        if isinstance(self.source, TruthState):
            return self._act(self.source.read(game))
        button = self._act(self.source.read(screen))
        self.source.pressed(button)
        return button

    def _act(self, state: Optional[State]) -> int:
        if state is None or not state.solids:
            return 0
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
