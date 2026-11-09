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
  BEHIND, or meeting Mario soon at their closing speed (the controller's
  meets_soon, which also ends runs), with no raised thing or pit between: stomp
  it (max points) or pass it (speed run), passing when a stomp is
  infeasible, jumping back (when it meets him soon) to land
  on Mario's side of it when passing is infeasible too, running from it
  (RETREAT) when none is feasible and it meets him soon, else waiting a
  moment;
- a raised thing ahead within RAISED_AHEAD (a pipe, a step): land on it;
- the floor ending within PIT_AHEAD at a pit: land beyond it, across at
  the same height within ACROSS before down into what lies between (a
  floor followed at once by something lower is a step down, run off);
- standing on something raised whose end is near: land on what follows;
- else run on toward the next of these, keeping speed into a jump.
A jump found infeasible is followed by a run closer to its target, or, at
a thing Mario touches, a step back (STEP_BACK) for room. No run ends past
the end of the floor Mario stands on (EDGE before it).
ScriptedPlayer plays it through the controller for the obstacle bench,
deciding only while Mario stands (a plan made in the air would start from
a standing jump), on game memory (TruthState) or, with ``vision``, on what
the Full SMB vision shows and the executor estimates (smb_controller.
perceived): the difference is what perception costs the controller.
"""

from typing import Optional

from retroagi.core.smb_controller import (
    ARRIVE,
    MEASURED,
    WALK,
    Choice,
    Controller,
    State,
    meets_soon,
    on_the_way,
    ItemTracker,
    perceived,
)
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import BLOCK_KINDS, Enemy, Goal, Solid, floor_behind, screen_edge, solids_of
from retroagi.core.smb_scene_labels import scene_from_labels

from .pixel_labels import ENEMY_KIND_TYPES, PLAYER_FLOAT_STATE, Unexplained, label_frame
from .truth import true_frame

ENEMY_AHEAD = 112
# Enemies coming at Mario from behind this close are met too (pixels).
BEHIND = 40
# A step back from a thing too close to jump onto (pixels).
STEP_BACK = 24
# How far Mario runs from an enemy no jump gets past (pixels).
RETREAT = 64
RAISED_AHEAD = 96
PIT_AHEAD = 48
# Where the floor ends, a landing at about the same height this near is
# taken before dropping down (pixels).
ACROSS = 64
# How far before a target an infeasible jump's run stops (pixels).
CLOSER = 40
# A run on a floor ending at a pit stops with Mario's middle this far
# before the floor's end (pixels): his front foot stays on it.
EDGE = 8
# The longest stretch a plain run covers before deciding again (pixels).
RUN_STRETCH = 96
# A raised thing: its top this far above Mario's feet or more.
RAISE = 4
# A floor followed by something at most this much lower is a step down,
# walked off (pixels: one block).
STEP_DOWN = 16
# A renumbered enemy is looked for among those whose feet are this close to
# Mario's (_the_enemy; the enemies on his way: smb_controller.on_the_way).
SAME_LEVEL = 24
# The solids are labelled again once the camera has scrolled this far
# (pixels) or this many frames have passed (blocks bumped, broken): they
# stay still otherwise, and a label costs 17 ms.
LABEL_SCROLL = 8
LABEL_FRAMES = 16
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
            elif item:
                vx, vy = ItemTracker.MUSHROOM_SPEED, 0.0
            else:
                vx, vy = (-WALK if box[0] > truth["mario"][0] else WALK), 0.0
            if item:
                items.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), stompable=False))
                continue
            enemies.append(Enemy(tuple(float(v) for v in box), vx, vy, ("slots", k), slot["stompable"]))
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


def _middle(box) -> float:
    return (box[0] + box[2]) / 2


def _floor_run(state: State, first):
    """``first`` and the surfaces of its height side by side after it, as
    one floor (a staircase's top two columns wide: by its first column
    alone the floor went on, and Mario ran off the second into the pit)."""
    end = first.x1
    while True:
        more = next((s for s in state.solids if s is not first and abs(s.x0 - end) <= 2
                     and abs(s.top - first.top) <= 2 and s.x1 > end and s.kind not in BLOCK_KINDS), None)
        if more is None:
            break
        end = more.x1
    if end == first.x1:
        return first
    return Solid(first.x0, first.top, end, first.bottom, first.kind, first.ref)


def _steps_down(state: State, under) -> bool:
    """Whether something as high or at most STEP_DOWN lower carries on right
    after the end of ``under`` (the floor going on, a staircase down):
    walking off it is safe (a pipe's end, with the floor far below, is
    jumped down from, its landing checked against enemies; two steps of one
    height side by side were a pit to jump across, and the jump over the
    seam was infeasible)."""
    after = under.x1 + 2
    return any(s is not under and s.x0 <= after < s.x1 and 0 <= s.top - under.top <= STEP_DOWN
               and s.kind not in BLOCK_KINDS for s in state.solids)


def _steps_on(state: State, first) -> float:
    """Where the staircase down that starts with ``first`` ends: each next
    step adjacent and at most STEP_DOWN lower (``first``'s end if none)."""
    end, top = first.x1, first.top
    while True:
        step = next((s for s in state.solids if abs(s.x0 - end) <= 2 and 0 < s.top - top <= STEP_DOWN
                     and s.kind not in BLOCK_KINDS), None)
        if step is None:
            return end
        end, top = step.x1, step.top


def _the_enemy(state: State, ref, mx: float, feet: float):
    """The enemy ``ref``; if vision has given it a new identity since, the
    nearest at Mario's level (None if none: a stomp target lost after an
    infeasible plan had Mario step back into the next Goombas)."""
    enemy = next((e for e in state.enemies if e.ref == ref), None)
    if enemy is not None:
        return enemy
    level = [e for e in state.enemies if abs(e.box[3] - feet) < SAME_LEVEL]
    return min(level, key=lambda e: abs(_middle(e.box) - mx), default=None)


def _walled_off(state: State, mx: float, feet: float, enemy) -> bool:
    """Whether a raised thing (a pipe, a step) stands between Mario and
    ``enemy``: the thing comes first (and turns a walker back)."""
    x = _middle(enemy.box)
    lo, hi = min(mx, x), max(mx, x)
    return any(s.kind in ("pipe", "ground") and s.top < feet - RAISE and s.x0 < hi and lo < s.x1
               for s in state.solids)


class ScriptedChooser:
    """Decides the next choice from the state (module notes)."""

    def __init__(self, mode: str):
        self.mode = mode

    def decide(self, state: State, last_event: Optional[str], last: Optional[Choice]) -> Choice:
        body = state.body
        mx, feet = _middle(body), body[3]
        # What Mario stands on: under any part of him (the game stands him at
        # a floor's end with his middle over the pit), the one under his
        # middle first.
        standing = sorted(
            (s for s in state.solids if s.x0 < body[2] and body[0] < s.x1 and abs(s.top - feet) <= 2),
            key=lambda s: not (s.x0 <= mx <= s.x1),
        )
        under = _floor_run(state, standing[0]) if standing else None
        # Where the floor ends at a pit (not a step down onto something
        # lower right after it), no run ends beyond it (a run 40 pixels short
        # of the far side's landing ran into the pit).
        pit = under is not None and not _steps_down(state, under)
        edge = under.x1 - EDGE if pit else float("inf")
        infeasible = last_event == "infeasible" and last is not None and last.goal is not None
        if infeasible and last.goal.kind != "land":
            enemy = _the_enemy(state, last.goal.ref, mx, feet)
            if enemy is not None:
                beyond = 1 if _middle(enemy.box) >= mx else -1
                if last.goal.kind == "stomp":
                    return Choice("jump", Goal("pass", enemy.ref, side=beyond))
                if last.goal.side == beyond and meets_soon(state, enemy):
                    # Back: a jump landing on Mario's side of it, braking in
                    # the air (after a stomp at full speed the next Goombas
                    # came into view too close to jump); not for one still
                    # far off (Mario jumped back over 1-1's second pit).
                    return Choice("jump", Goal("pass", enemy.ref, side=-beyond))
                if meets_soon(state, enemy):
                    # No jump will do: run from it (waiting, Mario stood
                    # beside a Goomba until it walked into him).
                    return Choice("run", x=mx - beyond * RETREAT)
                return Choice("wait", frames=8)
        elif infeasible:
            closer = min(last.goal.x0 - CLOSER, edge)
            if closer - mx > ARRIVE:
                return Choice("run", x=closer, keep_speed=True)
            return Choice("run", x=mx - STEP_BACK)
        ahead = [
            e for e in state.enemies
            if on_the_way(state, e)
            and (
                0 < _middle(e.box) - mx < ENEMY_AHEAD
                or (-BEHIND < _middle(e.box) - mx <= 0 and e.vx > 0)
                or meets_soon(state, e)
            )
            and not _walled_off(state, mx, feet, e)
            # one beyond the pit the floor ends at waits for the pit (Mario
            # at full speed tried to pass a Goomba across a pit, and ran in)
            and not (pit and mx < under.x1 < _middle(e.box))
        ]
        if ahead:
            enemy = min(ahead, key=lambda e: abs(_middle(e.box) - mx))
            if self.mode == "max_points" and enemy.stompable:
                return Choice("jump", Goal("stomp", enemy.ref))
            # Beyond it: ahead of Mario over it forward, from behind backward.
            return Choice("jump", Goal("pass", enemy.ref, side=1 if _middle(enemy.box) >= mx else -1))
        raised = sorted(
            (s for s in state.solids
             if s.kind in ("pipe", "ground") and s.top < feet - RAISE and -4 < s.x0 - mx < RAISED_AHEAD),
            key=lambda s: s.x0,
        )
        if raised:
            s = raised[0]
            return Choice("jump", Goal("land", s.ref, s.x0, s.x1, top=s.top))
        if pit and under.x1 - mx < PIT_AHEAD:
            beyond = sorted((s for s in state.solids if s.x0 >= under.x1 - 2 and s.kind not in BLOCK_KINDS),
                            key=lambda s: s.x0)
            # Across at the same height (from one staircase to the next)
            # before down into what lies between.
            level = [s for s in beyond if s.top <= feet + RAISE and s.x0 - under.x1 <= ACROSS]
            if beyond:
                s = level[0] if level else beyond[0]
                prefer = "near" if s.top >= feet - RAISE else "middle"
                end = _steps_on(state, s)
                if end > s.x1:
                    # A staircase down: any of its steps will do (at full
                    # speed its first, 16 pixels wide, was out of reach).
                    return Choice("jump", Goal("land", s.ref, s.x0, end, prefer="near"))
                return Choice("jump", Goal("land", s.ref, s.x0, s.x1, prefer=prefer, top=s.top))
        targets = [s.x0 for s in state.solids if s.x0 > mx + 8 and s.kind in ("pipe", "ground") and s.top < feet - RAISE]
        if under is not None:
            targets.append(under.x1)
        target = min([t for t in targets if t > mx + 8] + [mx + RUN_STRETCH])
        x = min(target - CLOSER, mx + RUN_STRETCH) if target - CLOSER > mx + 8 else mx + 16
        if min(x, edge) - mx <= ARRIVE:
            return Choice("wait", frames=8)  # at a floor's end, its far side not yet seen
        return Choice("run", x=min(x, edge), keep_speed=True)


class PerceivedState:
    """The controller's state from the Full SMB vision and the executor's
    estimates; the button pressed is told back (pressed)."""

    def __init__(self, vision):
        self.vision = vision

    def reset(self):
        from retroagi.core.smb_spatial_feedback import SpatialFeedback

        self.feedback, self.scene, self.items = SpatialFeedback(), None, ItemTracker()

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
