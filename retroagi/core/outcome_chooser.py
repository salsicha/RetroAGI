"""The chooser (docs/choose-and-control-plan.md phase 4): the game's three
rules on the outcome network's predictions (user, 2026-11-07: "1. Don't
die, 2. Progress to the right, 3. get coins and power-ups", in that order).

At a decision the outcome network predicts every candidate choice's outcome
in one pass (smb_choices.candidates, outcome_network.predict). Then:
1. don't die: of the choices the executor can carry out, keep those whose
   chance of death is at most SAFE, or within SLACK of the safest one when
   none is that safe; and of those, the ones not likely to leave Mario
   stuck (from where no forward choice gets him on: a valley between
   staircases, entered for its quick progress, held him there), if any;
2. progress right: of those, the one bringing the most progress a frame
   (frames counted from MIN_FRAMES up, so a short wait is no free win);
3. coins and power-ups: in max-points mode a point is worth POINT_VALUE
   pixels a frame of progress, so a stomp wins when it costs little
   progress; in a speed run points are worth nothing.
Waiting and retreating come out of the rules: when every forward choice
risks death, a wait or a step back that does not is the one kept.

What was tried is remembered for a short while: a choice taken from a spot
(SPOT pixels) is not taken from it again until Mario reaches NEW_GROUND
pixels beyond his furthest so far (then all is forgotten), unless nothing
else is left; waits with enemies on screen may repeat. One choice at a time
the chooser had tried two failing landings by turns, and gone round a
staircase it could not climb: land on a step, back down, run into it, land
again.
"""

import numpy as np

from .outcome_network import PREDICTIONS, predict
from .smb_choices import candidates, state_tokens
from .smb_controller import Choice, State

SAFE = 0.05
SPOT = 8
NEW_GROUND = 16
SLACK = 0.02
MIN_FRAMES = 8
POINT_VALUE = {"speed_run": 0.0, "max_points": 0.2}
DEATH, INFEASIBLE, FINISHED, STUCK, PROGRESS, FRAMES, POINTS = range(len(PREDICTIONS))


class OutcomeChooser:
    def __init__(self, model, mode: str, device="cpu"):
        self.model, self.mode, self.device = model, mode, device
        self.last = None  # (options, predictions, picked index): for inspection
        self.tried: dict = {}  # spot -> the choices taken from it
        self.furthest = float("-inf")

    def decide(self, state: State, last_event=None, last=None) -> Choice:
        options = candidates(state)
        mario, surfaces, enemies, slots = state_tokens(state)
        rows = [(kind, slots.get(ref, -1) if ref is not None else -1, m) for _, kind, ref, m in options]
        p = predict(self.model, (mario, surfaces, enemies), rows, self.device)
        feasible = p[:, INFEASIBLE] < 0.5
        x = state.body[0]
        if x >= self.furthest + NEW_GROUND:
            self.tried.clear()
        self.furthest = max(self.furthest, x)
        spot = (round(x / SPOT), round(state.body[3] / SPOT))
        signatures = [self._signature(o, state) for o in options]
        fresh = np.array([sig not in self.tried.get(spot, ()) for sig in signatures])
        if (feasible & fresh).any():
            feasible &= fresh
        if not feasible.any():
            feasible = np.ones(len(p), bool)
        risk = np.where(feasible, p[:, DEATH], np.inf)
        allowed = risk <= max(SAFE, risk.min() + SLACK)
        if (allowed & (p[:, STUCK] < 0.5)).any():
            allowed &= p[:, STUCK] < 0.5
        value = (p[:, PROGRESS] / np.maximum(p[:, FRAMES], MIN_FRAMES)
                 + POINT_VALUE[self.mode] * p[:, POINTS] + p[:, FINISHED] * 100.0)
        best = int(np.argmax(np.where(allowed, value, -np.inf)))
        self.last = (options, p, best)
        if not (options[best][0].verb == "wait" and state.enemies):
            self.tried.setdefault(spot, set()).add(signatures[best])
        return options[best][0]

    @staticmethod
    def _signature(option, state) -> tuple:
        """What a choice is, the same from one decision to the next: its
        kind and modifier, and for a landing where it lands (its slot moves
        as the camera scrolls)."""
        choice, kind, _, modifier = option
        if choice.goal is not None and choice.goal.kind == "land":
            return kind, modifier, round(choice.goal.x0 / SPOT)
        return kind, modifier
