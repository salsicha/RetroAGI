"""The chooser (docs/choose-and-control-plan.md phase 4): the game's three
rules on the outcome network's predictions (user, 2026-11-07: "1. Don't
die, 2. Progress to the right, 3. get coins and power-ups", in that order).

At a decision the outcome network predicts every candidate choice's outcome
in one pass (smb_choices.candidates, outcome_network.predict). Then:
1. don't die: of the choices the executor can carry out, keep those whose
   chance of death is at most SAFE, or within SLACK of the safest one when
   none is that safe;
2. progress right: of those, the one bringing the most progress a frame
   (frames counted from MIN_FRAMES up, so a short wait is no free win);
3. coins and power-ups: in max-points mode a point is worth POINT_VALUE
   pixels a frame of progress, so a stomp wins when it costs little
   progress; in a speed run points are worth nothing.
Waiting and retreating come out of the rules: when every forward choice
risks death, a wait or a step back that does not is the one kept. A choice
the executor has just ended at once (blocked, infeasible) is not taken again
from the same place.
"""

import numpy as np

from .outcome_network import PREDICTIONS, predict
from .smb_choices import candidates, state_tokens
from .smb_controller import Choice, State

SAFE = 0.05
SLACK = 0.02
MIN_FRAMES = 8
POINT_VALUE = {"speed_run": 0.0, "max_points": 0.2}
DEATH, INFEASIBLE, FINISHED, PROGRESS, FRAMES, POINTS = range(len(PREDICTIONS))


class OutcomeChooser:
    def __init__(self, model, mode: str, device="cpu"):
        self.model, self.mode, self.device = model, mode, device
        self.last = None  # (options, predictions, picked index): for inspection

    def decide(self, state: State, last_event=None, last=None) -> Choice:
        options = candidates(state)
        mario, surfaces, enemies, slots = state_tokens(state)
        rows = [(kind, slots.get(ref, -1) if ref is not None else -1, m) for _, kind, ref, m in options]
        p = predict(self.model, (mario, surfaces, enemies), rows, self.device)
        feasible = p[:, INFEASIBLE] < 0.5
        if last_event in ("blocked", "infeasible"):
            # The choice that just ended at once is not taken again from the
            # same place (a blocked run was taken 6,000 times at a wall).
            feasible &= np.array([choice != last for choice, _, _, _ in options])
        if not feasible.any():
            feasible = np.ones(len(p), bool)
        risk = np.where(feasible, p[:, DEATH], np.inf)
        allowed = risk <= max(SAFE, risk.min() + SLACK)
        value = (p[:, PROGRESS] / np.maximum(p[:, FRAMES], MIN_FRAMES)
                 + POINT_VALUE[self.mode] * p[:, POINTS] + p[:, FINISHED] * 100.0)
        best = int(np.argmax(np.where(allowed, value, -np.inf)))
        self.last = (options, p, best)
        return options[best][0]
