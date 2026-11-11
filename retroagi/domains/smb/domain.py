"""Super Mario Bros as a domain (framework/interfaces.Domain): what the
framework's tools (sampling, training, practice, bench, play) need to know
of the game, and nothing they do not.

- Practice worlds: random levels of the real game's pieces in the Block SMB
  simulator (sim/levels.py, sim/world.py), the state read from its truth.
- The real game: a level in the emulator (nes/world.py), the state from the
  game's memory ("memory") or from vision alone ("vision").
- The controller (controller.py), the choices (choices.py), the
  hand-written chooser as the behaviour that visits the states outcomes are
  sampled at (scripted_chooser.py), and the two ways to play: a speed run,
  where points are worth nothing, and max points.
"""

from copy import copy
from typing import Optional

from retroagi.framework.chooser import OutcomeChooser
from retroagi.framework.observed_dynamics import ObservedDynamics

from .choices import KINDS, RUN_BACK, RUN_LONG, SPACE
from .controller import Choice, Controller
from .planner import ENEMY_FALL, ENEMY_GRAVITY, FEET_BELOW, RULES, WALL_DEPTH, Goal
from .scripted_chooser import ScriptedChooser

# What a point is worth, in pixels of progress a frame, in each way to play.
POINT_VALUE = {"speed_run": 0.0, "max_points": 0.2}
# In sampling, how often the player bumps a question block in reach, or
# collects an item on screen (so that states with a mushroom out, and big
# Mario, are common enough to learn from).
TRY_POWER_UPS = 0.5
# After a choice, how many of the surfaces ahead are probed for a way on,
# and how far on a probe must take Mario (pixels).
PROBE_SURFACES = 3
PROBE_PROGRESS = 16
DEFAULT_VISION = "data/full_vit/full_vit_scene.pth"


class SuperMarioBros:
    name = "smb"
    modes = ("speed_run", "max_points")
    idle = 0
    # How a sampled choice can end: the controller's events, and the
    # sampler's own (the order is the datasets').
    ends = ("landed", "stomped", "arrived", "waited", "threat", "blocked", "infeasible", "timeout", "died",
            "finished", "edge")
    space = SPACE
    sources = ("memory", "vision")
    # The parts with alternatives (--use PART=NAME). The controller's model
    # of how enemies and items move: "observed", fitted to what is seen of
    # each thing as it plays (framework/observed_dynamics; a new one each
    # play, with the game's walkers' fall off a ledge for what it has not
    # yet seen), or "rules", written by hand for each kind of enemy.
    # The chooser on the outcome network's predictions: "wary" of choices
    # that leave Mario in danger, or "plain" (framework/chooser).
    parts = {
        "dynamics": {
            "rules": lambda: RULES,
            "observed": lambda: ObservedDynamics(gravity=ENEMY_GRAVITY, fastest=ENEMY_FALL, reach=FEET_BELOW + 2,
                                                 wall_depth=WALL_DEPTH, ignored=("edge",)),
        },
        "chooser": {"wary": True, "plain": False},
    }
    chosen = {"dynamics": "observed", "chooser": "wary"}

    def using(self, **choices) -> "SuperMarioBros":
        for part, name in choices.items():
            if name not in self.parts.get(part, ()):
                raise ValueError(f"no {part} called {name}: {sorted(self.parts.get(part, ())) or 'no such part'}")
        other = copy(self)
        other.chosen = {**self.chosen, **choices}
        return other

    def dynamics(self):
        """A new dynamics model of the chosen kind."""
        return self.parts["dynamics"][self.chosen["dynamics"]]()

    def controller(self) -> Controller:
        return Controller(self.dynamics())

    def chooser(self, predictor, mode: str) -> OutcomeChooser:
        return OutcomeChooser(predictor, self.space, POINT_VALUE[mode],
                              wary=self.parts["chooser"][self.chosen["chooser"]])

    def behaviour(self, mode: str) -> ScriptedChooser:
        return ScriptedChooser(mode)

    # ── practice (the simulator) ─────────────────────────────────────────────

    def practice(self, seed: int):
        from .sim.levels import random_level
        from .sim.world import SimWorld

        return SimWorld(random_level(seed))

    def truth_source(self):
        from .sim.world import SimSource

        return SimSource()

    def explore(self, state, candidates, rng) -> Optional[Choice]:
        """Collect an item on screen, or bump a question block in reach,
        TRY_POWER_UPS of the time each."""
        questions = {s.ref for s in state.solids if s.kind == "question"}
        bumps = [c for c in candidates if c.kind == KINDS.index("bump") and c.choice.goal.ref in questions]
        items = [c for c in candidates if c.kind == KINDS.index("collect")]
        if items and rng.random() < TRY_POWER_UPS:
            return rng.choice(items).choice
        if bumps and rng.random() < TRY_POWER_UPS:
            return rng.choice(bumps).choice
        return None

    def probes(self, state) -> list:
        """The forward choices tried after a choice to see whether Mario can
        go on: a long run, landing on each of the PROBE_SURFACES surfaces
        ahead (the part ahead of him of the one he is on), passing and
        stomping the nearest enemy ahead."""
        body = state.body
        mx = (body[0] + body[2]) / 2
        out = [Choice("run", x=mx + RUN_LONG, keep_speed=True)]
        ahead = sorted((s for s in state.solids if s.kind in ("ground", "pipe", "block", "question", "moving")
                        and s.x1 > mx + PROBE_PROGRESS + 8), key=lambda s: max(s.x0, mx))
        for s in ahead[:PROBE_SURFACES]:
            x0 = max(s.x0, mx + PROBE_PROGRESS)
            out.append(Choice("jump", Goal("land", s.ref, x0, s.x1, top=s.top)))
        coming = [e for e in state.enemies if (e.box[0] + e.box[2]) / 2 > mx]
        if coming:
            e = min(coming, key=lambda e: e.box[0])
            out += [Choice("jump", Goal("pass", e.ref, side=1)), Choice("jump", Goal("stomp", e.ref))]
        return out

    def escapes(self, state) -> list:
        """A brake back: what might save Mario where standing would not."""
        mx = (state.body[0] + state.body[2]) / 2
        return [Choice("run", x=mx - RUN_BACK)]

    # ── the real game (the emulator) ─────────────────────────────────────────

    def real_world(self, level: str, start_state: Optional[bytes] = None):
        from .nes.world import NESWorld

        return NESWorld(level, start_state)

    def source(self, kind: str, level: str, vision: str = DEFAULT_VISION):
        """The state source ``kind`` for the real game: "memory" (the game's
        own memory) or "vision" (the Full SMB vision saved at ``vision``,
        and nothing else)."""
        from .nes.sources import MemorySource, VisionSource

        if kind == "memory":
            return MemorySource(level)
        if kind == "vision":
            from .nes.vision import load_full_vit_checkpoint

            return VisionSource(load_full_vit_checkpoint(vision, device="cuda").model.eval())
        raise ValueError(f"no state source {kind!r}: one of {self.sources}")


DOMAIN = SuperMarioBros()
