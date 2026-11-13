"""Block SMB as a practice world (framework/interfaces.Practice) and the
controller's state from its truth.

SimWorld plays one level (a scenario: levels.random_level) in the simulator
(env.MarioScenarioEnv); its state can be saved and put back (env_state).
SimSource gives the controller its state from the simulator as nes/sources'
MemorySource does from the real game's memory: what is on screen (question
blocks not yet hit as their own kind, mushrooms as items), the screen's
left edge a wall, and each enemy's and item's track: its places while it
has been in sight (kept by SimWorld in the thing's own record, so they are
saved and put back with the world's state).
"""

from copy import copy

from retroagi.domains.smb.controller import State
from retroagi.domains.smb.planner import Enemy, Solid, screen_edge
from retroagi.framework.interfaces import Step

from .env import MarioScenarioEnv
from .env_state import restore_env_state, snapshot_env_state

# The places of a thing kept as its track (frames).
TRACK_FRAMES = 32


def in_sight(env, thing: dict) -> bool:
    """Whether an enemy or item of the simulator is seen: on the screen,
    and for an enemy alive, awake and, a plant, out of its pipe."""
    if thing["x"] + thing["w"] <= env.camera_x or thing["x"] >= env.camera_x + env.width:
        return False
    if "dead" not in thing:
        return True  # (an item)
    awake = thing.get("awake", not thing.get("game_rules"))
    hidden = thing.get("kind") == "piranha_plant" and thing["exposed"] <= 0
    return not (thing["dead"] or thing["h"] <= 0 or not awake or hidden)


class SimWorld:
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.env = MarioScenarioEnv(world_width=scenario["world_width"])
        self.env.draws = False
        # Frames a play of the level may take.
        self.budget = scenario.get("frame_budget", 6000)
        # A skill scene is passed once Mario stands beyond this (None: a level, which is finished).
        self.passed_at = scenario.get("metadata", {}).get("scene", {}).get("passed")

    def reset(self) -> None:
        self.env.reset(scenario=self.scenario)
        self._track()

    def step(self, action: int) -> Step:
        _, _, terminated, truncated, info = self.env.step(action)
        self._track()
        if not (terminated or truncated):
            return Step()
        return Step(over=True, died=bool(info["death"]), finished=bool(self.env._goal_credited))

    def _track(self) -> None:
        """Each enemy's and item's places while it has been in sight, this
        frame's added (out of sight, its track starts again)."""
        env = self.env
        for thing in (*env.enemies, *env.items):
            seen = thing.get("track", ()) + ((thing["x"], thing["y"]),) if in_sight(env, thing) else ()
            thing["track"] = seen[-TRACK_FRAMES:]

    @property
    def progress(self) -> float:
        return self.env.mario["x"]

    @property
    def furthest(self) -> float:
        return self.env._max_x_reached

    @property
    def score(self) -> float:
        return self.env.score

    @property
    def settled(self) -> bool:
        return bool(self.env.mario["on_ground"])

    @property
    def passed(self) -> bool:
        """Whether this skill scene has been got through."""
        return self.passed_at is not None and self.progress >= self.passed_at and self.settled

    def snapshot(self):
        return snapshot_env_state(self.env)

    def restore(self, saved) -> None:
        restore_env_state(self.env, saved)

    def close(self) -> None:
        pass  # (the simulator's window library is shared by every one in the process)


_KINDS = {"ground": "ground", "pipe": "pipe", "brick": "block", "question_block": "block", "moving_platform": "moving",
          "cannon": "ground"}  # (the game's labels call a cannon ground)


def enemy_kind(enemy: dict) -> str:
    """The planner's kind of a simulator enemy (smb_planner.ENEMY_KINDS)."""
    if enemy.get("kind") == "piranha_plant":
        return "plant"
    if enemy.get("kind") == "bullet":
        return "flier"
    if enemy.get("kind") == "hammer":
        return "thrown"
    if enemy.get("red") and not enemy.get("shell"):
        return "pacer"
    return "hopper" if enemy.get("wings") else "walker"


def plant_velocity(plant: dict) -> float:
    """A plant's vertical speed by its cycle (pixels a frame; up negative)."""
    rise, exposed = plant["rise_frames"], plant["exposed_frames"]
    phase = plant["plant_tick"] % (2 * rise + exposed + plant["hidden_frames"])
    speed = plant["plant_height"] / rise
    return -speed if phase < rise else speed if rise + exposed <= phase < 2 * rise + exposed else 0.0
# A question block not yet hit (the game draws it with its question mark).
QUESTION = "question"


class SimSource:
    """The controller's state from the simulator: what is on the screen,
    in world pixels."""

    def reset(self) -> None:
        pass

    def acted(self, action: int) -> None:
        pass

    def read(self, world: SimWorld) -> State:
        env = world.env
        left, right = env.camera_x, env.camera_x + env.width
        solids = []
        for k, (p, kind) in enumerate(zip(env.platforms, env.platform_kinds)):
            r = p["rect"]
            if r.width == 0 or r.right <= left or r.left >= right:
                continue
            seen = QUESTION if p.get("contents") and not p.get("used") else _KINDS.get(kind, "block")
            solids.append(Solid(max(r.left, left), r.top, min(r.right, right), r.bottom, seen, ("platforms", k)))
        enemies = []
        for k, e in enumerate(env.enemies):
            if not in_sight(env, e):
                continue  # (a plant down in its pipe among them)
            box = (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
            kind = enemy_kind(e)
            if kind == "plant":
                vy = plant_velocity(e)
            else:
                # (standing, its fall speed is gravity's step the floor stopped)
                vy = 0.0 if e["on_ground"] else e["vy"]
            enemies.append(Enemy(box, e["speed"] * e["direction"], vy, ("enemies", k),
                                 stompable=bool(e.get("stompable", True)), kind=kind, track=e.get("track") or None,
                                 type=kind))
        items = [Enemy((i["x"], i["y"], i["x"] + i["w"], i["y"] + i["h"]), i["vx"], i["vy"], ("items", k),
                       stompable=False, track=i.get("track") or None, type="item")
                 for k, i in enumerate(env.items) if in_sight(env, i)]
        m = env.mario
        return State(
            body=(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]),
            motion=copy(env.motion),
            grounded=bool(m["on_ground"]),
            solids=solids + [screen_edge(left)],
            enemies=enemies,
            items=items,
        )
