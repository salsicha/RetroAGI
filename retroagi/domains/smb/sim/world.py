"""Block SMB as a practice world (framework/interfaces.Practice) and the
controller's state from its truth.

SimWorld plays one level (a scenario: levels.random_level) in the simulator
(env.MarioScenarioEnv); its state can be saved and put back (env_state).
SimSource gives the controller its state from the simulator as nes/sources'
MemorySource does from the real game's memory: what is on screen (question
blocks not yet hit as their own kind, mushrooms as items), the screen's
left edge a wall.
"""

from copy import copy

from retroagi.domains.smb.controller import State
from retroagi.domains.smb.planner import Enemy, Solid, screen_edge
from retroagi.framework.interfaces import Step

from .env import MarioScenarioEnv
from .env_state import restore_env_state, snapshot_env_state


class SimWorld:
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.env = MarioScenarioEnv(world_width=scenario["world_width"])
        self.env.draws = False
        # Frames a play of the level may take.
        self.budget = scenario.get("frame_budget", 6000)

    def reset(self) -> None:
        self.env.reset(scenario=self.scenario)

    def step(self, action: int) -> Step:
        _, _, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            return Step()
        return Step(over=True, died=bool(info["death"]), finished=bool(self.env._goal_credited))

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
            awake = e.get("awake", not e.get("game_rules"))
            box = (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
            if e["dead"] or e["h"] <= 0 or not awake or box[2] <= left or box[0] >= right:
                continue
            kind = enemy_kind(e)
            if kind == "plant":
                if e["exposed"] <= 0:
                    continue  # down in its pipe: not seen
                vy = plant_velocity(e)
            else:
                # (standing, its fall speed is gravity's step the floor stopped)
                vy = 0.0 if e["on_ground"] else e["vy"]
            enemies.append(Enemy(box, e["speed"] * e["direction"], vy, ("enemies", k),
                                 stompable=bool(e.get("stompable", True)), kind=kind, type=kind))
        items = [Enemy((i["x"], i["y"], i["x"] + i["w"], i["y"] + i["h"]), i["vx"], i["vy"], ("items", k),
                       stompable=False, type="item")
                 for k, i in enumerate(env.items) if i["x"] + i["w"] > left and i["x"] < right]
        m = env.mario
        return State(
            body=(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]),
            motion=copy(env.motion),
            grounded=bool(m["on_ground"]),
            solids=solids + [screen_edge(left)],
            enemies=enemies,
            items=items,
        )
