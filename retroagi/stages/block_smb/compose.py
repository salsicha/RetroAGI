"""Block SMB layouts built from sections, each with its own tactics.

A section is a stretch of world with its own floor, obstacles and tactic
segments (tactic_schedule). ``compose`` lays sections left to right, numbers
their platforms and enemies, and joins their segments into the layout's
schedule: between special sections Mario advances, and each special section
adds its own tactics:

- plant: advance, holding the area while the plant is out (kind "plant");
- bridge: advance, holding the area while waiting for or riding the moving
  platform (kind "bridge");
- upper_route: the floor is cut by a pit too wide to jump (or a wall too tall
  to climb) and raised platforms cross it: alternate route over them;
- lower_route: a raised ledge is cut by an opening too wide to jump, over a
  lower floor: alternate route down, along and back up;
- dead_end: a corridor ends at a wall too tall to climb; a step behind leads
  up to a high path over it: retreat to the step, then alternate route;
- monster: a monster comes out of a tunnel too low to jump in: keep behind
  the tunnel's mouth (retreat, hold area) and jump over it (kind "monster").

Simple sections (flat, enemy, gap, pipe, stairs) are advanced through. The
standalone families start Mario inside a dead end or a tunnel.

Every layout is checked by the teacher (route_actions) in the sampler, which
redraws layouts the teacher cannot finish.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .tactic_schedule import segment

FLOOR = 220
DIFFICULTY = ("easy", "medium", "hard")


@dataclass
class Section:
    """One stretch of world, in world coordinates. Segment platform and enemy
    indices are the section's own (compose renumbers them)."""

    width: int
    platforms: list = field(default_factory=list)
    kinds: list = field(default_factory=list)  # one per platform: a drawn kind or None
    enemies: list = field(default_factory=list)
    coins: list = field(default_factory=list)
    segments: list = field(default_factory=list)  # its own tactic segments, if special
    entry: Optional[int] = None  # where its first segment begins (Mario's left edge)
    spawn: Optional[int] = None  # x to start Mario at, for a standalone family
    flags: dict = field(default_factory=dict)
    waits: int = 0  # frames the teacher may spend waiting here (frame budget)

    def add(self, platform, kind=None) -> int:
        self.platforms.append(platform)
        self.kinds.append(kind)
        return len(self.platforms) - 1


def _tier(difficulty: str) -> int:
    return DIFFICULTY.index(difficulty)


def flat(rng, difficulty, x, width=None) -> Section:
    width = width or rng.randint(48, 80)
    s = Section(width)
    s.add([x, FLOOR, width, 20])
    return s


def enemy(rng, difficulty, x) -> Section:
    width = rng.randint(128, 160)
    s = Section(width)
    s.add([x, FLOOR, width, 20])
    ex = x + rng.randint(64, 88)
    patrol = (0, 8, 16)[_tier(difficulty)]
    speed = round((0.0, 0.3, 0.5)[_tier(difficulty)] + rng.uniform(0, 0.1), 3)
    s.enemies.append([ex, 206, ex - patrol, ex + patrol, speed, rng.choice((-1, 1))])
    s.coins.append([ex, 170, 10, 10])
    return s


def gap(rng, difficulty, x) -> Section:
    width = (34, 42, 50)[_tier(difficulty)] + rng.randint(-2, 4)
    lead, landing = 56, 64
    s = Section(lead + width + landing)
    s.add([x, FLOOR, lead, 20])
    s.add([x + lead + width, FLOOR, landing, 20])
    return s


def pipe(rng, difficulty, x) -> Section:
    height = (34, 44, 54)[_tier(difficulty)] + rng.randint(-2, 4)
    lead, tail = 64, 56
    s = Section(lead + 32 + tail)
    s.add([x, FLOOR, s.width, 20])
    s.add([x + lead, FLOOR - height, 32, height], "pipe")
    return s


def stairs(rng, difficulty, x) -> Section:
    rise = (20, 24, 28)[_tier(difficulty)]
    step = rng.randint(24, 32)
    count = rng.choice((2, 3))
    lead, top, tail = 40, 32, 64
    s = Section(lead + count * step + top + tail)
    s.add([x, FLOOR, s.width, 20])
    for i in range(count):
        height = rise * (i + 1)
        width = step if i < count - 1 else step + top
        s.add([x + lead + i * step, FLOOR - height, width, height])
    return s


def plant(rng, difficulty, x, timed: Optional[bool] = None) -> Section:
    """A pipe with a piranha plant: a short plant can always be cleared; a tall
    timed one must be waited for until it goes back into its pipe."""
    timed = bool(rng.randrange(2)) if timed is None else timed
    height = (24, 30, 34)[_tier(difficulty)] + rng.randint(-2, 2)
    width = (32, 40, 48)[_tier(difficulty)] + (4 if timed else 0)
    lead, tail = 72, 64
    s = Section(lead + width + tail, entry=x)
    s.add([x, FLOOR, s.width, 20])
    s.add([x + lead, FLOOR - height, width, height], "pipe")
    rise = rng.randint(12, 20)
    plant_spec = {
        "kind": "piranha_plant",
        "x": x + lead + (width - 12) // 2,
        "pipe_top": FLOOR - height,
        "plant_height": 80 if timed else (16, 20, 20)[_tier(difficulty)],
        "rise_frames": rise,
        "exposed_frames": rng.randint(40, 64),
        "hidden_frames": rng.randint(24, 40) + (24 if timed else 0),
        "timed_crossing": timed,
    }
    period = 2 * rise + plant_spec["exposed_frames"] + plant_spec["hidden_frames"]
    plant_spec["phase"] = rng.randrange(period)
    s.enemies.append(plant_spec)
    s.segments = [segment("advance", 1, kind="plant", past_enemy=0)]
    s.waits = period
    return s


def bridge(rng, difficulty, x) -> Section:
    """A moving platform across a pit: wait for it, ride it, step off."""
    speed = round((2.2, 2.0, 1.8)[_tier(difficulty)] + rng.uniform(-0.2, 0.2), 3)
    # A shore long enough for a running Mario to slow to walking pace.
    shore, pit, far = 150, 200, 96
    s = Section(shore + pit + far, entry=x)
    s.add([x, FLOOR, shore, 20])
    low, high = x + shore - 10, x + shore + 160
    s.add(
        {
            "x": rng.randint(low, high),
            "y": FLOOR,
            "w": 100,
            "h": 20,
            "moving": [low, high, speed],
            "direction": rng.choice((-1, 1)),
        }
    )
    s.add([x + shore + pit, FLOOR, far, 20])
    s.segments = [segment("advance", 1, kind="bridge", bridge_crossed=True)]
    s.flags = {"require_bridge_before_goal": True, "bridge_then_terrain": True}
    s.waits = 240
    return s


def upper_route(rng, difficulty, x) -> Section:
    """Raised platforms over a pit too wide to jump, or a wall too tall to climb."""
    lead, tail = 96, 80
    if rng.random() < 0.5:
        pit = (172, 184, 196)[_tier(difficulty)] + rng.randint(0, 8)
        a, b = x + lead, x + lead + pit
        s = Section(lead + pit + tail, entry=x)
        s.add([x, FLOOR, lead, 20])
        s.add([b, FLOOR, tail, 20])
        route = [s.add([a - 40, 184, 40, 10])]
        right, high = a, True
        while right < b - 48:
            space = rng.randint(20, (28, 32, 36)[_tier(difficulty)])
            route.append(s.add([right + space, 160 if high else 168, 36, 10]))
            right, high = right + space + 36, not high
        route.append(s.add([max(right + 20, b - 12), 184, 40, 10]))
        s.segments = [segment("alternate_route", 1, route=route, reach_x=b)]
        return s
    wall = (100, 106, 112)[_tier(difficulty)] + rng.randint(0, 6)
    a = x + lead + 40
    s = Section(lead + 40 + 32 + tail, entry=x)
    s.add([x, FLOOR, s.width, 20])
    first = s.add([a - 104, 186, 36, 10])
    second = s.add([a - 44, 152, 40, 10])
    top = s.add([a, FLOOR - wall, 32, wall])
    s.segments = [segment("alternate_route", 1, route=[first, second, top], on=top)]
    return s


def lower_route(rng, difficulty, x) -> Section:
    """A raised ledge cut by an opening too wide to jump, over a lower floor."""
    opening = (176, 188, 200)[_tier(difficulty)] + rng.randint(0, 8)
    lead, ledge, tail = 32, 64, 64
    s = Section(lead + 24 + ledge + opening + ledge + tail, entry=x)
    s.add([x, FLOOR, lead, 20])
    step = s.add([x + lead, 190, 24, 30])
    near = s.add([x + lead + 24, 160, ledge, 60])
    o = x + lead + 24 + ledge
    low = s.add([o, FLOOR, opening, 20])
    back = s.add([o + opening - 28, 190, 28, 30])
    far = s.add([o + opening, 160, ledge, 60])
    s.add([o + opening + ledge, FLOOR, tail, 20])
    s.segments = [
        segment("advance", 1, route=[step, near], on=near),
        segment("alternate_route", 1, route=[low, back, far], on=far),
    ]
    return s


def dead_end(rng, difficulty, x, inside: bool = False) -> Section:
    """A corridor that ends at a wall too tall to climb. A step at its start
    leads up to a high path over the wall. Entered from the left, the wall
    comes into view only once Mario is in the corridor."""
    # Entered from the left, the corridor is short enough that the step stays
    # on screen (the screen never scrolls back) while Mario backs out to it.
    corridor = rng.randint(120, 150) if inside else rng.randint(176, 196)
    lead, tail = 24, 96
    s_left = x + lead
    s = Section(lead + 32 + corridor + 32 + tail, entry=x)
    s.add([x, FLOOR, lead + 32 + corridor, 20])
    step = s.add([s_left, 186, 32, 34])
    wall_x = s_left + 32 + corridor
    path = [step, s.add([s_left + 36, 140, 40, 10])]
    # Platforms over the corridor up to the wall, evenly spaced with gaps of
    # about ``space`` pixels, each 8 pixels higher than the last (to 120).
    start, space = s_left + 76, rng.randint(18, 24)
    count = max(1, (wall_x - start - space) // (36 + space))
    width = (wall_x - start - (count + 1) * space) / count
    top = 140
    for i in range(count):
        top = max(top - 8, 120)
        left = start + space + i * (width + space)
        path.append(s.add([round(left), top, round(width), 10]))
    wall = s.add([wall_x, 112, 32, FLOOR - 112])
    path.append(wall)
    s.add([wall_x + 32, FLOOR, tail, 20])
    back = s_left + 32 + 2
    segments = [
        segment("retreat", -1, reach_x=back),
        segment("alternate_route", 1, route=path, on=wall),
    ]
    if inside:
        s.spawn = s_left + 32 + rng.randint(40, 56)
    else:
        # Advance into the corridor until the wall is on screen (the camera
        # keeps Mario a third of the way across a 256-pixel screen).
        segments.insert(0, segment("advance", 1, avoid=path[1:], reach_x=wall_x - 170))
    s.segments = segments
    return s


def monster(rng, difficulty, x, inside: bool = False) -> Section:
    """A monster walks out of a tunnel too low to jump in; Mario keeps behind
    the tunnel's mouth and jumps over it in the open."""
    open_area, tunnel, tail = 72, rng.randint(140, 170), 64
    s = Section(open_area + tunnel + tail, entry=x)
    s.add([x, FLOOR, s.width, 20])
    mouth = x + open_area
    s.add([mouth, 40, tunnel, 150], "brick")
    speed = round((0.5, 0.65, 0.8)[_tier(difficulty)] + rng.uniform(-0.05, 0.05), 3)
    # It wakes once on screen (MarioScenarioEnv._update_enemy); a Mario
    # started inside the tunnel has it on screen from the start.
    start = mouth + (rng.randint(64, 120) if inside else rng.randint(tunnel // 2, tunnel - 24))
    s.enemies.append(
        {
            "kind": "monster",
            "x": start,
            "y": FLOOR - 20,
            "patrol_min": x + 4,
            "patrol_max": mouth + tunnel + 32,
            "speed": speed,
            "direction": -1,
        }
    )
    s.segments = [segment("advance", 1, kind="monster", past_enemy=0, keep_behind=mouth - 4)]
    if inside:
        s.spawn = mouth + rng.randint(20, 36)
    s.waits = int((start - x + 256) / speed)
    return s


SECTIONS: dict[str, Callable] = {
    "flat": flat,
    "enemy": enemy,
    "gap": gap,
    "pipe": pipe,
    "stairs": stairs,
    "plant": plant,
    "bridge": bridge,
    "upper_route": upper_route,
    "lower_route": lower_route,
    "dead_end": dead_end,
    "monster": monster,
}


def compose(rng: random.Random, difficulty: str, parts: list) -> tuple[dict, dict]:
    """A layout from section names (or (name, keyword arguments) pairs), left to
    right after a short start and before a finish stretch with the goal.

    Returns (scenario, parameters). The scenario has its schedule
    (scenario["tactics"]) and a frame budget for the teacher's route.
    """
    platforms, kinds, enemies, coins = [], [], [], []
    schedule: list = []
    flags: dict = {}
    spawn = None
    waits = 0
    x = 0
    names = []
    plan = [("flat", {"width": 40})] + [p if isinstance(p, tuple) else (p, {}) for p in parts]
    plan.append(("flat", {"width": rng.randint(72, 96)}))
    for name, options in plan:
        section = SECTIONS[name](rng, difficulty, x, **options)
        names.append(name)
        p0, e0 = len(platforms), len(enemies)

        def shifted(seg):
            seg = dict(seg)
            seg["route"] = [i + p0 for i in seg["route"]]
            seg["forbidden"] = [i + p0 for i in seg["forbidden"]]
            seg["avoid"] = [i + p0 for i in seg["avoid"]]
            end = dict(seg["end"])
            if "on" in end:
                end["on"] += p0
            if "past_enemy" in end:
                end["past_enemy"] += e0
            seg["end"] = end
            if "stomp" in seg:
                seg["stomp"] += e0
            return seg

        if section.segments:
            if section.spawn is None and schedule:
                # The advance before it ends where its own segments begin.
                schedule[-1]["end"] = {"reach_x": section.entry}
            elif section.spawn is None:
                schedule.append(segment("advance", 1, reach_x=section.entry))
            schedule.extend(shifted(seg) for seg in section.segments)
            schedule.append(segment("advance", 1))
        if section.spawn is not None:
            spawn = section.spawn
        platforms += section.platforms
        kinds += section.kinds
        enemies += section.enemies
        coins += section.coins
        flags.update(section.flags)
        waits += section.waits
        x += section.width
    if not schedule:
        schedule.append(segment("advance", 1))
    # Only the last segment ends at the goal.
    if any("goal" in seg["end"] for seg in schedule[:-1]):
        raise ValueError("a section ended its tactics at the goal")
    platforms, kinds, schedule = _merge_floors(platforms, kinds, schedule)
    scenario = {
        "world_width": x,
        "mario": [spawn if spawn is not None else 20, FLOOR - 16],
        "platforms": platforms,
        "platform_kinds": kinds,
        "enemies": enemies,
        "coins": coins,
        "goal": [x - 32, FLOOR - 20, 16, 20],
        "goal_requires_support": True,
        "reward_goal_distance_shaping": 2.0,
        "tactics": schedule,
        "frame_budget": int(160 + x / 1.2 + waits),
        **flags,
    }
    return scenario, {"sections": names[1:-1], "difficulty_bin": difficulty}


def _merge_floors(platforms: list, kinds: list, schedule: list) -> tuple[list, list, list]:
    """Join floor pieces that touch end to end into one.

    Each section brings its own floor, so neighbouring sections' floors meet at
    a seam. The teachers would take a seam for a gap. Pieces a tactic segment
    names are kept as they are; segment indices are renumbered.
    """
    named = {
        i
        for seg in schedule
        for i in (*seg["route"], *seg["forbidden"], *seg["avoid"], seg["end"].get("on"))
        if i is not None
    }
    floors = sorted(
        (
            i
            for i, p in enumerate(platforms)
            if isinstance(p, list) and p[1] == FLOOR and kinds[i] is None and i not in named
        ),
        key=lambda i: platforms[i][0],
    )
    merged = [list(p) if isinstance(p, list) else p for p in platforms]
    dropped = set()
    keeper = None
    for i in floors:
        if keeper is not None and merged[keeper][0] + merged[keeper][2] == merged[i][0]:
            merged[keeper][2] += merged[i][2]
            dropped.add(i)
        else:
            keeper = i
    new_index = {}
    for i in range(len(platforms)):
        if i not in dropped:
            new_index[i] = len(new_index)

    def renumbered(seg):
        seg = dict(seg)
        for key in ("route", "forbidden", "avoid"):
            seg[key] = [new_index[i] for i in seg[key]]
        if "on" in seg["end"]:
            seg["end"] = {"on": new_index[seg["end"]["on"]]}
        return seg

    keep = [i for i in range(len(platforms)) if i not in dropped]
    return (
        [merged[i] for i in keep],
        [kinds[i] for i in keep],
        [renumbered(seg) for seg in schedule],
    )


def route_actions(scenario: dict, max_frames: Optional[int] = None) -> list[int]:
    """The teacher's route through a layout from its start ([] if it fails)."""
    from .env import MarioScenarioEnv
    from .policy_recovery import coached_suffix

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        budget = max_frames or scenario.get("frame_budget", 320)
        return coached_suffix(env, max_frames=budget) or []
    finally:
        env.close()
