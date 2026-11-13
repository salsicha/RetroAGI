"""How a kind of enemy of the real game moves, read from its memory: Mario
is flown into a level (game_catalogue.fly's way of reaching it) until an
enemy of the kind is on the screen, then held in the air (still, or at a
speed given; nothing hurts him and he hurts nothing), and the
enemy's place, speed and collision box are read every frame. For making
Block SMB move as the game does (docs/requirements.md, section 3). The
levels scored on are refused.

    python scripts/dev/measure_enemy.py --level 1-3 --code 0x03 [--frames 600] [--output track.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import game_catalogue as catalogue

ENEMY_PAGE, ENEMY_X, ENEMY_Y_PAGE, ENEMY_Y = 0x6E, 0x87, 0xB6, 0xCF  # Enemy_PageLoc, _X_Position, _Y_HighPos, _Y_Position
ENEMY_SPEED, ENEMY_Y_SPEED, ENEMY_STATE, ENEMY_DIR = 0x58, 0xA0, 0x1E, 0x46  # Enemy_X_Speed, _Y_Speed, Enemy_State, Enemy_MovingDir
BOXES = 0x4B0  # EnemyBoundingBoxCoord: x0, y0, x1, y1 on the screen, four bytes a slot
CAMERA_PAGE, CAMERA_X = 0x71A, 0x71C  # ScreenLeft_PageLoc, ScreenLeft_X_Pos
X_SPEED, X_FORCE = 0x57, 0x705  # Player_X_Speed (sixteenths of a pixel a frame), Player_X_MoveForce
# While held Mario is not invincible (it would kill what touches him) but
# flashing as after a hit (InjuryTimer): nothing hurts him and he hurts
# nothing.
INJURY = 0x79E


def signed(byte: int) -> int:
    return byte - 256 if byte > 127 else byte


def measure(level: str, code: int, frames: int = 600, before: int = 20000, hold_x: int = 0,
            speed: float = 0.0, height: int = catalogue.FLY_HEIGHT) -> dict:
    """Every slot's enemy each frame, from the first frame an enemy of
    ``code`` is on the screen (``hold_x``: or once Mario is that far into
    the level); Mario then held ``height`` from the screen's top, moving
    ``speed`` pixels a frame (in the air the game keeps his speed: it is
    written each frame)."""
    import retro

    if level in catalogue.SCORED:
        raise ValueError(f"{level} is kept for scoring")
    world, number = map(int, level.split("-"))
    env = retro.make("SuperMarioBros-Nes", state="Level2-1", inttype=retro.data.Integrations.STABLE, render_mode=None)
    env.reset()
    idle = np.zeros(len(env.buttons), dtype=np.int8)
    run = idle.copy()
    run[env.buttons.index("RIGHT")] = run[env.buttons.index("B")] = 1
    memory = env.data.memory

    def write(address, value):
        memory.assign(address, "|u1", value)

    for _ in range(5):
        env.step(idle)
    for address, value in ((catalogue.WORLD, world - 1), (catalogue.LEVEL, number - 1),
                           (catalogue.AREA, catalogue.area_of(world, number)), (catalogue.HALFWAY, 0),
                           (catalogue.LIVES, 5), (catalogue.TIMER, 0), (catalogue.TIMER + 1, 0), (catalogue.TIMER + 2, 1)):
        write(address, value)
    here = (world - 1, number - 1)
    started, holding, rows = False, False, []
    try:
        for frame in range(before + frames):
            ram = env.get_ram()
            started = started or (frame > 300 and ram[catalogue.ENGINE] == catalogue.PLAYING
                                  and (ram[catalogue.WORLD], ram[catalogue.LEVEL]) == here)
            if started:
                for address, value in ((catalogue.Y, height if holding else catalogue.FLY_HEIGHT),
                                       (catalogue.Y_PAGE, 1), (catalogue.Y_SPEED, 0), (catalogue.Y_FORCE, 0),
                                       (catalogue.STAR, 0 if holding else 30), (catalogue.TIMER, 3)):
                    write(address, value)
                if holding:
                    write(X_SPEED, round(speed * 16) % 256)
                    write(X_FORCE, 0)
                    write(INJURY, 8)
            env.step(idle if holding or not started else run)
            if not started:
                continue
            ram = env.get_ram()
            mario_x = int(ram[catalogue.PAGE]) * 256 + int(ram[catalogue.X])
            slots = [s for s in range(catalogue.SLOTS) if ram[catalogue.ENEMY_ON + s]]
            if not holding:
                holding = (hold_x and mario_x >= hold_x) or (not hold_x and any(ram[catalogue.ENEMY_ID + s] == code for s in slots))
                if not holding:
                    continue
            elif speed > 0 and rows and mario_x <= rows[-1]["mario_x"]:  # against a wall: on through it
                write(catalogue.PAGE, (mario_x + 16) // 256)
                write(catalogue.X, (mario_x + 16) % 256)
            rows.append({"frame": len(rows), "mario_x": mario_x,
                         "camera": int(ram[CAMERA_PAGE]) * 256 + int(ram[CAMERA_X]),
                         "enemies": [{"slot": s, "code": int(ram[catalogue.ENEMY_ID + s]),
                                      "x": int(ram[ENEMY_PAGE + s]) * 256 + int(ram[ENEMY_X + s]),
                                      "y": (int(ram[ENEMY_Y_PAGE + s]) - 1) * 256 + int(ram[ENEMY_Y + s]),
                                      "speed": signed(int(ram[ENEMY_SPEED + s])), "y_speed": signed(int(ram[ENEMY_Y_SPEED + s])),
                                      "state": int(ram[ENEMY_STATE + s]), "direction": int(ram[ENEMY_DIR + s]),
                                      "box": [int(b) for b in ram[BOXES + 4 * s:BOXES + 4 * s + 4]]} for s in slots]})
            if len(rows) >= frames:
                break
    finally:
        env.close()
    return {"level": level, "code": code, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", required=True)
    parser.add_argument("--code", type=lambda s: int(s, 0), required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--hold-x", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.0, help="Mario's speed while held (pixels a frame)")
    parser.add_argument("--height", type=int, default=catalogue.FLY_HEIGHT, help="Mario's top while held")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = measure(args.level, args.code, args.frames, hold_x=args.hold_x, speed=args.speed,
                     height=args.height)
    tracks = {}
    for row in result["rows"]:
        for e in row["enemies"]:
            tracks.setdefault((e["slot"], e["code"]), []).append(e)
    for (slot, code), track in tracks.items():
        xs = [e["x"] for e in track]
        moves = np.diff(xs) if len(xs) > 1 else np.zeros(1)
        print(f"slot {slot} {catalogue.NAMES.get(code, hex(code)):36} {len(track):4} frames  x {xs[0]}..{xs[-1]}  "
              f"pixels a frame {np.mean(np.abs(moves)):.3f}  speeds {sorted(set(e['speed'] for e in track))}  "
              f"box {track[0]['box']}")
    if args.output:
        args.output.write_text(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
