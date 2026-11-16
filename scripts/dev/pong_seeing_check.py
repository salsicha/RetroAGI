"""How well seeing follows the real Pong: the vision network (vision.py) and
the tracker (seeing.py) replayed over recorded whole games it was not
trained on, the tracker's state compared with the game's memory frame by
frame. The share of frames each part is wrong, and how often the tracker
started again.

    python scripts/dev/pong_seeing_check.py [--vision data/pong_vision/vision.pt] [--parts PART.npz ...]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from retroagi.domains.pong.knob import Knob  # noqa: E402
from retroagi.domains.pong.seeing import Seeing  # noqa: E402
from retroagi.domains.pong.vision import FIELDS, NAMES, load, load_frames  # noqa: E402

PARTS = ("ball_x", "ball_y", "across", "speed", "ours", "drawn", "theirs", "frame%8", "points")


def readings(model, screens, batch: int = 256) -> dict:
    import torch

    out = {name: [] for name in NAMES}
    with torch.no_grad():
        for i in range(0, len(screens), batch):
            o = model(torch.from_numpy(screens[i:i + batch]).cuda())
            for name in NAMES:
                out[name].append(torch.log_softmax(o[name], -1).cpu().numpy())
    return {name: np.concatenate(v) for name, v in out.items()}


def check(model, frames: dict) -> dict:
    seen_all = readings(model, frames["screens"])
    memory = {f: frames["memory"][:, i].astype(int) for i, f in enumerate(FIELDS)}
    seeing, knob = Seeing(), Knob(step=0)
    wrong, counted, followed = Counter(), 0, 0
    started = time.perf_counter()
    for t in range(len(frames["actions"])):
        if frames["starts"][t]:
            seeing.reset()
            knob = Knob(step=0)
        action = int(frames["actions"][t])
        knob.press(action)
        seeing.acted(action)
        s = seeing.observe({name: seen_all[name][t] for name in NAMES}, knob)
        m = {f: memory[f][t] for f in FIELDS}
        if m["ball_y"] == 0:
            continue
        counted += 1
        if s is None:
            continue
        followed += 1
        truth = {"ball_x": m["ball_x"], "ball_y": m["ball_y"], "across": m["across"], "speed": m["speed"],
                 "ours": m["our_paddle"], "drawn": m["our_drawn"], "theirs": m["their_paddle"],
                 "frame%8": m["frame"] % 8, "points": (m["our_score"], m["their_score"])}
        got = {"ball_x": s.ball_x, "ball_y": s.ball_y, "across": s.across, "speed": s.speed, "ours": s.ours,
               "drawn": s.drawn, "theirs": s.theirs, "frame%8": s.frame % 8, "points": s.points}
        for part in PARTS:
            wrong[part] += got[part] != truth[part]
    seconds = time.perf_counter() - started
    return {"frames with the ball in play": counted, "followed": followed / counted, "starts": seeing.starts,
            "tracker ms a frame": 1000 * seconds / len(frames["actions"]),
            "share wrong (of followed)": {p: wrong[p] / max(followed, 1) for p in PARTS}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vision", default="data/pong_vision/vision.pt")
    parser.add_argument("--parts", nargs="*", default=None, help="recorded parts (the network's held-out ones by default)")
    args = parser.parse_args(argv)
    import torch

    model = load(args.vision)
    parts = args.parts or torch.load(args.vision, weights_only=False)["held_parts"]
    print(json.dumps(check(model, load_frames(parts)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
