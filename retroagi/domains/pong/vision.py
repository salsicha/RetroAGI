"""Seeing the real Pong: a network that reads the game's screen and gives,
for each thing the agent must know, how likely each of its values is: the
ball's left and top (top 0: no ball in play), our paddle's top as drawn,
their paddle's top, and both scores, in the game's own units (game.py).

It learns from the real game's frames, labelled from the game's memory
(docs/requirements.md: the real game may be watched to train seeing). It is
never run in Block Pong, where the state is a lookup.

What it reads, computes and gives:
- reads one screen (210 rows, 160 columns, colour);
- three layers of 3x3 convolutions at full resolution, then one map for
  each placed thing; the ball's left is read from its map's strongest value
  in each column, the heights from the strongest in each row, each mapped
  to the game's 256 values; a coarse view of the whole screen adds to them
  (a paddle half hidden in a wall is placed by what shows of it);
- each score from its own half of the band above the walls, 0 to 21;
- gives log-probabilities over every value of each (``read``).

The screen shows the paddles' and the ball's heights in two-line rows
(the game draws two scanlines at a time), so a height is often one of two
values: the network says so, and the tracker (seeing.py) settles it by
how things move.

    python -m retroagi.domains.pong.vision collect --frames 80000 --output data/pong_vision/frames
    python -m retroagi.domains.pong.vision train --data data/pong_vision/frames --output data/pong_vision/vision.pt
"""

from pathlib import Path

import numpy as np

ROWS, COLS = 210, 160
VALUES, SCORES = 256, 22
SCORE_ROWS = 24  # the scores are drawn above the top wall,
HALF = COLS // 2  # theirs on the left half, ours on the right
# What the network gives, in the order of its label columns.
PLACED = ("ball_x", "ball_y", "our_drawn", "theirs")
NAMES = PLACED + ("our_score", "their_score")
IGNORE = -100  # a label not learned (the ball's left while it is out of play)
# The memory fields kept beside each frame (for checking the tracker).
FIELDS = ("ball_x", "ball_y", "our_paddle", "their_paddle", "knob", "our_score", "their_score", "across", "speed",
          "our_drawn", "frame", "serve_wait")


def labels(m) -> np.ndarray:
    """The network's labels for the game's memory ``m`` (game.PongMemory)."""
    return np.array([m.ball_x if m.ball_y else IGNORE, m.ball_y, m.our_drawn, m.their_paddle,
                     min(m.our_score, SCORES - 1), min(m.their_score, SCORES - 1)], np.int64)


def network(width: int = 32):
    import torch
    from torch import nn

    class PongVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.width = width
            self.body = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, width, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            self.maps = nn.Conv2d(width, len(PLACED), 1)
            self.columns = nn.Linear(COLS, VALUES)
            self.rows = nn.ModuleList(nn.Linear(ROWS, VALUES) for _ in PLACED[1:])
            self.coarse = nn.Sequential(nn.AdaptiveMaxPool2d((14, 10)), nn.Flatten(), nn.Linear(width * 140, 256),
                                        nn.ReLU(), nn.Linear(256, VALUES * len(PLACED)))
            self.scores = nn.ModuleList(
                nn.Sequential(nn.AdaptiveMaxPool2d((6, 20)), nn.Flatten(), nn.Linear(width * 120, 128), nn.ReLU(),
                              nn.Linear(128, SCORES)) for _ in range(2))

        def forward(self, screens):
            """screens: uint8 [B, ROWS, COLS, 3] -> {name: logits [B, values]}."""
            x = screens.permute(0, 3, 1, 2).float() / 255.0
            f = self.body(x)
            maps = self.maps(f)
            coarse = self.coarse(f).view(-1, len(PLACED), VALUES)
            out = {"ball_x": self.columns(maps[:, 0].amax(1)) + coarse[:, 0]}
            for i, name in enumerate(PLACED[1:], 1):
                out[name] = self.rows[i - 1](maps[:, i].amax(2)) + coarse[:, i]
            # each score from its own half only (from the whole band it learned
            # which scores come together, and misread pairs it had not seen)
            out["our_score"] = self.scores[0](f[:, :, :SCORE_ROWS, HALF:])
            out["their_score"] = self.scores[1](f[:, :, :SCORE_ROWS, :HALF])
            return out

    return PongVision()


def load(path, device="cuda"):
    import torch

    saved = torch.load(path, map_location=device, weights_only=False)
    model = network(saved["width"]).to(device)
    model.load_state_dict(saved["state"])
    return model.eval()


def read(model, screen: np.ndarray) -> dict:
    """{name: log-probabilities over its values (numpy)} for one screen."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(screen))[None].to(device))
        return {name: torch.log_softmax(v[0], -1).cpu().numpy() for name, v in out.items()}


# ---- collecting frames from the real game ----

class RandomParts:
    """A chooser for collecting: returns off one of the paddle's edges (they
    win points) with chance ``edges``, else off any part (the middle keeps
    rallies going); rests anywhere."""

    def __init__(self, rng, edges: float = 0.5):
        self.rng, self.edges = rng, edges

    def decide(self, state, last_event=None, last_choice=None):
        from .controller import PARTS, Choice, coming

        if coming(state):
            if self.rng.random() < self.edges:
                return Choice("return", PARTS[0] if self.rng.random() < 0.5 else PARTS[-1])
            return Choice("return", PARTS[int(self.rng.integers(len(PARTS)))])
        return Choice("rest", int(self.rng.integers(3)))


def collect(frames: int, seed: int) -> dict:
    """``frames`` frames of whole real games: each point is played either by
    the agent on the game's memory, returning off random parts with the
    edges favoured (it wins most, rallies long), or by held random buttons
    (it loses); the share of points the agent plays is drawn for each game,
    so the games end anywhere from 21-0 to 0-21 and both scores are seen
    through their whole range (learned from games the agent mostly lost,
    the network misread our score past 7). Also one press in 30 at random."""
    from retroagi.framework.agent import Agent

    from .controller import Controller
    from .game import STAY
    from .real import MemorySource, RealPongWorld

    rng = np.random.default_rng(seed)
    world, source = RealPongWorld(), MemorySource()
    screens, labelled, memory, actions, starts = [], [], [], [], []
    start, agent, held, hold = True, None, 0, 0
    share = rng.uniform(0.1, 0.95)  # of points the agent plays, this game
    world.reset()
    source.reset()
    points = None
    while len(screens) < frames:
        m = world.game.memory()
        if (m.our_score, m.their_score) != points:  # a new point: who plays it
            points = (m.our_score, m.their_score)
            agent = None
            if rng.random() < share:
                agent = Agent(source, lambda mode: RandomParts(rng, 0.8), Controller, STAY)
                agent.chooser, agent.controller = agent.make_chooser("points"), Controller()
        if agent is not None:
            action = agent.step(source.read(world))
        else:
            if hold <= 0:
                held, hold = int(rng.integers(3)), int(rng.integers(1, 30))
            hold -= 1
            action = held
        if rng.random() < 1 / 30:
            action = int(rng.integers(3))
        source.acted(action)
        step = world.step(action)
        m = world.game.memory()
        screens.append(world.screen)
        labelled.append(labels(m))
        memory.append([getattr(m, f) for f in FIELDS])
        actions.append(action)
        starts.append(start)
        start = False
        if step.over:
            world.reset()
            source.reset()
            start, points = True, None
            share = rng.uniform(0.1, 0.95)
    world.close()
    return {"screens": np.stack(screens), "labels": np.stack(labelled), "memory": np.array(memory, np.int16),
            "actions": np.array(actions, np.int8), "starts": np.array(starts, bool)}


def _collect_part(args):
    frames, seed, path = args
    np.savez_compressed(path, **collect(frames, seed))
    return path


def load_frames(paths) -> dict:
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def frame_files(data) -> list:
    return sorted(Path(data).glob("part_*.npz"))


# ---- training ----

def train(data, output, steps: int = 8000, batch: int = 64, width: int = 32, held: float = 0.1, seed: int = 0):
    """Train the network on the frames in ``data`` (parts of whole games);
    the last ``held`` share of the parts is kept out and scored on: the
    share of frames where each thing's most likely value is the true one,
    and where the true one has at least 1% (what the tracker needs)."""
    import torch

    torch.manual_seed(seed)
    files = frame_files(data)
    cut = max(1, int(round(len(files) * held)))
    train_set, held_set = load_frames(files[:-cut]), load_frames(files[-cut:])
    device = "cuda"
    screens = torch.from_numpy(train_set["screens"]).to(device)
    targets = torch.from_numpy(train_set["labels"]).to(device)
    model = network(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=steps)
    loss_of = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    g = torch.Generator(device=device).manual_seed(seed)
    for step in range(1, steps + 1):
        index = torch.randint(len(screens), (batch,), device=device, generator=g)
        out = model(screens[index])
        loss = sum(loss_of(out[name], targets[index, i]) for i, name in enumerate(NAMES))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    report = evaluate(model.eval(), held_set)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": width, "state": model.state_dict(), "held": report,
                "train_parts": [str(f) for f in files[:-cut]], "held_parts": [str(f) for f in files[-cut:]]}, output)
    return report


def evaluate(model, frames: dict, batch: int = 256) -> dict:
    import torch

    device = next(model.parameters()).device
    by_value = {name: np.zeros((SCORES, 2)) for name in ("our_score", "their_score")}
    right = {name: 0 for name in NAMES}
    allowed = {name: 0 for name in NAMES}
    counted = {name: 0 for name in NAMES}
    with torch.no_grad():
        for i in range(0, len(frames["screens"]), batch):
            out = model(torch.from_numpy(frames["screens"][i:i + batch]).to(device))
            truth = torch.from_numpy(frames["labels"][i:i + batch]).to(device)
            for j, name in enumerate(NAMES):
                t = truth[:, j]
                keep = t != IGNORE
                p = torch.log_softmax(out[name], -1)
                right[name] += int(((p.argmax(-1) == t) & keep).sum())
                allowed[name] += int(((p.gather(1, t.clamp(min=0)[:, None])[:, 0] > np.log(0.01)) & keep).sum())
                counted[name] += int(keep.sum())
                if name in by_value:
                    for v, r in zip(t.cpu().numpy(), (p.argmax(-1) == t).cpu().numpy()):
                        by_value[name][v] += (r, 1)
    report = {name: {"most likely right": right[name] / counted[name],
                     "true value at least 1%": allowed[name] / counted[name], "frames": counted[name]} for name in NAMES}
    for name, table in by_value.items():
        seen = table[:, 1] >= 100  # (21 is shown one frame: the game is over)
        report[name]["worst value shown 100 frames or more"] = float((table[seen, 0] / table[seen, 1]).min())
        report[name]["values shown 100 frames or more"] = int(seen.sum())
    return report


def main(argv=None) -> int:
    import argparse
    import json
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="Seeing the real Pong: collect frames, train the network")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--frames", type=int, default=80000)
    c.add_argument("--parts", type=int, default=20)
    c.add_argument("--workers", type=int, default=10)
    c.add_argument("--first-seed", type=int, default=0)
    c.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--width", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "collect":
        Path(args.output).mkdir(parents=True, exist_ok=True)
        jobs = [(args.frames // args.parts, args.first_seed + i, str(Path(args.output) / f"part_{i:03d}.npz"))
                for i in range(args.parts)]
        with Pool(args.workers) as pool:
            for path in pool.imap_unordered(_collect_part, jobs):
                print(path, flush=True)
    else:
        print(json.dumps(train(args.data, args.output, args.steps, width=args.width), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
