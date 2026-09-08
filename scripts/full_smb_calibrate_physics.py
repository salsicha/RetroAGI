"""Measure actual jump heights and write a versioned duration mapping.

This is a calibration experiment, not policy evaluation. No model weights are
updated. The resulting profile explicitly records physical emulator frames for
each learned Block SMB duration bin and measures the residual height mismatch.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from retroagi.core import load_checkpoint, save_checkpoint
from retroagi.core.smb_runtime import SMBRuntimeContract
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage
from retroagi.stages.full_smb.geometry import _box


def measure():
    class NoVision:
        pass

    rows = []
    env = FullSMBStage(env_config=FullSMBEnvConfig(state="Level1-1"), vision=NoVision())
    try:
        for hold in range(1, 41):
            env.reset()
            start = _box(env.env.get_ram(), 0, 0).bottom
            best = start
            air = False
            for frame in range(100):
                env.step(5 if frame < hold else 0)
                ram = env.env.get_ram()
                best = min(best, _box(ram, 0, 0).bottom)
                air |= bool(ram[0x1D] != 0)
                if air and ram[0x1D] == 0:
                    break
            rows.append(dict(engine="nes", hold=hold, rise=start - best, flight=frame + 1))
    finally:
        env.close()
    env = MarioScenarioEnv()
    try:
        for hold in range(1, 17):
            env.reset(
                scenario={
                    "width": 256,
                    "mario": [40, 200],
                    "platforms": [[0, 220, 256, 20]],
                    "goal": [240, 200, 16, 20],
                }
            )
            start = env.mario["y"]
            best = start
            air = False
            for frame in range(100):
                env.step(5 if frame < hold else 0)
                best = min(best, env.mario["y"])
                air |= not env.mario["on_ground"]
                if air and env.mario["on_ground"]:
                    break
            rows.append(dict(engine="block", hold=hold, rise=start - best, flight=frame + 1))
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        raise FileExistsError("Calibration must write a new checkpoint")
    rows = measure()
    nes = [r for r in rows if r["engine"] == "nes"]
    block = [r for r in rows if r["engine"] == "block"]
    table = tuple(
        min(nes, key=lambda n: (abs(n["rise"] - b["rise"]), n["hold"]))["hold"] for b in block
    )
    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    contract = SMBRuntimeContract(**ck["config"]["smb_runtime_contract"])
    ck["config"]["smb_runtime_contract"] = replace(
        contract, jump_hold_frames=table, physics_profile="nes_height_calibration_v1"
    ).manifest()
    ck.setdefault("metadata", {})["physics_calibration"] = {
        "trials": rows,
        "mapping": table,
        "criterion": "nearest stationary jump apex; shortest hold breaks ties",
        "full_level_qualified": False,
    }
    save_checkpoint(args.output, ck)
    args.output.with_suffix(".physics.json").write_text(
        json.dumps(ck["metadata"]["physics_calibration"], indent=2) + "\n"
    )
    print(json.dumps({"mapping": table, "output": str(args.output)}))


if __name__ == "__main__":
    main()
