"""Differential NES/Block physics audit; never promotes a partial physics profile."""

import argparse
import json
from pathlib import Path

from retroagi.core.smb_physics import NES_PHYSICS_PROFILE, NESPlayerMotion
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage


def audit():
    class NoVision:
        pass

    scripts = {
        "run": [1] * 90,
        "reverse": [1] * 35 + [3] * 30,
        "brake": [1] * 30 + [0] * 30,
        "air_steer": [1] * 20 + [2] * 8 + [4] * 8 + [1] * 24,
        "held_landing": [5] * 100,
        "release_repress": [5] * 28 + [0] * 40 + [5] * 10 + [0] * 45,
    }
    scripts.update({f"jump_{hold}": [5] * hold + [0] * 65 for hold in (1, 3, 7, 13, 19, 28, 32)})
    stage = FullSMBStage(env_config=FullSMBEnvConfig(state="Level1-1"), vision=NoVision())
    reports = []
    try:
        for name, actions in scripts.items():
            stage.reset(seed=901)
            ram = stage.env.get_ram()
            block = MarioScenarioEnv(physics_profile=NES_PHYSICS_PROFILE)
            block.reset(
                scenario={
                    "world_width": 1024,
                    "mario": [43, 196],
                    "platforms": [[0, 208, 1024, 32]],
                }
            )
            block.motion = NESPlayerMotion.from_ram(ram)
            rows = []
            try:
                for frame, action in enumerate(actions):
                    block.step(action)
                    stage.step(action)
                    ram = stage.env.get_ram()
                    x = int(ram[0x6D]) * 256 + int(ram[0x86]) + 3
                    y = (int(ram[0xB5]) - 1) * 256 + int(ram[0xCE]) + 20
                    rows.append(
                        dict(
                            frame=frame,
                            action=action,
                            nes=[x, y],
                            block=[block.mario["x"], block.mario["y"]],
                            position_error=max(
                                abs(x - block.mario["x"]), abs(y - block.mario["y"])
                            ),
                            contact_matches=bool(
                                bool(block.mario["on_ground"]) == (ram[0x1D] == 0)
                            ),
                        )
                    )
            finally:
                block.close()
            reports.append(
                dict(
                    name=name,
                    max_error=max(r["position_error"] for r in rows),
                    contact_mismatches=sum(not r["contact_matches"] for r in rows),
                    frames=rows,
                )
            )
    finally:
        stage.close()
    return dict(
        profile=NES_PHYSICS_PROFILE,
        exact_motion_gate=all(
            r["max_error"] == 0 and r["contact_mismatches"] == 0 for r in reports
        ),
        coverage=[r["name"] for r in reports],
        trials=reports,
        unsupported=["water", "climbing", "power_state_transitions"],
        collision_profile_qualified=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exact_motion_gate": report["exact_motion_gate"],
                "trials": [{k: v for k, v in r.items() if k != "frames"} for r in report["trials"]],
            }
        )
    )


if __name__ == "__main__":
    main()
