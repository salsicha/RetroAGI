"""Contact trajectory comparisons on identical visible NES terrain and starts."""

import json
from pathlib import Path

import torch

from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.stages.block_smb.env import MarioScenarioEnv
from scripts.smb_emulator_curriculum import geometry, make_stage


def compare_contacts(directory):
    directory = Path(directory)
    cases = json.loads((directory / "approaches.json").read_text())["cases"]
    stage = make_stage()
    reports = []
    try:
        for case in cases:
            if (
                case["split"] != "train"
                or case["kind"] != "mount"
                or case["variation"] != {"wait": 0, "walk": 0}
            ):
                continue
            start = torch.load(directory / case["snapshot"], weights_only=False)
            for name, actions in {
                "wall": [1] * 40,
                "mount_short": [2] * 8 + [1] * 64,
                "mount_long": [2] * 32 + [1] * 64,
            }.items():
                stage.load_emulator_state(start)
                g = geometry(stage)
                scroll = g["scroll"]
                env = MarioScenarioEnv(physics_profile="nes_land_v1")
                env.reset(
                    scenario={
                        "mario": [g["world_x"], g["scene"].mario["y"]],
                        "world_width": int(scroll + 512),
                        "platforms": [
                            [p["rect"].x + scroll, p["rect"].y, p["rect"].w, p["rect"].h]
                            for p in g["scene"].platforms
                        ],
                    }
                )
                env.motion = NESPlayerMotion.from_ram(stage.env.get_ram())
                env.mario.update(
                    vx=g["scene"].mario["vx"],
                    vy=g["scene"].mario["vy"],
                    on_ground=g["scene"].mario["on_ground"],
                )
                env.camera_x = scroll
                rows = []
                try:
                    for frame, action in enumerate(actions):
                        if env.mario["x"] + env.mario["w"] + 3 >= scroll + 256:
                            break  # the frozen comparison terrain ends at the captured viewport
                        env.step(action)
                        _, _, done, truncated, info = stage.step(action)
                        actual = geometry(stage)
                        if actual.get("enemy_contact"):
                            break  # enemy contact is outside this terrain-only comparison
                        rows.append(
                            dict(
                                frame=frame,
                                action=action,
                                nes=[actual["world_x"], actual["scene"].mario["y"]],
                                block=[env.mario["x"], env.mario["y"]],
                                error=max(
                                    abs(actual["world_x"] - env.mario["x"]),
                                    abs(actual["scene"].mario["y"] - env.mario["y"]),
                                ),
                                contact_matches=env.mario["on_ground"]
                                == actual["scene"].mario["on_ground"],
                            )
                        )
                        if done or truncated:
                            break
                finally:
                    env.close()
                reports.append(
                    dict(
                        start=case["id"],
                        mechanic=name,
                        frames=rows,
                        max_error=max((r["error"] for r in rows), default=None),
                        contact_mismatches=sum(not r["contact_matches"] for r in rows),
                    )
                )
    finally:
        stage.close()
    result = dict(
        trials=reports,
        qualified=bool(reports)
        and all(
            r["max_error"] is not None and r["max_error"] <= 1 and r["contact_mismatches"] == 0
            for r in reports
        ),
        excluded=[
            "enemy_contact",
            "moving_platform_carry",
            "destructible_block_animation",
            "power_transitions",
        ],
    )
    (directory / "contact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
