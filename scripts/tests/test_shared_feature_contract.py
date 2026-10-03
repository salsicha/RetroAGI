"""The shared SMB observation: only what Full SMB can also show, in its screen frame."""

import numpy as np
import pytest

from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_ID,
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)

WIDE = {
    "world_width": 512,
    "mario": [40, 204],
    "platforms": [[0, 220, 330, 20], [370, 220, 142, 20]],
    "goal": [480, 200, 16, 20],
}
PATROL = {**WIDE, "enemies": [[200, 204, 150, 260]]}
LIFT = {"x": 120, "y": 160, "w": 40, "h": 10, "moving": [100, 180, 1.0]}


def run(block, actions):
    block.reset(seed=0)
    rows = [block.state_features()]
    for action in actions:
        block.step(action)
        rows.append(block.state_features())
    return np.array(rows)


def test_block_env_runs_nes_physics_only():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=WIDE)
        assert (env.mario["w"], env.mario["h"]) == (10, 12) and env.mario["on_ground"]
        assert "coyote_frames" not in env.mario and "jump_buffer" not in env.mario
        # Holding jump through a landing does not jump again.
        airborne = []
        for _ in range(90):
            env.step(5)
            airborne.append(not env.mario["on_ground"])
        first_landing = airborne.index(False, airborne.index(True))
        assert not any(airborne[first_landing:])
    finally:
        env.close()


@pytest.mark.parametrize("family", ["single_gap", "enemy_stomp", "pit_leap", "bridge_mount"])
def test_every_layout_has_a_route_verified_under_nes_physics(family):
    def draw():
        return sample_block_smb_monte_carlo_scenario(
            split="validation", seed=3, sample_index=0, family=family, difficulty="hard"
        )

    sample = draw()
    assert sample.scenario_id.startswith(BLOCK_SMB_MC_ID)
    assert validate_block_smb_monte_carlo_oracle(sample.scenario, sample.oracle["actions"])[
        "reachable"
    ]
    assert sample.scenario == draw().scenario
    if family == "enemy_stomp":
        width = sample.scenario["world_width"]
        assert all(enemy[2:4] == [0, width] for enemy in sample.scenario["enemies"])
        assert sample.scenario["task_objective"] == "stomp"
    if family == "pit_leap":
        assert sample.scenario["mario_velocity"] == [2.5, 0.0]


def test_platform_hop_jumps_from_the_first_frame_with_a_menu_hold():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=3, sample_index=0, family="platform_hop", difficulty="medium"
    )
    actions = sample.oracle["actions"]
    hold = next(i for i, a in enumerate(actions) if a != 2)
    assert actions[0] == 2 and hold in NES_JUMP_FRAMES
