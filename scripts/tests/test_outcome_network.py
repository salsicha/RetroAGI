"""The outcome network (core/outcome_network.py) and its trainer
(stages/block_smb/outcome_train.py)."""

import numpy as np
import torch

from retroagi.core.outcome_network import PREDICTIONS, OutcomeConfig, OutcomeNetwork, pick, predict
from retroagi.core.smb_choices import ENEMY_SLOTS, GENERAL, SURFACE_SLOTS, state_tokens
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.game_levels import BlockTruthState, random_level
from retroagi.stages.block_smb.outcome_sampling import OUTCOME_FIELDS
from retroagi.stages.block_smb.outcome_train import loss_of, targets

SMALL = OutcomeConfig(width=32, layers=1, heads=2)


def tokens():
    level = random_level(4)
    env = MarioScenarioEnv(world_width=level["world_width"])
    env.draws = False
    env.reset(scenario=level)
    return state_tokens(BlockTruthState().read(env))[:3]


def test_one_pass_gives_every_choice_its_predictions():
    model = OutcomeNetwork(SMALL)
    mario, surfaces, enemies = (torch.as_tensor(t)[None] for t in tokens())
    land, enemy, general = model(mario, surfaces, enemies)
    n = len(PREDICTIONS)
    assert land.shape == (1, SURFACE_SLOTS, 3, n) and enemy.shape == (1, ENEMY_SLOTS, 3, n)
    assert general.shape == (1, len(GENERAL), n)
    # pick takes each choice's row from its own head.
    out = tuple(o.expand(4, *o.shape[1:]) for o in (land, enemy, general))
    kind = torch.tensor([0, 1, 2, 4])
    slot = torch.tensor([2, 1, 1, -1])
    modifier = torch.tensor([1, 0, 1, 1])
    p = pick(out, kind, slot, modifier)
    assert torch.equal(p[0], land[0, 2, 1]) and torch.equal(p[1], enemy[0, 1, 0])
    assert torch.equal(p[2], enemy[0, 1, 2]) and torch.equal(p[3], general[0, GENERAL.index((4, 1))])
    chances = predict(model, tokens(), [(0, 0, 0), (3, -1, 1)], "cpu")
    assert chances.shape == (2, n) and ((0 <= chances[:, :3]) & (chances[:, :3] <= 1)).all()


def test_the_network_learns_a_handful_of_outcomes():
    torch.manual_seed(0)
    model = OutcomeNetwork(SMALL)
    mario, surfaces, enemies = (torch.as_tensor(t)[None].expand(6, *np.shape(t)) for t in tokens())
    kind = torch.tensor([0, 0, 3, 3, 4, 4])
    slot = torch.tensor([0, 1, -1, -1, -1, -1])
    modifier = torch.tensor([1, 1, 0, 2, 0, 1])
    outcomes = np.zeros((6, len(OUTCOME_FIELDS)), np.float32)
    outcomes[1, OUTCOME_FIELDS.index("died")] = 1.0  # the second landing kills
    outcomes[2, OUTCOME_FIELDS.index("progress")] = 32.0
    target = torch.as_tensor(targets(outcomes))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        loss = loss_of(pick(model(mario, surfaces, enemies), kind, slot, modifier), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    p = pick(model(mario, surfaces, enemies), kind, slot, modifier).detach()
    assert torch.sigmoid(p[1, 0]) > 0.8 and torch.sigmoid(p[0, 0]) < 0.2
    assert abs(float(p[2, 3]) * 64 - 32) < 4
