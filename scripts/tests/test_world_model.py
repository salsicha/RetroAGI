"""The world model: the policy's LSTM, run once per action after the skill
picks it, predicting the action's outcome (user, 2026-10-15/16)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import asdict

import numpy as np
import pytest
import torch

from retroagi.core.action_tokens import VERBS, ActionToken
from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_kinematics import MARIO_STATE, SPRITE_SLOTS, SPRITE_STATE
from retroagi.core.smb_observer import observation_layout
from retroagi.core.tokens import EXECUTION_WIDTH, token_layout
from retroagi.core.world_model import (
    ACTION_RECORD,
    COLUMN,
    MARIO_TOKEN,
    action_record,
    outcome_loss,
    scene_digest,
)
from retroagi.stages.block_smb import layered_train as lt

pytestmark = pytest.mark.timeout(900)


def scene(policy, count=2):
    from retroagi.core.smb_observer import SEQ_LEN_C

    torch.manual_seed(0)
    tokens, present = policy.encode_scene(
        (
            torch.zeros(count, 8, dtype=torch.long),
            torch.zeros(count, 16, dtype=torch.long),
            torch.rand(count, SEQ_LEN_C) * 0.1,
        )
    )
    sprites = torch.zeros(count, SPRITE_SLOTS, len(SPRITE_STATE))
    mario = torch.zeros(count, len(MARIO_STATE))
    feedback = torch.zeros(count, EXECUTION_WIDTH)
    return tokens, present, sprites, mario, feedback


@torch.no_grad()
def test_an_untrained_world_model_is_unsure():
    policy = LayeredSMBPolicy().eval()
    tokens, present, sprites, mario, feedback = scene(policy)
    verbs = torch.tensor([VERBS.index("wait"), VERBS.index("run")])
    pointers = torch.tensor([MARIO_TOKEN, MARIO_TOKEN])
    out, state = policy.memory.predict(
        tokens, present, sprites, mario, feedback, verbs, pointers, policy.memory.initial(2, "cpu")
    )
    assert torch.allclose(out["mean"][:, :4], torch.zeros(2, 4))
    assert torch.allclose(out["mean"][:, 4], torch.full((2,), 32.0))
    assert torch.allclose(torch.sigmoid(out["success"]), torch.full((2,), 0.5))
    assert state[0].shape == (2, policy.memory.width)


@torch.no_grad()
def test_trying_another_action_does_not_advance_the_state():
    policy = LayeredSMBPolicy().eval()
    model = policy.memory
    tokens, present, sprites, mario, feedback = scene(policy, 1)
    tokens = model.with_motion(tokens, sprites, mario)
    before = tuple(torch.randn(1, model.width) for _ in range(2))
    wait, run = torch.tensor([VERBS.index("wait")]), torch.tensor([VERBS.index("run")])
    pointer = torch.tensor([MARIO_TOKEN])
    executed = model.advance(tokens, feedback, wait, pointer, before)
    tried = model.advance(tokens, feedback, run, pointer, before)
    again = model.advance(tokens, feedback, wait, pointer, before)
    assert all(torch.equal(a, b) for a, b in zip(executed, again))
    assert not torch.equal(executed[0], tried[0])


def test_a_lower_bound_on_frames_is_wrong_only_when_predicted_below_it():
    # A one-frame wait of the teacher's: one it did not play out.
    outcome = {"dx": 0, "dy": 0, "frames": 1, "success": True, "won": False, "room": 64}
    hold = action_record(ActionToken("wait"), outcome, chosen=True)
    assert hold[COLUMN["frames_censored"]] == 1.0 and hold[COLUMN["pointer"]] == MARIO_TOKEN
    run = action_record(ActionToken("run", ("surfaces", 0)), outcome, chosen=True)
    assert run[COLUMN["frames_censored"]] == 0.0
    records = torch.as_tensor(np.stack([hold, hold]))
    records[1, COLUMN["frames"]] = 40.0  # a bound of 40 frames
    mean = torch.tensor([[0.0, 0, 0, 0, 30.0], [0.0, 0, 0, 0, 30.0]], requires_grad=True)
    predicted = {
        "mean": mean,
        "sigma": torch.full((2, 5), 8.0),
        "success": torch.zeros(2),
    }
    loss, _ = outcome_loss(predicted, records)
    loss.backward()
    # Above the first bound (1): no push; below the second (40): pushed up.
    assert mean.grad[0, 4] == 0 and mean.grad[1, 4] < 0


def test_the_world_model_reads_only_the_scene_encoder_it_was_trained_on():
    policy = LayeredSMBPolicy()
    digest = scene_digest(policy)
    with torch.no_grad():
        next(policy.memory.parameters()).add_(1)
        next(policy.skill.parameters()).add_(1)
    assert scene_digest(policy) == digest
    with torch.no_grad():
        next(policy.scene.parameters()).add_(1)
    assert scene_digest(policy) != digest


def test_an_older_checkpoint_gets_an_untrained_world_model_and_keeps_its_layers(tmp_path):
    policy = LayeredSMBPolicy()
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith("memory.")}
    # The old scene memory's weights.
    state["memory.cell.weight_ih_l0"] = torch.zeros(512, 96)
    state["memory.enemy_prediction.weight"] = torch.zeros(30, 128)
    path = tmp_path / "v4.pt"
    torch.save(
        dict(
            settings=asdict(policy.settings),
            state_dict=state,
            token_layout=token_layout(),
            observation_layout=observation_layout(),
            trained_layers=["skill"],
            memory_version=4,
        ),
        path,
    )
    loaded, checkpoint = lt.load_layered_checkpoint(path)
    assert checkpoint["trained_layers"] == ["skill"]
    assert "world_model_reshaped" in checkpoint["load_migrations"]
    assert checkpoint["world_model"] is None
    fresh = LayeredSMBPolicy().state_dict()
    assert set(k for k in loaded.state_dict() if k.startswith("memory.")) == set(
        k for k in fresh if k.startswith("memory.")
    )


def test_a_small_world_model_learns_a_recorded_episodes_outcomes():
    from retroagi.stages.block_smb.teacher_replay import reference_vision
    from retroagi.stages.block_smb.world_model_train import _forward, episode_data, kept

    task = lt.EpisodeTask(
        index=0,
        family="enemy_hop",
        split="validation",
        seed=0,
        sample_index=0,
        difficulty="easy",
        teacher_share=1.0,
        label=True,
        record_truth=True,
        record_actions=True,
    )
    policy = LayeredSMBPolicy().eval()
    record = lt.play_episodes(reference_vision()[0], policy, "skill", [task], "cpu", 1)[0]
    data = episode_data(kept(record))
    assert data["executed"].shape == (len(record.decision_frames), len(ACTION_RECORD))
    assert (data["tried"][:, :, COLUMN["verb"]] >= 0).any()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.memory.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.memory.parameters(), lr=3e-3)
    first = float(_forward(policy, [data], "cpu"))
    for _ in range(60):
        loss = _forward(policy, [data], "cpu")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(_forward(policy, [data], "cpu")) < 0.5 * first


@torch.no_grad()
def test_the_tactic_and_strategy_read_starts_with_no_influence():
    from retroagi.core.world_model import CONTEXT_WIDTH

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    model = policy.memory
    # A world model that has learned (an untrained one's outputs start at
    # zero whatever it reads), but not yet to read the tactic and strategy.
    for parameter in model.parameters():
        parameter.normal_(0.0, 0.1)
    model.context.weight.zero_()
    model.context.bias.zero_()
    tokens, present, sprites, mario, feedback = scene(policy, 2)
    verbs = torch.tensor([VERBS.index("wait"), VERBS.index("run")])
    pointers = torch.tensor([MARIO_TOKEN, MARIO_TOKEN])
    state = model.initial(2, "cpu")
    plain, _ = model.predict(tokens, present, sprites, mario, feedback, verbs, pointers, state)
    context = torch.rand(2, CONTEXT_WIDTH)
    read, _ = model.predict(tokens, present, sprites, mario, feedback, verbs, pointers, state, context=context)
    assert torch.equal(plain["mean"], read["mean"])
    # Once it has learned, the tactic and the strategy change the outcome.
    model.context.weight.normal_()
    changed, _ = model.predict(tokens, present, sprites, mario, feedback, verbs, pointers, state, context=context)
    assert not torch.equal(plain["mean"], changed["mean"])


def test_a_world_model_saved_before_it_read_the_tactic_loads_unchanged(tmp_path):
    from dataclasses import asdict

    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith("memory.context.")}
    path = tmp_path / "before.pt"
    torch.save(
        {
            "settings": asdict(policy.settings),
            "state_dict": state,
            "observation_layout": observation_layout(),
            "token_layout": token_layout(),
            "trained_layers": ["skill"],
            "memory_version": 5,
        },
        path,
    )
    loaded, checkpoint = load_layered_checkpoint(path)
    assert "world_model_reads_tactic_and_strategy" in checkpoint["load_migrations"]
    assert checkpoint["trained_layers"] == ["skill"]
    assert torch.equal(loaded.memory.cell.weight_ih, policy.memory.cell.weight_ih)
    assert not loaded.memory.context.weight.any()
