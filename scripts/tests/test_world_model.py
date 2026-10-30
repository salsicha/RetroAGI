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
from retroagi.core.smb_kinematics import MARIO_STATE, SPRITE_SLOTS
from retroagi.core.smb_tracked_objects import CAPACITY, COLUMN as OBJECT, FIELDS
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
    objects = torch.zeros(count, CAPACITY, len(FIELDS))
    mario = torch.zeros(count, len(MARIO_STATE))
    feedback = torch.zeros(count, EXECUTION_WIDTH)
    return tokens, present, objects, mario, feedback


@torch.no_grad()
def test_an_untrained_world_model_is_unsure():
    policy = LayeredSMBPolicy().eval()
    tokens, present, objects, mario, feedback = scene(policy)
    verbs = torch.tensor([VERBS.index("wait"), VERBS.index("run")])
    pointers = torch.tensor([MARIO_TOKEN, MARIO_TOKEN])
    out, state = policy.memory.predict(
        tokens, present, objects, mario, feedback, verbs, pointers, policy.memory.initial(2, "cpu")
    )
    assert torch.allclose(out["mean"][:, :4], torch.zeros(2, 4))
    assert torch.allclose(out["mean"][:, 4], torch.full((2,), 32.0))
    assert torch.allclose(torch.sigmoid(out["success"]), torch.full((2,), 0.5))
    assert state[0].shape == (2, policy.memory.width)


@torch.no_grad()
def test_trying_another_action_does_not_advance_the_state():
    policy = LayeredSMBPolicy().eval()
    model = policy.memory
    tokens, present, objects, mario, feedback = scene(policy, 1)
    tokens, present = model.with_objects(tokens, present, objects, mario)
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


@pytest.fixture(scope="module")
def enemy_hop_episode():
    """One enemy_hop episode the teacher played, recorded for the world model."""
    from retroagi.stages.block_smb.teacher_replay import reference_vision

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
    return lt.play_episodes(reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1)[0]


def test_a_small_world_model_learns_a_recorded_episodes_outcomes(enemy_hop_episode):
    from retroagi.stages.block_smb.world_model_train import _forward, episode_data, kept

    policy = LayeredSMBPolicy().eval()
    record = enemy_hop_episode
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


def test_episodes_are_learnt_joined_into_long_runs_of_actions(enemy_hop_episode):
    # User, 2026-10-30: training episodes are a few actions long; in 1-1 the
    # world model, its memory carried past anything it had learnt, sent the
    # run back from the fourth pipe into it.
    from retroagi.stages.block_smb.world_model_train import (
        WorldModelConfig,
        _forward,
        batches,
        chained,
        episode_data,
        kept,
    )

    data = episode_data(kept(enemy_hop_episode))
    narrow = dict(data, tried=data["tried"][:, :1])  # fewer tried rows a decision
    episodes = [data, narrow, data, data]
    n = len(data["executed"])
    runs = chained(episodes, 2 * n)
    assert [len(r["executed"]) for r in runs] == [2 * n, 2 * n]
    joined = runs[0]
    assert joined["tried"].shape[1] == data["tried"].shape[1]
    # The added tried rows name no action: no loss.
    assert (joined["tried"][n:, 1:, COLUMN["verb"]] == -1).all()
    assert all(len(rows) == 2 * n for rows in joined["rows"])
    assert torch.isfinite(_forward(LayeredSMBPolicy().eval(), runs, "cpu"))
    settings = WorldModelConfig(chain_actions=2 * n, batch_chains=1)
    assert [len(b) for b in batches(episodes, settings)] == [1, 1]


@torch.no_grad()
def test_the_world_model_reads_no_tactic_and_says_whether_a_run_keeps_its_speed():
    # User, 2026-10-28: it read where a jump ends from the tactic it was
    # chosen under; and every run it carried out braked (no run-ups).
    import inspect

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    model = policy.memory
    assert not hasattr(model, "context")
    assert "context" not in inspect.signature(model.predict).parameters
    tokens, present, objects, mario, feedback = scene(policy, 2)
    verbs = torch.tensor([VERBS.index("run"), VERBS.index("jump")])
    pointers = torch.tensor([MARIO_TOKEN, MARIO_TOKEN])
    out, _ = model.predict(tokens, present, objects, mario, feedback, verbs, pointers, model.initial(2, "cpu"))
    assert out["carry"].shape == (2,)
    # Untrained, a run brakes (about a third of the teacher's runs are run-ups).
    assert (torch.sigmoid(out["carry"]) < 0.5).all()


def test_a_world_model_saved_reading_the_tactic_loads_without_it(tmp_path):
    from dataclasses import asdict

    from retroagi.core.world_model import CONTEXT_WIDTH, OUTCOME_WIDTH
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    state = dict(policy.state_dict())
    width = state["memory.verb.weight"].shape[1]
    # As saved before: the tactic and strategy read, no run-up output.
    state["memory.context.weight"] = torch.randn(width, CONTEXT_WIDTH)
    state["memory.context.bias"] = torch.randn(width)
    state["memory.out.weight"] = torch.randn(2 * OUTCOME_WIDTH + 1, width)
    state["memory.out.bias"] = torch.randn(2 * OUTCOME_WIDTH + 1)
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
    assert {"world_model_reads_no_tactic", "world_model_predicts_run_ups"} <= set(checkpoint["load_migrations"])
    assert checkpoint["trained_layers"] == ["skill"]
    assert torch.equal(loaded.memory.cell.weight_ih, policy.memory.cell.weight_ih)
    # Every outcome it predicted is kept; the run-up output starts untrained.
    assert torch.equal(loaded.memory.out.weight[: 2 * OUTCOME_WIDTH + 1], state["memory.out.weight"])
    assert not loaded.memory.out.weight[-1].any()


@torch.no_grad()
def test_the_world_model_reads_the_list_in_place_of_visions_enemy_tokens():
    from retroagi.core.world_model import SPRITE_TOKENS, object_pointers

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    model = policy.memory
    for parameter in model.parameters():
        parameter.normal_(0.0, 0.1)
    tokens, present, objects, mario, feedback = scene(policy, 1)
    present[:, SPRITE_TOKENS] = True
    # A walker out of sight, kept where it vanished: 40 pixels ahead.
    objects[0, 3, OBJECT["present"]] = 1.0
    objects[0, 3, OBJECT["x"]] = 40.0
    objects[0, 3, OBJECT["unseen"]] = 0.5
    objects[0, 3, OBJECT["walker"]] = 1.0
    read, listed = model.with_objects(tokens, present, objects, mario)
    count = tokens.shape[1]
    assert read.shape[1] == count + CAPACITY
    assert not listed[0, SPRITE_TOKENS].any()  # vision's enemy and platform tokens: not read
    assert listed[0, count:].tolist() == [row == 3 for row in range(CAPACITY)]
    # An action at the enemy in sprite slot 1 points at its row on the list.
    slot_keys = np.zeros((1, SPRITE_SLOTS), np.int32)
    slot_keys[0, 1] = 7
    keys = np.zeros((1, CAPACITY), np.int32)
    keys[0, 3] = 7
    assert object_pointers([SPRITE_TOKENS[1]], slot_keys, keys, count)[0] == count + 3
    assert object_pointers([SPRITE_TOKENS[0]], slot_keys, keys, count)[0] == SPRITE_TOKENS[0]
    # What the list says changes the outcome.
    verbs, pointers = torch.tensor([VERBS.index("wait")]), torch.tensor([MARIO_TOKEN])
    state = model.initial(1, "cpu")
    near, _ = model.predict(tokens, present, objects, mario, feedback, verbs, pointers, state)
    objects[0, 3, OBJECT["x"]] = 120.0
    far, _ = model.predict(tokens, present, objects, mario, feedback, verbs, pointers, state)
    assert not torch.equal(near["mean"], far["mean"])


def test_a_world_model_saved_before_the_list_learns_again(tmp_path):
    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    state = {k: v for k, v in policy.state_dict().items() if not k.startswith(("memory.objects.", "memory.object_row"))}
    state["memory.sprite_motion.weight"] = torch.zeros(policy.memory.verb.weight.shape[1], 3)
    state["memory.sprite_motion.bias"] = torch.zeros(policy.memory.verb.weight.shape[1])
    path = tmp_path / "before.pt"
    torch.save(
        {
            "settings": asdict(policy.settings),
            "state_dict": state,
            "observation_layout": observation_layout(),
            "token_layout": token_layout(),
            "trained_layers": ["skill"],
            "memory_version": 5,
            "world_model": {"reads": scene_digest(policy), "history": []},
        },
        path,
    )
    loaded, checkpoint = lt.load_layered_checkpoint(path)
    assert "world_model_reads_the_tracked_list" in checkpoint["load_migrations"]
    assert checkpoint["world_model"] is None and checkpoint["trained_layers"] == ["skill"]
    assert torch.equal(loaded.memory.cell.weight_ih, policy.memory.cell.weight_ih)


def test_a_teachers_wait_is_recorded_aimed_at_nothing_and_counts_when_survived():
    from retroagi.core.action_outcomes import COLUMN as EXECUTED, executed_outcomes
    from retroagi.core.action_tokens import executed_pointer
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    # Recorded before 2026-10-21: a teacher's wait stored with slot 0.
    assert executed_pointer(VERBS.index("wait"), 0) == -1
    assert executed_pointer(VERBS.index("jump"), 0) == 0
    task = lt.EpisodeTask(
        index=0,
        family="action_wait",
        split="validation",
        seed=0,
        sample_index=0,
        difficulty="easy",
        teacher_share=1.0,
        label=True,
        record_truth=True,
    )
    torch.manual_seed(0)
    (record,) = lt.play_episodes(reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1)
    waits = np.flatnonzero(record.action_verb == VERBS.index("wait"))
    assert len(waits) and (record.action_pointer[waits] == -1).all()
    outcomes = executed_outcomes(record, lt.END_REASONS.index("no_progress"))
    assert record.won and (outcomes[waits, EXECUTED["success"]] == 1.0).all()
    assert not outcomes[waits, EXECUTED["has_object"]].any()
