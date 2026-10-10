"""The skill chooses its command in order, and its destination is taught with a
loss that grows with the pixel error.

The mode comes first, then the destination's x reading the chosen mode, then
its y reading both, so one command's mode or x is never paired with another
command's x or y.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy, choice_log_prob, choose
from retroagi.core.smb_observer import policy_input
from retroagi.core.smb_scene_labels import EnemyView, MarioView, SceneObservation, Surface
from retroagi.core.tokens import SKILL_MODES, SKILL_X, SKILL_Y, encode_tactic, tactic_token


def inputs(policy):
    scene = SceneObservation(
        mario=MarioView(
            box=(100, 196, 110, 208), facing_right=True, support="ground", on_something=True
        ),
        enemies=(EnemyView((150, 194, 162, 208), "walker"),),
        surfaces=(Surface(8, 248, 208, False),),
    )
    rows = policy_input(scene)
    now = policy.encode_scene((rows.src_a[None], rows.src_b[None], rows.src_c[None]))
    _, expected = policy.expect(policy.remember(now, None))
    return now, expected, encode_tactic(tactic_token("advance"))[None]


def picks(mode, x, y=0):
    return {
        "mode": torch.tensor([SKILL_MODES.index(mode)]),
        "x": torch.tensor([SKILL_X.index(x)]),
        "y": torch.tensor([SKILL_Y.index(y)]),
    }


def test_x_reads_the_chosen_mode_and_y_reads_the_mode_and_x():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    now, expected, tactic = inputs(policy)
    with torch.no_grad():
        run = policy.run_skill(now, expected, tactic, picks=picks("run", 72))
        jump = policy.run_skill(now, expected, tactic, picks=picks("jump", 72))
        near = policy.run_skill(now, expected, tactic, picks=picks("jump", 13))
        # Untrained, what was chosen has no influence yet (a warm start keeps
        # the outputs of the skill that chose the three apart).
        assert torch.equal(run["x"], jump["x"]) and torch.equal(jump["y"], near["y"])
        for given in (policy.skill.given_mode, policy.skill.given_mode_and_x):
            given.net[-1].weight.normal_(std=0.5)
        run = policy.run_skill(now, expected, tactic, picks=picks("run", 72))
        jump = policy.run_skill(now, expected, tactic, picks=picks("jump", 72))
        near = policy.run_skill(now, expected, tactic, picks=picks("jump", 13))
    assert not torch.allclose(run["x"], jump["x"])  # x depends on the mode
    assert not torch.allclose(jump["y"], near["y"])  # y depends on x
    assert torch.equal(run["mode"], jump["mode"])  # the mode reads neither


@pytest.mark.parametrize("sample", [False, True])
def test_the_chosen_command_is_the_one_the_conditional_outputs_were_read_with(sample):
    torch.manual_seed(1)
    policy = LayeredSMBPolicy().eval()
    for given in (policy.skill.given_mode, policy.skill.given_mode_and_x):
        given.net[-1].weight.data.normal_(std=0.5)
    now, expected, tactic = inputs(policy)
    with torch.no_grad():
        out = policy.run_skill(now, expected, tactic, sample=sample)
        token, chosen = choose(
            "skill",
            {k: v[0] if k != "picks" else {h: p[0] for h, p in v.items()} for k, v in out.items()},
        )
        assert chosen == {head: int(out["picks"][head][0]) for head in ("mode", "x", "y")}
        again = policy.run_skill(now, expected, tactic, picks=out["picks"])
        assert torch.allclose(again["x"], out["x"]) and torch.allclose(again["y"], out["y"])
        if not sample:
            assert chosen["mode"] == int(out["mode"].argmax())
            assert chosen["x"] == int(out["x"].argmax()) and chosen["y"] == int(out["y"].argmax())
        stacked = {head: torch.tensor([value]) for head, value in chosen.items()}
        log_prob, _ = choice_log_prob("skill", out, stacked)
        expected_log_prob = sum(
            out[head].log_softmax(-1)[0, chosen[head]] for head in ("mode", "x", "y")
        )
    assert float(log_prob) == pytest.approx(float(expected_log_prob), abs=1e-5)
    assert token.mode == SKILL_MODES[chosen["mode"]]


def test_a_checkpoint_from_before_loads_with_the_same_outputs(tmp_path):
    from retroagi.stages.block_smb.layered_train import (
        LayeredTrainConfig,
        load_layered_checkpoint,
        save_layered_checkpoint,
    )

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    path = tmp_path / "old.pt"
    save_layered_checkpoint(path, policy, LayeredTrainConfig(), ["skill"], [])
    saved = torch.load(path, weights_only=False)
    for name in list(saved["state_dict"]):
        if name.startswith(("skill.given_mode.", "skill.given_mode_and_x.")):
            del saved["state_dict"][name]
    torch.save(saved, path)
    loaded, checkpoint = load_layered_checkpoint(path)
    assert "skill_ordered_choice_zero_initialized" in checkpoint["load_migrations"]
    now, expected, tactic = inputs(loaded.eval())
    with torch.no_grad():
        a = loaded.run_skill(now, expected, tactic, picks=picks("run", 4))
        b = loaded.run_skill(now, expected, tactic, picks=picks("jump", 60, -10))
    assert torch.equal(a["x"], b["x"]) and torch.equal(a["y"], b["y"])


def test_the_destination_loss_grows_with_the_pixel_error():
    from retroagi.stages.block_smb.layered_train import position_loss

    pixels = torch.tensor(SKILL_X, dtype=torch.float32)
    label = torch.tensor([SKILL_X.index(13)])

    def loss(centre):
        logits = (-0.5 * ((pixels - centre) / 2.0) ** 2)[None]
        return float(position_loss(logits, label, "x"))

    losses = [loss(c) for c in (13, 14, 16, 30, 70)]
    assert losses == sorted(losses)
    assert losses[1] - losses[0] < 0.25  # one pixel off is nearly free
    assert losses[-1] > 100 * losses[1]  # 57 pixels off is expensive
    # The loss of y uses y's pixels.
    y_label = torch.tensor([SKILL_Y.index(-10)])
    y_pixels = torch.tensor(SKILL_Y, dtype=torch.float32)
    y_logits = (-0.5 * ((y_pixels + 10) / 2.0) ** 2)[None]
    assert float(position_loss(y_logits, y_label, "y")) < 0.2
