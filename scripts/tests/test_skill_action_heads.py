"""The skill's action heads: a verb (run, jump or wait), then the object a
run or a jump ends on."""

import torch

from retroagi.core.action_tokens import POINTER_TOKENS, VERBS, pointer_index
from retroagi.core.layered_policy import POINTER_START, LayeredSMBPolicy, PolicySettings
from retroagi.core.smb_observer import C_SPANS, SEQ_LEN_A, SEQ_LEN_B, SEQ_LEN_C
from retroagi.core.tokens import encode_tactic, tactic_token


def scene(batch=2, enemy=(True, False), surface=(True, True)):
    """Rows with Mario and (per row) surface slot 0 and enemy slot 0."""
    a = torch.zeros(batch, SEQ_LEN_A, dtype=torch.long)
    b = torch.zeros(batch, SEQ_LEN_B, dtype=torch.long)
    c = torch.zeros(batch, SEQ_LEN_C)
    c[:, C_SPANS["c_mario"][0]] = 1
    for row, (has_enemy, has_surface) in enumerate(zip(enemy, surface)):
        c[row, C_SPANS["c_enemies"][0]] = float(has_enemy)
        c[row, C_SPANS["c_surfaces"][0]] = float(has_surface)
    return a, b, c


def run(policy, rows, picks=None):
    encoded = policy.encode_scene(rows)
    tactic = torch.stack([encode_tactic(tactic_token("advance"))] * len(rows[0]))
    return policy.run_skill(encoded, tactic, picks=picks)


def test_verbs_and_objects_are_masked_by_what_is_in_view():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings()).eval()
    # Row 1 sees nothing to end on: a run or a jump needs something in view;
    # a wait is always a choice.
    out = run(policy, scene(enemy=(True, False), surface=(True, False)))
    usable = out["verb"] > -1e3
    for verb in ("run", "jump"):
        assert usable[0, VERBS.index(verb)] and not usable[1, VERBS.index(verb)]
    assert usable[:, VERBS.index("wait")].all()
    assert out["pointer"].shape == (2, POINTER_TOKENS)
    enemy = pointer_index(("enemies", 0)) - POINTER_START
    surface = pointer_index(("surfaces", 0)) - POINTER_START
    # A run or a jump ends on a surface or on an enemy (a stomp) in view.
    for verb in ("run", "jump"):
        logits = run(
            policy,
            scene(),
            picks={"verb": torch.full((2,), VERBS.index(verb))},
        )["pointer"]
        assert (logits[0] > -1e3).nonzero().flatten().tolist() == sorted([surface, enemy])
        assert (logits[1] > -1e3).nonzero().flatten().tolist() == [surface]


def test_action_heads_learn_the_teachers_action():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings())
    rows = scene()
    verbs = torch.tensor([VERBS.index("jump"), VERBS.index("run")])
    pointers = (
        torch.tensor([pointer_index(("enemies", 0)), pointer_index(("surfaces", 0))])
        - POINTER_START
    )
    optimizer = torch.optim.Adam(policy.skill.parameters(), lr=3e-3)
    for _ in range(60):
        out = run(policy, rows, picks={"verb": verbs})
        loss = torch.nn.functional.cross_entropy(
            out["verb"], verbs
        ) + torch.nn.functional.cross_entropy(out["pointer"], pointers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    picks = run(policy, rows)["picks"]
    assert picks["verb"].tolist() == verbs.tolist()
    assert picks["pointer"].tolist() == pointers.tolist()


@torch.no_grad()
def test_the_decision_layers_read_motion_from_no_influence():
    from retroagi.core.smb_kinematics import MARIO_STATE, SPRITE_SLOTS, SPRITE_STATE
    from retroagi.core.tokens import EXECUTION_WIDTH, encode_strategy, DEFAULT_STRATEGY

    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings()).eval()
    rows = scene()
    encoded = policy.encode_scene(rows)
    tactic = torch.stack([encode_tactic(tactic_token("advance"))] * 2)
    sprites = torch.rand(2, SPRITE_SLOTS, len(SPRITE_STATE))
    sprites[..., 0] = 1.0  # tracked
    motion = (sprites, torch.rand(2, len(MARIO_STATE)))
    still = policy.run_skill(encoded, tactic)["verb"]
    assert torch.equal(still, policy.run_skill(encoded, tactic, motion=motion)["verb"])
    policy.skill.sprite_motion.weight.normal_()
    policy.skill.mario_motion.weight.normal_()
    assert not torch.equal(still, policy.run_skill(encoded, tactic, motion=motion)["verb"])
    # The tactic layer too, and it reads how the last action went.
    switch = encode_strategy(DEFAULT_STRATEGY).expand(2, -1)
    from retroagi.core.layered_policy import held_features

    held = held_features(None).expand(2, -1)
    memory = torch.zeros(2, policy.tactic_memory.width)
    feedback = torch.rand(2, EXECUTION_WIDTH)
    base = policy.tactic_outputs(encoded, switch, held, memory)["tactic"]
    same = policy.tactic_outputs(encoded, switch, held, memory, motion=motion, feedback=feedback)["tactic"]
    assert torch.equal(base, same)
    policy.tactic.feedback.weight.normal_()
    moved = policy.tactic_outputs(encoded, switch, held, memory, feedback=feedback)["tactic"]
    assert not torch.equal(base, moved)


def test_the_skill_learns_the_teachers_actions_in_a_tactic_episode():
    # LayeredTrainConfig.skill_learns (user, 2026-10-22: "the skills should
    # continue learning during the tactic training").
    from retroagi.stages.block_smb import layered_train as lt
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    task = lt.EpisodeTask(
        index=0, family="speed_run_advance", split="validation", seed=0, sample_index=0,
        difficulty="easy", teacher_share=1.0, label=True, teacher_skill=True,
    )
    torch.manual_seed(0)
    policy = LayeredSMBPolicy(PolicySettings())
    (record,) = lt.play_episodes(reference_vision()[0], policy.eval(), "tactic", [task], "cpu", 1)
    d = lt.tactic_episode_decisions([record], "cpu")
    assert d is not None and d["labelled"].all()  # every action the teacher's
    assert (d["label_verb"] == torch.as_tensor(record.action_verb, dtype=torch.long)).all()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.skill.parameters():
        parameter.requires_grad_(True)
    policy.train()
    optimizer = torch.optim.Adam(policy.skill.parameters(), lr=3e-3)
    first, _ = lt.skill_losses(policy, [record], "cpu")
    for _ in range(40):
        losses, stats = lt.skill_losses(policy, [record], "cpu")
        optimizer.zero_grad()
        sum(losses.values()).backward()
        optimizer.step()
    assert float(sum(losses.values())) < 0.5 * float(sum(first.values()))
    assert stats["verb_accuracy"] == 1.0
