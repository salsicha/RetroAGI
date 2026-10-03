"""The game-neutral parts of the four-layer agent: reader, tokens, executor, layers."""

import numpy as np
import pytest
import torch

from retroagi.core.actions import SMBAction
from retroagi.core.layered_policy import (
    LayeredSMBPolicy,
    action_plan,
    skill_token,
)
from retroagi.core.smb_executor import FRAME_COUNTS, ActionPlan, SMBExecutor
from retroagi.core.smb_observer import (
    C_SPANS,
    CODE,
    SCENE_SLOTS,
    SEQ_LEN_C,
    column_codes,
    pack_c,
    policy_input,
)
from retroagi.core.smb_scene_labels import (
    EnemyView,
    Gap,
    MarioView,
    SceneObservation,
    Surface,
)
from retroagi.core.tokens import (
    NO_TARGET,
    SKILL_WIDTH,
    TARGETS,
    SkillToken,
    StrategyToken,
    encode_skill,
)


def scene(
    mario=(100, 192, 112, 208),
    support="ground",
    enemies=(),
    surfaces=None,
    gaps=(),
    on_something=None,
):
    return SceneObservation(
        mario=MarioView(
            box=mario,
            facing_right=True,
            support=support,
            on_something=support != "air" if on_something is None else on_something,
        ),
        enemies=tuple(enemies),
        surfaces=tuple(surfaces if surfaces is not None else (Surface(8, 248, 208, False),)),
        gaps=tuple(gaps),
    )


# ── Reader ────────────────────────────────────────────────────────────────────


def test_positions_are_relative_to_mario_and_nothing_else_is_packed():
    row = pack_c(scene(enemies=[EnemyView((150, 198, 160, 208), "walker")]))
    assert row.shape == (SEQ_LEN_C,)
    mario = row[slice(*C_SPANS["c_mario"])]
    assert mario[0] == 1.0 and mario[6 + 1] == 1.0  # present, standing on ground
    enemy = row[slice(*C_SPANS["c_enemies"])][:9]
    assert enemy[0] == 1.0
    assert enemy[1] == pytest.approx((150 - 106) / 256)
    surface = row[slice(*C_SPANS["c_surfaces"])][:5]
    assert surface[3] == pytest.approx(0.0)  # level with Mario's feet
    assert not row[slice(*C_SPANS["c_reserved"])].any()


def test_overflowing_lists_keep_what_is_nearest_mario():
    far = [EnemyView((x, 198, x + 10, 208), "walker") for x in range(130, 250, 12)]
    row = pack_c(scene(enemies=far))
    slots = row[slice(*C_SPANS["c_enemies"])].reshape(SCENE_SLOTS["enemies"], -1)
    assert slots[:, 0].sum() == SCENE_SLOTS["enemies"]
    nearest_left = slots[:, 1].min() * 256 + 106
    assert nearest_left == pytest.approx(130)


def test_column_codes_name_the_most_important_thing_in_each_band():
    codes = column_codes(
        scene(enemies=[EnemyView((200, 198, 210, 208), "plant")], gaps=[Gap(32, 64)]), 8
    )
    assert codes[3] == CODE["mario"]
    assert codes[6] == CODE["plant"]
    assert codes[1] == CODE["gap"]
    assert codes[0] == CODE["level"]


def test_policy_input_has_the_three_rows():
    inputs = policy_input(scene())
    assert inputs.src_a.shape == (8,) and inputs.src_b.shape == (16,)
    assert inputs.src_c.shape == (SEQ_LEN_C,)


# ── Tokens ────────────────────────────────────────────────────────────────────


def test_tokens_refuse_unknown_values_and_encode_pointers():
    with pytest.raises(ValueError):
        StrategyToken("hurry")
    with pytest.raises(ValueError):
        SkillToken("jump_gap", direction=0)
    token = SkillToken("stomp", 1, ("enemies", 2))
    vector = encode_skill(token)
    assert vector.shape == (SKILL_WIDTH,)
    assert TARGETS[token.pointer] == ("enemies", 2)
    assert SkillToken("advance").pointer == NO_TARGET


# ── Executor ──────────────────────────────────────────────────────────────────


def test_the_executor_presses_its_button_for_its_frame_count():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT_JUMP, 5))
    pressed = []
    while not executor.finished:
        pressed.append(executor.press())
    assert pressed == [SMBAction.RIGHT_JUMP] * 5
    with pytest.raises(RuntimeError):
        executor.press()
    executor.end("done")
    assert executor.idle and executor.history[-1][1:] == (5, "done")


def test_every_action_takes_any_frame_count_from_1_to_32():
    assert FRAME_COUNTS == tuple(range(1, 33))
    for action in SMBAction:
        for frames in (1, 7, 15, 32):
            assert ActionPlan(action, frames).frames == frames
        for frames in (0, 33):
            with pytest.raises(ValueError):
                ActionPlan(action, frames)


def test_a_landing_is_the_land_detector_turning_on_after_mario_was_in_the_air():
    from retroagi.core.smb_agent import LandingWatch

    def watch(pictures):
        seen = LandingWatch()
        return [seen.landed(picture) for picture in pictures]

    air = scene(support="air")
    assert watch([air, scene()]) == [False, True]
    assert watch([air, scene(support="moving_platform")]) == [False, True]
    # Landing on an enemy: still in the air, but the detector says his feet are on something.
    stomp = scene(support="air", on_something=True)
    assert watch([air, stomp, air]) == [False, True, False]
    assert watch([scene(), scene()]) == [False, False]
    assert watch([scene(), air]) == [False, False]  # leaving the ground
    # Nothing but the land detector counts: an enemy under his feet does not.
    goomba = EnemyView((100, 208, 112, 218), "walker")
    assert watch([air, scene(support="air", enemies=[goomba])]) == [False, False]
    # A picture without Mario changes nothing.
    assert watch([air, scene(mario=None, support="ground"), scene()]) == [False, False, True]


class SceneList:
    """A stand-in vision transformer that reports a given scene for each frame in turn."""

    def __init__(self, pictures):
        self.pictures = list(pictures)

    def eval(self):
        return self

    def scene(self, screens):
        return [self.pictures.pop(0) for _ in screens]


def test_a_landing_ends_the_action_early_and_nothing_else_does():
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    goomba = EnemyView((112, 196, 122, 208), "walker")
    pictures = [scene(), scene(enemies=[goomba]), scene(mario=None), scene(support="air")]
    pictures += [scene(support="air"), scene()]
    agents = SMBAgents(VisionObserver(SceneList(pictures)), LayeredSMBPolicy().eval(), "cpu")
    plan = ActionPlan(SMBAction.RIGHT, 20)

    def given(copies, scenes):
        return {"skill": [SkillToken("advance")], "action": [plan]}

    screen = [np.zeros((240, 256, 3), np.uint8)]
    steps = [agents.act(screen, [0], given=given)[0] for _ in pictures]
    # Touching an enemy, Mario vanishing and leaving the ground change nothing;
    # the landing on the last frame ends the action and a new one starts.
    assert [step.ended for step in steps] == [None, None, None, None, None, "landed"]
    assert [step.decision is not None for step in steps] == [True] + [False] * 4 + [True]
    assert [step.button for step in steps] == [SMBAction.RIGHT] * 6


# ── Layers ────────────────────────────────────────────────────────────────────


def test_layers_read_only_the_inputs_and_the_token_above():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    inputs = policy_input(scene(enemies=[EnemyView((150, 198, 160, 208), "walker")]))
    rows = (inputs.src_a[None], inputs.src_b[None], inputs.src_c[None])
    with torch.no_grad():
        now = policy.encode_scene(rows)
        _, memory = policy.expect(policy.remember(now, None))
        tactic = torch.zeros(1, 5)
        tactic[0, 0] = tactic[0, -1] = 1.0
        out = policy.layer_outputs(
            rows,
            memory,
            {"skill": encode_skill(SkillToken("jump_gap"))[None], "tactic": tactic},
        )
        other = policy.layer_outputs(
            rows, memory, {"skill": encode_skill(SkillToken("stomp"))[None]}
        )
    first = {name: value[0] for name, value in out["action"].items()}
    assert isinstance(action_plan(first), ActionPlan)
    assert not torch.equal(out["action"]["action"], other["action"]["action"])
    # The action layer's output does not depend on the tactic or strategy tokens.
    with torch.no_grad():
        tactic_changed = policy.layer_outputs(
            rows,
            memory,
            {"skill": encode_skill(SkillToken("jump_gap"))[None], "tactic": torch.ones(1, 5)},
        )
    assert torch.equal(out["action"]["action"], tactic_changed["action"]["action"])
    pointer = {name: value[0] for name, value in out["skill"].items()}
    token = skill_token(pointer)
    assert token.target is None or token.target[0] in ("surfaces", "enemies", "moving_platforms")
    absent = out["skill"]["pointer"][0][: len(TARGETS)]
    enemy_slots = [i for i, (name, slot) in enumerate(TARGETS) if name == "enemies" and slot > 0]
    assert np.isneginf(absent[enemy_slots].numpy()).all()  # absent slots cannot be pointed at


# ── Agent and trainer pieces ──────────────────────────────────────────────────


class SceneEcho:
    """A stand-in vision transformer: every screen shows the same fixed scene."""

    def __init__(self, picture):
        self.picture = picture

    def eval(self):
        return self

    def scene(self, screens):
        return [self.picture] * len(screens)


def test_given_tokens_replace_the_policy_only_where_given():
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    agents = SMBAgents(VisionObserver(SceneEcho(scene())), LayeredSMBPolicy().eval(), "cpu", 2)
    screens = [np.zeros((240, 256, 3), np.uint8)] * 2
    teacher_skill = SkillToken("jump_gap", 1)
    teacher_plan = ActionPlan(SMBAction.RIGHT_JUMP, 6)

    def given(copies, scenes):
        assert copies == [0, 1] and len(scenes) == 2
        return {"skill": [teacher_skill, teacher_skill], "action": [teacher_plan, None]}

    steps = agents.act(screens, [0, 1], given=given, run_given=("action",))
    first, second = steps[0].decision, steps[1].decision
    assert first.skill == second.skill == teacher_skill
    assert first.plan == teacher_plan
    assert second.plan == second.chosen["action"]  # not given: the policy's own
    assert "action" in first.chosen  # run anyway, to compare with the given plan
    assert "skill" not in first.chosen  # given for every copy, so not run
    # Play passes nothing: every layer's token is the policy's (strategy: the default).
    agents.reset(0)
    (alone,) = agents.act(screens[:1], [0])
    assert alone.decision.strategy.kind == "speed_run" and "skill" in alone.decision.chosen


def test_jumps_are_taught_the_middle_of_the_longest_certified_run():
    from retroagi.core.smb_physics import NES_JUMP_FRAMES
    from retroagi.stages.block_smb.layered_train import jump_frame_label

    menu = list(NES_JUMP_FRAMES)
    assert jump_frame_label(menu[0], ()) == menu[0]
    assert jump_frame_label(menu[0], (menu[2], menu[3], menu[4], menu[9])) == menu[3]
    assert jump_frame_label(menu[0], (menu[9],)) == menu[9]


def _record(learner: str, frames: int = 30):
    from retroagi.core.tokens import STRATEGY_WIDTH, TACTIC_WIDTH
    from retroagi.stages.block_smb.layered_train import EpisodeRecord

    rng = np.random.default_rng(0)
    rows = [policy_input(scene(mario=(100 + t, 192, 112 + t, 208))) for t in range(frames)]
    decisions = np.array([0, 10, 20], np.int32)
    width = {"action": SKILL_WIDTH, "skill": TACTIC_WIDTH, "tactic": STRATEGY_WIDTH}[learner]
    labels = {
        "action": {"action": np.array([1, 2, 0]), "frames": np.array([3, 0, 5])},
        "skill": {
            "skill": np.array([0, 1, 2]),
            "direction": np.array([1, 1, 0]),
            "pointer": np.array([NO_TARGET, NO_TARGET, NO_TARGET]),
        },
        "tactic": {"tactic": np.array([0, 2, 0]), "direction": np.array([1, 1, 1])},
    }[learner]
    return EpisodeRecord(
        family="flat_run",
        split="train",
        sample_index=0,
        won=True,
        difficulty="easy",
        end="goal",
        src_a=np.stack([r.src_a.numpy() for r in rows]).astype(np.int8),
        src_b=np.stack([r.src_b.numpy() for r in rows]).astype(np.int8),
        src_c=np.stack([r.src_c.numpy() for r in rows]).astype(np.float16),
        buttons=rng.integers(0, 6, frames).astype(np.int8),
        decision_frames=decisions,
        targets=np.zeros((3, 5), np.float32),
        given=np.eye(width, dtype=np.float32)[[0, 1, 0]],
        labels={**labels, "valid": np.ones(3, bool)},
        played_teacher=np.ones(3, bool),
        agreed=np.zeros(3, bool),
    )


@pytest.mark.parametrize("learner", ["action", "skill", "tactic"])
def test_training_one_layer_leaves_every_other_part_unchanged(learner):
    from retroagi.stages.block_smb.layered_train import learner_losses

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    learning = policy.parameters_of(learner)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in learning:
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(learning, lr=0.1)
    losses, _ = learner_losses(policy, learner, [_record(learner)], 1.0, "cpu")
    assert ("expectation" in losses) == (learner == "action")
    optimizer.zero_grad()
    sum(losses.values()).backward()
    optimizer.step()
    owned = {id(p) for p in learning}
    for name, parameter in policy.named_parameters():
        moved = not torch.equal(before[name], parameter.detach())
        if id(parameter) not in owned:
            assert not moved, f"training {learner} changed {name}"
    assert any(not torch.equal(before[n], p.detach()) for n, p in policy.named_parameters())


def test_reward_credit_runs_back_from_the_end():
    from retroagi.stages.block_smb.layered_train import reward_advantages

    record = _record("action", frames=30)
    record.rewards = np.zeros(30, np.float32)
    record.rewards[25] = 50.0  # the goal, during the last decision's plan
    record.old_value = np.zeros(3, np.float32)
    record.terminal = True
    advantages, returns = reward_advantages(record, discount=1.0, smoothing=1.0)
    assert np.allclose(advantages, 50.0) and np.allclose(returns, 50.0)
    advantages, _ = reward_advantages(record, discount=0.9, smoothing=1.0)
    assert advantages[2] == pytest.approx(50.0 * 0.9**5)
    assert advantages[0] == pytest.approx(50.0 * 0.9**25)


def test_reward_rounds_train_from_sampled_choices():
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, learner_losses

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    record = _record("action")
    record.explored = True
    record.terminal = True
    record.rewards = np.linspace(0, 1, record.frames).astype(np.float32)
    record.old_value = np.zeros(3, np.float32)
    record.picks = {"action": np.array([1, 2, 0]), "frames": np.array([3, 0, 5])}
    record.old_log_prob = np.full(3, -3.0, np.float32)
    record.played_teacher = np.zeros(3, bool)
    config = LayeredTrainConfig(reward_rounds=1)
    losses, stats = learner_losses(policy, "action", [record], 1.0, "cpu", rl=config)
    assert {"reward", "estimate", "entropy", "action", "frames"} <= set(losses)
    sum(losses.values()).backward()
    assert policy.action.value.weight.grad is not None


# ── Position waves and the frame window ───────────────────────────────────────


def test_a_pixel_apart_is_a_clear_difference_in_the_waves():
    from retroagi.core.layered_policy import PositionWaves

    waves = PositionWaves((1,), frequencies=8)
    here = torch.tensor([[1.0, 0.25]])
    pixel_on = torch.tensor([[1.0, 0.25 + 1 / 256]])
    raw = (pixel_on - here).abs().max()
    wave = (waves(pixel_on) - waves(here)).abs().max()
    assert raw < 0.005 and wave > 0.5  # the shortest wave (4 pixels) turns a quarter
    assert waves(here).shape == (1, 2 + 2 * 8)


def _rows(xs):
    picked = [
        policy_input(
            scene(mario=(100, 192, 112, 208), enemies=[EnemyView((x, 198, x + 10, 208), "walker")])
        )
        for x in xs
    ]
    return [(p.src_a[None], p.src_b[None], p.src_c[None]) for p in picked]


def test_the_memory_reads_only_the_latest_picture():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    # It takes the latest scene's summary and nothing else: no action, no button.
    assert policy.memory.cell.input_size == policy.settings.width
    with torch.no_grad():
        frames = [policy.encode_scene(rows) for rows in _rows([150, 160])]
        near = policy.remember(frames[0], None)
        far = policy.remember(frames[1], None)
        later = policy.remember(frames[1], near)
        expected, _ = policy.expect(near)
    assert not torch.allclose(near.hidden, far.hidden)
    # What it saw at an earlier action stays in its state.
    assert not torch.allclose(far.hidden, later.hidden)
    assert expected.shape == (1, SEQ_LEN_C)


def test_replaying_an_episode_gives_the_memory_play_gave():
    from retroagi.stages.block_smb.layered_train import action_memory

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    frames = _rows([150 + 3 * t for t in range(12)])
    starts = [0, 1, 5, 9]  # the frames on which actions started
    with torch.no_grad():
        state, played = None, []
        for t in starts:
            state = policy.remember(policy.encode_scene(frames[t]), state)
            played.append(state.hidden[0])
        a, b, c = (torch.cat([r[i] for r in frames]).unsqueeze(0) for i in range(3))
        d = {
            "episode": torch.zeros(len(starts), dtype=torch.long),
            "frame": torch.tensor(starts),
        }
        replayed = action_memory(policy, a, b, c, d)
    assert torch.allclose(torch.stack(played), replayed, atol=1e-5)


# ── Each layer's own previous choices ─────────────────────────────────────────


def test_a_layer_reads_its_own_previous_choices():
    from retroagi.core.layered_policy import CHOICE_WIDTH, HISTORY, encode_choice

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    inputs = policy_input(scene(enemies=[EnemyView((150, 198, 160, 208), "walker")]))
    rows = (inputs.src_a[None], inputs.src_b[None], inputs.src_c[None])
    tactic = torch.zeros(1, 5)
    tactic[0, 0] = tactic[0, -1] = 1.0

    def skill_scores(previous):
        choices = torch.zeros(1, HISTORY, CHOICE_WIDTH["skill"])
        present = torch.zeros(1, HISTORY, dtype=torch.bool)
        for age, token in enumerate(previous):
            choices[0, age] = encode_choice("skill", token)
            present[0, age] = True
        with torch.no_grad():
            now = policy.encode_scene(rows)
            _, expected = policy.expect(policy.remember(now, None))
            out = policy.layer_outputs(
                rows, expected, {"tactic": tactic}, {"skill": (choices, present)}
            )
        return out["skill"]["skill"]

    nothing = skill_scores([])
    gap = skill_scores([SkillToken("jump_gap")] * 3)
    enemy = skill_scores([SkillToken("stomp", 1)] * 3)
    assert not torch.allclose(nothing, gap) and not torch.allclose(gap, enemy)


def test_the_agent_keeps_its_last_16_used_choices_newest_first():
    from retroagi.core.layered_policy import HISTORY, encode_choice
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    agents = SMBAgents(VisionObserver(SceneEcho(scene())), LayeredSMBPolicy().eval(), "cpu")
    plans = [ActionPlan(SMBAction.RIGHT, 1 + i % 32) for i in range(20)]
    turn = iter(plans)

    def given(copies, scenes):
        return {"skill": [SkillToken("advance")], "action": [next(turn)]}

    screen = [np.zeros((240, 256, 3), np.uint8)]
    while True:
        try:
            agents.act(screen, [0], given=given)
        except StopIteration:
            break
        while not agents.copies[0].executor.finished:
            agents.copies[0].executor.press()
    kept = agents.copies[0].choices["action"]
    assert len(kept) == HISTORY
    expected = [encode_choice("action", plan) for plan in reversed(plans)][:HISTORY]
    assert all(torch.equal(a, b) for a, b in zip(kept, expected))


def test_training_rebuilds_each_decisions_history_within_its_episode():
    from retroagi.core.layered_policy import HISTORY
    from retroagi.stages.block_smb.layered_train import choice_history

    used = torch.arange(7, dtype=torch.float32).unsqueeze(-1)  # one number per decision
    episode = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    choices, exists = choice_history(used, episode)
    assert choices.shape == (7, HISTORY, 1)
    # The third decision of episode 0 sees the second, then the first.
    assert exists[2].tolist()[:3] == [True, True, False]
    assert choices[2, :2, 0].tolist() == [1.0, 0.0]
    # Episode 1's first decision sees nothing from episode 0.
    assert not exists[3].any()
    assert choices[6, :3, 0].tolist() == [5.0, 4.0, 3.0] and not exists[6, 3]


def test_a_remembered_skill_target_is_where_it_was_not_which_slot():
    from retroagi.core.layered_policy import SKILL_HISTORY_WIDTH, encode_choice
    from retroagi.core.smb_observer import target_row

    here = scene(enemies=[EnemyView((150, 198, 160, 208), "walker")])
    box = (150, 198, 160, 208)
    first = encode_choice("skill", SkillToken("stomp", 1, ("enemies", 0)), target_row(here, box))
    other_slot = encode_choice(
        "skill", SkillToken("stomp", 1, ("enemies", 4)), target_row(here, box)
    )
    elsewhere = encode_choice(
        "skill",
        SkillToken("stomp", 1, ("enemies", 0)),
        target_row(here, (180, 198, 190, 208)),
    )
    assert first.shape == (SKILL_HISTORY_WIDTH,)
    assert torch.equal(first, other_slot)  # the slot number is not remembered
    assert not torch.equal(first, elsewhere)  # where the target was is
    assert first[-4].item() == pytest.approx((150 - 106) / 256)  # its left edge, from Mario
    torch.manual_seed(0)
    layer = LayeredSMBPolicy().skill
    assert layer.history.in_features == SKILL_HISTORY_WIDTH + 2 * 4 * 8  # box edges also as waves


def test_the_skills_are_the_seven_moves_and_avoiding_enemies_is_none_of_them():
    from retroagi.core.tokens import SKILLS
    from retroagi.stages.block_smb.teacher_tokens import OBJECTIVE_SKILLS

    assert SKILLS == ("advance", "jump_gap", "climb", "descend", "stomp", "retreat", "wait")
    assert OBJECTIVE_SKILLS["enemy"] == "advance"  # getting past an enemy is advancing
    assert set(OBJECTIVE_SKILLS.values()) <= set(SKILLS)


def test_waiting_for_a_moving_platform_is_holding_the_area_and_waiting():
    from retroagi.core.smb_scene_labels import scene_from_labels
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_tokens import (
        episode_teacher,
        teacher_skill,
        teacher_tactic,
    )

    scenario = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=3, sample_index=0, family="bridge_wait", difficulty="hard"
    ).scenario
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    teacher = episode_teacher(scenario)
    tactic = teacher_tactic(env, teacher)
    skill = teacher_skill(env, scene_from_labels(env.scene_labels()), teacher, tactic)
    env.close()
    assert (tactic.stance, skill.kind) == ("hold_area", "wait")
