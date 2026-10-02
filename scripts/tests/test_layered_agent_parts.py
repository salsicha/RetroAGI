"""The game-neutral parts of the four-layer agent: reader, tokens, executor, layers."""

import numpy as np
import pytest
import torch

from retroagi.core.actions import SMBAction
from retroagi.core.layered_policy import (
    LayeredSMBPolicy,
    absent_like,
    action_plan,
    skill_token,
)
from retroagi.core.smb_executor import (
    JUMP_FRAMES,
    STEADY_FRAMES,
    TAKEOFF_GRACE,
    ActionPlan,
    SMBExecutor,
)
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


def scene(mario=(100, 192, 112, 208), support="ground", enemies=(), surfaces=None, gaps=()):
    return SceneObservation(
        mario=MarioView(box=mario, facing_right=True, support=support),
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
        SkillToken("clear_gap", direction=0)
    token = SkillToken("enemy_clear", 1, True, ("enemies", 2))
    vector = encode_skill(token)
    assert vector.shape == (SKILL_WIDTH,)
    assert TARGETS[token.pointer] == ("enemies", 2)
    assert SkillToken("advance").pointer == NO_TARGET


# ── Executor ──────────────────────────────────────────────────────────────────


def play(executor, scenes):
    pressed, reasons = [], []
    for picture in scenes:
        reason = executor.ended(picture)
        if reason is not None:
            reasons.append(reason)
            break
        pressed.append(executor.press())
    return pressed, reasons


def test_a_walk_lasts_its_frame_count():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT, 6), scene())
    pressed, reasons = play(executor, [scene()] * 20)
    assert pressed == [SMBAction.RIGHT] * 6 and reasons == ["done"]
    assert executor.idle


def test_a_jump_holds_then_keeps_direction_until_the_vision_reports_landing():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT_JUMP, 4), scene())
    pictures = [scene()] * 2 + [scene(support="air")] * 10 + [scene()] * 3
    pressed, reasons = play(executor, pictures)
    assert pressed[:4] == [SMBAction.RIGHT_JUMP] * 4
    assert set(pressed[4:]) == {SMBAction.RIGHT}
    assert reasons == ["landed"] and len(pressed) == 12


def test_a_jump_that_never_leaves_the_ground_ends_after_a_grace():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.JUMP, 2), scene())
    pressed, reasons = play(executor, [scene()] * 20)
    assert reasons == ["landed"] and len(pressed) == 2 + TAKEOFF_GRACE


def test_enemy_contact_and_walking_off_an_edge_interrupt():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT, 32), scene())
    touching = scene(enemies=[EnemyView((112, 196, 122, 208), "walker")])
    pressed, reasons = play(executor, [scene(), scene(), touching])
    assert reasons == ["enemy_contact"] and len(pressed) == 2
    executor.start(ActionPlan(SMBAction.RIGHT, 32), scene())
    pressed, reasons = play(executor, [scene(), scene(support="air")])
    assert reasons == ["left_ground"]
    executor.start(ActionPlan(SMBAction.NOOP, 32), scene())
    defeated = scene(enemies=[EnemyView((112, 196, 122, 208), "defeated")])
    pressed, reasons = play(executor, [scene(), defeated, scene(mario=None)])
    assert reasons == ["mario_missing"]


def test_plans_must_use_the_frame_menus():
    assert ActionPlan(SMBAction.RIGHT_JUMP, JUMP_FRAMES[-1]).frames == 32
    assert ActionPlan(SMBAction.NOOP, STEADY_FRAMES[-1]).frames == 96
    with pytest.raises(ValueError):
        ActionPlan(SMBAction.RIGHT_JUMP, 15)


# ── Layers ────────────────────────────────────────────────────────────────────


def test_layers_read_only_the_inputs_and_the_token_above():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    inputs = policy_input(scene(enemies=[EnemyView((150, 198, 160, 208), "walker")]))
    rows = (inputs.src_a[None], inputs.src_b[None], inputs.src_c[None])
    with torch.no_grad():
        now = policy.encode_scene(rows)
        _, memory = policy.expect(policy.remember(now, absent_like(now), None))
        tactic = torch.zeros(1, 5)
        tactic[0, 0] = tactic[0, -1] = 1.0
        out = policy.layer_outputs(
            rows,
            memory,
            {"skill": encode_skill(SkillToken("clear_gap"))[None], "tactic": tactic},
        )
        other = policy.layer_outputs(
            rows, memory, {"skill": encode_skill(SkillToken("enemy_clear"))[None]}
        )
    first = {name: value[0] for name, value in out["action"].items()}
    assert isinstance(action_plan(first), ActionPlan)
    assert not torch.equal(out["action"]["action"], other["action"]["action"])
    # The action layer's output does not depend on the tactic or strategy tokens.
    with torch.no_grad():
        tactic_changed = policy.layer_outputs(
            rows,
            memory,
            {"skill": encode_skill(SkillToken("clear_gap"))[None], "tactic": torch.ones(1, 5)},
        )
    assert torch.equal(out["action"]["action"], tactic_changed["action"]["action"])
    pointer = {name: value[0] for name, value in out["skill"].items()}
    token = skill_token(pointer)
    assert token.target is None or token.target[0] in ("surfaces", "enemies", "moving_platforms")
    absent = out["skill"]["pointer"][0][: len(TARGETS)]
    enemy_slots = [i for i, (name, slot) in enumerate(TARGETS) if name == "enemies" and slot > 0]
    assert np.isneginf(absent[enemy_slots].numpy()).all()  # absent slots cannot be pointed at


def test_a_jump_after_a_held_jump_first_releases_the_button():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT_JUMP, 32), scene())
    pictures = [scene()] * 2 + [scene(support="air")] * 10 + [scene()]
    play(executor, pictures)  # lands while the button is still held
    executor.start(ActionPlan(SMBAction.RIGHT_JUMP, 4), scene())
    assert executor.press() == SMBAction.RIGHT  # released for one frame
    assert executor.press() == SMBAction.RIGHT_JUMP


def test_a_plan_started_in_the_air_ends_when_mario_lands():
    executor = SMBExecutor()
    executor.start(ActionPlan(SMBAction.RIGHT, 32), scene(support="air"))
    pressed, reasons = play(executor, [scene(support="air")] * 3 + [scene()])
    assert reasons == ["landed"] and len(pressed) == 3


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
    teacher_skill = SkillToken("clear_gap", 1)
    teacher_plan = ActionPlan(SMBAction.RIGHT_JUMP, JUMP_FRAMES[3])

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
    assert alone.decision.strategy.kind == "progress" and "skill" in alone.decision.chosen


def test_jumps_are_taught_the_middle_of_the_longest_certified_run():
    from retroagi.stages.block_smb.layered_train import jump_frame_label

    menu = list(JUMP_FRAMES)
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
            "contact": np.array([0.0, 1.0, 0.0]),
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


def test_the_memory_reads_only_the_two_latest_pictures():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    # It takes the window's summary and nothing else: no action, no button.
    assert policy.memory.cell.input_size == policy.settings.width
    with torch.no_grad():
        frames = [policy.encode_scene(rows) for rows in _rows([150, 152, 160])]
        coming = policy.remember(frames[1], frames[0], None)
        going = policy.remember(frames[1], frames[2], None)
        alone = policy.remember(frames[1], absent_like(frames[1]), None)
        expected, _ = policy.expect(coming)
    # The same picture with the enemy arriving from elsewhere, or from nowhere: different memory.
    assert not torch.allclose(coming.hidden, going.hidden)
    assert not torch.allclose(coming.hidden, alone.hidden)
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
            now = policy.encode_scene(frames[t])
            before = policy.encode_scene(frames[t - 1]) if t else absent_like(now)
            state = policy.remember(now, before, state)
            played.append(state.hidden[0])
        a, b, c = (torch.cat([r[i] for r in frames]).unsqueeze(0) for i in range(3))
        d = {
            "episode": torch.zeros(len(starts), dtype=torch.long),
            "frame": torch.tensor(starts),
        }
        replayed = action_memory(policy, a, b, c, d)
    assert torch.allclose(torch.stack(played), replayed, atol=1e-5)
