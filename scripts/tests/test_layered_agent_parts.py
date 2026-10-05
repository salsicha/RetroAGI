"""The game-neutral parts of the layered agent: reader, tokens, executor, layers."""

from collections import defaultdict

import numpy as np
import pytest
import torch

from retroagi.core.actions import SMBAction
from retroagi.core.layered_policy import LayeredSMBPolicy, action_plan
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
    SKILL_WIDTH,
    TACTIC_WIDTH,
    TACTICS,
    SkillToken,
    StrategyToken,
    TacticToken,
    encode_skill,
    encode_tactic,
    tactic_token,
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


def test_tokens_refuse_unknown_values_and_wrong_directions():
    with pytest.raises(ValueError):
        StrategyToken("hurry")
    with pytest.raises(ValueError):
        TacticToken("jump_gap")
    with pytest.raises(TypeError):
        TacticToken("advance", -1)  # no direction argument
    vector = encode_tactic(tactic_token("climb_backward"))
    assert vector.shape == (TACTIC_WIDTH,)
    assert TACTIC_WIDTH == len(TACTICS) == 7
    assert vector[TACTICS.index("climb_backward")] == 1.0
    assert vector.sum() == 1.0 and vector[-1] == 0.0
    hold = encode_tactic(tactic_token("hold_ground"))
    assert hold[-1] == 1.0 and hold.sum() == 1.0


def test_tactics_include_holding_and_have_no_direction_field():
    assert TACTICS == (
        "advance",
        "retreat",
        "climb_forward",
        "climb_backward",
        "descend_forward",
        "descend_backward",
        "hold_ground",
    )
    assert all(not hasattr(tactic_token(t), "direction") for t in TACTICS)


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
        return {"tactic": [tactic_token("advance")], "action": [plan]}

    screen = [np.zeros((240, 256, 3), np.uint8)]
    steps = [agents.act(screen, [0], given=given)[0] for _ in pictures]
    # Touching an enemy, Mario vanishing and leaving the ground change nothing;
    # the landing on the last frame ends the action and a new one starts.
    assert [step.ended for step in steps] == [None, None, None, None, None, "landed"]
    assert [step.decision is not None for step in steps] == [True] + [False] * 4 + [True]
    assert [step.button for step in steps] == [SMBAction.RIGHT] * 6


def test_the_agent_updates_the_hold_controller_between_policy_decisions():
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_executor import HOLD_GROUND
    from retroagi.core.smb_observer import VisionObserver

    pictures = [scene(), scene(mario=(104, 192, 116, 208)), scene(mario=(96, 192, 108, 208))]
    agents = SMBAgents(VisionObserver(SceneList(pictures)), LayeredSMBPolicy().eval(), "cpu")

    def given(copies, scenes):
        return {"tactic": [tactic_token("hold_ground")], "action": [ActionPlan(HOLD_GROUND, 8)]}

    screen = [np.zeros((240, 256, 3), np.uint8)]
    steps = [agents.act(screen, [0], given=given)[0] for _ in pictures]
    assert [step.button for step in steps] == [SMBAction.NOOP, SMBAction.LEFT, SMBAction.RIGHT]
    assert [step.decision is not None for step in steps] == [True, False, False]


# ── Layers ────────────────────────────────────────────────────────────────────


def test_the_action_layer_reads_only_the_spatial_command(monkeypatch):
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("action network read vision or memory")

    monkeypatch.setattr(policy.scene, "forward", forbidden)
    monkeypatch.setattr(policy.memory, "forward", forbidden)
    with torch.no_grad():
        out = policy.run_action(encode_skill(SkillToken("jump", 40, -24))[None])
        other = policy.run_action(encode_skill(SkillToken("jump", -40, -24))[None])
    assert isinstance(action_plan({name: value[0] for name, value in out.items()}), ActionPlan)
    assert not torch.equal(out["action"], other["action"])


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
    teacher_tactic = tactic_token("climb_forward")
    teacher_plan = ActionPlan(SMBAction.RIGHT_JUMP, 6)

    def given(copies, scenes):
        assert copies == [0, 1] and len(scenes) == 2
        return {"tactic": [teacher_tactic, teacher_tactic], "action": [teacher_plan, None]}

    steps = agents.act(screens, [0, 1], given=given, run_given=("action",))
    first, second = steps[0].decision, steps[1].decision
    assert first.tactic == second.tactic == teacher_tactic
    assert first.plan == teacher_plan
    assert second.plan == second.chosen["action"]  # not given: the policy's own
    assert "action" in first.chosen  # run anyway, to compare with the given plan
    assert first.tactic_step is None  # given for every copy, so the tactic layer did not run
    # Play passes nothing: every layer's token is the policy's (strategy: the default).
    agents.reset(0)
    (alone,) = agents.act(screens[:1], [0])
    assert alone.decision.strategy.kind == "speed_run" and "tactic" in alone.decision.chosen


def test_jumps_are_taught_the_middle_of_the_longest_certified_run():
    from retroagi.core.smb_physics import NES_JUMP_FRAMES
    from retroagi.stages.block_smb.layered_train import jump_frame_label

    menu = list(NES_JUMP_FRAMES)
    assert jump_frame_label(menu[0], ()) == menu[0]
    assert jump_frame_label(menu[0], (menu[2], menu[3], menu[4], menu[9])) == menu[3]
    assert jump_frame_label(menu[0], (menu[9],)) == menu[9]


def _record(learner: str, frames: int = 30):
    from retroagi.core.tokens import STRATEGY_WIDTH
    from retroagi.stages.block_smb.layered_train import EpisodeRecord

    rng = np.random.default_rng(0)
    rows = [policy_input(scene(mario=(100 + t, 192, 112 + t, 208))) for t in range(frames)]
    decisions = np.array([0, 10, 20], np.int32)
    width = {"action": SKILL_WIDTH, "skill": TACTIC_WIDTH, "tactic": STRATEGY_WIDTH}[learner]
    labels = {
        "action": {"action": np.array([1, 2, 0]), "frames": np.array([3, 0, 5])},
        "skill": {
            "mode": np.array([0, 1, 2]),
            "x": np.array([280, 216, 256]),
            "y": np.array([240, 216, 240]),
        },
        # The teacher's tactic, and whether the held one is finished.
        "tactic": {"tactic": np.array([0, 2, 2]), "end": np.array([False, True, False])},
    }[learner]
    # The tactic learner: advance from the start, ended at the second decision
    # for retreat, held through the third.
    tactic = {
        "held": np.array([-1, 0, 1]),
        "actions": np.array([0, 1, 1]),
        "frames": np.array([0, 10, 10]),
        "started": np.array([True, True, False]),
        "used": np.array([0, 1, 1]),
        "own_end": np.array([True, True, False]),
        "end_probability": np.array([-1.0, 0.7, 0.2]),
    }
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
        given=np.eye(width, dtype=np.float32)[[0, 1, 0]],
        labels={**labels, "valid": np.ones(3, bool)},
        played_teacher=np.ones(3, bool),
        agreed=np.zeros(3, bool),
        rewards=np.zeros(frames, np.float32),
        old_log_prob=np.zeros(3, np.float32),
        potentials=np.zeros(frames, np.float32),
        tactic=tactic if learner == "tactic" else {},
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
    assert ("expectation" in losses) == (learner == "skill")
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


def test_the_skill_layer_reads_its_own_previous_destinations():
    from retroagi.core.layered_policy import CHOICE_WIDTH, HISTORY, encode_choice

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    inputs = policy_input(scene(enemies=[EnemyView((150, 198, 160, 208), "walker")]))
    rows = (inputs.src_a[None], inputs.src_b[None], inputs.src_c[None])
    tactic = encode_tactic(tactic_token("advance"))[None]

    def action_scores(previous):
        choices = torch.zeros(1, HISTORY, CHOICE_WIDTH["skill"])
        present = torch.zeros(1, HISTORY, dtype=torch.bool)
        for age, plan in enumerate(previous):
            choices[0, age] = encode_choice("skill", plan)
            present[0, age] = True
        with torch.no_grad():
            now = policy.encode_scene(rows)
            _, expected = policy.expect(policy.remember(now, None))
            out = policy.run_skill(now, expected, tactic, (choices, present))
        return out["x"]

    nothing = action_scores([])
    walked = action_scores([SkillToken("run", 24, 0)] * 3)
    jumped = action_scores([SkillToken("jump", 40, -24)] * 3)
    assert not torch.allclose(nothing, walked) and not torch.allclose(walked, jumped)


def test_the_agent_keeps_its_last_16_used_choices_newest_first():
    from retroagi.core.layered_policy import HISTORY, encode_choice
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    agents = SMBAgents(VisionObserver(SceneEcho(scene())), LayeredSMBPolicy().eval(), "cpu")
    plans = [SkillToken("run", 1 + i, 0) for i in range(20)]
    turn = iter(plans)

    def given(copies, scenes):
        return {
            "tactic": [tactic_token("advance")],
            "skill": [next(turn)],
            "action": [ActionPlan(SMBAction.RIGHT, 1)],
        }

    screen = [np.zeros((240, 256, 3), np.uint8)]
    while True:
        try:
            agents.act(screen, [0], given=given)
        except StopIteration:
            break
        while not agents.copies[0].executor.finished:
            agents.copies[0].executor.press()
    kept = agents.copies[0].choices["skill"]
    assert len(kept) == HISTORY
    expected = [encode_choice("skill", plan) for plan in reversed(plans)][:HISTORY]
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


def test_waiting_for_a_moving_platform_is_holding_ground():
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_tokens import (
        episode_teacher,
        schedule_stance,
        teacher_tactic,
    )

    scenario = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=3, sample_index=0, family="bridge_wait", difficulty="hard"
    ).scenario
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    teacher = episode_teacher(scenario)
    stance, _ = schedule_stance(env, teacher)
    tactic = teacher_tactic(env, teacher)
    env.close()
    assert (stance, tactic.stance) == ("hold_area", "hold_ground")


# ── The tactic layer: an option-critic with its own memory ───────────────────


def _held_agents(end_bias: float, copies: int = 1, pictures=None):
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    policy.tactic.heads["end"].bias.data.fill_(end_bias)
    vision = SceneList(pictures) if pictures is not None else SceneEcho(scene())
    return SMBAgents(VisionObserver(vision), policy, "cpu", copies)


def test_the_agent_holds_its_tactic_until_its_end_check_says_finished():
    screen = [np.zeros((240, 256, 3), np.uint8)]
    for end_bias, starts in ((-20.0, 1), (20.0, None)):
        agents = _held_agents(end_bias)
        made = []
        for _ in range(100):
            (step,) = agents.act(screen, [0])
            if step.decision is not None:
                made.append(step.decision.tactic_step)
        assert len(made) > 3
        # The first decision always chooses; after it the end check decides.
        assert made[0].held is None and made[0].started
        if starts == 1:
            assert [m.started for m in made] == [True] + [False] * (len(made) - 1)
            assert [m.actions for m in made[1:]] == list(range(1, len(made)))
        else:
            assert all(m.started for m in made) and all(m.actions <= 1 for m in made)


def test_the_tactic_memory_steps_only_when_a_tactic_starts():
    screen = [np.zeros((240, 256, 3), np.uint8)]
    agents = _held_agents(-20.0)
    states = []
    for _ in range(60):
        (step,) = agents.act(screen, [0])
        if step.decision is not None:
            states.append(agents.copies[0].tactic_hidden.clone())
    assert not torch.equal(states[0], torch.zeros_like(states[0]))  # stepped at the start
    assert all(torch.equal(states[0], later) for later in states[1:])  # held: no step


def _played_tactic_episode(end_bias: float = 0.0, frames: int = 90):
    """Play one episode with the tactic layer deciding (its end check sampled), and
    record it as the trainer does."""
    from retroagi.core.layered_policy import encode_choice
    from retroagi.core.tokens import TACTICS, encode_strategy
    from retroagi.stages.block_smb.layered_train import EpisodeRecord

    pictures = [
        scene(enemies=[EnemyView((150 + 3 * (t % 20), 198, 160 + 3 * (t % 20), 208), "walker")])
        for t in range(frames)
    ]
    agents = _held_agents(end_bias, pictures=list(pictures))
    torch.manual_seed(1)
    screen = [np.zeros((240, 256, 3), np.uint8)]
    rows, decided, d = [], [], defaultdict(list)
    for t in range(frames):
        (step,) = agents.act(screen, [0], sample=("tactic",))
        rows.append(step.rows)
        if step.decision is not None:
            decision, tactic = step.decision, step.decision.tactic_step
            decided.append(tactic)
            d["frame"].append(t)
            d["given"].append(encode_strategy(decision.strategy).numpy())
            d["used"].append(encode_choice("tactic", decision.tactic).numpy())
            d["held"].append(-1 if tactic.held is None else TACTICS.index(tactic.held))
            d["actions"].append(tactic.actions)
            d["frames"].append(tactic.frames)
            d["started"].append(tactic.started)
            d["used_index"].append(TACTICS.index(decision.tactic.stance))
            d["log_prob"].append(tactic.choice_log_prob)
    count = len(d["frame"])
    a, b, c = (np.stack([np.asarray(r[i]) for r in rows]) for i in range(3))
    record = EpisodeRecord(
        family="flat_run",
        split="train",
        sample_index=0,
        won=False,
        difficulty="easy",
        end="timeout",
        src_a=a.astype(np.int8),
        src_b=b.astype(np.int8),
        src_c=c.astype(np.float32),  # (training stores half precision; exact here)
        buttons=np.zeros(frames, np.int8),
        decision_frames=np.asarray(d["frame"], np.int32),
        given=np.stack(d["given"]).astype(np.float32),
        labels={
            "tactic": np.asarray(d["used_index"]),
            "end": np.zeros(count, bool),
            "valid": np.ones(count, bool),
        },
        played_teacher=np.zeros(count, bool),
        agreed=np.zeros(count, bool),
        rewards=np.zeros(frames, np.float32),
        old_log_prob=np.asarray(d["log_prob"], np.float32),
        potentials=np.zeros(frames, np.float32),
        tactic={
            "held": np.asarray(d["held"]),
            "actions": np.asarray(d["actions"]),
            "frames": np.asarray(d["frames"]),
            "started": np.asarray(d["started"]),
            "used": np.asarray(d["used_index"]),
        },
    )
    return agents.policy, record, decided


def test_replaying_an_episode_gives_the_tactic_layer_what_play_gave():
    from retroagi.core.tokens import TACTICS
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, tactic_forward

    policy, record, decided = _played_tactic_episode()
    starts = [k for k, step in enumerate(decided) if step.started]
    assert 1 < len(starts) < len(decided)  # both holding and ending happened
    config = LayeredTrainConfig(learner="tactic")
    with torch.no_grad():
        d, out = tactic_forward(policy, [record], config, "cpu", memory_grad=False)
    # The end check gives what it gave in play, wherever a tactic was held.
    holding = out["holding"].tolist()
    replayed = torch.sigmoid(out["hold"]["end"])
    for k, chance in zip(holding, replayed):
        assert decided[k].end_probability == pytest.approx(float(chance), abs=1e-5)
    # The choice, where the layer chose, from the memory as play stepped it.
    choosing = torch.log_softmax(out["choose"]["tactic"], -1)
    for k, step in enumerate(decided):
        if step.own_end:
            mine = TACTICS.index(step.own)
            assert step.choice_log_prob == pytest.approx(float(choosing[k, mine]), abs=1e-5)


def test_tactic_returns_blend_the_critics_next_estimate_with_what_followed():
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, critic_targets

    tactics = 4
    count = 3
    d = {
        "episode": torch.zeros(count, dtype=torch.long),
        "used": torch.tensor([0, 0, 1]),
        "earned": torch.tensor([1.0, 2.0, 4.0]),
        "length": torch.tensor([1, 1, 1]),
        "last": torch.tensor([False, False, True]),
        "terminal": torch.tensor([True, True, True]),
    }
    values = torch.zeros(count, tactics)
    values[1, 0] = 10.0  # holding tactic 0 at the second decision
    choose = {"tactic": torch.zeros(count, tactics), "values": values}
    hold = {"end": torch.tensor([-100.0, 100.0])}  # decision 1 keeps it; decision 2 ends it
    out = {
        "choose": choose,
        "hold": hold,
        "holding": torch.tensor([1, 2]),
        "before": None,
        "ends": torch.tensor([], dtype=torch.long),
    }
    config = LayeredTrainConfig(learner="tactic", discount=1.0)
    # Nothing blended: each return is the rewards that followed.
    returns, _ = critic_targets(d, out, config, smoothing=1.0)
    assert returns.tolist() == pytest.approx([7.0, 6.0, 4.0])
    # Only the critic: decision 0 continues tactic 0 into decision 1 (worth 10).
    returns, _ = critic_targets(d, out, config, smoothing=0.0)
    assert returns[0].item() == pytest.approx(1.0 + 10.0)
    # Decision 1's tactic ends at decision 2: the value of choosing anew there (0).
    assert returns[1].item() == pytest.approx(2.0)


def test_the_end_check_holds_a_tactic_worth_more_and_drops_one_worth_less():
    from retroagi.stages.block_smb.layered_train import termination_loss

    for held_value, direction in ((5.0, -1), (-5.0, 1)):
        logit = torch.zeros(1, requires_grad=True)
        termination_loss(logit, torch.tensor([held_value]), torch.tensor([0.0]), 0.01).backward()
        # A gradient step moves the chance of ending down (worth more) or up.
        assert np.sign(-logit.grad.item()) == direction


def test_the_shaping_cannot_change_which_tactic_is_best():
    from retroagi.stages.block_smb.layered_train import EpisodeRecord, shaped_rewards

    def shaped_return(distances, discount=0.9):
        weight = 2.0
        frames = len(distances) - 1
        rewards = [weight * (distances[t] - distances[t + 1]) for t in range(frames)]
        record = EpisodeRecord(
            family="flat_run",
            split="train",
            sample_index=0,
            won=False,
            difficulty="easy",
            end="death",
            src_a=np.zeros((frames, 8)),
            src_b=np.zeros((frames, 16)),
            src_c=np.zeros((frames, SEQ_LEN_C)),
            buttons=np.zeros(frames),
            decision_frames=np.array([0]),
            given=np.zeros((1, 4)),
            labels={},
            played_teacher=np.zeros(1, bool),
            agreed=np.zeros(1, bool),
            rewards=np.asarray(rewards, np.float32),
            potentials=np.asarray([weight * x for x in distances[1:]], np.float32),
            terminal=True,
        )
        shaped = shaped_rewards(record, discount, 1.0)
        return float((shaped * discount ** np.arange(frames)).sum())

    # However Mario moved before the episode ended, the shaping pays the same:
    # the weight times the distance he started at.
    assert shaped_return([0.5, 0.4, 0.3, 0.2]) == pytest.approx(1.0)
    assert shaped_return([0.5, 0.6, 0.9]) == pytest.approx(1.0)


def test_tactic_changes_agree_when_within_the_tolerance():
    from retroagi.stages.block_smb.layered_train import end_agreement

    record = _record("tactic")
    record.tactic["used"] = np.array([0, 0, 1, 1, 2, 2])
    record.labels = {"tactic": np.array([0, 1, 1, 1, 1, 2]), "valid": np.ones(6, bool)}
    # Learner changes at 2 and 4; teacher at 1 and 5: both within 1 decision.
    assert end_agreement([record], tolerance=1)["overall"] == pytest.approx(1.0)
    assert end_agreement([record], tolerance=0)["overall"] == pytest.approx(0.0)


def test_reward_rounds_improve_the_tactic_choice_and_end_check_once_the_critic_is_ready():
    from retroagi.stages.block_smb.layered_train import LayeredTrainConfig, tactic_losses

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    record = _record("tactic")
    record.explored = True
    record.played_teacher = np.zeros(3, bool)
    record.rewards = np.linspace(0, 1, record.frames).astype(np.float32)
    config = LayeredTrainConfig(learner="tactic", reward_rounds=1)
    waiting, _ = tactic_losses(policy, [record], config, "cpu", by_reward=True, critic_ready=False)
    assert {"reward", "ending"}.isdisjoint(waiting) and "estimate" in waiting
    losses, _ = tactic_losses(policy, [record], config, "cpu", by_reward=True, critic_ready=True)
    assert {"reward", "entropy", "ending", "estimate", "tactic", "end"} <= set(losses)
    sum(losses.values()).backward()
    assert policy.tactic.heads["end"].weight.grad is not None
    imitation, _ = tactic_losses(policy, [record], config, "cpu")
    assert "tactic_expectation" in imitation  # the memory learns the end scene


def test_a_checkpoint_with_other_tokens_is_refused_unless_only_untrained_layers_read_them(
    tmp_path,
):
    from dataclasses import asdict

    from retroagi.core.smb_observer import observation_layout
    from retroagi.core.tokens import token_layout
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    old_strategies = ["speed_run", "max_coins", "careful"]
    state = dict(policy.state_dict())
    # The tactic layer reads the switch: one more strategy, one more input.
    state["tactic.above.weight"] = torch.zeros(96, len(old_strategies) + 1)

    def saved(tokens, layers, name):
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "settings": asdict(policy.settings),
                "state_dict": state,
                "observation_layout": observation_layout(),
                "token_layout": tokens,
                "trained_layers": layers,
            },
            path,
        )
        return path

    older = {**token_layout(), "strategies": old_strategies}
    loaded, _ = load_layered_checkpoint(saved(older, ["action"], "action"))
    assert torch.equal(loaded.action.network[0].weight, policy.action.network[0].weight)
    assert loaded.tactic.above.weight.shape == policy.tactic.above.weight.shape  # fresh
    with pytest.raises(ValueError, match="different tokens"):
        load_layered_checkpoint(saved(older, ["action", "tactic"], "tactic"))
    other_tactics = {**token_layout(), "tactics": ["advance", "alternate_route"]}
    with pytest.raises(ValueError, match="different tokens"):
        load_layered_checkpoint(saved(other_tactics, ["action"], "tactics"))
    old_actions = {**token_layout(), "executor_actions": token_layout()["executor_actions"][:-1]}
    with pytest.raises(ValueError, match="different tokens"):
        load_layered_checkpoint(saved(old_actions, ["action"], "buttons_only"))
