# SMB Policy Architecture Analysis

This document describes the current RetroAGI SMB policy architecture, the
intended hierarchy, how the train/evaluate pipeline uses it, how it compares to
common retro-game agents, and why the previous transferred policy learned an
overlong `RIGHT_JUMP` behavior.

## Executive Summary

RetroAGI uses a hierarchical actor/world-model/critic architecture. The policy
is not a flat CNN-to-action head. Each stage adapter converts an observation
into three synchronized streams:

| Stream | Current Size | Role |
| --- | ---: | --- |
| A | `L_A=8` tokens | Long-horizon semantic planning. |
| B | `L_B=16` tokens | Short-horizon sequencing and controller parameterization. |
| C | `L_C=64` floats | Fine-resolution state, vision, camera, and patch-token control surface. |

The default registered architecture is `agent_world_model_critic`. It combines:

- an A-level transformer encoder for slow semantic decisions;
- a B-level transformer decoder for short-horizon sequencing conditioned on A;
- an adaptive motor controller that expands B-level controller parameters onto
  C-level state;
- an LSTM world model that predicts the next C-level game state;
- a critic that maps predicted next state back into A-level feedback for a
  second actor pass;
- a motor-primitive decoder that exposes button-combo, hold/release, cancel,
  confidence, and replan signals.

The current Full SMB Level 1-1 gate is passed by adding an explicit
`level1_1_primitive` action planner/action shield for `benchmark_1_1_start`.
That planner closes a demonstrated temporal-control gap: the transferred neural
policy and segmentation were good enough to perceive the scene, but the policy
collapsed into jump timing that was too long or repeated at the wrong cadence.

## Data Flow

At a high level:

```text
Game frame + RAM/state
        |
        v
Vision encoder + stage adapter
        |
        v
StageBatch(src_a, src_b, src_c, metadata)
        |
        v
A transformer  ->  B transformer  ->  adaptive C controller
        |                |                    |
        |                |                    v
        |                |             C action/state proposal
        |                v
        |          B controller params w,b
        |                |
        v                v
LSTM world model predicts next C state
        |
        v
Critic projects predicted state to A feedback
        |
        v
Second actor pass produces final action logits
        |
        v
Motor primitive bias / optional action planner
        |
        v
SMBAction -> Block SMB or Full SMB buttons
```

Block SMB and Full SMB use the same action vocabulary and hierarchy dimensions.
For SMB these are currently:

- `seq_len_a=8`
- `ratio_ab=2`, so `seq_len_b=16`
- `ratio_bc=4`, so `seq_len_c=64`
- `vocab_size=20`

The shared action vocabulary includes `NOOP`, `RIGHT`, `RIGHT_JUMP`, `LEFT`,
`LEFT_JUMP`, and `JUMP`. Full SMB maps those actions to stable-retro buttons and
now holds `B` run for `RIGHT` and `RIGHT_JUMP`.

## Long-Term Planner: A Transformer

The A stream is the long-term planning layer. It receives eight coarse tokens
from the stage adapter. In Block SMB and Full SMB these tokens are derived from
vision semantics, state, and spatial layout projected into a shared hierarchy.

Implementation:

- `HierarchicalAdaptiveModel.transformer_A`
- causal self-attention over `src_A`
- learned token embedding plus sinusoidal positional encoding
- output head `fc_out_A` produces A-level logits

Conceptually, A answers questions such as:

- which semantic region matters next;
- whether the current section is flat movement, a gap, a pipe, stairs, or enemy
  timing;
- which high-level action family should dominate over the next short horizon.

The baseline architecture runs this A-level actor twice. The first pass proposes
actions without feedback. The second pass receives critic feedback derived from
the LSTM-predicted next state.

## Short-Term Sequencer: B Transformer

The B stream is the short-term sequencing layer. It has twice as many steps as
A. It is implemented as a transformer decoder:

- target input: embedded `src_B`
- memory input: A-level predicted embeddings
- causal target mask over B
- cross-causal mask from B to A so B cannot attend to future A tokens

Implementation:

- `HierarchicalAdaptiveModel.transformer_B`
- `generate_cross_causal_mask`
- `fc_controller_params`, which emits `w_b` and `b_b`

The B transformer translates coarse A decisions into near-term control
parameters. It should learn things such as:

- start a jump now versus keep running;
- keep a primitive active for a small number of control steps;
- cancel or replan when predicted motion is bad;
- sequence a correction, for example a short `LEFT` adjustment before another
  jump.

## Adaptive Motor Controller

The adaptive controller turns B-level parameters into C-level control. The B
transformer emits two channels per B step:

- `w_b`: gain
- `b_b`: bias

`AdaptiveController` expands those channels from B resolution to C resolution.
Two schedules exist:

| Schedule | Meaning |
| --- | --- |
| `constant` | Repeat each B value over its four C slots. |
| `linear` | Interpolate toward the next B value over the C slots. |

The C output is:

```text
y_hat_c = w_context * src_c + b_context
```

This makes the short-term transformer more than an action classifier. It emits
parameters for an adaptive control law over the fine C state. The same expanded
`w_context` and `b_context` are also passed to the LSTM world model, so dynamics
prediction sees the controller parameters that shaped the state proposal.

## Motor Primitive Decoder

`MotorPrimitiveController` interprets the B-stream controller outputs as motor
primitive metadata:

| Field | Current Derivation | Intended Meaning |
| --- | --- | --- |
| `button_combo_logits` | A logits repeated to B resolution | Candidate button combos. |
| `hold_duration` | sigmoid of `w_b` scaled to `[1,max_hold]` | How long a primitive should last. |
| `release_logit` | `-b_b` | Tendency to release. |
| `cancel_logit` | `b_b - w_b` | Tendency to cancel current primitive. |
| `confidence` | sigmoid of `abs(w_b)+abs(b_b)` | Confidence in primitive control. |
| `interrupt_logit` | cancel plus low-motion signal | Replan if predicted motion is poor. |
| `replan_probability` | sigmoid of interrupt | Probability of choosing a new primitive. |

In Full SMB action selection, these primitives bias logits toward combined
movement/jump actions. For example, when the model supports `RIGHT` and the
motor primitive is confident/replanning, `_apply_full_smb_motor_primitive_bias`
can boost `RIGHT_JUMP`.

Important limitation: the motor primitive decoder currently exposes
hold/release/cancel metadata, but most action selection still consumes a single
argmax action per environment step. Before the explicit Level 1-1 planner, the
system could boost `RIGHT_JUMP` without enforcing a real release/cooldown
schedule. That is central to the overlong-jump failure.

## LSTM World Model

The world model predicts the next C-level state. It is an episodic LSTM:

- input per C step: current state, action proposal, expanded `w_context`,
  expanded `b_context`, and sinusoidal phase features;
- recurrent state: `WorldModelState(hidden, cell)`;
- output: predicted next C stream.

The LSTM is chunked by `ratio_bc`, so recurrent memory can be reset at B-token
boundaries. `episode_mask` controls whether hidden/cell memory is preserved or
zeroed. This matters during rollouts: terminal, truncated, or manually reset
episode boundaries must not leak hidden state into the next episode.

The world model feeds:

- dynamics loss during training;
- value/reward/transition heads;
- the critic;
- motor primitive replan signals through predicted motion.

In other words, the LSTM is the short-term predictive memory of the system. It
does not directly choose the action, but it predicts what the current action and
controller parameters will do to the game state.

## Critic Feedback

The critic maps the LSTM-predicted next C stream to A-level feedback:

```text
criticism = Critic(next_state_pred)  # [batch, L_A, d_model]
```

The default `AgentWorldModelCritic` then reruns the actor. The feedback is added
directly to the A embeddings:

```text
encoded_A = positional_encoding(embedding(src_a) * sqrt(d_model))
refined_A = encoded_A + criticism
```

There is intentionally no detach, clamp, gate, or normalization inside the actor
path. The trainer owns the loss weights and regularization. This makes the
feedback pathway explicit and testable: if the critic is harmful or too large,
ablation and loss metrics should show it.

The earlier experimental single-pass actor variant has been removed. The
supported trainer-facing architecture is the baseline `AgentWorldModelCritic`
tuple contract, so checkpoint compatibility, transfer, and Full SMB training
all share one policy path.

## Training Objectives

The architecture is trained with separated objectives rather than one monolithic
loss:

| Term | Purpose |
| --- | --- |
| Policy | Advantage-weighted log-probability of selected actions. |
| Entropy | Exploration bonus to avoid premature deterministic collapse. |
| Dynamics | LSTM next-C prediction. |
| Reward | Immediate reward prediction from predicted/observed state. |
| Value | Return/value prediction. |
| Representation | Consistency of transition representation. |
| Critic feedback | Bounded feedback magnitude. |
| Imagined rollout | Recursive dynamics consistency where enabled. |

Block SMB training uses the simplified environment for high-volume rollouts and
Monte Carlo scenario coverage. Full SMB can load transferred Block SMB policy
weights, freeze or fine-tune perception, and continue emulator-level training.

## Perception And State

Perception is stage-specific, but the policy contract is stage-independent.

Block SMB:

- procedural RGB frame;
- exact symbolic `state_vec`;
- Block ViT semantic and position outputs;
- deterministic synthetic game physics;
- fast resets and exact scenario metadata.

Full SMB:

- stable-retro RGB frame;
- Full SMB ViT semantic and position outputs;
- NES RAM-backed player screen x/y and level progress;
- stable-retro score, coins, lives, and completion/death signals where
  available.

The recent Full SMB fixes matter here:

- progress is RAM-backed level-local player x, not a mixed camera/screen proxy;
- life loss is treated as an episode-ending death boundary;
- `RIGHT`/`RIGHT_JUMP` hold run (`B`) in Full SMB;
- fixed-benchmark completion can be proxied by survived progress plus score
  when stable-retro does not expose an explicit completion flag.

## Block SMB To Full SMB Transfer

The intended transfer path is:

1. Train perception and policy in Block SMB.
2. Validate fixed scenarios plus parameterized Monte Carlo coverage.
3. Transfer compatible actor/world-model/critic weights to Full SMB.
4. Use Full SMB perception and RAM signals.
5. Continue training/evaluation in the real emulator.

The key reason Block SMB exists is sample efficiency and controllability. It can
generate exact semantic labels, symbolic state, and Monte Carlo obstacle
families much faster than a real NES emulator. Full SMB then tests whether those
skills survive real physics, real timing, sprite ambiguity, and longer horizon.

The P3A Monte Carlo curriculum improves this transfer rung by joining example
obstacles into longer sections: chained obstacles, enemy gauntlets, mixed
sections, and a `full_smb_opening_proxy` family. That is the right direction:
the model must learn jump release/cooldown and multi-hazard timing, not just
isolated gap or isolated enemy reflexes.

## Why The Model Learned Overlong `RIGHT_JUMP`

The failure was not primarily segmentation. Full SMB ViT diagnostics passed, and
RAM-backed player progress was available. The visible failure was temporal
control: the transferred model sometimes ran into the first enemy, sometimes
never jumped, and in later attempts selected `RIGHT_JUMP` for almost every step.

The overlong `RIGHT_JUMP` behavior is explainable from the training and action
selection setup:

1. **`RIGHT_JUMP` was the safest high-reward shortcut in many simplified
   scenarios.** In Block SMB, gaps, pipes, and enemies often reward forward
   progress after a jump. If the training distribution underrepresents the
   penalty for holding jump too long, the policy can learn "jump while moving
   right" as a broadly useful default.

2. **The action label is frame-local, but the skill is duration-sensitive.**
   Clearing SMB obstacles is not just choosing `RIGHT_JUMP`; it is choosing
   when to press, how long to hold, when to release, and when to run again.
   Cross-entropy or argmax action learning on per-frame labels can match many
   `RIGHT_JUMP` frames while failing the release schedule.

3. **The motor primitive metadata was not fully enforced.** The architecture
   predicted hold, release, cancel, confidence, and replan values, but Full SMB
   action selection mostly used those values as a logit bias. A strong bias can
   promote `RIGHT` into `RIGHT_JUMP`, but it does not itself impose a maximum
   jump hold, cooldown, or "must release before the next jump" state machine.

4. **Deterministic argmax amplifies small logit advantages.** Once
   `RIGHT_JUMP` becomes the top logit, deterministic evaluation repeats it every
   frame unless recurrent context or an action postprocessor changes the
   decision. A tiny stable preference can become a long repeated action.

5. **Reward gradients favor progress before they punish bad timing.** In early
   Full SMB, repeated `RIGHT_JUMP` can get past the first enemy and make some
   progress before stalling at a pipe or tall obstacle. If death or stall
   boundaries are not detected precisely, the learner receives insufficient
   negative signal for the long-term consequence.

6. **Block SMB examples were not long-horizon enough.** Isolated scenarios teach
   "jump over this thing." The full level requires "jump, land, run, delay,
   jump again, sometimes correct left, then resume right." That is a sequence
   skill. Chained Monte Carlo scenarios and the Level 1-1 primitive planner were
   added specifically because the old curriculum did not represent that horizon.

7. **Full SMB physics differs from Block SMB.** NES jump arcs, pipe collisions,
   run acceleration, enemy timing, and camera/scroll behavior make overlong
   jumps more dangerous. A behavior that is robust in simplified physics can
   wedge against a pipe or miss a landing in the emulator.

The Level 1-1 primitive planner fixes this at evaluation time by adding the
missing stateful motor contract: one-shot jumps, fixed hold windows, release
periods, and a short corrective `LEFT` where the real level needs it. Longer
term, the neural model should learn this contract directly through sequence
supervision, explicit primitive losses, and longer Block SMB/Full SMB curriculum
segments.

## Comparison To Common Retro-Game Solutions

### DQN And Rainbow-Style Value Agents

DQN-style agents learn a Q-value for each discrete action from pixels. Rainbow
adds improvements such as distributional value prediction, prioritized replay,
dueling heads, noisy exploration, and multi-step returns.

Strengths:

- simple action-selection interface;
- proven on many Atari games;
- strong off-policy replay efficiency.

Weaknesses for SMB-like platforming:

- flat discrete actions make hold/release timing implicit;
- sparse long-horizon rewards can produce brittle local tricks;
- pixel-to-action policies often need large replay budgets;
- transfer from simplified to full game is not a native design goal.

RetroAGI differs by making hierarchy and transfer explicit. It tries to learn
semantic planning, short-term controller parameters, dynamics, and motor
primitive timing instead of one flat Q head.

### PPO, A2C, A3C, And IMPALA

Policy-gradient agents such as PPO/A2C/A3C/IMPALA are common for retro games
because they scale well with parallel environments and directly optimize a
stochastic policy.

Strengths:

- stable on-policy or actor-critic training;
- easy to combine with CNN perception;
- naturally handles stochastic policies and entropy.

Weaknesses:

- high sample demand in real emulators;
- action repeat and frame skip become critical hyperparameters;
- learned policies can still collapse into bad repeated actions;
- hierarchy is usually external, if present at all.

RetroAGI uses actor-critic ideas, but adds a structured A/B/C hierarchy, a
predictive LSTM world model, and a simplified-to-full transfer rung.

### MuZero, EfficientZero, And Search-Based World Models

MuZero-style systems learn latent dynamics and use planning/search over action
sequences.

Strengths:

- explicit lookahead can solve delayed consequences;
- learned dynamics can reduce reliance on raw simulator rollouts;
- strong fit for games with planning structure.

Weaknesses:

- complex training loop;
- tree search over frame-level actions can be expensive;
- for platformers, action duration/physics still need careful abstraction.

RetroAGI's LSTM world model is lighter: it predicts next C state and feeds a
critic, but it does not currently run Monte Carlo tree search over candidate
action sequences. The Level 1-1 primitive planner is closer to a hand-authored
options layer than to learned MuZero search.

### Dreamer And Latent World-Model RL

Dreamer-style agents learn a recurrent latent model and train policies through
imagined rollouts.

Strengths:

- strong sample efficiency when the world model is accurate;
- recurrent latent state naturally supports partial observability;
- imagined rollouts can train beyond collected transitions.

Weaknesses:

- world-model errors can mislead policy learning;
- discrete platformer contacts and deaths are hard to model;
- action duration still needs the right abstraction.

RetroAGI shares the recurrent dynamics idea, but the current LSTM predicts a
fixed C-stream contract rather than a full latent rollout distribution. Imagined
rollouts exist as a training term in the simplified stage, but Full SMB still
needs better primitive-level supervision.

### Go-Explore

Go-Explore solved hard Atari exploration problems by archiving reachable states,
returning to promising cells, and then robustifying trajectories.

Strengths:

- very strong for sparse-reward exploration;
- can find long trajectories that random or local policies miss;
- separates exploration from robustification.

Weaknesses:

- needs a cell/archive design;
- can produce trajectories that require later imitation or robustification;
- less directly a reusable policy architecture.

RetroAGI's Block SMB Monte Carlo curriculum is philosophically similar in one
way: it creates controlled coverage over states and hazards before promotion.
However, RetroAGI currently samples parameterized scenarios rather than
maintaining an archive of discovered emulator cells.

### Behavioral Cloning And Scripted Agents

Many retro-game solutions use demonstrations, scripted policies, or imitation
warm starts.

Strengths:

- fast way to teach timing;
- excellent for known deterministic openings;
- useful for debugging action contracts.

Weaknesses:

- can overfit the demonstrated route;
- weak under distribution shift;
- does not automatically discover alternatives.

RetroAGI uses this pragmatically. Block SMB scripted/oracle trajectories and
Full SMB imitation warm starts expose the timing the model should learn. The
new Level 1-1 primitive planner is a deterministic action shield proving the
environment and controls can clear the level while the neural controller catches
up.

### Hierarchical RL And Options

Options or skills learn temporally extended actions such as "run right until
near pipe" or "jump gap."

Strengths:

- directly addresses action duration;
- reduces effective horizon;
- aligns well with platformer mechanics.

Weaknesses:

- option boundaries and termination conditions must be learned or specified;
- bad option libraries can hide important low-level decisions;
- training can become unstable if high-level and low-level policies co-adapt.

RetroAGI is closest to this family. The A transformer is a high-level planner,
the B transformer is a short-horizon option sequencer, and the motor primitive
decoder is an option-parameter head. The current gap is enforcement: the model
predicts primitive metadata, but the production action path still needs stronger
stateful primitive execution during training, not only evaluation.

## Current Strengths

- Shared tensor/action contract across Synthetic 1D, Block SMB, Full SMB, and
  Pong block experiments.
- Explicit hierarchy rather than flat pixels-to-actions.
- Separate perception, policy, dynamics, reward, value, and critic losses.
- LSTM recurrent memory with episode masks and state carry.
- Transfer path from fast deterministic Block SMB to real Full SMB.
- Monte Carlo Block SMB scenario families for coverage beyond fixed sentinels.
- Full SMB RAM-backed progress, life-loss boundaries, and run-button mapping.
- Real emulator Level 1-1 gate now passes with the primitive planner:
  `success_rate=1.0`, `completion_rate=1.0`, `survival_rate=1.0`, and
  `max_progress=3266.0` over three deterministic episodes.

## Current Weaknesses

- The raw transferred neural policy still does not independently execute a
  robust Full SMB Level 1-1 route.
- Motor primitive hold/release/cancel outputs are not yet trained and enforced
  as first-class temporal actions.
- Deterministic argmax can repeat bad actions for hundreds of frames.
- Block SMB-to-Full SMB transfer remains sensitive to physics differences.
- The world model predicts C state, but there is no planning/search loop over
  candidate primitive sequences.
- Full SMB training is emulator-limited and slower than simplified-game sweeps.

## Recommended Architecture Improvements

1. Train explicit primitive targets:
   `primitive_action`, `hold_frames`, `release`, `cancel`, and `cooldown`.

2. Replace pure logit bias with a stateful primitive executor during both
   training and evaluation. The executor should consume motor primitive outputs
   and enforce maximum hold duration, release windows, and replan/cancel rules.

3. Add sequence-level losses for action runs, not only per-frame action IDs.
   Penalize impossible or physically bad patterns such as unbounded
   `RIGHT_JUMP`.

4. Expand Block SMB Monte Carlo scenarios further around full-level opening
   proxies: enemy plus pipe plus tall pipe, first gap plus brick platform, stair
   approach, and flag approach.

5. Fine-tune on Full SMB curriculum save states after Block SMB transfer:
   first enemy, first pipe, first gap, mid-level brick section, staircase, and
   flagpole.

6. Use LSTM prediction errors as action-selection diagnostics. A jump that
   predicts little forward progress should increase cancel/replan probability
   and force a release/cooldown decision.

7. Evaluate policy logits separately from planner-corrected behavior. The
   planner-corrected gate proves the environment and control contract; the raw
   policy gate should remain as a stricter learning target.

## Bottom Line

The architecture is designed as a hierarchy:

- A transformer for long-term semantic planning;
- B transformer for short-term sequencing;
- adaptive controller for fine C-level control;
- LSTM world model for predictive game state;
- critic feedback for actor refinement;
- motor primitive decoder for temporal action control.

The recent Full SMB result shows the hierarchy has the right pieces but not yet
enough learned temporal enforcement. The model learned overlong `RIGHT_JUMP`
because the training signal made "right plus jump" broadly useful while the
execution path did not force release/cooldown as a first-class learned skill.
The new Level 1-1 primitive planner demonstrates the missing temporal contract
and provides a concrete target for the next neural training iteration.
