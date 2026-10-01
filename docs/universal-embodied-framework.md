# Universal Embodied Learning Framework

Status: design for future implementation. Nothing in this document changes
current behavior. It records how RetroAGI can become one framework for any
video game or robot, where each game or robot supplies bespoke content
(families, teachers, calibration, safety limits) and the framework supplies
everything else.

The target pipeline for robots is a fidelity ladder:

1. **Low-fidelity simulation**: fast, deterministic, full access to true state.
   Most learning happens here.
2. **Photoreal simulation**: realistic rendering and contact physics.
   Perception and sensor-only policies are trained and checked here.
3. **Real robot**: no access to true state, expensive resets, safety limits.
   Only calibration and narrow adaptation happen here.

Block SMB → Full SMB is already a two-rung version of this ladder, and it has
not yet transferred (0/3 Level 1-1 completions in
[the transfer contract audit](full-smb-transfer-contract.md)). The first proof
of the framework should therefore be making that game ladder work, before any
robot.

## Plain-Language Overview

1. A **domain plugin** describes one game or robot: its actions, sensors,
   objects, physics settings, safety limits, and practice families.
2. The **framework** provides the learning machinery that does not care which
   domain it is: the policy hierarchy, the world model and critic, the shared
   primitive executor, teacher tooling, curriculum and exams, logging, and the
   fidelity ladder with its promotion gates.
3. **Contracts** connect the two. Every observation, action, primitive, and
   unit is declared, saved inside checkpoints, and checked before a checkpoint
   is used anywhere.
4. A policy is trained at the cheapest fidelity that can teach a skill, then
   promoted one rung at a time. Each promotion needs calibration, a bounded
   adaptation phase, and a held-out exam run in a fresh process.
5. Failures at a higher rung are rebuilt as new practice families at a lower
   rung, so the cheap simulator keeps learning from the expensive one.

## What Exists Today

### Already game-neutral

| Piece | Where | Role in the framework |
| --- | --- | --- |
| Game profiles and stage ladders | `retroagi/core/games.py` (`GameSpec`, `BlockGameSpec`, `StageLadderEntry`) | Becomes the domain plugin's top-level declaration. |
| Stage names | `retroagi/core/stage_resolution.py` (`synthetic`, `block`, `full`, plus optional rungs) | Becomes fidelity rung names; add `photoreal` and `real`. |
| Promotion plans and gates | `retroagi/core/game_promotion.py` (`GamePromotionPlan`, metric/artifact/runtime gates) | Becomes the rung-to-rung promotion machinery. |
| Backend adapters | `retroagi/core/backends.py` (`BackendAdapter`, `GymnasiumBackendAdapter`, `BackendCapabilitySpec`, capability probes) | Becomes the simulator/robot driver interface; capabilities gain new fields (below). |
| Stage lifecycle and tensors | `retroagi/core/interfaces.py` (`StageAdapter`, `StageBatch`, `VisionEncoder`) | Kept; batch contents come from the scene and sensor contracts. |
| Generic action specs | `retroagi/core/actions.py` (`ActionSpec`, `ContinuousControlSpec`) | Already supports continuous axes; becomes the base of the embodiment's action space. |
| Temporal spans | `retroagi/core/temporal.py` (`TemporalGoal`, `HierarchicalTransition`) | Kept as the universal episode log, including real-robot logs. |
| Perception, signals, rewards, tasks | `retroagi/core/perception.py`, `signals.py`, `rewards.py`, `tasks.py` | Kept as domain-declared schemas. |
| Checkpoint-owned runtime contract | `retroagi/core/smb_runtime.py` (`SMBRuntimeContract`) | Generalizes into the embodiment contract. |
| Second registered game | Pong in `games.py` / `game_plugins.py`; `stages/synthetic_1d` | Useful as a non-platformer check during the refactor. |

### Hard-wired to Mario

| Assumption | Where | What it must become |
| --- | --- | --- |
| Six discrete actions; jump/walk/wait primitives with a 16-step duration menu | `core/actions.py` (`SMBAction`, `SMBAdaptiveController`), `core/models.py` (`MotorPrimitiveController`, `DEFAULT_PRIMITIVE_DURATION_BINS`) | Primitives declared by the domain, including continuous parameters. |
| Controller logic about button presses, releases, and landings | `core/actions.py` (`resolve_landing`, re-jump suppression) | A generic committed-primitive executor with domain-declared termination events and handoff rules. |
| Fixed tactic stances and skill types | `core/models.py` (`TACTIC_STANCES`), `core/skills.py` (`SKILL_GOAL_TYPES`) | Vocabularies declared by the domain. |
| Feature positions with physical meaning baked in (e.g., "slots 9-12 are support") | `core/models.py` (`_MOTOR_PRIMITIVE_SUPPORT_SPANS`, `_TERMINAL_SPANS`, `_PROGRESS_SLOTS`) | Roles declared by the scene contract and looked up by name. |
| Seven Mario modules inside the shared core | `core/smb_*.py` | Moved into a Mario domain package. |
| Teachers that freeze the simulator, try every parameter, and restore | `stages/block_smb/local_traversal.py` (`safe_jump_holds`), `policy_recovery.py`, `piranha_tactics.py` | A teacher toolkit whose teachers declare the backend capabilities they need. |

## Design Rules Learned From This Project

Each rule below comes from a failure that cost real training time.

1. **Meaning must be declared, not implied by shape.** The Full SMB transfer
   found tensors of the same shape carrying different physical meanings, and
   jump durations in different units (a 16-frame Block jump flies about 33
   frames; the same NES hold flies about 53). Every observation and action
   layout carries names and units, and is checked at load time.
2. **One executor for everyone.** The two-frame landing lockout lived in the
   controller and was separately mirrored in teacher code. It blocked the
   only survivable frame in enemy_on_platform. Teachers, training, evaluation,
   and deployment must all run primitives through the same implementation,
   and teacher routes must be replayed through it before they are used.
3. **Label the learner's own states, not only the teacher's.** The piranha
   stance head predicted "wait" correctly 73-84% of the time on teacher routes
   and 0 of 184 times in its own play, because it never reached the states the
   teacher visited. Relabeling the learner's states and demonstrating from
   learner-like arrivals is built in, not added per family.
4. **Families must prove the skill is necessary and has margin.** Random plant
   timing let most "timed" layouts be passed without waiting, and the hard
   clearance tier left at most 2 pixels of margin (0-3 of 720 running jumps).
   A family generator certifies both that the target skill is required and
   that a working solution has a stated margin.
5. **True state steers until perception earns it.** The vision support head
   misread airborne frames 86.6% of the time until it was trained against
   engine truth. Perception is trained against simulator truth and only takes
   control after passing a gate.
6. **Safe sets need a preference for the middle.** Any certified duration is
   correct, but the edges of the safe range sit beside failures. Set-valued
   labels include an interior preference.
7. **Judge with fresh-process exams.** In-run scores swing between rounds and
   are measured on few layouts. Promotion uses fixed, stratified, held-out
   exams run in a separate process.

## Architecture

### 1. Domain plugin

A domain plugin is the only place bespoke content lives. It provides:

- the embodiment contract (actions, primitives, sensors, rates, limits);
- the scene schema (object classes, attributes, roles);
- backend adapters for each fidelity rung it supports;
- perception vocabularies and datasets per rung;
- skill-goal and tactic vocabularies;
- teachers (physics-probe, planner, learned, or human);
- practice families with difficulty tiers, necessity and margin checks;
- success predicates, reward terms, and exam definitions;
- calibration procedures and randomization ranges per rung;
- safety envelopes (robots).

Mario becomes the first plugin: `retroagi/domains/platformer/` (or
`domains/smb/`) holding today's `core/smb_*.py` and `stages/block_smb`
content. Pong is the second, and a tabletop arm the first robot.

### 2. Embodiment contract

The generalization of `SMBRuntimeContract`. It is saved in every checkpoint and
refuses to load where it does not match.

```python
@dataclass(frozen=True)
class EmbodimentContract:           # proposal
    name: str
    control_rate_hz: float          # rate of the lowest (controller) level
    decision_rates_hz: Mapping[str, float]   # e.g. {"strategy": 1, "skill": 10}
    action_space: tuple[ActionSpec, ...]     # discrete and continuous axes
    primitives: tuple["PrimitiveSpec", ...]
    sensors: tuple["SensorSpec", ...]        # proprioception, cameras, force, ...
    units: Mapping[str, str]                 # "position": "m" or "px", ...
    safety: "SafetyEnvelope | None"          # required for real hardware
```

### 3. Backend adapters and capabilities

`BackendCapabilitySpec` already declares `save_load_state`, `reset_seed`,
`render`, and similar. Add the capabilities that decide which teachers and
checks are allowed:

| Capability | Low-fidelity sim | Photoreal sim | Real robot |
| --- | --- | --- | --- |
| `save_load_state` (snapshot and restore) | yes | usually | no |
| `privileged_state` (true object poses and contacts) | yes | yes | no |
| `deterministic_replay` | yes | often not (GPU physics) | no |
| `cheap_reset` | yes | yes | no |
| `real_hardware` | no | no | yes |

Teachers and coaching steps declare required capabilities. The framework
refuses to run, for example, a "try every hold from a snapshot" teacher on a
backend without `save_load_state`, instead of failing silently.

### 4. Scene and sensor contract

The universal replacement for Mario's fixed feature layout (the shared SMB
observation in `retroagi/core/smb_scene.py`):

- **Entities**: class, pose, size, velocity, and declared attributes (e.g.,
  `stompable`, `graspable`, `moving`), each with units and a reference frame.
- **Ego state**: the agent's own pose, velocity, contacts, and joint state.
- **Roles**: named meanings the learning machinery uses, such as `support`,
  `hazard`, `progress`, `goal`, and `terminal`. The world model and critic look
  up these roles by name instead of reading fixed positions.
- **Providers**: `oracle` (from simulator truth) or `perceived` (from sensors).
  The SMB implementation currently has one geometry observer per game: Block
  SMB reads simulator truth (`block_oracle_scene`) and Full SMB reads NES RAM
  (`NESGeometry`). A pixel-based observer is not implemented.
- **Availability masks**: a measurement missing at a rung or on a frame (e.g.,
  a moving platform's velocity on the first Full SMB frame it is seen, true
  contact forces on a real robot) is explicitly marked unavailable rather than
  silently zeroed.

Model input layers are built from the schema, so a new domain changes the
schema, not the model code.

### 5. Perception

Per rung, perception maps sensors to the scene contract:

- trained against simulator truth wherever that exists (low-fidelity and
  photoreal rungs);
- gated: a perceived scene only drives the policy after meeting per-role
  accuracy thresholds (support, contact, hazard) on held-out data;
- on the real robot, fine-tuned with a small labeled set and self-consistency
  checks, with the photoreal-trained model as the starting point.

### 6. Primitives and the shared executor

Today's committed jump (choose a hold, commit, adjust mid-flight, end on a
physical event, hand control back) is the general pattern. A primitive is:

```python
@dataclass(frozen=True)
class PrimitiveSpec:                # proposal
    name: str                       # "jump", "reach", "grasp", "step", "wait"
    parameters: Mapping[str, "ParameterSpec"]   # discrete menu or continuous range
    commits: bool                   # executes to a termination event once started
    termination_events: tuple[str, ...]         # "landed", "contact", "timeout", ...
    adjustable: tuple[str, ...]     # parameters the policy may retune mid-execution
    handoff: "HandoffRule"          # when control returns to the policy
```

- Mario: `jump(hold)` ends on `landed` or `enemy_contact`, with handoff on the
  first grounded frame when the button was released in the air.
- Arm: `reach(target_pose, speed)` ends on `arrived`, `contact`, or `timeout`;
  `grasp(width, force)` ends on `closed` or `slip`.
- Legged robot: `step(foothold)` ends on `touchdown`.

A single executor implementation serves teachers, training, evaluation, and
deployment. It emits the same event records to the temporal log at every rung.

### 7. Policy hierarchy

The current strategy → tactics → skill → primitive → controller hierarchy is
kept, with two generalizations:

- **Vocabularies come from the domain.** Tactic stances and skill-goal types
  are declared lists with parameters (for example `clear_gap(distance)` or
  `place(object, pose)`), encoded the same way `skill_goal_encoding` encodes
  them today.
- **Levels run at declared rates.** Robots need the controller at hundreds of
  hertz under a skill layer at around 10 Hz and a strategy layer at around
  1 Hz. The levels exchange `TemporalGoal` records, so the rates are
  independent.

Action heads are factorized as "which primitive" plus "its parameters", so
continuous parameters (target poses, forces) and discrete menus (jump holds)
share one structure.

### 8. World model and critic

The world model (LSTM memory with primitive-outcome prediction) and critic are
kept. Their Mario-specific internals are replaced:

- terms that read fixed feature positions read schema roles instead;
- primitive-outcome prediction predicts the primitive's declared termination
  event and end state;
- on the real robot, the world model is also the fallback teacher (below).

### 9. Teachers and coaching toolkit

The Block SMB teacher machinery generalizes into a toolkit:

| Tool | Today | General form |
| --- | --- | --- |
| Certified parameter sets | `safe_jump_holds` | From a snapshot, try each primitive parameter, keep the ones that achieve the objective and leave a recoverable state. |
| Middle-of-range preference | `interior_duration_targets`, `INTERIOR_DURATION_WEIGHT` | Same, for any discrete menu; distance-to-failure weighting for continuous parameters. |
| Direction-only corrections | `sign_coached_hold` | "More" or "less" feedback when only the sign of the error is known. |
| Unreachable-attempt suppression | `jump_overreach`, `primitive_unreachable` | Push down the decision to start a primitive from a state where no parameter works. |
| Launch-window labels | `takeoff_timing_actions` | "Keep approaching", "either", or "act now" from the size and trend of the certified window. |
| Learner-state relabeling | `repair_policy_actions`, online tactic labels | Relabel the learner's own states with the teacher's answer. |
| Learner-like arrivals | `arrival_demonstrations` | Demonstrate corrections from the states learners typically reach. |

Teacher kinds, by the capabilities they need:

1. **Physics-probe teachers** (snapshot and restore): low-fidelity rung, and
   the photoreal rung when deterministic. Use repeated trials and success rates
   where replays are not deterministic.
2. **Planner teachers** (true state, no snapshots): motion planners or
   trajectory optimizers in simulation.
3. **Learned teachers**: the learned oracle in
   [the universal retro oracle roadmap](universal-retro-oracle.md), and on real
   robots the world model's predicted outcomes with conservative margins.
4. **Human teachers**: teleoperation and interventions, logged as correction
   labels.

Every teacher's output is replayed through the shared executor
(`teacher_route_reachable` today) before it becomes training data.

### 10. Families, curriculum, and exams

Kept as framework machinery (today in `stages/block_smb/monte_carlo.py` and
`train.py`):

- family generators with easy, medium, and hard tiers;
- solvability check (a teacher solves the layout);
- **necessity check** (a baseline that lacks the target skill fails, as the
  timed piranha layouts now guarantee);
- **margin check** (the solution survives stated perturbations, as the hard
  clearance tier now guarantees);
- stratified, fixed, held-out exams per family;
- mastery-gated sampling and retention;
- failure-driven families: situations that fail at a higher rung are rebuilt
  at a lower rung (`transfer_failure_families.py` does this for Full SMB).

### 11. Logging and provenance

`HierarchicalTransition` is already game-neutral. Every rung, including the
real robot, writes the same spans with `source` (real, simulated, teacher,
human) and `policy_version`. This is what makes failure-driven families and
cross-rung comparisons possible.

## The Fidelity Ladder

### Rung 1: Low-fidelity simulation

- **Examples**: Block SMB; MuJoCo or Brax with simple rendering for robots.
- **Available**: snapshots, true state, deterministic replay, cheap resets.
- **Trained here**: the full hierarchy on simulator-truth scene geometry, with
  physics-probe teachers and the full coaching toolkit, at high volume.
- **Robustness**: physics parameters (masses, friction, latency, actuator
  gains) vary around measured ranges so the policy never depends on one value.
- **Promotion gate**: per-family exam thresholds at every difficulty tier.

### Rung 2: Photoreal simulation

- **Examples**: Full SMB emulator for games; Isaac Sim, Unreal-based
  simulators, or Habitat for robots.
- **Available**: true state; snapshots and determinism vary by simulator.
- **Trained here**:
  - perception, against simulator truth, until it passes its gate;
  - a sensor-only student policy that learns to match the rung-1 policy and
    teachers (which still see true state), corrected on the student's own
    states;
  - primitive calibration: measure how declared parameters map to physical
    outcomes and store the mapping in the checkpoint;
  - bounded adaptation: only the heads listed for this rung may change.
- **Visual robustness**: lighting, textures, camera pose, and distractors vary.
- **Promotion gate**: the same family exams, now with a scene derived from
  sensors, plus perception gates and a check that the adaptation stayed in
  bounds. The current Full SMB rung does not do this yet: it is an explicit
  RAM-assisted player whose geometry comes from NES RAM, with only scene
  semantics coming from its frozen ViT segmenter.

### Rung 3: Real robot

- **Available**: sensors only; no snapshots, no true state, costly resets.
- **Trained here**: calibration from measured responses, perception
  fine-tuning, and narrow adaptation of low-level heads only.
- **Teachers**: world-model predictions with conservative margins, plus human
  interventions.
- **Safety supervisor** (always on, outside the learned policy): joint, speed,
  force, and workspace limits; emergency stop; watchdogs on control rate and
  sensor freshness. A limit violation stops the primitive and is logged as a
  failure event.
- **Feedback**: every real failure is logged as spans, reconstructed in the
  lower rungs, and turned into practice families.
- **Promotion gate**: a small fixed real-world exam suite with success rates
  and zero safety violations.

## Robot-Specific Requirements

- **Continuous control**: primitives with continuous parameters; the executor
  interpolates and tracks at the control rate.
- **Latency**: the contract declares sensing and actuation delays; simulators
  inject them; policies see time-stamped observations.
- **Resets**: families declare how a scene is reset at each rung (automatic in
  simulation; scripted or human-assisted on hardware).
- **Data budgets**: real-robot adaptation assumes hundreds, not millions, of
  episodes; only small heads adapt.
- **Hardware interface**: a ROS 2 bridge implements `BackendAdapter` for the
  real rung.

## Implementation Roadmap

Each phase keeps all existing Block SMB and Full SMB results reproducible.

| Phase | Work | Done when |
| --- | --- | --- |
| 0. Separate Mario from the core | Move `core/smb_*.py`, fixed vocabularies, and fixed slot spans into a Mario domain package behind a domain-plugin interface. | All current tests pass; a fresh Block SMB run matches recent results within normal variation. |
| 1. Primitives and executor | `PrimitiveSpec`, `ParameterSpec`, `HandoffRule`; one executor with event hooks; Mario jump/walk/wait expressed through it. | Teacher replay, training, evaluation, and emulator play produce identical actions for the same inputs. |
| 2. Scene and sensor contract | Schema with units, roles, and availability masks; model inputs, world model, and critic built from the schema. | Pong and Mario both train through the schema path; no fixed slot positions remain in `core/models.py`. |
| 3. Vocabularies and hierarchy rates | Domain-declared skills and stances; per-level decision rates. | A domain with different skills trains without editing core code. |
| 4. Teacher toolkit | Capability-gated teachers; toolkit functions generalized from Block SMB. | Running a snapshot teacher on a backend without snapshots is refused with a clear error. |
| 5. Ladder toolkit, proven on Mario | Calibration, randomization, teacher-student distillation, adaptation bounds, promotion gates with fresh-process exams. | A Block-trained policy passes Full SMB Level 1-1 exams through the ladder. |
| 6. First robot, rungs 1 and 2 | Tabletop arm in MuJoCo, then in a photoreal simulator; reach, grasp, and place families. | Sensor-only student passes photoreal exams at every tier. |
| 7. First robot, rung 3 | ROS 2 bridge, safety supervisor, real calibration and exams, real-to-sim failure families. | Real exam suite passes with zero safety violations. |

## Risks and Open Questions

- **The ladder has not worked yet, even simulation to simulation.** Phase 5 is
  the real test of the design and should not be skipped.
- **Non-deterministic simulators** weaken snapshot-and-try teachers; certified
  sets become estimated success rates, which need margins of their own.
- **Continuous parameters** change the coaching tools: safe ranges instead of
  safe menus, and interior preference becomes distance to the nearest failure.
- **Learned teachers can be confidently wrong** where no probe teacher exists;
  real-rung labels need conservative margins and human review of failures.
- **Scope of adaptation per rung** must be decided per domain and enforced, or
  late rungs will overwrite skills learned cheaply earlier.

## Related Documents

- [Hierarchical self-supervised planning](hierarchical-self-supervised-planning.md):
  the temporal hierarchy this framework keeps.
- [Universal retro oracle roadmap](universal-retro-oracle.md): learned teachers
  across games, the long-term replacement for bespoke teachers.
- [Full SMB transfer contract](full-smb-transfer-contract.md): the evidence for
  the contract and calibration rules.
- [Tensor contracts](tensor-contracts.md) and [stage semantics](stage-semantics.md):
  today's per-stage contracts that the scene and embodiment contracts extend.

## Glossary

- **Calibration**: measuring how a primitive's declared parameters map to
  physical outcomes at one rung, and storing that mapping.
- **Certified set**: the primitive parameters a teacher has verified succeed
  from a given state.
- **Fidelity rung**: one level of the ladder (low-fidelity simulation,
  photoreal simulation, real robot).
- **Handoff rule**: when a committed primitive returns control to the policy.
- **Necessity check**: proof that a family's layouts cannot be solved without
  the skill the family teaches.
- **Privileged state**: true simulator information (exact poses, contacts)
  that a real robot cannot sense.
- **Promotion gate**: the checks a policy must pass before moving up a rung.
- **Teacher-student distillation**: training a sensor-only policy to match a
  teacher that sees privileged state, on the student's own states.
