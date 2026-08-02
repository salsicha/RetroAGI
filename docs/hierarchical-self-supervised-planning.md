# Hierarchical Self-Supervised Planning Plan

RetroAGI needs hierarchy that persists through real time, not only tensors at
different resolutions. The current A, B, and C streams remain useful for
semantic context, action sequencing, and fine control, but their lengths
describe representation sizes rather than plans that last for seconds, rooms,
or complete levels.

This plan adds learned temporal levels above and around the current controller.
Each level proposes a goal to the level below, observes what actually happened,
and improves from stored play. The first target is Block SMB because its
physics, scenario families, scripted bootstrap, and held-out gates make the
learning loop measurable. The same contracts should later support Full SMB and
other retro-style games.

## Plain-Language Model

The hierarchy should work like this:

1. A route planner decides which meaningful place should be reached next.
2. A tactic planner chooses the obstacle or situation to handle on the way.
3. A skill policy chooses a reusable behavior such as clearing a gap.
4. A motor primitive chooses a button combination and how long to hold it.
5. The existing controller turns that primitive into frame-by-frame actions.

Information moves in both directions. Goals move downward; observed outcomes,
failures, and confidence move upward. A high-level plan is therefore checked
against real play rather than assumed to have succeeded.

The learning process is self-supervised where possible: the game itself tells
the model where the player ended, how long the action lasted, whether support
was lost, and whether the player died. It is not fully reward-free. Progress,
survival, task success, coverage, and reduced uncertainty remain the outer
signals that keep discovered skills useful.

## Current Gap

RetroAGI already has several pieces needed for this work:

- A slow semantic stream, a medium action stream, and a fine control stream;
- learned hold, release, cancel, replan, and post-release outputs;
- an LSTM world model and critic feedback;
- scripted Block SMB traces, Monte Carlo scenario families, failure bins, and
  fixed plus held-out promotion gates;
- a universal oracle roadmap for learned labels, outcome predictions,
  uncertainty, and expectation-maximization relabeling.

These pieces do not yet form a long-horizon planner. A/B/C sequence lengths are
representation lengths, not durable decisions. The current policy can predict
that jumping is useful without preserving the complete action schedule needed
to approach, jump, release, land, and recover. It also has no learned tactic,
route, or cross-episode curriculum level.

The new hierarchy should extend the current controller rather than replace it.

## Proposed Temporal Levels

| Level | Typical duration | Decision | Observable result |
| --- | ---: | --- | --- |
| Perception | 1 frame | What objects, surfaces, motion, and hazards are present? | Updated scene state and confidence |
| Physical control | 1-4 frames | Which buttons should be pressed now? | Position, velocity, contacts, and button state |
| Motor primitive | 4-60 frames | Run, jump, wait, retreat, release, land, or recover | Displacement, landing state, duration, interruption, and risk |
| Skill | 0.5-5 seconds | Clear a gap, mount a platform, stomp an enemy, or pass a gate | Goal state reached or a named failure |
| Tactic | 3-20 seconds | Cross an obstacle group or reach a safe local position | Landmark reached and remaining hazards |
| Route | 10 seconds to level end | Choose a sequence of landmarks toward the level goal | Route progress, cost, and completion |
| Curriculum | Many episodes | Choose what situation or skill to practice next | Improvement, coverage, uncertainty reduction, and regressions |

Durations are ranges, not fixed clocks. A level finishes when its success,
failure, timeout, or interruption condition fires. Higher levels may replan
early when a lower level reports an unexpected outcome.

The initial implementation should add levels one at a time. More names do not
create more intelligence; each level must own a distinct decision, duration,
training signal, and evaluation gate.

## Shared Contracts

All temporal levels should use one versioned transition format so an episode
can be viewed as frames, primitives, skills, tactics, or a route without
duplicating incompatible logs.

### `TemporalGoal`

A goal passed to a lower level should contain:

- `level`: the level expected to execute it;
- `goal_type`: a stable game-neutral goal name;
- `target`: measurable desired state or landmark;
- `constraints`: death risk, time, action, or region limits;
- `expected_duration`: predicted duration range;
- `success_condition`, `failure_condition`, and `interrupt_condition`;
- `parent_goal_id`: the higher-level reason for this goal;
- `confidence` and model version.

### `HierarchicalTransition`

Every completed or interrupted span should record:

- episode, game, stage, task, seed, distribution, and scenario identifiers;
- hierarchy level, transition ID, parent ID, and child IDs;
- start and end frame plus real duration;
- state before, intended goal, and lower-level command sequence;
- predicted final state, reward, risk, duration, and confidence;
- actual final state, reward, terminal state, and termination reason;
- success, failure category, interruption source, and uncertainty;
- whether the record came from real play, scripted play, human play, model
  imagination, or later relabeling;
- policy, world-model, labeler, and schema versions.

### Episode And Span Boundaries

An episode begins when a stage environment is reset to a declared task and
seed. It ends when the environment reports task success, death or another true
terminal state, a configured time limit, or an explicit evaluator truncation.
A checkpoint, room, or obstacle does not silently create a new episode unless
the stage contract resets the environment there.

Each episode contains nested spans. A primitive span can end while its skill
continues; a failed skill can end while its tactic replans; a tactic can finish
while the route remains active. The stored reason must distinguish success,
failure, interruption, timeout, environment termination, and evaluator
truncation. This prevents a timeout from being learned as a successful natural
ending.

## Runtime Flow

A Block SMB decision might be represented as:

```text
Route:     reach the next checkpoint
Tactic:    reach the safe platform beyond the enemy
Skill:     clear the gap and finish grounded
Primitive: move right, jump, adapt the hold, then release
Control:   press RIGHT+A now
```

On every control step, the controller receives the active primitive and local
state. When an event occurs, such as liftoff, hazard entry, release, landing,
lost support, no progress, or death, the relevant levels update. The primitive
may adjust or stop; the skill may retry or return failure; the tactic may choose
a recovery skill; the route replans only when the local result changes the
landmark graph.

This keeps fast reactions fast while avoiding frame-by-frame route planning.

## Learning Flywheels

Every level should run the same basic improvement loop:

1. Play in the real stage with a declared goal and policy version.
2. Record the intended goal, lower-level commands, duration, prediction,
   outcome, confidence, and failure reason.
3. Segment the episode using observed events and learned boundary proposals.
4. Relabel each span with what it actually accomplished.
5. Train the level's policy, outcome model, stopping rule, and confidence.
6. Send failures, novel states, and uncertain transitions to focused practice.
7. Promote a new version only when held-out real play improves.

### Per-Level Signals

| Level | Learns from play | Focused replay |
| --- | --- | --- |
| Perception | Masked observations, next observations, object motion, and consistency across adjacent frames | Frames with disagreement, occlusion, fast motion, or transfer errors |
| Physical control | Multi-horizon position, velocity, support, collision, and death outcomes | States where predicted physics disagrees with Block or Full execution |
| Motor primitive | Actual displacement, duration, release timing, landing, interruption, and recovery | Sparse waits, left recovery, jump windows, overshoots, and failed landings |
| Skill | Final state relabeled as the achieved goal, plus success and failure boundaries | Unreliable goals, novel starts, near misses, and underused skills |
| Tactic | Transitions between safe local landmarks and obstacle configurations | Unknown graph edges, unreliable sequences, and recovery choices |
| Route | Landmark reachability, cumulative cost, risk, and level completion | Routes with stale estimates, bottlenecks, or unexpected local failures |
| Curriculum | Which practice produced measurable held-out improvement | Weak skills, uncertain transitions, regressions, and transfer gaps |

### Hindsight Relabeling

A failed attempt can still be useful data. If the agent tries to land on a far
platform but lands safely on a nearer one, store both facts: the intended goal
failed, and the nearer goal succeeded. The skill policy can then learn which
starts can reach the nearer state without pretending the original attempt was
successful.

Relabeling should prefer measurable state changes such as displacement,
support surface, landmark, hazard clearance, and terminal outcome. Free-form
names may be added for inspection, but training targets must remain grounded in
state the environment or perception stack can verify.

### Skill Discovery

Before reliable labels exist, candidate skills can be discovered by rewarding
three properties:

- diversity: different skills reach meaningfully different states;
- controllability: requesting a skill changes what the agent can cause;
- predictability: the outcome model can tell what the skill is likely to do.

Discovery should be constrained by useful outer objectives. A behavior that is
different but repeatedly dies or makes no progress should not consume the
skill library. Skills also need usage and redundancy checks so many IDs do not
collapse to the same movement.

## Physics Learning

The model should be physics-informed in the practical sense: it learns the
physical properties that matter for control from observed transitions. It does
not need equations for Mario's engine embedded in the network.

The physical model should predict at several horizons:

- player position and velocity change;
- grounded, airborne, and supported state;
- collision, support loss, hazard contact, and death risk;
- object and platform motion;
- likely landing region and time;
- progress and reward;
- whether the active primitive should continue, release, cancel, or replan.

Use a shared encoded state with level-specific transition heads. The primitive
head predicts frames to seconds; the skill head predicts goal states; the
tactic head predicts landmark transitions. Separate heads keep each prediction
measurable while allowing common visual and semantic features to transfer.

Train on real transitions first. Imagined transitions may expand practice only
after uncertainty is calibrated, and they must remain marked as imagined. A
policy cannot pass promotion solely by exploiting its own learned model.

Block-to-Full disagreement is valuable supervision. Store paired or comparable
states where the simplified and emulator physics diverge, then focus the world
model and curriculum on those regions.

## Iterative Estimation And Retraining

The hierarchy can use the maximum-likelihood and expectation-maximization
pattern from the universal oracle roadmap at every level.

1. **Estimate hidden structure:** infer likely skill identity, start and end
   boundaries, achieved goal, alternate tactic, and failure cause from stored
   play. This is the expectation step.
2. **Retrain from the estimates:** update policies, stopping rules, world-model
   heads, values, and confidence using observed labels plus filtered inferred
   labels. This is the maximization step.
3. **Play again:** collect trajectories that expose weak estimates.
4. **Repeat:** keep old held-out episodes unchanged so improvement is real.

Known labels, such as scripted primitives or environment events, use ordinary
maximum-likelihood training. Inferred labels need confidence thresholds and
provenance. Low-confidence records can train representation learning but should
not become authoritative action targets.

Lower-level policy updates make old high-level commands harder to interpret.
For example, the meaning of "jump forward" changes when a new primitive policy
jumps farther. Before retraining a parent, relabel old transitions with the
current child model when possible, retain the original executed actions, and
measure parent performance after every child promotion.

## Replay And Practice Selection

The replay store should balance records by level, goal, outcome, scenario
family, duration, and provenance. Sampling only common successful movement will
repeat the current imbalance where rightward actions overwhelm wait and
recovery behavior.

Each level receives a practice score based on:

```text
practice priority = failure + uncertainty + novelty + regression + transfer gap
                    - recent mastery
```

Weights should be explicit configuration, logged in artifacts, and evaluated
through ablations. Curriculum decisions must use training data only. Fixed
validation, held-out Monte Carlo test, and benchmark episodes remain read-only.

Replay must retain:

- successful, failed, interrupted, and near-miss spans;
- rare goals and recovery modes;
- the policy and world-model versions that generated each transition;
- actual child actions, not only the parent's requested goal;
- a stable validation slice that is never used for practice selection.

## Planning

Planning should begin over skills and landmarks, not raw frames.

The tactic model predicts which skill can move from the current local state to
a desired safe state, including success probability, duration, and risk. The
route planner treats known landmarks as nodes and learned tactic outcomes as
edges. It searches for a short, safe path and replans when an edge produces an
unexpected result.

Frame-level tree search is intentionally deferred. It is expensive, sensitive
to small physics errors, and duplicates work already handled by the controller.
Later search can refine a small set of high-risk skill sequences when the world
model is calibrated.

## Evaluation Gates

Every level needs direct measurements as well as end-to-end return.

| Measurement | Question answered |
| --- | --- |
| Goal success by family | Does the level achieve what was requested? |
| Duration calibration | Does predicted duration match actual duration? |
| Boundary and stopping accuracy | Does the span end for the right reason and at the right time? |
| Outcome error | Is the predicted final state close to the real final state? |
| Risk and confidence calibration | Are wrong predictions less confident? |
| Skill diversity and usage | Are learned skills distinct and actually selected? |
| Parent lift | Does adding the level improve the level above it? |
| Fixed and held-out pass rate | Does real task performance generalize? |
| Transfer lift | Does Block learning improve Full performance at equal budget? |
| Retention | Did the new version preserve previously mastered behavior? |

Each milestone should compare against the current flat or shallower baseline
under the same environment-step and compute budget. Promotion requires held-out
real-environment improvement, calibrated confidence, no collapse to one skill,
and no material regression on existing fixed gates.

## Milestones

### HSP0: Contracts And Instrumentation

- Define versioned temporal goal, transition, outcome, and termination schemas.
- Emit nested primitive spans from current Block SMB execution without changing
  policy behavior.
- Add event detectors for liftoff, release, landing, support loss, progress
  stall, hazard clearance, death, success, timeout, and truncation.
- Add replay manifests, provenance, policy versions, and hierarchy diagnostics.
- Add an episode viewer or report that reconstructs nested spans and goals.

Exit gate: existing fixed and Monte Carlo Block SMB runs can be reconstructed
as valid episodes and primitive spans with no missing frames or ambiguous end
reasons.

### HSP1: Persistent Primitive Flywheel

- Convert B-level primitive outputs into explicit outcome-conditioned commands.
- Train primitive duration, release, landing, displacement, cancel, and replan
  targets from complete spans rather than isolated frames.
- Add hindsight outcome labels and balanced replay for wait, left recovery,
  jump timing, and failed landings.
- Select checkpoints using held-out primitive metrics and scenario success.
- Keep the current scripted oracle as bootstrap data and regression sentinel.

Exit gate: persistent primitives improve fixed and held-out gap, platform,
wait, and recovery families over the current per-frame baseline without
regressing already mastered families.

### HSP2: Goal-Conditioned Skill Layer

- Define measurable local goals using displacement, support surface, hazard
  state, landing region, and safe-state predicates.
- Segment play into candidate skills and relabel achieved goals in hindsight.
- Train a skill policy, stopping rule, outcome model, and confidence estimate.
- Add constrained discovery for useful skills not covered by scripted labels.
- Merge redundant skills and retire consistently unused or unsafe skills.

Exit gate: one policy can execute at least clear-gap, mount-platform, wait-pass,
enemy-clear, and retreat-recover goals from held-out starts, and reports
calibrated failures when a goal is not reachable.

### HSP3: Tactic Manager

- Add a manager above the current A/B/C controller that selects a skill goal
  every 8-32 control decisions or when an event interrupts the active skill.
- Train tactic transitions from skill outcomes and local landmark changes.
- Add recovery selection after interrupted or failed skills.
- Correct old parent transitions when lower-level skill behavior changes.
- Compare learned tactics with scripted and single-skill baselines.

Exit gate: the tactic manager improves multi-obstacle, mixed-section, and
chained-gauntlet held-out success at equal environment-step budget.

### HSP4: Multi-Horizon World Models And Imagination

- Add primitive-, skill-, and tactic-horizon prediction heads on shared state.
- Train ensembles or equivalent uncertainty estimates on real transitions.
- Validate position, landing, risk, duration, and landmark predictions by
  scenario family and horizon.
- Permit short imagined practice only inside well-calibrated state regions.
- Detect model exploitation by comparing imagined gains with real rollouts.

Exit gate: model-assisted practice improves real held-out pass rate, and
prediction error plus confidence remain within declared thresholds.

### HSP5: Landmark And Route Planning

- Discover or define stable landmarks such as safe platforms, obstacle exits,
  checkpoints, and level goals.
- Build a graph whose edges are learned tactic outcomes with success, duration,
  risk, and uncertainty.
- Search the graph for routes and update edge estimates from real play.
- Replan on changed landmarks, failed edges, or excessive uncertainty.
- Add optional search over a small number of high-risk tactic sequences.

Exit gate: route planning completes longer held-out layouts more reliably than
the tactic-only model and can explain failure as a specific missing or
unreliable graph edge.

### HSP6: Cross-Episode Curriculum

- Rank practice by failure, uncertainty, novelty, regression, and transfer gap.
- Allocate play budgets across skills, tactics, routes, and scenario families.
- Track learning progress so mastered tasks decay in priority without being
  forgotten.
- Generate or select Block scenarios that isolate weak transitions.
- Keep all benchmark and promotion sets read-only.

Exit gate: automatic practice selection reaches the same promotion gate with
fewer environment steps than uniform sampling and preserves mastered skills.

### HSP7: Universal Oracle And Cross-Game Integration

- Extend `UniversalOracleTrace` with temporal goals, nested spans, achieved
  goals, boundaries, and parent-child provenance.
- Share game-neutral primitive and skill concepts while retaining game-specific
  mappings and outcome fields.
- Use oracle EM relabeling for uncertain boundaries, skills, and tactics.
- Feed Full-fidelity failures back into Block practice and physics learning.
- Evaluate transfer to at least one non-SMB retro-style game profile.

Exit gate: the same temporal contracts and learning loop improve held-out play
for at least two game profiles without adding trainer-specific hierarchy logic.

## First Experiment

The smallest useful experiment is a persistent primitive and one skill layer on
a restricted Block SMB suite:

1. Use flat run, single gap, variable pit, platform, wait bridge, and left
   recovery scenarios.
2. Collect scripted and current-policy episodes with full event traces.
3. Segment action runs into approach, jump, release, landing, wait, and recovery
   spans.
4. Train primitive outcome and stopping heads on complete spans.
5. Relabel safe final positions as achieved local goals.
6. Train a goal-conditioned skill policy for grounded displacement and safe
   landing.
7. Evaluate fixed scenarios, untouched Monte Carlo seeds, calibration, rare
   action usage, and span-level timing against the current policy.

This experiment is successful only if executable action spans improve. Lower
per-frame imitation loss by itself is not sufficient.

## Implementation Map

Likely ownership boundaries are:

- `retroagi/core/hierarchy.py`: level specifications and parent-child runtime
  coordination;
- a new core temporal-contract module: versioned goals, transitions,
  termination reasons, and serialization;
- `retroagi/core/models.py`: level-conditioned policy and outcome heads after
  contracts stabilize;
- Block SMB environment and training modules: event extraction, span replay,
  balanced practice, and first evaluation gates;
- Full SMB training and diagnostics: compatible event extraction and transfer
  comparison;
- universal oracle tooling: EM labels, confidence, provenance, and cross-game
  temporal traces;
- artifacts: hierarchy manifests, per-level metrics, replay summaries, graph
  snapshots, and promotion reports.

The core modules should remain game-neutral. Stage adapters translate native
observations and actions into measurable goals and events. A new level should
be registered through contracts and configuration, not hard-coded into every
trainer.

## Risks And Controls

| Risk | Control |
| --- | --- |
| Many skill IDs learn the same behavior | Diversity, usage, and outcome-distance gates; merge redundant skills |
| Novel but useless behavior dominates | Anchor discovery to progress, survival, coverage, or uncertainty reduction |
| A policy exploits errors in its world model | Promote on real held-out play; constrain imagination by uncertainty |
| Lower-level updates invalidate high-level replay | Store executed child actions, version policies, and relabel parent transitions |
| Common movement overwhelms rare recovery | Balance replay by goal, outcome, and family; report coverage |
| The hierarchy becomes slow or brittle | Event-driven decisions, bounded candidate sets, and level-by-level ablations |
| Training forgets mastered skills | Frozen validation slices, retention gates, and replay of prior competencies |
| Evaluation data leaks into curriculum | Immutable split manifests and read-only promotion sets |

## Non-Goals

- Do not add every proposed level in one change.
- Do not rename the current A/B/C representations and claim that duration was
  solved.
- Do not replace the existing short-range controller before persistent
  primitives are demonstrably better.
- Do not promote a policy from imagined rollouts alone.
- Do not require hand-authored names for every discovered skill.
- Do not let benchmark failures directly train the model unless the run is
  explicitly declared adaptive and evaluated later on a fresh held-out set.

## Research Basis

The implementation should borrow mechanisms, not copy a single architecture:

| Work | Useful idea |
| --- | --- |
| [Options](https://www.sciencedirect.com/science/article/pii/S0004370299000521) | A temporally extended action has a start rule, internal policy, and stopping rule. |
| [HIRO](https://arxiv.org/abs/1805.08296) | A higher level can set state-space goals while replay is corrected as the lower policy changes. |
| [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) | Failed attempts can be relabeled with goals they actually achieved. |
| [DIAYN](https://arxiv.org/abs/1802.06070) and [DADS](https://arxiv.org/abs/1907.01657) | Reward-free discovery can seek diverse, controllable, predictable skills. |
| [Director](https://arxiv.org/abs/2206.04114) | A manager can learn long-horizon goals through a world model. |
| [Successor Feature Landmarks](https://arxiv.org/abs/2111.09858) | Discovered landmarks can support planning across long distances. |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | A learned world model can support broad behavior learning when grounded by real experience. |
| [MuZero](https://arxiv.org/abs/1911.08265) | Search can use a learned model without reconstructing every environment detail. |

The practical recommendation is a hybrid: persistent options for execution,
hindsight for dense learning, world models for outcome prediction, landmarks
for long-range search, and a curriculum that focuses real play on uncertainty
and failure. Every part remains answerable to held-out environment outcomes.
