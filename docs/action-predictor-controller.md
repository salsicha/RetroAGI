# Action, predictor and adaptive controller (approved redesign)

Status: approved (2026-10-10), being built; the target tracker's architecture
(section 3) is the recommended one, and the adaptation network is to be worked
out later. The
current design is described in the [architecture and training
guide](layered-agent.md); decisions and their status in
[design decisions](design-decisions.md).

## Why

Today the skill outputs a movement mode and an exact destination in pixels
relative to Mario's feet, and the executor turns that into button presses.
The skill therefore has to be a precise geometry calculator. The problems of
2026-10-09 and 10 came from that:

- **Single values and short lists.** The skill learned destinations such as
  72 pixels and applied them where they were wrong (running into an enemy
  with a jump-over's distance), and families taught a few discrete values
  instead of a relationship with the scene.
- **Imprecise distances.** Runs to a ledge stopped 7 to 23 pixels short, and
  the next run overshot into the pit: the skill learned typical run lengths,
  not the distance to the edge.
- **A prediction it cannot use.** The scene memory predicts the scene at the
  end of the next action, but it is not told which action; its prediction
  cannot be the target of that action.

## The hierarchy after the change

```text
strategy switch -> tactic (unchanged)
                 -> skill: chooses an ACTION (a verb aimed at an object it sees)
                 -> action predictor: the successful END STATE of that action,
                    relative to its target object, and when it is reached
                 -> target tracker: where that object will be at that time;
                    every frame it checks the world against its prediction
                    and, when they differ, sends an updated target
                 -> adaptive predictive controller: buttons and durations to
                    reach the target; it re-plans on an updated target but
                    models and tracks nothing in the world itself
                    <- adaptation network: updates the controller's motion
                       model from sensed changes in how Mario moves
```

The skill decides what to do; the action predictor works out how it ends; the
target tracker says where the target will be; the controller works out how to
get there.

## 1. Actions

An action is a verb and the object it is aimed at, one of the objects the
vision sees (a surface, an enemy, the edge of a gap). The verbs
(`core/action_tokens.py`):

| Verb | Object | Meaning |
|---|---|---|
| hold | none | stay put (also on a moving platform) |
| run to | a surface | walk or run along a floor to a point on it, or off its end onto a lower one |
| approach | what the next jump is aimed at | run up to the takeoff of the jump that follows |
| land on | a surface | jump onto a platform, step or floor |
| jump over | an enemy or a gap | jump past it |
| stomp | an enemy | jump onto it |
| back away | an enemy | run away from it, away from the goal's side |

The object is named by a pointer to one of the scene encoder's object tokens
(a list and a slot of `smb_observer.packed_lists`: 60 slots). Each verb only
applies to some kinds of object (stomp to enemies, land on to surfaces);
others are masked.

The skill keeps its transformer; its output heads become a verb choice and a
pointer that picks one of its scene tokens. It no longer outputs x and y.

**The teacher's actions** (`stages/block_smb/teacher_actions.py`, training
labels only). Each command the teacher chooses is named by what its trial
did:

- a jump that kills an enemy stomps it;
- one that ends on the other side of an enemy that stays alive jumps over it;
- any other jump lands on the surface it ends on;
- a run before a planned jump approaches what that jump is aimed at;
- a run away from the goal's side with an enemy within 96 pixels backs away
  from the nearest one;
- any other run runs to the surface it ends on.

The simulator's object is matched to the slot the vision reports: an enemy
by its box's middle (within 12 pixels), a surface by its top (within 3
pixels) under the point (a point beyond the visible window is matched at its
edge). Every jump Mario can make at a decision (the teacher's jump table) is
named the same way; jumps naming the same action are versions of it, and
the action's outcome is the safe version with the most room.

An outcome is measured from Mario's feet at the decision (pixels): where
they end, where the action's object is then (an enemy's middle and top, a
surface's left edge and top, moved by their own rules), how many frames it
takes, and whether Mario survives and wins.

## 2. The action-conditioned predictor

Told the action, it predicts that action's successful end:

- where Mario's feet end (relative x and y),
- how long it takes,
- where the target object is then (a walking enemy, a moving platform),
- the chance that the action succeeds, and the uncertainty of each.

It reads the scene memory's state (vision only, as now) and the encoded
action (the verb and the chosen object's token). Built as a separate small
network (`core/action_predictor.py`) reading a trained policy's scene encoder
and memory, which stay fixed: its query is the verb, the chosen object's
scene token (and, for an enemy, the memory's forecast of where it will be)
and the memory's state; it attends over the whole scene, so other threats
count. It outputs each quantity with an uncertainty (a normal likelihood is
its loss) and the chance of survival. Trained by
`stages/block_smb/predictor_train.py` on the teacher's episodes played through
the production pipeline with `EpisodeTask.record_actions`.

**Training data comes from the teacher's trials.** At every decision the
dense teacher already tries every jump Mario can make and screens every run,
each with its simulated outcome (2026-10-10). Every tried action, chosen or
not, becomes a training example: (scene, action) -> outcome. The outcome the
predictor learns for an action is its best execution: the most room from
every threat (the same objective the teacher uses). Targets are continuous
positions and times, so no list of values can be learned; the variety and
collapse checks stay.

## 3. The target tracker

The controller has no model of the world, so it cannot know where an enemy or
a moving platform will be. A separate, small model answers: for an object in
the scene and a time some frames ahead, where will it be?

Recommended architecture:

- **Object-centred and shared.** Each visible enemy, moving platform or plant
  runs through the same small recurrent network (a GRU of about 32 to 64
  units), all objects of all episodes batched together: microseconds a frame.
- **Inputs every frame, from vision:** the object's observed movement since
  the last frame, its kind and whether it is seen, its surroundings (the
  distance to the edge or wall ahead and behind on the surface it stands on,
  so a turn at a ledge can be predicted) and Mario's position relative to it.
  The recurrent state carries what one picture cannot show: direction,
  speed, and where a plant or platform is in its cycle.
- **Output, once per frame:** the whole path ahead, its position at a few
  times (1, 2, 4, 8, 16, 32, 64 and 96 frames) with an uncertainty and the
  chance it is still seen. Any time is read between those points without
  running the network again.
- **Corrections to steady motion.** It predicts how the object departs from
  continuing at its current velocity: walkers are almost free, and that is
  the safe default.
- **Sensing changes.** Each frame the observation is compared with the path
  predicted a few frames before; outside the predicted uncertainty (a
  walker turned, a platform reversed), the target is updated and only the
  new target is sent to the controller.
- **Training without a teacher:** every training episode logs each object's
  position every frame; from frame t, predict its positions up to 96 frames
  later (the simulator's truth where vision loses an object's identity).

Built as described: a 48-unit GRU; play records every frame's tracked
objects (`EpisodeTask.record_objects`: visual identity, kind, box in world
pixels, the ends of the surface under it, Mario's position), and
`ObjectTracker` keeps one recurrent state per identity at play, answers
where an object will be any number of frames ahead (read between the
horizons), and reports a change when the object is outside three
uncertainties of the forecast made 4 frames before (forecasts made before
its movement was seen are not checked).

Not chosen: the scene memory itself (scene-wide, stepped every 4 frames and
at decisions, trained for one moment: too heavy to run every frame and a
different job); a fixed window of recent frames instead of a recurrent state
(cannot know the phase of a long cycle); a transformer over all objects (only
needed when objects affect each other, as a shell hitting enemies in Full
SMB: the upgrade path).

## 4. The adaptive predictive controller

It has no learned weights in its control loop and no model of the world.
Given a target (a position and a time), it computes the buttons and how long
to hold them, predicting only Mario's own motion. When the target tracker
sends an updated target, it re-plans from where Mario is. Re-predicting
Mario's path every frame during a flight (`smb_trajectory.REPLAN_IN_FLIGHT`,
off for Block SMB training today) is needed only for that re-planning, and
must be made faster first (it was half the cost of every teacher trial).
Today a flight re-plans once (`Flight.replan`) when vision corrects Mario's
speed during it: a jump made before his speed was seen (one picture cannot
show it; `pit_leap` and `platform_hop` start him running) plays the rest of
its jump for the speed he really has.

**Adaptive.** Its motion model of Mario has parameters: acceleration,
friction, gravity, jump forces, air control, now fixed NES constants. A
further deep network, the adaptation network, will update these parameters
from sensed changes, comparing the motion the controller predicted for Mario
with the motion vision then shows (ice, water, another game's physics). To
work out later:

- its inputs (recent prediction errors, the scene, the action);
- its outputs (corrections to the motion parameters, bounded so the
  controller stays stable);
- how it is trained (Block SMB layouts with varied physics);
- how often it updates (per frame, per action).

**Proposed design (to confirm).**

- *What it adapts:* the groups of constants of Mario's motion model
  (`NESPlayerMotion`): ground acceleration and top speeds, braking and
  skidding, the jump's starting speed for each running speed, gravity while
  A is held and after, steering in the air, and the fastest fall.
- *Inputs:* the last 32 frames of what the controller predicted Mario would
  do with the buttons it pressed against what vision then showed (his
  movement per frame, the camera's scroll removed), with the buttons, whether
  he was on the ground, and the parameter values in use. Nothing about the
  world: the errors carry the physics.
- *Network:* a small recurrent network (a GRU of about 32 units) over that
  window, giving one bounded correction per group (a factor between 0.5
  and 1.5) and how sure it is; a correction is applied only when sure, and
  smoothed over time so the controller never sees a jump in its model.
- *When:* at action boundaries (never during a flight, whose plan was made
  with the old model), and only after enough frames of each kind of motion
  have been seen (a group with no new evidence keeps its value).
- *Training:* Block SMB layouts played with varied physics (ice, water, low
  gravity, faster running: each episode draws a variant), where the true
  parameters are known: the target is the variant's parameters, and the
  loss is also the controller's own prediction error over the next window
  with the corrected model. Held-out variants test it.
- *Test:* Full SMB's water levels, whose physics differ from the rest.

## 5. Teacher and training

- **Labels.** The teacher labels the action it chooses (the move choice by
  strategy and room, as now) and gives the outcome of every action it tried.
- **Order.** The predictor first (a model of what actions do), then the skill
  (which action), then the tactic. Possibly the predictor and the skill
  together.
- **Qualification.** As now: the teacher must win every layout a run can use,
  through the production pipeline; additionally, the predictor and
  controller must reach the teacher's outcomes.
- **Validation.** The formal validation proposed in the design decisions
  (a specification per family, teacher verification, data validation and
  policy validation) applies to actions and outcomes.

## 6. What carries over

- **Kept:** the dense teacher (its tried jumps and runs become the candidate
  actions and their outcomes), the variety of the pit and enemy families,
  room around every threat, the strategy's choice between stomping and
  jumping over, the collapse checks, regularization, and validating the
  teacher on every layout before training.
- **Becomes the predictor's targets:** the teacher's best landing for each
  action.
- **Obsolete:** the skill's x and y outputs and their ordered choice (mode,
  then x, then y), and the destination labels themselves.

## 7. Open questions

1. The verb list, and how a run-up before a jump is expressed (its own verb,
   or part of "land on" and "jump over", with the controller placing the
   takeoff). A run along the floor Mario stands on has no end its object
   decides: the teacher runs 8 pixels to wait for an enemy and 196 to the
   goal, both "run to" that floor, so the predictor can only learn a mean
   (36 pixels off in the first test). The run's end must come from an object
   (a gap's edge, an enemy, the end of the floor) or a further choice. "Land
   on" a wide floor has the same gap: where on it depends on the threats
   around (8 pixels in one scene, 45 in a similar one). A surface split at
   the threats on it (the floor between a pit and an enemy as one object)
   would make the end follow from the object.
2. Whether the predictor is a readout of the scene memory or its own network.
3. Whether the skill reads the predicted outcomes of its candidate actions
   (choosing by predicted success and room) or only the scene.
4. The controller's speed budget, and how often the target tracker runs and
   how far a divergence must go before a target is updated.
5. The adaptation network: inputs, outputs, training, update rate.
6. Full SMB: objects across screens; the checkpoint training plan (a
   checkpoint about every 500 pixels, scored by time or points).

## 8. Proposed steps

1. Define the verbs and the action encoding; make the teacher emit an action
   (verb and object) for each choice, and the outcome of every action tried.
   Done for jumps (2026-10-10); runs are labelled for the chosen run only.
2. Build the predictor; train it on the teacher's outcomes; measure its error
   in pixels and frames on held-out layouts. Built and trained (2026-10-10,
   `artifacts/block_smb/action_predictor_20261010b`): 912 teacher episodes
   of the validated layouts (3 rounds x 8 per skill family), 342 held out.
   Held-out error of where Mario's feet end: land on 10.7 pixels, jump over
   9.6, stomp 8.7, approach 9.0, run to 13.6 (hold 1.9); duration 6 to 11
   frames; survival predicted better than its base rate (stomp 96.8% against
   85.5% survived). Fitting the values by squared error (not only the
   likelihood, which let a wide uncertainty excuse a poor value) halved the
   errors. A plain linear fit from the target surface's numbers gets "land
   on" to 18 pixels: the rest needs the threats around it.
3. Build the target tracker; train it on logged object positions; measure
   its error at each time ahead and how fast it notices a change. Built
   (2026-10-10, `core/target_tracker.py`, `stages/block_smb/tracker_train.py`):
   on synthetic walkers it learns where they turn (64 frames ahead: 5 to 8
   pixels off, against 26 for steady motion) and its check notices a turn
   within 4 frames. Trained on the same episodes' tracked objects (450
   tracks; walkers, a few moving platforms): it matches, but does not yet
   beat, steady motion averaged over 16 sightings (3.3 pixels off 64 frames
   ahead; the last frame's movement continued was 21 pixels off). These
   walkers seldom turn within the horizon; it needs data with turns and
   reversals (patrols, moving platforms, plants).
4. Give the skill verb and pointer heads; train it on the teacher's actions.
   Built (2026-10-10): `SkillLayer` chooses a verb (verbs with nothing in
   view to aim at are excluded), then a pointer reading the verb (the
   decision token scored against each object's scene token after the
   blocks; absent objects and kinds the verb cannot aim at excluded). Every
   labelled skill decision also records the teacher's action
   (`label_verb`, `label_pointer`, `label_action_valid`), taught with cross
   entropy beside the old mode, x and y heads, which still drive the
   executor until the controller takes actions (step 5). Old checkpoints
   load with the new heads untrained.
5. Make the controller reach targets and re-plan on updated ones, fast
   enough for training. Wired (2026-10-10, `core/action_controller.py`):
   `SMBAgents(predictor=, tracker=)` executes the skill's own action at the
   predictor's end state (in the verb's mode), binds an action on an enemy
   or moving platform to its visual identity (Mario's predicted end from the
   object's predicted point), steps the tracker every frame, and when the
   object leaves its forecast moves the destination: a flight plans the
   rest of its jump again (`Flight.replan`), a run moves where it stops.
   Without the two models the agent plays exactly as before. Checked with
   untrained models only (wiring), and on a synthetic turning walker.
6. Document the adaptation network's design (details later). Proposed in
   section 4 (2026-10-10), to confirm.
7. Validate the teaching on every layout, then train.
