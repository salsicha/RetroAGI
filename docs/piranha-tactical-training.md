# Joint piranha tactics, skills, and primitive training

Piranha family revision 3 adds a real departure-timing problem and explicit
tactical supervision. Family IDs are unchanged; demonstration contract 11
requires freshly collected data. Old revision-2 scores are not directly
comparable with revision-3 scores.

## Practice tasks

Every difficulty samples two types of crossing, independently of difficulty:

- **Clearance:** the original short plant and conservative route. Mario can
  clear the obstacle while the plant is fully exposed.
- **Timed:** an 80-pixel plant makes full-exposure clearance impossible from
  the ground. Mario approaches a staging point 32 pixels before the pipe,
  brakes early enough to stop inside it, observes retraction, and jumps during
  a certified hidden interval. A certified departure found earlier on the
  approach, including a running one, is taken instead of stopping.

Timed pipe widths are 36/44/52 pixels, versus 32/40/48 for clearance pipes.
The policy's state features do not encode pipe width, so while a plant is
hidden the two classes look identical until the plant has been seen. The
peak-exposure memory input carries that sighting forward. A private scenario
flag never enters the policy input or the executor. Pipe heights remain
22–26/30–34/38–42 pixels. Timed plants remain hidden for 48–64 frames;
rise/retraction lasts 12–20 frames and full exposure lasts 40–64. Initial phase
is sampled over the entire cycle.

The timed teacher uses the same visible-enemy history as the policy adapter.
An initially empty pipe has unknown phase and is not a departure cue. The
teacher requires an observed disappearance and certifies candidate holds
against the shortest allowed hidden interval and fastest emergence, with a
rounding margin, so any disappearance younger than 48 frames may be probed.
It never uses the episode's hidden phase or actual remaining timer to choose a
departure. Physics probes restore the full environment state. Timed routes are
regenerated after a phase change; replaying a stale time-indexed action list is
intentionally not guaranteed safe.

Each timed layout also contributes an overshoot correction: an unsupervised
prefix walks past the staging window, then the teacher brakes, retreats, waits
and departs. Canonical routes no longer overshoot, so without these rows
retreat would be taught only by sparse policy-recovery rows, upweighted the
same way.

The existing goal, progress, contact-death, and time rewards remain in use.
There is no reward for predicting a stance or for accumulating wait time.

## What learns

1. **Tactics:** explicit cross-entropy targets for `advance`, `hold_area`, and
   `retreat` at grounded, uncommitted decision states. Other families and flight
   continuations have no tactical label. Timed online rollouts receive labels
   at their actual observed states; clearance demonstrations label their
   successful action sequence.
2. **Skill A:** existing action imitation and reward-based learning choose the
   motor command. Tactical stance probabilities now feed the context supplied
   to A, so the stance head is a functional input rather than only a diagnostic.
3. **Primitive B:** existing safe-set duration supervision teaches certified
   jump holds, with an interior preference in demonstrations. Unsafe temporal
   departures do not receive a misleading "hold jump longer" target.

The tactical transformer now has feature-position information: its old
attention-and-mean-pooling path could not distinguish permutations of the
scene's scalar fields. Positional gain and the stance-to-context projection
start at zero to preserve old checkpoint inference before new training.

The default `tactic_loss_weight` is 0.5, applied both online and during
demonstration bootstrap/rehearsal. The full-volume recipe enables motion and
hazard observations. Joint batches update tactics, A, and B; no oracle stance
or action is injected during autonomous evaluation.

Plant waits reobserve every physics frame in both crossing modes. Their duration
head is not trained, because the executor does not consume a duration for those
waits. The executor previously keyed this on the private `timed_crossing` flag,
so a wait chosen at a clearance pipe, which looks identical while its plant is
hidden, committed 4–64 frames and could not react to a retraction. Recovery
collection detects premature departures and missed opportunities, preserves
observation history across the failed prefix, and labels only successful
corrected suffixes.

## Reporting and verification

Training records `loss_tactic`, `tactic_supervised_steps`, and
`demonstration_tactic_loss`. Evaluation adds `piranha_crossing_modes` by
mode/difficulty, and `piranha_tactics` decision accuracy. Accuracy measures
agreement at visited labeled states; it is not a substitute for episode success.

Regression tests cover shifted phases, fatal contact, unknown empty pipes,
hidden-timer independence, physical executor replay, one-frame waits, repaired
prefixes, joint gradients, and an intervention showing that learned stance
changes alter A's logits. These checks establish the training mechanism, not a
policy above 90% success. A new training run and held-out evaluation are needed
to measure learned performance.

Use the existing full-volume launcher with its updated default recipe and a
fresh output directory. Recollect demonstrations rather than loading contract-10
caches. Existing running processes do not acquire these changes automatically.


## Robust teachers and peak-exposure memory (contract 13)

A piranha-only diagnosis traced most failures to clearance pipes. While a plant
was hidden, a timed pipe and its clearance twin produced identical state
features, so the policy paused at clearance pipes, and the executor committed
those waits for 4–64 frames. Every timed route overshot its staging window and
backed up; timed routes ran up to about 250 of the 320 evaluation frames.

`hazard_memory_observations` appends `enemy_peak_exposure`: the tallest exposed
height of the most recently seen enemy this episode, divided by 64 and capped at
1, and zero before any sighting. The shared Block/NES history computes it; it is
versioned separately from the six v1 hazard features, and checkpoints, runtime
contracts and cached demonstrations must match. It occupies one C-stream slot
taken from the pooled vision summary, so no model shape changes, but earlier
checkpoints are rejected. The full-volume recipe now obtains this memory from
the world-model LSTM instead (next section); the joint and family learning
tools still offer the input through `--hazard-memory-observations`.

Timed-plant training rollouts now last at least the teacher's completion time
plus one plant cycle and 32 frames, so a missed window can be retried within
the episode. Evaluation budgets are unchanged.

Demonstration contract 13 requires regenerated plant demonstrations; the
joint-learning tool rejects older cached plant routes. Start a fresh
full-volume run; running processes do not acquire these changes.

Clearance demonstrations take off at the first feasible frame, with one
certified hold in 84% of routes. A revised clearance teacher also credited
landings anywhere past the plant, took off at the middle of the largest
certified set, varied takeoffs between routes and waited for unseen plants. It
lowered held-out clearance success in the piranha-only probe below and was not
adopted. Nor was restricting timed tactic-action targets to the teacher's own
motor command (walk only while approaching, jump only when certified): it cut
clearance success from 22 to 12 of 22 in the same probe, apparently by
suppressing jumps near every visible plant.

Verification: across 60 training layouts with four routes each, every timed
teacher route replays through the real executor without a death. Canonical
timed routes contain no retreat frames and average 145 frames (maximum 204);
28 of 116 departures are running jumps.

A piranha-only probe fitted 4,000 demonstration updates on 90 training layouts
and evaluated 60 held-out validation layouts greedily with the normal executor,
counting successes within 320 steps. The previous code scored 41/60, with 15 of
22 clearance layouts timing out. With the same weights, the executor fix alone
scored 53/60. This revision scored 58/60 and 52/60 on two seeds, and 51/60 on one
seed with overshoot corrections. Seed-to-seed differences that large mean the
probe does not separate the memory and teacher changes from the executor fix;
the probe also has no online learning or recovery, which the overshoot
corrections target. A full-volume run and held-out evaluation are required.


## LSTM episodic memory

The world-model LSTM can hold the same memory as the peak-exposure input, so the
adapter need not compute it for the policy. `architecture_config`
`world_model_memory_dim: 1` adds two pieces to the world model:

- a 32-unit encoder that gives the LSTM individual C-stream slots;
- a linear memory head on the updated hidden state, trained to report
  `enemy_peak_exposure` with weight `world_model_memory_weight`.

The actor reads the carried hidden and cell state through
`world_model_actor_context`, as before, and has no explicit memory input. The
strategy network's rolling stance history stays reset in this mode, because
batched demonstration fitting cannot train it.

Demonstration rows are sampled independently, so each row needs the state that
a rollout would carry into it. Collected demonstrations now record each row's
position in its episode and the observable memory target.
`refresh_demonstration_memory` replays every stored episode in order with its
demonstrated actions and the current weights, starting from an empty state, and
stores the state entering each row. Updates then fit each row from its stored
state, so gradients reach the LSTM through the memory head, the dynamics
prediction and the actor context. States are refreshed every
`memory_refresh_interval` updates and at the start of every fit. Recovery
demonstrations start their memory at the first supervised row. Online rollouts
record the prediction and its target, and the trainer adds the same loss.
Batched demonstrations with recurrent state are rejected unless the model has a
memory head and a positive refresh interval.

The full-volume recipe enables recurrent state, `world_model_memory_dim: 1`,
`world_model_memory_weight: 1.0` and `memory_refresh_interval: 250`, and turns
`hazard_memory_observations` off. The joint and family learning tools evaluate
in batches without carried state, so they drop these settings and stay
feedforward. A refresh replays every stored row once: 49,000 piranha rows took
2.2 seconds on the development GPU, about the time of 50 updates. Its cost
grows linearly with the dataset, and it runs 40 times during a 10,000-update
bootstrap and four times per 1,000-update rehearsal.

Verification: the piranha-only probe from the previous section (90 training
layouts, 4,000 updates, 60 held-out layouts, successes within 320 steps) was
run on four seeds for each form of memory:

| Seed | LSTM memory | Peak-exposure input |
| --- | --- | --- |
| default | 58 | 52 |
| 7 | 42 | 52 |
| 11 | 53 | 48 |
| 13 | 55 | 46 |

Across the four seeds the LSTM memory succeeded on 145 of 152 timed layouts and
the explicit input on 137. Clearance results were 63 and 61 of 88. With the
LSTM state reset every frame, the default-seed weights fell from 58 to 45. The
policy therefore uses the carried memory.

The memory is approximate. Across evaluation frames its mean absolute error
was 0.16–0.24, against targets between 0 and 1. States go stale between
refreshes: 250 updates after a refresh, the squared error on the states used
for training was half that on freshly replayed states. Refreshing every 50
updates or raising the weight to 5 did not reduce the error after 4,000
updates. More updates did reduce it: freshly replayed error fell from 0.23 to
0.10 between 2,000 and 4,000 updates. A full-volume run and held-out
evaluation remain the deciding test.
