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
  stops, observes retraction, and jumps during a certified hidden interval.
  Overshooting the staging point teaches retreat and another approach.

Timed pipe widths are 36/44/52 pixels, versus 32/40/48 for clearance pipes.
This observable geometry distinguishes the classes even when the plant is
hidden; a private scenario flag never enters the policy input. Pipe heights
remain 22–26/30–34/38–42 pixels. Timed plants remain hidden for 48–64 frames;
rise/retraction lasts 12–20 frames and full exposure lasts 40–64. Initial phase
is sampled over the entire cycle.

The timed teacher uses the same visible-enemy history as the policy adapter.
An initially empty pipe has unknown phase and is not a departure cue. The
teacher requires an observed disappearance within the past eight frames and
certifies candidate holds against the shortest allowed hidden interval and
fastest emergence, with a rounding margin. It never uses the episode's hidden
phase or actual remaining timer to choose a departure. Physics probes restore
the full environment state. Timed routes are regenerated after a phase change;
replaying a stale time-indexed action list is intentionally not guaranteed safe.

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

Timed waits reobserve every physics frame. Their duration head is not trained,
because the executor does not consume a duration for those waits. Recovery
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
