# Choose and control: the plan

Approved by the user on 2026-11-07 ("Yes I agree, make it so"; the design
below as revised the same day: "Yes, and go ahead and implement"). This
replaces the layered agent's approach (docs/layered-agent.md), where networks
predicted where actions end and how long waits last. The background, the
evidence from this project and the design are in the report
[Choose and Control](https://claude.ai/artifact/1CaL7MDpVQsL4VCCS2DA6V).

## The design

User, 2026-11-07: "Learning a skill means 1 learning what is about to
happen, and 2 learning what to do about it"; the fixed rules of SMB are
"1. Don't die, 2. Progress to the right, 3. get coins and power-ups"; a
general player "doesn't learn long-term consequences ... The levels could
be randomly generated and I could still pass them."

1. **The executor**, a fast recurrent adaptive controller, every frame. An
   action is injected into it as parameters: jump onto an object (near,
   middle, far), stomp or jump over an enemy, run to an object or a
   distance (keeping speed or stopping), wait until a condition (an enemy
   clear of a point, a jump due, at most N frames), retreat. It plans with
   a model of the player's motion from fresh observations, carries the
   action out, reports events (arrived, landed, a wait's condition met, a
   threat, infeasible) and can override with a reflex. Its motion model is
   the NES physics for now; for any game it will be learned online.
2. **The outcome network**: from the current state and the action in
   progress, one pass predicts the consequences of continuing and of every
   alternative action at once (a head on each object's vector for the
   actions aimed at it, a few general outputs for the rest): the chance of
   dying within the action and a short window after it, progress to the
   right and the frames it takes, coins and power-ups expected, the chance
   of getting stuck. Trained on real outcomes: on randomly generated levels
   faithful to the real game's rules, states are saved and every candidate
   action is played from them by the same executor (no teacher, no level
   memory, nothing learned on Full SMB).
3. **The chooser**, the three rules on those predictions: the action in
   progress is kept while its predicted death chance is under a threshold
   and it still progresses (or is a needed wait or retreat); otherwise,
   among the actions least likely to die, the most progress right per
   frame, then coins and power-ups worth a little progress. Waiting and
   retreating come out of the rules: when every forward action risks death,
   a wait that does not wins, until a forward action is safe again. The
   same pass runs every few frames on the action in progress, so a rising
   risk picks again at once.

Timing: the executor every frame (about a millisecond); the outcome network
every few frames and at events (about a millisecond a pass); the next choice
decided ahead while the current action runs, checked at its event.

## What is dropped

- The old world model, which predicted where a whole action ends from
  teacher labels: the outcome network predicts every action's consequences
  from sampled real outcomes, and the executor computes motion.
- The collision tracker networks (the executor's rollouts check contact).
- The teacher and copying its choices (fragile: a mistake leads where it
  never showed); value functions and long-term consequences; level memory.
- The tactic memory network; the strategy, tactic and skill layers as
  separate networks.
- Waits in 8-frame slices (waits are conditions the executor watches).
- The long curriculum of hand-made lessons and 12-round recordings: random
  levels from the real game's pieces and rules.
- Next-frame prediction and pretraining.

Nothing learns on Full SMB (user, 2026-10-28). Reading the real game for
layouts and test states is allowed.

## Phases

Each phase ends with a measurement that decides whether to go on.

| Phase | Work | Done when |
|---|---|---|
| 0. Test bench | Saved game states just before each 1-1 obstacle; success per obstacle | One command scores any player on every obstacle in minutes (done) |
| 1. Executor | Goal regions, the jump planner, waits, events; driven by a scripted chooser on game memory, then on perception | The scripted chooser passes most obstacles on perception as on game memory |
| 2. Random levels | Block SMB levels generated from the real game's pieces and rules: its enemy wake-up and turning rules, its physics checked against game memory | The scripted chooser passes random levels as it passes 1-1 |
| 3. Outcome sampling | The action set; at sampled states every candidate action played from a saved state by the executor, its outcome recorded | Millions of labelled (state, action, outcome) examples an hour |
| 4. Outcome network and chooser | One pass scores every action; the three rules choose and monitor | On fresh random levels it plays as well as the scripted chooser; then the Full SMB bench and full 1-1 playthroughs, nothing learned on Full SMB |
| 5. Decide ahead | The next choice computed while the current action runs | No frame waits for a decision |

## Progress

- **Phase 0 (2026-11-07, done).** data/full_smb_obstacles/Level1-1: 24
  obstacles from the first Goomba to the flag, each with a checked start
  state (retroagi/stages/full_smb/obstacles.py, obstacle_bench.py). The
  learned agent of the old design (tactic_20261104) passes 60%.
- **Phase 1 (2026-11-07, done with game memory).** The jump planner
  (retroagi/core/smb_planner.py) and controller (smb_controller.py), driven
  by a scripted chooser reading game memory (stages/full_smb/
  scripted_chooser.py), pass all 24 obstacles in both modes and finish 1-1
  from its start: speed run 2,990 frames, max points 3,473 (score 500). The
  motion model's predicted paths matched the game frame for frame; what had
  to be fixed was everything around it (walls, enemy speed estimates, the
  stomp rule, the bounce, walking off ledges, narrow steps, the bench's
  starts). Next: the same chooser on perceived state (vision and the
  executor's estimates), then phase 2.
- **Phase 1, 2026-11-08: truly resting starts and the fixes they showed.**
  The bench had saved the game state from the frame *before* the one whose
  memory said "at rest", so some starts were moving (1528's at half a
  pixel a frame, which perception, starting from zero, could not know);
  starts are now the checked frame's state, at rest, at least 40 pixels
  from the screen's left edge (where vision drew cut-off things as a 3-pixel
  pipe and an enemy). From those starts the game-memory player first
  passed 71%; the fixes:
  - a floor followed at once by something at most a block lower is a step
    down, run off (jumping onto 16-pixel steps from rest slid off them),
    while a pipe's end is still jumped down from with its landing checked;
  - no run ends past the end of a floor at a pit (a "run closer" after an
    infeasible jump aimed 40 pixels short of the far side: in the pit);
  - an enemy with a pipe or step between it and Mario waits for the pipe
    (Mario stood against 1-1's third pipe for 250 frames trying to pass a
    Goomba beyond it);
  - a gap in the floor labels exactly where an enemy stands is floor (the
    Goomba drawn over the ground split it: a fake pit);
  - in the air the controller checks its plan every 4 frames and corrects
    it (held shorter, released or braking a few frames) when the landing no
    longer meets the goal: Goombas woken by the scroll after takeoff waited
    where Mario came down (smb_planner.plan_flight);
  - game memory's motion now includes the jump's origin height and whether
    A was held (0x708, bit 0x80 of 0x0D): without them a mid-air roll-out
    kept the low gravity after A was released;
  - land goals match their target by its top and place, not its slot (the
    slots renumber as the camera scrolls; a mid-flight check took another
    step for the target).
  Result: all 24 obstacles in both modes, and 1-1 finished from its start:
  speed run 2,017 frames, max points 2,109 frames (score 500).

