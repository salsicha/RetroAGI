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
  state (retroagi/domains/smb/nes/obstacles.py, obstacle_bench.py). The
  learned agent of the old design (tactic_20261104) passes 60%.
- **Phase 1 (2026-11-07, done with game memory).** The jump planner
  (retroagi/domains/smb/planner.py) and controller (smb_controller.py), driven
  by a scripted chooser reading game memory (legacy/full_smb/
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
- **Phase 1 on perception, 2026-11-08.** The same chooser and controller on
  what the Full SMB vision shows and the executor estimates: 0.73 of the
  bench before, **0.98** after, and **1-1 finished from vision alone** in
  the speed run (2,098 frames; max points died at x 1556). Game memory
  stays at all 24 and finishes both modes (2,380 and 2,420 frames, score
  1,200). What perception needed:
  - a floor gap where an enemy stands is floor also when vision's box runs
    a pixel below the floor's top;
  - Mario's body is his collision box (10 wide, 4 below the drawn top)
    inside the drawn sprite, which is 16 wide: the planner stood him on a
    16-pixel step past its edge (and the jump's origin height moves with
    it);
  - the executor's motion estimate takes a seen bounce (falling in the
    model, clearly rising on screen) as a stomp, and corrects its speed
    when its own moves over four frames are 2 pixels off the seen ones (a
    wall bump it did not know left it 0.37 pixels a frame fast);
  - the screen's left edge is a wall, and nothing solid lies inside
    Mario's drawn box (his cut-off sprite read as a 3-pixel pipe there);
  - a run ends when an enemy ahead will meet Mario within 48 frames at
    their closing speed, and the chooser counts such an enemy as ahead (a
    96-pixel run blind to enemies met a Goomba pair too close to jump);
  - a bounce can stomp the next walker too, and in the air with no landing
    12 frames clear of enemies, one 4 frames clear is taken over a plan
    known to hit;
  - after an infeasible pass: a jump back when the enemy meets Mario soon,
    else running from it, else waiting; a stomp target vision renumbered is
    found by place.
  Left: 11_enemies_1682 stalls in half its runs; max points on perception
  over the whole level. Next: phase 2, random levels in Block SMB.
- **Phase 2, 2026-11-08: random levels of the game's pieces.**
  retroagi/domains/smb/sim/levels.py builds a level from a seed out
  of 1-1's pieces, in the game's pixels (16-pixel tiles, the floor at 208):
  pits 2 to 4 tiles, pipes 2 to 4 tiles tall, staircases up and down with a
  valley or a pit between, brick rows at the game's two heights (Goombas
  sometimes on the upper one), single Goombas and pairs, a closing staircase
  and the flag. Goombas follow the game's rules (wake on screen, walk left
  at half a pixel a frame, turn at walls, fall off ledges) and the camera
  keeps Mario where the real game does (his left side at 115 on screen, a
  new level setting `scroll_x`; other layouts keep 85). BlockTruthState
  gives the controller what is on screen, as game memory does in the real
  game; `python -m retroagi.domains.smb.sim.levels --levels 100`
  plays the scripted chooser through them (0.5 ms a frame). On 100 levels
  it finished 99 speed runs and 98 max-points runs (from 92 and 98 at
  first). What the levels showed:
  - the simulator counted a head reaching a block's bottom edge as a bump,
    one pixel early (the game and the planner do not);
  - two steps of one height side by side are floor, not a pit;
  - an enemy beyond the pit the floor ends at waits for the pit;
  - a jump across a pit may land anywhere on a staircase down beyond it (a
    land goal over the steps, with no single height);
  - enemies on Mario's way include those at the foot of a staircase he is
    going down (smb_controller.on_the_way), for the chooser and for runs;
  - predicted walkers fall off ledges (a Goomba off a brick row) and walk
    off the screen's edge; standing ones report no fall speed.
  - a gap in the floor two Goombas wide, under a pair, is floor too; vision
    tracks smaller than 6 pixels (fragments in a pit) and ones starting in
    the screen's first 16 columns are not enemies.
  Effect on 1-1: game memory 24/24 obstacles; vision 0.92 (was 0.98; max
  points fails half its runs at the Goomba pairs of 05, 06, 10, 11). Full
  1-1 shifted with these changes: one run all four playthroughs finished,
  on the final code only game memory's max points did (the others died
  between x 1676 and 2020).
  Left: Goombas waking at the screen's right edge while Mario is in the air
  at full speed (seed 43), the kind of risk the outcome network is to learn;
  max points on vision at Goomba pairs. Next: phase 3, outcome sampling
  from saved states on these levels.
- **Phase 3, 2026-11-08: outcome sampling.** The choices
  (retroagi/domains/smb/choices.py): land near, middle or far on each of the
  8 nearest surfaces, stomp each of the 4 nearest enemies or pass it
  beyond or back, run 32 or 96 pixels forward keeping speed or 48 back,
  wait 8 or 24 frames (about 20 to 35 at a decision). At every decision
  point on a random level, retroagi/framework/sampling.py
  saves the simulator's state and plays every candidate from it through
  the executor, then puts the state back, recording: how the choice ended,
  death during it or after it (buttons released, until Mario has stood 16
  frames, waiting up to 64 for him to come down), progress right, frames,
  points, reaching the flag. The states come from the scripted chooser's
  play, a random candidate instead 30% of the time. The simulator no
  longer draws pictures when nothing looks at them (drawing was 88% of a
  step). 2,000 levels in both modes: 180,214 states, 3.75 million choices
  in 32 minutes on 14 processes (6.9 million an hour), 36 MB
  (data/datasets/block_smb/outcomes_20261108); 100 fresh levels for
  measuring (outcomes_fresh_20261108, seeds from 100000).
  Later the same day the outcomes gained "stuck" (below), the sampler ends
  a choice exactly as play does (the frame it ends on is played: a run's
  "blocked" comes 2 pixels short of the wall), and a choice that ends
  before its first frame counts as infeasible. The final set: 181,233
  states, 3.77 million choices from 2,000 levels with staircase tops one
  or two columns wide (2.7 million an hour with the stuck probes), packed
  in 8 parts.
- **Phase 4, 2026-11-08: the outcome network and the three-rule chooser.**
  retroagi/framework/outcome_network.py: a small transformer over the state's
  tokens (Mario, 24 surfaces, 8 enemies, x relative to Mario); a head on
  each surface's token gives its three landings, a head on each enemy's
  its stomp and two passes, a head on Mario's the runs and waits; each
  choice: the chance of death (during or soon after), of being infeasible,
  of reaching the flag, of leaving Mario stuck, and its progress, frames
  and points. Trained in 9 minutes on the V100
  (framework/training.py; data/outcome/outcome_20261108.pt).
  On 190,661 choices sampled on 100 fresh levels: deaths ranked at 0.989
  (ROC area), infeasible called right 98.3%, stuck 95.2%, progress within
  3.4 pixels. retroagi/framework/chooser.py applies the rules: keep
  feasible choices with death chance at most 5% (or within 2 points of the
  safest), then those not likely to leave Mario stuck, then the most
  progress a frame (in max points a point is worth 0.2 pixels a frame). It
  remembers the choices taken from each spot until Mario reaches new
  ground. Playing 100 fresh random levels: **100 finished in both modes,
  no deaths** (the scripted chooser: 100 and 99). On 1-1, nothing learned
  there, from game memory: full 1-1 **finished in both modes** (speed run
  1,895 frames, faster than the scripted chooser's 2,017; max points
  2,116), the obstacle bench 0.85 (the earlier model, without two-column
  staircase tops: 0.94 and both modes finished too). From vision alone:
  the bench 0.81 and full 1-1 **finished in max points** (3,047 frames,
  score 800; the speed run died at x 1686).
  What it took, each found by a stall on the fresh levels:
  - "stuck": one choice at a time the chooser walked into a valley between
    staircases (quick progress) and waited there; now after each sampled
    choice a few forward choices are tried from where it ended (Mario's
    speed kept) and the choice counts as leaving him stuck if none is
    feasible, safe and gains 16 pixels;
  - a run blocked at once counted as feasible and was taken 6,000 times
    at a wall;
  - the stuck probes first ran after Mario had stood and slid to rest, and
    before the frame a run ends on (which stops him against the wall);
  - a short memory of what was tried at each spot (two failing landings
    taken by turns; a staircase gone round and round);
  - staircase tops two columns wide in the random levels (1-1's second
    pair), else the network learned that a staircase top before a pit is a
    dead end and refused 1-1's.
  Left: on vision, big Mario (after a mushroom) confuses perception (his
  box changes height; his top half was read as an enemy, now dropped);
  power-ups are not in Block SMB. On game memory the bench's misses are
  two-step traps (passing back off a pipe into a valley with Goombas) that
  one choice's outcome and the stuck probe do not see.
- **Power-ups, 2026-11-09.** Block SMB now has the game's question blocks
  (hit from below: a coin, or a mushroom), mushrooms (moving at a pixel a
  frame, turned by walls, falling off ledges), big Mario (12 by 24; a hit
  shrinks him, then 120 frames unhurt; he breaks bricks) and points in the
  game's proportions. Random levels put question blocks among the bricks
  and over the floor (a mushroom in 35% of them; Mario starts big in 30% of
  levels, both above the game's so the network sees them). The planner
  gained "bump" and "collect" goals, the network bump and collect outputs
  and item tokens; sampling play bumps question blocks and collects items
  half the time it can. 4.24 million choices from 2,000 levels
  (data/datasets/block_smb/outcomes_20261109); the network
  (data/outcome/outcome_20261109.pt) ranks deaths at 0.991 on fresh levels.
  On 100 fresh levels with power-ups: 98 speed runs and 97 max-points runs
  finished (the scripted chooser 97 and 96); in max points it scores 59.9 a
  level (the scripted chooser, which ignores blocks, 7.7) for 6% more
  frames. In the real game's state question blocks are told from bricks,
  the power-up slot (game memory) or power-ups seen (vision) are items, and
  big Mario has his own collision box.
  On 1-1 that model looped at the second staircase again: it took landing
  on the two-column top for deadly. Tried on the real game from that state,
  every jump across from the top was safe; the label was wrong. In the
  window after a sampled choice Mario stands, buttons released, and landing
  at speed on that top he slid off into the pit. A death in the window now
  counts only when no probe from where the choice ended (the forward ones,
  or a brake back) survives. Retrained on 4.24 million choices
  (data/datasets/block_smb/outcomes_20261109b;
  data/outcome/outcome_20261109b.pt): fresh random levels 97 speed runs and
  94 max-points runs (more deaths than before: 3 and 6); on 1-1 from game
  memory the bench **0.90** (the staircases pass; deaths at the second
  pit and the 1965 Goombas), full 1-1 in max points finished with score
  1,600 (the speed run died at the second pit).
- **Koopas and grounded probes, 2026-11-09.** The planner now rejects a
  landing Mario would slide off into a pit (a pass over 1-1's second pit
  landed him on the floor's last pixels at speed). Block SMB gained Koopa
  Troopas as 1-1 has them (a 12 by 12 body; a stomp leaves a shell, a touch
  kicks it at 3 pixels a frame, a moving shell kills enemies and hurts
  Mario), a quarter of the random levels' enemy pieces. And a sampled
  choice that ends with Mario in the air plays on, buttons released, until
  he lands before anything is probed, as in play (probed in mid-air, a jump
  seemed to save him from the Goomba below 1-1's third pipe). Retrained on
  4.52 million choices (data/datasets/block_smb/outcomes_20261109d;
  data/outcome/outcome_20261109d.pt): fresh random levels 99 speed runs and
  100 max-points runs (the scripted chooser 97 and 95). **1-1 from game
  memory: all 24 obstacles, and full 1-1 finished in both modes** (the
  speed run in 1,789 frames, faster than the scripted chooser's best of
  2,017; max points with score 1,750). From vision: the bench 0.96; full
  1-1 died (x 1991 in the speed run, x 682 in max points, by the third
  pipe where vision also misses on the bench).

- **Vision finishes 1-1, 2026-11-09.** Five fixes to perception, the
  planner and the chooser, the network unchanged (outcome_20261109d):
  - perceived walkers get the game's collision box inside the drawn
    sprite (from its left side: a Goomba 10 by 6 ending 5 above the drawn
    bottom, a Koopa 12 by 12 ending 4 above it; read from game memory).
    With the drawn boxes a pass over a Goomba pair looked like a hit in
    the air, and the flight check braked Mario down in front of them;
  - a jump starts only while Mario would still stand on something 4
    pixels further the way he moves (TAKEOFF_KEEP): vision placed the
    second pipe's end 2 pixels too far, and a run-up and jump from its last
    pixels dropped him unjumped behind a Goomba;
  - a floor piece is dropped as a pipe's or block's own top only when the
    pipes and blocks of its height cover at least half of it: a made-up
    2-pixel pipe at a step's end had hidden the step Mario stood on;
  - a wait with enemies on screen forgets what was tried from the spot (the
    enemies have moved on), but landings, runs and bumps found infeasible
    or blocked there stay refused until new ground: beside the third pipe
    the chooser had waited, every way on tried, while a Goomba walked up
    from behind.
  **Full 1-1 from vision finished in both modes** (speed run 1,733 frames;
  max points 2,038 frames, score 800); from game memory both finished
  (1,789 frames; max points score 2,200). Benches: vision 0.96 (one slow
  max-points start, x 592, outlasts the 900-frame limit, counted for two
  obstacles), game memory 1.00. Fresh random levels 98 of 100 in each mode:
  seed 100036 lost to the takeoff margin (Mario slides to a pit's edge
  after dropping off a pipe, which the network, trained without the
  margin, does not foresee), seed 100018 to a Goomba on a high brick row.
- **Retrained with the margins; the slide check, 2026-11-09.** Outcomes
  resampled with the takeoff margin and the chooser's new memory (4.50
  million choices, data/datasets/block_smb/outcomes_20261109e; model
  data/outcome/outcome_20261109e.pt, deaths ranked at 0.992 on fresh
  levels). It plays fresh random levels better (99 of 100 in each mode)
  but 1-1 from vision worse: it landed Mario on the far end of the last
  staircase top, sliding, where every jump was refused. So the planner's
  slide check after a landing now uses the takeoff footing too: a landing
  he would still be sliding on within TAKEOFF_KEEP of an end with nothing
  below is no landing. With it, on 1-1:

  | | outcome_20261109d (kept) | outcome_20261109e |
  |---|---|---|
  | Vision bench | **1.00** (all 24) | misses at 17 and 19 (stopped after 22) |
  | Game memory bench | **1.00** | not run |
  | Vision full 1-1 | finished: 1,788 frames; max points score 1,300 | finished: 2,290 frames; score 500 |
  | Game memory full 1-1 | finished: 1,898 frames; score 2,300 | finished: 1,869 frames; score 1,300 |
  | Fresh random levels | 97 / 97 | **99 / 99** |

  outcome_20261109d stays the default: the real game is the test, and
  nothing learns on it.
- **World 5-1, 2026-11-10.** The second test level, played and measured
  only. Its game memory showed three things the random levels lacked, now
  in Block SMB as the game has them (measured): piranha plants (rising 24
  pixels at half a pixel a frame, 64 frames out and 64 down, staying down
  while Mario is within 33 pixels, colliding by a 10x6 box 4 to 10 pixels
  over the pipe's top), hopping Koopa Paratroopas (3.25 pixels a frame up,
  0.12 gravity), and Bullet Bill cannons (10x6 bullets at 1.5 pixels a
  frame through everything). The planner's enemies have kinds (walker,
  hopper, plant, flier) with their own paths, and the network's enemy
  tokens carry the kind. 5-1 also exposed, and the fixes cover:
  - perception: the camera estimate stood still on bare floor and in the
    game's partial-follow zone (now dead-reckoned from Mario's buttons
    from screen x 77 on); only walkers are assumed to come at Mario until
    measured; a hopper's speed comes from its last 4 moves; two enemies
    drawn as one keep their whole width;
  - planner: a wall blocks only moving deeper into it (Mario pressed
    against a pipe could not jump); a head bump goes on to a safe landing;
    a run-up whose goal fails jumps to any safe landing; a staircase's next
    step is a wall, not an end;
  - controller: a wait brakes to a stop; a run stops before a pit (a pipe
    or step ahead is no pit);
  - chooser: reaching the flag counts only when likely; landings, runs
    and bumps refused at a spot stay refused across waits.
  Three networks were trained on the way. Two were discarded: their
  sampled outcomes carried planner bugs (staircase steps as ends, pipes
  as pits) that taught false dead ends. The third, outcome_20261110h (6.26
  million choices, data/datasets/block_smb/outcomes_20261110h), plays
  random levels and 5-1 better than outcome_20261109d but 1-1's benches
  worse (0.92 from game memory, 0.88 from vision: the run off the second
  pipe toward a Goomba walking away, the Goombas after the second pit, the
  second staircase pair); outcome_20261109d stays the default for now.
  A check before each long sampling run now compares how sampled choices
  end with the last set (runs ending at an edge fell from 35% to 13% once
  pipes stopped counting as pits).
