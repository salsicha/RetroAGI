# The four-layer SMB agent

One agent plays both Super Mario Bros games: Block SMB (a simulator drawn to
look like the NES game) and Full SMB (the real game in an emulator). It sees
each game only through that game's vision transformer, and it learns
everything first in Block SMB.

## What the agent sees

Each game has its own trained copy of one vision transformer
(`retroagi/core/vision.py`, `SceneVisionTransformer`). From a 256x240 screen it
outputs:

- every pixel's type: background, Mario, ground, brick, question block, pipe,
  coin, enemy, moving platform, power-up;
- for each 8x8 cell, the kind of enemy drawn there: walker, plant, other,
  defeated;
- Mario's facing; whether he is standing, in the air or on a moving
  platform; and whether his feet are on something (the land detector). Each
  game's land detector learns from its own engine's truth: the standing flag,
  plus stomps. In Full SMB the stomps are read from the emulator's memory; in
  Block SMB they come from the simulator's stomp handling.

The pixel types never reach a policy. `retroagi/core/smb_scene_labels.py` turns
them into one description of the screen, a `SceneObservation`:

- **Objects:** Mario and every enemy, coin, power-up and moving platform.
  Each is a group of touching pixels of its type; pieces at most one pixel
  apart join, and groups under 8 pixels are dropped as specks.
- **Blocks and pipes:** groups of their pixels.
- **Surfaces:** the top edge of everything Mario can stand on.
- **Gaps:** stretches of the floor with nothing to stand on.

Everything is reported as drawn. Nothing hidden is reported, such as a plant
inside its pipe or the game's collision boxes, and there is no speed: one
picture, one description.

The truth the vision transformers learn from uses the same rules.
- **Block SMB:** the simulator draws exact labels.
- **Full SMB:** the frame is rebuilt from game memory
  (`retroagi/stages/full_smb/pixel_labels.py`).

`scripts/vision/evaluate_block_vision.py` and
`scripts/vision/evaluate_full_vision.py` measure both models the same way.

`retroagi/core/smb_observer.py` packs the description into the policy's three
inputs:

- **C (352 numbers):** Mario, then fixed-length lists, each item "present"
  plus its box relative to Mario, and the current skill's target. The lists
  are enemies, coins, power-ups, moving platforms, pipes, blocks, surfaces
  and gaps.
- **A and B (8 and 16 numbers):** for each vertical band of the screen, the
  most important thing drawn in it.

## How it decides

`retroagi/core/layered_policy.py` holds the policy: four layers, a shared scene
encoder and a memory.
- **Scene encoder:** one token per reported object, plus the band codes.
  Every position number also enters as 8 sine and 8 cosine waves, with
  wavelengths from 512 pixels (480 vertically) down to 4. A difference of one
  pixel is therefore a clear difference of input, while the long waves still
  tell near from far.
- **Memory**, a long short-term memory network, steps once per action, at
  the action's start, before anything decides:
  - **Input:** vision information only, never the action Mario is about to
    take: the scene encoder's summary of the latest picture, the frame the
    action starts on. What happened at earlier actions, including how things
    moved, it carries in its own state.
  - **Output:** its expected scene, a prediction of the scene numbers the
    vision transformer will report when the coming action ends, completed
    or interrupted.
  - **Who reads it:** the expected scene is encoded like a real one and given
    to the decision layers beside the current scene.
  - **Training:** action by action. It's scored against the scene actually
    reported when each action ended.
- **The four layers** are each a transformer whose context holds:
  - the current scene;
  - the memory's expected scene;
  - the token from the layer above;
  - its own last 16 choices, each marked with how many decisions ago it was
    made. In play these are its own choices; in training they're whatever
    was actually used, including the teacher's. The skill layer remembers
    where each past target was (its box measured from Mario at that moment),
    not which slot it filled, because slots are re-filled for every picture.

  So every layer knows what it chose before, and keeping or changing course
  is something it learns.
- **Strategy layer:** emits what the run is for, and a direction:
  - speed run: finish as fast as possible;
  - max coins: collect as many coins as possible;
  - careful: finish without risk, however long it takes.
- **Tactic layer:** emits advance, alternate route, hold area or retreat, and a
  direction:
  - advance: toward the goal along the main path;
  - alternate route: toward the goal along another path, higher platforms or a
    lower floor;
  - hold area: stay put until something changes (a moving platform comes, a
    plant goes back into its pipe, a monster comes close enough to jump);
  - retreat: back away from the goal for a while, out of a dead end or away
    from a monster.
- **Skill layer:** emits a skill and a direction, and points at a target
  object. The skills are advance, jump gap, climb, descend, stomp, retreat and
  wait. Avoiding enemies is part of every skill, not a skill of its own.
  Walking up to a moving platform is advancing, jumping onto one is jumping a
  gap, and riding one is waiting.
- **Action layer:** emits a button action (nothing, right, right + jump, left,
  left + jump or jump) and how many frames to press it: any whole number from
  1 to 32, for every action.

The executor (`retroagi/core/smb_executor.py`) takes only those two numbers
from the action layer and presses that button action on each of those frames;
it reads nothing else. An action ends in one of two ways, and only these two
make the layers choose a new one:
- **It ran its course:** it pressed all of its frames. A jump is the jump
  button held for its frames, so the layers may choose the next action with
  Mario still in the air.
- **Mario landed:** the vision transformer decides. Its land detector says
  whether Mario's feet are on something in the picture: the ground, a moving
  platform, or an enemy he is stomping. When it switches on after being off
  (he was in the air), the running action ends.

`retroagi/core/smb_agent.py` (`SMBAgents`) runs all of this from screens
alone, for both games.

## How it learns, in Block SMB

`retroagi/stages/block_smb/layered_train.py` trains one layer at a time, from the
bottom up, with the others frozen.
- **Action layer:** given the skill.
- **Skill layer:** given the tactic.
- **Tactic layer:** given the strategy.

The token from above comes from a teacher that reads the simulator
(`retroagi/stages/block_smb/teacher_tokens.py`). The teacher is used only in
training.

### Every layout states its tactics

Each layout carries its tactics in order: a list of segments
(`retroagi/stages/block_smb/tactic_schedule.py`). A segment has a stance, a
direction, and optionally a route (platforms to stand on, in order) and
forbidden platforms. It ends at a platform, at a line, after a number of
frames, once a moving platform is crossed, or once an enemy is passed. The
simulator follows the segments during training:
- the goal counts only in the last segment;
- standing on a forbidden platform, or leaving a hold area early, ends the
  episode as a loss.

Three kinds of segment change their stance inside, by their teacher's rule:
- moving platform: hold the area while waiting for it or riding it;
- plant: hold the area while waiting for it to go back into its pipe;
- monster: retreat from it, hold the area, then jump over it.

The teachers follow the segments. The tactic is the current segment's. The
skill is decided by the tactic:
- hold area: wait;
- retreat: retreat;
- advance or alternate route: the next step of that path (climb, descend,
  jump gap, stomp or advance).

The action teacher's route follows the same segments.

| Tactic | Families that teach it |
|---|---|
| Advance | every family; the simple ones are advance only |
| Alternate route | upper_route, lower_route, dead_end_retreat, chained_obstacles, tactics_obstacle_sequence, tactics_bridge_then_gap, tactics_mixed_sequence, mixed_section, choice_alternate_route, low_choice_alternate_route |
| Hold area | bridge_wait, wait_timing, moving_bridge, bridge_mount, bridge_dismount, piranha_avoidance, monster_retreat, chained_enemy_gauntlet, full_smb_opening_proxy, tactics_bridge_sequence, tactics_obstacle_sequence, tactics_bridge_then_gap, tactics_mixed_sequence, mixed_section, choice_hold_area |
| Retreat | dead_end_retreat, monster_retreat, tactics_bridge_then_gap, chained_enemy_gauntlet, tactics_mixed_sequence, choice_retreat; the plant teacher when Mario overshoots its waiting spot |

The new families (`retroagi/stages/block_smb/tactic_families.py`):
- **upper_route:** the floor is cut by a pit too wide to jump, or a wall too
  tall to climb; raised platforms cross it.
- **lower_route:** a raised ledge is cut by an opening too wide to jump; Mario
  drops to the floor below and climbs back up.
- **dead_end_retreat:** Mario starts in a corridor that ends at a wall too
  tall to climb. He backs out to a step behind him, then takes a high path
  over the wall.
- **monster_retreat:** a monster that can't be stomped (the vision reports it
  as the "other" enemy kind) walks out of a tunnel too low to jump in. Mario
  backs out of the tunnel, waits, and jumps over it in the open.

The chained and sequence families are composed from sections
(`retroagi/stages/block_smb/compose.py`), so their tactics change along the
way. tactics_bridge_sequence is a moving platform, then a plant.

**Clones**, for the skill layer, play the very same layouts and differ only in
their tactics, so the right skill is decided by the tactic token:
- choice_advance, choice_alternate_route, choice_hold_area and choice_retreat
  share a floor with a pit ahead, a raised path above it and room behind;
- low_choice_advance and low_choice_alternate_route share a ledge path with
  gaps over a lower floor.

Not following the tactic loses the episode. The tactic layer doesn't train on
the clones, because their tactic isn't decided by the scene.

### Strategy courses

The strategy token matters only in the strategy courses: speed_run_course,
max_coins_course and careful_course. They are composed scenes only, and the
three families play the very same layouts. The only differences are:
- the strategy given to the tactic layer;
- the tactics the teacher follows;
- what the episode pays and what counts as winning.

Each course has two or three sections that offer two routes, among other
sections:
- **Coin detour:** coins on a zig-zag tower above the floor. Walk past
  (advance) or climb it for the coins (alternate route).
- **Hazard bypass:** enemies patrol the floor under a high walkway with none,
  reached by a zig-zag. Go through the enemies (advance) or over them
  (alternate route).
- **Lift shortcut:** a pit too wide to jump. Ride a moving platform (advance;
  how long that takes depends on where the platform is when Mario arrives),
  or climb a zig-zag to a walkway across (alternate route, with a few
  coins).

The teacher plays every combination of the routes and measures each one:
frames to the goal, coins collected and enemies passed. Each strategy takes
the best combination for its objective:
- **speed run:** the fewest frames;
- **max coins:** the most coins, then the fewest frames;
- **careful:** the fewest enemies passed, then the fewest frames.

So speed run takes a raised route whenever that is quicker: past a far-away
moving platform, or over enemies that would slow it down. In checks over 24
layouts it took at least one raised route in 12. Turning back and forth on a
zig-zag costs time; jumping does not, because Mario keeps his running speed
through a jump.

What each strategy is paid for (`MarioScenarioEnv.STRATEGY_REWARDS`) and how it
wins:

| Strategy | Paid for | Wins when |
|---|---|---|
| Speed run | a time bonus at the goal for each frame left before its deadline, and a frame cost five times the usual | it reaches the goal within 10% of the fastest route's time |
| Max coins | coins, at two and a half times the usual reward | it reaches the goal with at least three quarters of the extra coins its route gathers beyond speed run's |
| Careful | no frame cost; dying costs five times as much | it reaches the goal |

A course layout is kept only if max coins' best route gathers more coins than
speed run's. Each layout records what the teacher measured for every
combination (`route_results`).

Every other family is played as a speed run.

1. **Imitation rounds.** Workers play episodes through the vision transformer.
   At every decision the teacher says what it would do from that exact state;
   the learner trains on all such labels so far. In the first round every
   decision plays the teacher's choice, and the share falls to none by the
   last imitation round.
2. **Reward rounds.** The learner samples its own choices and learns from the
   rewards each family gives until the next decision. The teacher's labels
   still count.

A layer counts as trained when every family passes the bar on held-out
layouts. After the tactic layer, the whole agent is also scored as deployed:
every token is its own, and the strategy is "speed run" until the strategy
layer has learned.

```bash
python -m retroagi.stages.block_smb.cli train-layer --learner action --reward-rounds 8
```

```bash
python -m retroagi.stages.block_smb.cli train-layer --learner skill --init artifacts/block_smb/layered/best.pt
```

## Full SMB

The layers trained in Block SMB play Full SMB unchanged, through the Full SMB
vision transformer. `retroagi/stages/full_smb/play.py` gives the agent screens;
game memory is read only to score how far Mario got.

```bash
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/layered/best.pt
```

The strategy layer learns only from playing Full SMB.
