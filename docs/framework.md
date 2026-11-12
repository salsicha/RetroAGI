# The framework and its domains

RetroAGI is a game-neutral framework (`retroagi/framework`) plus one package
per game or robot (`retroagi/domains/<name>`). Super Mario Bros is the first
domain. The framework knows nothing of Mario: it is given a domain by name
and uses it only through the interfaces below. What all of it must be, in
the project owner's words, is in [requirements.md](requirements.md).

## The agent's five parts

```text
world ──source──▶ state ──choice space──▶ candidate choices
                            │
                            ▼
                    outcome predictor ──▶ each candidate's predicted outcome
                            │
                            ▼
                         chooser ──▶ one choice ──▶ controller ──▶ an action a frame
                                                        │
                                                        └─▶ an event when the choice has ended
```

| Part | What it reads | What it gives | Who supplies it |
|---|---|---|---|
| World | an action | the next moment; whether the agent died or finished | the domain: a simulator, or the real thing |
| State source | the world | the state the other parts read | the domain: from the world's truth, or from perception |
| Choice space | the state | the choices open now, and the state as numbers for a learned predictor | the domain |
| Outcome predictor | the state and the candidates | for each: chance of death, of being infeasible, of finishing, of getting stuck; progress, frames, points | the framework's outcome network, or anything with the same answer |
| Chooser | the predictions | one choice, by fixed priorities: don't die, make progress, gather points | the framework; a domain may add a hand-written one |
| Controller | a choice and the state each frame | an action each frame, and how the choice ended | the domain, planning with its dynamics model |
| Dynamics model | the things and solids in each state | where each thing will be for the frames ahead | the domain's rules, or the framework's model fitted to what is seen |

The declarations are in `retroagi/framework/interfaces.py` (`World`,
`Practice`, `StateSource`, `ChoiceSpace`, `OutcomePredictor`, `Chooser`,
`Controller`, `Dynamics`, `Domain`). The state and the choices are the
domain's own types; the framework only passes them between the parts. The
one thing it asks of a state is `decidable`: whether a choice can be made
now.

The controller is shown every state (`observe`), a choice in progress or
not, and passes what moves in it to its dynamics model: a model that
adapts learns from every frame, not only from the frames it plans in. A
controller can make another that knows what its model has learned
(`another`): outcome sampling plays each candidate with one.

## Parts with alternatives

A domain names the parts it has more than one of (`Domain.parts`), and
every tool takes `--use PART=NAME` to play with another. Super Mario Bros
has two dynamics models:

| `--use dynamics=` | What it is |
|---|---|
| `rules` (the default) | The simulator's own rule for each kind of enemy (`domains/smb/planner.py`): Block SMB's behaviour, measured from the game. |
| `observed` | `framework/observed_dynamics.py`: told nothing of what a thing is, it fits each thing's motion to its own track and remembers by type what things have been seen to do (how they fall, bounce, come out of solids, pace). Its memory starts empty at each play. An experiment: it was shaped against recordings of the real game, which no part of the policy may be, and plays fresh Block SMB levels no better than the rules. |

```bash
python -m retroagi.framework.practice --domain smb --use dynamics=observed --levels 100 --first-seed 100000 --model data/outcome/outcome_20261111k.pt
python -m retroagi.domains.smb.nes.dynamics_check --bench data/full_smb_obstacles/Level2-1 --model data/outcome/outcome_20261111k.pt
```

The second command scores every dynamics model of the domain on the same
recorded plays: predicted against true places of each enemy, by kind and
by frames ahead. [How general is it?](generalization.md) has the results.

## What the framework provides

| File | What it does |
|---|---|
| `agent.py` | The one loop: read the state, let the controller press on, and when a choice has ended ask the chooser for the next. |
| `chooser.py` | The fixed priorities over predicted outcomes, with a short memory of what was tried where. |
| `observed_dynamics.py` | A dynamics model identified from what is seen, for worlds of boxes moving among solid boxes: any domain of that shape can use it. |
| `registry.py` | Domains by name, and their parts' alternatives (`--use`). |
| `outcome_network.py` | A transformer over the state's tokens with a head per thing: every candidate's outcome in one pass. Its shape comes from the domain's `ChoiceSpec`; a saved network carries that spec and is checked against the domain when loaded. |
| `sampling.py` | Training data, measured: in practice worlds every candidate is played from a saved state by the domain's controller. |
| `training.py` | Trains the outcome network on sampled outcomes and measures it on fresh ones. |
| `practice.py` | Plays a domain's practice worlds and counts how they end. |
| `bench.py` | Scores an agent on each obstacle of a real level from saved states. |
| `play.py` | Plays a real level from its start. |

```bash
python -m retroagi.framework.sampling --domain smb --levels 2000 --workers 14 --output data/datasets/block_smb/outcomes_NEW
python -m retroagi.framework.training --domain smb --data data/datasets/block_smb/outcomes_NEW --fresh data/datasets/block_smb/outcomes_fresh_NEW --output artifacts/block_smb/outcome_NEW
python -m retroagi.framework.practice --domain smb --levels 100 --first-seed 100000 --model data/outcome/outcome_20261111k.pt
python -m retroagi.framework.bench --domain smb --bench data/full_smb_obstacles/Level1-1 --source vision --model data/outcome/outcome_20261111k.pt
python -m retroagi.framework.play --domain smb --level Level1-1 --source vision --model data/outcome/outcome_20261111k.pt
```

## What a domain supplies

A domain is `retroagi/domains/<name>/domain.py` exposing `DOMAIN`. For Super
Mario Bros (`retroagi/domains/smb`):

| Supplied | Where |
|---|---|
| The controller and its jump planner | `controller.py`, `planner.py`, with the motion model in `physics.py` |
| The choices and the state as tokens | `choices.py` |
| A hand-written chooser (the behaviour that visits the states outcomes are sampled at, and the baseline to beat) | `scripted_chooser.py` |
| Perception: vision, tracking, motion estimate, vision to state | `perception/` |
| The practice world: random levels of the game's pieces in a simulator | `sim/` |
| The real game in the emulator, its game-memory state source, the obstacle benches | `nes/` |
| The ways to play and what a point is worth in each, and the parts with alternatives | `domain.py` |

Adding a domain means writing these for it; nothing in `framework/` changes.
`scripts/tests/test_architecture.py` keeps it so: the framework imports no
domain, and no domain imports the retired code.

## The retired layered agent

`retroagi/legacy` holds the earlier design (strategy, tactic and skill
layers, its teachers and curriculum) and `scripts/tests/legacy` its tests.
It still runs; the framework and the domains import nothing from it.

## Checking a change that should not change behaviour

Play is deterministic, so a refactoring must reproduce recorded behaviour
exactly. `scripts/dev/compare_reference.py REFERENCE NEW` compares two
folders of recordings: random levels played, outcomes sampled, obstacle
benches from game memory and from vision.

## What is not general yet

- **Most of the policy is written by hand, and was adjusted while watching
  the real game**: the tracking, the list of choices, the chooser's rules
  and the controller's. Only the outcome network is learned (from Block
  SMB). The rule ([generalization.md](generalization.md)) is that all of
  it is learned or set from Block SMB alone.
- **Block SMB does not yet behave as all of the game does**: it lacks
  enemies the real levels have (Lakitu and his Spinies, Hammer Bros, Buzzy
  Beetles) and lifts, and its random levels do not draw every piece and
  spacing real levels are made of. (It is not meant to have the game's
  graphics or its levels: [requirements.md](requirements.md).) The agent
  finishes 92% to 96% of fresh Block SMB levels and one real level.
- **The controller's parameters are set by hand**, not learned in Block
  SMB and passed in.
- **Mario's own motion is transcribed from the game**
  (`domains/smb/physics.py`) and compiled into the planner: nothing about
  the agent's own body is identified.
- **Perception is trained on labels read from this game's memory**, and the
  collision boxes inside drawn sprites are entered by hand.
- **The simulator is a hand copy of the game's rules**; where it differs
  from the game, sampled outcomes teach the wrong thing.
- **Vision's errors cost the agent in the real game** (9 to 15 obstacles in
  100 on levels nobody looked at). In Block SMB the state is a fast lookup
  by design, so this is seeing's problem to solve, not the policy's.
- **Leftovers of the retired design in live files:** vision's scene lists
  are still capped at the old policy's slot counts
  (`perception/scene_lists.py`), and the simulator still carries the
  curriculum's reward and schedule features (`sim/env.py`).
