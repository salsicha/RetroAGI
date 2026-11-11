# The framework and its domains

RetroAGI is a game-neutral framework (`retroagi/framework`) plus one package
per game or robot (`retroagi/domains/<name>`). Super Mario Bros is the first
domain. The framework knows nothing of Mario: it is given a domain by name
and uses it only through the interfaces below.

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

The declarations are in `retroagi/framework/interfaces.py` (`World`,
`Practice`, `StateSource`, `ChoiceSpace`, `OutcomePredictor`, `Chooser`,
`Controller`, `Dynamics`, `Domain`). The state and the choices are the
domain's own types; the framework only passes them between the parts. The
one thing it asks of a state is `decidable`: whether a choice can be made
now.

## What the framework provides

| File | What it does |
|---|---|
| `agent.py` | The one loop: read the state, let the controller press on, and when a choice has ended ask the chooser for the next. |
| `chooser.py` | The fixed priorities over predicted outcomes, with a short memory of what was tried where. |
| `outcome_network.py` | A transformer over the state's tokens with a head per thing: every candidate's outcome in one pass. Its shape comes from the domain's `ChoiceSpec`; a saved network carries that spec and is checked against the domain when loaded. |
| `sampling.py` | Training data, measured: in practice worlds every candidate is played from a saved state by the domain's controller. |
| `training.py` | Trains the outcome network on sampled outcomes and measures it on fresh ones. |
| `practice.py` | Plays a domain's practice worlds and counts how they end. |
| `bench.py` | Scores an agent on each obstacle of a real level from saved states. |
| `play.py` | Plays a real level from its start. |

```bash
python -m retroagi.framework.sampling --domain smb --levels 2000 --workers 14 --output data/datasets/block_smb/outcomes_NEW
python -m retroagi.framework.training --domain smb --data data/datasets/block_smb/outcomes_NEW --fresh data/datasets/block_smb/outcomes_fresh_NEW --output artifacts/block_smb/outcome_NEW
python -m retroagi.framework.practice --domain smb --levels 100 --first-seed 100000 --model data/outcome/outcome_20261109d.pt
python -m retroagi.framework.bench --domain smb --bench data/full_smb_obstacles/Level1-1 --source vision --model data/outcome/outcome_20261109d.pt
python -m retroagi.framework.play --domain smb --level Level1-1 --source vision --model data/outcome/outcome_20261109d.pt
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
| The ways to play and what a point is worth in each | `domain.py` |

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

- **The dynamics model is written by hand.** The controller takes its model
  of how enemies and items move as a part that can be swapped
  (`planner.RULES`, `Controller(dynamics=...)`), but the only one there is
  a hand-written rule per kind of enemy. Mario's own motion is transcribed
  from the game (`domains/smb/physics.py`) and compiled into the planner.
  Nothing is identified or learned from what the agent sees yet: this is
  where adaptive control has to go.
- **Perception is trained on labels read from this game's memory**, and the
  collision boxes inside drawn sprites are entered by hand.
- **The simulator is a hand copy of the game's rules**; where it differs
  from the game, sampled outcomes teach the wrong thing.
- **The outcome network learns from the simulator's exact state** but plays
  from vision's noisy one.
- **Leftovers of the retired design in live files:** vision's scene lists
  are still capped at the old policy's slot counts
  (`perception/scene_lists.py`), and the simulator still carries the
  curriculum's reward and schedule features (`sim/env.py`).
