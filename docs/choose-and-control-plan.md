# Choose and control: the plan

Approved by the user on 2026-11-07 ("Yes I agree, make it so"). This
replaces the layered agent's approach (docs/layered-agent.md), where networks
predicted where actions end and how long waits last. The background, the
evidence from this project and the design are in the report
[Choose and Control](https://claude.ai/artifact/1CaL7MDpVQsL4VCCS2DA6V).

## The design

- **Perception** turns each frame into a short list of objects (the Full SMB
  vision network; exact labels from game memory for tests).
- **A model-based controller runs every frame** from that list. It
  estimates each object's motion, plans with the game's real physics (the
  NES motion model), re-plans, presses buttons, and reports events: arrived,
  landed, a wait's condition met, a threat due, the goal infeasible. It is
  arithmetic, about a millisecond a frame.
- **A small decision network runs only at decision points** (an event from
  the controller). It reads the object list and the strategy switch, and
  outputs one choice: a mode, a verb (run, jump, wait), which object, and a
  coarse modifier (stomp or pass; land near, middle or far; keep speed or
  stop). It never outputs a distance, a frame count or a height.
- **A faithful cheap simulator** (Block SMB) copies the real levels: their
  geometry read from the game, the game's enemy wake-up rule (an enemy
  appears when the screen reaches it), and physics checked against game
  memory. Windows of the real levels are varied around the real geometry.
- **A privileged teacher** in the simulator picks the best choice by trying
  each candidate through the same controller, with full state. It labels
  only decision points.
- **Learning** copies the teacher, then rounds of the network playing in the
  simulator with the teacher labelling the states it reaches (user,
  2026-11-07: allowed in the simulator).

## What is dropped

- The world model as an outcome predictor (the controller computes outcomes).
- The collision tracker networks (model rollouts from the state estimates).
- The tactic memory network, unless a need shows.
- The strategy, tactic and skill layers as separate networks: one network
  with output heads.
- Waits in 8-frame slices (waits become conditions the controller watches).
- The long curriculum of hand-made lessons and 12-round recordings: windows
  of real levels, plus a few generic lessons.
- Next-frame prediction and pretraining.

Nothing learns on Full SMB (user, 2026-10-28). Reading the real game for
layouts and test states is allowed.

## Phases

Each phase ends with a measurement that decides whether to go on.

| Phase | Work | Done when |
|---|---|---|
| 0. Test bench | Saved game states just before each 1-1 obstacle (the first Goomba, each pipe, the pits, the brick rows, the stairs, the flag); success per obstacle | One command scores any agent on every obstacle in minutes |
| 1. Controller first | Goal regions from the chosen object; a jump planner on perceived state (the teacher's jump search moved into play); waits as conditions; "infeasible" events. Tested in 1-1 with a scripted chooser that reads game memory | A correct chooser plus the controller passes most 1-1 obstacles |
| 2. Faithful simulator | 1-1's layout imported into Block SMB; the enemy wake-up rule; jump arcs and run speed-up checked against game memory; varied windows of the level | The scripted chooser does as well in the simulator as in the real game |
| 3. Teacher | The privileged chooser in the simulator, trying candidate choices through the shared controller, labelling decision points only | Wins nearly all windows, in seconds an episode |
| 4. Learn | Copy the teacher, then rounds of the network playing and the teacher labelling its states | The network's choices match the teacher's on fresh windows |
| 5. Real game | The obstacle bench and full 1-1 playthroughs; then windows of another level | 1-1 obstacles mostly pass; what transfers is known |

Phase 1 comes before any learning: if a correct chooser and the controller
cannot pass an obstacle, nothing learned on top will.
