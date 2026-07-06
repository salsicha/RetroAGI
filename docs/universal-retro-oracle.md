# Universal Retro Oracle Roadmap

RetroAGI should not rely on a bespoke heuristic oracle forever. The long-term
teacher for Block-level learning should be a general purpose AI oracle trained
across many retro-style games, then adapted to each game's synthetic, block,
asset-mock, and full-fidelity stages.

The current scripted Block SMB oracle remains useful as bootstrap data and a
regression sentinel. It should not become the permanent source of truth. The
target system is a learned oracle that proposes actions, primitive parameters,
expected outcomes, and confidence estimates, then improves those labels as the
student architecture trains.

## Research Basis

The best fit is a hybrid oracle, not a single-paper copy.

| Research direction | Useful lesson for RetroAGI |
| --- | --- |
| [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241) | A single transformer can learn return-conditioned behavior across many Atari games from offline trajectories. This is the closest match for the oracle policy backbone. |
| [Gato](https://arxiv.org/abs/2205.06175) | A tokenized generalist policy can share one model across different observation and action modalities, which matches the multi-game oracle goal. |
| [DreamerV3](https://arxiv.org/abs/2301.04104), [IRIS](https://arxiv.org/abs/2209.00588), and [STORM](https://arxiv.org/abs/2310.09615) | A latent world model can predict future outcomes from pixels or semantic tokens, supporting the expectation/planning step instead of pure behavior cloning. |
| [MuZero](https://arxiv.org/abs/1911.08265) and [EfficientZero](https://arxiv.org/abs/2111.00210) | Search over a learned model can produce stronger action labels, especially when explicit emulator rules are unavailable. This is a later, higher-compute teacher mode. |
| [Video PreTraining](https://arxiv.org/abs/2206.11795) and [Genie](https://arxiv.org/abs/2402.15391) | Inverse-dynamics or latent-action models can bootstrap labels from unlabeled gameplay video or replay data, which is important for scaling beyond games with hand-written action scripts. |

Recommended architecture:

1. **Universal transformer oracle policy:** a multi-game, return or
   goal-conditioned transformer that consumes semantic grids, symbolic entities,
   patch tokens, recent action primitives, game/task tokens, and recurrent
   memory. It outputs game-native actions plus game-neutral primitive labels.
2. **Latent world model:** a Dreamer/IRIS/STORM-style dynamics model that
   predicts k-step primitive outcomes, uncertainty, rewards, progress, hazards,
   and terminal risk.
3. **Inverse or latent action labeler:** a VPT/Genie-style component that can
   infer plausible actions and primitive boundaries from unlabeled gameplay
   sequences.
4. **Optional search teacher:** a MuZero/EfficientZero-style planner that uses
   the world model to refine labels for difficult states, high-risk hazards, or
   low-confidence student failures.

This lets RetroAGI keep the fast Block rung as the main training engine while
replacing per-game heuristics with a reusable learned teacher.

## Oracle Output Contract

The oracle should emit a versioned `UniversalOracleTrace` for each scenario or
trajectory segment:

- game ID, stage name, task ID, seed, distribution ID, and source provenance;
- observation tokens: semantic classes, symbolic entities, patch features, RAM
  or signal fields when available, and recent history;
- game-native action label and button vector;
- primitive labels: `button_combo`, `hold_duration`, `release`,
  `post_release_action`, `cancel`, `replan`, and `hazard_window_timing`;
- outcome targets: progress delta, support loss, collision risk, death risk,
  reward, terminal probability, and continue/cancel/replan target;
- confidence and uncertainty for each label family;
- alternate actions considered by planning, with expected outcome summaries;
- provenance fields showing whether a label came from a scripted bootstrap,
  human trace, emulator replay, model prediction, search, or EM relabeling.

Block-level trainers should consume this contract the same way they currently
consume scripted oracle actions: as supervised labels and outcome targets. The
student policy should still be evaluated independently through fixed scenarios,
Monte Carlo held-out splits, and full-fidelity transfer gates.

## Iterative ML And EM Loop

The oracle should learn while it guides learning.

1. **Collect trajectories:** ingest scripted Block SMB traces, Monte Carlo
   oracle traces, human or imitation traces, emulator rollouts, and future
   retro-game traces into a shared multi-game trajectory store.
2. **M-step, supervised maximum likelihood:** train the universal transformer
   policy, primitive heads, inverse action model, value/outcome heads, and
   latent dynamics model on known labels and high-confidence pseudo-labels.
3. **E-step, expectation and relabeling:** use the world model, uncertainty
   estimates, and optional search to infer better latent actions, primitive
   boundaries, expected returns, terminal risks, and cancel/replan targets for
   unlabeled or weakly labeled trajectories.
4. **Student update:** train Block-level policies using the oracle trace
   contract, including action imitation, primitive imitation, and k-step
   outcome prediction.
5. **Failure feedback:** route student failures, low-confidence oracle states,
   and held-out Monte Carlo failure bins back into data collection and EM
   relabeling.
6. **Promotion:** require the learned oracle-guided student to beat or match
   the scripted-bootstrap baseline on held-out Block tasks before allowing it
   to replace scripted labels for a game family.

The loop is maximum likelihood for known behavior labels and expectation
maximization for uncertain latent actions, primitive boundaries, and future
outcomes.

## Milestones

### URO0: Contracts And Data

- Define `UniversalOracleTrace`, primitive target, outcome target, confidence,
  and provenance schemas.
- Add a game-neutral action ontology that maps per-game actions to button
  combos, hold/release behavior, no-op/release actions, and cancel/replan
  semantics.
- Add a multi-game trajectory manifest format with split, seed, source,
  license/provenance, model version, and label-confidence metadata.

Exit gate: existing Block SMB scripted traces can be exported into the new
contract without losing current oracle action, primitive, or Monte Carlo
metadata.

### URO1: Bootstrap Dataset

- Export Block SMB fixed and Monte Carlo oracle trajectories.
- Add at least one non-SMB block-game trajectory source, starting with the
  existing Pong profile or another simple retro-style block profile.
- Add importers for human/imitation traces and emulator rollouts where content
  provenance allows local use.
- Store failure bins, near-miss traces, and successful traces together so the
  oracle learns both good actions and bad outcomes.

Exit gate: a reproducible dataset build command writes train, validation, test,
and stress splits for at least two game profiles.

### URO2: Supervised Universal Oracle Baseline

- Train a multi-game transformer oracle using maximum likelihood on exported
  action and primitive labels.
- Condition the model on game ID, task ID, return/goal tokens, and stage
  resolution.
- Add teacher-forcing validation metrics: action accuracy, primitive accuracy,
  outcome calibration, per-game metrics, and held-out scenario pass-rate lift
  when used for Block policy distillation.

Exit gate: the learned oracle matches the scripted oracle on held-out easy
Block SMB samples and improves student distillation over no-oracle training.

### URO3: World Model And Outcome Oracle

- Add latent dynamics training for k-step primitive outcomes across games.
- Predict progress, support loss, collision/death risk, reward, terminal
  outcome, and continue/cancel/replan targets.
- Calibrate uncertainty so low-confidence labels can be filtered or sent to
  search/replay.

Exit gate: outcome predictions are calibrated on held-out Block distributions
and improve motor-primitive cancel/replan decisions.

### URO4: EM Relabeling And Search

- Add an inverse/latent action model for trajectories without reliable action
  labels.
- Add an EM job that alternates supervised retraining with pseudo-label
  inference and confidence filtering.
- Add optional search over candidate primitive sequences for high-risk or
  low-confidence states.

Exit gate: EM relabeling increases held-out Block pass rates or reduces
student failure bins without increasing overconfident wrong labels.

### URO5: Block Trainer Integration

- Add a pluggable `OracleProvider` interface for Block trainers and distillers.
- Support `scripted`, `learned`, and `hybrid` providers.
- In `hybrid` mode, let the learned oracle supply labels when confidence is
  high and fall back to scripted/bootstrap labels only for validation or
  coverage sentinels.
- Record oracle model version, provider mode, confidence thresholds, and label
  provenance in every checkpoint and promotion artifact.

Exit gate: Block SMB can train from learned-oracle labels and still pass fixed
plus held-out Monte Carlo gates.

### URO6: Cross-Game Scaling

- Add more retro-style games by implementing game profiles, block simulators,
  action mappings, semantic vocabularies, and trajectory exporters.
- Train the oracle with mixed-game batches and measure transfer to held-out
  game families or held-out mechanics.
- Add architecture reports that compare scripted oracle, learned oracle,
  hybrid oracle, and no-oracle training.

Exit gate: adding a new block-game profile requires data/schema adapters, not
new hand-written oracle logic in the trainer.

### URO7: Full-Fidelity Feedback

- Feed Full SMB and other emulator rollouts back into the oracle dataset.
- Use full-fidelity failures to request new Block scenarios or EM relabeling
  around physics gaps.
- Keep benchmark evaluation read-only, but add explicit adaptive-training runs
  where full-fidelity outcomes can update the oracle and student models.

Exit gate: the oracle-guided Block policy transfers better to Full SMB than the
scripted-oracle baseline on the same promotion budget.

## Evaluation Gates

The learned oracle should be promoted only when it satisfies objective gates:

- equal or better student pass rate than the scripted oracle on fixed Block
  scenarios;
- equal or better held-out Monte Carlo validation pass rate;
- per-family failure-bin reduction, especially for transfer-sensitive hazards;
- calibrated confidence, measured by lower confidence on wrong labels;
- cross-game action and primitive accuracy on held-out game profiles;
- improved Block-to-Full transfer metrics under the same budget;
- no dependence on trainer-specific hand-written oracle logic at inference.

The first implementation can keep scripted labels as bootstrap data, but the
roadmap goal is a learned teacher that becomes stronger as it observes student
failures and full-fidelity outcomes.
