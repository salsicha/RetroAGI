• Yes: the evaluated model collapsed. Not to LEFT anymore, but to RIGHT.

  Training-time sampled actions were mixed, but deterministic eval was:

  - Fixed eval: RIGHT=1262, all others 0
  - MC validation: RIGHT=8007, all others 0
  - MC test: RIGHT=15934, all others 0

  So the model is not learning a conditional policy. It learned “always hold right,” which solves a few simple/right-friendly scenarios and fails most tasks requiring jump, wait,
  recovery, enemy timing, or left.

  Key hyperparameters from the run:

  architecture: agent_world_model_critic
  params: ~132,913
  hidden_dim: 32
  Transformer A: 2 layers, 4 heads, ff=128
  Transformer B: 2 layers, 4 heads, ff=128
  world model: LSTM hidden 32
  learning_rate: 3e-4
  gamma: 0.95
  rollout_steps: 32
  epochs: 5
  episodes per epoch after fix: 528
  optimizer updates per epoch: 33
  update_batch_episodes: 16
  entropy_weight: 0.01
  policy_loss_weight: 1.0
  value_loss_weight: 0.25
  world_model_weight: 0.1
  reward_loss_weight: 0.01
  action_aux_weight: 0.01
  noop_loss_weight: 0.25
  critic_loss_weight: 0.001

  The fundamental suspects are more serious than “needs more epochs”:

  1. No oracle-action supervision.
     The Monte Carlo scenarios have oracle actions, but Block SMB policy training is not learning from them. It samples its own actions and uses scalar returns. So “training data” is not
     actually labeled behavior data.

  2. Action head mismatch risk.
     The architecture vocabulary is size 20, but Block SMB env actions are only first 6. Env action selection slices logits [:6], while the actor internals use Gumbel over the full
     vocab. That is a serious contract smell.

  3. Action auxiliary loss is self-labeling.
     It trains motor primitive heads toward the action the policy already sampled, not the correct/oracle action. That can reinforce bias.

  4. Argmax deployment is not trained directly.
     Stochastic rollout explores all actions, but deterministic eval takes argmax. If logits are only weakly shaped, argmax collapses to one action.

  5. The world model learned; actor did not.
     Dynamics semantic accuracy hit about 1.0. So perception/dynamics are not the main bottleneck. The control policy is.

  My read: the next fix should not be “more RL.” It should be to train the Block SMB actor with oracle imitation on the generated/fixed scenarios, and add a hard test that a trained
  model overfits a tiny scenario set to near 1.0 before doing full curriculum training.

