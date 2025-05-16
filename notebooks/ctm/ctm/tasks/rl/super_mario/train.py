import sys
import os
import warnings
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the other directory
other_dir = os.path.join(current_dir, '..', '..', '..')
# Add the other directory to sys.path
sys.path.insert(0, other_dir)

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from models.ctm_rl import ContinuousThoughtMachineRL

# Example CTM config (tune as needed)
CTM_CONFIG = dict(
    iterations=2,
    d_model=128,
    d_input=128,
    n_synch_out=16,
    synapse_depth=1,
    memory_length=10,
    deep_nlms=True,
    memory_hidden_dims=4,
    do_layernorm_nlm=False,
    backbone_type='classic-control-backbone',
    prediction_reshaper=[-1],
    dropout=0.0,
    neuron_select_type='first-last',
)

class MarioAgent(nn.Module):
    def __init__(self, ctm, n_actions):
        super().__init__()
        self.ctm = ctm
        self.policy = nn.Sequential(
            nn.Linear(ctm.n_synch_out, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(ctm.n_synch_out, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, obs, hidden_states):
        # obs: (B, C, H, W)
        # hidden_states: (state_trace, activated_trace)
        synchronisation, hidden_states = self.ctm(obs, hidden_states)
        logits = self.policy(synchronisation)
        value = self.value(synchronisation).squeeze(-1)
        return logits, value, hidden_states

def preprocess(obs):
    obs = np.transpose(obs, (2, 0, 1))  # (C, H, W)
    obs = obs.astype(np.float32) / 255.0
    return torch.tensor(obs, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Version check and warning
    import gym_super_mario_bros, nes_py, gym
    # print(f"gym_super_mario_bros version: {gym_super_mario_bros.__version__}")
    # print(f"nes_py version: {nes_py.__version__}")
    # print(f"gym version: {gym.__version__}")
    if hasattr(gym_super_mario_bros, 'make'):
        try:
            env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode=None)
        except TypeError:
            env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    else:
        env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    n_actions = env.action_space.n

    ctm = ContinuousThoughtMachineRL(**CTM_CONFIG).to(device)
    agent = MarioAgent(ctm, n_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4)

    num_steps = 128
    num_envs = 1
    num_updates = 1000
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_t = preprocess(obs).unsqueeze(0).to(device)
    state_trace = ctm.start_trace.unsqueeze(0).expand(num_envs, -1, -1).to(device)
    activated_trace = ctm.start_activated_trace.unsqueeze(0).expand(num_envs, -1, -1).to(device)
    hidden_states = (state_trace, activated_trace)
    done = False

    for update in range(num_updates):
        obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        h_states_buf = []

        for step in range(num_steps):
            obs_buf.append(obs_t.cpu().numpy())
            h_states_buf.append((hidden_states[0].detach().cpu().numpy(), hidden_states[1].detach().cpu().numpy()))
            logits, value, hidden_states = agent(obs_t, hidden_states)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            actions_buf.append(action.item())
            logprobs_buf.append(logprob.item())
            values_buf.append(value.item())
            dones_buf.append(done)

            obs, reward, done, info = env.step(action.item())
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_t = preprocess(obs).unsqueeze(0).to(device)
            rewards_buf.append(reward)
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs_t = preprocess(obs).unsqueeze(0).to(device)
                state_trace = ctm.start_trace.unsqueeze(0).expand(num_envs, -1, -1).to(device)
                activated_trace = ctm.start_activated_trace.unsqueeze(0).expand(num_envs, -1, -1).to(device)
                hidden_states = (state_trace, activated_trace)
                done = False

        # Convert buffers to tensors
        obs_buf = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device).squeeze(1)
        actions_buf = torch.tensor(actions_buf, device=device)
        logprobs_buf = torch.tensor(logprobs_buf, device=device)
        rewards_buf = torch.tensor(rewards_buf, dtype=torch.float32, device=device)
        dones_buf = torch.tensor(dones_buf, dtype=torch.float32, device=device)
        values_buf = torch.tensor(values_buf, dtype=torch.float32, device=device)

        # Compute last value for GAE
        with torch.no_grad():
            logits, next_value, _ = agent(obs_t, hidden_states)
        advantages = torch.zeros_like(rewards_buf, device=device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t+1]
                nextvalues = values_buf[t+1]
            delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values_buf

        # PPO update
        b_inds = np.arange(num_steps)
        clipfracs = []
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, 32):
                end = start + 32
                mb_inds = b_inds[start:end]
                mb_obs = obs_buf[mb_inds]
                mb_actions = actions_buf[mb_inds]
                mb_logprobs = logprobs_buf[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_values = values_buf[mb_inds]
                mb_obs_t = mb_obs.to(device)
                mb_actions_t = mb_actions.to(device)
                mb_advantages_t = mb_advantages.to(device)
                mb_returns_t = mb_returns.to(device)
                mb_logprobs_t = mb_logprobs.to(device)

                logits, value, _ = agent(mb_obs_t, hidden_states)
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(mb_actions_t)
                entropy = dist.entropy().mean()
                ratio = (newlogprob - mb_logprobs_t).exp()
                surr1 = ratio * mb_advantages_t
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - mb_returns_t) ** 2).mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        print(f"Update {update+1}: mean reward {rewards_buf.mean().item():.2f}")

    env.close()

if __name__ == "__main__":
    main()
