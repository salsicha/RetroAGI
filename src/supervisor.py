import torch
import os
import random
from collections import deque

class Supervisor:
    """
    The Supervisor handles the online training of the lobes, calculates
    prediction errors based on game rewards/penalties, and manages serialization.
    Now uses Experience Replay and Batched Updates for stability and performance.
    """
    def __init__(self, lobes, checkpoint_dir='data/checkpoints/', batch_size=32, update_freq=8):
        self.lobes = lobes
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.memory = deque(maxlen=5000) # Experience replay buffer
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.step_count = 0

    def update(self, state, action, next_state, reward, info, latents=None):
        self.step_count += 1
        life_lost = info.get('life', 2) < 2
        
        transition = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'x_pos': info.get('x_pos', 0),
            'coins': info.get('coins', 0),
            'collision': life_lost,
            'latents': latents or {}
        }
        self.memory.append(transition)

        # Decouple optimization step from environment step
        if self.step_count % self.update_freq == 0 and len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            self._train_on_batch(batch)

    def _train_on_batch(self, batch):
        # Robust device detection
        if self.lobes and hasattr(self.lobes[0], 'parameters'):
            device = next(self.lobes[0].parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Efficient batched tensor creation
        rewards = torch.as_tensor([b['reward'] for b in batch], dtype=torch.float32, device=device)
        collisions = torch.as_tensor([b['collision'] for b in batch], dtype=torch.bool, device=device)
        x_pos = torch.as_tensor([b['x_pos'] for b in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([b['action'] for b in batch], dtype=torch.long, device=device)
        
        has_latents = all(['v' in b['latents'] for b in batch])
        if has_latents:
            # Re-upload latents from RAM to GPU memory for the batch
            latents_v = torch.cat([b['latents']['v'] for b in batch], dim=0).to(device)
            latents_t = torch.cat([b['latents']['t'] for b in batch], dim=0).to(device)
            latents_h = torch.cat([b['latents']['h'] for b in batch], dim=0).to(device)
            plans = torch.cat([b['latents']['plan'] for b in batch], dim=0).to(device)
        
        signal = {
            'reward': rewards,
            'collision': collisions,
            'x_pos': x_pos,
            'actual_next_state': next_states,
            'latents_v': latents_v if has_latents else None,
            'latents_t': latents_t if has_latents else None,
            'latents_h': latents_h if has_latents else None,
            'plans': plans if has_latents else None,
            'actions': actions
        }

        # Broadcast the batched signal to all lobes for backprop
        for lobe in self.lobes:
            lobe.learn(signal)

    def checkpoint(self):
        print(f"Serializing models to {self.checkpoint_dir}...")
        for lobe in self.lobes:
            lobe_name = lobe.__class__.__name__
            path = os.path.join(self.checkpoint_dir, f"{lobe_name}.pth")
            lobe.save(path)
            
        supervisor_state = {
            'step_count': self.step_count
        }
        torch.save(supervisor_state, os.path.join(self.checkpoint_dir, 'supervisor.pth'))

    def load_checkpoint(self):
        supervisor_path = os.path.join(self.checkpoint_dir, 'supervisor.pth')
        if os.path.exists(supervisor_path):
            state = torch.load(supervisor_path)
            self.step_count = state.get('step_count', 0)
            print(f"Resuming supervisor state from step {self.step_count}")

        checkpoint_found = False
        for lobe in self.lobes:
            lobe_name = lobe.__class__.__name__
            path = os.path.join(self.checkpoint_dir, f"{lobe_name}.pth")
            if os.path.exists(path):
                if not checkpoint_found:
                    print(f"Loading checkpoints from {self.checkpoint_dir}...")
                    checkpoint_found = True
                lobe.load(path)
                print(f"Loaded {lobe_name} from checkpoint.")
        return checkpoint_found

    def set_planning_mode(self, value):
        for lobe in self.lobes:
            if hasattr(lobe, 'planning_mode'):
                lobe.planning_mode = value
                print(f"Planning mode set to: {'Speedrun' if value == 0 else 'Max Coins' if value == 1 else value}")
