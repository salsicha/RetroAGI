"""
Feedback Loop Manager
Handles the storage of high-level goals and implements the 'Stop and Learn' logic.
"""
import torch
import torch.nn.functional as F

class FeedbackLoop:
    def __init__(self, models, optimizers, device='cpu'):
        """
        Args:
            models (dict): Dictionary of model instances (e.g., {'motor': motor_lobe}).
            optimizers (dict): Dictionary of optimizers.
            device (str): 'cpu' or 'cuda'.
        """
        self.models = models
        self.optimizers = optimizers
        self.device = device
        
        # Storage for the current "Plan" (Frontal/Parietal outputs)
        self.current_plan = {
            'frontal_latent': None,
            'parietal_latent': None,
            'motor_seq': None,
            'motor_log_probs': None,
            'value': None
        }

    def start_planning_phase(self, frontal_latent, parietal_latent):
        """
        Call this when the Frontal/Parietal lobes generate a new plan.
        Saves the outputs so they aren't recomputed every frame.
        """
        # We keep these attached to the graph if we want to backpropagate 
        # the outcome all the way to the Frontal lobe later.
        self.current_plan['frontal_latent'] = frontal_latent
        self.current_plan['parietal_latent'] = parietal_latent
        
        # Reset motor execution state
        self.current_plan['motor_seq'] = None
        self.current_plan['motor_log_probs'] = None
        self.current_plan['value'] = None

    def set_motor_execution(self, sequence, log_probs, value):
        """
        Save the generated motor sequence and its log probabilities.
        """
        self.current_plan['motor_seq'] = sequence
        self.current_plan['motor_log_probs'] = log_probs
        self.current_plan['value'] = value

    def apply_feedback(self, reward, scope='motor'):
        """
        Closes the loop: Applies reinforcement learning update based on the reward.
        
        Args:
            reward (float): Positive for success (Goal Achieved), Negative for failure (Death).
            scope (str): Which part of the brain to update.
        """
        if scope == 'motor' and self.current_plan['motor_log_probs'] is not None:
            # PPO Hyperparameters
            ppo_epochs = 4
            clip_param = 0.2
            
            # Data
            parietal_latent = self.current_plan['parietal_latent']
            motor_seq = self.current_plan['motor_seq']
            old_log_probs = self.current_plan['motor_log_probs'].detach()
            old_value = self.current_plan['value'].detach()
            
            # Returns and Advantage
            returns = torch.tensor([reward], device=self.device).expand_as(old_value)
            advantage = returns - old_value
            
            # Prepare input for re-evaluation (Prepend SOS)
            batch_size = motor_seq.size(1)
            sos = torch.zeros(1, batch_size, dtype=torch.long, device=self.device)
            input_seq = torch.cat([sos, motor_seq], dim=0)[:-1] # Inputs: SOS, A, B... Targets: A, B, C...
            
            for _ in range(ppo_epochs):
                # Re-evaluate
                logits, current_value = self.models['motor'](parietal_latent, target_seq=input_seq)
                
                # Calculate new log probs
                log_probs = F.log_softmax(logits, dim=-1)
                # Gather log probs for the actions taken
                action_log_probs = log_probs.gather(2, motor_seq.unsqueeze(-1)).squeeze(-1)
                
                # Sum over sequence to get trajectory log prob
                trajectory_log_prob = action_log_probs.sum(dim=0)
                old_trajectory_log_prob = old_log_probs.sum(dim=0)
                
                # Ratio
                ratio = torch.exp(trajectory_log_prob - old_trajectory_log_prob)
                
                # Losses
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(current_value, returns)
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizers['motor'].zero_grad()
                loss.backward()
                self.optimizers['motor'].step()
                
            self.current_plan['motor_log_probs'] = None

    def clear_plan(self):
        """Resets the current plan."""
        self.current_plan = {k: None for k in self.current_plan}