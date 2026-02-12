"""
Feedback Loop Manager
Handles the storage of high-level goals and implements the 'Stop and Learn' logic.
"""
import torch

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
            'motor_log_probs': None
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

    def set_motor_execution(self, sequence, log_probs):
        """
        Save the generated motor sequence and its log probabilities.
        """
        self.current_plan['motor_seq'] = sequence
        self.current_plan['motor_log_probs'] = log_probs

    def apply_feedback(self, reward, scope='motor'):
        """
        Closes the loop: Applies reinforcement learning update based on the reward.
        
        Args:
            reward (float): Positive for success (Goal Achieved), Negative for failure (Death).
            scope (str): Which part of the brain to update.
        """
        if scope == 'motor' and self.current_plan['motor_log_probs'] is not None:
            # Policy Gradient Update (REINFORCE)
            log_probs = self.current_plan['motor_log_probs']
            trajectory_log_prob = log_probs.sum()
            loss = -reward * trajectory_log_prob
            
            if 'motor' in self.optimizers:
                self.optimizers['motor'].zero_grad()
                loss.backward()
                self.optimizers['motor'].step()
                
            self.current_plan['motor_log_probs'] = None

    def clear_plan(self):
        """Resets the current plan."""
        self.current_plan = {k: None for k in self.current_plan}