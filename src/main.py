"""
src/main.py
Main entry point for RetroAGI.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import cv2
import retro

# Import our models
# Assuming models.py is in the same directory or src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import OccipitalLobe, TemporalLobe, Hippocampus, PrefrontalLobe, MotorLobe

# Configuration
FRAME_SIZE = 64
NUM_KEYPOINTS = 16
LATENT_DIM = NUM_KEYPOINTS * 2
# We will determine NUM_ACTIONS based on retro action space later
NUM_ACTIONS = 9 # Usually 9 buttons for NES in retro
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "data/weights/"

def preprocess(frame):
    """Resize and normalize frame."""
    if frame is None: return torch.zeros(1, 3, FRAME_SIZE, FRAME_SIZE)
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    frame = frame.transpose(2, 0, 1) # HWC -> CHW
    frame = frame / 255.0
    return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def save_models(occipital, hippocampus, prefrontal, motor, epoch):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    torch.save({
        'occipital': occipital.state_dict(),
        'hippocampus': hippocampus.state_dict(),
        'prefrontal': prefrontal.state_dict(),
        'motor': motor.state_dict(),
        'temporal': temporal.state_dict(),
    }, f"{SAVE_PATH}/brain_epoch_{epoch}.pth")
    print(f"Saved weights to {SAVE_PATH}/brain_epoch_{epoch}.pth")

def main():
    # 1. Initialize Environment
    env = retro.make(game='SuperMarioBros-Nes')

    # 2. Initialize Lobes
    occipital = OccipitalLobe(num_keypoints=NUM_KEYPOINTS).to(DEVICE) # Out: 32
    temporal = TemporalLobe(num_keypoints=NUM_KEYPOINTS) # Out: 32
    
    # Hippocampus Input: Occipital (32) + Temporal (32) = 64
    hippocampus = Hippocampus(input_dim=LATENT_DIM * 2).to(DEVICE) 
    
    # Prefrontal Input: Temporal (32) + Hippocampus (64) = 96
    # Output: Goal for Motor (64)
    prefrontal = PrefrontalLobe(input_dim=LATENT_DIM * 3, output_dim=LATENT_DIM * 2).to(DEVICE)
    
    # Motor Input: Prefrontal Output (64)
    motor = MotorLobe(input_dim=LATENT_DIM * 2, num_actions=NUM_ACTIONS).to(DEVICE)

    # Optimizers
    opt_vision = optim.Adam(occipital.parameters(), lr=1e-4)
    opt_brain = optim.Adam(list(hippocampus.parameters()) + 
                           list(prefrontal.parameters()) + 
                           list(motor.parameters()), lr=1e-4)

    # Prefrontal Setting (0: speed, 1: coins)
    prefrontal.set_mode(1.0) 

    frame_count = 0
    epoch = 0
    
    obs = env.reset()
    done = False
    
    print("Starting RetroAGI Loop...")
    
    while True:
        # --- PERCEPTION (Occipital) ---
        img_tensor = preprocess(obs)
        
        # Reconstructive Learning (Unsupervised)
        z_vision, recon = occipital(img_tensor)
        loss_recon = torch.mean((img_tensor - recon) ** 2)
        
        opt_vision.zero_grad()
        loss_recon.backward(retain_graph=True)
        opt_vision.step()
        
        # --- MEMORY & DYNAMICS (Temporal) ---
        # Detach z for temporal processing (pyhgf uses numpy)
        z_np = z_vision.detach().cpu().numpy()
        
        # Update HGF beliefs
        # Temporal lobe tracks the "where" (keypoints) dynamics
        pred_z_np = temporal.update(z_np)
        pred_z = torch.tensor(pred_z_np, dtype=torch.float32).to(DEVICE).view(z_vision.shape)
        
        # --- SPATIAL MAPPING (Hippocampus) ---
        # Input: Vision Latent + Temporal Prediction
        # GEMINI.md: "takes the latent output of the occipital and temporal models"
        hippo_input = torch.cat([z_vision, pred_z], dim=1)
        
        # Reconstructive/Associative Learning
        mem_recon, spatial_map = hippocampus(hippo_input)
        loss_mem = torch.mean((hippo_input - mem_recon) ** 2)
        
        # --- PLANNING (Prefrontal) ---
        # Determine Goal
        # GEMINI.md: "takes latent input from the temporal and hippocampus"
        pf_input = torch.cat([pred_z, mem_recon], dim=1)
        motor_goal = prefrontal(pf_input)
        
        # --- ACTION (Motor) ---
        logits = motor(motor_goal)
        action_prob = torch.softmax(logits, dim=1)
        action = torch.multinomial(action_prob, 1).item()
        
        # --- ENVIRONMENT STEP ---
        next_obs, reward, done, info = env.step(action)
        
        # --- REINFORCEMENT / PREDICTIVE LEARNING ---
        # For this example, we use a simple Reinforce-like signal or Prediction Error
        # In pure Predictive Coding, we minimize Proprioceptive Prediction Error.
        # Here we mix in standard RL reward for practical gameplay.
        
        # Total Brain Loss
        loss_brain = loss_mem - torch.tensor(reward).to(DEVICE) * 0.1 # Simple reward maximization
        
        opt_brain.zero_grad()
        loss_brain.backward()
        opt_brain.step()
        
        obs = next_obs
        frame_count += 1
        
        if done:
            obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            print(f"Episode finished. Frames: {frame_count}")
        
        # Periodic Save
        if frame_count % 5000 == 0:
            epoch += 1
            save_models(occipital, hippocampus, prefrontal, motor, epoch)
            
        if frame_count > 100000: # Safety break
            break

    env.close()

if __name__ == "__main__":
    main()
