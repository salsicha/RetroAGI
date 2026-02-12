"""Main script to run the agent."""
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import retro
from torchvision import transforms

from src.models.occipital import OccipitalLobe
from src.models.temporal import TemporalLobe
from src.models.parietal import ParietalLobe
from src.models.frontal import FrontalLobe
from src.models.motor import MotorLobe
from src.utils.tutor import Tutor

def main():
    """Main function to run the agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    try:
        env = retro.make(game='SuperMarioBros-Nes')
    except Exception as e:
        print(f"Error creating retro env: {e}")
        return

    # Dimensions
    dim_what = 64
    dim_where = 64
    dim_temporal_hidden = 128
    dim_frontal_hidden = 128
    dim_parietal_hidden = 128
    
    # Initialize models
    occipital_lobe = OccipitalLobe(latent_dim=128).to(device)
    temporal_lobe = TemporalLobe(input_dim=dim_what + dim_parietal_hidden, hidden_dim=dim_temporal_hidden, vocab_size=50).to(device)
    frontal_lobe = FrontalLobe(input_dim=dim_parietal_hidden, latent_dim=dim_frontal_hidden, vocab_size=50).to(device)
    parietal_lobe = ParietalLobe(input_dim=dim_where + dim_temporal_hidden + dim_frontal_hidden, latent_dim=dim_parietal_hidden).to(device)
    motor_lobe = MotorLobe(input_dim=dim_parietal_hidden, action_space=env.action_space.shape[0]).to(device)

    # Initialize Tutor
    tutor = Tutor(device=device)

    # Optimizers
    opt_occipital = optim.Adam(occipital_lobe.parameters(), lr=1e-4)
    opt_motor = optim.Adam(motor_lobe.parameters(), lr=1e-4)
    opt_frontal = optim.Adam(frontal_lobe.parameters(), lr=1e-4)
    
    # Loss functions
    criterion_recon = nn.MSELoss()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    obs = env.reset()
    
    prev_parietal = torch.zeros(1, dim_parietal_hidden).to(device)
    prev_frontal = torch.zeros(1, dim_frontal_hidden).to(device)
    prev_temporal = torch.zeros(1, dim_temporal_hidden).to(device)

    # Training Flags
    use_teacher_forcing = True # If True, pass Tutor ground truth to next lobe instead of model output

    try:
        step = 0
        while True:
            # 1. Capture and Preprocess
            img_tensor = transform(obs).unsqueeze(0).to(device)

            # 2. Forward Pass
            # Occipital
            reconstructed, (what, where) = occipital_lobe(img_tensor), occipital_lobe.get_latent(img_tensor)
            
            # Temporal
            generated_seq, new_temporal = temporal_lobe(what, prev_parietal)
            new_temporal = new_temporal.squeeze(0)
            
            # Parietal
            # Teacher Forcing: If enabled, we could replace 'where' with tutor's sprite map (if dimensions matched)
            # For now, we just pass the model output, but we will train against the tutor.
            parietal_latent, objectives_map = parietal_lobe(where, new_temporal, prev_frontal)
            
            # Frontal
            frontal_latent, goals, goal_map = frontal_lobe(parietal_latent)
            
            # Motor
            # If we had a "Perfect Action" from tutor, we could force it here.
            action_probs = motor_lobe(parietal_latent)

            # 3. Action Selection (Sample for exploration)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            
            action = np.zeros(env.action_space.shape[0], dtype=np.int8)
            action[action_idx.item()] = 1 # Press one button
            
            # 4. Environment Step
            next_obs, rew, done, info = env.step(action)
            
            # 5. Tutor Ground Truth
            tutor_data = tutor.get_ground_truth(obs, info, env.action_space)

            # 5. Online Training
            # Train Occipital (Reconstruction)
            loss_recon = criterion_recon(reconstructed, img_tensor)
            
            # Train Parietal (Objective Map vs Tutor Truth)
            # Assuming tutor_data['objective_map'] matches parietal output shape
            loss_parietal = criterion_recon(objectives_map, tutor_data['objective_map'])
            
            # Train Frontal (Goal Map vs Tutor Truth)
            loss_frontal = criterion_recon(goal_map, tutor_data['goal_map'])
            
            opt_occipital.zero_grad()
            opt_frontal.zero_grad()
            
            loss_recon.backward(retain_graph=True) # Retain because other lobes might need grads from occipital latent
            loss_frontal.backward(retain_graph=True)
            
            opt_occipital.step()
            opt_frontal.step()

            # Train Motor (Policy Gradient - REINFORCE one step)
            # Very basic: Minimize -log_prob * reward
            log_prob = dist.log_prob(action_idx)
            loss_motor = -log_prob * rew
            opt_motor.zero_grad()
            loss_motor.backward()
            opt_motor.step()

            if step % 10 == 0:
                print(f"Step {step}: Recon Loss={loss_recon.item():.4f}, Par Loss={loss_parietal.item():.4f}, Reward={rew}")

            # 6. Visualization (Headless safe)
            try:
                env.render()
                cv2.imshow("Reconstructed", cv2.cvtColor(reconstructed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR))
                cv2.imshow("Parietal Objectives", objectives_map.squeeze(0).squeeze(0).detach().cpu().numpy())
                cv2.imshow("Frontal Goal", goal_map.squeeze(0).squeeze(0).detach().cpu().numpy())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass # Ignore render errors

            obs = next_obs
            prev_parietal = parietal_latent.detach()
            prev_frontal = frontal_latent.detach()
            prev_temporal = new_temporal.detach()
            step += 1

            if done:
                obs = env.reset()
                prev_parietal = torch.zeros(1, dim_parietal_hidden).to(device)
                prev_frontal = torch.zeros(1, dim_frontal_hidden).to(device)
                prev_temporal = torch.zeros(1, dim_temporal_hidden).to(device)

    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()