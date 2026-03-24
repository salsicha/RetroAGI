import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import cv2
import retro

# Add src to path so we can import models
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))
from models import OccipitalLobe, TemporalLobe, Hippocampus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SIZE = 64
NUM_KEYPOINTS = 16
LATENT_DIM = NUM_KEYPOINTS * 2

def main():
    # 1. Initialize Environment
    env = retro.make(game='SuperMarioBros-Nes')

    # 2. Load Classically Trained Tutor
    print("Loading Tutor...")
    try:
        tutor = models.segmentation.deeplabv3_resnet50(pretrained=False)
    except TypeError:
        tutor = models.segmentation.deeplabv3_resnet50(weights=None)
        
    tutor.classifier = DeepLabHead(2048, 6)
    
    # Path to tutor weights
    tutor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tutor/segmentation/MarioSegmentationModel.pth')
    
    if os.path.exists(tutor_path):
        tutor.load_state_dict(torch.load(tutor_path, map_location=DEVICE))
        print("Tutor weights loaded successfully.")
    else:
        print(f"Error: Tutor weights not found at {tutor_path}")
        return
        
    tutor = tutor.to(DEVICE)
    tutor.eval()
    tutor_transform = transforms.ToTensor()

    # 3. Initialize PyHGF Models
    print("Initializing RetroAGI Lobes...")
    occipital = OccipitalLobe(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
    temporal = TemporalLobe(num_keypoints=NUM_KEYPOINTS)
    hippocampus = Hippocampus(input_dim=LATENT_DIM * 2).to(DEVICE)
    
    # Joint optimizer for bootstrapped model
    optimizer = optim.Adam(
        list(occipital.parameters()) + list(hippocampus.parameters()), 
        lr=1e-3
    )

    # 4. Bootstrap Loop
    num_steps = 1000
    if len(sys.argv) > 1 and sys.argv[1] == '--steps':
        num_steps = int(sys.argv[2])
        
    print(f"Starting Bootstrap Process for {num_steps} steps...")
    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    
    for step in range(num_steps):
        # Determine random action to explore
        action = env.action_space.sample()
        next_obs, _, done, info = env.step(action)
        
        # --- A. Tutor Pass (Ground Truth Generation for Hippocampus) ---
        with torch.no_grad():
            img_tensor_tutor = tutor_transform(obs).unsqueeze(0).to(DEVICE)
            out = tutor(img_tensor_tutor)['out'][0]
            # Get the exact labels map for semantic classes
            seg_mask = torch.argmax(out.squeeze(), dim=0) # shape: (240, 256)
        
        # Resize to match our model's resolution (64x64) and create target cross-entropy tensor
        target_mask_np = seg_mask.cpu().numpy().astype(np.uint8)
        target_mask_resized = cv2.resize(target_mask_np, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_NEAREST)
        # Class targets need shape (Batch, 64, 64) for CrossEntropyLoss
        target_tensor = torch.tensor(target_mask_resized, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # --- B. Predictive Coding Pass ---
        # 1. Occipital Input is the RAW game frame
        obs_resized = cv2.resize(obs, (FRAME_SIZE, FRAME_SIZE))
        obs_input = obs_resized.transpose(2, 0, 1) / 255.0
        input_tensor = torch.tensor(obs_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Occipital processing
        z_vision, recon = occipital(input_tensor)
        loss_occ = F.mse_loss(recon, input_tensor) # Occipital target is raw frame
        
        # 2. Temporal processing
        z_np = z_vision.detach().cpu().numpy()
        pred_z_np = temporal.update(z_np)
        pred_z = torch.tensor(pred_z_np, dtype=torch.float32).to(DEVICE).view(z_vision.shape)
        
        # 3. Hippocampus processing
        hippo_input = torch.cat([z_vision, pred_z], dim=1)
        mem_recon, spatial_map = hippocampus(hippo_input)
        
        # associative loss for mapping
        loss_mem = F.mse_loss(mem_recon, hippo_input.detach())
        # The key bootstrapping loss! Train hippocampus decoder to predict semantic locations
        loss_hippo_spatial = F.cross_entropy(spatial_map, target_tensor)
        
        # Combined loss
        loss = loss_occ + loss_mem + loss_hippo_spatial
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps} | Occ Loss: {loss_occ.item():.4f} | Spatial Loss: {loss_hippo_spatial.item():.4f}")
            
        if done:
            obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        else:
            obs = next_obs
            
    # 5. Save Bootstrapped weights
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/weights')
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(occipital.state_dict(), os.path.join(save_dir, 'bootstrapped_occipital.pth'))
    torch.save(hippocampus.state_dict(), os.path.join(save_dir, 'bootstrapped_hippocampus.pth'))
    # Optional: save temporal
    torch.save(temporal.state_dict(), os.path.join(save_dir, 'bootstrapped_temporal.pkl') if hasattr(temporal, 'save') else os.path.join(save_dir, 'bootstrapped_temporal.pth'))
    
    print(f"Bootstrapping complete! Weights saved to {save_dir}")

if __name__ == '__main__':
    main()
