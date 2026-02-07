"""
Game Recorder and Data Generator
Runs the game with a heuristic agent and uses the Synthesizer to create a dataset.
"""
import os
import cv2
import json
import torch
import numpy as np
import retro
from .synthesizer import DataSynthesizer

class DataGenerator:
    def __init__(self, game='SuperMarioBros-Nes', output_dir='data/synthetic'):
        self.game = game
        self.output_dir = output_dir
        self.synthesizer = DataSynthesizer()
        
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def run(self, num_episodes=5, max_steps=500):
        try:
            env = retro.make(game=self.game)
        except Exception as e:
            print(f"Error loading game {self.game}: {e}")
            return

        total_frames = 0
        
        for episode in range(num_episodes):
            print(f"Generating Episode {episode+1}/{num_episodes}...")
            obs = env.reset()
            prev_info = None
            
            for step in range(max_steps):
                # Heuristic Policy: Simple "Go Right and Jump Randomly"
                # NES Actions: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
                # Index mapping might vary, assuming standard Retro:
                # 0: B, 1: NULL, 2: SELECT, 3: START, 4: UP, 5: DOWN, 6: LEFT, 7: RIGHT, 8: A
                action = np.zeros(env.action_space.shape[0], dtype=np.int8)
                
                # Always Right (7)
                action[7] = 1
                
                # Randomly Jump (8: A) or Run (0: B)
                if np.random.rand() < 0.1: # Jump
                    action[8] = 1
                if np.random.rand() < 0.5: # Run
                    action[0] = 1
                
                next_obs, rew, done, info = env.step(action)
                
                # --- Synthesis ---
                # 1. Temporal Label
                temporal_text = self.synthesizer.generate_temporal_label(info, prev_info)
                
                # 2. Parietal Target (Map)
                parietal_map = self.synthesizer.generate_parietal_target(info)
                
                # 3. Frontal Label
                frontal_text = self.synthesizer.generate_frontal_label(info)
                
                # --- Save Data ---
                frame_id = f"ep{episode}_step{step}"
                
                # Save Image
                # Swap RGB to BGR for OpenCV
                if next_obs.shape[2] == 3:
                    img_bgr = cv2.cvtColor(next_obs, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = next_obs
                cv2.imwrite(os.path.join(self.images_dir, f"{frame_id}.png"), img_bgr)
                
                # Save Label Info
                label_data = {
                    "frame_id": frame_id,
                    "action": action.tolist(),
                    "reward": float(rew),
                    "temporal_text": temporal_text,
                    "frontal_text": frontal_text,
                    # Parietal map is large, maybe save as npy or just recreate it during training? 
                    # Let's save it as a separate npy file for "ground truth" usage.
                    "parietal_path": f"labels/{frame_id}_parietal.npy"
                }
                
                np.save(os.path.join(self.labels_dir, f"{frame_id}_parietal.npy"), parietal_map)
                
                with open(os.path.join(self.labels_dir, f"{frame_id}.json"), 'w') as f:
                    json.dump(label_data, f)
                
                obs = next_obs
                prev_info = info
                total_frames += 1
                
                if done:
                    break
        
        env.close()
        print(f"Finished. Total frames generated: {total_frames}")
