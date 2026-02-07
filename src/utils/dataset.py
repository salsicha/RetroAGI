"""
RetroAGI Dataset
PyTorch Dataset for loading synthetic RetroAGI data.
"""
import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class RetroAGIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.transform = transform
        
        self.frames = [f.replace('.png', '') for f in os.listdir(self.images_dir) if f.endswith('.png')]
        self.frames.sort()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_id = self.frames[idx]
        
        # Load Image
        img_path = os.path.join(self.images_dir, f"{frame_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        # Load Labels
        label_path = os.path.join(self.labels_dir, f"{frame_id}.json")
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            
        # Load Parietal Map
        parietal_path = os.path.join(self.data_dir, label_data['parietal_path'])
        parietal_map = np.load(parietal_path)
        
        # Prepare targets
        action = torch.tensor(label_data['action'], dtype=torch.float32)
        temporal_text = label_data['temporal_text']
        frontal_text = label_data['frontal_text']
        parietal_target = torch.from_numpy(parietal_map).float()
        
        return {
            "image": image,
            "action": action,
            "temporal_text": temporal_text,
            "frontal_text": frontal_text,
            "parietal_map": parietal_target
        }
