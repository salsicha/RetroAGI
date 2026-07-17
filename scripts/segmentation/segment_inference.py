"""EXAMPLE ONLY: legacy CNN (DeepLab) segmentation inference.

Kept as a reference for the original CNN-based approach; not part of the
supported pipeline. All supported vision paths use the ViT-backed
FullSMBSegmentationVision in retroagi.stages.full_smb.vision.
"""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import csv
import os
import random
import sys
import time
from os.path import join
from pathlib import Path

import numpy as np

# import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as TF

# from torchvision.datasets.utils import download_file_from_google_drive
from PIL import Image
from torch import optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core import select_device

# Defaults resolved relative to this file (not the CWD).
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "MarioSegmentationModel.pth")
DEFAULT_FRAME_PATH = str(PROJECT_ROOT / "data" / "vit" / "preview_0.png")


class SegmenInf:

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="auto"):
        self.device = select_device(device)
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        self.model.classifier = DeepLabHead(2048, 6)

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.to(self.device)
        self.model.eval()

    # Define the helper function
    def decode_segmap(self, image, nc=21):
        ## Color palette for visualization of the 21 classes
        label_colors = np.array(
            [
                (0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (0, 0, 255),
                (127, 127, 0),
                (0, 255, 0),
                (255, 0, 0),
                (255, 255, 0),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128),
                (128, 128, 128),
                (64, 0, 0),
                (192, 0, 0),
                (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0),
                (64, 0, 128),
                (192, 0, 128),
                (64, 128, 128),
                (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0),
                (128, 64, 0),
                (0, 192, 0),
                (128, 192, 0),
                (0, 64, 128),
            ]
        )

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for class_id in range(0, nc):
            idx = image == class_id
            r[idx] = label_colors[class_id, 0]
            g[idx] = label_colors[class_id, 1]
            b[idx] = label_colors[class_id, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def segment(self, net, path, show_orig=True, transform=transforms.ToTensor(), dev=None):
        dev = self.device if dev is None else select_device(dev)
        img = Image.open(path)
        # if show_orig: plt.imshow(img); plt.axis('off'); plt.show()

        input_image = transform(img).unsqueeze(0).to(dev)
        out = net(input_image)["out"][0]

        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        segm_rgb = self.decode_segmap(segm)

        return segm_rgb

        # plt.imshow(segm_rgb)
        # plt.axis('off')
        # plt.savefig('1_1.png', format='png',dpi=300,bbox_inches = "tight")
        # plt.show()

    def compare(self, net, net2, path, show_orig=True, transform=transforms.ToTensor(), dev=None):
        dev = self.device if dev is None else select_device(dev)
        img = Image.open(path)
        # if show_orig: plt.imshow(img); plt.axis('off'); plt.show()

        input_image = transform(img).unsqueeze(0).to(dev)
        out = net(input_image)["out"][0]

        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        segm_rgb = self.decode_segmap(segm)

        # plt.imshow(segm_rgb)
        # plt.axis('off'); plt.show()

        input_image = transform(img).unsqueeze(0).to(dev)
        out = net2(input_image)["out"][0]

        segm2 = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        segm_rgb2 = self.decode_segmap(segm2)
        # plt.imshow(segm_rgb2); plt.axis('off'); plt.show()

        return segm_rgb, segm_rgb2

    def run(self, frame=DEFAULT_FRAME_PATH):

        # print(frame)

        return self.segment(self.model, frame)
