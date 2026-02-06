## Mario

import cv2
from PIL import Image
from torchvision import datasets, transforms, models
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead,DeepLabV3
from torchvision import models
import matplotlib.pyplot as plt

import cv2
import numpy as np

import os

# sys.path.append("../scripts/")

os.chdir('/scripts')

import matplotlib.pyplot as plt
import sys

import retro

# game_list = retro.data.list_games()
# print(game_list)


env = retro.make(game='SuperMarioBros-Nes')
# env = retro.make(game='SuperMarioBros3-Nes')

obs = env.reset()

count = 2000

while True:
    count -= 1
    if count < 0:
        break

    obs, rew, done, term, info = env.step(env.action_space.sample())

    # print(f"Observation shape: {obs.shape} \n")
    # print(f"Reward: {rew} \n")
    # print(f"Done?: {done} \n")
    # print(f"Terminated?: {term} \n")
    # print(f"info: {info} \n")

    env.render()
    if done:
        obs = env.reset()
env.close()
