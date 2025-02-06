import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
#from torchvision.datasets.utils import download_file_from_google_drive

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import sys
import argparse
import os
import time
from os.path import join
import csv

import cv2

from torchvision.models.segmentation.deeplabv3 import DeepLabHead,DeepLabV3
from torchvision import models

print('PyTorch version:', torch.__version__)





class Configuration:
  def __init__(self):
    self.experiment_name = "Training the Super Mario Segmentation Model"
    
    # Paramters for the first part
    self.pre_load    = "True" ## Load dataset in memory
    self.pre_trained = "True"
    self.num_classes = 6
    self.ignore_label = 255
    self.training_data_proportion = 0.8 # Proportion of images of the dataset to be used for training

    self.lr    = 0.001  # 0.001 if pretrained weights from pytorch. 0.1 if scratch
    self.epoch = 45     # Play with this if training takes too long
    self.M = [37,42]         # If training from scratch, reduce learning rate at some point

    self.batch_size = 4  # Training batch size
    self.test_batch_size = 4  # Test batch size
    self.model_file_name = "MarioSegmentationModel.pth"
    
    self.dataset_root = "/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset"
    self.download   = False
    
    self.seed = 271828


class MarioDataset(data.Dataset):
    def __init__(self, args, mode, transform_input=transforms.ToTensor(), transform_mask=transforms.ToTensor()):
        self.args = args
        # self.folder = args.dataset_root
        self.folder = "/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset"

        #If you change how you create the dataset you may need to modify this:
        self.images_in_dataset = len(os.listdir(self.folder+"/PNG"))
        training_images_no = int(self.images_in_dataset*0.8)
        self.imgs = np.arange(training_images_no) if mode == 'train' else np.arange(training_images_no,self.images_in_dataset)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set or proportions')

        self.mode = mode
        self.transform_input = transform_input
        self.transform_mask = transform_mask

    # Default trasnformations on train data
    def transform(self, image, mask):

        i, j, h, w = transforms.RandomCrop.get_params(image, (224,224))
        
        image = TF.crop(image,i,j,h,w)
        mask  = TF.crop(mask,i,j,h,w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        return image, mask

    # Default trasnformations on test data
    def test_transform(self, image, mask):
        #224x224 center crop: 
        image = TF.center_crop(image,[224,224])
        mask  = TF.center_crop(mask,[224,224])

        return image, mask
    
    def __getitem__(self, index):

        if self.mode == 'test':
            img = Image.open(+self.folder+"/PNG/"+str(index)+".png").convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return str(index), img

            #Load RGB image
        img = Image.open(self.folder+"/PNG/"+str(index)+".png").convert('RGB')

        if self.mode == 'train':

            #Load class mask
            mask = Image.open(self.folder+"/Labels/"+str(index)+".png")
        else:
            mask = Image.open(self.folder+"/Labels/"+str(index)+".png")

            ##Transform using default transformations
        if self.mode=="train":
              img, mask = self.transform(img,mask)
        else:
              img, mask = self.test_transform(img,mask)

        if self.transform_input is not None:
           img = self.transform_input(img)
        if self.transform_mask is not None:
            mask = 255*self.transform_mask(mask)

        return img, mask.long()

    def __len__(self):
        return len(self.imgs)




class GridGenerator():

    def __init__(self, width=17, height=15):
        self.grid_w = width
        self.grid_h = height
        self.grid = np.array([[['[background]', 'Empty', 'Empty'] for a in np.arange(
            self.grid_w)] for b in np.arange(self.grid_h)], dtype='object')
        self.free_floor = np.arange(self.grid_w)
        self.havefloor = []
        self.floor_level = 12

    def GenerateClouds(self):
        # Rows 9,10 over the floor
        block_n = np.random.choice(np.arange(3), p=[0.1, 0.5, 0.4])
        # To not overlap hills and clouds
        #hill_av = np.arange(16)
        if block_n != 0:
            # where are they placed
            block_l = np.random.permutation(self.grid_w)[:block_n]

            for i in np.arange(block_n):
                # how many blocks stick together:
                size = np.random.choice(np.arange(3, 6), p=[0.4, 0.3, 0.3])
                # what type are the blocks. Bricks? or boxes?
                for k in np.arange(size):
                    # to place lengthy blocks around initial position
                    offset = np.power(-1, k)*((k+1)//2)
                    position = block_l[i]+offset
                    # if position is inside the grid, set the label
                    if position >= 0 and position < self.grid_w:
                        # np.delete(hill_av,position)
                        if k < size - 2:
                            self.grid[2, position, 0] = '[cloud_tm]'
                            self.grid[3, position, 0] = '[cloud_bm]'
                        else:
                            if offset < 0:
                                if 'cloud' in self.grid[2, position, 0]:
                                    self.grid[2, position, 0] = '[cloud_tm]'
                                    self.grid[3, position, 0] = '[cloud_bm]'
                                else:
                                    self.grid[2, position, 0] = '[cloud_tl]'
                                    self.grid[3, position, 0] = '[cloud_bl]'
                            else:
                                if 'cloud' in self.grid[2, position, 0]:
                                    self.grid[2, position, 0] = '[cloud_tm]'
                                    self.grid[3, position, 0] = '[cloud_bm]'
                                else:
                                    self.grid[2, position, 0] = '[cloud_tr]'
                                    self.grid[3, position, 0] = '[cloud_br]'

    def GenerateHills(self):
        # Generate hills.

        # Number of hills
        block_n = np.random.choice(np.arange(3), p=[0.1, 0.45, 0.45])
        if block_n != 0:
            block_h = np.random.choice([0, 1], size=block_n, p=[0.5, 0.5])
            # generate placements (doesnt take into account hill height)
            block_l = [5*i + x for i,
                       x in enumerate(sorted(np.random.randint(0, 7, size=3)))]

            free_floor_for_hills = self.free_floor
            for i in np.arange(block_n):
                # place tip
                self.grid[10+block_h[i], block_l[i], 0] = '[hill_t]'
                # middle row
                self.grid[11+block_h[i], block_l[i], 0] = '[hill_mm]'
                if block_l[i]-1 >= 0:
                    self.grid[11+block_h[i], block_l[i]-1, 0] = '[hill_ml]'
                    # bottom row, saves 1 comparison
                    self.grid[12+block_h[i], block_l[i]-1, 0] = '[hill_bl]'

                if block_l[i]+1 < self.grid_w:
                    self.grid[11+block_h[i], block_l[i]+1, 0] = '[hill_mr]'
                    # bottom row, saves 1 comparison
                    self.grid[12+block_h[i], block_l[i]+1, 0] = '[hill_br]'

                # The rest of the bottom row
                if block_l[i] - 2 >= 0:
                    self.grid[12+block_h[i], block_l[i]-2, 0] = '[hill_bll]'
                if block_l[i]+2 < self.grid_w:
                    self.grid[12+block_h[i], block_l[i]+2, 0] = '[hill_brr]'

                self.grid[12+block_h[i], block_l[i], 0] = '[hill_bm]'

                free_floor_for_hills = [x for x in free_floor_for_hills if x not in [
                    block_l[i]-2, block_l[i]-1, block_l[i], block_l[i]+1, block_l[i]+2]]

    def GenerateBushes(self):
        # Generates bushes (1 to 2 per image)
        block_n = np.random.choice(np.arange(1, 3), p=[0.6, 0.4])
        #print("Free floor for bushes: ",free_floor)
        if block_n != 0:
            # where are they placed
            block_l = np.random.permutation(self.free_floor)[:block_n]

            for i in np.arange(block_n):
                # how many blocks stick together:
                size = np.random.choice(np.arange(3, 6), p=[0.6, 0.3, 0.1])
                # what type are the blocks. Bricks? or boxes?
                for k in np.arange(size):
                    # to place lengthy blocks around initial position
                    offset = np.power(-1, k)*((k+1)//2)
                    position = block_l[i]+offset
                    # if position is inside the grid, set the label

                    if position >= 0 and position < self.grid_w:
                        # np.delete(hill_av,position)
                        if k < size - 2:
                            self.grid[12, position, 1] = '[bush_m]'
                        else:
                            if offset < 0:
                                if 'bush' in self.grid[12, position, 1]:
                                    self.grid[12, position, 1] = '[bush_m]'
                                else:
                                    self.grid[12, position, 1] = '[bush_l]'
                            else:
                                if 'bush' in self.grid[12, position, 1]:
                                    self.grid[12, position, 1] = '[bush_m]'
                                else:
                                    self.grid[12, position, 1] = '[bush_r]'

    def GenerateFloor(self):
        # Generate floor tiles
        # Last two rows will be floor, with random holes
        prob = 0.02
        holes = []
        for i in np.arange(self.grid_w):
            # randomly decide if there is a hole or not
            if np.random.uniform() <= prob:  # 2% of generating hole
                # if it generates a hole, its more probable that it has another one next to it
                if prob == 0.02:
                    prob = 0.85
                else:
                    prob = 0.02
                holes.append(i)
                continue
            else:  # no hole, fill with floor
                self.grid[13, i, 1] = '[floor]'
                self.grid[14, i, 1] = '[floor]'
                self.havefloor.append(i)

        for i in holes:
            self.free_floor = np.delete(
                self.free_floor, np.where(self.free_floor == i), axis=0)

    def GenerateTrees(self):
        # Get the number of available slots in the floor.
        trees = np.random.choice(np.arange(0, 4), p=[0.25, 0.45, 0.23, 0.07])

        if trees != 0:
            # 2 Types of trees, big top and small top. Both share trunk
            for tree in np.arange(trees):
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before placing trees? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)

                tree_type = np.random.choice(np.arange(0, 2))

                if tree_type == 0:  # trunk 2 top
                    self.grid[self.floor_level, place, 1] = '[tree_trunk]'
                    self.grid[self.floor_level-1, place, 1] = '[tree_tb]'
                    self.grid[self.floor_level-2, place, 1] = '[tree_tt]'
                else:
                    self.grid[self.floor_level, place, 1] = '[tree_trunk]'
                    self.grid[self.floor_level-1, place, 1] = '[tree_small]'

    def GenerateBlocks(self):
        # This function places unbrickable blocks
        # Must work similar to placing bricks
        max_blocks = 4
        block_number = np.random.choice(
            np.arange(0, max_blocks), p=[0.7, 0.20, 0.07, 0.03])
        #print("Free floor before blocks {}".format(self.free_floor))

        placed_block = []

        if block_number != 0:
            for block in np.arange(block_number):
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before generating blocks? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)

                block_group_height = np.random.choice(np.arange(1, 5))

                # First place the main block
                for k in np.arange(block_group_height):
                    self.grid[self.floor_level-k, place, 2] = '[block_hard]'

                placed_block.append(place)

                # Now, roll to check wether it grows or not and in which direction. The more blocks there are, less growth width.

                grow = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])

                if grow != 0:
                    grow_n = np.random.randint(0, max_blocks-block_number)
                    for block_strip in np.arange(grow_n):
                        loc = place + block_strip*grow

                        if loc > 0 and loc < self.grid_w:
                            height = np.random.choice(
                                np.arange(1, block_group_height+1))
                            for i in np.arange(height):
                                self.grid[self.floor_level-i,
                                          loc, 2] = '[block_hard]'

                            placed_block.append(loc)

            for i in placed_block:
                self.free_floor = np.delete(
                    self.free_floor, np.where(self.free_floor == i), axis=0)
            #print("Free floor after blocks {}".format(self.free_floor))

    def GeneratePipes(self):
        # Over those that have floor, pipes can appear, with low probability.
        pipes = np.random.choice(np.arange(0, 4), p=[0.7, 0.20, 0.07, 0.03])
        placed_pipes = []
        #print("Free floor before pipes {}".format(self.free_floor))
        if pipes != 0:
            for pipe in np.arange(pipes):
                # find where to place it (part of the pipe can be over a hole)
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before generating pipes? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)
                pipeH = np.random.choice(np.arange(2, 4))

                if place == 0:  # only the right part of the pipe will be visible
                    # if there is already a pipe there skip this one.
                    if ('pipe' in self.grid[12, place, 2]):
                        continue

                    self.grid[12, place, 2] = '[pipe_br]'

                    if pipeH == 2:
                        self.grid[11, place, 2] = '[pipe_tr]'
                    else:
                        self.grid[11, place, 2] = '[pipe_br]'
                        self.grid[10, place, 2] = '[pipe_tr]'

                    placed_pipes.append(place)
                    enemy = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])

                    if enemy > 0:
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                else:
                    if ('pipe' in self.grid[12, place, 2]) or ('pipe' in self.grid[12, place-1, 2]):
                        continue

                    self.grid[12, place, 2] = '[pipe_br]'
                    self.grid[12, place-1, 2] = '[pipe_bl]'

                    if pipeH == 2:
                        self.grid[11, place, 2] = '[pipe_tr]'
                        self.grid[11, place-1, 2] = '[pipe_tl]'
                    else:
                        self.grid[11, place, 2] = '[pipe_br]'
                        self.grid[11, place-1, 2] = '[pipe_bl]'
                        self.grid[10, place, 2] = '[pipe_tr]'
                        self.grid[10, place-1, 2] = '[pipe_tl]'

                    placed_pipes.append(place-1)
                    placed_pipes.append(place)

                    # Does it have an enemy?
                    enemy = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
                    if enemy > 0:
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                        self.grid[12-pipeH, place-1,
                                  2] = '[piranha_'+str(enemy)+']'

        for i in placed_pipes:
            self.free_floor = np.delete(
                self.free_floor, np.where(self.free_floor == i), axis=0)
        #print("Free floor after pipes {}".format(self.free_floor))

    def PlaceBlocks(self, level=0):
        if level == 0:
            # At heights 4, 8 blocks may appear. If so, they appear in chunks of random size.
            # At height 4, higher probability of blocks of length 3-5.
            # Also, at height 4 different probabilities of 1 or more appearing.
            # HEIGHT 4
            # how many blocks?
            block_n = np.random.choice(np.arange(3), p=[0.15, 0.5, 0.35])
            if block_n != 0:
                # where are they placed
                block_l = np.random.permutation(self.grid_w)[:block_n]

                for i in np.arange(block_n):
                    # how many blocks stick together:
                    size = np.random.choice(
                        np.arange(1, 7), p=[0.05, 0.1, 0.23, 0.3, 0.22, 0.1])
                    # what type are the blocks. Bricks? or boxes?
                    b_type = np.random.choice(
                        np.arange(2), size=size, p=[0.8, 0.2])
                    for k in np.arange(size):
                        b_label = '[brick]'
                        if b_type[k] == 1:
                            b_label = '[box]'

                        # to place lengthy blocks around initial position
                        offset = np.power(-1, k)*((k+1)//2)
                        position = block_l[i]+offset
                        # if position is inside the grid, set the label
                        if position >= 0 and position < self.grid_w:
                            self.grid[9, position, 1] = b_label
            # Height 8
            # how many blocks?
            block_n = np.random.choice(np.arange(3), p=[0.6, 0.25, 0.15])
            if block_n != 0:
                # where are they placed
                block_l = np.random.permutation(self.grid_w)[:block_n]

                for i in np.arange(block_n):
                    # how many blocks stick together:
                    size = np.random.choice(
                        np.arange(1, 4), p=[0.65, 0.25, 0.1])
                    # what type are the blocks. Bricks? or boxes?
                    b_type = np.random.choice(
                        np.arange(2), size=size, p=[0.7, 0.3])
                    for k in np.arange(size):
                        b_label = '[brick]'
                        if b_type[k] == 1:
                            b_label = '[box]'

                        # to place lengthy blocks around initial position
                        offset = np.power(-1, k)*((k+1)//2)
                        position = block_l[i]+offset
                        # if position is inside the grid, set the label
                        if position >= 0 and position < self.grid_w:
                            self.grid[5, position, 1] = b_label

    def GenerateMushrooms(self):
        # place some mushrooms
        block_n = np.random.choice(np.arange(4), p=[0.1, 0.25, 0.35, 0.3])
        # print("Mushrooms:",block_n)
        if block_n != 0:
            # where are they placed
            random_free_floor = np.random.permutation(self.free_floor)
            block_l = random_free_floor[:block_n]
            self.free_floor = random_free_floor[block_n:]
            size = np.random.choice(np.arange(1, 4), p=[0.65, 0.25, 0.1])

            for i in np.arange(block_n):
                b_type = np.random.choice(
                    np.arange(2), size=size, p=[0.5, 0.5])
                for k in np.arange(size):
                    b_label = '[mush_1]'
                    if b_type[k] == 1:
                        b_label = '[mush_2]'

                    self.grid[12, block_l[i], 2] = b_label

    def GenerateKoopas(self):
        # Place koopas

        koopa_n = np.random.choice(np.arange(3), p=[0.8, 0.15, 0.05])
        # print("Mushrooms:",block_n)
        if koopa_n != 0:
            # where are they placed
            random_free_floor = np.random.permutation(self.free_floor)
            block_l = random_free_floor[:koopa_n]
            self.free_floor = random_free_floor[koopa_n:]
            size = np.random.choice(np.arange(1, 4), p=[0.65, 0.25, 0.1])

            for i in np.arange(koopa_n):
                b_type = np.random.choice(
                    np.arange(2), size=size, p=[0.5, 0.5])
                for k in np.arange(size):
                    b_label = '[koopa_1]'
                    if b_type[k] == 1:
                        b_label = '[koopa_2]'

                    self.grid[12, block_l[i], 2] = b_label

    def GenerateEnemies(self):
        self.GenerateMushrooms()
        self.GenerateKoopas()

    def GenerateMario(self):
        # at last, place mario. For this, first choose if he's on the floor or jumping
        mario_state = np.random.choice(['floor', 'jump'], p=[0.8, 0.2])

        if mario_state == 'floor':
            # can be one of four states
            mario_state = np.random.choice(['idle', 'walk1', 'walk2', 'walk3'], p=[
                                           0.25, 0.25, 0.25, 0.25])
            # has to be placed on a free slot in the floor
            label = '[mario_' + mario_state + ']'

            self.grid[12, self.free_floor[0], 2] = label
            # print(grid[12,free_floor[0],2])
        else:
            # if not, choose a random position in the air, check if its free or choose again.
            # there can be mario as high as the clouds (position 2 and 3) so from 11 to 2
            # have 9 x 17 slots. Do random number between 0 and 9x17 and select via modulo
            # CHECK AGAIN SEEMS LIKE NO MARIOS ON SECOND ROW
            placed = False
            while placed == False:
                position = np.random.randint(0, 9*17)
                x = position % 17
                y = position//17
                #print("Jump", position,"x", x,"y",y)
                if self.grid[2+y, x, 1] == 'Empty' and self.grid[2+y, x, 2] == 'Empty':
                    self.grid[2+y, x, 2] = '[mario_jump]'
                    placed = True

    def GenerateGrid(self, level):
        '''This function generates the grid with the elements that will be displayed in the image.
            Elements may vary depeding on the type of level being played, using a flag variable.

            Level = 'xyz' with:
            x - Level tileset to use (not relevant for grid generation)
            y - type of level:
                - 0 means default level, with bushes and hills in the background
                - 1 means default level with trees in the background
                - 2 means underground level. No trees or bushes in the background
                - 3 means castle level (not implemented)
                - 4 means mushroom level (not implemented)

                - O means default level with alternate bushes   
                - I means default level with alternate trees
            z - background color (not relevant for grid generation)
                - 0 default blue
                - 1 black 

        '''
        # If level is type default
        if level[1] == '0' or level[1] == '1' or level[1] == 'O' or level[1] == 'I':
            self.GenerateClouds()

        # Common for all levels
        self.GenerateFloor()
        self.GenerateBlocks()
        self.PlaceBlocks()

        # Background decoration
        if level[1] == '0' or level[1] == 'O':
            self.GenerateHills()
            self.GenerateBushes()
        elif level[1] == '1' or level[1] == 'I':
            self.GenerateTrees()

        # Generate pipes
        self.GeneratePipes()

        # Generate enemies
        self.GenerateEnemies()

        # Place mario
        self.GenerateMario()

        return self.grid




'''
This file stores all loaders for all the supported sprites. All loaders receive a parameter to properly load
correct sprite based on appearance.
'''
class SpriteLoader():
    ''' Class with different load functtions which return a dictionary with pairs
        of <str:label,cv2.mat: sprite>
    '''

    ###ENVIRONMENT LOADERS

    def loadFloor(level=0):
        level = str(level)
        floor =  cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_0_7.png")[...,::-1]
        return floor

    def loadBox(level=0):
        level = str(level)
        box = cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_0_3.png")[...,::-1]
        return box

    def loadBrick(level=0):
        level = str(level)
        box = cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_1_5.png")[...,::-1]
        return box

    def loadBlock(level=0):
        level = str(level)
        block = cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_5_3.png")[...,::-1]
        return block

    def loadClouds(level=0):
        #Convert to str to be able to concatenate it
        level = str(level)

        #Read clouds sprites
        clouds = {'[cloud_tl]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_13_3.png")[...,::-1],
                    '[cloud_bl]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_13_4.png")[...,::-1],
                    '[cloud_tm]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_3.png")[...,::-1],
                    '[cloud_bm]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_4.png")[...,::-1],
                    '[cloud_tr]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_15_3.png")[...,::-1],
                    '[cloud_br]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_15_4.png")[...,::-1]
                    }
        return clouds

    def loadBushes(level=0):

        #Convert to str to be able to concatenate it.
        level = str(level)
        #Read bushes sprites
        bushes = {'[bush_l]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_13_5.png")[...,::-1],
                        '[bush_m]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_5.png")[...,::-1],
                        '[bush_r]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_15_5.png")[...,::-1],
                        }
        return bushes

    def loadHills(level=0):

        #Convert to str to be able to concatenate it.
        level = str(level)
        #Load hill sprites
        hill     =   {'[hill_t]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_10_3.png")[...,::-1],
                       '[hill_ml]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_9_4.png")[...,::-1],
                       '[hill_mm]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_10_4.png")[...,::-1],
                       '[hill_mr]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_11_4.png")[...,::-1],
                       '[hill_bll]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_8_5.png")[...,::-1],
                       '[hill_bl]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_9_5.png")[...,::-1],
                       '[hill_bm]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_10_5.png")[...,::-1],
                       '[hill_br]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_11_5.png")[...,::-1],
                       '[hill_brr]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_12_5.png")[...,::-1]
                        }

        return hill

    def loadTrees(level=0):
        #to str for concatenation
        level = str(level)
        #load tree sprites
        tree = {'[tree_trunk]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_2.png")[...,::-1],
                '[tree_tb]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_1.png")[...,::-1],
                '[tree_tt]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_14_0.png")[...,::-1],
                '[tree_small]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_13_1.png")[...,::-1]
                }
        return tree



    def loadPipes(level=0):
        #Convert to str to be able to concatenate it.
        level = str(level)
        #Load sprites
        pipe = {'[pipe_tl]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_6_2.png")[...,::-1],
                '[pipe_tr]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_7_2.png")[...,::-1],
                '[pipe_bl]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_6_3.png")[...,::-1],
                '[pipe_br]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite"+level+"_7_3.png")[...,::-1]
            }
        return pipe

    #MARIO LOADER
    def loadMario():
        mario = {'[mario_idle]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/MarioSprites/idle.png")[...,::-1],
                      '[mario_walk1]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/MarioSprites/walk1.png")[...,::-1],
                      '[mario_walk2]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/MarioSprites/walk2.png")[...,::-1],
                      '[mario_walk3]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/MarioSprites/walk3.png")[...,::-1],
                      '[mario_jump]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/MarioSprites/jump.png")[...,::-1]
                      }
        return mario
    
    #ENEMY LOADERS
    def loadGoombas(tileset = 0 ):
        mushroom = {'[mush_1]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/mushroom_"+str(tileset)+"_0.png")[...,::-1],
                    '[mush_2]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/mushroom_"+str(tileset)+"_1.png")[...,::-1]}
        return mushroom
    
    def loadKoopa(tileset = 0):
        koopa = {'[koopa_1]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/koopa_"+str(tileset)+"_0.png")[...,::-1],
                 '[koopa_2]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/koopa_"+str(tileset)+"_1.png")[...,::-1]}
        
        return koopa

    def loadPiranha(tileset = 0):
        piranha = {'[piranha_1]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/piranha_"+str(tileset)+"_0.png")[...,::-1],
                   '[piranha_2]':cv2.imread("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/EnemySprites/piranha_"+str(tileset)+"_1.png")[...,::-1]}
        
        return piranha


    #GENERATE SEGMENTATION GT FOR SOME SPRITES

    def GenerateSSGT(object,class_color, no_label_color = [0,0,0], background_color= np.array([147,187,236])):
        #First create dictionary to hold images and keys
        ssgt = {}

        #Iterate over already loaded object
        for key, value in object.items():
                #copy the sprite
                ssgt[key] = value.copy()

                #modify the sprite
                for i in np.arange(value.shape[0]):
                    for j in np.arange(value.shape[1]):                
                        if (value[i,j,:] == background_color).all():
                            ssgt[key][i,j,:] = no_label_color
                        else:
                            ssgt[key][i,j,:] = class_color

        return ssgt
    def SpriteSSGT(frame,class_color, no_label_color = [0,0,0], background_color= np.array([147,187,236])):

        ssf = frame.copy()

        for i in np.arange(frame.shape[0]):
                    for j in np.arange(frame.shape[1]):                
                        if (frame[i,j,:] == background_color).all():
                            ssf[i,j,:] = no_label_color
                        else:
                            ssf[i,j,:] = class_color

        return ssf





class FrameGenerator():
    ''' Super mario frames have a dimension of
        256 x 240 x 3 (x,y,c)
        The idea is to place labels inside a grid
        and then fill with the corresponding image and
        its segmentation
        '''

    def __init__(self, sprite_dataset="/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/label_assignment/sprite_labels_correspondence.csv", cores=1):
        '''Initializes the frame generator'''
        # For faster generation
        self.cores = cores
        # load all sprites (routes of)
        self.sprites = pd.read_csv(sprite_dataset, sep=',')
        # available classes
        self.labels = self.sprites.Label.unique()
        # background color (for level 1)

        self.background_colors = {0: np.array(
            [147, 187, 236]), 1: np.array([0, 0, 0])}
        self.sprite_bg_color = np.array([147, 187, 236])
        # load all sprites and semantically segment them

        # Set size for the label grid, in terms of squared sprite blocks.
        # Set to 15 and 17 to do random crops on training to get horizontal "displacement"
        # to avoid "grid"-y look
        self.grid_h = 15
        self.grid_w = 17

        # Set existing classes
        self.classes = {"default": 0,
                        'floor': 1,
                        'brick': 2,
                        'box': 3,
                        'enemy': 4,
                        'mario': 5}

        # For segmented frames, set a color per class.
        self.classcolors = {"default": [0, 0, 0],
                            'floor': [0, 0, 255],
                            'brick': [127, 127, 0],
                            'box': [0, 255, 0],
                            'enemy': [255, 0, 0],
                            'mario': [255, 255, 0]}

    def SetLevelSprites(self, level):
        '''This function loads sprites and textures that will be used in the image.
            Level = 'xyz' with:
            x - Level tileset to use (not relevant for grid generation)
            y - type of level:
                - 0 means default level, with bushes and hills in the background
                - 1 means default level with trees in the background
                - 2 means underground level. No trees or bushes in the background
                - 3 means castle level (not implemented)
                - 4 means mushroom level (not implemented)
                - O means default level with alternate bushes
                - I means default level with alternate trees

            z - background color (not relevant for grid generation)
                - 0 default blue
                - 1 black 

        '''
        # Parameter is tileset to choose from.
        tileset = int(level[0])

        if level[1] == '2':
            tileset = 1

       # For background elements
        self.floor = SpriteLoader.loadFloor(tileset)
        self.box = SpriteLoader.loadBox(tileset)
        self.brick = SpriteLoader.loadBrick(tileset)
        self.pipe = SpriteLoader.loadPipes(tileset)
        self.block = SpriteLoader.loadBlock(tileset)

        if level[1] == '0' or level[1] == '1':
            tileset = 0
        elif level[1] == 'O' or level[1] == 'I':
            tileset = 3
        else:
            tileset = 0
        # For background elements
        self.hill = SpriteLoader.loadHills(tileset)
        self.clouds = SpriteLoader.loadClouds(0)
        self.bushes = SpriteLoader.loadBushes(tileset)
        self.trees = SpriteLoader.loadTrees(tileset)

        # Generate segmentation gt for level elements
        self.spipe = SpriteLoader.GenerateSSGT(
            self.pipe, self.classcolors['floor'])
        self.seg_floor = SpriteLoader.SpriteSSGT(
            self.floor, self.classcolors['floor'])
        self.sbox = SpriteLoader.SpriteSSGT(self.box, self.classcolors['box'])
        self.sbrick = SpriteLoader.SpriteSSGT(
            self.brick, self.classcolors['brick'])
        self.sblock = SpriteLoader.SpriteSSGT(
            self.brick, self.classcolors['floor'])

    def LoadSprites(self, level):
        '''Load sprites for mario, enemies and generates their ground truth.'''
        # cave sprites
        if level[1] == '2':
            tileset = 1
        else:
            tileset = 0

        # Load Enemies
        self.mushroom = SpriteLoader.loadGoombas(tileset)
        self.smushroom = SpriteLoader.GenerateSSGT(
            self.mushroom, self.classcolors['enemy'])
        self.koopa = SpriteLoader.loadKoopa(tileset)
        self.skoopa = SpriteLoader.GenerateSSGT(
            self.koopa, self.classcolors['enemy'])
        self.piranha = SpriteLoader.loadPiranha(tileset)
        self.spiranha = SpriteLoader.GenerateSSGT(
            self.piranha, self.classcolors['enemy'])

        # Load mario sprites
        self.mario = SpriteLoader.loadMario()
        self.smario = SpriteLoader.GenerateSSGT(
            self.mario, self.classcolors['mario'])

    def generate_frame(self, level='000', grid=[]):
        # Generate the grid for the level
        if grid == []:
            grid = self.generate_grid(level)

        self.SetLevelSprites(level)

        self.LoadSprites(level)

        if level[1] == '2':
            bg_color = 1
        else:
            bg_color = int(level[2])

        self.background_color = self.background_colors[bg_color]
        frame = np.zeros((16*15, 16*self.grid_w, 3))
        frame = frame + self.background_color
        sframe = np.zeros((16*15, 16*self.grid_w, 3))
        classframe = np.zeros((16*15, 16*self.grid_w))

        missing_right = False  # for piranha

        # Generate frame and semantically segmented frame
        for row in np.arange(grid.shape[0]):
            frow = row * 16  # index iterator
            for column in np.arange(grid.shape[1]):
                fcol = column*16  # index iterator
                # Print tile
                # First print background for the tile
                if grid[row, column, 0] == '[background]':
                    # Iterate over pixels of the grid
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j, :] = self.background_color
                            sframe[frow+i, fcol+j] = [0, 0, 0]
                elif grid[row, column, 0][1:6] == 'cloud':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.clouds[grid[row, column, 0]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.clouds[grid[row, column, 0]][i, j]
                elif grid[row, column, 0][1:5] == 'hill':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.hill[grid[row, column, 0]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.hill[grid[row, column, 0]][i, j]
                else:
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j] = [255, 255, 255]
                            sframe[frow+i, fcol+j] = [255, 255, 255]

                # Paint "mid depth"
                # paint floor
                if grid[row, column, 1] == '[floor]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j, :] = self.floor[i, j]
                            sframe[frow+i, fcol+j, :] = self.seg_floor[i, j]
                            classframe[frow+i, fcol+j] = self.classes['floor']
                elif grid[row, column, 1] == '[box]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.box[i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.box[i, j]
                                sframe[frow+i, fcol+j, :] = self.sbox[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['box']

                elif grid[row, column, 1] == '[brick]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.brick[i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.brick[i, j]
                                sframe[frow+i, fcol+j, :] = self.sbrick[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['brick']

                elif grid[row, column, 1][1:5] == 'bush':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.bushes[grid[row, column, 1]][i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.bushes[grid[row, column, 1]][i, j]

                elif grid[row, column, 1][1:5] == 'tree':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.trees[grid[row, column, 1]][i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.trees[grid[row, column, 1]][i, j]

                # Print characters
                if grid[row, column, 2][1:5] == 'mush':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.mushroom[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.mushroom[grid[row, column, 2]][i, j]
                                sframe[frow+i, fcol+j,
                                       :] = self.smushroom[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['enemy']

                if grid[row, column, 2][1:6] == 'koopa':
                    for i in np.arange(32):
                        for j in np.arange(16):
                            row_off = frow+i-16
                            # Only print non background pixels
                            if (self.koopa[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[row_off, fcol+j,
                                      :] = self.koopa[grid[row, column, 2]][i, j]
                                sframe[row_off, fcol+j,
                                       :] = self.skoopa[grid[row, column, 2]][i, j]
                                classframe[row_off, fcol +
                                           j] = self.classes['enemy']

                if grid[row, column, 2][1:8] == 'piranha':
                    # Columns go from 0 to x, increasing, so first reaches left side of a "piranha cell"
                    # Special case is for column 0 where it could be a right side so it should offset left.
                    if missing_right == False:
                        y = np.random.randint(1, 21)

                        piranha_height = np.random.choice([0, y])

                    if grid[row+1, column, 2] == '[pipe_tl]':  # then only print half piranha

                        for i in np.arange(32):
                            for j in np.arange(8):
                                row_off = frow+i-16 + piranha_height
                                col_off = fcol+j + 8
                                if (self.piranha[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                    frame[row_off, col_off,
                                          :] = self.piranha[grid[row, column, 2]][i, j]
                                    sframe[row_off, col_off,
                                           :] = self.spiranha[grid[row, column, 2]][i, j]
                                    classframe[row_off,
                                               col_off] = self.classes['enemy']
                        missing_right = True

                    if grid[row+1, column, 2] == '[pipe_tr]':  # then only print half piranha

                        for i in np.arange(32):
                            for j in np.arange(8):
                                row_off = frow+i-16 + piranha_height
                                col_off = fcol+j

                                if (self.piranha[grid[row, column, 2]][i, j+8] != self.sprite_bg_color).any():
                                    frame[row_off, col_off,
                                          :] = self.piranha[grid[row, column, 2]][i, j+8]
                                    sframe[row_off, col_off,
                                           :] = self.spiranha[grid[row, column, 2]][i, j+8]
                                    classframe[row_off,
                                               col_off] = self.classes['enemy']
                        missing_right = False

                if grid[row, column, 2][1:5] == 'pipe':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.pipe[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.pipe[grid[row, column, 2]][i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j,
                                       :] = self.spipe[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['floor']

                if grid[row, column, 2][1:6] == 'block':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.block[i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.block[i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j, :] = self.sblock[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['floor']

                # Print characters
                if grid[row, column, 2][1:6] == 'mario':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # print("Color",self.floor[i,j])
                            # Only print non background pixels
                            if (self.mario[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.mario[grid[row, column, 2]][i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j,
                                       :] = self.smario[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['mario']

        return frame, sframe, classframe

    def generate_grid(self, level):

        grid_gen = GridGenerator()

        return grid_gen.GenerateGrid(level)

    def GenerateSamples(self, init_filenumber, end_filenumber, seed, w_tqdm=False):
        # This function generates frames and their label and semantic segmentation ground truths.
        np.random.seed(seed)

        files = None
        if w_tqdm == True:
            files = tqdm(np.arange(init_filenumber, end_filenumber))
        else:
            files = np.arange(init_filenumber, end_filenumber)
        for i in files:
            x = '0'  # np.random.choice(['0','1'])
            y = np.random.choice(['0', '1', '2', 'O', 'I'])
            z = np.random.choice(['0', '1'], p=[.8, .2])
            level = x + y + z
            #level = np.random.choice([0,1])
            frame, sframe, classframe = self.generate_frame(level)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/PNG/%d.png" % (i), frame)
            sframe = cv2.cvtColor(sframe.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Segmentation/%d.png" % (i), sframe)
            # labels = framegen.GenerateLabelImageFromSegmentation(sframe) #Esto habra que cambiarlo para que segun genere la segmentacion lo haga
            cv2.imwrite("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Labels/%s.png" % (i), classframe)
        # image_list.write(str(filename))

    def GenerateDataset(self, samples):
        '''Generates a dataset of a given size.'''
        start = time.time()
        # gets number of available threads
        threads = self.cores
        # generates different random seeds for each thread to avoid repetitions in the dataset
        seeds = np.random.randint(100, size=threads)

        #creates the folder (removes if previously exists)
        dir = '/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset'
        if os.path.exists(dir):
            # shutil.rmtree(dir)
            os.system("sudo rm -rf /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset")
        # os.makedirs(dir)
        os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/")
        
        #and subfolders
        # os.makedirs('/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/PNG/')
        # os.makedirs('/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Segmentation/')
        # os.makedirs('/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Labels/')
        os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/PNG/")
        os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Segmentation/")
        os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/Labels/")
        
        os.system("sudo chown -R 1000:1000 /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/dataset/")
        
        # If only uses one core, execute the function once
        if self.cores == 1:
            level = np.random.choice([0, 1])
            self.GenerateSamples(0, samples, level, w_tqdm=True)
        else:
            # otherwise, distribute amount of samples between threads.
            step = samples//self.cores
            ppool = multiprocessing.Pool(threads)
            ranges = step*np.arange(self.cores+1)
            print("ranges: ", ranges)
            ppool.starmap(self.GenerateSamples, zip(
                ranges[:-1], ranges[1:], seeds))

        # The above fails so trying this for now:
        # level = np.random.choice([0, 1])
        # self.GenerateSamples(0, samples, level, w_tqdm=True)

        end = time.time()

        print("Elapsed time:", end-start)




class TrainingUtils():
    ''' 
    '''

    def __init__(self, ):
        '''Initializes utils'''
        # 




## Step 1: 

## Set up Hyperparameters and enable GPU acceleration


## Create arguments object
args = Configuration()

device = 'cuda'


# Set random seed for reproducibility
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(args.seed)  # CPU seed
torch.cuda.manual_seed_all(args.seed)  # GPU seed
random.seed(args.seed)  # python seed for image transformation
np.random.seed(args.seed)

print("done")



## Step 2: Training Epoch

## per pixel cross-entropy loss is to be computed

def train_epoch(args, model, device, train_loader, optimizer, epoch):
    # switch to train mode
    model.train()

    train_loss = []
    counter = 1

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    gts_all, predictions_all = [], []

    for batch_idx, (images, mask) in enumerate(train_loader):

        images, mask = images.to(device), mask.to(device)

        outputs = model(images)['out']
 
        #Aggregated per-pixel loss
        loss = criterion(outputs, mask.squeeze(1))
        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, int(counter * len(images)), len(train_loader.dataset),
                100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    
    return sum(train_loss) / len(train_loss) # per batch averaged loss for the current epoch.



## Step 3: Validation Epoch

## Per pixel cross entropy loss 

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def testing(args, model, device, test_loader):

    model.eval()

    loss_per_batch = []
    test_loss = 0

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    gts_all, predictions_all = [], []
    with torch.no_grad():
        for batch_idx, (images, mask) in enumerate(test_loader):

            images, mask = images.to(device), mask.to(device)

            outputs = model(images)['out']

            loss = criterion(outputs,mask.squeeze(1))
            loss_per_batch.append(loss.item())

            # Adapt output size for histogram calculation.
            preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts_all.append(mask.data.squeeze(0).cpu().numpy())
            predictions_all.append(preds)

    loss_per_epoch = [np.average(loss_per_batch)]

    hist = np.zeros((args.num_classes, args.num_classes))
    for lp, lt in zip(predictions_all, gts_all):
        hist += _fast_hist(lp.flatten(), lt.flatten(), args.num_classes)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    plt.figure()
    plt.bar(np.arange(args.num_classes), iou)
    plt.title('Class Accuracy in the validation set ')
    plt.show()

    mean_iou = np.nanmean(iou)

    print('\nTest set ({:.0f}): Average loss: {:.4f}, mIoU: {:.4f}\n'.format(
        len(test_loader.dataset), loss_per_epoch[-1], mean_iou))

    return (loss_per_epoch, mean_iou)


## Step 4: 

## Populate sprites


# First, run  It will populate the Sprites folder.
# python extract_subimages.py
# !python3 /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/extract_subimages.py

# Now, run:
# python generate_frames.py -s NUMBER_OF_SAMPLES_TO_GENERATE

# If you are running on linux, you can use multiple cores to make the generation faster. Instead, run the command:
# python generate_frames.py -s NUMBER_OF_SAMPLES_TO_GENERATE -c NUMBER_OF_CORES


## Step 5: Extract sprites


'''
This script is used to extract the single sprites included in the tileset.png in the same folder.
'''
def ExtractTiles(path):
    # Load the tileset
    tileset = cv2.imread(path)[..., ::-1]

    y, x, z = tileset.shape  # y,x,z

    # the image includes 5 levels, from which only 4 are from the original super mario
    # (Overworld, Underwold,Underwater and Castle)

    # Iterate over each level region and extract sprites.
    # Each region has 9 x 16 (hxw) sprites, with the 6 in the bottom right corner being 2x as high.

    # Width and height for each level set of tiles
    lvl_wide = x//3
    lvl_height = y//2

    # size for each tile
    grid_size = (16, 16)  # y,x

    # Extract all sprites for all valid levels
    #(Overworld, Underwold,Underwater and Castle)
    for level in np.arange(4):
        # offsets for each level
        x_offset_lvl = 1 + (lvl_wide+1)*(level % 3)
        y_offset_lvl = 12 + (level//3)*(37+136)

        # extract per row
        for y_i in np.arange(136//grid_size[0]):
            # extract per column
            y_offset = (grid_size[0]+1)*y_i + y_offset_lvl
            for x_i in np.arange((x//3)//grid_size[1]-1):
                x_offset = (grid_size[1]+1)*x_i + x_offset_lvl

                # if row is 6 and column is 10 or bigger, skip as those are 2x height sprites
                # probably resized on input tho
                if (y_i == 6 and x_i > 9):
                    continue
                # Get the 2x height sprites
                elif (y_i == 7 and x_i > 9):
                    sprite = tileset[y_offset-grid_size[0]-1:y_offset +
                                    grid_size[0]-1, x_offset:x_offset+grid_size[0], :]

                else:
                    sprite = tileset[y_offset:y_offset+grid_size[0],
                                    x_offset:x_offset+grid_size[0], :]

                sprite_write = cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR)
                cv2.imwrite("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites/Sprite%d_%d_%d.png" %
                            (level, x_i, y_i), sprite_write)

#########################################################################

path_tileset = "/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/tilesets/tileset.png"

sprites_dir = '/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites'
if os.path.exists(sprites_dir):
    os.system("sudo rm -rf /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites")
# os.makedirs(sprites_dir)
os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites")
os.system("sudo chown -R 1000:1000 /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/Sprites")

# Cut sprites from tileset
print(path_tileset)
ExtractTiles(path_tileset)


## Step 6: Frame generator

cores = 1 # numbers greater than 1 don't seem to work
samples = 1000 # how many should this be???? 1000s???

# Generate the dataset
framegen = FrameGenerator(cores=cores)
framegen.GenerateDataset(samples)


## Step 7: data loader definition

workers = 2 #Anything over 0 will crash on windows. On linux it works fine.

trainset = MarioDataset(args, 'train')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=True)

testset = MarioDataset(args, 'val')
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

print("done")


## Step 8: define model and download pretrained weights

device = 'cuda'
model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
model.classifier = DeepLabHead(2048, 6)
model = model.to(device)


## Step 9: define the optimizer and scheduler

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=0.1)


## Step 10: training loop for semantic segmentation

loss_train_epoch = []
loss_test_epoch = []
acc_train_per_epoch = []
acc_test_per_epoch = []
new_labels = []

cont = 0

os.system("sudo mkdir /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/models/")
os.system("sudo chown -R 1000:1000 /Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/models/")

for epoch in range(1, args.epoch + 1):
    st = time.time()
    
    print("DeepLabV3_Resnet50 training, epoch " + str(epoch))
    loss_per_epoch = train_epoch(args,model,device,train_loader,optimizer,scheduler)

    loss_train_epoch += [loss_per_epoch]

    scheduler.step()

    loss_per_epoch_test, acc_val_per_epoch_i = testing(args,model,device,test_loader)

    loss_test_epoch += loss_per_epoch_test
    acc_test_per_epoch += [acc_val_per_epoch_i]

    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i
        
    else:
        if acc_val_per_epoch_i > best_acc_val:
            best_acc_val = acc_val_per_epoch_i

    if epoch==args.epoch:
        torch.save(model.state_dict(), "/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/models/"+args.model_file_name)

    

    cont += 1


## Step 11: accuracy and loss curves

#Accuracy
acc_test  = np.asarray(acc_test_per_epoch)

#Loss per epoch
loss_test  = np.asarray(loss_test_epoch)
loss_train = np.asarray(loss_train_epoch)

numEpochs = len(acc_test)
epochs = range(numEpochs)

plt.figure(1)
plt.plot(epochs, acc_test, label='Test Semantic, max acc: ' + str(np.max(acc_test)))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.figure(2)
plt.plot(epochs, loss_test, label='Test Semantic, min loss: ' + str(np.min(loss_test)))
plt.plot(epochs, loss_train, label='Train Semantic, min loss: ' + str(np.min(loss_train)))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.show()


## Step 12: load and test model

model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)
# Added a Sigmoid activation after the last convolution layer
model.classifier = DeepLabHead(2048, 6)

# model.load_state_dict(torch.load("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/models/resnet_50.pth"))
model.load_state_dict(torch.load("/Semantic-Segmentation-Boost-Reinforcement-Learning/dataset_generator/models/MarioSegmentationModel.pth"))
model.cuda()


## Step 13: Helpers???

# Define the helper function
def decode_segmap(image, nc=21):
  ## Color palette for visualization of the 21 classes
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (0, 0,255), (127, 127, 0), (0, 255, 0), (255, 0, 0), (255, 255, 0),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(net, path, show_orig=True,transform=transforms.ToTensor(), dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  
  input_image = transform(img).unsqueeze(0).to(dev)
  out = net(input_image)['out'][0]
  
  segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  segm_rgb = decode_segmap(segm)
  plt.imshow(segm_rgb)
  plt.axis('off')
  #plt.savefig('1_1.png', format='png',dpi=300,bbox_inches = "tight")
  plt.show()

def compare(net,net2, path, show_orig=True,transform=transforms.ToTensor(), dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  
  input_image = transform(img).unsqueeze(0).to(dev)
  out = net(input_image)['out'][0]
  
  segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  segm_rgb = decode_segmap(segm)
  plt.imshow(segm_rgb)
  plt.axis('off'); plt.show()

  input_image = transform(img).unsqueeze(0).to(dev)
  out = net2(input_image)['out'][0]
  
  segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  segm_rgb = decode_segmap(segm)
  plt.imshow(segm_rgb); plt.axis('off'); plt.show()



## Step 14: eval

model.eval() #Or batch normalization gives error

# frame = "/Semantic-Segmentation-Boost-Reinforcement-Learning/Semantic segmentation/real_frames/1_1/4.png"
# frame = "/Semantic-Segmentation-Boost-Reinforcement-Learning/Semantic segmentation/real_frames/1_2/4.png"
# frame = "/Semantic-Segmentation-Boost-Reinforcement-Learning/Semantic segmentation/real_frames/6_2/4.png"
# frame = "/Semantic-Segmentation-Boost-Reinforcement-Learning/Semantic segmentation/real_frames/4_1/4.png"
# frame = "/Semantic-Segmentation-Boost-Reinforcement-Learning/Semantic segmentation/real_frames/6_1/4.png"

print(frame)
segment(model,frame)




## TODO:


# Move functions:
# ExtractTiles(), decode_segmap(), segment(), compare()
# Into class:
# TrainingUtils