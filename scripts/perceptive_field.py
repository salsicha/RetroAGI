

## Class colors
# 'floor': [0, 0, 255],
# 'brick': [127, 127, 0],
# 'box': [0, 255, 0],
# 'enemy': [255, 0, 0],
# 'mario': [255, 255, 0]


import cv2
import numpy as np

class PerceptiveField():
    def __init__(self, img_seg, grid_width=20):
        self.img_seg = img_seg
        self.grid_width = grid_width

        self.x_cell_size = 20
        self.y_cell_size = 20


    def get_sprite_positions(self, img_seg, color):

        hsv_image = cv2.cvtColor(img_seg, cv2.COLOR_RGB2HSV)

        rgb_yellow = np.uint8([[color]])
        hsv_yellow = cv2.cvtColor(rgb_yellow, cv2.COLOR_RGB2HSV)[0][0]
        lower = hsv_yellow
        upper = hsv_yellow

        mask = cv2.inRange(hsv_image, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = []

        for cnt in contours:
            # area = cv2.contourArea(cnt)

            M = cv2.moments(cnt)
            cX = M['m10'] / M['m00']
            cY = M['m01'] / M['m00']

            result.append([cX, cY])

        return result


    def hash_f(self, sprite_pos, mario_pos):
        cell_x = (sprite_pos[0] - mario_pos[0]) // self.grid_width
        cell_y = (sprite_pos[1] - mario_pos[1]) // self.grid_width
        hash_e = cell_y * self.grid_width + cell_x + 300
        return hash_e

