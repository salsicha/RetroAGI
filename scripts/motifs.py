import stumpy
import csv
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda


class Motifs():
    def __init__(self, max_motifs=2, window_size=24, max_distance=20.0):
        self.max_motifs = max_motifs
        self.window_size = window_size
        self.max_distance = max_distance

    def find_motifs(self, data):

        self.p = stumpy.stump(data, self.window_size)

        all_gpu_devices = [device.id for device in cuda.list_devices()]
        p = stumpy.gpu_stump(data, self.window_size, device_id=all_gpu_devices)

        distances, indices = stumpy.motifs(data, self.p[:, 0], max_motifs=self.max_motifs)

        match_count = []
        match_dict = {}

        for z in range(0, indices.shape[0]):
            start = indices[z, 0]
            stop = indices[z, 0] + self.window_size
            matches = stumpy.match(data[start:stop], data, max_distance = self.max_distance)
            match_count.append(matches.shape[0])
            match_dict[z] = matches

        most_frequent_motif = np.argmax(match_count)

        start = indices[most_frequent_motif, 0]
        stop = indices[most_frequent_motif, 0] + self.window_size

        # Return most frequent motif
        return data[start:stop], match_dict[most_frequent_motif], p
    