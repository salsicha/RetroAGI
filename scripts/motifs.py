import stumpy
import csv
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda


class Motifs():
    def __init__(self):
        self.max_motifs = 2
        self.m = 24

        # fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        # plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
        # axs[0].plot(c)
        # axs[0].set_ylabel('CO2 ppm', fontsize='20')

        # self.cols = ['red' , 'green', 'blue' ]

    def find_motifs(self, data):

        # self.c = np.random.rand(10000)
        self.p = stumpy.stump(data, self.m)

        all_gpu_devices = [device.id for device in cuda.list_devices()]
        p = stumpy.gpu_stump(data, self.m, device_id=all_gpu_devices)

        self.dists, self.inde = stumpy.motifs(data, self.p[:, 0], max_motifs=self.max_motifs)

        match_count = {}
        i = 0

        for z in range(0, self.inde.shape[0]):
            # col = self.cols[z]
            start = self.inde[z, 0]
            stop = self.inde[z, 0] + self.m
            matches = stumpy.match(data[start:stop], data, max_distance = 200.0)
            match_count[i] = matches.shape[0]
            i += 1

            # for mt in range(matches.shape[0]):
            #     s = matches[mt, 1]
            #     st = s + self.m

                # axs[0].plot(np.arange(s, st), c[s : st], c=col)

        # axs[1].plot(p[:, 0])
        # axs[1].set_ylabel('Matrix profile', fontsize='20')
        # plt.show()

        sorted_matches = sorted(match_count.items(), key=lambda x: x[1], reverse=True)
        most_frequent_motifs = sorted_matches[:self.max_motifs]
        most_frequent_motifs = {k: v for k, v in most_frequent_motifs}

        print(most_frequent_motifs)

        z = sorted_matches[0]
        start = self.inde[z, 0]
        stop = self.inde[z, 0] + self.m

        # Return most frequent motif
        return data[start:stop]
    