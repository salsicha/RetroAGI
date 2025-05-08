import stumpy
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [20, 6]  # width, height
plt.rcParams['xtick.direction'] = 'out'

m = 24
c = np.random.rand(10000)
p = stumpy.stump(c, m)

dists, inde = stumpy.motifs(c, p[:, 0], max_motifs=2)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
axs[0].plot(c)
axs[0].set_ylabel('CO2 ppm', fontsize='20')

cols = ['red' , 'green', 'blue' ]

for z in range(0, inde.shape[0]):
    col = cols[z]
    start = inde[z, 0]
    stop = inde[z, 0] + m
    matches = stumpy.match(c[start:stop],c, max_distance=2.0) 
    for mt in range(matches.shape[0]):
        s = matches[mt, 1]
        st = s + m
        axs[0].plot(np.arange(s, st), c[s : st], c=col)

axs[1].plot(p[:, 0])
axs[1].set_ylabel('Matrix profile', fontsize='20')
plt.show()
