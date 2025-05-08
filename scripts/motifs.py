

## WIP


import numpy as np
import stumpy
import matplotlib.pyplot as plt

T = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 5, 6, 7, 8, 7, 6, 5])
m = 4  # subsequence length

mp = stumpy.stump(T, m)

motif_groups = stumpy.motifs(T, mp[:, 0], max_motifs=3, k=3)

for i, group in enumerate(motif_groups):
    print(f"Motif Group {i+1}: start indices = {group}")
    for idx in group:
        plt.plot(range(idx, idx + m), T[idx:idx + m], label=f"Motif {idx}")
    plt.legend()
    plt.title(f"Motif Group {i+1}")
    plt.show()

