# This code calculates the mean cluster size of the motifs and cluster analysis is done using ClusterAnalysisModifier of OVITO.

import matplotlib
matplotlib.use('Agg')

from ovito.io import import_file
from ovito.modifiers import ClusterAnalysisModifier
import numpy as np
import os

# Parameters
m = '6'
clu = 'FCC'
cut = 2.1

# Paths
path = f'F:/p3_TCC/tcc/codes_new/tcc_shear/cluster_xyzs/sr1.5e-5/d{m}/'
savepath = f'F:/p3_TCC/tcc/codes_new/further_analysis/data/avgCs/d{m}/'
os.makedirs(savepath, exist_ok=True)

# Load data
pipeline = import_file(path + f'STS_ASM385_70_2-12_{m}_featum_y0-55.xyz.{clu}_clusts.xyz')
pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cut))

# Prepare list for frame-wise stats
frame_stats = []

# Process each frame
for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)
    cluster_sizes = data.tables['clusters']['Cluster Size'].array

    filtered = cluster_sizes[cluster_sizes >= 13]
    if len(filtered) == 0:
        continue

    mean_size = np.mean(filtered)
    std_size = np.std(filtered)
    frame_stats.append((mean_size, std_size))

# Save only the required values, one row per frame: mean \t std
summary_file_path = savepath + f'avg_cluster_size_{clu}_per_frame_MeanSD_2-12-{m}_cut={cut}.txt'

with open(summary_file_path, 'w') as f:
    for mean_val, std_val in frame_stats:
        f.write(f"{mean_val:.4f}\t{std_val:.4f}\n")

print(f"Saved frame-wise mean and std to: {summary_file_path}")
