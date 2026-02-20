#This code calculates the length of the trajecotries of individual particles during shear. You get the fraction of times a particle is present in a particular LFS

import numpy as np
import os
import pandas as pd
from matplotlib.pyplot import *
from scipy.spatial import KDTree


# Read the TCC output file containing the positions of clusters and convert it to a numpy array
def load_xyz_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    frames_xyz = []
    i = 0
    while i < len(lines):
        num_particles = int(lines[i].strip())
        i += 2  # Skip the first two rows having nop and frame number
        
        frame_xyz = []
        for j in range(num_particles):
            columns = lines[i + j].strip().split()
            x = float(columns[1])
            y = float(columns[2])
            z = float(columns[3])
            frame_xyz.append([x, y, z])
        
        frames_xyz.append(np.array(frame_xyz))
        i += num_particles
    return frames_xyz


def find_matching_ids(xyz_frame, ref_frame, tolerance=1e-2):
    match = np.zeros(len(ref_frame), dtype=int)

    for xyz in xyz_frame:
        diff = np.abs(ref_frame - xyz)
        within_tolerance = np.all(diff <= tolerance, axis=1)
        
        match[within_tolerance] = 1

    return match


# Paths and parameters
m = 6
clu = 'FCC'

xyz_file_path = '/media/rati/One Touch1/p3_TCC/tcc/codes_new/tcc_shear/cluster_xyzs/sr1.5e-5/d'+str(m)+'/STS_ASM385_70_2-12_'+str(m)+'_featum_y0-55.xyz.'+str(clu)+'_clusts.xyz'


dat_path = '/media/rati/One Touch1/p3_TCC/tcc/xyzLinked/dec/bulk/d'+str(m)+'/STS_ASM385_70_2-12_'+str(m)+'_featumd_y0-55_t='


savepath = '/media/rati/One Touch1/p3_TCC/tcc/codes_new/further_analysis/data/id_dmin/'+str(clu)+'/d'+str(m)+'/'

nf=14

for t in range(nf-4):
	print('t=',t)
	frames_xyz = load_xyz_data(xyz_file_path)

	x_dat = np.loadtxt(dat_path+str(t)+'-'+str(t+4)+'_PosX.dat')
	y_dat = np.loadtxt(dat_path+str(t)+'-'+str(t+4)+'_PosY.dat')
	z_dat = np.loadtxt(dat_path+str(t)+'-'+str(t+4)+'_PosZ.dat')

	Np, frames = x_dat.shape

	traj = np.zeros((Np,frames))

	# Iterate through each frame and process
	for i in range(t, t + 5):
		xyz_frame = frames_xyz[i]
		print(f"Processing frame {i}...")
		#print('xyz_frame=', xyz_frame)
  
		txt_link = np.vstack((x_dat[:, i-t], y_dat[:, i-t], z_dat[:, i-t])).T
		#print(np.shape(txt_link))
		
		matching = find_matching_ids(xyz_frame, txt_link, tolerance=1e-2)
		
		traj[:,i-t] = matching
	np.savetxt(savepath + f'traj_in_'+str(clu)+'_d'+str(m)+str(t)+'-'+str(t+4)+'.txt', traj, fmt='%d')
    
