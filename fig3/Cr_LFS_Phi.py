# This code calculates the pair correlation function of high caging potential particles and the lfs particles


import numpy as np
import math 
import time
import pandas as pd
import numba
from numba import jit
import scipy.integrate as intg
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': 'serif'})
import os


start_time = time.time()
@jit(nopython=True, parallel=True, nogil=True, cache=True)
def particle_gr(tag_x, tag_y, tag_z, loop_x, loop_y, loop_z, tag_n, loop_n,Lx,Ly,Lz):
    histbb = np.zeros((maxbin, tag_n))
    bb = np.zeros((maxbin, tag_n))
    for j in numba.prange(tag_n):
        for k in numba.prange(loop_n):
            #if j != k:
            rxjk = tag_x[j] - loop_x[k]
            ryjk = tag_y[j] - loop_y[k]
            rzjk = tag_z[j] - loop_z[k]

            rjksq = rxjk * rxjk + ryjk * ryjk + rzjk * rzjk
            rjk = math.sqrt(rjksq)
            kbin = int(rjk / delr)
            if (kbin <= maxbin - 1) :
                histbb[kbin, j] = histbb[kbin, j] + 1

    gr = np.zeros((maxbin, tag_n + 1))
    rho = loop_n / (Lx * Ly * Lz)
    for jj in numba.prange(tag_n):
        for ii in numba.prange(maxbin):
            rlower = ii * delr
            rupper = rlower + delr

            xr = (rupper + rlower) / 2
            ts = (1.0 / (4.0 * np.pi * xr * xr *delr* rho))
            gr[ii, jj] = ts * histbb[ii, jj]
            gr[ii, -1] = xr
    return gr



m=6
Nf = 10
cutoff = 3
clu = '10b'
path_bulk = '/media/rati/rati/p3_TCC/tcc/codes_new/sr1.5e-5/d'+str(m)+'/'
path_id = '/media/rati/rati/p3_TCC/tcc/codes_new/further_analysis/data/ids/'+str(clu)+'/d'+str(m)+'/'
path_sop = '/media/rati/rati/p3_TCC/tcc/softness/d'+str(m)+'/sop/'
path_dmin = '/media/rati/rati/p3_TCC/tcc/y55_dmin/dmin_data/all/bulk/d'+str(m)+'/'

path_link = '/media/rati/rati/p3_TCC/tcc/xyzLinked/dec/bulk/d'+str(m)+'/'

bulk_sop = []
clust_sop = []
c_sop=[]


for t in range(Nf):
    print(t)
    bulk = np.loadtxt(path_bulk+'STS_ASM385_70_2-12_'+str(m)+'_featum_y0-55_t='+str(t)+'.txt')
    idt = np.loadtxt(path_id+'idx_cutoff3_'+str(clu)+'_d'+str(m)+'_t='+str(t)+'.txt',dtype = 'int')
    idc = np.loadtxt(path_id+'idx_central_cutoff3_'+str(clu)+'_d'+str(m)+'_t='+str(t)+'.txt',dtype = 'int')
    
    # remove the boundary particles
    
    min_x, max_x = np.min(bulk[:, 0]), np.max(bulk[:, 0])
    min_y, max_y = np.min(bulk[:, 1]), np.max(bulk[:, 1])
    min_z, max_z = np.min(bulk[:, 2]), np.max(bulk[:, 2])
    
    x_bounds = (min_x + cutoff, max_x - cutoff)
    y_bounds = (min_y + cutoff, max_y - cutoff)
    z_bounds = (min_z + cutoff, max_z - cutoff)
    
    within_bounds = (
        (bulk[:, 0] >= x_bounds[0]) & (bulk[:, 0] <= x_bounds[1]) &
        (bulk[:, 1] >= y_bounds[0]) & (bulk[:, 1] <= y_bounds[1]) &
        (bulk[:, 2] >= z_bounds[0]) & (bulk[:, 2] <= z_bounds[1])
    )
    print(np.shape(bulk))
    
    # load the order paramters
    x = load('STS_ASM385_70_2-12_6_featumd_y0-55_t='+str(t)+'-'+str(t+0)+'_PosX.dat')
    y = load('STS_ASM385_70_2-12_6_featumd_y0-55_t='+str(t)+'-'+str(t+0)+'_PosY.dat')
    z = load('STS_ASM385_70_2-12_6_featumd_y0-55_t='+str(t)+'-'+str(t+0)+'_PosZ.dat')
    xyz = np.vstack((x[:,0],y[:,0],z[:,0])).T
    print(np.shape(xyz))
    dmin = np.loadtxt(path_dmin+'Dmin_bulk_Na=1.15_ra=1.5_2-12-6_t='+str(t)+'-'+str(t+5)+'t0-4_aveps.txt')
    
    print(np.shape(dmin))

    clust = bulk[idt,:]
    clust_c = bulk[idc,:]
      
    msop = np.mean(bulk_sop)
    print(np.shape(bulk_sop))
    
    high = np.where(bulk_sop>1.46*msop) # 1.1853
    low = np.where(bulk_sop<0.58*msop)  # 0.814
    
    print(high)
    print(np.shape(low))
    
    fracH = len(high[0])/len(bulk_s)
    fracL = len(low[0])/len(bulk_s)
    print('fracH = ',fracH)
    print('fracL = ',fracL)
    xyzH = bulk_s[high[0],0:3]
    xyzL = bulk_s[low[0],0:3]
    
    xyz = clust_s[:,0:3]
    xyzc = s_c[:,0:3]
    
    maxbin = 10
    delr = 1
    Lx = np.max([np.max(xyzc[:, 0]), np.max(xyzH[:, 0]), np.max(xyzL[:, 0])])-np.min([np.min(xyzc[:, 0]), np.min(xyzH[:, 0]), np.min(xyzL[:, 0])])
    Ly = np.max([np.max(xyzc[:, 1]), np.max(xyzH[:, 1]), np.max(xyzL[:, 1])])-np.min([np.min(xyzc[:, 1]), np.min(xyzH[:, 1]), np.min(xyzL[:, 1])])
    Lz = np.max([np.max(xyzc[:, 2]), np.max(xyzH[:, 2]), np.max(xyzL[:, 2])])-np.min([np.min(xyzc[:, 2]), np.min(xyzH[:, 2]), np.min(xyzL[:, 2])])
    print('high bulk=',len(xyzH[:,0]))
    print('low bulk=',len(xyzL[:,0]))
    print('N loop',len(xyz[:,0]))
    #print('Lz=',Lz)
    grH = np.zeros((maxbin, len(xyz[:,0]) + 1))
    grH = particle_gr(xyzH[:,0], xyzH[:,1], xyzH[:,2], xyzc[:,0], xyzc[:,1], xyzc[:,2], len(xyzH[:,0]), len(xyzc[:,0]),Lx,Ly,Lz)
    grL = particle_gr(xyzL[:,0], xyzL[:,1], xyzL[:,2], xyzc[:,0], xyzc[:,1], xyzc[:,2], len(xyzL[:,0]), len(xyzc[:,0]),Lx,Ly,Lz)


r = grH[:,-1]
bulkgrH =  np.delete(grH, -1, axis=1) 
bulkgrH =  np.mean(bulkgrH,axis=1).reshape(-1, 1)
bulkgrL =  np.delete(grL, -1, axis=1) 
bulkgrL = np.mean(bulkgrL,axis=1).reshape(-1, 1)


spath = '/media/rati/rati/p3_TCC/paper2/final/draft1/fig3/gr/'
ldata = np.vstack((r,bulkgrL[:,0])).T
print(np.shape(ldata))

hdata = np.vstack((r,bulkgrH[:,0])).T
print(np.shape(hdata))
#np.savetxt(spath+str(clu)+'_low_l=0_10per_central_dr=1.txt',ldata)
#np.savetxt(spath+str(clu)+'_high_l=0_10per_central_dr=1.txt',hdata)

plt.plot(r,bulkgrH,'*-',label='high')
plt.plot(r,bulkgrL,'o-',label='low')
#plt.axhline(y=1, color='r', linestyle='--', label='y=10')
#plt.title('10b l=1 10% central particles dr=1')
plt.xlabel('$r$')
plt.ylabel('$g(r)$')
plt.legend()
plt.show()
