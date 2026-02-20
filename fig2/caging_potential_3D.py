# This code calculates the particle level gr for a binary system
# Inputs : linked coordinates of big and small particles respectively
# Outputs: caging potential of each time frame.


import warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
import math 
import time
import pandas as pd
import numba
from numba import jit
from scipy.spatial import cKDTree
import scipy.integrate as intg
import time
start_time = time.time()

import scipy.integrate as intg
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18})
plt.rcParams.update({'font.family': 'serif'})

##########    load data    ###########
dia = 1.4
path = '/media/hdd2/ShearedColloids/sr15e-6/all_tracks/pos/STS_ASM385_70_2-12_3_T=0-23_'
x = np.loadtxt(path+'PosX.dat')/dia
y = np.loadtxt(path+'PosY.dat')/dia
z = np.loadtxt(path+'PosZ.dat')/dia
print('maxx',max(x[:,0]))
print('maxx',max(y[:,0]))
print('maxx',max(z[:,0]))
print(np.shape(x))
savepath = '/media/hdd2/P2-Entropy/final_calculations/softness/data/'
#################  Initialize variables   ##############

N = np.shape(x)[0]
Nf = np.shape(x)[1]
delr = 0.005
rho = N/(max(x[:,0])*max(y[:,0])*max(z[:,0]))
sigma = 0.02  # for sheared 3d data keep sigma 0.02
d = sigma
ds = 3*d
s = math.sqrt(2*np.pi*d*d)
maxbin = 600

start_time = time.time()
###############   g(r) algorithm   ###############

@jit(nopython=True, parallel=True, nogil=True, cache=True)
def pargr(x,y,z,t):
	N = x.shape[0]
	print(N)
	histaa = np.zeros((615,N))
	aa = np.zeros((615,N))
	for j in numba.prange(N):
		for k in numba.prange(N):
			if (j != k ):
				rxjk= x[j,t]-x[k,t]
				ryjk= y[j,t]-y[k,t]
				rzjk= z[j,t]-z[k,t]
					
				rjksq = rxjk*rxjk+ryjk*ryjk+rzjk*rzjk
				rjk= math.sqrt(rjksq)
				kbin = int(rjk/delr)
				if (kbin <= maxbin-1):
					histaa[kbin,j] = histaa[kbin,j] + 1
					
##############   Gaussian brodening for continuous field   #############
	gr = np.zeros((maxbin,N+1))
	for jj in numba.prange(N):
		for ii in numba.prange(maxbin):
			rr = ii*delr
			m1 = int((rr - ds)/delr)
			l1 = int((rr + ds)/delr)
			if ( m1 >= ds ):
				for kk in numba.prange(m1,l1+1):
					ss =   kk * delr
					a11 = (rr-ss)**2

					aa[ii,jj] = aa[ii,jj] + histaa[kk,jj]*(1/s)*np.exp(-a11/(2*d*d))
			rlower = ii*delr
			rupper = rlower +delr

			xr=(rupper+rlower)/2
			ts = (1.0/(4.0*np.pi*xr*xr*rho)) 
			gr[ii,-1] = xr
			gr[ii,jj] = (ts*aa[ii,jj])
			
	return gr
	
############   calculate   ############

softness = np.zeros((N,Nf))
avg_softness = np.zeros((N,Nf))
bulkgr = np.zeros((600))
for f in range(Nf):
	print('t=',f)
	rho = N/(max(x[:,f])*max(y[:,f])*max(z[:,f]))
	par_gr = pargr(x,y,z,f)
	bulkgr += np.mean(par_gr[:,0:N],axis=1)
	rall = par_gr[:,-1]
	#plt.plot(par_gr[:,-1],par_gr[:,10])
	#plt.plot(par_gr[:,-1],bulkgr)
	#plt.show()
	
	bulk = np.array([x[:,f], y[:,f], z[:,f]])
	bulk = bulk.T
	print(np.shape(bulk))
	bulk_df = pd.DataFrame(bulk[:,0:3],columns=['x','y','z'])
	ckdtree = cKDTree(bulk_df[['x', 'y', 'z']])
	dist_bulk, idxs = ckdtree.query(bulk_df, k= 16, distance_upper_bound= 1.5)

	############# Individual softness of Particle ###########
	# Start calculation form a lower rmin = 1.2, from where g(r) is nonzero.
	
	r_edges = par_gr[int(0.85/delr):int(1.5/delr),-1]    
	par_gr = par_gr[int(0.85/delr):int(1.5/delr), 0:N]
	s2 = np.zeros(N)

	for i in range(N):
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (4*np.pi*rho)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j]**2)
		s2[i] = intg.simpson(itg,r_edges, dx = 0.005)
	softness[:,f] = s2	
	#plt.hist(s2,bins=150)
	#plt.show()

	############# Average Local Entropy ##################
	
	av_s2 = np.zeros(N)
	ra = 1.5   # ra=2 in general, 1st minimum of gr
	for i in range(N):
		sum1 = 0
		sum_fij = 0
		for k in range(16):
			fij = (1-(dist_bulk[i,k]/ra)**6)/(1-(dist_bulk[i,k]/ra)**12)
			if (i != idxs[i,k]) & (idxs[i,k] != N) :
				sum1 = sum1 + s2[idxs[i,k]]*fij
				sum_fij = sum_fij + fij
		av_s2[i] = (sum1 + s2[i])/(sum_fij +1 )
	print("--- %s seconds ---" % (time.time() - start_time))
	avg_softness[:,f] = av_s2
	#plt.hist(av_s2,bins=150)
	#plt.show()
bulkgr = bulkgr/Nf
grall = np.vstack((rall,bulkgr)).T
np.savetxt(savepath+'bulkgr_sts_2-12_3_T=1-23.txt',grall)
np.savetxt(savepath+'phi_sts_2-12_3_T=1-23_rmin=0.85_rmax=1.5.txt',softness)
np.savetxt(savepath+'avg_phi_sts_2-12_3_T=1-23_rmin=0.85_rmax=1.5_ra=1.5.txt',avg_softness)
