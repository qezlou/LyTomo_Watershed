""" An MPI script to plot 2D-slice comparison between Hydro and FGPA"""
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import plot_simulation as ps
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rank_load = int(205/size)*np.ones(shape=(size,))
remainder = 205%size
if remainder != 0 :
    rank_load[0:remainder] += 1

startz = np.sum(rank_load[0:rank])

with h5py.File('./FGPA/FGPA_DMonly_flux_z2.4.hdf5','r') as f:
    fgpa_map = f['map'][:]
with h5py.File('./spectra/maps/map_TNG_true_1.0_z2.4.hdf5','r') as ftrue :
    true_map = gaussian_filter1d(ftrue['map'][:],1, mode='wrap')
true_map = gaussian_filter(true_map,4, mode='wrap')
true_map /= np.std(true_map)
fgpa_map /= np.mean(fgpa_map)
fgpa_map -= 1
fgpa_map = gaussian_filter(fgpa_map,4, mode='wrap')
fgpa_map /= np.std(fgpa_map)

with h5py.File('./FGPA/lmap_FGPA_TNG_true_z2.4.hdf5','r') as f:
    lmap= f['map'][:]
with h5py.File('./thresh/labeled_map_TNG_true_z2.4_n1_sigma4_th2.35.hdf5','r') as f:
    lmap_hydro = f['map'][:]
peaks_hydro = h5py.File('./thresh/peaks_TNG_true_z2.4_n1_sigma4_th2.30.hdf5','r')
peaks = h5py.File('./FGPA/peaks_FGPA_TNG_true_z2.4.hdf5','r')

ind_colored = h5py.File('./FGPA/outlier_subregions_fgpa.hdf5','r')

for z in np.arange(startz, startz+rank_load[rank], 1).astype(int):
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w',
        'text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k'}):
        fig, ax = plt.subplots(1,2,figsize=(60,30))
        ps.plot_flux(fig=fig, ax=ax[0], z=z, title='Hydro', dFmap=true_map, lmap=lmap_hydro, peaks=peaks_hydro, ind_colored=(ind_colored['hydro_ids'][:]-1).astype(int))
        ps.plot_flux(fig=fig, ax=ax[1], z=z, title='FGPA', dFmap=fgpa_map, lmap=lmap, peaks=peaks, ind_colored=(ind_colored['fgpa_ids'][:]-1).astype(int))
        plt.suptitle('z = '+str(z), fontsize=50)
        plt.tight_layout()
        plt.savefig('./TNG_map/'+str(z)+'_fgpa.png')
        plt.close(fig)
