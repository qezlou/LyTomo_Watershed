## A parallel code to get peaks and label maps for a sequence of logmassratio parameters in minima.py
from mpi4py import MPI
import numpy as np
import time
import h5py
import argparse


def runit(lmr0):
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   tss = time.asctime()
   print('Rank =', rank, 'started!', tss, flush=True)

   import minima

   mock = np.fromfile('../h_minima/map_TNG_z2.4_n1_averageF.dat').reshape(205,205,205)
   from scipy.ndimage import gaussian_filter
   mock = gaussian_filter(mock, sigma=4, mode='wrap')
   mtrue=h5py.File('../h_minima/map_TNG_true_1.0_z2.4.hdf5','r')['map']
   from scipy.ndimage import gaussian_filter
   mtrue = gaussian_filter(mtrue, sigma=4, mode='wrap')

   lmr = (rank+1)*0.05 + lmr0

   print('lmr = ', lmr, flush=True)

   peaks_mock, lmap_mock = minima.mtomo_partition_v2(mapconv=mock, minlogmassratio=lmr)
   peaks_true, lmap_true = minima.mtomo_partition_v2(mapconv=mtrue,minlogmassratio=lmr)

   #Record on hdf5 files
   with h5py.File('./results/labeled_map_TNG_z2.4_n1_lmr'+str(lmr)[0:4].ljust(4,'0')+'.hdf5','w') as fw:
        fw['map'] = lmap_mock
   with h5py.File('./results/peaks_TNG_z2.4_n1_lmr'+str(lmr)[0:4].ljust(4,'0')+'.hdf5','w') as fw:
        for a in list(peaks_mock.keys())[0:16]:
            fw[a] = peaks_mock[a]
   with h5py.File('./results/labeled_map_TNG_true_z2.4_n1_lmr'+str(lmr)[0:4].ljust(4,'0')+'.hdf5','w') as fw:
        fw['map'] = lmap_true
   with h5py.File('./results/peaks_TNG_true_z2.4_n1_lmr'+str(lmr)[0:4].ljust(4,'0')+'.hdf5','w') as fw:
        for a in list(peaks_true.keys())[0:16]:
            fw[a] = peaks_true[a]

   tsd = time.asctime()
   print('Rank = ', rank, 'is done !', tsd, flush=True)




if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--lmr0', type=float, required=True)

   args = parser.parse_args()

   runit(lmr0 = args.lmr0)

