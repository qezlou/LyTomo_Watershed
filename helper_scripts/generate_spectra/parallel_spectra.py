"""
   Helper function to submit parallel jobs on fake_spectra. 
   - You can directly run this with mpirun
   - Examples of slurm submit files are also provided, you can adjust the arguments

"""

from fake_spectra.randspectra import RandSpectra
from fake_spectra.ratenetworkspectra import RateNetworkGas
from mpi4py import MPI
import numpy as np
import time
import argparse

def get_spec(i, savefile,folder ,res, numlos, snap):
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()

   tss = time.asctime()
   print('Rank =', rank, 'started!', tss, flush=True)

   numlos = numlos
   res = res
   snap_num = snap
   thresh_t = 0
   base = '/lustre/scratch/mqezlou/TNG300-1/output/snapdir_0'+str(snap_num)+'/'
   savedir='/lustre/scratch/mqezlou/TNG300-1/postprocessing/spectra/Snap_0'+str(snap_num)+folder

   rand_seed = np.loadtxt('../../../rand_seed_spectra.txt').astype(int)

   rank_str = str(rank)
   rr = RandSpectra(num = snap_num,  seed = rand_seed[i], numlos=numlos, ndla=numlos, thresh=thresh_t,
                    res=res, base= base , MPI = MPI, savedir=savedir, savefile=savefile, gasprop=RateNetworkGas, 
                    gasprop_args={"selfshield":True, "treecool_file":"./TREECOOL_ep_2018p", "cool":"KWH","recomb":"Cen92"}, 
                    kernel='tophat')
   
   #rr.get_tau("C",3,)
   rr.get_tau("H",1,1215)
   rr.get_col_density("H",1)
   rr.save_file()

   tsd = time.asctime() 
   print('Rank = ', rank, 'is done !', tsd, flush=True)
   del rr

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--i', type=int, required=True)
   parser.add_argument('--savefile', type=str, required=True)
   parser.add_argument('--folder', type=str, required=True)
   parser.add_argument('--res',  type=float, required=True)
   parser.add_argument('--numlos', type=int, required=True)
   parser.add_argument('--snap', type=int, required=True)
   
   args = parser.parse_args()

   get_spec(i=args.i, savefile=args.savefile, savedir=args.folder, res=args.res, numlos=args.numlos, snap=args.snap)


