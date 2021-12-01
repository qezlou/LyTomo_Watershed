"""
   Helper function to submit parallel jobs on fake_spectra to get spectra on a grid.
   - You can directly run this with mpirun
   - Examples of slurm submit files are also provided, you can adjust the arguments."mpi_submit_grid"

"""



import fake_spectra
from fake_spectra.griddedspectra import GriddedSpectra
from fake_spectra.ratenetworkspectra import RateNetworkGas
from mpi4py import MPI
import numpy as np
import time
import argparse

def get_spec(snap, nspec, res, savefile):
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()

   tss = time.asctime()
   print('Rank =', rank, 'started!', tss, flush=True)

   base = '/lustre/scratch/mqezlou/TNG300-1/output/snapdir_0'+str(snap)+'/'
   savedir='/lustre/scratch/mqezlou/TNG300-1/postprocessing/spectra/Snap_0'+str(snap)+'/true/'

   rank_str = str(rank)
   gs = GriddedSpectra(num= snap, base= base, nspec= nspec, MPI = MPI, res= res, savedir= savedir, savefile= savefile, axis=3, 
                       gasprop=RateNetworkGas, gasprop_args={"selfshield":True, "treecool_file":"./TREECOOL_ep_2018p", "cool":"KWH","recomb":"Cen92"}, 
                       kernel='tophat')
       
   gs.get_tau("H",1,1215)
   gs.get_col_density("H",1)
   #gs.get_tau("C", 4, 1548)
   #gs.get_col_density("C",4)
   gs.save_file()

   tsd = time.asctime() 
   print('Rank = ', rank, 'is done !', tsd, flush=True)
   del gs

if __name__ == '__main__':

   parser = argparse.ArgumentParser()

   parser.add_argument('--snap', type=int, required=True, help='snap number')
   parser.add_argument('--nspec', type=int, required=True, help='Make a grid of nspec*nspec of spectra')
   parser.add_argument('--res', type=float, required=True, help='Pixel resolution in km/s')
   parser.add_argument('--savefile', type=str, required=True, help='File name for the spectra')



   args = parser.parse_args()

   get_spec(snap=args.snap, nspec= args.nspec, res= args.res, savefile= args.savefile) 
