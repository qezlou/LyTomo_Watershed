from mpi4py import MPI
import h5py
from codes import halos
import argparse

def run_it(peaksfile, lmapfile, savefile):
   
   comm = MPI.COMM_WORLD
   peaks = h5py.File(peaksfile, 'r')
   lmap = h5py.File(lmapfile, 'r')['map'][:]
   halos.highz_halos(MPI=MPI, comm=comm, peaks=peaks, lmap=lmap, savefile=savefile, z=2.4442257045541464, min_radius=2, mass_thresh=0) 


if __name__ == '__main__':
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--savefile', type=str, required=True)
   parser.add_argument('--peaksfile', type=str, required=True)
   parser.add_argument('--lmapfile', type=str, required=True)
   args = parser.parse_args()

   run_it(args.peaksfile, args.lmapfile, args.savefile)
   
