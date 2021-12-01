# A parallel code to get progenitor particles (of any type) for TNG snapshots. Run this code directly. The core code is progenitor_particles.py
import argparse
import h5py
from mpi4py import MPI
import numpy as np
import time
import glob
import os
import progenitor_particles


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts = time.asctime()
print('\n Rank =', rank, 'started!', ts, flush=True)

def _get_files(basepath, snap):
    """Get all files whithin each rank """
    rank = comm.Get_rank()
    size = comm.Get_size()
    snap=str(snap).rjust(3,'0')
    dir_name = os.path.join(basepath, "snapdir_"+snap)
    if not os.path.isdir(dir_name):
       raise OSError('The snapshot directory not found')
    fnames = []
    # A list of files needs to be examined
    fnames = glob.glob(os.path.join(dir_name, "snap_"+snap+"*.hdf5"))

    num_files = len(fnames)
    if rank == 0 :
        print('num_files = ', num_files)
    return fnames

    """
    else:
        files_per_rank = int(num_files/size)
        #a list of file names for each rank
        fnames_rank = fnames[rank*files_per_rank : (rank+1)*files_per_rank]
        # Some ranks get 1 more snaphot file
        remained = int(num_files - files_per_rank*size)
        print('remained files ', remained)
        if rank in range(1,remained+1):
            fnames_rank.append(fnames[files_per_rank*size + rank-1 ])
            print('Ranks with more files ', rank, fnames_rank)
        return fnames_rank
    """ 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--PartType', type=int, required=True)
    parser.add_argument('--basepath', type=str, required=True, help='The path to parent basepath for snapshots')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--coord_dir', type=str, required=True, help='A temporary directory is made within this to save the coordinates of the particles at high z')
    parser.add_argument('--savedir', type=str, required=True, help='The direcrtory to save the full density map of this cluster on')
    parser.add_argument('--cluster', type=int, default=0, required=True, help='The cluster index')
    args = parser.parse_args()
    
    fnames = _get_files(basepath=args.basepath, snap = args.snap)
    # Call the core code
    progenitor_particles.get_part_coord_parallel(MPI=MPI, cluster_ind= args.cluster, basepath = args.basepath, fnames=fnames, coord_dir= args.coord_dir, savedir=args.savedir, PartType=args.PartType)

    te = time.asctime()
    print('\n Rank =', rank, 'Ended at :', te, flush=True)
    # Make sure earlierst rank waits till all are done
    comm.Barrier()
