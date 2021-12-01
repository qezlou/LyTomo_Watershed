# codes to make a 3D map from the stored coordinates of DM progenitores
# 1. Get the DM progenitor positions with progenitor_parallel.py (the mpi_submi file for submission)
# 2. Run the MPI method here : organize_prog_map() to organize the data
# 3. Run make_full_mesh() to make a sinlgle file containing the 3D map of the progenitors

import glob
import os
from nbodykit.lab import HDFCatalog 
from nbodykit import CurrentMPIComm
comm = CurrentMPIComm.get()
import h5py
import numpy as np

def organize_prog_map(Nmesh=205):
    """Reads the progenitor DM coordinates have been written on file to organize the data. 'make_full_mesh()' should 
       be called after this to make the full map. This method is using MPI feature in nbodykit
    """
    with h5py.File('./PartType1/R200/temp_maps/map_PC_'+str(comm.rank)+'_prog.hdf5','w') as f:
        cat = HDFCatalog('./PartType1/R200/prog_snap*.hdf5', dataset='PartType1')
        #cat.attrs['BoxSize']=205000
        # Peculiar velocity has already been taken into acount
        mesh =  cat.to_mesh(Nmesh=205, BoxSize=205000)
        real = mesh.compute()
        L = np.arange(0, Nmesh, 1)
        f['DM'] = real[:]
        f['x'] = L[real.slices[0]]
        f['y'] = L[real.slices[1]]
        f['z'] = L[real.slices[2]]

def make_full_mesh(box=205, savefile='map_PC_prog_R200_10percent.hdf5'):
    """This uses the data already have been generated via 'organize_prog_map()' and spites the full map out """
    m = np.empty((box,box,box))
    fnames = glob.glob(os.path.join('./PartType1/R200/temp_maps/'+'map_*.hdf5'))
    for fn in fnames:
        print('file ', fn, ' started!')
        with h5py.File(fn, 'r') as f:
            x = slice(f['x'][0], f['x'][-1]+1)
            y = slice(f['y'][0], f['y'][-1]+1)
            z = slice(f['z'][0], f['z'][-1]+1)
            m[x,y,z] = np.around(f['DM'], decimals=4)
        with h5py.File(savefile,'w') as fw:
         fw['map']=m

