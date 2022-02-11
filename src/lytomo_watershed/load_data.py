# A set of loaders for TNG info, being used by paint_map.py
import numpy as np
import glob
import os

def load_BH():
    """ BHs at high z snap """
    return np.load('./BH/BH_TNG.npz')

def load_halos():
    """ Halos at high z  """
    fhalos = h5py.File('./halos/TNG/halos_snap30.hdf5', 'r')
    return  np.around(fhalos['Coords'][:]/1000).astype(int)
    
def load_progen_DM():
    DM_coords = np.zeros((1,3))
    fnames = glob.glob(os.path.join("./Progenitor/","PartType1_rank*.npz"))
    for fn in fnames:
        DM_coords = np.append( DM_coords, np.load(fn)['coords'].astype(int), axis=0 )
    return DM_coords
def load_progen_Gas():
    Gas_coords = np.zeros((1,3))
    fnames = glob.glob(os.path.join("./Progenitor/","PartType0_rank*.npz"))
    for fn in fnames:
        Gas_coords = np.append( Gas_coords, np.load(fn)['coords'].astype(int), axis=0 ) 
    return Gas_coords










