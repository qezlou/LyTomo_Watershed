"""Find the Error in Mtomo for many mocks and save the data on an hdf5 file"""
import h5py
from LATIS.codes import mpi4py_helper
import numpy as np
from scipy.ndimage import gaussian_filter
import importlib

def get_id_max_overlap(lmap_mock, lmap_true):
    """Slightly improved compared to the one used for old plots of Mtomo vs Mtomo. In this case,
    the minima in the labeled map are not numbered in order, so we can miss some numbers in between.
    returns : A dictionary of the corresponding ids of overlapping structures, just returns those structures which have
    overlapping structures in true map"""
    minima_mock = np.unique(lmap_mock)
    minima_true = np.unique(lmap_true)
    minima_mock = np.delete(minima_mock, np.where(minima_mock==0))
    minima_true = np.delete(minima_true, np.where(minima_true==0))
    
    id_max_overlap = {'mock':np.array([]),'true':np.array([])}
    for i in minima_mock:
        indm = np.where(lmap_mock==i)
        idtrue, counts = np.unique(lmap_true[indm], return_counts=True)
        if idtrue[0] == 0:
            idtrue = np.delete(idtrue, 0)
            counts = np.delete(counts, 0)
            if counts.size== 0 :
                continue
        counts_sorted = np.sort(counts)
        # Here, If 2 sub-contours overlap identically, we pick just the one with lower id
        indt = np.where(counts == counts_sorted[-1])[0][0]
        if idtrue[indt]!=0 :
            id_max_overlap['mock'] = np.append(id_max_overlap['mock'], i)
            id_max_overlap['true'] = np.append(id_max_overlap['true'], idtrue[indt])
    id_max_overlap['true'].astype(int); id_max_overlap['mock'].astype(int)
    return id_max_overlap

def get_err_Mtomo(n, th_range, z, sigma, lc):

    err_Mtomo_th = []
    mad_th = []
    dev_all = []
    
    
    for th in np.round(th_range,2):
        #print(th)
        with h5py.File('../LyTomo_data/watersheds_z'+str(z)+'/mocks/n'+str(n)
                       +'/labeled_map_TNG_z'+str(z)+'_n'+str(n)+'_sigma4_th'
                       +str(np.around(th,2)).ljust(4,'0')+'_lc'
                       +str(np.around(lc,2)).ljust(4,'0')
                       +'.hdf5','r') as f:
            lmap_mock = f['map'][:]
        with h5py.File('../LyTomo_data/watersheds_z'+str(z)+'/noiseless/labeled_map_TNG_true_z'
                       +str(z)+'_n1_sigma4_th'+str(np.around(th,2)).ljust(4,'0')+'_lc'
                       +str(np.around(lc,2)).ljust(4,'0')+'.hdf5','r') as f :
            lmap_true = f['map'][:]
        peaks_mock = h5py.File('../LyTomo_data/watersheds_z'+str(z)+'/mocks/n'
                               +str(n)+'/peaks_TNG_z'+str(z)+'_n'+str(n)
                               +'_sigma4_th'+str(np.around(th,2)).ljust(4,'0')
                               +'_lc'+str(np.around(lc,2)).ljust(4,'0')+'.hdf5', 'r')
        peaks_true = h5py.File('../LyTomo_data/watersheds_z'+str(z)+'/noiseless/peaks_TNG_true_z'
                           +str(z)+'_n1_sigma4_th'+str(np.around(th,2)).ljust(4,'0')
                           +'_lc'+str(np.around(lc,2)).ljust(4,'0')+'.hdf5','r')
        
        id_max_overlap = get_id_max_overlap(lmap_mock, lmap_true)
        mtomo_mock = peaks_mock['mtomo'][:]
        Mtomo_mock_overlap = mtomo_mock[id_max_overlap['mock'].astype(int)-1]
        mtomo_true = peaks_true['mtomo'][:]
        Mtomo_true_overlap = mtomo_true[id_max_overlap['true'].astype(int)-1]

        
        # Statistics :
        # The non isolated peaks
        ind_mock = id_max_overlap['mock'].astype(int) - 1
        ind_true = id_max_overlap['true'].astype(int) - 1
        
        dev = mtomo_true[ind_true] - mtomo_mock[ind_mock]
        dev_all.append(dev)
        err_Mtomo_th.append(np.sqrt(np.dot(dev,dev)/np.size(dev)))
        mad_th.append(np.median(np.abs(dev)))

    return err_Mtomo_th, mad_th, dev_all


if __name__== '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sigma=4
    z=2.4
    lc=2.00
    th_range = np.around(np.arange(2.00,2.90, 0.05),2)

    
    nrange = np.arange(1,21,1)
    n_rank = mpi4py_helper.distribute_array(MPI, comm, nrange)
    
    results = np.zeros(shape=(nrange.size, th_range.size))
    for n in n_rank:
        print('n ='+str(n), flush=True)
        err_Mtomo_th, mad_th, dev_all_th  = get_err_Mtomo(n=n, th_range=th_range, z=z, sigma=sigma, lc=lc)
        results[int(n-1)] = err_Mtomo_th
    comm.Barrier()
    comm.Allreduce(MPI.IN_PLACE, results, op=MPI.SUM)

    if rank==0 :
        fw = h5py.File('../LyTomo_data/plotting_data/error_Mtomo_mocks_z'+str(z)+'_sigma'+str(sigma)+'.hdf5','w')
        fw['Mtomo_err'] = results
        


