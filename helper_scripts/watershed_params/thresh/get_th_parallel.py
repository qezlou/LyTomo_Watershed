## A parallel code to get peaks and label maps for a sequence of threshold parameters in minima.py
from mpi4py import MPI
import numpy as np
import time
import h5py
import argparse
import os

def runit(n, th0, mock_file, true_file, DM_file, z, sigma, linking_contour, periodic_bound, savedir, coeff):
    """ Uses the minima.mtomo_partiion_v2() to get the labeled map and peaks for a sequence of value for
       parameter thresh 
       Look at main function below for the description on each argument.
       Since different values for thresh take different run time, we need to break a wide domain of the parameter to smaller range values, so that
       MPI code does not break. The stable intervals I used are thresh=[2.0,2.3], [2.3,2.6], 2.6,2.9]. Therefore it is not using all CPUs within a 
       node. Otherwise, works perfectly well.

   """
    from scipy.ndimage import gaussian_filter
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    tss = time.asctime()
    print('Rank =', rank, 'started!', tss, flush=True)

    import codes.minima as minima
    if periodic_bound :
        mode = 'wrap'
    else :
        mode = 'reflect'
        print('Not periodic', flush=True)

    mock = np.fromfile(mock_file).reshape(205,205,205)
    std_map =np.std(mock)
    mock = gaussian_filter(mock, sigma=sigma, mode=mode)
    if true_file != 'None':
        mtrue=h5py.File(true_file,'r')['map']
        # We need to scale the amplitude of the noise in true map to sigma_IGM as set
        # in the coveriance matrix of the mock map
        mtrue*=std_map/np.std(mtrue)
        mtrue = gaussian_filter(mtrue, sigma=sigma, mode=mode)
    # The last factor multiplied is to correct the mean density of the simulation into rho_c
    DMconv = gaussian_filter(h5py.File(DM_file,'r')['DM/dens'][:], sigma=sigma, mode=mode)

    lc= linking_contour
    lc, th = distribute_work(rank, lc, th0, mode='easy calc')
    
    print('kappa = ', th, 'nu = ', lc, flush=True)
    for i in range(3):
        coeff[i]=float(coeff[i])
    peaks_mock, lmap_mock = minima.mtomo_partition(mapconv=mock, DMconv=DMconv, z_acc=2.4442257045541464, coeff=coeff, thresh=th, linking_contour=lc, periodic_bound=periodic_bound, minimalist=False, rank=rank)
    print('Start Working on true maps!', flush=True)
    if true_file != 'None':
        peaks_true, lmap_true = minima.mtomo_partition(mapconv=mtrue, DMconv=DMconv, z_acc=2.4442257045541464, coeff=None, thresh=th, linking_contour=lc, periodic_bound=periodic_bound, minimalist=False, rank=rank)

    #Write on hdf5 files
    fname = os.path.join(savedir,'labeled_map_TNG_z'+str(z)+'_n'+str(int(n))+'_sigma'+str(sigma)+'_th'+str(th)[1:5].ljust(4,'0')+'_lc'+str(lc)[0:4].ljust(4,'0')+'.hdf5')
    with h5py.File(fname,'w') as fw:
        fw['map'] = lmap_mock
    fname = os.path.join(savedir,'peaks_TNG_z'+str(z)+'_n'+str(int(n))+'_sigma'+str(sigma)+'_th'+str(th)[1:5].ljust(4,'0')+'_lc'+str(lc)[0:4].ljust(4,'0')+'.hdf5')
    with h5py.File(fname,'w') as fw:
        for a in list(peaks_mock.keys())[0:16]:
            fw[a] = peaks_mock[a]
    if true_file != 'None':
        fname = os.path.join(savedir,'labeled_map_TNG_true_z'+str(z)+'_n'+str(int(n))+'_sigma'+str(sigma)+'_th'+str(th)[1:5].ljust(4,'0')+'_lc'+str(lc)[0:4].ljust(4,'0')+'.hdf5')
        with h5py.File(fname,'w') as fw:
            fw['map'] = lmap_true
        fname = os.path.join(savedir,'peaks_TNG_true_z'+str(z)+'_n'+str(int(n))+'_sigma'+str(sigma)+'_th'+str(th)[1:5].ljust(4,'0')+'_lc'+str(lc)[0:4].ljust(4,'0')+'.hdf5')
        with h5py.File(fname,'w') as fw:
            for a in list(peaks_true.keys())[0:16]:
                fw[a] = peaks_true[a]

    tsd = time.asctime()
    print('Rank = ', rank, 'is done !', tsd, flush=True)
    comm.Barrier()

def distribute_work(rank, lc, th0, mode='heavy calc'):

    if mode=='heavy calc':
        th = np.round(-1*(rank*0.05 + th0), 2)
        if th >= th0+1.2:
            th = lc +0.05 +(th - (th0+1.2))
            lc += 0.05

        return lc, th
    if mode=='easy calc':
        th = np.round(-1*(rank*0.05 + th0), 2)
        return lc, th





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True, help='Mock map id')
    parser.add_argument('--th0', type=float, required=True, help='The absolute value of min Peak threshold to start with')
    parser.add_argument('--mock_file', type=str, required=True, help='The address for the mock file')
    parser.add_argument('--true_file', type=str, required=True, help='The address for the true file')
    parser.add_argument('--DM_file', type=str, required=True, help='The address for the DM density map')
    parser.add_argument('--periodic', type=int, required=True, help='Periodic boundary if non zero passed ')
    parser.add_argument('--z', type=float, required=True, help='Redshift ')
    parser.add_argument('--sigma', type=int, required=True, help='smoothing scale')
    parser.add_argument('--lc', type=float, required=True, help='linking contour')
    parser.add_argument('--savedir', type=str, required=True, help='Directory to save the results in')
    parser.add_argument('--coeff', nargs='+', required=True, help='The coefficient in rho_DM/<rho_DM> -vs- delta_F relation')


   
    args = parser.parse_args()
    if args.periodic == 0:
        args.periodic = False
    else :
        args.periodic = True

    runit(n=args.n, th0 = args.th0, mock_file= args.mock_file, true_file= args.true_file, DM_file= args.DM_file, z= args.z, sigma= args.sigma, linking_contour= args.lc, periodic_bound= args.periodic, savedir=args.savedir, coeff=args.coeff)

