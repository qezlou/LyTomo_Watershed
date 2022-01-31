"""Generating spectra with Fluctuating Gunn Peterson Approximation (FGPA)

- The code is MPI working on Illustris and MP-Gadget snapshots. The packages needed are :

    - astropy 
    - fake_spectra

To get the FGPA spectra, follow the steps below :

1. From codes.get_density_field use TNG() or MP-Gadget() to construct DM density
   field on a grid with desired size. For FGPA, the grid cells should on avergae 
   have 1 particle per cell.
  
2. Save the results from the previous step in savedir directory. The density 
   has been saved on several files depending on number of ranks used. 

3. Run get_noiseless_map() or get_sample_spectra() functions here with the same
   number of MPI ranks and pass the directories for the density field above as 
   savedir argument. 

4. The output is a single hdf5 file containing either the full true flux map or the
   random spectra sample. Note : In case your desired map is too large to fit on your 
   memory, modify the last lines of the functions to store results of each rank separately. 

"""

import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import gaussian_filter1d
import fake_spectra.fluxstatistics as fs
from . import spectra_mocking as sm


def get_sample_spectra(MPI, z, num, seed=13, savedir='density_highres/', 
                       savefile='spectra_z2.4_FGPA_n1.hdf5',boxsize=205,
                       Ngrids=205, Npix=205, SmLD=1, SmLV=1):
    """Get a sample of spectra to be used for mock map reconstruction
    z : redshift
    seed : rand seed to get x and y coordinates of the sample spectra 
    savedir :  the directory containing the density map
    savefile : The name for hdf5 file to save final map on
    Ngrids : int, the size of the x and y dimensions of the desired map
    Npix : number of desired pxiels along final map
    """
    # Initialize the MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tau_conv, xrank, yrank = get_tau_conv(comm=comm, z=z, savedir=savedir,
                                          boxsize=boxsize, Ngrids=Ngrids,
                                          SmLD=SmLD, SmLV=SmLV)
    
    if seed is not None:
        cofm =get_cofm(seed=seed, num=num, boxsize= tau_conv.shape[0]).astype(int)
    else:
        x, y = np.meshgrid(np.arange(Ngrids), np.arange(Ngrids))
        cofm = np.zeros(shape=(Ngrids*Ngrids,2), dtype=int)
        cofm[:,0] = np.ravel(x)
        cofm[:,1] = np.ravel(y)
        del x, y
    
    tau_sampled = np.zeros(shape=(cofm.shape[0], tau_conv.shape[2]))
    xrank, yrank = np.arange(xrank[0], xrank[1]+1,1), np.arange(yrank[0], yrank[1]+1,1)
    # Find the sample spectra on this rank
    ind_cofm = np.where(np.isin(cofm[:,0], xrank)*np.isin(cofm[:,1], yrank))[0]
    tau_sampled[ind_cofm] = tau_conv[cofm[ind_cofm,0], cofm[ind_cofm,1],:]
    ### MPI part
    # Make sure the data is contiguous in memeory
    tau_sampled = np.ascontiguousarray(tau_sampled, np.float64)
    # Add the results from all ranks
    comm.Allreduce(MPI.IN_PLACE, tau_sampled, op=MPI.SUM)
    # Scale the tau to get mean flux right
    scale = fs.mean_flux(tau_sampled, mean_flux_desired=sm.get_mean_flux(z=z))
    tau_sampled *= scale
    if rank==0 :
        print('Scaling tau with :', scale)
    # Change cofm to kpc/h to record on spctra
    cofm  = cofm.astype(float)*(boxsize*1000/tau_conv.shape[0])

    if rank == 0 :
        with h5py.File(savedir+savefile,'w') as fw:
            # We need all Header info to load the file with fake_spectra
            # Some attrs are copied from hydro spectra, a more stable way
            # should be implemented
            fw.create_group('Header')
            fw['Header'].attrs.create('redshift', z)
            fw['Header'].attrs.create('box', boxsize*1000)
            fw['Header'].attrs.create('discarded', 0)
            fw['Header'].attrs.create('hubble', 0.6774)
            fw['Header'].attrs.create('nbins', tau_sampled.shape[1])
            fw['Header'].attrs.create('npart', np.array([0, 15625000000, 0, 0, 0, 0]))
            fw['Header'].attrs.create('omegab', 0.04757289217927339)
            fw['Header'].attrs.create('omegal', 0.6911)
            fw['Header'].attrs.create('omegam', 0.3089)
            fw['tau/H/1/1215'] = tau_sampled
            fw.create_group('spectra')
            fw['spectra/axis'] = 3*np.ones(shape=(cofm.shape[0],))
            fw['spectra/cofm'] = cofm
            fw['colden/H/1'] = np.zeros(shape=(1,))
            fw.create_group('density_Weight_debsity')
            fw.create_group('num_important')
            fw.create_group('velocity')
            fw.create_group('temperature')
            fw.create_group('tau_obs')
 
    
def get_cofm(seed, num, boxsize=205):
    """ A copy of fake_spectra.rand_spectra.get_cofm() to replicate the
        spectra used for hydro snalysis. 
    seed : the seed for random sample
    num : number of spectra
    """
    np.random.seed(seed)
    cofm = boxsize*np.random.random_sample((num,3))
    return cofm

def get_noiseless_map(MPI, z, savedir='density_highres/', savefile='FGPA_flux_z2.4.hdf5',
                      boxsize=205, Ngrids=205, Npix=205, SmLD=1, SmLV=1, fix_mean_flux=True):
    """Calculate the true map on a mesh grid of size (Ngrids*Ngrids*Npix)
    z : redshift
    savedir :  the directory containing the density map
    savefile : The name for hdf5 file to save final map on
    Ngrids : int, the size of the x and y dimensions of the desired map
    Npix : number of desired pxiels along final map
    """
    # MPI communications
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tau_conv, xrank, yrank = get_tau_conv(comm=comm, z=z, savedir=savedir, boxsize=boxsize, Ngrids=Ngrids, SmLD=SmLD, SmLV=SmLV)
    
    # Fix the mean flux
    # This may not be accurate since mean flux on each data chunck is corrected seperately
    if fix_mean_flux and (xrank!=0) and (yrank!=0):
        mean_flux = sm.get_mean_flux(z=z)
        scale = fs.mean_flux(tau_conv[xrank[0]:xrank[1]+1, yrank[0]:yrank[1]+1,:], mean_flux_desired=mean_flux)
    else :
        scale = 1
    ### Resampling pixels along spectra
    flux_conv = resample_flux(scale*tau_conv, Npix)

    ### MPI part
    # Make sure the data is contiguous in memeory
    flux_conv = np.ascontiguousarray(flux_conv, np.float64)
    # Add the results from all ranks
    comm.Allreduce(MPI.IN_PLACE, flux_conv, op=MPI.SUM)
    # We should subtract the amount below since in absense of 
    # absorption flux is 1 and the Allreduce() is adding them.
    flux_conv -= (comm.Get_size() - 1 )
    if rank == 0:
        with h5py.File(savedir+savefile,'w') as fw:
            fw['map'] = flux_conv
    comm.Barrier()


def resample_flux(tau, Npix):
    """Resample spectra to get Npix pixels along line of sight. It is done by averaging over
    few consecutive pixels.
    tau : Optical depth.
    Npix : The desired number of pxiels 
    """
    Nz = tau.shape[2]
    # Number of adjacent pixels along spectrum need to be averaged over
    addpix = int(Nz / Npix)
    flux = np.zeros(shape=(tau.shape[0], tau.shape[1], Npix), dtype=np.float64)
    for t in range(Npix):
        flux[:,:,t] = np.sum(np.exp(-tau[:,:,t*addpix:(t+1)*addpix]), axis=2)/addpix 
    flux = gaussian_filter1d(flux, sigma=1, mode='wrap')
    return flux


def get_tau_conv(comm, z, savedir='density_highres/', savefile='FGPA_flux_z2.4.hdf5',boxsize=205, Ngrids=205, SmLD=1, SmLV=1):
    """ Calculate tau in redshift space
        Convolving tau in real space with Voight profile
        comm : Communicator for MPI
        z : redshift
        savedir :  the directory containing the density map
        savefile : The name for hdf5 file to save final map on
        Ngrids : int, the size of the x and y dimensions of the desired map
        Returns :
        tau_conv : convoluted optical depth
        [xstart, xend], [ystart, yend] : the ranges of x and y coordinates covered by any rank
        """

    rank = comm.Get_rank()
    size = comm.Get_size()
    with h5py.File(savedir+str(rank)+'_densfield.hdf5','r') as f:
        # nbodykit does not break the data along the z direction, so Nz is the 
        # the size of the initial density map in all 3 dimentions
        Nz = f['DM/dens'][:].shape[2]
        dvbin = cosmo.H(z).value*boxsize/(cosmo.h*Nz*(1+z))
        up = np.arange(Nz)*dvbin
        tau_conv = np.zeros(shape=(Ngrids, Ngrids, Nz))
        # Approx position of the desired sightlines. The approximation should be ok
        # for FGPA since the density map has very fine voxels
        x, y = int(Nz/Ngrids)*np.arange(Ngrids), int(Nz/Ngrids)*np.arange(Ngrids)
        # Which sightlines are on this rank
        indx = np.where(np.isin(x, f['DM/x'][:]))[0]
        if indx.size == 0:
            # Some ranks may not hold any sightlines at all
            print('The sightline coordinates are not on density grids on rank=', 
                  rank, flush=True)
            print("The y desnity grid coordinates are = ", f['DM/y'][:], flush=True)
            return tau_conv, [0,0], [0,0]
        
        xstart, xend = indx[0], indx[-1]
        indy = np.where(np.isin(y, f['DM/y'][:]))[0]
        if indy.size == 0:
            # Some ranks may not hold any sightlines at all
            print('The sightline coordinates are not on density grids on rank=', 
                  rank, flush=True)
            print("The y desnity grid coordinates are = ", f['DM/y'][:], flush=True)
            return tau_conv, [0,0], [0,0]
        
        ystart, yend = indy[0], indy[-1]
        print('Sightlines on Rank =', rank, (int(xstart), int(xend)), (int(ystart), int(yend)) ,flush=True)
        # i, j are indices for the final flux map (Ngrids * Ngrids)
        for i in range(xstart, xend+1):
            if rank ==0:
                print(str(int(100*(i-xstart)/indx.size))+'%', flush=True )
            # Indices on f['DM/dens'] map
            ic = x[i] - f['DM/x'][0]
            for j in range(ystart, yend+1):
                # Indices on f['DM/dens'] map
                jc = y[j] - f['DM/y'][0]
                dens = f['DM/dens'][ic,jc,:]
                tau_real = get_tau_real(f['DM/dens'][ic,jc,:], z=z)
                # Peculiar velocity addition
                ind = np.where((dens != 0))
                vel_pec = np.zeros_like(f['DM/pz'][ic,jc,:])
                vel_pec[ind] = f['DM/pz'][ic,jc,:][ind]/(dens[ind]*np.sqrt(1+z))
                vel_pec = gaussian_filter1d(vel_pec, SmLV)
                dens = gaussian_filter1d(dens, SmLD)
                u0 = up + vel_pec
                btherm = get_btherm(dens)
                # To avoide devision by 0, if b_threm == 0, pass a nonzero value since
                # tau_real is 0 in that voxel anyway, tau_conv will be 0.
                btherm[np.where(btherm==0)] = 1.0
                for k in range(Nz):
                    dvel = np.abs(up[k]-u0)
                    # Periodic Boundary
                    indv = np.where(dvel > dvbin*Nz/2)
                    dvel[indv] = dvbin*Nz - dvel[indv]
                    Voight = (1/btherm)*np.exp(-(dvel/btherm)**2)
                    tau_conv[i,j,k] = np.sum(tau_real*Voight*dvbin)
        comm.Barrier()
        print('Rank', rank, 'is done with tau_conv', flush=True)
        return tau_conv, [xstart,xend], [ystart,yend]
    
def get_tau_real(Delta, z):
    """ Get tau in real space
        The amplitude needs to get fixed with mean
        observed flux or 1D power
        z : redshift
    """
    lambda_Lya, sigma_Lya = 1215.67, 1
    return (lambda_Lya*sigma_Lya/cosmo.H(z).value)*get_nHI(Delta)

def get_nHI(Delta, gamma=1.46):
    """ Calculate Neutral Hydrogen Density
        The amplitude needs to get fixed with mean flux
    """
    return Delta**(2-0.7*(gamma -1))

def get_btherm(Delta, mp=1.67*10**(-27), kB=1.38*10**(-23)):
    """ Thermal Doppler parameter in km/s"""
    return np.sqrt(2*kB*get_Temp(Delta)/mp)/1000

def get_Temp(Delta, T0=1.94*10**4, gamma=1.46):
    """ Temperature density relation 
        Delta : (1 + delta_b)
    """
    return T0*Delta**(gamma-1)
