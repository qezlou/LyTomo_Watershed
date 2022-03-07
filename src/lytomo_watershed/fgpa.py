"""Generating spectra with Fluctuating Gunn Peterson Approximation (FGPA)

- The code is MPI working on Illustris and MP-Gadget snapshots. The packages needed are :

    - astropy 
    - fake_spectra

To get the FGPA spectra, refer to the helper script at
https://github.com/mahdiqezlou/LyTomo_Watershed/tree/dist/helper_scripts/FGPA 
which follows the steps below :

1. From density.py use Gadget() to construct DM density
   field on a grid with desired size. For FGPA, the grid cells should on avergae 
   have 1 particle per cell.
  
2. Save the results from the previous step in savedir directory. The density 
   has been saved on several files depending on number of ranks used. 

3. Run get_noiseless_map() or get_sample_spectra() functions here and pass the 
   directories for the density field above as savedir argument. 

4. The output is a single hdf5 file containing either the full true flux map or the
   random spectra sample.

"""

import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import gaussian_filter1d
import fake_spectra.fluxstatistics as fs
from . import spectra_mocking as sm
from . import density

class Fgpa:
    """A class for FGPA method"""
    def __init__(self, MPI, comm, z, boxsize, Ngrids, Npix, SmLD, SmLV, savedir, fix_mean_flux=True, 
                 mean_flux=None, gamma=1.46, T0=1.94*10**4):
        """
        Params : 
            comm : instanse of the MPI communicator
            z : redshift
            savedir :  the directory containing the density map
            Ngrids : int, the size of the x and y dimensions of the desired map
            Npix : number of desired pxiels along final map
            gamma : The slope in temperature-density relation
            T0 : Temperature at mean density
            """
        # Initialize the MPI communication
        self.MPI = MPI
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.z = z
        self.boxsize = boxsize
        self.Ngrids = Ngrids
        self.Npix = Npix
        self.SmLD = SmLD
        self.SmLV = SmLV
        self.savedir = savedir
        self.fix_mean_flux = fix_mean_flux
        self.mean_flux = mean_flux
        # Thermal paramters of the IGM
        self.gamma= gamma
        self.T0 = T0
        # Physical constants
        # Lya constants do not matter at all, since we fix the mean absorption 
        self.lambda_Lya = 1215.67
        self.sigma_Lya = 1
        self.mp = 1.67*10**(-27) # proton mass in Kg
        self.kB=1.38*10**(-23) # Boltzman const in SI
                
        
    def get_sample_spectra(self, num, seed=13, savefile='spectra_z2.4_FGPA_n1.hdf5'):
        """Get a sample of spectra to be used for mock map reconstruction
        num : Numebr of desired random spectra
        savefile : The name for hdf5 file to save the spectra file
        """

        tau_conv = self.get_tau_conv()
        if seed is not None:
            cofm = self.get_cofm(num=num, Nvoxles=tau_conv.shape[0], seed=seed).astype(int)
        else:
            x, y = np.meshgrid(np.arange(self.Ngrids), np.arange(self.Ngrids))
            cofm = np.zeros(shape=(self.Ngrids*self.Ngrids,2), dtype=int)
            cofm[:,0] = np.ravel(x)
            cofm[:,1] = np.ravel(y)
            del x, y
        ind = np.where(tau_conv!= -1)
        tau_sampled = np.zeros(shape=(cofm.shape[0], tau_conv.shape[2]))
        # Find the sample spectra on this rank
        ind_cofm = np.where(np.isin(cofm[:,0], ind[0])*np.isin(cofm[:,1], ind[1]))[0]
        tau_sampled[ind_cofm] = tau_conv[cofm[ind_cofm,0], cofm[ind_cofm,1],:]
        ### MPI part
        # Make sure the data is contiguous in memeory
        tau_sampled = np.ascontiguousarray(tau_sampled, np.float64)
        # Add the results from all ranks
        self.comm.Allreduce(self.MPI.IN_PLACE, tau_sampled, op=self.MPI.SUM)
        # Scale the tau to get mean flux right
        if self.fix_mean_flux:
            if self.mean_flux is None:
                mean_flux = sm.get_mean_flux(z=self.z)
            else:
                mean_flux =self.mean_flux
            print('mean flux is ', mean_flux, flush=True)
            scale = fs.mean_flux(tau_sampled, mean_flux_desired=mean_flux)
        else:
            scale=1
        tau_sampled *= scale
        if self.rank==0 :
            print('Scaling tau with :', scale)
        # Change cofm to kpc/h to record on spctra
        cofm  = cofm.astype(float)*(self.boxsize*1000/tau_conv.shape[0])

        if self.rank == 0 :
            with h5py.File(self.savedir+savefile,'w') as fw:
                # We need all Header info to load the file with fake_spectra
                # Some attrs are copied from hydro spectra, a more stable way
                # should be implemented
                fw.create_group('Header')
                fw['Header'].attrs.create('redshift', self.z)
                fw['Header'].attrs.create('box', self.boxsize*1000)
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


    def get_cofm(self, num, Nvoxels, seed):
        """ A copy of fake_spectra.rand_spectra.get_cofm() to replicate the
            spectra used for hydro snalysis. 
        seed : the seed for random sample
        num : number of spectra
        Nvoxels: the number of vxoels along each side of the simulations
        """
        np.random.seed(seed)
        cofm = Nvoxels*np.random.random_sample((num,3))
        return cofm

    def get_noiseless_map(self, savefile='FGPA_flux_z2.4.hdf5'):
        """Calculate the true map on a mesh grid of size (Ngrids*Ngrids*Npix)
        savefile : The name for hdf5 file to save final map on
        """
        tau_conv = self.get_tau_conv()
        ind = tau_conv != -1
        if self.fix_mean_flux:
            if self.mean_flux is None:
                mean_flux = sm.get_mean_flux(z=self.z)
            else:
                mean_flux = self.mean_flux
            print('mean flux is ', mean_flux, flush=True)
            scale = fs.mean_flux(tau_conv[ind], mean_flux_desired=mean_flux)
        else :
            scale = 1
        tau_conv[~ind] = 0
        ### Resampling pixels along spectra
        flux_conv = self.resample_flux(scale*tau_conv)

        ### MPI part
        # Make sure the data is contiguous in memeory
        flux_conv = np.ascontiguousarray(flux_conv, np.float64)
        # Add the results from all ranks
        self.comm.Allreduce(self.MPI.IN_PLACE, flux_conv, op=self.MPI.SUM)
        # We should subtract the amount below since in absense of 
        # absorption flux is 1 and the Allreduce() is adding them.
        flux_conv -= (self.comm.Get_size() - 1 )
        if self.rank == 0:
            with h5py.File(self.savedir+savefile,'w') as fw:
                fw['map'] = flux_conv
        self.comm.Barrier()


    def resample_flux(self, tau):
        """
        Resample spectra to get Npix pixels along line of sight. It is done by averaging the flux over
        few consecutive pixels.
        Params :
            tau : Optical depth.
        """
        Nz = tau.shape[2]
        # Number of adjacent pixels along spectrum need to be averaged over
        addpix = int(Nz / self.Npix)
        flux = np.zeros(shape=(tau.shape[0], tau.shape[1], self.Npix), dtype=np.float64)
        for t in range(self.Npix):
            flux[:,:,t] = np.sum(np.exp(-tau[:,:,t*addpix:(t+1)*addpix]), axis=2)/addpix 
        flux = gaussian_filter1d(flux, sigma=1, mode='wrap')
        return flux 

    def get_tau_conv(self):
        """ 
        Calculate tau in redshift space
        Convolving tau in real space with an apprimation of the Voight profile (Gaussian profile)
        Returns :
            tau_conv : convoluted optical depth
        """
        import glob
        import os
        from . import mpi4py_helper

        fnames = glob.glob(os.path.join(self.savedir,'*_densfield.hdf5'))
        fnames = mpi4py_helper.distribute_files(comm=self.comm, fnames=fnames)

        tau_conv = None
        c=0
        for fn in fnames:
            c+=1
            print(self.rank, fn, flush=True)
            if not os.path.exists(fn):
                raise IOError('File '+fn+' does not exist!')
            with h5py.File(fn,'r') as f:
                if tau_conv is None:
                    # nbodykit does not break the data along the z direction, so Nz is the 
                    # the size of the initial density map in all 3 dimentions
                    Nz = f['DM/dens'][:].shape[2]
                    dvbin = cosmo.H(self.z).value*self.boxsize/(cosmo.h*Nz*(1+self.z))
                    up = np.arange(Nz)*dvbin
                    tau_conv = -1*np.ones(shape=(self.Ngrids, self.Ngrids, Nz))
                    # Approx position of the desired sightlines. The approximation should be ok
                    # for FGPA since the density map has very fine voxels
                    x, y = int(Nz/self.Ngrids)*np.arange(self.Ngrids), int(Nz/self.Ngrids)*np.arange(self.Ngrids)
                # Which sightlines are on this rank
                indx = np.where(np.isin(x, f['DM/x'][:]))[0]
                if indx.size == 0:
                    # Some ranks may not hold any sightlines at all
                    print('The sightline coordinates are not on density grids on rank=', 
                          self.rank, flush=True)
                    print("The y desnity grid coordinates are = ", f['DM/y'][:], flush=True)
                    return tau_conv, [0,0], [0,0]

                xstart, xend = indx[0], indx[-1]
                indy = np.where(np.isin(y, f['DM/y'][:]))[0]
                if indy.size == 0:
                    # Some ranks may not hold any sightlines at all
                    print('The sightline coordinates are not on density grids on rank=', 
                          self.rank, flush=True)
                    print("The y desnity grid coordinates are = ", f['DM/y'][:], flush=True)
                    return tau_conv, [0,0], [0,0]

                ystart, yend = indy[0], indy[-1]
                print('Sightlines on Rank =', self.rank, (int(xstart), int(xend)), (int(ystart), int(yend)) ,flush=True)
                # i, j are indices for the final flux map (Ngrids * Ngrids)
                for i in range(xstart, xend+1):
                    if self.rank ==1:
                        print(str(int(100*c/len(fnames)))+'%', flush=True )
                    # Indices on f['DM/dens'] map
                    ic = x[i] - f['DM/x'][0]
                    for j in range(ystart, yend+1):
                        # Indices on f['DM/dens'] map
                        jc = y[j] - f['DM/y'][0]
                        dens = f['DM/dens'][ic,jc,:]
                        tau_real = self.get_tau_real(f['DM/dens'][ic,jc,:])
                        # Peculiar velocity addition
                        ind = np.where((dens != 0))
                        vel_pec = np.zeros_like(f['DM/pz'][ic,jc,:])
                        # Convert momentum to velocity
                        vel_pec[ind] = f['DM/pz'][ic,jc,:][ind]/dens[ind]
                        vel_pec = gaussian_filter1d(vel_pec, self.SmLV)
                        dens = gaussian_filter1d(dens, self.SmLD)
                        u0 = up + vel_pec
                        btherm = self.get_btherm(dens)
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
        self.comm.Barrier()
        print('Rank', self.rank, 'is done with tau_conv', flush=True)
        return tau_conv

    def get_tau_real(self, Delta):
        """ Get tau in real space
            The amplitude needs to get fixed with mean
            observed flux or 1D power
            z : redshift
        """
        return (self.lambda_Lya*self.sigma_Lya/cosmo.H(self.z).value)*self.get_nHI(Delta)

    def get_nHI(self, Delta):
        """ Calculate Neutral Hydrogen Density
            The amplitude needs to get fixed with mean flux
        """
        return Delta**(2-0.7*(self.gamma -1))

    def get_btherm(self, Delta):
        """ Thermal Doppler parameter in km/s"""
        return np.sqrt(2*self.kB*self.get_Temp(Delta)/self.mp)/1000

    def get_Temp(self, Delta):
        """ Temperature density relation 
            Delta : (1 + delta_b)
        """
        return self.T0*Delta**(self.gamma-1)