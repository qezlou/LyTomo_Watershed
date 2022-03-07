"""For a helper script to run this module refer to https://github.com/mahdiqezlou/LyTomo_Watershed/tree/dist/helper_scripts/DensityField """
import h5py
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from nbodykit.lab import *
from nbodykit.lab import HDFCatalog
from nbodykit import CurrentMPIComm

class Density:
    """ A parallel code to get the density field in Gadget simulations"""
    def __init__(self, snaps, savedir, savefile, Nmesh, zspace=True, momentumz=False, parttype='PartType1'):
        """
        snaps : The path to the simulation snapshots in the format like : ./output/snapdie_029/snap_029*
        savedir : The path to the directory to store the results in. It saves the intermdeiate results and the final map.
        savefile: the file to save the full map on.
        Nmesh : The grid size
        zspace : Whether the density be on redshift space or not
        momentumz : If True, calculate the weighted velocity along z direction. The 
                    recorded field is (1+ delta)*Vpec_z, we have also saved (1+delta) as density
                    feild.
        """
        # Files' info
        self.snaps = snaps
        self.savedir = savedir
        self.savefile = savefile
        # The method parameters
        self.Nmesh = Nmesh
        self.zspace = zspace
        self.momentumz = momentumz
        self.parttype = parttype
        if self.parttype =='PartType0':
            self.typestr='Gas'
        if self.parttype =='PartType1':
            self.typestr='DM'
        # MPI
        self.comm = CurrentMPIComm.get()
        
    def Gadget(self):
        """
        Generate the density feild on a grid for Gadget simulations, e.g. Iluustris.
        """
        with h5py.File(self.snaps[0:-1]+'1.hdf5','r') as fr:
            z = fr['Header'].attrs['Redshift']
            boxsize = fr['Header'].attrs['BoxSize']
        with h5py.File(self.savedir+str(self.comm.rank)+"_densfield.hdf5",'w') as f :
            cat = HDFCatalog(self.snaps, dataset=self.parttype, header='Header')
            if self.zspace :
                ## peculiar velocity correction
                # Old Dask used in nbodykit does not accept elemnt-wise assignment, so we need to project V_pec along z 
                cat['Coordinates'] = (cat['Coordinates'] + (1000*cosmo.h*np.sqrt(1+z)/cosmo.H(z).value)*cat['Velocities']*[0,0,1])%boxsize

            print('Rank ', self.comm.rank, ' cat,size= ', cat.size, flush=True)
            mesh = cat.to_mesh(Nmesh=self.Nmesh, position='Coordinates')
            dens = mesh.compute()
            if self.momentumz :
                # Average line-of-sight velocity in each voxel
                cat['Vz'] = cat['Velocities'][:,2]/np.sqrt(1+z)
                mesh_momen = cat.to_mesh(Nmesh=self.Nmesh, position='Coordinates', value='Vz')
                pz = mesh_momen.compute()
            L = np.arange(0, self.Nmesh, 1)
            f[self.typestr+'/dens'] = dens[:]
            if self.momentumz :
                f[self.typestr+'/pz'] = pz[:]
            f[self.typestr+'/x'] = L[dens.slices[0]]
            f[self.typestr+'/y'] = L[dens.slices[1]]
            f[self.typestr+'/z'] = L[dens.slices[2]]
            f[self.typestr+'/num_parts'] = cat.size

        self.comm.Barrier()
        if self.comm.rank==0:
            print('Saving the results', flush=True)
            self.make_full_mesh()
        self.comm.Barrier()
        
    def make_full_mesh(self):
        """ Loop over the saved hdf5 files for each rank to constrcut the full mesh and save it 
        """
        dens = np.empty((self.Nmesh, self.Nmesh, self.Nmesh))
        pz = np.empty((self.Nmesh, self.Nmesh, self.Nmesh))
        num_parts=0
        for i in range(self.comm.Get_size()) :
            print('file '+str(i)+' started!')
            with h5py.File(self.savedir+str(i)+'_densfield.hdf5','r') as f:
                x = slice(f[self.typestr+'/x'][0], f[self.typestr+'/x'][-1]+1)
                y = slice(f[self.typestr+'/y'][0], f[self.typestr+'/y'][-1]+1) 
                z = slice(f[self.typestr+'/z'][0], f[self.typestr+'/z'][-1]+1)
                dens[x,y,z] = f[self.typestr+'/dens'][:]
                if self.momentumz :
                    pz[x,y,z] = f[self.typestr+'/pz'][:]
                num_parts += f[self.typestr+'/num_parts'][()]
        with h5py.File(self.savefile, 'w') as f_w:
            f_w[self.typestr+'/dens'] = dens
            f_w[self.typestr+'DM/momentumz'] = pz
            f_w[self.typestr+'/num_parts']=num_parts