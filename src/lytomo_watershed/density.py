"""For a helper script to run this module refer to https://github.com/mahdiqezlou/LyTomo_Watershed/tree/dist/helper_scripts/DensityField """
import h5py
import numpy as np
#from mpi4py import MPI
from astropy.cosmology import Planck15 as cosmo
from nbodykit.lab import *
from nbodykit import CurrentMPIComm
#CurrentMPIComm.set(MPI.COMM_WORLD)
class Density:
    """ A parallel code to get the density field in Gadget simulations"""
    def __init__(self, snaps, Nmesh, savedir, savefile=None,  sim_type='Gadget', boxsize=None, z=None,
                 parttype='PartType1', ls_vec=[0,0,1], zspace=True, momentumz=False):
        """
        snaps : The path to the simulation snapshots in the format like : ./output/snapdie_029/snap_029*
        savedir : The path to the directory to store the results in. 
                  It saves the intermdeiate results and the final map.
        savefile: Optional, the file to save the full map on.
        Nmesh : The grid size
        sim_type : 'Gadget' for Gadget/Arepo simulations with hdf5 output fromat
                    'Gadget_old' for old Gadget simulations with non-hdf5 output format
        boxsize : In ckpc/h, only if sim_type='Gadget_old', othrwise default in None
        z : accurate redshift, only if sim_type='Gadget_old', othrwise default in None
        zspace : Whether the density be on redshift space or not
        momentumz : If True, calculate the weighted velocity along z direction. The 
                    recorded field is (1+ delta)*Vpec_z, we have also saved (1+delta) as density
                    feild.
        ls_vec : list, len of 3
                 Unit line-of-sight vector
        """
        # Files' info
        self.snaps = snaps
        self.savedir = savedir
        self.savefile = savefile
        # The method parameters
        self.Nmesh = Nmesh
        self.sim_type = sim_type
        self.boxsize = boxsize
        self.z = z
        self.zspace = zspace
        self.momentumz = momentumz
        self.parttype = parttype
        self.ls_vec = ls_vec
        if self.parttype =='PartType0':
            self.typestr='Gas'
        if self.parttype =='PartType1':
            self.typestr='DM'
        # MPI
        self.comm = CurrentMPIComm.get()
    
    def _apply_RSD(self, coord, vel):
        """Apply redshift space distortion along the slightline"""
        # Old Dask used in nbodykit does not accept elemnt-wise assignment, 
        # so we need to project V_pec along ls_vec
        coord = (coord + vel*self.ls_vec*1000*cosmo.h*np.sqrt(1+self.z)/cosmo.H(self.z).value)%self.boxsize
        return coord
        
    
    def get_gadget_old_cat(self):
        """
        Retrun a particle catalog for Gadget1 type simulations, e.g. MultiDarkMatter 
        It has only bee ntested on MDPL2 simulations:
        https://www.cosmosim.org/cms/files/simulation-data/
        Using the Gadget1Catalog from nbodykit package
        Params:
            box size in ckpc/h
            z : accurate redshift 
        """
        cat = Gadget1Catalog(path=self.snaps, comm=self.comm)
        # MDPL2 has the length units in cMpc/h
        cat['Coordinates'] = cat['Position']*1000
        # The velocity is in units of sqrt(a)*km/s
        cat['Velocities'] = cat['GadgetVelocity']
        return cat 
    
    def get_gadget_cat(self):
        """
        Retrun a particle catalog for Gadget/Arepo type simulations, e.g. TNG/Illustris
        """
        from nbodykit.lab import HDFCatalog
        with h5py.File(self.snaps[0:-1]+'0.hdf5','r') as fr:
            self.z = fr['Header'].attrs['Redshift']
            self.boxsize = fr['Header'].attrs['BoxSize']
        cat = HDFCatalog(self.snaps, dataset=self.parttype, header='Header')
        return cat
        
    def Gadget(self):
        """
        Generate the density feild on a grid for Gadget simulations, e.g. Iluustris.
        """ 
        if self.sim_type == 'Gadget':
            cat = self.get_gadget_cat()
        elif self.sim_type == 'Gadget_old':
            cat = self.get_gadget_old_cat()
        else :
            raise TypeError('The snapshot type is not supported]')
        if self.zspace:
            cat['Coordinates'] = self._apply_RSD(coord=cat['Coordinates'], vel=cat['Velocities'])

        print('Rank ', self.comm.rank, ' cat,size= ', cat.size, flush=True)
        mesh = cat.to_mesh(Nmesh=self.Nmesh, position='Coordinates')
        dens = mesh.compute()
        if self.momentumz :
            # Average line-of-sight velocity in each voxel, the Gadget/Arepo units are in
            # sqrt(a)*km/s units
            cat['Vz'] = cat['Velocities'][:,2]/np.sqrt(1+self.z)
            mesh_momen = cat.to_mesh(Nmesh=self.Nmesh, position='Coordinates', value='Vz')
            pz = mesh_momen.compute()
        L = np.arange(0, self.Nmesh, 1)
        # Write each ranks' results on a file
        with h5py.File(self.savedir+str(self.comm.rank)+"_densfield.hdf5",'w') as f :
            f[self.typestr+'/dens'] = dens[:]
            if self.momentumz :
                f[self.typestr+'/pz'] = pz[:]
            f[self.typestr+'/x'] = L[dens.slices[0]]
            f[self.typestr+'/y'] = L[dens.slices[1]]
            f[self.typestr+'/z'] = L[dens.slices[2]]
            f[self.typestr+'/num_parts'] = cat.size
        if self.savefile is not None:
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