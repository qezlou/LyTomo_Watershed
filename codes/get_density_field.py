# A parallel code to get density Field on a grid in  MP-Gadget and TNG  simulations
# 1. Run MP-Gadget() or TNG() to get the ranks store the density info on a separate fall
# 2. Run make_full_mesh() to get the full mesh
import numpy as np
from nbodykit.lab import *
from nbodykit.lab import HDFCatalog
from nbodykit.lab import BigFileCatalog
import h5py
from nbodykit import CurrentMPIComm
from astropy.cosmology import Planck15 as cosmo
comm = CurrentMPIComm.get()

def MP_Gadget(snap_dir = '/lustre/scratch/mqezlou//Base2/output/PART_006/', outfile='Base2_z2.3_densfield_compute.hdf5'):
    with h5py.File('./results/'+str(comm.rank)+'.hdf5','w') as f :
        for dataset in ['1', '0']:
           cat = BigFileCatalog(snap_dir, dataset=dataset, header='Header')
           Nmesh = 60
           mesh = cat.to_mesh(Nmesh = Nmesh)
           real = mesh.compute()
           L = np.arange(0,60, 1)
           #print('rank', comm.rank, 'slices :', real.slices)
           if dataset =='0':
             f['Gas'] = np.array(real[:])
           if dataset == '1':
             f['DM'] = np.array(real[:])
             f['x'] = L[real.slices[0]]
             f['y'] = L[real.slices[1]]
             f['z'] = L[real.slices[2]]


def TNG(snaps='/lustre/scratch/mqezlou/TNG300-1/output/snapdir_029/snap_029.*', savedir='./',Nmesh=205, zspace=True, momentumz=False, parttype=['PartType1']):
    """
      Read the explanation at the top of this code
      Nmesh : The grid size
      zspace : Whether the density be on redshift space or not
      momentumz : If True, calculate the weighted velocity along z direction. The 
      recorded field is (1+ delta)*Vpec_z, we have also saved (1+delta) as density
      feild.
    """

    with h5py.File(snaps[0:-1]+'1.hdf5','r') as fr:
        z = fr['Header'].attrs['Redshift']
        boxsize = fr['Header'].attrs['BoxSize']
    with h5py.File(savedir+str(comm.rank)+"_densfield.hdf5",'w') as f :
        for dataset in parttype:
           cat = HDFCatalog(snaps, dataset=dataset, header='Header')
           if zspace :
              ## peculiar velocity correction
              # Old Dask used in nbodykit does not accept elemnt-wise assignment, so we need to project V_pec along z 
              cat['Coordinates'] = (cat['Coordinates'] + (1000*cosmo.h/(cosmo.H(z).value*(1+z)**0.5))*cat['Velocities']*[0,0,1])%boxsize

           print('Rank ', comm.rank, ' cat,size= ', cat.size, flush=True)
           mesh = cat.to_mesh(Nmesh=Nmesh, position='Coordinates')
           dens = mesh.compute()
           if momentumz :
              # Average line-of-sight velocity in each voxel
              cat['Vz'] = cat['Velocities'][:,2]
              mesh_momen = cat.to_mesh(Nmesh=Nmesh, position='Coordinates', value='Vz')
              pz = mesh_momen.compute()
           L = np.arange(0, Nmesh, 1)
           if dataset=='PartType0':
             f['Gas/dens'] = dens[:]
             if momentumz :
                f['Gas/pz'] = pz[:]
             f['Gas/x'] = L[dens.slices[0]]
             f['Gas/y'] = L[dens.slices[1]]
             f['Gas/z'] = L[dens.slices[2]]
             f['Gas/num_parts'] = cat.size

           if dataset=='PartType1':
             f['DM/dens'] = dens[:]
             if momentumz :
                f['DM/pz'] = pz[:]
             f['DM/x'] = L[dens.slices[0]]
             f['DM/y'] = L[dens.slices[1]]
             f['DM/z'] = L[dens.slices[2]]
             f['DM/num_parts'] = cat.size

def make_full_mesh(savedir, rank_start, rank_end, num_grid_side, savefile, velocity=False):
   """ Loop over the saved hdf5 files for each rank to constrcut the full mesh and save it 
       parameters:
       savedir: the directoey the files saaved in via TNG() or MP_Gadget()
       rank_start : the to start with, rnaks are those used in TNG() or MP_Gadget()
       rank_end : the rank to finish with
       num_grid_side : number of grid cells on one side
       savefile: the file to save the full map on
   """
   m_DM = np.empty((num_grid_side, num_grid_side, num_grid_side))
   m_Gas = np.empty((num_grid_side, num_grid_side, num_grid_side))
   momentumz = np.empty((num_grid_side, num_grid_side, num_grid_side))
   num_parts=0
   for i in range(rank_start,rank_end) :
       print('file '+str(i)+' started!')
       with h5py.File(savedir+str(i)+'_densfield.hdf5','r') as f:
            x = slice(f['DM/x'][0], f['DM/x'][-1]+1)
            y = slice(f['DM/y'][0], f['DM/y'][-1]+1) 
            z = slice(f['DM/z'][0], f['DM/z'][-1]+1)
            m_DM[x,y,z] = f['DM/dens'][:]
            if velocity :
               momentumz[x,y,z] = f['DM/Vz'][:]
           
            #m_Gas[x,y,z] = f['Gas']
            num_parts += f['DM/num_parts'][()]
   with h5py.File(savefile, 'w') as f_w:
        f_w['DM/dens'] = m_DM
        f_w['DM/momentumz'] = momentumz
        f_w['Gas'] = m_Gas
        f_w['DM/num_parts']=num_parts

