import numpy as np
import h5py
from . import illustris_python as il
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage.filters import gaussian_filter
from . import minima
from . import mpi4py_helper




def highz_halos(MPI, comm, peaks, basepath, m=None ,snap=29, lmap= None,savefile=None, mass_thresh=None, z=2.3161107439568918, boxsize=205, linking_contour=-2.0, sigma=4, min_radius=None):
    """ Get the halos with mass > mass_thresh around each peak within a volume whihc is union of labels passed to the function and a spehre of radius minrad"""
    # Get the flux contours
    from scipy.ndimage import label
    # if labeled map is not provided, use the
    # simple contours
    if lmap is None :
        m = gaussian_filter(m, sigma=sigma, mode='wrap')
        m = m / np.std(m)
        lmap, num_features = label((m < linking_contour))
    # Get all halos at high z
    halos = il.groupcat.loadHalos(basepath, snap, fields=['GroupMass', 'GroupPos','GroupVel'])
    # In 1 cMpc/h resolution, v_peculiar of the group is also taken into account (it changes the result by a few cMpc/h )
    halos_coord = np.around(halos['GroupPos']/1000 + (halos['GroupVel']/cosmo.H(z).value)*cosmo.h*(1+z)**2).astype(int)%boxsize
    dims = halos_coord.max(0) + 1
    
    PeakIDs = np.unique(lmap).astype(np.int)[1:]
    PeakIDsRank = mpi4py_helper.distribute_array(MPI, comm, PeakIDs)
    PeakIDsHalos = np.array([], dtype=np.int)
    
    # Read the cotours
    for c,i in enumerate(PeakIDsRank):
        #print(int(c/((PeakIDsRank.size-1)*100)), ' percent done!', flush=True)
        #find the voxels within the same label and Rsigma from the peaks : 
        #But, if watershed algorithm is being used, just check the labels
        invol = lmap == i
        # Whether to put a minimum structure radius (like 4cMpc/h in S15)
        if min_radius is not None :
            L = lmap.shape[0]
            xgr, ygr, zgr = minima.tophat_mask(peaks['x'][i-1], peaks['y'][i-1], peaks['z'][i-1], Lx=L, Ly=L, Lz=L, maxrad=min_radius ,BC=True)
            invol[xgr, ygr, zgr] = True
    
        involx, involy, involz = np.where(invol)
        pc_coord = np.zeros((involx.size,3))
        pc_coord[:,0], pc_coord[:,1], pc_coord[:,2] = involx, involy, involz
        #Finding halos within countour i
        halos_ind = np.where(np.in1d(np.ravel_multi_index(halos_coord.T, dims), np.ravel_multi_index(pc_coord.T.astype(int), dims)))[0]
        if mass_thresh is not None :
            # Pick massive halos M > mass_thresh*10^10 M_sol
            massive_ind_temp = np.where(halos['GroupMass'][halos_ind] > mass_thresh )[0]
        else :
            # Pick the most massive halo
            if halos_ind.size == 0 :
                print('No halos found for pc with ID ', i, flush=True)
                continue
            else :
                massive_ind_temp = np.where(halos['GroupMass'][halos_ind] == np.max(halos['GroupMass'][halos_ind]))[0]
            if massive_ind_temp[0].size >1 :
                raise ValueError("More than one most massive halo")
        print(massive_ind_temp.size, ' halos found', flush=True)
        if c==0:
            massive_ind = halos_ind[massive_ind_temp]
        else:
            massive_ind = np.append(massive_ind, halos_ind[massive_ind_temp])
            # These ids are actually the pc indicies+1 not labled contours which are no longer important
        PeakIDsHalos = np.append(PeakIDsHalos, np.ones_like(massive_ind_temp)*i)
    
    massive_mass = (halos['GroupMass'][massive_ind]).astype(np.float32)
    x, y, z = halos_coord[massive_ind][:,0], halos_coord[massive_ind][:,1], halos_coord[massive_ind][:,2]
    
    x, y, z = np.ascontiguousarray(x, dtype=np.int), np.ascontiguousarray(y, dtype=np.int), np.ascontiguousarray(z, dtype=np.int)
    massive_mass = np.ascontiguousarray(massive_mass, dtype=np.float32)
    massive_ind = np.ascontiguousarray(massive_ind.astype(np.int), dtype=np.int)
    PeakIDsHalos = np.ascontiguousarray(PeakIDsHalos, dtype=np.int)
    comm.barrier()
    
    massive_ind = mpi4py_helper.Allgatherv_helper(MPI, comm, data=massive_ind, data_type=np.int)
    comm.barrier()
    PeakIDsHalos = mpi4py_helper.Allgatherv_helper(MPI, comm, data=PeakIDsHalos, data_type=np.int)
    comm.barrier()
    x = mpi4py_helper.Allgatherv_helper(MPI, comm, data=x, data_type=np.int)
    comm.barrier()
    y = mpi4py_helper.Allgatherv_helper(MPI, comm, data=y, data_type=np.int)
    comm.barrier()
    z = mpi4py_helper.Allgatherv_helper(MPI, comm, data=z, data_type=np.int)
    comm.barrier()
    massive_mass = mpi4py_helper.Allgatherv_helper(MPI, comm, data=massive_mass, data_type=np.float32)
    comm.barrier()
    
    if comm.Get_rank() == 0 :
        coords = np.zeros(shape=(x.size,3), dtype=np.int)
        coords[:,0], coords[:,1], coords[:,2] = x, y, z
        
        if savefile is not None:
            with h5py.File(savefile, 'w') as f_id:
                f_id['HaloID'] = massive_ind
                f_id['PeakIDs'] = PeakIDsHalos
                f_id['Mass'] = massive_mass
                f_id['coords'] = coords
    else :
        return massive_ind, PeakIDs, massive_mass
    
def highz_subhalos(MPI, comm, peaks, snap, lmap, basepath, savefile=None, mass_thresh=0, z=2.3161107439568918, boxsize=205, linking_contour=-2.0, sigma=4, min_radius=None):
    """ Get the suhalos with mass > mass_thresh around each peak within a volume which is union of labele passed to
    the function and a spehre of radius minrad. This is the actual function we use rather highz_halos().
    peaks and lmap : HDF5 files we get from minima.motmo_partition_v2()
    mass_thresh : record subhalos with mas slarger than this
    min_radius : Determines the minimum volume around peaks to look for subhalos
    basepath : The pathto the simulation output directory, we followe the recommended organization in TNG website. The 
                directory path should  look like this : `~/TN G300-1/output`
    """
    # Get all subhalos at high z
    subhalos = il.groupcat.loadSubhalos(basepath, snap, fields=['SubhaloMass', 'SubhaloPos','SubhaloVel'])
    
    # In 1 cMpc/h resolution, v_peculiar of the group is also taken into account (it changes the result by a few cMpc/h )
    subhalos_coord = np.around(np.around(subhalos['SubhaloPos']/1000 + (subhalos['SubhaloVel']/cosmo.H(z).value)*cosmo.h*(1+z)**2).astype(np.int)%boxsize).astype(np.int)
    
    dims = subhalos_coord.max(0) + 1
    
    PeakIDs = np.unique(lmap).astype(np.int)[1:]
    PeakIDsRank = mpi4py_helper.distribute_array(MPI, comm, PeakIDs)
    PeakIDsSubhalos = np.array([], dtype=np.int)
    
    # Read the cotours
    for c, i in enumerate(PeakIDsRank):
        #print(int(i/(np.unique(lmap).size-1)*100), ' percent done!')
        #find the voxels within the same label
        invol = lmap == i
        # Make sure at least a sphere of radius minrad is searched around the strucutre
        if min_radius is not None:
            L = lmap.shape[0]
            xgr, ygr, zgr = minima.tophat_mask(peaks['x'][i-1], peaks['y'][i-1], peaks['z'][i-1], Lx=L, Ly=L, Lz=L, maxrad=min_radius ,BC=True)
            invol[xgr, ygr, zgr] = True
        involx, involy, involz = np.where(invol)
        pc_coord  = np.zeros((involx.size,3))
        pc_coord[:,0], pc_coord[:,1], pc_coord[:,2] = involx, involy, involz
        #Finding subhalos within countour i
        subhalos_ind = np.where(np.in1d(np.ravel_multi_index(subhalos_coord.T, dims), np.ravel_multi_index(np.floor(pc_coord.T).astype(int), dims)))[0]
        if subhalos_ind.size == 0:
            raise ValueError('No subhalo found within the contour at all')
        
        # Pick massive subhalos M > mass_thresh*10^10 M_sol
        massive_ind_temp = np.where(subhalos['SubhaloMass'][subhalos_ind] > mass_thresh )[0]
        if massive_ind_temp.size == 0 :
            raise ValueError('No massive enough subhalo found for structure '+str(i+1))
            
        print(massive_ind_temp.size, ' subhalos found', flush=True)
        
        if c==0:
            massive_ind = subhalos_ind[massive_ind_temp]
        else:
            massive_ind = np.append(massive_ind, subhalos_ind[massive_ind_temp])
        PeakIDsSubhalos = np.append(PeakIDsSubhalos, np.ones(shape=(massive_ind_temp.size,), dtype=np.int)*i)
            
            
    massive_mass = (subhalos['SubhaloMass'][massive_ind]).astype(np.float32)
    x, y, z = subhalos_coord[massive_ind][:,0], subhalos_coord[massive_ind][:,1], subhalos_coord[massive_ind][:,2]
    x, y, z = np.ascontiguousarray(x, dtype=np.int), np.ascontiguousarray(y, dtype=np.int), np.ascontiguousarray(z, dtype=np.int)
    massive_mass = np.ascontiguousarray(massive_mass, dtype=np.float32)
    massive_ind = np.ascontiguousarray(massive_ind.astype(np.int), dtype=np.int)
    PeakIDsSubhalos = np.ascontiguousarray(PeakIDsSubhalos, dtype=np.int)
    comm.barrier()
    massive_ind = mpi4py_helper.Allgatherv_helper(MPI, comm, data=massive_ind, data_type=np.int)
    comm.barrier()
    PeakIDsSubhalos = mpi4py_helper.Allgatherv_helper(MPI, comm, data=PeakIDsSubhalos, data_type=np.int)
    comm.barrier()
    x = mpi4py_helper.Allgatherv_helper(MPI, comm, data=x, data_type=np.int)
    comm.barrier()
    y = mpi4py_helper.Allgatherv_helper(MPI, comm, data=y, data_type=np.int)
    comm.barrier()
    z = mpi4py_helper.Allgatherv_helper(MPI, comm, data=z, data_type=np.int)
    comm.barrier()
    massive_mass = mpi4py_helper.Allgatherv_helper(MPI, comm, data=massive_mass, data_type=np.float32)
    comm.barrier()
    
    if comm.Get_rank() == 0 :
        coords = np.zeros(shape=(x.size,3), dtype=np.int)
        coords[:,0], coords[:,1], coords[:,2] = x, y, z
        if savefile is not None:
            with h5py.File(savefile, 'w') as f_id:
                # The indices to the subhalo catalogue :
                f_id['SubhaloInd'] = massive_ind 
                f_id['PeakIDs'] = PeakIDsSubhalos
                f_id['Mass'] = massive_mass
                f_id['coords'] = coords
    comm.barrier()

def pos_mass_halos(savefile=None):
    """ Get Position and Mass of the massive halos within flux contours, to over-plot on  z=2.3 map  """
    halos = il.groupcat.loadHalos(basepath, 30, fields=['GroupMass', 'GroupPos'])
    try :
        f_id = h5py.File('./halos/massive_halos.hdf5','r')
        massive_ind, contour_id, massive_mass = f['HaloID'], f['ContourID']
    except: 
        massive_ind, massive_contour_id = highz_halos(savefile='massive_halos.hdf5')
    halos_coord = (halos['GroupPos']/1000).astype(int)
    massive_pos, massive_mass = halos_coord[massive_ind], halos['GroupMass'][massive_ind]
    
    if savfile is not None :
      f_halos = h5py.File(savefile,'w')
      f_halos['Coord'] =  massive_ind
      f_halos['Mass'] =  massive_pos
      f_halos['Contour_id'] = massive_contour_id
      f_halos.close()


def within_random_spheres(r=2, num=358, seed=1456, box=205):
    """A function to get Most Massive halos within random spheres with radii r""" 
    #pc = {'x':[], 'y':[], z:''}
    pc={}
    np.random.seed(seed)
    coords = np.random.randint(0,box+1, size=(num,3))
    pc['x'], pc['y'], pc['z'] = coords[:,0], coords[:,1], coords[:,2]
    lmap = np.zeros((box,box,box))
    for i in range(num):
      x, y, z = np.arange(coords[i,0]-r,coords[i,0]+r+1), np.arange(coords[i,1]-r,coords[i,1]+r+1), np.arange(coords[i,2]-r,coords[i,2]+r+1)
      d = np.zeros((3,2*r+1))
      d[0,:],d[1,:],d[2,:] = x, y, z
      for j in range(3):
          ind = np.where(d[j,:] < 0)
          d[j,ind] += box
          ind = np.where(d[j,:] >= box)
          d[j,ind] -= box
      d = d.astype(int)
      lmap[d[0,:], d[1,:], d[2,:]] = i
    highz_halos(pc=pc, m=None ,lmap= lmap, savefile='MMhalos_random_spheres.hdf5', mass_thresh=None, z=2.3161107439568918, boxsize=205, linking_contour=-2.0, sigma=4,max_structure_size=None)
    """
      for j in range(3):
         x,y,z = np.arange(coords[i,0]-r,coords[i,0]+r+1), np.arange(coords[i,1]-r,coords[i,1]+r+1), np.arange(coords[i,2]-r,coords[i,2]+r+1)
      ind = np.where(np.in1d(halos_coord[:,0], x)*np.in1d(halos_coord[:,1], y)*np.in1d(halos_coord[:,2], z))
      
    """
