# modules being used either by progenitor_parallel or induvidually to find any type (Gas/DM/BH) particles 
#lied within center of the z~0 DM halos and write their cordinates at higher redhsift on files
import h5py
import numpy as np
import illustris_python as il
import time
import os
from .. import mpi4py_helper
#from astropy.cosmology import Planck15 as cosmo

def get_clusters_low_z(min_mass = 10**4):
    """Script to write the position of z ~ 0 large mass halos on file """ 
    
    basepath='/lustre/scratch/mqezlou/TNG300-1/output'
    halos = il.groupcat.loadHalos(basepath,  98, fields=['GroupMass', 'GroupPos','Group_R_Crit200'])
    ind = np.where(halos['GroupMass'][:] > min_mass)
    with h5py.File('clusters_TNG300-1.hdf5','w') as f :
        f['Mass'] = halos['GroupMass'][ind]
        f['Group_R_Crit200'] = halos['Group_R_Crit200'][ind]
        f['x'], f['y'], f['z'] = halos['GroupPos'][ind[0],0], halos['GroupPos'][ind[0],1], halos['GroupPos'][ind[0],2]
        f.close()
    return 0

def get_center_part_IDs(MPI, basepath, PartType='BH', cluster_ind=0) :
    """ Returns the IDs of the partcles within Group_R_Critic200 of the center of the halos at z= 0
        MPI : MPI communicator from mpi4py
        PartType : PartType of the particles of interest, 1 for DM
        cluster_ind : the index to the group catalouge list for the desired halo
    """ 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Load the Cluster at z=0
    f = h5py.File('clusters_TNG300-1.hdf5', 'r')
    IDs = np.array([])
    #for i in range(f['x'].size): 
    i = cluster_ind
    parts = il.snapshot.loadHalo(basepath, snapNum=98, id=i, partType=PartType,fields = ['ParticleIDs','Coordinates'])
    parts_per_rank = int(parts['ParticleIDs'][:].size / size)
    if rank == size-1:
       coordinates = parts['Coordinates'][(size-1)*parts_per_rank:-1]
       ParticleIDs = parts['ParticleIDs'][(size-1)*parts_per_rank:-1]
    else:
       coordinates = parts['Coordinates'][rank*parts_per_rank : (rank+1)*parts_per_rank]
       ParticleIDs = parts['ParticleIDs'][rank*parts_per_rank : (rank+1)*parts_per_rank]
    if coordinates.size == 0:        
       print('Rank ', rank, 'coordinates.size', coordinates.size, flush=True)
    halo_center = np.array([f['x'][i], f['y'][i], f['z'][i]]) 
    d =  coordinates - halo_center
    dist = np.sqrt(np.sum(d*d, axis=1))
    ind = np.where(dist-f['Group_R_Crit200'][i] < 0)
    #subsam = np.random.randint(1, ind[0].size, int(ind[0].size/10))
    #IDs = np.append(IDs, parts['ParticleIDs'][ind[0][subsam]])
    IDs = np.append(IDs, parts['ParticleIDs'][ind[0]])
    IDs = np.ascontiguousarray(IDs, dtype=np.uint)
    IDs_size = np.zeros(shape=(size,), dtype=np.int)
    IDs_size[rank] = IDs.size
    IDs_size = np.ascontiguousarray(IDs_size, np.int)
    comm.Allreduce(MPI.IN_PLACE, IDs_size, op=MPI.SUM)
    comm.Barrier()
    IDs_all_ranks = np.empty(np.sum(IDs_size), dtype=np.uint)
    disp = np.zeros_like(IDs_size, dtype=np.int)
    for i in range(1, IDs_size.size):
       disp[i] = np.sum(IDs_size[0:i])
    #if rank == 0:
       #print('IDs_size', IDs_size, flush=True)
       #print('disp ', disp, flush=True)
    comm.Allgatherv(IDs, [IDs_all_ranks, tuple(IDs_size.astype(np.int)), tuple(disp.astype(np.int)), MPI.UNSIGNED_LONG])
    #print('Rank :', rank, 'IDs_all_ranks', IDs_all_ranks, flush=True)
       
    f.close()
    return IDs_all_ranks

def get_part_coord_parallel(MPI, cluster_ind, basepath, fnames, coord_dir, savedir, PartType=5, boxsize=205000, h=0.6774, Hz=247.26097145, axis=2, Nmesh=205):
    """ (FOR TNG) Records the IDs and positions of the particles around center of z = 0 halos, the particles 
          which also exist at higher z. Both arrays are saved. 
        - Loops over all snapshot files and find the common particle amonf z=0 cluster and that snapshot.
          Then, distributes the common particles among ranks to calculate the position of the partciles by adding
          their peculiar velocity into it. Each rank saves the coordinates for each file on a seperate file. The output
          can be loaded with HDFCatalog in nbodykit. 
        - astropy breaks sometimes, so h and Hz are passed as arguments in this method 
        
        Arguments :
        - MPI : mpi4py.MPI
        - cluster_ind : The index to the cluster in the GroupCatalog
        - basepath : path to the directory containing the snapshots
        - fnames : A list of snapshot files
        - coord_dir : The directory to write some temporary files on
        - h and Hz : 100*h and Hz are Hubble parameters at z=0 and z=z
        - axis : The line of sight axis, the default is the xis=2
        - Nmesh : Number of mesh cells, total =  (Nmesh)**3
        
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
      print('Finding the central part_IDs', flush=True)
    # Get the particle IDs belonging to the cluster
    central_part_IDs = get_center_part_IDs(MPI=MPI, basepath=basepath, PartType=PartType, cluster_ind=cluster_ind)
    if rank ==0:
        print('Central_prt_IDs found!', flush=True)
    #print('Rnak ', rank, 'central part IDs size ', central_part_IDs.size)
    # To save coordinates of paticles found on each rank :
    x0,x1,x2 = np.array([]), np.array([]), np.array([])
    for fc, fn in enumerate(fnames):
        if rank==0:
           print(str(int(100*((fc+1)/len(fnames))))+'%', flush=True )
        # Particle IDs at higher z on rank = rank
        try : 
           f = h5py.File(fn, 'r')
        except OSError:
           print('Snapshot is ', fn, 'and is not loaded')
           raise 
        IDs_highz = f['PartType'+str(PartType)]['ParticleIDs'][:]
        # Common IDs between z= 0 and the high z 
        _ ,ind_common_set, _ = np.intersect1d(IDs_highz, central_part_IDs, return_indices=True)

        ind_common_set = np.sort(ind_common_set)
        # All coordinates
        z = f['Header'].attrs['Redshift']
        # The first few snapshots contain most of the DM progenitors, so we break the list of particles
        # break data to 'size' parts
        part_per_rank = int(ind_common_set.shape[0]/size)
       
        #  Spread particles among ranks
        if rank==size-1:
           ind_common_set = ind_common_set[rank*part_per_rank:-1]
        else:
           ind_common_set = ind_common_set[rank*part_per_rank:(rank+1)*part_per_rank]
        
        if ind_common_set.size==0:
           #print('Rank ', rank, 'fn ', fn, ' len(ind_comment_set) = 0', flush=True)
           continue
        else :
           print('Rank ', rank, 'common indices on file ', fn, flush=True)
 
        ind_common_set = np.sort(np.unique(ind_common_set))
        ind_common_set = list(ind_common_set)
 
        common_coords =  f['PartType'+str(PartType)]['Coordinates'][ind_common_set]
        assert type(common_coords[0,0])== np.float32
        # Add peculiar velocity along the line of sight in units of kpc/h
        common_coords[:,axis] = common_coords[:,axis] + (1000*(f['PartType'+str(PartType)]['Velocities'][ind_common_set, int(axis)]/(Hz*(1+z)**0.5))*h)
        commob_coords = (common_coords[:]%boxsize).astype(np.float32)
        assert type(common_coords[0,0]) is np.float32
        assert np.shape(common_coords)==(int(common_coords.size/3), 3)
        #print('fn :', fn, 'common_coords.size', common_coords.size)
        assert type(cluster_ind) is not list
        assert type(fn) is not list

        x0 = np.append(x0, common_coords[:,0])
        x1 = np.append(x1, common_coords[:,1])
        x2 = np.append(x2, common_coords[:,2])
    
    x0all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm, data=x0, data_type=np.float64)
    comm.Barrier()
    x1all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm, data=x1, data_type=np.float64)
    comm.Barrier()
    x2all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm, data=x2, data_type=np.float64)
    comm.Barrier()
    del x0
    del x1
    del x2
 
    if rank == 0:
        common_coords = np.zeros(shape=(x0all.size,3))
        common_coords[:,0] = x0all
        del x0all
        common_coords[:,1] = x1all
        del x1all
        common_coords[:,2] = x2all
        del x2all
        with h5py.File(os.path.join(coord_dir, 'prog_coords_cluster'+str(cluster_ind)+'.hdf5'),'w') as fw:
           # In cKpc/h
           fw['PartType1/Position'] = common_coords[:]
    comm.Barrier()
    get_density_map(cluster_ind=cluster_ind, coord_dir=coord_dir, savedir=savedir, Nmesh=Nmesh, boxsize=boxsize)
    comm.Barrier()
    if rank==0 :
        os.system('rm -rf '+os.path.join(coord_dir,'prog_coords_cluster'+str(cluster_ind)+'.hdf5'))

def get_density_map(cluster_ind, coord_dir, savedir, Nmesh=205, boxsize=205000):
    """
    Reads the progenitor DM coordinates have been written on file earlier via progenitor_particles()
    - This method is using MPI feature in nbodykit
    - coord_dir : The directory in which the coordinates are saved
    - savedir : The directory to save the full density map
    - boxsize : in cKpc/h
    """
    # Load the packages

    from nbodykit.lab import HDFCatalog
    import nbodykit as nk
    comm = nk.CurrentMPIComm.get()

    if not os.path.isdir(coord_dir):
       raise FileNotFoundError('Directory '+coord_dir+' does not exist!')

    cat = HDFCatalog(os.path.join(coord_dir,'prog_coords_cluster'+str(cluster_ind)+'.hdf5'), dataset='PartType1')
    #print('Rank ', comm.rank, ' cat.size: ', cat.size)
    #cat.attrs['boxsize']=205000
    # Peculiar velocity has already been taken into acount
    mesh =  cat.to_mesh(Nmesh=Nmesh, BoxSize=boxsize)
    density_rank = mesh.compute()
    density_full = np.zeros(shape=(Nmesh,Nmesh,Nmesh), dtype=np.float32)
    density_full[density_rank.slices] = density_rank[:]
    del density_rank
    # Make sure the data is contiguous in memory
    density_full =  np.ascontiguousarray(density_full, np.float32)
    # Make the full density map
    comm.Allreduce(nk.MPI.IN_PLACE, density_full, op=nk.MPI.SUM)
    if comm.rank == 0:
       with h5py.File(os.path.join(savedir,'map_PC_prog'+str(cluster_ind)+'.hdf5'),'w') as fw:
            fw.create_dataset('DM', data=density_full)
            fw.create_dataset('num_parts', data=[cat.size,])
