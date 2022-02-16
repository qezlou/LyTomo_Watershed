""" modules being used either by progenitor_parallel or induvidually to find any type (Gas/DM/BH) particles 
    lied within center of the z~0 DM halos and write their cordinates at higher redhsift on files"""
import h5py
import numpy as np
import time
import os
from . import mpi4py_helper
from . import illustris_python as il

def get_clusters_low_z(min_mass = 10**4, basepath='/lustre/scratch/mqezlou/TNG300-1/output'):
    """Script to write the position of z ~ 0 large mass halos on file """ 
    halos = il.groupcat.loadHalos(basepath,  98, fields=['GroupMass', 'GroupPos','Group_R_Crit200'])
    ind = np.where(halos['GroupMass'][:] > min_mass)
    with h5py.File('clusters_TNG300-1.hdf5','w') as f :
        f['Mass'] = halos['GroupMass'][ind]
        f['Group_R_Crit200'] = halos['Group_R_Crit200'][ind]
        f['x'], f['y'], f['z'] = halos['GroupPos'][ind[0],0], halos['GroupPos'][ind[0],1], halos['GroupPos'][ind[0],2]
        f.close()
    return 0

def get_center_part_IDs(MPI, basepath, PartType='BH', cluster_ind=0) :
    """ Returns the IDs of the partcles within Group_R_Critic200 of the center of a halo at z= 0
        MPI : MPI communicator from mpi4py
        PartType : PartType of the particles of interest, 1 for DM
        cluster_ind : the index to the group catalouge list for the halo of interest
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
    comm.Allgatherv(IDs, [IDs_all_ranks, tuple(IDs_size.astype(np.int)), tuple(disp.astype(np.int)), MPI.UNSIGNED_LONG])
       
    f.close()
    return IDs_all_ranks

def get_part_coord_parallel(MPI, cluster_ind, basepath, fnames, coord_dir, savedir, PartType=5,
                            boxsize=205000, h=0.6774, Hz=247.26097145, axis=2, Nmesh=205):
    """ (FOR TNG) Traces back the particles around center of z = 0 halos to z=2.5. 
        - Loops over all snapshot files and finds the common particle among z=0 cluster
          and that snapshot at z=2.5. Then, distributes the common particles among ranks
          to calculate their paositions by adding their peculiar velocity into it. 
        - Then `get_density_map()` is called to make a density map of the progenitor 
          particles. The temporary files in coord_dir are deleted at the end.
        - `astropy` breaks sometimes, so h and Hz are passed as arguments in this method 
        
        Arguments :
        - MPI : mpi4py.MPI
        - cluster_ind : The index to the cluster of interest in the GroupCatalog
        - basepath : path to the directory containing the simulation snapshots
        - fnames : A list of snapshot files
        - coord_dir : The directory to write some temp
        - h and Hz : 100*h and Hz are Hubble parameters at z=0 and z=z
        - axis : The line of sight axis, the default is the xis=2 (z-axis)
        - Nmesh : Number of mesh cells, total =  (Nmesh)**3
        - savedir : directory to save final DM density map of the desired progenitor
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
        print('Finding the central part_IDs', flush=True)
    # Get the particle IDs belonging to the cluster
    central_part_IDs = get_center_part_IDs(MPI=MPI, basepath=basepath, 
                                           PartType=PartType, cluster_ind=cluster_ind)
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
        common_coords[:,axis] = common_coords[:,axis] + (1000*(f['PartType'
                                                                 +str(PartType)]['Velocities'][ind_common_set,
                                                                                               int(axis)]/(Hz*(1+z)**0.5))*h)
        commob_coords = (common_coords[:]%boxsize).astype(np.float32)
        assert type(common_coords[0,0]) is np.float32
        assert np.shape(common_coords)==(int(common_coords.size/3), 3)
        #print('fn :', fn, 'common_coords.size', common_coords.size)
        assert type(cluster_ind) is not list
        assert type(fn) is not list

        x0 = np.append(x0, common_coords[:,0])
        x1 = np.append(x1, common_coords[:,1])
        x2 = np.append(x2, common_coords[:,2])
    
    x0all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm,
                                            data=x0, data_type=np.float64)
    comm.Barrier()
    x1all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm,
                                            data=x1, data_type=np.float64)
    comm.Barrier()
    x2all = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm,
                                            data=x2, data_type=np.float64)
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
        with h5py.File(os.path.join(coord_dir,
                                    'prog_coords_cluster'
                                    +str(cluster_ind)
                                    +'.hdf5'),'w') as fw:
           # In cKpc/h
           fw['PartType1/Position'] = common_coords[:]
    comm.Barrier()
    get_density_map(cluster_ind=cluster_ind, coord_dir=coord_dir, savedir=savedir, Nmesh=Nmesh, boxsize=boxsize)
    comm.Barrier()
    if rank==0 :
        os.system('rm -rf '+os.path.join(coord_dir,'prog_coords_cluster'+str(cluster_ind)+'.hdf5'))

def get_density_map(cluster_ind, coord_dir, savedir, Nmesh=205, boxsize=205000):
    """
    Reads the progenitor DM coordinates have been written on file earlier via 
    progenitor_particles()
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

def get_cofm(savefile='./LyTomo_data/progenitors/cofm_progenitors.hdf5', L=205):
    """ A code to find the Center of Mass of each individual progenitor """
    from scipy.ndimage import label
    X, Y, Z, cluster_id = np.array([]), np.array([]), np.array([]),np.array([])
    with h5py.File('./clusters_TNG300-1.hdf5','r') as f:
        ind = np.where((f['Mass'][:]>10**3.75) * (f['Mass'][:]<10**4.0))[0]
  # interate over cluster progenitors
    a = np.append(np.arange(248),ind[1::])
    for j in a:
        new_num_features = 0
        with h5py.File('./prog_maps/map_PC_prog_R200_cluster'+str(j)+'.hdf5','r') as f:
            m = (f['map'][:]*f['num_parts'][()]/(205**3))
        labeled_array, num_features = label(m[:] > 0 )
        xcm, ycm, zcm, num_parts = np.array([]), np.array([]), np.array([]), np.array([])
        # interate over all islands a cluster progenitors is spread over (Peridoc boundary condition)
        # breaks cluster progenitor into pieces
        for i in range(num_features):
            indp = np.where(labeled_array==i+1)
            num_parts= np.append(num_parts, np.sum(m[indp]))
            xcm= np.append(xcm, np.sum(indp[0]*m[indp])/num_parts[-1])
            ycm= np.append(ycm, np.sum(indp[1]*m[indp])/num_parts[-1])
            zcm= np.append(zcm, np.sum(indp[2]*m[indp])/num_parts[-1])
        # Take care of periodic boundary condition to get Center Of Mass
        if num_features > 1:
            if np.any(xcm > L/2)*np.any(xcm < L/2):
                ind = np.where(xcm < L/2)
                xcm[ind] += L
            if np.any(ycm > L/2)*np.any(ycm < L/2):
                ind = np.where(ycm < L/2)
                ycm[ind] += L
            if np.any(zcm > L/2)*np.any(zcm < L/2):
                ind = np.where(zcm < L/2)
                zcm[ind] += L
        # Store the center of mass for entire inividual cluster progenitors
        X = np.append(X, (np.sum(xcm*num_parts)/np.sum(num_parts))%L)
        Y = np.append(Y, (np.sum(ycm*num_parts)/np.sum(num_parts))%L)
        Z = np.append(Z, (np.sum(zcm*num_parts)/np.sum(num_parts))%L)
        cluster_id = np.append(cluster_id, j)
    with h5py.File(savefile, 'w') as fw:
        fw['x'] = X
        fw['y'] = Y
        fw['z'] = Z
        fw['cluster_id'] = cluster_id

def make_full_prog_map(save=False, force_compute=False):
    """Create a map containing all progenitors"""
    import glob
    import os
    data_dir = '/run/media/mahdi/HD2/Lya/LyTomo_data/'
    savefile = os.path.join(data_dir,'progenitor_maps/Full_prog_map.hdf5')
    ### Do not calculate it, if you already have it
    if os.path.exists(savefile) and not force_compute:
        return h5py.File(savefile,'r')['DM'][:]
    else :    
        map_pc_prog_files = os.path.join(data_dir,'progenitor_maps/map_PC_prog*.hdf5')
        fnames = glob.glob(map_pc_prog_files)
        DM_full = np.zeros(shape=(205,205,205))
        for fn in fnames:
            with h5py.File(fn,'r') as f:
                DM_prog = f['DM'][:]
                num_parts = f['num_parts'][()]
                # There is a hack here, number of progenitors on each file is only the number on rank=0
                # So, we need to correct that by multiplying ti by number of ranks used in generating the 
                # progenitor maps, It is easy solve but since it is only used for visualization, 
                #the accuracy is good enough. There is an issue raised on GitHub to solve this later.
                DM_prog *=  (48*num_parts) / (2500**3)
                DM_full += DM_prog
        if save:
            with h5py.File(savefile,'w') as fw:
                fw['DM'] = DM_full     
        return DM_full