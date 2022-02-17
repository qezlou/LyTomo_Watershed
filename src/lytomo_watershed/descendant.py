#  A Script to find any PartType/FOF_halo descendents of the massive halos within flux contours < thresh
import numpy as np
import h5py
import glob
import os
import time
from . import illustris_python as il
from . import mpi4py_helper


def parts_within_halos( PartType, basepath):
    """ takes halos and returns PartType particles withing them  """
    f = h5py.File('./halos/massive_halos.hdf5','r')
    countour_id = []
    for i in range(f['HaloID'][:].size): 
        parts = il.snapshot.loadHalo(basepath, snapNum=30, id=f['HaloID'][i], partType=PartType,fields = ['ParticleIDs','Coordinates'])
        if i == 0 :
           part_pos = parts['Coordinates'][:]
           part_id = parts['ParticleIDs'][:]
           halo_id = f['HaloID'][i]*np.ones_like(parts['ParticleIDs'][:])
           contour_id = f['ContourID'][i]*np.ones_like(parts['ParticleIDs'][:])
        else :
           part_pos = np.append(part_pos, parts['Coordinates'][:], axis=0)
           part_id = np.append(part_id, parts['ParticleIDs'][:])
           halo_id = np.append(halo_id, f['HaloID'][i]*np.ones_like(parts['ParticleIDs'][:]))
           contour_id = np.append(contour_id, f['ContourID'][i]*np.ones_like(parts['ParticleIDs'][:]))
    return part_id, part_pos, contour_id, halo_id 
        

def find_corres_halo(PartType, basepath):
    """ I do not remember the funcionality of this method"""
    part_id, part_pos, contour_id, halo_id = parts_within_halos(PartType)
    halos = il.groupcat.loadHalos(basepath, 98, fields=['GroupMass', 'SubhaloGrNr','Group_R_Crit200'])
    massive_ind = np.where(halos['GroupMass'] >= 100)[0]
    desc_halo_mass = np.array([])
    contour_id_f = np.array([])
    for i in range(massive_ind.size):
       #print(i)
       parts = il.snapshot.loadHalo(basepath, snapNum=98, id=massive_ind[i], partType=PartType,fields = ['ParticleIDs','Coordinates'])
       # Ther might be no descendants in that specific halo at z =0
       if parts['count']== 0: continue
       ind1 = np.where(np.in1d(parts['ParticleIDs'], part_id))[0]
       ind2 = np.where(np.in1d(part_id, parts['ParticleIDs']))[0]
       desc_halo_mass = np.append(desc_halo_mass, halos['GroupMass'][massive_ind[i]])
       contour_id_f = np.append(contour_id_f, contour_id[ind2])

    #for i in range(part_id.size):
        #ind = np.less_equal(np.sqrt(np.sum((halos['SubhaloGrNr'][:] - part_pos[i])**2, axis=1 )), halos['Group_R_Crit200'][:] )
    return desc_halo_mass, contour_id_f, halo_id
    
def find_roots(MPI, snap, savefile, halos_file, subhalos, basepath):
    """Finds the root descendant subhaloInd, it's mass and the corresponding FOF Group mass of the sub/halos file from z ~2.5 passed to it
       An MPI code.
       halos_file : A list of all halos or subhalos at snapshot = snap
       subhalos : Boolean. True if the halos_file are subhalos, False if they are halos
       savefile : The output file name
    """
    # MPI initializations
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    num_ranks = int(comm.Get_size())
    tss = time.asctime()
    print('\n Rank =', rank, 'started!', tss, flush=True)

    
    f = h5py.File(halos_file,'r')
    # This ensures the sub/halos are sorted based on PeakIDs
    indsort_all = np.argsort(f['PeakIDs'][:])
    
    if not subhalos :
        # Find the most massive subhalo within each FOF halo to trace the descendant
        GroupFirstSub = il.groupcat.loadHalos(basepath,snap,fields=['GroupFirstSub'])

    # Distributing the sub/halos among ranks
    indsort = mpi4py_helper.distribute_array(MPI, comm, indsort_all)
    if subhalos :
        halos = f['SubhaloInd'][:][indsort]
    else :
        halos = GroupFirstSub[:][f['HaloID'][:][indsort]]
        
    not_on_tree = np.array([])
    percent_old = []# Just a counter to print progress
    roots, root_mass, PeakIDs, halo_mass, desc_subhalo_mass, SnapNum, SubhaloGrNr = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    for i in range(halos.size):
        #print the progress
        if rank==0 :
            percent =  np.round(i/halos.size, 1)*100
            if percent not in percent_old:
                percent_old.append(percent)
                print(percent, ' percent is done!', flush=True)
        
        tree = il.sublink.loadTree(basepath,snap,halos[i],fields=['SnapNum', 'SubhaloID','GroupMass', 'SubhaloGrNr', 'RootDescendantID','SubhaloMass'],onlyMDB=True)
        # This try and except deals with the cases, the subhalo at high-z 
        # is not on the tree
        try :
            if np.size(tree['GroupMass']) == 0:
                not_on_tree = np.append(not_on_tree, halos[i])
                continue
        except TypeError:
            continue
            
        root_mass = np.append(root_mass, tree['GroupMass'][0])
        roots = np.append(roots, tree['RootDescendantID'][0])
        SnapNum = np.append(SnapNum, tree['SnapNum'][0])
        SubhaloGrNr = np.append(SubhaloGrNr, tree['SubhaloGrNr'][0])
        desc_subhalo_mass = np.append(desc_subhalo_mass, tree['SubhaloMass'][0])
        PeakIDs = np.append(PeakIDs, f['PeakIDs'][:][indsort][i])
        halo_mass = np.append(halo_mass, f['Mass'][:][indsort][i])
    
    # Add the results of all ranks
    comm.barrier()
    root_mass = mpi4py_helper.Allgatherv_helper(MPI, comm, data=root_mass, data_type=type(root_mass[0]))
    roots = mpi4py_helper.Allgatherv_helper(MPI, comm, data=roots, data_type=type(roots[0]))
    SnapNum = mpi4py_helper.Allgatherv_helper(MPI, comm, data=SnapNum, data_type=type(SnapNum[0]))
    SubhaloGrNr = mpi4py_helper.Allgatherv_helper(MPI, comm, data=SubhaloGrNr, data_type=type(SubhaloGrNr[0]))
    desc_subhalo_mass = mpi4py_helper.Allgatherv_helper(MPI, comm, data=desc_subhalo_mass, data_type=type(desc_subhalo_mass[0]))
    PeakIDs = mpi4py_helper.Allgatherv_helper(MPI, comm, data=PeakIDs, data_type=type(PeakIDs[0]))
    halo_mass = mpi4py_helper.Allgatherv_helper(MPI, comm, data=halo_mass, data_type=type(halo_mass[0]))
    
    if rank==0:
        with h5py.File(savefile, 'w') as fw:
            #f_roots['Group_M_Crit200'] = root_mass
            fw['GroupMass'] = root_mass
            fw['desc_subhalo_mass'] = desc_subhalo_mass
            fw['SnapNum'] = SnapNum
            fw['SubhaloGrNr'] = SubhaloGrNr
            fw['RootDescendantID'] = roots
            fw['PeakIDs'] = PeakIDs
            fw['halo_mass'] = halo_mass
            fw['not_on_tree'] = not_on_tree
    comm.barrier()
    tss = time.asctime()
    print('\n Rank =', rank, 'Finished!', tss, flush=True)

def find_most_massive_roots(savefile=None, rootfile='./halos/RootDescendants_massive_halos.hdf5'):
    """Reads the root descendant file and in each contour,  picks the one within the most massive FOF halo  """
    f = h5py.File(rootfile, 'r')
    mm_id = np.array([])
    mm_mass = np.array([])
    PeakIDs = np.array(list(set(f['PeakIDs'][:])))
    for i in PeakIDs:
        ind = np.where(f['ContourID'][:]==i)
        mm_ind = np.where(f['GroupMass'][ind]== np.max(f['GroupMass'][ind]))[0][0]
        mm_mass = np.append(mm_mass, f['GroupMass'][ind][mm_ind])
        mm_id = np.append(mm_id, f['RootDescendantID'][ind][mm_ind])
    if savefile is not None :
        fsave = h5py.File(savefile,'w')
        fsave['GroupMass'] = mm_mass
        fsave['RootDescendantID'] = mm_id
        fsave['PeakIDs'] = PeakIDs
    else :
        return mm_mass,  PeakIDs, mm_id

def find_voted_subhalo_roots(savefile=None, rootfile='./halos/RootDescendants_massive_halos.hdf5', snap=30):
    """Reads the root descendants made with find_roots_subhalos() and weighs each root by the mass of the sub/halos at z ~2.5. The root with highest score
       within each volume is being selected """
    f = h5py.File(rootfile, 'r')
    PeakIDs = np.unique(f['PeakIDS'][:])
    winner_roots, GroupMass, mass_ratio, desc_subhalo_mass, SnapNum, SubhaloGrNr = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for p in PeakIDs:
        indp = np.where(f['PeakIDs'][:]==p)
        roots = np.unique(f['RootDescendantID'][:][indp])
        scores = np.array([])
        for r in roots:
            indr = np.where(f['RootDescendantID'][:][indp]==r)
            scores = np.append(scores, np.sum(f['halo_mass'][:][indp][indr]))
        winner_roots = np.append(winner_roots, roots[np.argmax(scores)])
        indw = np.where(f['RootDescendantID'][:][indp]==winner_roots[-1])
        GroupMass = np.append(GroupMass,f['GroupMass'][:][indp][indw][0])
        SnapNum = np.append(SnapNum, f['SnapNum'][:][indp][indw][0])
        SubhaloGrNr = np.append(SubhaloGrNr, f['SubhaloGrNr'][:][indp][indw][0])
        desc_subhalo_mass = np.append(desc_subhalo_mass, f['desc_subhalo_mass'][:][indp][indw][0])
        mass_ratio = np.append(mass_ratio, np.max(scores)/np.sum(scores))
    if savefile is not None :
        fsave = h5py.File(savefile,'w')
        fsave['GroupMass'] = GroupMass
        fsave['SnapNum'] = SnapNum
        fsave['SubhaloGrNr'] = SubhaloGrNr
        fsave['desc_subhalo_mass'] = desc_subhalo_mass
        fsave['RootDescendantID'] = winner_roots
        fsave['MassRatio'] = mass_ratio

def find_voted_FOF_roots(MPI, savefile=None, rootfile='./halos/RootDescendants_massive_halos.hdf5', snap=29):
    """Reads the root descendants made with find_roots_subhalos() and weighs each FOF Group of the subhalo Root descendants
       by the mass of the sub/halos at z ~2.5. The root with highest score
       within each volume is being selected """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    f = h5py.File(rootfile, 'r')
    PeakIDs = np.unique(f['PeakIDs'][:])
    # To balcance load on ranks, randonly distribute the subregions among them
    rand = np.arange(0, PeakIDs.size)
    np.random.seed(56)
    np.random.shuffle(rand)
    
    PeakIDsRank = mpi4py_helper.distribute_array(MPI, comm, PeakIDs[rand]).astype(int)
    
    
    WinnerGroups = np.zeros(shape=(PeakIDs.size,), dtype=np.int32)
    GroupMass = np.zeros(shape=(PeakIDs.size,), dtype=np.float32)
    MassRatio = np.zeros(shape=(PeakIDs.size,), dtype=np.float32)
    SnapNum =  np.zeros(shape=(PeakIDs.size,), dtype=np.int16) 
    
    print('rank ', comm.Get_rank(), 'PeakIDsRank : ', PeakIDsRank, flush=True)

    percent_old = []# Just a counter to print progress
    
    for c, p in enumerate(PeakIDsRank):
        if rank == 0:
            percent = np.round(c/PeakIDsRank.size, 1)*100
            if percent not in percent_old:
                percent_old.append(percent)
                print(percent, ' % is done! ', flush=True)
            
        indp = np.where(f['PeakIDs'][:]==p)
        groups = np.unique(f['SubhaloGrNr'][:][indp])
        scores = np.array([])
        
        for g in groups:
            indg = np.where(f['SubhaloGrNr'][:][indp]==g)
            scores = np.append(scores, np.sum(f['halo_mass'][:][indp][indg]))
        
        WinnerGroups[p-1] = groups[np.argmax(scores)]
        
        indw = np.where(f['SubhaloGrNr'][:][indp]==WinnerGroups[p-1])
        GroupMass[p-1] = f['GroupMass'][:][indp][indw][0]
        SnapNum[p-1] = f['SnapNum'][:][indp][indw][0]
        MassRatio[p-1] = np.max(scores)/np.sum(scores)
    
    WinnerGroups = np.ascontiguousarray(WinnerGroups, np.int32)
    GroupMass = np.ascontiguousarray(GroupMass, np.float32)
    SnapNum = np.ascontiguousarray(SnapNum, np.int16)
    MassRatio = np.ascontiguousarray(MassRatio, np.float32)
    #PeakIDsRank = np.ascontiguousarray(PeakIDsRank, np.int)
    
    comm.Allreduce(MPI.IN_PLACE, WinnerGroups, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, GroupMass, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, SnapNum, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, MassRatio, op=MPI.SUM)
    #PeakIDsRank = mpi4py_helper.Allgatherv_helper(MPI, comm, PeakIDsRank, int)
    
    if (savefile is not None) and (comm.Get_rank()==0) :
        fsave = h5py.File(savefile,'w')
        fsave['GroupMass'] = GroupMass
        fsave['SnapNum'] = SnapNum
        fsave['winner_groups'] = WinnerGroups
        fsave['score'] = MassRatio
        fsave['peak_id'] = PeakIDs
    
    comm.barrier()


def sum_all_roots(savefile=None, rootfile='./halos/RootDescendant_massive_halos.hdf5'):
    """ Reads the root descendant file and in each contour, sums over all FOF halos of root descendants """
    f = h5py.File(rootfile, 'r')
    tot_mass = np.array([])
    PeakIDs = np.array(list(set(f['PeakIDs'][:])))
    for i in PeakIDs:
        ind = np.where(f['ContourID'][:]==i)
        #_, unique_root_ind = np.unique(f['RootDescendantID'][ind], return_index=True)
        unique_halo_mass = np.unique(f['GroupMass'][ind])
        tot_mass = np.append(tot_mass, np.sum(unique_halo_mass))
    if savefile is not None:
        fsave = h5py.File(savefile,'w')
        fsave['GroupMass'] = tot_mass
        fsave['PeakIDsd'] = PeakIDs
    else:
        return tot_mass, PeakIDs

    

