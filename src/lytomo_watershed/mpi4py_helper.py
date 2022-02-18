import numpy as np

def distribute_array(MPI, comm, data):
    """
    Distribute array "data" equally between ranks and return the laod
    for each rank individually.
    """
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    LoadRank = np.ones(shape=(size,), dtype=int)*int(data.size/size)
    remainder = int(data.size%size)
    if remainder != 0 :
        LoadRank[0:remainder] += 1
    if rank ==0 :
        DataRank = data[0:LoadRank[0]]
    else:
        start = np.sum(LoadRank[0:rank]).astype(int)
        DataRank = data[start:start+LoadRank[rank]]
    # The data for rank = comm.Get_rank()
    return DataRank


def Allgatherv_helper(MPI, comm, data, data_type):
    """
    Each rank should call this with data on that rank
    MPI : pass the mpi4py.MPI
    comm : The mpi communicator
    data : The 1D array on each rank. The size of data on each
    rank coul be different.
    data_type: Type of each elemnt of data array
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    data_size_ranks = np.zeros(shape=(size,), dtype=np.int)
    data_size_ranks[rank] = data.size
    data_size_ranks = np.ascontiguousarray(data_size_ranks, dtype=np.int)
    comm.Allreduce(MPI.IN_PLACE, data_size_ranks, op=MPI.SUM)
    comm.Barrier()

    data_all_ranks = np.empty(np.sum(data_size_ranks), dtype=data_type)
    disp = np.zeros(shape=(size,), dtype=np.int)
    for i in range(1, size):
        disp[i] = np.sum(data_size_ranks[0:i])
    if data_type == np.float32:
        mpi_type = MPI.FLOAT
    if data_type == np.float64:
        mpi_type = MPI.DOUBLE
    if data_type == np.uint64:
        mpi_type = MPI.UNSIGNED_LONG
    if data_type == np.int:
        mpi_type = MPI.LONG
        
    comm.Allgatherv(data, [data_all_ranks, tuple(data_size_ranks.astype(int)), tuple(disp.astype(np.int)), mpi_type])
    return data_all_ranks

def distribute_files(comm, fnames):
    """Distribute a list of files among available ranks
    comm : MPI communicator
    fnames : a list of file names
    Returns : A list of files for each rank
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_files = len(fnames)
    files_per_rank = int(num_files/size)
    #a list of file names for each rank
    fnames_rank = fnames[rank*files_per_rank : (rank+1)*files_per_rank]
    # Some ranks get 1 more snaphot file
    remained = int(num_files - files_per_rank*size)
    if rank in range(1,remained+1):
        fnames_rank.append(fnames[files_per_rank*size + rank-1 ])
    return fnames_rank