import numpy as np
from  mpi4py import MPI
import mpi4py_helper

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Test Algatherv_helper()
for data_type in [np.float32, np.uint]:
    data = rank*np.ones(shape=(rank,), dtype=data_type)
    data_all_ranks = mpi4py_helper.Allgatherv_helper(MPI=MPI, comm=comm, data=data, data_type=data_type)
    true_answer = np.array([])
    for i in range(size) :
        true_answer = np.append(true_answer, i*np.ones(shape=(i,), dtype=data_type))
    assert np.all(data_all_ranks==true_answer)

## Test distribute_array()
data = np.arange(10)
DataRank = mpi4py_helper.distribute_array(MPI, comm, data)
print('rank ', rank, ' DataRank ', DataRank)


