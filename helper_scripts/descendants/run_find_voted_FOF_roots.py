from mpi4py import MPI
import LATIS
from LATIS.codes import descendant
import argparse

def run_it(savefile, rootfile):
   descendant.find_voted_FOF_roots(MPI=MPI, savefile=savefile, rootfile=rootfile, snap=29) 

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--savefile', type=str, required=True)
   parser.add_argument('--rootfile', type=str, required=True)
   args = parser.parse_args()

   run_it(savefile= args.savefile, rootfile= args.rootfile)



