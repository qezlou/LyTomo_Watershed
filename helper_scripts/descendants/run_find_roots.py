from mpi4py import MPI

from codes import descendant
import argparse

def run_it(savefile, halosfile):
   descendant.find_roots(MPI=MPI, snap=29, savefile= savefile, halos_file= halosfile, subhalos=False)


if __name__ == '__main__':
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--savefile', type=str, required=True)
   parser.add_argument('--halosfile', type=str, required=True)
   args = parser.parse_args()

   run_it(savefile= args.savefile, halosfile= args.halosfile)
   
