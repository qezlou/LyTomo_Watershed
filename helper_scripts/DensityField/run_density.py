import argparse
from lytomo_watershed.density import Density

def runit(snaps, savedir, savefile, Nmesh):
    
    dens = Density(snaps=snaps, savedir=savedir, savefile=savefile, Nmesh=Nmesh, zspace=True, momentumz=True)
    dens.Gadget()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--snaps', type=str, required=True, help='adress to snapshots in globe pattern, like "./snap_0*"')
    parser.add_argument('--savedir', type=str, required=True, help='the dir to save the results of each rank in')
    parser.add_argument('--savefile', type=str, required=True, help='The file name to save the full density map')
    parser.add_argument('--Nmesh', type=int, required=True, help='Number of mesh cells along each axis')
    
    args = parser.parse_args()
    runit(snaps=args.snaps, savedir=args.savedir, savefile=args.savefile, Nmesh=args.Nmesh)
      


