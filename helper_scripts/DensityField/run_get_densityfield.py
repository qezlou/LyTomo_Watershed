import lytomo_watershed
import argparse

def runit(snaps, savedir, Nmesh):
    from lytomo_watershed import get_density_field
    get_density_field.TNG(snaps=snaps, savedir=savedir,Nmesh=Nmesh, zspace=True, momentumz=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--snaps', type=str, required=True, help='adress to snapshots in globe pattern, like "./snap_0*"')
    parser.add_argument('--savedir', type=str, required=True, help='the dir to save the results of each rank in')
    parser.add_argument('--Nmesh', type=int, required=True, help='Number of mesh cells along each axis')
    
    args = parser.parse_args()
    runit(snaps=args.snaps, savedir=args.savedir, Nmesh=args.Nmesh)
      


