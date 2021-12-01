import LATIS.codes.DensityField.get_density_field as get_density_field
import argparse
def runit(z, snap):
    get_density_field.TNG(snaps='/lustre/scratch/mqezlou/TNG300-1/output/snapdir_0'+str(snap)+'/snap_0'+str(snap)+'.*', savedir='./results_'+str(z)+'/',Nmesh=205, zspace=True, momentumz=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--z', type=float, required=True, help='e.g. z=2.4')
    parser.add_argument('--snap', type=int, required=True, help='snap number')
    args = parser.parse_args()
    runit(z=args.z, snap=args.snap)
    


