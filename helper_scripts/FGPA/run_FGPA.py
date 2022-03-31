"""A helper function to run different methods in fgpa.py
   In this version, you need to set number of the MPI 
   ranks such that you are not low on memory. On the other hand
   too many ranks will mess up with get_noiseless_map() since
   some ranks would not overlap with any sightline.
   There are ways to improve this if needded later."""
import argparse
import lytomo_watershed

def get_density_field(snaps, savedir, Nmesh, sim_type, boxsize, z):
    """First you need to make a density map on a grid with average
        one particle per voxel. 
    Arguments:
        snaps: The path to DM-only sanapshot files, e.g ../snapshots/snap_028*.hdf5, notice the *
        savedir: The path to directory to save the density files in
        Nmesh: Number of mesh voxles along each axis, the density map would be of shape 
        (Nmesh, Nmesh, Nmesh)"""
    from lytomo_watershed.density import Density
    dens = Density(snaps=snaps, savedir=savedir, Nmesh=Nmesh, zspace=False, momentumz=True, sim_type=sim_type, z=z, boxsize=boxsize)
    dens.Gadget()

def get_noiseless_map(z, savedir, savefile, boxsize=205, Ngrids=410, Npix=1780):
    """Get the noiseless flux FGPA map.
    Argumentrs:
        z: redshift
        savedir: The path to directory to save the density files in
        savfile: The path and name of the file you want to store the result in
        boxsize: The boxsize in cMpc/h
        Ngrids: Number of grids required along the transverse direction
        Npix : Number of pixels along the line of sight
    """
    from lytomo_watershed.fgpa import Fgpa
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    fgpa = Fgpa(MPI=MPI, comm=comm, z=z, savedir=savedir, boxsize=boxsize, Ngrids=Ngrids,
                Npix=Npix, SmLD=1, SmLV=1, fix_mean_flux=True)
    fgpa.get_noiseless_map(savefile=savefile)
    
def iterate_over_SmLV():
    """Not being used regularly"""
    from lytomo_wtaershed import fgpa
    from mpi4py import MPI
    SmLD = int(1)
    comm = MPI.COMM_WORLD
    for SmLV in range(1,15,1):
        fgpa.get_sample_spectra(MPI, z=2.4442257045541464, num= 2500, savedir='./Fix_Vel_Smoothing/',
                                savefile='./spectra_FGPA_z2.4_50by50_SmLD'+str(SmLD)+'_SmLV'+str(SmLV)+'.hdf5',
                                Ngrids=50 ,Npix=2460, SmLD=SmLD, SmLV=SmLV, seed=None)
        comm.Barrier()

def iterate_over_SmLD():
    """Not being used regularly"""
    from lytomo_watershed import fgpa
    from mpi4py import MPI
    SmLV = int(1)
    comm = MPI.COMM_WORLD
    for SmLD in range(15,30,2):
        fgpa.get_sample_spectra(MPI, z=2.4442257045541464, num= 2500, savedir='./DM_only/', 
                                savefile='./spectra_FGPA_z2.4_50by50_SmLD'+str(SmLD)+'_SmLV'+str(SmLV)+'.hdf5',
                                Ngrids=50 ,Npix=2460, SmLD=SmLD, SmLV=SmLV, seed=None)
        comm.Barrier()


if __name__ == '__main__':
    """ Uncomment the function you may want to use """
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=True)
    parser.add_argument('--snaps', type=str, required=False, default=None, help='adress to snapshots in globe pattern, like "./snap_0*"')
    parser.add_argument('--Nmesh', type=int, required=False, default=None, help='Number of mesh cells along each axis, the density field will have the shape of Nmesh*Nmesh*Nmesh')
    parser.add_argument('--simtype', type=str, required=False, default='Gadget', help='Either "Gadget" or "Gadget_old"')
    parser.add_argument('--z', type=float, required=False, default=None, help='Accurate redshift')
    parser.add_argument('--savedir', type=str, required=True, help='the dir to save the results of each rank in')
    parser.add_argument('--savefile', type=str, required=False, help='The file name to save the full FGPA map')
    parser.add_argument('--boxsize', type=int, required=False, default=None, help='boxsize in cMpc/h, only if simtype="Gadget_old"')
    parser.add_argument('--Ngrids', type=int, required=False, help='We will get Ngrids*Ngrids spectra streched along z')
    parser.add_argument('--Npix', type=int, required=False, help='number of pxiels in each spectra')

    
    args = parser.parse_args()
    
    if args.func == 'density':
        get_density_field(snaps= args.snaps, savedir= args.savedir, Nmesh= args.Nmesh, sim_type= args.simtype, z=args.z, boxsize=args.boxsize)
    if args.func == 'map':
        get_noiseless_map(z=args.z, savedir=args.savedir, savefile=args.savefile,
                          boxsize=args.boxsize, Ngrids=args.Ngrids, Npix=args.Npix)
    #iterate_over_SmLV()
    #iterate_over_SmLD()
    
  

