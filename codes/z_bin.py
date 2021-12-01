import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units
def best_z_bin():
    """ NOT WORKING
    Finds the best binnng of redshift (the one with reduced chi^2 closest to o1)
    returns the best z binning

    """

    deltaF_all = latis.get_deltaF_spec()
    deltaF = deltaF_all[1]
    chi_2_array = []
    for i in range(2,7):

        z_bins= []
        
        dz = 0.6/(1.0*i)

        bins = np.arange(2.2,2.8001, dz)
        z_bins = np.array([(bins[k]+bins[k+1] )/2. for k in range(0,np.size(bins)-1)])

        r_bins = radial_distance(z_bins) 
        #ind2 = np.where(rbins < )


        
        chi_2 = []
        chi_2.append([ (deltaF[indices[k]] - deltaF[indices[k+1]])**2 for k in range(np.size(z_bins)-1)])
        chi_2 = np.array(chi_2)
        chi_2 = np.sum(chi_2)/(np.size(z_bins))

        chi_2_array.append(chi_2)

    #return (np.max(chi_2_array), np.where(chi_2_array == np.max(chi_2_array)))
    return chi_2_array

def best_z_bin_3D():
    """ NOT WORKING
    The same method as above except working on 3D map instead of spectra"""
    m = np.fromfile('./spectra/maps/map_obs.dat').reshape((63, 51, 483))

    ## Check among different redshift bins


def get_pix_ind(z_bins):
    """ Takes an array of reshifts and returns the pixels' indicies in radial direction   """

    return np.around(radial_distance(z_bins)).astype(int)


def radial_distance(z_bins):
    """ takes an array of redshifts and returns the radial distcance in observation's box  ( unit : h^-1 cMpc ) """
    """
    distance = []
    for i in range(np.size(z_bins)):

        z_bin_integrand = np.arange(2.2,z_bins[i],0.001)
        integrand =  (3*10**5)/Hubble(z_bin_integrand)
        distance.append(np.trapz(integrand, z_bin_integrand))
    
    #return np.around(distance-distance[0], decimals=0)
    return np.array(distance)
    """
    return np.array(cosmo.comoving_distance(z_bins) - cosmo.comoving_distance(2.2))* cosmo.h

def cmpch_to_redshift(d, z0=2.2):
    """
    d: an array of line-of-sight coordinatis in cMpc/h
    z0 : the minimum redshift of the map
    Return : corresponding redshift
    """
    zarr = np.linspace(2.2,3.00,100)
    zcoords = (cosmo.comoving_distance(zarr) - cosmo.comoving_distance(2.2)).to(units.Mpc).value * cosmo.h
    redshift = np.interp(d, zcoords, zarr)

    return redshift


def cMpc_to_velocity(z):
    ''' Converts  1 cMpch^-1 to velocity at reshift z, being used for resolution of the spectra '''

    return cosmo.H(z)/((1+z)*cosmo.h)

