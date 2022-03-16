# Modified version of the code borrowed from Drew Newman
import numpy as np
import scipy 
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
import h5py

def boundary_condition_label(labeled_array):
    """A method to implement periodic boundary condition in a labeled map with scipy.ndimage.label()"""
    print('Periodic BC. Init num of countours: ', np.unique(labeled_array).size-1)
    for a in range(labeled_array.shape[0]):
        for b in range(labeled_array.shape[1]):
            if labeled_array[a,b,0]>0 and labeled_array[a,b,-1]>0 :
                labeled_array[labeled_array==labeled_array[a,b,-1]] = labeled_array[a,b,0]
    for a in range(labeled_array.shape[1]):
        for b in range(labeled_array.shape[2]):
            if labeled_array[0,a,b]>0 and labeled_array[-1,a,b]>0 :
                labeled_array[labeled_array==labeled_array[-1,a,b]] = labeled_array[0,a,b]
    for a in range(labeled_array.shape[2]):
        for b in range(labeled_array.shape[0]):
            if labeled_array[b,0,a]>0 and labeled_array[b,-1,a]>0 :
                labeled_array[labeled_array==labeled_array[b,-1,a]] = labeled_array[b,0,a]
    labels = np.unique(labeled_array)
    print('Final Num of countours: ', len(labels)-1)
    new_labeled_array = np.zeros_like(labeled_array)
    for i in range(np.size(labels)):
        ind = np.where(labeled_array==labels[i])
        new_labeled_array[ind] = i
        
    return new_labeled_array

def tophat_mask(x0,y0,z0,Lx,Ly,Lz,maxrad,BC=True):
    """ A spherical mask around (x0,y0,z0) with radius=maxrad in a box (Lx,Ly,Lz)
    if BC is True, the box is periodic
    returns: The x,y,z coordiantes of the grid points within the sphere
    """
    if not BC :
        xlo = np.max([0, np.round(x0) - maxrad])
        xhi = np.min([np.round(x0) + maxrad, Lx-1])
        ylo = np.max([0, np.round(y0) - maxrad])
        yhi = np.min([np.round(y0) + maxrad, Ly-1])
        zlo = np.max([0, np.round(z0) - maxrad])
        zhi = np.min([np.round(z0) + maxrad, Lz-1])
        xbin, ybin, zbin = np.arange(xlo, xhi), np.arange(ylo, yhi), np.arange(zlo,zhi)
    else:
        xlo, xhi = (np.around(x0) - maxrad)%Lx, (np.around(x0) + maxrad)%Lx
        ylo, yhi = (np.around(y0) - maxrad)%Ly, (np.around(y0) + maxrad)%Ly
        zlo, zhi = (np.around(z0) - maxrad)%Lz, (np.around(z0) + maxrad)%Lz
        if xlo<xhi :
            xbin = np.arange(xlo,xhi)
        else:
            xbin = np.append(np.arange(xlo,Lx), np.arange(0,xhi+1))
        if ylo<yhi :
            ybin = np.arange(ylo,yhi)
        else:
            ybin = np.append(np.arange(ylo,Ly), np.arange(0,yhi+1))
        if zlo<zhi :
            zbin = np.arange(zlo,zhi)
        else:
            zbin = np.append(np.arange(zlo,Lz), np.arange(0,zhi+1))
    xgr, ygr, zgr = np.meshgrid(xbin,ybin,zbin)
    return (xgr, ygr, zgr)


def radial_flux_profile(mapconv, x0, y0, z0, maxrad=25, BC=True):
    """get sphericaly averaged flux profile around (x0,y0,z0)
    arguments :
    Bc : Periodic Boundary Condition
    returns :
    prof_x, an array of the radii of voxels
    prog_y, an array of the spherecically averaged flux at corresponding radii
    """
    prof_r_bins = np.arange(maxrad + 1)
    Lx,Ly,Lz = np.shape(mapconv)
    xgr, ygr, zgr = tophat_mask(x0,y0,z0,Lx,Ly,Lz,maxrad=maxrad,BC=BC)
    cutout = mapconv[xgr,ygr,zgr] / np.std(mapconv)
    rad = np.sqrt((xgr - x0)**2 + (ygr - y0)**2 + (zgr - z0)**2)
    assert cutout.shape == rad.shape

    prof_x = np.zeros(maxrad)
    prof_y = prof_x * 0.
    for i in range(len(prof_r_bins)-1):
        w = (rad >= prof_r_bins[i]) & (rad < prof_r_bins[i+1]) & np.isfinite(cutout)
        prof_x[i] = np.mean(rad[w])
        prof_y[i] = np.mean(cutout[w])
    return prof_x, prof_y

def our_minima_finder(dfmap, thresh):
    """Drew's code for finding the local minima
    Since we are using np.roll(), the boundary condition is periodic
    Parameters :
    dfmap : flux map
    thresh : minima lower than this or within thersh[0] and thresh[2] (depending on the dimention of thresh) are accepted
    returns : a boolean map with local minima as true
    """
    ismin = np.ones_like(dfmap, dtype=bool)
    for rx in [-1,0,1]:
        xrolled = np.roll(dfmap, rx, axis=0)
        for ry in [-1,0,1]:
            yrolled = np.roll(xrolled, ry, axis=1)
            for rz in [-1,0,1]:
                if rx==0 and ry==0 and rz==0: continue
                ismin = ismin & (dfmap < np.roll(yrolled, rz, axis=2))
    # Apply threshold
    try :
        ismin = ismin & (dfmap < - np.abs(thresh[1])) & (dfmap > - np.abs(thresh[0]))
    except TypeError:
        ismin = ismin & (dfmap < -np.abs(thresh))
    return ismin

def find_extrema(mapconv, thresh=-2.0, linking_contour=2.0, periodic_bound=True, max_structure_size=None, keepallminima=True, hmin=None, minimalist=True):
    """A function to get the contours with significance (i.e. delta_f /sigma_map) < -linking_contour which host at least an absorptionminima with significance < thresh. 
    Inputs:
        mapconv : ndarray, required
            Smoothed delta_F / sigma_map. Note: If boundary is periodic make sure to set mode='wrap' when smoothing with `scipy.ndimage.gaussian_filter()`
        thresh: float
            The threshold on absoprtion peaks
        linking_contour : float (must be the absolute value, i.e. positive)
            The abs value of the significance of the parent contours
        periodic_bound : bool, default:True
            True if the boundary is periodic
        max_structure_size : Int ot float, Optional, default=None
            In cMpc/h, the maximum size of any strucutre
        keepallminima : bool, default=True
            If False, link absorption peaks within the same +- LINKING_CONTOUR sigma contour
            as long as they are separated by less than MAX_STRUCTURE_SIZE.
        minimalist: bool, default=True
            If True, do not calculate or save some characterisitcs of the watersheds whcih are
            not esential, e.g. the position of their centroid, etc.
    Returns:
        st : a dictionary storing info about each absorption peak
        labels : An ndarray with the same shape as mapconv. The voxels are numbered the same if they bemong to the same parent contour. Look at scipy.ndimage.label() for more info
        
    """

    assert linking_contour > 0 
    domaxima = False
    # Identify extrema with maps smoothed by 4 cMpc/h kernel
    std = np.std(mapconv)
    m = mapconv/std
    neg = m if not domaxima else -m
    
    from skimage.morphology.extrema import local_minima
    #pad the array for the purpose of periodic boundary condition
    if hmin is None :
        a = 2
        offset = [(a,a),(a,a),(a,a)]
        ismin = local_minima(np.pad(neg,offset, mode='wrap'))[a:neg.shape[0]+a, a:neg.shape[1]+a, a:neg.shape[2]+a]
        ismin *=(neg < thresh)
    else :
        from skimage.morphology import h_minima
        a = int(m.shape[0]/4)
        offset = [(a,a),(a,a),(a,a)]
        negp = neg
        negp[np.where(negp>-linking_contour)]=10
        ismin = h_minima(np.pad(negp,offset, mode='wrap'), h=hmin)[a:neg.shape[0]+a, a:neg.shape[1]+a, a:neg.shape[2]+a]
        ismin *=(neg < thresh)
    
    l = np.where(ismin)
    del ismin
    # Remove peaks in masked zones
    #interior = np.array([~mapmasks[field][l[0][i], l[1][i], l[2][i]] for i in range(len(l[0]))])
    x, y, z = l[0], l[1], l[2]
    del l
    # Record significance in units of map stddev
    signif = np.array([m[x[i], y[i], z[i]] for i in range(len(x))])
    print ('Initially got %d peaks, ' % len(x), flush=True)
    if minimalist:
        # Extract a 1D flux profile and measure two radii: one where the flux
        # reaches half the peak, and one where it reaches -1sigma
        Rhalf, Rsig = x*0., x*0.
        for i in range(len(x)):
            # Constructing radial profile about the peak
            px, py = radial_flux_profile(m, x[i], y[i], z[i], BC=periodic_bound)
            Rhalf[i] = np.interp(0.5, py[::-1]/py[0], px[::-1])
            Rsig[i] = np.interp(-linking_contour, py, px)
    if max_structure_size is None:
        max_structure_size = np.inf*np.ones_like(x)
    else :
        max_structure_size = max_structure_size*np.ones(shape=(len(x),))
    
    from scipy.ndimage import label
    print('label started', flush=True)
    labels, _ = label((m > linking_contour).astype(int) if domaxima else (m < -linking_contour).astype(int), structure=np.ones((3,3,3)))
    if periodic_bound :
        # enforce periodic boundary condition in Simulations
        labels = boundary_condition_label(labels)

    if not keepallminima :
        # Link points within the same +- LINKING_CONTOUR sigma contour
        # as long as they are separated by less than MAX_STRUCTURE_SIZE.
        friends=[]
        feature_label = labels[x,y,z]
        for i in range(x.size):
            # Can't throw out objects that aren't even within their own contour
            if np.abs(signif[i]) < linking_contour: continue
            # Perodic boundary for merging the peaks
            if periodic_bound :
                box_size = mapconv.shape[0]
                dist = np.abs(np.array([x-x[i], y-y[i], z-z[i]])).T
                for j in range(3):
                    ind = np.where(dist[:,j] > box_size/2)
                    dist[ind,j] = (box_size/2)-dist[ind,j]
                dist = np.sqrt(np.sum(dist*dist,1))
            else:
                dist = np.sqrt((x - x[i])**2 + (y - y[i])**2 + (z - z[i])**2)
            f = np.where((feature_label == feature_label[i]) & (dist < max_structure_size[i]))[0]
            if f.size > 1:
                friends.append(tuple(f))
        delete = np.repeat(False, len(x))
        for n in fof_networks(len(x), friends):
            if len(n) ==1 : continue 
            keep = np.argmax(np.abs(signif[list(n)]))
            delete[list(n)] = True
            delete[n[keep]] = False
        x, y, z, signif = x[~delete], y[~delete], z[~delete], signif[~delete]
        Rsig, Rhalf = Rsig[~delete], Rhalf[~delete]


    s = np.argsort(np.abs(signif))[::-1]
    ##k = np.argsort(z)
    ## Get the detection significance, i.e., the peak height relative to the map noise
    ##detsigmap = maps_conv[field][4] / maps_conv_noise[field][4]
    ##detsig = np.array([detsigmap[x[i], y[i], z[i]] for i in range(len(x))])
    ## Also record the map S/N_e at peak and the smoothed map value
    ##sne = np.array([mapsn[field][4][x[i], y[i], z[i]] for i in range(len(x))])
    mapval = np.array([mapconv[x[i], y[i], z[i]] for i in range(len(x))])
    if minimalist:
        st = {'x': x[s], 'y': y[s], 'z': z[s],'signif': signif[s], 'mapval': mapval[s], 'Rhalf': Rhalf[s], 'Rsig': Rsig[s],'thresh': thresh,
          'linking_contour': linking_contour, 'label': labels[x,y,z][s]}
    else:
        st = {'x': x[s], 'y': y[s], 'z': z[s],'signif': signif[s], 'thresh': thresh,
          'linking_contour': linking_contour, 'label': labels[x,y,z][s]}
    return st, labels

def mtomo_partition(mapconv, DMconv, z_acc, thresh=-2.0, linking_contour=2.0, coeff=None, periodic_bound=True, max_structure_size=None, keepallminima=True, minlogmassratio=None, hmin=None, minimalist=False, rank=0):
    """Performs the watershed algorithm on a mock-observed map.
    
    Parameters:
        mapconv : ndarray, required
            Smoothed delta_F / sigma_map. Note: If boundary is periodic make sure to set mode='wrap' when smoothing with `scipy.ndimage.gaussian_filter()`
        DMconv : ndarray, required:
            The smoothed dark matter density field. This rho_DM / <rho_DM>.
        thresh: float
            The threshold on absoprtion peaks
        linking_contour : float (must be the absolute value, i.e. positive)
            The abs value of the significance of the parent contours
        coeff: A list of floats with length of 3, default: None
            The 2nd-degree polynomial estimator of rho_DM / <rho_DM> -vs- delta_F. If None, 
            this estimator is obtained by fitting a polynomial between mapconv and DMconv.
        periodic_bound : bool, default:True
            True if the boundary is periodic
        max_structure_size : Int ot float, Optional, default=None
            In cMpc/h, the maximum size of any strucutre
        keepallminima : bool, default=True
            If False, link absorption peaks within the same +- LINKING_CONTOUR sigma contour 
            as long as they are separated by less than MAX_STRUCTURE_SIZE.
        minimalist: bool, default=True
            If True, do not calculate or save some characterisitcs of the watersheds whcih are
            not esential, e.g. the position of their centroid, etc.
        minlogmassratio : float or None, default=None
            If not None, merge the absorption peaks within the same parent countrours (i.e.
            linking_contour) if the ratio of the mass of the bigger to smaller is larger than
            this ratio. It is not used in our standard method.
        rank : int, default 0.
            MPI rank label. Only used to print progress log on the fly.
    
    """
    
    from skimage.segmentation import watershed
    # We need a correction for DM dsnsity map, to convert nbodykit's output from rho / <rho>
    # rho / rho_c
    if coeff is None:
        coeff = np.polyfit(np.ravel(mapconv), np.ravel(DMconv), 2)
    print ("Using delta_dm = %0.3f dF**2 %0.4f dF + %0.4f", tuple(coeff))
    delta_dm = lambda delta_F: np.polyval(coeff, delta_F)
    # The conversion factor needed to calculate M_tomo
    Delta_to_Msol_ph = convert_Delta_Msol_ph(z=z_acc, volume=1)
    print('find_extrema started!', flush=True)
    m, l = find_extrema(mapconv, thresh=thresh, linking_contour=linking_contour, periodic_bound=periodic_bound, max_structure_size=max_structure_size, keepallminima=keepallminima, hmin=hmin, minimalist=np.invert(minimalist))
    print('find_extrema is done!', flush=True)
    Nl = np.size(np.unique(np.ravel(l[l>0])))

    thresh = m['thresh']
    data = mapconv / np.std(mapconv)
    data_shape = data.shape
    # Make a map of the locations of minima
    atminimum = np.zeros(data.shape, dtype=int)
    atminimum[m['x'], m['y'], m['z']] = np.arange(1,len(m['x'])+1)

    mtomopart = np.zeros(len(m['x'])) * np.nan
    parentid = np.repeat(-1, len(m['x']))
    full_map = np.zeros_like(data)
    if not minimalist :
        mtomoparent = mtomopart.copy()
        centroid = np.ones((len(m['x']),3))
        radius = np.ones((len(m['x']),))
    if periodic_bound:
        #Pad the matrices to get the boundary condition for watershed as periodic
        offset = [(data_shape[0],data_shape[0]),(data_shape[1],data_shape[1]),(data_shape[2],data_shape[2])]
        data_pad = np.pad(data, offset, mode='wrap')
        del data
        atminimum_pad = np.pad(atminimum, offset, mode='wrap')
        del atminimum
    # Loop through contours
    for i in range(1,Nl+1):
        print('Rank ', rank, str(np.around((i/Nl)*100)), '% done!', flush=True)
        wpar = (l==i)
        assert np.sum(wpar) > 0
        wmin = np.where((atminimum_pad[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]] > 0) & wpar)    
        if len(wmin[0]) > 1:
            # Break apart the contour
            wshed_bc = False # A flag for whether the segment is on the boundary or not
            if periodic_bound :
                if np.any(wpar[0,:,:]) or np.any(wpar[:,0,:]) or np.any(wpar[:,:,0]):
                    wshed_bc = True
                    wpar_pad = np.pad(wpar, offset, mode='wrap')
                    seg = watershed(data_pad, markers=atminimum_pad, mask=wpar_pad)
                    seg = seg[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]]
                    del wpar_pad
                else :
                    seg = watershed(data_pad[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]], markers=atminimum_pad[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]], mask=wpar)
            else :
                seg = watershed(data, markers=atminimum, mask=wpar)
            Npartsinit = len(np.unique(seg)) - 1
            # Measure mass ratios if we're imposing a minimum
            if minlogmassratio is not None:
                minimalist = np.unique(seg[seg > 0])
                masses = np.log10(np.array([np.sum(delta_dm(mapconv[seg==j])) for j in minimalist]))
                mass_diff = np.max(masses) - masses
                mass_diff = np.delete(mass_diff, np.where(mass_diff == 0))
                cut = np.where(mass_diff > minlogmassratio)[0]
                if len(cut) > 0:
                    newminima = atminimum.copy()
                    for c in minimalist[cut]:
                        newminima[newminima == c] = 0
                    if wshed_bc :
                        newminima = np.pad(newminima, offset, mode='wrap')
                        wpar_pad = np.pad(wpar, offset, mode='wrap')
                        seg = watershed(data_pad, markers=newminima, mask=wpar_pad)
                        seg = seg[data.shape[0]:-data.shape[0], data.shape[1]:-data.shape[1], data.shape[2]:-data.shape[2]]
                    else :
                        seg = watershed(data, markers=newminima, mask=wpar)
            Npartsfin = len(np.unique(seg)) - 1
            print ('paernt ', i, 'Num parts initial :', Npartsinit,'Num parts final:',  Npartsfin, flush=True)
        else:
            # It's just one piece
            if len(wmin[0]) == 0:
                print (" contour"+str(i)+" hs no minima within it.", flush=True)
                continue
            seg = np.zeros(data_shape, dtype=int)
            seg[wpar] = atminimum_pad[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]][wmin[0][0], wmin[1][0], wmin[2][0]]
            #print ('parent ', i, 'Num parts initial :', len(wmin[0]),'Num parts final:',  len(wmin[0]))
        overlap = full_map*seg
        if np.any(overlap):
            print('overlap contour :', i, flush=True)
        # Add the strcutures within this contour to the full label map
        full_map += seg
        std = np.std(mapconv)
        deltaF = mapconv/std
        for j in np.unique(seg[seg > 0]):
            wpart = seg==j
            # Catalog masses of each segment (i.e. each local minimum) in this contour...
            mtomopart[j-1] = np.log10(np.sum(delta_dm(mapconv[wpart])) * Delta_to_Msol_ph)
            # ...and the id of the contour...
            parentid[j-1] = i
            if not minimalist:
                mtomoparent[j-1] = np.log10(np.sum(delta_dm(mapconv[wpar])) * Delta_to_Msol_ph)
                # Record the flux weighted centroid within a subcontour
                # In these units, integers are centered on cells (so that coordinates
                # equal the peak coordinates for an isotropic feature).
                # Also record the radius given by the sphere with vol. equal to this contour.
                # Define volume of map associated with this structure
                if periodic_bound:
                    (centroid[j-1,0], centroid[j-1,1], centroid[j-1,2], radius[j-1]) = get_centroid(dfmap=data_pad[data_shape[0]:-data_shape[0], data_shape[1]:-data_shape[1], data_shape[2]:-data_shape[2]], wpart=wpart, periodic_bound=periodic_bound)
                else:
                    (centroid[j-1,0], centroid[j-1,1], centroid[j-1,2], radius[j-1]) = get_centroid(dfmap=data, wpart=wpart, periodic_bound=periodic_bound)


    print("trimmed to ", np.unique(full_map).size-1, " Peaks")
    m['mtomo'] = mtomopart
    m['parentid'] = parentid
    if not minimalist:
        m['centroid'] = centroid
        m['radius'] = radius

    return m,full_map

def get_centroid(dfmap, wpart, periodic_bound):
    """
       Record the flux weighted centroid within a subcontour
       In these units, integers are centered on cells (so that coordinates
       equal the peak coordinates for an isotropic feature).
       Also record the radius given by the sphere with vol. equal to this contour.
       Define volume of map associated with this structure
    dfmap : the deltaF map
    wpart : A mask on dfmap being True on the subregion
    periodic_bound : True if boundary is periodic
    """

    involx, involy, involz = np.where(wpart)
    (Lx, Ly, Lz) = dfmap.shape
    if periodic_bound:
        if (0 in involx)*( Lx-1 in involx):
            involx[np.where(involx < Lx/2)] += Lx
        if (0 in involy)*( Ly-1 in involy):
            involy[np.where(involy < Ly/2)] += Ly
        if (0 in involz)*( Lz-1 in involz):
            involz[np.where(involz < Lz/2)] += Lz
        offset = [(0,int(Lx/2)), (0,int(Ly/2)), (0,int(Lz/2))]
        data = np.pad(dfmap, offset, mode='wrap')
    else : 
        data = dfmap
    del dfmap
    
    wht = np.copy(-data[involx, involy, involz])
    ind_underdens = np.where(wht < 0)
    wht[ind_underdens] = 0
    wht /= np.sum(wht)
    radius = (len(involx) / (4./3 * np.pi))**(1/3.)
    return (np.sum(involx * wht)%Lx, np.sum(involy * wht)%Ly, np.sum(involz * wht)%Lz, radius)
 

def displace_randomly(lmap, peaks, save_lmap, save_peaks, seed=69):
    """Displace the labeled map and the corresponding peak
    cataloge randomly"""
    np.random.seed(seed)
    d = np.random.randint(low=0,high=205,size=3)
    print('displace vector :', d)
    lmap = np.roll(lmap, (d[0],d[1],d[2]), (0,1,2))
    
    with h5py.File(save_lmap, 'w') as f:
        f['map'] = lmap
    with h5py.File(save_peaks,'w')  as f:
        pmap = np.zeros_like(lmap)
        for i in range(peaks['x'].size):
            pmap[peaks['x'][i], peaks['y'][i], peaks['z'][i]] = i+1
        pmap = np.roll(pmap, (d[0],d[1],d[2]), (0,1,2))
        X, Y, Z =np.array([]), np.array([]), np.array([])
        for i in range(peaks['x'].size):
            x,y,z = np.where(pmap==i+1)
            X, Y, Z = np.append(X, x), np.append(Y, y), np.append(Z, z)
        f['x'], f['y'], f['z'] = X.astype(int), Y.astype(int), Z.astype(int)
        fields = set(peaks.keys()) - {'x', 'y', 'z'}
        for a in fields:
            f[a] = peaks[a][()]
    def _test_displace_randomly():
        """Tests both labeled map and the corresponding peak cataloge are displaced consistently"""
        with h5py.File(save_peaks, 'r') as f:
            with h5py.File(save_lmap,'r') as fl :
                l = fl['map'][:].astype(int)
                for i in range(f['x'].size):
                    assert(l[f['x'][i], f['y'][i], f['z'][i]] == i+1)
    _test_displace_randomly()

def correct_Delta(z = 2.4442257045541464, mdm = 0.003983427498675485, Nt=2500**3, L=205):
    """
    The output of nbodykit for DM density is Delta = rho / <rho> where <rho> is the total_num_particles / total_volume.
    This fuction converts it to rho/rho_c at redshift z.
    mdm : Mass of each DM particle in units of 1e10*M_sol/h
    Nt : total number of DM particles
    L : simulation box in cMpc/h
    """
    from astropy.cosmology import Planck15 as cosmo
    import astropy.units as u

    return (((Nt*mdm*1e10*u.solMass/cosmo.h)/(L*u.Mpc/((1+z)*cosmo.h))**3).to(u.g/u.cm**3) / cosmo.critical_density(z=z)).value

def convert_Delta_Msol_ph(z, volume=1):
    """
    Convert Delta = rho /<rho_DM> to Msol/h  within each vocxel of volume (cMpc/h) at redshift z
    """
    from astropy.cosmology import Planck15 as cosmo
    import astropy.units as u
    
    return volume*(cosmo.critical_density(z=z).to(u.solMass/u.Mpc**3)*(1/(1+cosmo.Ob0/cosmo.Om0))*((1*u.Mpc/((1+z)*cosmo.h))**3)).value


def get_islands(thresh=-2.0, sigma=4):
    """Get the simple contours (islands) with df/sigma < thresh
       It is used for visualization
    Retrun : A 3D map where voxels are labeled by the islands tag number"""
    from scipy.ndimage import label
    from scipy.ndimage import gaussian_filter as gf
    import os
    data_dir = '/run/media/mahdi/HD2/Lya/LyTomo_data/'
    
    mock_map = np.fromfile(os.path.join(data_dir, 'mock_maps_z2.4/map_TNG_z2.4_n1.dat')).reshape(205,205,205)
    mock_map = gf(mock_map, sigma, mode='wrap')
    mock_map /= np.std(mock_map)
    contours, _ = label(mock_map < thresh)
    contorus = boundary_condition_label(contours)    
    return contorus


def get_id_max_overlap(lmap_mock, lmap_true):
    """returns : A dictionary of the corresponding ids of overlapping structures, 
    just returns those structures which have overlapping structures in true map"""
    minima_mock = np.unique(lmap_mock)
    minima_true = np.unique(lmap_true)
    minima_mock = np.delete(minima_mock, np.where(minima_mock==0))
    minima_true = np.delete(minima_true, np.where(minima_true==0))
    
    
    id_max_overlap = {'mock':np.array([]),'true':np.array([])}
    for i in minima_mock:
        indm = np.where(lmap_mock==i)
        idtrue, counts = np.unique(lmap_true[indm], return_counts=True)
        if idtrue[0] == 0:
            idtrue = np.delete(idtrue, 0)
            counts = np.delete(counts, 0)
            if counts.size== 0 :
                continue
        counts_sorted = np.sort(counts)
        # Here, If 2 sub-contours overlap identically, we pick just the one with lower id
        indt = np.where(counts == counts_sorted[-1])[0][0]
        if idtrue[indt]!=0 :
            id_max_overlap['mock'] = np.append(id_max_overlap['mock'], i)
            id_max_overlap['true'] = np.append(id_max_overlap['true'], idtrue[indt])
    id_max_overlap['true'].astype(int); id_max_overlap['mock'].astype(int)
    return id_max_overlap

def get_Mtomo_MDM(z_accurate, lmap_mock, lmap_true, peaks_mock, peaks_true, DM_file):
    """Returns Mtomo and DM mass within the mock watersheds and the companion ones
    in the nosieless map
    z_accurate : The accurate redshift of teh snapshot
    lmap_mock, lmap_true: labeled array of the mock and noiseless maps
    peaks_mock, peaks_true:  arrays containing the peaks of the mock and noiseless maps
                             Refer to motmo_partition for more details
    DM_file : path to the 3D DM overdensity map, i.e Delta = rho / <rho>
    
    Returns :
             (all mock Mtomos, all noiseless mtomos, DM mass within all mock contours, 
             DM masses within all noiseless contours, DM mass within noiseless contours which overlap with each mock contour,
             an array connecting ids of the overlapping contours in mock and noiseless maps)
    """
    DM = h5py.File(DM_file, 'r')['DM/dens'][:]
    DM *= convert_Delta_Msol_ph(z=z_accurate)
    MDM_true = np.array([])
    MDM_mock = np.array([])
    MDM_mock_true = np.array([])
    for i in range(1, peaks_mock['mtomo'].size+1):
        ind = np.where(lmap_mock == i)
        MDM_mock = np.append(MDM_mock, np.log10(np.sum(DM[ind])))
    for i in range(1, peaks_true['mtomo'].size+1):
        ind = np.where(lmap_true == i)
        MDM_true = np.append(MDM_true, np.log10(np.sum(DM[ind])))
    
    id_max_overlap = get_id_max_overlap(lmap_mock=lmap_mock, lmap_true=lmap_true)
    for i in id_max_overlap['true'][:]:
        ind = np.where(lmap_true==i)
        MDM_mock_true = np.append(MDM_mock_true, np.log10(np.sum(DM[ind])))
    return peaks_mock['mtomo'][:], peaks_true['mtomo'][:], MDM_mock, MDM_true, MDM_mock_true, id_max_overlap

def write_Mtomo_MDM(z, z_acc, th, lc, offset):
    """Writes the Mtomo and Dm masses on a file since they are slow to produce"""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_Size()
    
    DM = h5py.File('/run/media/mahdi/HD2/Lya/LyTomo_data_new/DM_Density_field/TNG_DM_z'
               +str(z)+'.hdf5','r')['DM/dens'][:]
    fname = ('/run/media/mahdi/HD2/Lya/LyTomo_data_new/plotting_data/Mtomo_MDM_z'+str(z)+'_th'
             +str(th).ljust(4,'0')+'_lc'+str(lc).ljust(4,'0')+'.hdf5')
    with h5py.File(fname,'w') as fw:
        for n in range(1,21):
            lmap_mock, peaks_mock, lmap_true, peaks_true =  load_watersheds(z=z, n=n, th=th, lc=lc)
            Mtomo_mock, Mtomo_true, MDM_mock, MDM_true, MDM_mock_true, id_max_overlap = minima.get_Mtomo_MDM(z_accurate=z_acc, lmap_mock=lmap_mock,
                                                                                                             peaks_mock=peaks_mock, lmap_true=lmap_true,
                                                                                                             peaks_true=peaks_true, DM=DM)
            fw[str(n)+'/Mtomo_mock'] = Mtomo_mock[:]+offset
            fw[str(n)+'/MDM_mock'] = MDM_mock[:]
            fw[str(n)+'/Mtomo_mock_overlap'] = Mtomo_mock[:][id_max_overlap['mock'][:].astype(int)-1]+offset
            fw[str(n)+'/MDM_true_overlap'] = MDM_mock_true
            fw[str(n)+'/id_max_overlap/mock'] = id_max_overlap['mock'][:]
            fw[str(n)+'/id_max_overlap/true'] = id_max_overlap['true'][:]