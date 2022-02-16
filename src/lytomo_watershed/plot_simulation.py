# A code to make a movie of the recovered map + overplotting the progenitors and compare it with the DM/Gas densitry field
# it uses paint_map.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
import imageio
import matplotlib

from scipy.ndimage.filters import gaussian_filter
def plot_2D(m, sigma=4., figname='Base2', boxsize=60):
    ''' Takes a 3D Mesh of the log(density/Nmesh) field. It can be found using mesh.preview() method in NbodyKit '''
    m = gf(m=m, sigma=sigma, mode='wrap')
    fig = plt.figure(figsize=(15,15)) 
    def plot_func(z):
        cmap = plt.get_cmap('jet')
        plt.imshow(m[:,:,z], extent=(0,boxsize,0,boxsize), origin='lower', cmap=cmap)
        #plt.clim(np.min(m), np.max(m))
        plt.title('z = '+str(z)+r'h^{-1} cMpc')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image
    imageio.mimsave(figname+'.gif', [plot_func(z) for z in range(0,int(boxsize))], fps=8)

def plot_minima_comparison(m, minima1, minima2, label1, label2, boxsize=205):
    """Plot flux map and compare our local minima with the skimage.morphology.extrema.local_minima()"""  
    cmap = plt.get_cmap('jet').reversed()
    def _within_linking_length(coords, z, deltaz=3):
        """Getting the objects within +- b parameter of FOF"""
        mask = np.ones((coords.shape[0],), dtype=bool)
        #mask *= coords[:,2] == z 
        mask *= coords[:,2] >= (z-deltaz)%m.shape[2]
        mask *= coords[:,2] <= (z+deltaz)%m.shape[2]
        return coords[mask].astype(int), mask
    def plot_func(z):
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        print('z= ', z)
        m_2D = m[:,:,z]
        im = ax.imshow(m_2D, cmap=cmap, origin='lower', extent=[0, m_2D.shape[1], 0, m_2D.shape[0]], interpolation='bilinear', vmin=-4, vmax=3)
        coords1, _ = _within_linking_length(minima1, z)
        coords2, _ = _within_linking_length(minima2, z)
        ax.scatter(coords1[:,1], coords1[:,0], color='w', marker='D', s=70 )
        ax.scatter(coords2[:,1], coords2[:,0], color='k', marker='*', s=20 )
        #ax.legend(fontsize='30')
        ax.set_title('z_coordinate = '+ str(z) +' cMpc/h', fontsize=30)
        ax.tick_params(labelsize=30, width=5, length=10)

        fig.savefig('./TNG_map/'+str(z)+'_flux.png')
        plt.close('all')
    for z in range(m.shape[2]):  plot_func(z)

    
#def comparison_plot(gas_field, DM_field, DM_field_prog, m_dachshund, m_true=None, m_noisy=None,  boxsize=205, halos_coords=False, sigma=4):
    ''' maeks a (1,2) subplot to compare Dachshund output with Density(DM/Gas) Field for simulation '''
def comparison_plot(labeled_mock, DM_field, boxsize=205, sigma=4):

    #if m_true is None:
    #    m = np.fromfile('./spectra/maps/map_'+map_file+'.dat').reshape((boxsize-1, boxsize-1, boxsize-1))
    #m_dachshund = np.fromfile(map_file).reshape(boxsize,boxsize,boxsize)

    """
    paint = Paint(m_dach=m_dachshund, dim=[boxsize,boxsize,boxsize], sigma=4, thresh=-3, load_BH=False,load_progen_DM=False, load_progen_Gas=False, load_protoc=False, load_halos=False)
    paint_true = Paint(m_dach=m_true, dim=[boxsize,boxsize,boxsize], sigma=4, thresh=-3, load_BH=False,load_progen_DM=True, load_progen_Gas=False, load_protoc=False, load_halos=False)
    """

    #m_true, m_noisy, m_dachund = sm.convolve(m=m_true, sigma=sigma), sm.convolve(m=m_noisy, sigma=sigma), sm.convolve(m=m_dachund, sigma=sigma)
    #m_true /= np.std(m_true)
    #m_noisy/= np.std(m_noisy) 
    #gas_field = np.exp(-gas_field)
    #DM_field = np.exp(-DM_field)
    
    #gas_field, DM_field, DM_field_prog = np.log10(gas_field), np.log10(DM_field), np.log10(DM_field_prog)
    #gas_field = (gas_field - np.mean(gas_field)) / np.abs(np.mean(gas_field))
    #DM_field = (DM_field - np.mean(DM_field)) / np.abs(np.mean(DM_field))
    #DM_field -= 1
    #gas_field, DM_field, DM_field_prog = sm.convolve(m=gas_field, sigma=sigma), sm.convolve(m=DM_field, sigma=sigma), sm.convolve(m=DM_field_prog, sigma=sigma)
    #gas_field /= np.std(gas_field)
    #DM_field  /= np.std(DM_field)
    ind = DM_field < 1
    DM_field[ind] = 1
    ind = DM_prog_all < 1
    DM_prog_all = h5py.File
    DM_prog_all[ind] = 1
    DM_field, DM_prog_all = np.log10(DM_field), np.log10(DM_prog_all)
    
    for i in range(len(DM_prog)):
        ind = DM_prog[i] < 1
        DM_prog[i][ind] = 1
        DM_prog[i] = np.log10(DM_prog[i])
        DM_prog[i] = gaussian_filter(DM_prog[i], sigma=sigma, mode='wrap')
        DM_prog[i] /=np.std(DM_prog[i])
    m_dachshund, DM_field, DM_prog_all = gaussian_filter(m_dachshund, sigma, mode='wrap'), gaussian_filter(DM_field,sigma, mode='wrap'), gaussain_filter(DM_prog_all, sigma, mode='wrap')
    m_dachshund /= np.std(m_dachshund)
    DM_field /= np.std(DM_field)
    DM_prog_all /= np.std(DM_prog_all)

    ### Center of Mass for cluster progenitors
    fcofm = h5py.File('cofm_progenitors.hdf5','r')
    ## Indices of those progenitors being slected as the closest to the absorption peaks
    ind_closest = h5py.File('ind_closest_prog_link_contour2.hdf5','r')['ind']

    def plot_func(z):
        print('z = ', z)
        fig, ax = plt.subplots(1,2,figsize=(20,17))
        cmap = plt.get_cmap('jet')
        #ax[0,0].set_title('True map (smoothed)', fontsize=30)
        #pcm1 = ax[0,0].pcolormesh(x, y, m_true[:,:,z], cmap=cmap.reversed(), vmin=-4, vmax=3)
        #ax[0,1].set_title('Regular Grid (smoothed) +noise + spec_res ', fontsize=30)
        #pcm2 = ax[0,1].pcolormesh(x, y, m_noisy[:,:,z], cmap=cmap.reversed(), vmin=-4, vmax=3)
        # Dachshund output (less sightlines)
        #ax[1].imshow( m_dachund[:,:,z].T, cmap=cmap.reversed(), vmin=-4, vmax=3, origin = 'lower', extent=(0,boxsize, 0, boxsize), interpolation='bilinear')
        """
        ax[2], im2 = paint.plot_2D_flux(ax[2], z=z)
        ax[2].set_title('Mock observed flux', fontsize=50)
        ax[2].set_xlabel('cMpc/h', fontsize=50)
        ax[2].tick_params(labelsize=30, width=5, length=10)
        """
        # Assuming the z axis is the redshift axis
        #ax[2].imshow(gas_field[:,:,z], extent=[0,boxsize-1,0,boxsize-1], origin='lower', cmap=cmap, interpolation='bilinear')
        #ax[2].set_title('Gas Density, smoothed', fontsize=30)
        #ax[2] = paint_true.plot_2D_flux(ax[2], z=z)
        #ax[2].set_title('True map (smoothed, no noise)', fontsize=30)
        """
        im0 = ax[0].imshow(DM_field[:,:,z], extent=[0,boxsize-1,0,boxsize-1], origin='lower', cmap=cmap, interpolation='bilinear')
        ax[0].set_title('Smoothed Dark matter density', fontsize=40)
        ax[0].set_xlabel('cMpc/h', fontsize=50)
        ax[0].tick_params(labelsize=30, width=5, length=10)
        """
        
        im0 = ax[0].imshow(DM_field[:,:,z], extent=[0,boxsize,0,boxsize], origin='lower', cmap=cmap, interpolation='bilinear', vmin=0, vmax=3.)
        ### Ordinary flux contour map
        #ax[0].contour(m_dachshund[:,:,z], levels=[-7,-6,-5,-4,-3,-2], origin='lower', colors=['w','w','w','w','w','w'])
        #ax[1].contour(m_dachshund[:,:,z], levels=[-3,-2], origin='lower', colors=['w','w'])

        ### watershed contours
        ax[0].contour(watershed_map[:,:,z], levels=np.arange(1, np.unique(watershed_map).size), origin='lower', colors='w', linewidths=1.0)
        ax[1].contour(watershed_map[:,:,z], levels=np.arange(1, np.unique(watershed_map).size), origin='lower', colors='w', linewidths=1.0)

        ax[0].set_title('DM Field', fontsize=30)
        ax[0].set_xlabel('cMpc/h', fontsize=50)
        ax[0].set_ylabel('cMpc/h', fontsize=50)
        ax[0].tick_params(labelsize=30, width=5, length=10)


        for i in range(len(DM_prog)):
                    
            ax[1].contour(DM_prog[i][:,:,z], levels=[10], origin='lower', colors=['orange'])
        # Overlaying the peaks found im mass_calcs.get_M_tomo_raw
        #fpc = h5py.File('PC_TNG_z2.3.hdf5','r')
        # Overlaying all minima found with find_xtrema_v3
        fpc = h5py.File('watershed_minima_no_logmassratio_TNG_z2.3.hdf5','r')

        mask = np.ones((fpc['z'][:].shape[0],), dtype=bool)
        mask *= fpc['z'][:] >= z-3
        mask *= fpc['z'][:] <= z+3
        ax[0].scatter(fpc['y'][mask], fpc['x'][mask], marker='*', s=50, color='lime')
        ax[1].scatter(fpc['y'][mask], fpc['x'][mask], marker='*', s=50, color='lime')

        im1 = ax[1].imshow(DM_prog_all[:,:,z], extent=[0,boxsize,0,boxsize], origin='lower', cmap=cmap, interpolation='bilinear', vmin=0, vmax=3.)
        ax[1].set_title('progenitor of DM within R200', fontsize=30)
        ax[1].set_xlabel('cMpc/h', fontsize=50)
        ax[1].set_ylabel('cMpc/h', fontsize=50)
        ax[1].tick_params(labelsize=30, width=5, length=10)

        ### Overlaying the center of Mass for each individual cluster progenitor
        mask = np.ones((fcofm['z'][:].shape[0],), dtype=bool)
        mask *= fcofm['z'][:] >= z-3
        mask *= fcofm['z'][:] <= z+3
        ax[0].scatter(fcofm['y'][mask], fcofm['x'][mask], marker='D', s=50, color='lime')
        ax[1].scatter(fcofm['y'][mask], fcofm['x'][mask], marker='D', s=50, color='lime')
        # Color the closest progenitors differently
        #cs = np.where(mask)[0]
        #for t in cs :
        #    if t in ind_closest :
        #        ax[0].scatter(fcofm['y'][t], fcofm['x'][t], marker='D', s=50, color='w')
        #        ax[1].scatter(fcofm['y'][t], fcofm['x'][t], marker='D', s=50, color='w')

        

        cb0 = fig.colorbar(im0 , ax=ax[0], orientation='horizontal')
        cb1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
        #cb2 = fig.colorbar(im2, ax=ax[2], orientation='horizontal')

        cb0.ax.tick_params(labelsize=30, width=5, length=10)
        cb1.ax.tick_params(labelsize=30, width=5, length=10)
        #cb2.ax.tick_params(labelsize=30, width=5, length=10)

        
        cb0.set_label(r'$log_{10} (\rho_m / \bar{\rho}_m)$', size=45)
        cb1.set_label(r'$log_{10} (\rho_m / \bar{\rho}_m)$', size=45)
        #cb2.set_label(r'$\delta_F / \sigma_{map}$', size= 45)
                                  
        fig.suptitle('z_coordinate = '+str(z)+r'$h^-1 cMpc$', fontsize=36)
        # overplot the halos within +- 4 cMpch^-1 of this depth
        
        """
        if type(halos_coords)==np.ndarray:
            h_coords = pc._within_linking_length(halos_coords, z=z)
            print(h_coords)
            ax[1].scatter(h_coords[:,0], h_coords[:,1], color='g', marker='^', s=70)
            ax[2].scatter(h_coords[:,0], h_coords[:,1], color='g', marker='^', s=70)
        """
        plt.tight_layout(pad=1.0)
        plt.savefig('./TNG_map/'+str(z)+'_prog_mock.png')
        plt.close('all')

    for z in range(boxsize-1):  plot_func(z)
    #plot_func(z=60)

def get_cofm_progenitors(prog_dir='./progenitors', files='map_PC_prog', min_mass = 13.5):
    """ Load all progenitor density map and record the cofm
    prog_dir : the directory containing all progenitor density maps
    files : the file names containing the file names of the maps
    min_mass : minimum mass of the cluster 
    """
    from mpi4py import MPI
    import glob
    import os

    all_masses = h5py.File('clusters_TNG300-1.hdf5','r')['Mass'][:]
    ind = np.where(10+np.log10(all_masses) > min_mass)[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    prog_ranks = int(ind.size/size)*np.ones(shape=(size,), dtype=np.int)
    remainder = ind.size%size
    if remainder != 0:
        prog_ranks[0:remaninder] += 1
    if rank == 0:
        start = 0
    else:
        start = np.sum(prog_ranks[0:rank])
    end = start + prog_ranks[rank]
    cofm = np.zeros((ind.size,3), dtype=np.int)
    for i in np.arange(start,end):
        try :
            with h5py.File(os.path.join(prog_dir, files+str(i)+'.hdf5'), 'r') as fp:
                r = np.arange(0,fp['DM'][:].shape[0])
                weight = np.sum(fp['DM'][:], axis=(1,2))
                cofm[i,0] = np.around(np.sum(r*weight) / np.sum(weight))
                weight = np.sum(fp['DM'][:], axis=(0,2))
                cofm[i,1] = np.around(np.sum(r*weight) / np.sum(weight))
                weight = np.sum(fp['DM'][:], axis=(0,1))
                cofm[i,2] = np.around(np.sum(r*weight) / np.sum(weight))
        except OSError:
            print('This file not found : ', os.path.join(prog_dir, files+str(i)+'.hdf5'))
    comm.Barrier()
    cofm = np.ascontiguousarray(cofm, np.int)
    comm.Allreduce(MPI.IN_PLACE, cofm, op=MPI.SUM)
    if rank == 0:
        with h5py.File('cofm_progenitors.hdf5','w') as fw:
            fw['x'], fw['y'], fw['z'] = cofm[:,0], cofm[:,1], cofm[:,2]
            fw['cluster_ind'] = ind

    

def prog_parts(fig, ax, z, peaks, title='', vmin=-2.5, vmax=-1.3, sigma=4, lmap= None, ax_label=True):
    """A plot to show how contours encapsulate the massive progenitors and how the absorption peaks
    are proxy to the center of  mass of these protoclusters"""
    """
    if z is not None :
        zs, ze = z, z+1
        xs, xe = 0, 205
        zs, ze = 0, 205
    if x is not None :
        xs, xe = x. x+1
        ys, ye = 0, 205
        zs, ze = 0, 205
    if y is not None:
        ys, ye = y, y+1
        xs, xe = 0, 205
        ys, ye = 0,205

    """
    DM_prog_all = h5py.File('./progenitors/Full_prog_map.hdf5','r')['DM'][:]
    #num_prog_parts = h5py.File('./spectra/maps/map_PC_prog_R200_all_clusters.hdf5','r')['num_parts'][()]
    
    # Limit the range of variation for plotting purposses
    DM_prog_all = gf(DM_prog_all, sigma=4 , mode='wrap')
    ind = DM_prog_all == 0
    DM_prog_all[ind] = 1e-10
    DM_prog_all = np.log10(DM_prog_all)

    all_clusters = h5py.File('clusters_TNG300-1.hdf5','r')
    fcofm = h5py.File('cofm_progenitors.hdf5','r')
    cmap = plt.get_cmap('jet')
    # 0, 2
    im = ax.imshow(DM_prog_all[:,:,z], extent=[0,DM_prog_all.shape[0],0,DM_prog_all.shape[1]], origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)

    if lmap is not None :
        ax.contour(lmap[:,:,z], levels=np.arange(.1, np.unique(lmap).size), origin='lower', colors='w', linewidths=1.8)

      
    # Draw the absorption peaks
    mask = np.ones((peaks['z'][:].shape[0],), dtype=bool)
    mask *=  peaks['z'][:] >= z-np.around(sigma)
    mask *=  peaks['z'][:] <= z+np.around(sigma)
    ax.scatter(peaks['y'][mask], peaks['x'][mask], marker='*', s=150, color='lime', label='Absorption peaks')
        
    # Draw the Center of Mass of the proto-clusters/gropus
    mask = np.ones((fcofm['z'][:].shape[0],), dtype=bool)
    mask *= fcofm['z'][:] >= z-np.around(sigma)
    mask *= fcofm['z'][:] <= z+np.around(sigma)
    masses = all_clusters['Mass'][:][fcofm['cluster_ind'][:][mask].astype(int)]
    ind1, ind2, ind3 = np.where(masses<10**4.0), np.where((masses < 10**4.5)*(masses > 10**4.0)), np.where(masses > 10**4.5)
    ax.scatter(fcofm['y'][mask][ind1], fcofm['x'][mask][ind1], marker='D', s=150, c='orchid', label=r'$ 10^{13.5} < M < 10^{14} $')
    ax.scatter(fcofm['y'][mask][ind2], fcofm['x'][mask][ind2], marker='D', s=150, c='cyan', label=r'$ 10^{14} < M < 10^{14.5} $')
    ax.scatter(fcofm['y'][mask][ind3], fcofm['x'][mask][ind3], marker='D', s=150, c='w', label=r'$ 10^{14.5} < M$')

        
    ax.set_title(title, fontsize=50)
    if ax_label:
        ax.set_xlabel('cMpc/h', fontsize=50)
        ax.set_ylabel('cMpc/h', fontsize=50)
        ax.tick_params(labelsize=50, width=5, length=10)
        ax.set_xticks(np.arange(0,DM_prog_all.shape[0],200))
        ax.set_yticks(np.arange(0,DM_prog_all.shape[0],200))

    else :
        ax.set_xticks([])
        ax.set_yticks([])



    cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.047, pad=0.01)
    cb.ax.tick_params(labelsize=30, width=5, length=10)
    cb.set_label(r'$log_{10} (\rho_m / \bar{\rho}_m)$', size=45)
    
    plt.legend(loc=(1.005,0), fontsize=25)
       

    
def DM_plot(fig, ax, z, DM, title='', vmin=-0.35, vmax=0.35, plot_contour=False, plot_peaks=False, plot_prog_cofm=False, ax_label=True):
    
    cmap = plt.get_cmap('jet')
    im = ax.imshow(DM[:,:,z], extent=[0, DM.shape[0], 0, DM.shape[1]], origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.047, pad=0.01)
    cb.ax.tick_params(labelsize=30, width=5, length=10)
    cb.set_label(r'$log_{10} (\rho_m / \bar{\rho_m})$', size=45)
    ax.set_title(title, fontsize=50)
    if ax_label :
        ax.set_ylabel('cMpc/h', fontsize=50)
        ax.tick_params(labelsize=50, width=5, length=10)
        ax.set_xticks([])
        ax.set_yticks(np.arange(0,DM.shape[0],200))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

        
    
    if plot_contour:
        lmap = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')['map'][:]
        ax.contour(lmap[:,:,z], levels=np.arange(1, np.unique(lmap).size), origin='lower', colors='w', linewidths=1.0)
    if plot_peaks :
        peaks = h5py.File('./thresh/peaks_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')
        # Draw the absorption peaks
        mask = np.ones((peaks['z'][:].shape[0],), dtype=bool)
        mask *=  peaks['z'][:] >= z-4
        mask *=  peaks['z'][:] <= z+4
        ax.scatter(peaks['y'][mask], peaks['x'][mask], marker='*', s=120, color='lime', label='Absorption peaks')
    if plot_prog_cofm :
        fcofm = h5py.File('cofm_progenitors.hdf5','r')
        # Draw the Center of Mass of the proto-clusters/gropus
        mask = np.ones((fcofm['z'][:].shape[0],), dtype=bool)
        mask *= fcofm['z'][:] >= z-4
        mask *= fcofm['z'][:] <= z+4
        all_clusters = h5py.File('clusters_TNG300-1.hdf5','r')
        masses = all_clusters['Mass'][:][fcofm['cluster_id'][:][mask].astype(int)]
        ind1, ind2 = np.where((masses > 10**3.75 )*(masses<10**4.0)), np.where(masses > 10**4.0)
        
        ax.scatter(fcofm['y'][mask][ind1], fcofm['x'][mask][ind1], marker='D', s=120, c='orchid', label=r'$ 10^{13.75} < M < 10^{14} $')
        ax.scatter(fcofm['y'][mask][ind2], fcofm['x'][mask][ind2], marker='D', s=120, c='w', label=r'$ 10^{14} < M$')
    if plot_prog_cofm or plot_peaks :
        plt.legend(loc=(1.005,0), fontsize=32)
        

def plot_flux(fig, ax, z, dFmap, title='', vmin=-3, vmax=4, lmap = None, peaks=None, ax_label=True):
    
    cmap = plt.get_cmap('jet').reversed()
    im = ax.imshow(dFmap[:,:,z], extent=[0,dFmap.shape[0],0,dFmap.shape[1]], origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.047, pad=0.01)
    cb.ax.tick_params(labelsize=30, width=5, length=10)
    cb.set_label(r'$\rm{\delta_F / \sigma_{ map}}$', size=45)
    ax.set_title(title, fontsize=50)
    if ax_label :
        ax.set_ylabel('cMpc/h', fontsize=50)
        ax.tick_params(labelsize=50, width=5, length=10)
        ax.set_xticks([])
        ax.set_yticks(np.arange(0,dFmap.shape[0],200))
    else :
        ax.set_xticks([])
        ax.set_yticks([])


    if lmap is not None:
        ax.contour(lmap[:,:,z], levels=np.arange(0.1, np.unique(lmap).size), origin='lower', colors='w', linewidths=1.8)

    if peaks is not None :
        # Draw the absorption peaks
        mask = np.ones((peaks['z'][:].shape[0],), dtype=bool)
        mask *=  peaks['z'][:] >= z-4
        mask *=  peaks['z'][:] <= z+4
        ax.scatter(peaks['y'][mask], peaks['x'][mask], marker='*', s=90, color='lime', label='Absorption peaks')
 
def mega_plot(z=10):
    
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w','text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k', 'font.family':'Serif'}):
        fig = plt.figure(figsize=(42,28))
        
        ax1 = plt.subplot2grid((4,6),(0,0), colspan=2, rowspan=2)
        DM_field = h5py.File('TNG_DensField/TNG_DM_z2.4.hdf5','r')['DM/dens'][:]
        DM_field = np.log10(gf(DM_field, 4, mode='wrap'))
        DM_plot(fig, ax1, z=z, DM=DM_field, vmin=-0.35,vmax=0.35, title='smoothed DM')
        
        ax2 = plt.subplot2grid((4,6),(0,2), colspan=2, rowspan=2)
        mtrue=h5py.File('./spectra/maps/map_TNG_true_1.0_z2.4.hdf5','r')['map'][:]
        dFmap = gf(mtrue,4)
        dFmap /= np.std(dFmap)
        plot_flux(fig, ax2, z=z, dFmap=dFmap, title='True flux', ax_label=False)
        
        ax3 = plt.subplot2grid((4,6),(0,4), colspan=2, rowspan=2)
        mock = np.fromfile('spectra/maps/map_TNG_z2.4_n1_averageF.dat').reshape(205,205,205)
        mock = gf(mock, 4)
        mock /= np.std(mock)
        plot_flux(fig, ax3, z=z, dFmap=mock, title='Mock observation', ax_label=False)
        peaks = h5py.File('./thresh/peaks_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')
        lmap = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')['map'][:]
        ax4 = plt.subplot2grid((4,6),(2,1), colspan=2, rowspan=2)
        plot_flux(fig, ax4, z=z, dFmap=mock, title='Mock observation', lmap=lmap, peaks=peaks, ax_label=False)
        
        ax5 = plt.subplot2grid((4,6),(2,3), colspan=2, rowspan=2)
        prog_parts(fig, ax5, z=z, title='Progenitors', lmap=lmap, peaks=peaks, vmin=-2.5, vmax=-1.3, ax_label=False)


        plt.subplots_adjust(hspace=0.45, wspace=0.02)
        plt.suptitle('z = '+ str(z), fontsize='50')
        plt.savefig('./pres/'+str(z)+'_5plots.png')
    
def plot_for_presentation(z, dpi=75):
    
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w','text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k'}):
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        DM_field = h5py.File('TNG_DensField/TNG_DM_z2.4.hdf5','r')['DM/dens'][:]
        DM_field = np.log10(DM_field)
        DM_plot(fig, ax, z=z, DM=DM_field, vmin=-1, vmax=1., title='DM')
        plt.tight_layout()
        plt.savefig('./pres/0_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')

        fig, ax = plt.subplots(1,1, figsize=(15,15))
        DM_plot(fig, ax, z=z, DM=gf(DM_field,4), vmin=-0.6,vmax=0.1, title='DM smoothed')
        plt.tight_layout()
        plt.savefig('./pres/1_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')
        
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        mtrue=h5py.File('./spectra/maps/map_TNG_true_1.0_z2.4.hdf5','r')['map'][:]
        dFmap = gf(mtrue,4)
        dFmap /= np.std(dFmap)
        plot_flux(fig, ax, z=z, dFmap=dFmap, title='True flux')
        plt.tight_layout()
        plt.savefig('./pres/2_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')
        
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        mtrue=h5py.File('./spectra/maps/map_TNG_true_1.0_z2.4.hdf5','r')['map'][:]
        dFmap = gf(mtrue,4)
        dFmap /= np.std(dFmap)
        plot_flux(fig, ax, z=z, dFmap=dFmap, title='True flux')
        # Plot sightlines
        sightlines = h5py.File('./spectra/spectra_TNG_z2.4_1.hdf5','r')['spectra/cofm'][:]
        ax.scatter(sightlines[:,1]/1000,sightlines[:,0]/1000, marker='+', s=30, color='w', alpha=0.5)
        #plt.legend(loc=(1.005,0), fontsize=32)
        plt.tight_layout()
        plt.savefig('./pres/3_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')

        fig, ax = plt.subplots(1,1, figsize=(15,15))
        mock = np.fromfile('spectra/maps/map_TNG_z2.4_n1_averageF.dat').reshape(205,205,205)
        mock = gf(mock, 4)
        mock /= np.std(mock)
        plot_flux(fig, ax, z=z, dFmap=mock, title='Mock observation')
        plt.tight_layout()
        plt.savefig('./pres/4_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')
        
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        peaks = h5py.File('./thresh/peaks_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')
        lmap = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')['map'][:]
        plot_flux(fig, ax, z=z, dFmap=mock, title='Mock observation', lmap=lmap, peaks=peaks)
        plt.tight_layout()
        plt.savefig('./pres/5_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')

        
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        prog_parts(fig, ax, z=z, title='Progenitors', lmap=lmap, peaks=peaks, vmin=-.6, vmax=.1)
        plt.tight_layout()
        plt.savefig('./pres/6_pres_z'+str(z)+'.jpg', dpi=dpi)
        plt.close('all')

    
    # our contours
    #lmap_mock = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_th2.35.hdf5','r')['map'][:]
    #for a in [ax[0,0],ax[0,1],ax[1,0],ax[1,1]] :
    #    a.contour(lmap_mock[:,:,z], levels=np.arange(1, np.unique(lmap_mock).size), origin='lower', colors='k', linewidths=1.0)
    #plt.tight_layout()
    #plt.savefig('4_test.png')

def for_undetected_clusters(lmap, peaks, dFmap):
    """ Being used to see the undetected massive clusters
    """
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w','text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k'}):
        def plot_func(z):
            fig, ax = plt.subplots(1,2, figsize=(30,20))
            print(z)
            plot_flux(fig=fig, ax=ax[0], z=z, dFmap=dFmap, title='True flux', lmap=lmap, peaks=peaks)
            prog_parts(fig=fig, ax=ax[1], z=z, title='progenitors', lmap=lmap, peaks=peaks)
            plt.tight_layout()
            plt.suptitle('z= '+str(z), fontsize=45)
            plt.savefig('./TNG_map/'+str(z)+'_prog_maps.png')
            plt.close('all')
        for z in range(205): plot_func(z=z)
    
    
def plot_map_contour_numbered(vmin=-3, vmax=4):
    """Same as plot_flux(), but idf of the contour displayed for the peaks"""    
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w','text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k'}) :
        merged_peaks = np.array([  0,  24,  31., 123.,  85., 175.,  37.,  74., 144., 168.,  66.,148.]).astype(int)
        dFmap = np.fromfile('spectra/maps/map_TNG_z2.4_n1_averageF.dat').reshape(205,205,205)
        dFmap = gf(dFmap, 4, mode='wrap')
        dFmap /= np.std(dFmap)
        peaks = h5py.File('./thresh/peaks_TNG_z2.4_n1_th2.35.hdf5','r')
        markers = np.arange(1, peaks['x'].size+1).astype(str)
        #markers = [str(i) for i in range(peaks['x'].size+1)]
        def func(z):
            print('z = '+ str(z))
            fig, ax = plt.subplots(1,1, figsize=(40,40))
            cmap = plt.get_cmap('jet').reversed()
            im = ax.imshow(dFmap[:,:,z], extent=[0,dFmap.shape[0],0,dFmap.shape[1]], origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im , ax=ax, orientation='horizontal')
            cb.ax.tick_params(labelsize=50, width=5, length=10)
            cb.set_label(r'$\delta_F / \sigma_{IGM}$', size=45)

            ax.set_xlabel('cMpc/h', fontsize=50)
            ax.set_ylabel('cMpc/h', fontsize=50)
            ax.tick_params(labelsize=50, width=5, length=10)

            lmap = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_th2.35.hdf5','r')['map'][:]
            ax.contour(lmap[:,:,z], levels=np.arange(0.5, np.unique(lmap).size), origin='lower', colors='w', linewidths=1.0)

            # Draw the absorption peaks
            mask = np.ones((peaks['z'][:].shape[0],), dtype=bool)
            mask *=  peaks['z'][:] >= z-4
            mask *=  peaks['z'][:] <= z+4
            mask = np.where(mask)[0]
            for i in range(mask.size):
                if mask[i] in merged_peaks :
                    color='lime'
                else :
                    color='w'
                plt.text(peaks['y'][mask[i]], peaks['x'][mask[i]], markers[mask[i]] , color= color, fontsize=30)
            #plt.legend(loc=(1.005,0), fontsize=32)
            fig.savefig('./TNG_map/'+str(z)+'_flux.png')
            plt.close('all')
        for z in range(205): func(z=z)
 
def compare_different_smoothings(vmin=-3, vmax=4.0, selected_peak_ind=None):
    """Compare different smoothings scales"""
    mtrue = h5py.File('./spectra/maps/map_TNG_true_1.0_z2.4.hdf5','r')['map'][:]
    dFmap = []
    lmap = []
    peaks = []
    markers = []
    
    sigma = [4,2]
    th = [2.35, 3.25]
    
    for i in range(len(sigma)):
        dFmap_temp = gf(mtrue, sigma[i], mode='wrap')
        dFmap_temp /= np.std(dFmap_temp)
        dFmap.append(dFmap_temp)
        del dFmap_temp
        peaks.append(h5py.File('./thresh/peaks_TNG_true_z2.4_n1_sigma'+str(sigma[i])+'_th'+str(th[i])[0:4].ljust(4,'0')+'.hdf5','r'))
        markers.append(np.arange(1, peaks[-1]['x'].size+1).astype(str))
        lmap.append(h5py.File('./thresh/labeled_map_TNG_true_z2.4_n1_sigma'+str(sigma[i])+'_th'+str(th[i])[0:4].ljust(4,'0')+'.hdf5','r')['map'][:])
        
    with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'white', 'figure.facecolor':'k', 'axes.titlecolor':'w','text.color':'w', 'axes.labelcolor':'w', 'legend.facecolor':'k'}) :
        def func(z):
            print('z =', z)
            fig, ax = plt.subplots(1,len(sigma), figsize=(20*len(sigma),20))
            plt.suptitle('z= '+str(z), fontsize=60)
            cmap = plt.get_cmap('jet').reversed()
            
            for i in range(len(sigma)):
                im = ax[i].imshow(dFmap[i][:,:,z], extent=[0,205,0,205], origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
                ax[i].contour(lmap[i][:,:,z], levels=np.arange(0.5, np.unique(lmap[i]).size), origin='lower', colors='w', linewidths=0.5)
                mask = np.ones((peaks[i]['z'][:].shape[0],), dtype=bool)
                mask *=  peaks[i]['z'][:] >= z-sigma[i]
                mask *=  peaks[i]['z'][:] <= z+sigma[i]
                mask = np.where(mask)[0]
                
                if (i==1)*(selected_peak_ind is not None) :
                    indpm = mask[np.in1d(mask,selected_peak_ind)]
                    if indpm.size == 0:
                        indpm = None
                else :
                    indpm = None
                for j in range(mask.size):
                    if indpm is None :
                        color = 'w'
                    else :
                        if mask[j] in indpm :
                            color='lime'
                        else :
                            color='w'
                    ax[i].text(peaks[i]['y'][mask[j]], peaks[i]['x'][mask[j]], markers[i][mask[j]] , color=color, fontsize=20)

            #axes = [ax[0,0], ax[0,1],ax[1,0],ax[1,1]]    
            for i in range(2):
                a = ax[i]
                cb = fig.colorbar(im , ax=a, orientation='horizontal')
                cb.ax.tick_params(labelsize=30, width=5, length=4)
                cb.set_label(r'$\delta_F / \sigma_{IGM}$', size=30)
                #a.set_xlabel('cMpc/h', fontsize=50)
                #a.set_ylabel('cMpc/h', fontsize=50)
                a.set_title(r'$\sigma = $'+str(sigma[i]), fontsize=30)
                a.tick_params(labelsize=30, width=5, length=10)
            plt.tight_layout()
            fig.savefig('./TNG_map/'+str(z)+'_flux24.png')
            plt.close('all')
        for z in range(205): func(z=z)


def compare_contours(m_dachshund, mtrue, boxsize=205, sigma=4, levels=[-2], thresh=-3, thresh_true=-2, closest_peak=False):
    """ Comapres contours for reconstructed flux map and the true flux map"""
    paint_true = Paint(m_dach=mtrue, dim=[boxsize,boxsize,boxsize], sigma=4, thresh=thresh_true, linking_contour=levels[0], load_BH=False,load_progen_DM=False, load_progen_Gas=False, load_protoc=True, load_halos=False)
    paint = Paint(m_dach=m_dachshund, dim=[boxsize,boxsize,boxsize], sigma=4, thresh=thresh, linking_contour=levels[0], load_BH=False,load_progen_DM=False, load_progen_Gas=False, load_protoc=True, load_halos=False)
    pc_true_ind, pc_ind = np.array([]), np.array([])
    # this section finds the closest peaks in true map to the peaks in mock map, using mass_calcs.closest_true(), only plot those with a companion in the true map
    if closest_peak :
        for i in range(paint.pc['x'].size):
            ind = mass_calcs.closest_true_peak(pc_true = paint_true.pc,  pc = paint.pc, pc_ind=i)
            if ind.size > 0:
                pc_true_ind = np.append(pc_true_ind, ind)
                pc_ind = np.append(pc_ind, i)
            print('# of peaks with a companion in true map : ', pc_ind.size)
    def plot_func(z):
        print('z = ', z)
        fig, ax = plt.subplots(1,1,figsize=(20,20))
        matplotlib.rcParams['contour.negative_linestyle'] = 'dashed'
        paint.plot_2D_flux(ax, z=z, only_contour=True, contour_color='r', levels=levels, pc_ind=pc_ind.astype(int), label='mock')
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        paint_true.plot_2D_flux(ax, z=z, only_contour=True, contour_color='b', levels=levels, pc_ind=pc_true_ind.astype(int), label='True')
        ax.set_title('z_coordinate = '+ str(z) +' cMpc/h, contours = '+ str(levels[0])+', thresh = '+str(thresh), fontsize=30)
        fig.savefig('./TNG_map/contours_'+str(z)+'.png')
        plt.close('all')

    for z in range(boxsize): plot_func(z)
def compare_subcontours(lmap1, lmap2, boxsize=205):
    """ Compare 2 different watershed subcontours, like mock and true """
    def plot_func(z):
        fig, ax = plt.subplots(1,1,figsize=(20,20))
        matplotlib.rcParams['contour.negative_linestyle'] = 'dashed'
        ax.contour(lmap1[:,:,z], levels=np.arange(1, np.unique(lmap1).size+1), origin='lower', colors='r', linewidths=1.0)
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        ax.contour(lmap2[:,:,z], levels=np.arange(1, np.unique(lmap2).size+1), origin='lower', colors='b', linewidths=1.0)
        ax.set_title('z_coordinate = '+ str(z) +' cMpc/h, contours = -2', fontsize=30)
        fig.savefig('./TNG_map/contours_'+str(z)+'.png')
        plt.close('all')
    for z in range(boxsize): plot_func(z)



def watershed_Rsigma(mockmap, boxsize=205, sigma=4, levels=[-2], thresh=-2):
    """ Compare watershed and Rsigma """
    
    peaks_mock, lmap_mock = minima.mtomo_partition_v2(mapconv=gf(mockmap, sigma, mode='wrap'))
    paint = Paint(m_dach=mockmap, dim=[boxsize,boxsize,boxsize], sigma=sigma, thresh=thresh, linking_contour=levels[0], load_BH=False,load_progen_DM=False, load_progen_Gas=False, load_protoc=True, pc=peaks_mock, load_halos=False)
    def plot_func(z):
       print('z=', z)
       fig, ax = plt.subplots(1,1,figsize=(20,20))
       paint.plot_2D_flux(ax, z=z, only_contour=False, contour_color='g', plot_Rsig=True,levels=levels)
       ax.contour(lmap_mock[:,:,z], levels=np.arange(1, np.unique(lmap_mock).size+1), origin='lower', colors='w', linewidths=1.0)
       ax.set_title('z_coordinate = '+ str(z) +' cMpc/h, Contours = '+ str(levels[0])+', Rsigma = '+str(thresh), fontsize=30)
       ax.tick_params(labelsize=30, width=5, length=10)

       fig.savefig('./TNG_map/'+str(z)+'_flux.pdf')
                        
       plt.close('all')

    
    for z in range(boxsize): plot_func(z)

def prog_parts_paper(fig, ax, volume, peaks, DM_prog_all, all_clusters, fcofm, lmap= None, mock_map=None, 
                     roll= (0,0,0), title='', vmin=-2.5, vmax=-1.3, sigma=4, markersize= 150, islands=None,
                     ax_label=['x','y'], colorbar=True, legend=True, boxsize=205):
    """A modified version of prog_parts() for the fugure in paper
    - volume : A boolean array with shape of the map, but True only for the slice we want to illlustrate. In one direction,
    it should a slice of size 1 (we want to draw 2D planes here)
    - peaks : a dictionary with keys = ['x', 'y', 'z'] which each contain the list of absorption peak coordinates
    - mock_map : If None, plot DM progenitor particles. Otherwise, plot the mock flux map passed with this argument.
    - roll : tuple of size 3, Default=(0,0,0). If the object of interest is along the edge, roll all maps with this vector. The argument
    "volume" should be in the new coordinates. 
    """
    DM_prog_all = np.roll(DM_prog_all, roll)
    ind_volume = np.where(volume)
    shape = np.array([np.unique(ind_volume[0]).size, np.unique(ind_volume[1]).size, np.unique(ind_volume[2]).size])
    frame = np.where(shape > 1)[0]
    
    dim = []
    for f in frame :
        dim.append((ind_volume[f][0], ind_volume[f][-1]))
    
    if mock_map is None :
        cmap = plt.get_cmap('jet')
        # Limit the range of variation for plotting purposses
        DM_prog_all = gf(DM_prog_all, sigma=sigma , mode='wrap')
        ind = DM_prog_all == 0
        DM_prog_all[ind] = 1e-10
        DM_prog_all = np.log10(DM_prog_all)
        im = ax.imshow(np.squeeze(DM_prog_all[volume].reshape(shape[frame])), 
                       extent=[dim[1][0],dim[1][1],dim[0][0],dim[0][1]], origin='lower', 
                       cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
    else :
        cmap = plt.get_cmap('jet').reversed()
        im = ax.imshow(np.squeeze(mock_map[volume].reshape(shape[frame])), 
                       extent=[dim[1][0],dim[1][1],dim[0][0],dim[0][1]], origin='lower',
                       cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
    
    # Draw contours of flux
    def _plot_contours(contours, color='darkgray'):
        """A helper function to draw contours
        contours : In the syle of scipy.ndimage.label()
        """
        contours = np.roll(contours, roll)
        x = np.arange(dim[1][0],dim[1][1]+1)
        y = np.arange(dim[0][0],dim[0][1]+1)
        xx, yy = np.meshgrid(x,y)
        ax.contour(xx, yy, np.squeeze(contours[volume].reshape(shape[frame])),
                   levels=np.arange(.1, np.unique(contours).size), origin='lower',
                   colors=color, linewidths=2)
        
    if islands is not None:
        _plot_contours(islands, color='b')    
    if lmap is not None :
        _plot_contours(lmap, color='darkgray')

    
    def _plot_points(x, y, z, marker, s, edgecolor, label, facecolor="None", alpha=1.0):
        """A helper function to plot cofm of ptogenitors or the absorption peaks"""
        coords = np.zeros((x.size, 3))
        coords = (coords[:] + np.array(roll))%(boxsize)
        coords[:,0], coords[:,1], coords[:,2] = x, y, z
        axis = np.where(shape == 1)[0][0]
        axis_coord = np.unique(ind_volume[axis])
        mask = np.ones((x.shape[0],), dtype=bool)
        mask *=  coords[:,axis] >= axis_coord-np.around(sigma)
        mask *=  coords[:,axis] <= axis_coord+np.around(sigma)
        mask *=  coords[:,frame[0]] >= np.unique(ind_volume[frame[0]])[0]
        mask *=  coords[:,frame[0]] <= np.unique(ind_volume[frame[0]])[-1]
        mask *=  coords[:,frame[1]] >= np.unique(ind_volume[frame[1]])[0]
        mask *=  coords[:,frame[1]] <= np.unique(ind_volume[frame[1]])[-1]
        if legend is False :
            label=None
        ax.scatter(coords[mask,frame[1]], coords[mask,frame[0]], marker=marker, s=s,
                   edgecolor=edgecolor, label=label, facecolor= facecolor, alpha=alpha)
        del coords
    
    # Plot absorption peaks
    _plot_points(peaks['x'][:], peaks['y'][:], peaks['z'][:],  marker='x', s=markersize,
                 edgecolor='lime', facecolor='lime',
                 label='Absorption peaks')
    # Plot Progenitor cofm
    masses = all_clusters['Mass'][:][fcofm['cluster_ind'][:].astype(int)]
    ind1, ind2, ind3 = (np.where(masses<10**4.0), 
                        np.where((masses < 10**4.5)*(masses > 10**4.0)),
                        np.where(masses > 10**4.5))
    
    _plot_points(fcofm['x'][:][ind1], fcofm['y'][:][ind1], fcofm['z'][:][ind1],  marker='D',
                 s=int(markersize*0.3), edgecolor='orchid', 
                 label=r'$\mathrm{ 10^{13.5} \ M_{\odot}/h < M(z=0) < 10^{14} \ M_{\odot}/h }$')
    _plot_points(fcofm['x'][:][ind2], fcofm['y'][:][ind2], fcofm['z'][:][ind2],  marker='D', 
                 s=int(markersize*0.6), edgecolor='cyan',
                 label=r'$\mathrm{10^{14} \ M_{\odot}/h < M(z=0) < 10^{14.5} \ M_{\odot}/h}$')
    _plot_points(fcofm['x'][:][ind3], fcofm['y'][:][ind3], fcofm['z'][:][ind3],
                 marker='D', s=markersize, edgecolor='w', label=r'$\mathrm{10^{14.5} \ M_{\odot}/h < M(z=0)  }$')

    ax.set_title(title, y=1.0, pad=-14, bbox=dict(facecolor='w', alpha=1.0))

    if len(ax_label)==0:
        ax.set_xticks([])
        ax.set_yticks([])

    else :
        ax.tick_params(labelsize=25, width=5, length=10)
        if 'x' in ax_label:
            ax.set_xlabel('cMpc/h', fontsize=30)
            if len(ax_label)==1:
                ax.set_yticks([])
        if 'y' in ax_label:
            ax.set_ylabel('cMpc/h', fontsize=30)
            #ax.set_yticks(np.arange(0,t110,50))
            if len(ax_label)==1:
                ax.set_xticks([])

    if colorbar :
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="100%", height="100%",  bbox_to_anchor=(1.2,0.25, 0.1, 1.5), 
                           bbox_transform=ax.transAxes)
        cb_ticks_pos = 'left'
        cb = fig.colorbar(im, cax=axins, orientation='vertical')
        #cb = fig.colorbar(im , cax=cax, ax=ax, orientation='vertical', fraction=0.06, pad=0.1)
        #cb.ax.tick_params(labelsize=30, width=5, length=10)
        cb.set_label(r'$\mathrm{\frac{\delta_F^{sm} }{ \delta_{map}}}$', fontsize=40)
    #if legend :
    #    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.05), ncol=4, fontsize=20, facecolor='darkgray')

def plot_3D_prog_map(MPI):
    """ A function to plot 3D progenitors. It hover over a range of vieing angel
    and save each figure
    """
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection='3d')

    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    def _plot_progs(value, sigma=4):
        """A helper function to plot all progenitors
        value : is the isocontour density value we want to plot. Plotting is like onion layers.
        """
        DM_prog_all = h5py.File('./progenitors/Full_prog_map.hdf5','r')['DM'][:]
        DM_prog_all = gf(DM_prog_all, sigma, mode='wrap')
        ind = np.where(DM_prog_all == 0)
        DM_prog_all[ind] = 1e-10
        DM_prog_all = np.log10(DM_prog_all)
        return measure.marching_cubes(DM_prog_all, value)
    
    # Plot progenitors in layers of density
    for i, value in enumerate(np.arange(-1.5,-1.0, 0.2)):
        verts, faces, normals, values = _plot_progs(value=value)
        mesh = Poly3DCollection(verts[faces], alpha=0.5*i/6)
        mesh.set_edgecolor('None')
        ax.add_collection3d(mesh)
    
    # Plot flux contours
    lmap = h5py.File('./thresh/labeled_map_TNG_z2.4_n1_sigma4_th2.35.hdf5','r')['map'][:]
    verts, faces, normals, values = measure.marching_cubes(lmap, 0.99)
    mesh = Poly3DCollection(verts[faces], alpha=0.15, color='C1')
    mesh.set_edgecolor('None')
    ax.add_collection3d(mesh)
    ax.set_xlim((0,205))
    ax.set_ylim((0,205))
    ax.set_zlim((0,205))
    
    # Change view angel and save each figure
    phi, theta = np.meshgrid(np.arange(10,360,10), np.arange(10,71,20))
    phi = np.ravel(phi)
    theta = np.ravel(theta)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    load_per_rank = int(phi.size / size)*np.ones(shape=(size,), dtype=int)
    remainder = phi.size%size
    if remainder !=0:
        load_per_rank[0:remainder]+=1
    if rank==0:
        start=0
    else:
        start = np.sum(load_per_rank[0:rank])
    end = start + load_per_rank[rank]
    for i in range(start, start+load_per_rank[rank]):
        ax.view_init(theta[i],phi[i])
        fig.savefig('3D_figures/3D_progs'+str(i)+'.png')


def run_mpi(func, boxsize):
    """Pass any of the functions above to be run in parallel
    Distribute slices among mpi ranks"""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    z_ranks = int(boxsize/size)*np.ones(shape=(size,)).astype(int)
    remainder = boxsize%size
    if remainder !=0:
        z_ranks[0:remainder]+=1
    if rank == 0:
        start = 0
    else:
        start = np.sum(z_ranks[0:rank])
    end = start + z_ranks[rank]
    #print(rank, start, end)
    for z in range(start, start+z_ranks[rank]):
        func(z=z)

def make_mp4(savefile='TNG_DM_Gas.mp4',contour=False, flux=False, prog=False):
    """ Convert the comparison_plot() output to mp4 movie"""
    writer = imageio.get_writer(savefile, fps=4)
    if prog:
        files = ['./3D_figures/3D_progs'+str(i)+'.png' for i in range(0,140)]
    if contour :
        #files = ['./TNG_map/contours_'+str(i)+'.png' for i in range(0,205)]
        files = ['./TNG_map/contours_'+str(i)+'.png' for i in range(0,205)]
    if flux :
        files = ['./TNG_map/'+str(i)+'_flux.png' for i in range(0,205)]
    for im in files:
        writer.append_data(imageio.imread(im))
    writer.close()

    
