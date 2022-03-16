# A list methods to plot figures presented in the paper
import os
import h5py
import numpy as np
import scipy.stats
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt


class Df_dm():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def deltaF_true_mock(self, fig, ax, z, nmocks=20, vmin=0.01, vmax=10):
        #matplotlib.rc('axes',edgecolor='k')

        # Set the axes
        plt.style.use('Notebook.mystyle')
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        x = np.arange(-0.4,0.4,0.01)
        bins=[np.linspace(-.4,.4,100), np.linspace(0,3.5,200)]
        extent = ([bins[0][0], bins[0][-1], bins[0][0], bins[0][-1]], [bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]])
        f = h5py.File(os.path.join(self.data_dir,'plotting_data/df_mock_true_dm_z'+str(z)+'.hdf5'),'r')
        print(f.keys())
        keys = ['df_mock_true','df_dm']
        for i, k in enumerate(keys):
            hplot = f[k+'/median'][:]
            hplot[hplot < vmin] = vmin
            im = ax[i].imshow(np.rot90(hplot), cmap=plt.cm.viridis, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), extent = extent[i], aspect='auto' )
            if k=='df_dm':
                axins = inset_axes(ax[i],width="100%", height="100%",  bbox_to_anchor=(0.9,0.5, 0.02, 0.4), bbox_transform=ax[i].transAxes)
                cb_ticks_pos = 'left'
                cb = fig.colorbar(im, cax=axins, orientation='vertical', ticks=[1e-2,1e-1, 1, 10,100])
                cb.ax.yaxis.set_tick_params(color='k')
                cb.ax.yaxis.set_ticks_position(cb_ticks_pos)
                cb.ax.yaxis.set_label_position(cb_ticks_pos)
                plt.setp(plt.getp(axins, 'yticklabels'), color='w')
                cb.ax.set_ylabel('2-D PDF', labelpad=5, color='w', fontsize=20)


        ax[0].set_xlabel(r'$\mathrm{\delta^{\rm sm}_{\rm F, mock}}$')
        ax[0].set_ylabel(r'$\mathrm{\delta^{\rm sm}_{\rm F, noiseless}}$')
        #ax[1].set_xlabel(r'$\mathrm{\delta^{\rm sm}_{\rm F, mock}}$')
        ax[1].set_ylabel(r'$\rm  \left(\frac{\rho_{DM}}{ \langle \rho_{DM} \rangle}\right)^{\rm sm}$', fontsize=40)
        ax[1].set_xlim(-0.35,0.25)
        ax[0].set_xlim(-0.35,0.25)
        ax[0].set_ylim(-0.35,0.25)
        ax[1].set_ylim(0,3.5)
        ax[0].set_yticks(ticks = np.arange(-.3,.3,.1))
        ax[0].set_xticks(ticks = np.arange(-.3,.3,.1))
        ax[1].set_xticks(ticks = np.arange(-.3,.3,.1))
        ax[0].plot([-.3,.2],[-.3,.2], label='1:1', color='C1', ls='--')
        ax[0].legend(loc='lower right', facecolor='w')
        ax[1].legend(loc='lower left', fontsize=20, facecolor='w')
        plt.setp(ax[0].get_xticklabels(), rotation=20, fontsize=20)

        ax[1].grid(True)
        ax[0].grid(True)
        plt.tight_layout()
        
class Mtomo_MDM():
    def __init__(self, mass_file):
        self.mass_file = mass_file

    def plot_Mtomo_MDM(self, fig, ax, Mtomo, MDM, z, th, lc, signif, mass_file, title='', 
                       xlabel=r'$\rm log[M_{tomo}]$',  ylabel=r'$\rm log[M_{DM, noiseless}]$',co=None,
                       legend=True, plot_kde=False, first_plot=True, vmin=0.01, vmax=1):

        mmin, mmax = 13, 16
        ax.plot([13,16],[13,16], alpha=0.4, color='r', label='1:1', ls='--', lw=8)

        if plot_kde :

            Z = self.get_mean_KDE(z=z, th=th, lc=lc, mass_file=mass_file, first_plot=first_plot)
            #Z[Z < 0.01*vmin] = 0.01*vmin
            im = ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,extent = [mmin, mmax, mmin, mmax] )
            cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.25, shrink=0.7)

            cb.ax.set_xlabel('Gaussian KDE')

        if co is None:
            co = linregress(Mtomo-14, MDM)
            #print('fitting line :', co)

        x = np.array([13-14,16])
        ax.plot(x, co[0]*(x-14)+co[1], alpha=0.7, label='power-law estimator',ls='dotted', lw=8)

        if signif is None :
            ax.scatter(Mtomo, MDM, alpha=0.3)
        else :
            ind1 = (signif > -2.5)*(signif < -2.35)
            ind2 = (signif > -3.0)*(signif < -2.5)
            ind3 = (signif > -3.5)*(signif < -3.0)
            ind4 = (signif < -3.5)
            ax.scatter( Mtomo[ind1], MDM[ind1], alpha=0.6, label=r'$-2.5 < \sigma < -2.35$',
                       s=50, facecolor='None', edgecolor='r', marker='^', lw=2)
            ax.scatter( Mtomo[ind2], MDM[ind2], alpha=0.65, label=r'$-3.0 < \sigma < -2.5$',
                       s=50, facecolor='None', edgecolor='C1', marker='s', lw=2)
            ax.scatter( Mtomo[ind3], MDM[ind3], alpha=0.6, label=r'$-3.5 < \sigma < -3.0$',
                       s=50, facecolor='None', edgecolor='violet', marker='o', lw=2)
            ax.scatter( Mtomo[ind4], MDM[ind4], alpha=0.6, label=r'$ \sigma < -3.5 $', s=50,
                       facecolor='None', edgecolor='w', marker='D', lw=2)



        bins = [13,14,14.5,15.5]
        std = []
        co = [co[0], co[1]]
        for b in range(1,len(bins)):
            ind = np.where((Mtomo[:] > bins[b-1])*(Mtomo[:] < bins[b]))
            #print('diff = ', MDM[ind] - Mtomo[ind])
            dev = MDM[ind] - np.polyval(co, Mtomo[ind])
            #print('dev = ', dev)
            std.append(np.sqrt(np.mean(dev*dev)))
        #print('RMS scatter around the fit in bins of ', bins, ' are :', std)
        dev = MDM - np.polyval(co, Mtomo)
        #print(' The total RMS scatter around the fit is : ', np.sqrt(np.mean(dev*dev)))

        ax.set_xlim((13,16))
        ax.set_ylim((13,16))
        ax.set_xticks(np.arange(13.5,16,.5))
        ax.set_yticks(np.arange(13.5,16,.5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.7)
        ax.set_title(title)
        if legend:
            ax.legend( framealpha=1, loc=(1.05, 0.2), fontsize=18, facecolor='gray',
                      frameon=True)        
        plt.setp(ax.get_xticklabels(), rotation=40)

    def get_kde(self, M1, M2, mmin= 13, mmax= 16):

        from scipy.stats import gaussian_kde as gkde

        data = np.zeros((2,M1.size))
        data[0,:] = M1
        data[1,:] = M2

        kernel = gkde(data)  
        X, Y = np.mgrid[mmin:mmax:50j, mmin:mmax:50j]
        grid_points = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(grid_points).T, X.shape)
        del X, Y
        return Z

    def get_mean_KDE(self, z, th, lc, mass_file, nmocks=20, first_plot=True):
        import os

        f = h5py.File(mass_file,'r')
        Zall = []
        if first_plot:
            for n in np.arange(1,nmocks+1):
                Zall.append(self.get_kde(f[str(n)+'/Mtomo_mock'][:], f[str(n)+'/MDM_mock'][:]))
        else:
            for n in np.arange(1,nmocks+1):
                Zall.append(self.get_kde(f[str(n)+'/Mtomo_mock_overlap'][:], 
                                         f[str(n)+'/MDM_true_overlap'][:]))
        f.close()

        Zmean = np.zeros(shape=(50,50))
        for Z in Zall:
            Zmean += Z
        Zmean /= nmocks

        return Zmean
    

    def make_a_pool(self, nrange, z=2.4, th=2.35, lc=2.00):
        """Make a pool of all strucutres in all mocks"""
        Mtomo = np.array([])
        MDM = np.array([])

        f = h5py.File(self.mass_file,'r')
        for n in nrange:
            Mtomo = np.append(Mtomo, f[str(int(np.around(n)))+'/Mtomo_mock_overlap'][:])
            MDM = np.append(MDM, f[str(int(np.around(n)))+'/MDM_true_overlap'][:])

        # Randomize the pool
        ind_rnd = np.random.choice(np.arange(Mtomo.size), size= Mtomo.size, replace=False)
        Mtomo = Mtomo[ind_rnd]
        MDM = MDM[ind_rnd]

        return Mtomo, MDM


    def plot_different_fits(self, train_size=16, z=2.4, th=2.35, lc=2.00):

        fig, ax = plt.subplots(figsize=(5,5))
        x = np.arange(13,16.5, 1)

        f = h5py.File(self.mass_file,'r')

        slopes = []
        intercepts = []
        ntrain = np.random.choice(np.arange(1,21), size= train_size, replace=False)
        y = np.zeros((train_size, x.size))
        for i, n in enumerate(ntrain):
            Mtomo_mock = f[str(n)+'/Mtomo_mock_overlap'][:]
            co = linregress(Mtomo_mock[:] - 14, f[str(n)+'/MDM_true_overlap'][:])
            slopes.append(co[0])
            intercepts.append(co[1])
            ax.plot(x, co[0]*(x-14)+co[1], alpha=0.7, lw=1)
            y[i] = co[0]*(x-14)+co[1]
        mean = np.mean(y, axis=0)
        #ax.plot(x, mean, lw=4)
        std = np.std(y, axis=0)

        slope_mean = np.mean(slopes)
        intercept_mean = np.mean(intercepts)
        print('Slope : ', str(slope_mean)[:5]+'+-'+str(np.std(slopes))[:5])
        print('Intercept : ', str(intercept_mean)[:5]+'+-'+str(np.std(intercepts))[:5])
        ax.fill_between(x=x, y1=mean-std, y2=mean+std, color='C3', alpha=0.4, edgecolor='None')


        ax.set_xlim((13,16))
        ax.set_ylim((13,16))
        ax.set_xlabel(r'$\mathrm{M_{tomo, mock}}$')
        ax.set_ylabel(r'$\mathrm{M_{DM, noiseless}}$')

        return ntrain, (slope_mean, intercept_mean)

    def test_fit(self, co, ntest, z=2.4, th=2.35, lc=2.00, bins=np.array([13, 14, 14.5, 15.5])):
        """Find the errors for in mass bins of bins for the mock maps array of ntest"""

        fit = lambda x : co[0]*(x-14) +co[1]
        Mtomo, MDM = self.make_a_pool(nrange=ntest,z=z, th=th, lc=lc)

        err = []
        for i in range(bins.size-1):
            ind = np.where( (Mtomo > bins[i])*(Mtomo < bins[i+1]))
            dev = fit(Mtomo[ind]) - MDM[ind]
            err.append(np.sqrt(np.sum(dev*dev)/dev.size))

        return err

class Mtomo_Mdesc():
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def read_files(self, n, z, sigma, th, lc, noiseless, random):

        if noiseless:
            fname = (os.path.join(self.data_dir,'descendants/z'+str(z)+'_sigma'+sigma+'_th'+th
                                  +'_lc'+str(lc)+'/Mass_voted_FOF_halos_true_z'+str(z)+'_n'+str(n)
                                  +'_sigma'+sigma+'_th'+th+'_halos_lc'+lc+'.hdf5'))
            pname = (os.path.join(self.data_dir,'watersheds_z'+str(z)+'/noiseless/peaks_TNG_true_z'
                                  +str(z)+'_n'+str(n)+'_sigma'+sigma+'_th'+th+'_lc'+lc+'.hdf5'))
            lmapname = (os.path.join(self.data_dir,'watersheds_z'+str(z)+'/noiseless/labeled_map_TNG_true_z'
                                     +str(z)+'_n'+str(n)+'_sigma'+sigma+'_th'+th+'_lc'+lc+'.hdf5'))
        elif random:
            print('Reading the random watersheds')
            fname = (os.path.join(self.data_dir,'descendants/z'+str(z)+'_sigma'+sigma+'_th'+th
                                  +'_lc'+str(lc)+'/Mass_voted_FOF_halos_z'+str(z)+'_n'+str(n)
                                  +'_sigma'+sigma+'_th'+th+'_halos_lc'+lc+'_rnd.hdf5'))
            pname = (os.path.join(self.data_dir,'watersheds_z'+str(z)+'/mocks/n'+str(n)+'/peaks_TNG_z'
                                  +str(z)+'_n'+str(n)+'_sigma'+sigma+'_th'+th+'_lc'+lc+'_rnd.hdf5'))
            lmapname=None
        else:
            fname = (os.path.join(self.data_dir,'descendants/z'+str(z)+'_sigma'+sigma+'_th'+th
                                  +'_lc'+str(lc)+'/Mass_voted_FOF_halos_z'+str(z)+'_n'+str(n)
                                  +'_sigma'+sigma+'_th'+th+'_halos_lc'+lc+'.hdf5'))
            pname = (os.path.join(self.data_dir,'watersheds_z'+str(z)+'/mocks/n'+str(n)+'/peaks_TNG_z'
                                  +str(z)+'_n'+str(n)+'_sigma'+sigma+'_th'+th+'_lc'+lc+'.hdf5'))
            lmapname = (os.path.join(self.data_dir,'watersheds_z'+str(z)+'/mocks/n'+str(n)+'/labeled_map_TNG_z'
                                     +str(z)+'_n'+str(n)+'_sigma'+sigma+'_th'+th+'_lc'+lc+'.hdf5'))
            

        f = h5py.File(fname,'r')
        peaks = h5py.File(pname,'r')
        lmap = h5py.File(lmapname, 'r')['map'][:]
        return f, peaks, lmap

    def get_ind_non_overlapping(self, lmap1, lmap2):
        """Returns the indices in the lmap1 which overlap with no satershed in lmap2"""
        ind = np.where(lmap2!=0)
        id_overlapping = np.unique(lmap1[ind])[1:]
        id_non_overlapping = np.unique(lmap1)[1:][np.isin(np.unique(lmap1)[1:],
                                                          id_overlapping, invert=True)]
        return id_overlapping.astype(int) - 1, id_non_overlapping.astype(int) - 1
    

    def get_Mtomo_GroupMass(self, f, peaks, offset, noiseless):

        GroupMass = 10+np.log10(f['GroupMass'][:])
        ind = f['peak_id'][:].astype(int)-1
        Mtomo = peaks['mtomo'][:][ind] + offset
        if noiseless:
            signif = 0
        else :
            signif = peaks['signif'][ind]

        return Mtomo, GroupMass, signif


    def get_the_mean_fit(self, z, sigma, th, lc, offset, nrange=np.arange(1,17,1), random=False):

        from scipy.stats import linregress
        slopes, intercepts, M1test, M2test = [], [], [], []

        for n in nrange:
            f, peaks, _ = self.read_files(n=n, z=z, sigma=sigma, th=th, lc=lc, noiseless=False, random=random)
            M1, M2, _ = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=False)
            co = linregress(M1 - 14, M2) 
            slopes.append(co[0])
            intercepts.append(co[1])

        slope_mean, intercept_mean = np.mean(slopes), np.mean(intercepts)

        print('Slope : ', str(slope_mean)[:5]+'+-'+str(np.std(slopes))[:5])
        print('The prefactor : ', str(intercept_mean)[:5]+'+-'+str(np.std(intercepts))[:5])

        return (slope_mean, intercept_mean)

    def _get_err(self, co, M1, M2, bins=np.array([13, 13.75, 14.25, 14.75, 15.5])):

        fit = lambda x : co[0]*(x-14) +co[1]
        err = []
        for i in range(bins.size-1):
            ind = np.where( (M1 > bins[i])*(M1 < bins[i+1]))
            dev = fit(M1[ind]) - M2[ind]
            err.append(np.sqrt(np.sum(dev*dev)/dev.size))
        return np.array(err)


    def test_fit(self, co, z, sigma, th, lc, offset, ntest=np.arange(17,21),
                 bins= np.array([13, 14, 14.25, 14.75, 15.5]), random=False):
        """Find the errors for in mass bins of bins for the mock maps array of ntest"""

        errall = np.zeros(shape=(ntest.size, bins.size-1))
        for i, n in enumerate(ntest):
            f, peaks, _ = self.read_files(n=n, z=z, sigma=sigma, th=th, lc=lc, noiseless=False, random=random)
            M1, M2, _ = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=False)
            errall[i,:] = self._get_err(co, M1, M2, bins)

        return bins, errall

    def get_kde(self, M1, M2, minmax=(13,16,10,16), ngridsx=80):

        from scipy.stats import gaussian_kde as gkde

        data = np.zeros((2,M1.size))
        data[0,:] = M1
        data[1,:] = M2

        kernel = gkde(data)
        if ngridsx == 80:
            X, Y = np.mgrid[minmax[0]:minmax[1]:80j, minmax[2]:minmax[3]:160j]
        else:
            X, Y = np.mgrid[minmax[0]:minmax[1]:95j, minmax[2]:minmax[3]:160j]
        grid_points = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(grid_points).T, X.shape)
        del X, Y
        return Z


    def get_mean_KDE(self, offset,  z, sigma, th, lc, nmocks=20):

        Zall = []
        if sigma == '4':
            minmax=(13.0,16,10,16)
            ngridsx = 80
        else : 
            minmax=(12.5,16,10,16)
            ngridsx= 95

        for n in np.arange(1,nmocks+1):
            f, peaks, _ = self.read_files(n=n, z=z, sigma=sigma, th=th, lc=lc, noiseless=False, random=False)
            Mtomo, GroupMass, _ = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=False)
            Zall.append(self.get_kde(Mtomo, GroupMass, minmax, ngridsx))

        Zmean = np.zeros(shape=(ngridsx,160))
        for Z in Zall:
            Zmean += Z
        Zmean /= nmocks

        return Zmean
        
    def scatter_hist(self, n, fig, ax, ax_histx, ax_histy, ax_err, ax_err_label, offset, 
                     z=2.4, sigma='4', th='2.35', lc='2.00', bin_size=0.4, plot_kde=False, 
                     plot_noiseless=True, bins_err=np.array([13.3,14.0,14.5,15.0,15.5]), lmap2=None, lmap4=None):
        import matplotlib
        
        f, peaks, _ = self.read_files(n=n, z=z, sigma=sigma, th=th, lc=lc, noiseless=False, random=False)
        Mtomo, GroupMass, signif = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=False)
        f, peaks, _ = self.read_files(n=n, z=z, sigma=sigma, th=th, lc=lc, noiseless=True, random=False)
        Mtomo_noiseless, GroupMass_noiseless, _ = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=True)
        print('Number of watersheds in mock = ', Mtomo.size)
        print('Number of watersheds in noiseless = ', Mtomo_noiseless.size)
        if sigma=='2' :
            vmax=0.6
            edgecolor='w'
            extent = [12.5, 16, 10, 16]
            if (lmap4 is None) or (lmap2 is None):
                raise "pass lmap2 and lmap4 to this function"
            ind_overlapping, ind_non_overlapping = self.get_ind_non_overlapping(lmap2, lmap4)
            err_yticks = np.arange(0.3,1.0,0.1)
            ax_err.set_xlim(12.5, 15.5)
            ax_histx.set_yticks(ticks = np.arange(10,215, 50))
            ax_histy.set_xticks(ticks = np.arange(10,215, 50))
            ax.set_xlim(12.5,15.5)        
        else:
            vmax=0.8
            edgecolor='w'
            ind_overlapping = np.ones_like(Mtomo, dtype=bool)
            ind_non_overlapping = np.zeros_like(Mtomo, dtype=bool)
            extent = [13.0, 16, 10, 16]
            err_yticks = np.arange(0.3,0.65,0.05)
            ax_err.set_xlim(13.0, 15.75)
            ax_histy.set_xticks(ticks = np.arange(10,75, 20))
            ax_histx.set_yticks(ticks= np.arange(10,100,20))
            ax.set_xlim(13.0,15.75)
        ax.set_ylim(10.5,15.75)

        if plot_kde :
            legendcolor='w'
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            Zmean = self.get_mean_KDE(offset=offset, z=z, sigma=sigma, th=th, lc=lc)
            im = ax.imshow(np.rot90(Zmean), cmap=plt.cm.viridis, extent = extent,
                           aspect='auto', vmin=0, vmax=vmax)
            if sigma=='2' :
                axins = inset_axes(ax,width="100%", height="100%", 
                                   bbox_to_anchor=(0.98,0.3, 0.02, 0.6),
                                   bbox_transform=ax.transAxes)
                cb_ticks_pos = 'left'
            else:
                axins = inset_axes(ax,width="100%", height="100%",
                                   bbox_to_anchor=(0.05,0.05, 0.02, 0.6),
                                   bbox_transform=ax.transAxes)
                cb_ticks_pos = 'right'
            cb = fig.colorbar(im, cax=axins, orientation='vertical',
                              ticks=np.arange(0,vmax+vmax+0.1,0.2))
            cb.ax.yaxis.set_tick_params(color='k')
            cb.ax.yaxis.set_ticks_position(cb_ticks_pos)
            cb.ax.yaxis.set_label_position(cb_ticks_pos)
            plt.setp(plt.getp(axins, 'yticklabels'), color=legendcolor)
            cb.ax.set_ylabel('Gaussian KDE', labelpad=5, color=legendcolor, fontsize=20)

        ind1 = (signif[ind_overlapping] > -2.5)*(signif[ind_overlapping] < -2.0)
        ind2 = (signif[ind_overlapping] > -3.0)*(signif[ind_overlapping] < -2.5)
        ind3 = (signif[ind_overlapping] > -3.5)*(signif[ind_overlapping] < -3.0)
        ind4 = (signif[ind_overlapping] < -3.5)

        with plt.rc_context({'scatter.marker':'o', 'patch.facecolor':'None',
                             'lines.markersize':'5','font.family':'serif'}) :
            if sigma=='4':
                ax.scatter(Mtomo[ind_overlapping][ind1], GroupMass[ind_overlapping][ind1],
                           label=r'$-2.5 < \rm \delta_F / \sigma_{map} < -2.35$',
                           edgecolors='r', facecolor='None', alpha=0.65 ,marker='D',
                           lw=3, s=50)
                ax.scatter(Mtomo[ind_overlapping][ind2], GroupMass[ind_overlapping][ind2],
                           label=r'$-3.0 < \rm \delta_F / \sigma_{map} < -2.5$',
                           edgecolors='C1', facecolor='None', alpha=0.65, marker='s',
                           lw=3, s=50)
            else:
                ax.scatter(Mtomo[ind_non_overlapping], GroupMass[ind_non_overlapping],
                           label='New watersheds',edgecolors='C1', facecolor='none',
                           alpha=0.8, marker='*', lw=3, s=50)
            ax.scatter(Mtomo[ind_overlapping][ind3], GroupMass[ind_overlapping][ind3],
                       label=r'$-3.5 < \rm \delta_F / \sigma_{map} < -3.0$',edgecolors='b',
                       facecolor='None', alpha=0.65, marker='o', lw=3, s=50)
            ax.scatter(Mtomo[ind_overlapping][ind4], GroupMass[ind_overlapping][ind4],
                       label=r'$ \ \rm \delta_F / \sigma_{map} < -3.5$',edgecolors=edgecolor,
                       facecolor='none', alpha=0.6, marker='D', lw=3, s=50)

            ax.plot(np.arange(13,16.5),np.arange(13,16.5), label='1:1', color='r', alpha=0.6, ls='--')


            ax.set_yticks(ticks= np.arange(11,16,0.5))
            ax.set_xticks(ticks=[])
            ax.set_ylabel('log'+r'$ \rm \left(\  M_{desc} (h^{-1} M_{\odot})\right) $', fontsize=30)


            plt.setp(ax.get_xticklabels(), fontsize=20)
            plt.setp(ax.get_yticklabels(), fontsize=20, rotation=20)
            ax.tick_params(axis="x", labelbottom=False)
            ax.grid(True, alpha=0.7)
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histx.set_ylabel('count')
            ax_histy.set_xlabel('count')

            co = self.get_the_mean_fit(z=z, th=th, lc=lc, offset=offset, sigma=sigma)

            #co_rand = get_the_mean_fit(random=True)
            #Mfit_rand = co_rand[0]*x + co_rand[1]
            #ax.plot(x, Mfit_rand, label='random')

            bins = np.arange(np.min(Mtomo), np.max(Mtomo)+2*bin_size,bin_size)
            ax_histx.hist(Mtomo, bins=bins, histtype='step', color='C0')
            plt.setp(ax_histx.get_yticklabels(), fontsize=20)

            bins = np.arange(np.min(GroupMass), np.max(GroupMass)+2*bin_size,bin_size)
            ax_histy.hist(GroupMass, bins=bins, histtype='step', 
                          label='Detected \n Watersheds', color='C0', orientation='horizontal')

            if (sigma =='4')*(z==2.4):
                f, peaks, _ = self.read_files(n=1, z=z, sigma=sigma, th=th, lc=lc, noiseless=False, random=True)
                M1, M2, _ = self.get_Mtomo_GroupMass(f, peaks, offset=offset, noiseless=False)
                bins = np.arange(np.min(M2), np.max(M2)+2*bin_size,bin_size)
                ax_histy.hist(M2, bins=bins, histtype='step', label='random \n watersheds',
                              color='C1', orientation='horizontal', ls='--')
            else :
                bins = np.arange(np.min(GroupMass), np.max(GroupMass)+2*bin_size,bin_size)
                ax_histy.hist(GroupMass[ind_non_overlapping], bins=bins, histtype='step',
                              label="Detected \n only in \n this map", color='C1', 
                              orientation='horizontal', ls='--')

            plt.setp(ax_histy.get_xticklabels(), fontsize=20)
            if sigma=='4':
                ax_histy.set_xticks(ticks = np.arange(10,75, 20))
            if sigma=='2':
                ax_histy.set_xticks(ticks = np.arange(10,115, 50))
            ax_histx.grid(True, alpha=0.7)
            ax_histy.grid(True, alpha=0.7)
            ax_histy.legend(loc=(0,0.01), fontsize=17, frameon=False)
            def _plot_uncertainty_M0_paper(ax_err):                    
                with plt.rc_context({'scatter.marker':'o', 'patch.facecolor':'k',
                                     'lines.markersize':'7', 'font.family':'serif'}) :

                    #bins = np.arange(np.min(Mtomo),np.max(Mtomo)+bin_size+0.05,bin_size)
                    #bins = np.delete(bins, -2)
                    mbin = np.array([(bins_err[b]+bins_err[b+1])/2 for b in range(0, bins_err.size-1)])
                    _ , err = self.test_fit(co=co, z=z, th=th, lc=lc, offset=offset, bins=bins_err, sigma=sigma)
                    ax_err.errorbar(x=mbin, y=np.mean(err, axis=0), yerr=np.std(err, axis=0),
                                    marker='o', label=ax_err_label,  markersize=10)

                    Mfit = co[0]*(mbin-14) + co[1]

                    ax.plot(mbin, Mfit, label='power-law estimator', color='k', alpha=0.5)

                    from scipy.stats import linregress
                    if plot_noiseless:
                        co_noiseless = linregress(Mtomo_noiseless-14, GroupMass_noiseless)
                        bins_noiseless = np.arange(np.min(Mtomo_noiseless),
                                                   np.max(Mtomo_noiseless)+bin_size,bin_size)
                    
                        bins_noiseless = np.delete(bins_noiseless, -2)
                        
                        mbin_noiseless = np.array([(bins_noiseless[b]+
                                                    bins_noiseless[b+1])/2 for b in
                                                   range(0, bins_noiseless.size-1)])
                        err_true = self._get_err(co=co_noiseless, M1=Mtomo_noiseless, M2=GroupMass_noiseless, bins=bins_noiseless)
                        ax_err.plot(mbin_noiseless, err_true, marker='s', ls='--',
                                    label='Noiseless', color='C2')

                    #ax_err.legend(framealpha=0, loc=(0.02,0.15))
                    ax_err.legend(framealpha=0, loc='upper right')

                    ax_err.set_xlabel('log'+ r'$\ \rm \left( M_{tomo} (h^{-1} M_{\odot})\right) $',
                                      fontsize=30)
                    ax_err.set_ylabel('error on '+r'$\rm M_{desc}$'+'(dex)',
                                      fontsize=25)

                    ax.legend(fontsize=20, loc='lower right', framealpha=0, labelcolor=legendcolor)


                    ax_err.set_yticks(ticks= err_yticks)
                    ax_err.set_xticks(ticks= np.arange(13.0,16,0.5))
                    if sigma=='4':
                        ax_err.set_ylim((0.20,0.7))

                    plt.setp(ax_err.get_xticklabels(), rotation=20, fontsize=20)
                    plt.setp(ax_err.get_yticklabels(), rotation=20, fontsize=20)
                    ax_err.grid(True, alpha=0.7)


            _plot_uncertainty_M0_paper(ax_err)

class Environments():
    def __init__(self, lmap, peaks, cofm_file, all_clusters_file):
        self.lmap = lmap
        self.peaks = peaks
        self.cofm_file = cofm_file
        self.all_clusters_file = all_clusters_file
        
        self._get_detected_progs()

    def _get_detected_progs(self, mlim=[13.5, np.inf]):
        """Find and count the detected progenitors more massive than mlim"""
        #parentid = peaks['parentid'][:]
        fcofm = h5py.File(self.cofm_file,'r')
        all_clusters = h5py.File(self.all_clusters_file,'r')
        # Select only massive progenitors
        Mass = np.log10(all_clusters['Mass'][:])+10
        indm = np.where((Mass > mlim[0])*(Mass < mlim[1]))[0]
        indm = np.isin(fcofm['cluster_ind'][:], indm, assume_unique=True)
        prog = np.zeros((205,205,205))
        x, y, z = fcofm['x'][:][indm].astype(int), fcofm['y'][:][indm].astype(int), fcofm['z'][:][indm].astype(int)
        # Count progenitors from 1 not 0
        prog[x,y,z] = fcofm['cluster_ind'][:][indm]+1

        detected_progs, peakid = np.array([]), np.array([])

        for i in range(1,self.peaks['x'].size+1):
            contour = np.zeros((205,205,205), dtype=int)
            contour[self.lmap==i] = 1
            overlap_progs = np.unique(prog*contour).astype(int)[1::]
            # Return the prog counting to start from 0 (index of the cluster at z=0)
            detected_progs = np.append(detected_progs, overlap_progs-1)
            peakid = np.append(peakid, np.ones_like(overlap_progs)*i)
    
        self.detected_progs = detected_progs
        self. undetected_progs = np.array(list(set(fcofm['cluster_ind'][:]) 
                                           - set(self.detected_progs)))
        self.peak_id = peakid
    
    def large_over_density(self, DMConv, z, r):
        from . import minima
        from astropy.cosmology import Planck15 as cosmo
        
        fcofm = h5py.File(self.cofm_file,'r')
        
        x, y, z = fcofm['x'][:].astype(int), fcofm['y'][:].astype(int), fcofm['z'][:].astype(int)

        def _iterate_over_structs(ClusterInd):
            """Iterate over clusters and save the mass in Mass"""
            MeanDens = np.array([])
            for c, d in enumerate(ClusterInd):
                neighbourhood = np.zeros((205,205,205), dtype=int)
                ind = np.where(fcofm['cluster_ind'][:]==np.int(d))[0]
                xn, yn, zn = minima.tophat_mask(x[ind], y[ind], z[ind], Lx=205,
                                                Ly=205, Lz=205, maxrad=r)
                neighbourhood[xn, yn, zn] = 1
                #DMMeanDen_to_rhoc = (cosmo.Om0 + cosmo.Ob0) / cosmo.Om0
                MeanDens = np.append(MeanDens, np.sum(DMConv*neighbourhood)/xn.size)  
            return MeanDens

        DeltaMean_det = _iterate_over_structs(ClusterInd=self.detected_progs)


        DeltaMean_un = _iterate_over_structs(ClusterInd=self.undetected_progs)

        return DeltaMean_det, DeltaMean_un

    def plotMdesc_Msph(self, DMConv, z, r, savefig=None):
        # Setup the figure
        DeltaMean_det, DeltaMean_un = self.large_over_density(DMConv=DMConv, z=z, r=r)
        left, width = 0.15, 0.60
        bottom, height = 0.1, 0.60
        spacing = 0
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # Plotting part
        fcofm = h5py.File(self.cofm_file,'r')
        all_clusters = h5py.File(self.all_clusters_file,'r')
        Mass = np.log10(all_clusters['Mass'][:])+10


        print(DeltaMean_un.size, self.undetected_progs.size)
        print(DeltaMean_det.size, self.detected_progs.size)
        ax.scatter(DeltaMean_un, Mass[self.undetected_progs.astype(int)], s=160, facecolor='w', marker='.', edgecolor='C1', alpha=0.8, label='undetected', linewidth=3)
        ax.scatter(DeltaMean_det, Mass[self.detected_progs.astype(int)], marker='o',
                   s=100, facecolor='none', edgecolor='C0', alpha=0.4, label='detected', linewidth=3)
        ax.grid(True, alpha=0.6)
        ax.legend()

        ax_histx.hist(DeltaMean_det, histtype='step', label='Detected')
        ax_histx.hist(DeltaMean_un, histtype='step', ls='--', label='Undetected')
        ax_histx.legend(loc=(1.01,0.5))
        ax_histx.set_yticks([50,150,300])

        ax_histy.hist(Mass[self.detected_progs.astype(int)], histtype='step',
                      orientation='horizontal')
        ax_histy.hist(Mass[self.undetected_progs.astype(int)], histtype='step',
                      orientation='horizontal', ls='--')
        ax_histy.set_xticks([50,150, 300])

        ax.set_xlabel(r'$\rm \frac{<\rho_{DM}>_{4 cMpc/h}}{<\rho_{DM}>} $', fontsize=45)
        ax.set_ylabel(r'$\rm M_{z=0} \ (h^{-1} \  M_{\odot})$')
        plt.tight_layout()
        #plt.suptitle('r = '+str(r), fontsize=40)

    def plotMdesc_Msph_contours(self, DMConv, z, r, savefig=None, xmin=0, xmax=4, ymin=13.5, ymax=15.25):
        # Setup the figure
        DeltaMean_det, DeltaMean_un = self.large_over_density(DMConv=DMConv, z=z, r=r)
        left, width = 0.15, 0.60
        bottom, height = 0.1, 0.60
        spacing = 0
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # Plotting part
        fcofm = h5py.File(self.cofm_file,'r')
        all_clusters = h5py.File(self.all_clusters_file,'r')
        Mass = np.log10(all_clusters['Mass'][:])+10


        print(DeltaMean_un.size, self.undetected_progs.size)
        print(DeltaMean_det.size, self.detected_progs.size)
        kde_un = self.get_kde(x=DeltaMean_un, y=Mass[self.undetected_progs.astype(int)], xmin=xmin, xmax=xmax,
                              ymin=ymin, ymax=ymax)
        kde_det = self.get_kde(x=DeltaMean_det, y=Mass[self.detected_progs.astype(int)], xmin=xmin, xmax=xmax,
                               ymin=ymin, ymax=ymax)
        binsx = np.linspace(xmin,xmax,50)
        binsy = np.linspace(ymin,ymax,50)
        xmbin = [(binsx[b]+binsx[b+1])/2 for b in range(binsx.size-1)]
        ymbin = [(binsy[b]+binsy[b+1])/2 for b in range(binsy.size-1)]
        X, Y = np.meshgrid(binsx,binsy)
    
        cs = ax.contour(X.T, Y.T, kde_det, colors='C0', levels=[0.05,0.32])
        cs = ax.contour(X.T, Y.T, kde_un, colors='C1', levels=[0.05,0.32], linestyles='dashed')
        ax.grid(True, alpha=0.6)
        ax.legend()

        ax_histx.hist(DeltaMean_det, histtype='step', label='Detected')
        ax_histx.hist(DeltaMean_un, histtype='step', ls='--', label='Undetected')
        ax_histx.legend(loc=(1.01,0.5))
        ax_histx.set_yticks([50,150,300])

        ax_histy.hist(Mass[self.detected_progs.astype(int)], histtype='step',
                      orientation='horizontal')
        ax_histy.hist(Mass[self.undetected_progs.astype(int)], histtype='step',
                      orientation='horizontal', ls='--')
        ax_histy.set_xticks([50,150, 300])

        ax.set_xlabel(r'$\rm \frac{<\rho_{DM}>_{4 cMpc/h}}{<\rho_{DM}>} (z=2.5)$', fontsize=45)
        ax.set_ylabel(r'$\rm M_{z=0} \ (h^{-1} \  M_{\odot})$')
        plt.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
        
    def get_kde(self, x, y, xmin, xmax, ymin, ymax):

        from scipy.stats import gaussian_kde as gkde

        data = np.zeros((2,x.size))
        data[0,:] = x
        data[1,:] = y

        kernel = gkde(data)  
        X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        grid_points = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(grid_points).T, X.shape)
        del X, Y
        return Z
    
