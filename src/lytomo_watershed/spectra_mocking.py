import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
import fake_spectra
from fake_spectra.plot_spectra import PlottingSpectra as PS
from fake_spectra.spectra import Spectra
from fake_spectra import spec_utils
import matplotlib.pyplot as plt
import array
#import compat_tomo # Some funcs from tomo package By Drew Newman made compatible with my version of python
from fake_spectra import fluxstatistics as fstat
from . import z_bin
from . import minima
from astropy.cosmology import Planck15 as cosmo

def get_submap_hist(m, bins, lratio):
    """
    Get the histogram for subvolumes of TNG with the same size as LATIS
    """
    sidelen = int(205/lratio)
    hist = np.zeros(shape=(lratio**3,bins.size-1))
    for i in range(lratio):
        for j in range(lratio):
            for k in range(lratio):
                mask = np.zeros(shape=m.shape, dtype=bool)
                mask[i*sidelen:(i+1)*sidelen,j*sidelen:(j+1)*sidelen,k*sidelen:(k+1)*sidelen]=True
                hist[(lratio**2)*k+lratio*j+i] = np.histogram(np.ravel(m[mask]), bins=bins, density=True)[0]
    return hist


def plot_flux_pdf_v3(ax, lratio, z_range, filename, label, over_sigma=True) :
    """A function to plot the flux pdf. It shows the cosmic variance. 
    lratio : ratio of the side-length of LATIS to our mock. It is used for cosmic variance.
    z_range : The z range in LATIS to be used
    file_name : Path to the mock map
    label : label on the elegend for the mock
    over_sigma : If true it devides deltaF by it's std
    """
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    mock_num = 1
    npiece=lratio**3
    if over_sigma :
        bins = np.arange(-6,5,0.2)
    else :
        bins = np.arange(-.8, .8, .02)
        
    mbins = [(bins[i]+bins[i+1])/2 for i in range(bins.size-1)]
    sigma = [1,2,4]
    #fig, ax = plt.subplots(2,3,figsize=(18,11))
    for i in range(len(sigma)):
        hist_all = np.zeros(shape=(mock_num*npiece,np.size(mbins)))
        # LATIS
        mL = get_latis_map(latis_file='../spectra/maps/mapsv13/dcv13map.dat', map_version=13, z_range=z_range, sigma=sigma[i])
        mL = np.ravel(mL)
        if over_sigma :
            histl = np.histogram(np.ravel(mL/(np.std(mL))), bins=bins, density=True)[0]
        else:
            histl = np.histogram(np.ravel(mL), bins=bins, density=True)[0]
        
        #Mocks
        for j in range(1,mock_num+1):
            mmask = np.fromfile(filename).reshape(205,205,205)
            mmask = gaussian_filter(mmask,sigma=sigma[i],mode='wrap')
            if over_sigma :
                hist_all[(j-1)*npiece:j*npiece] = get_submap_hist(m=mmask/np.std(mL), bins=bins, lratio=lratio)
            else:
                hist_all[(j-1)*npiece:j*npiece] = get_submap_hist(m=mmask, bins=bins, lratio=lratio)
            
        
        median = np.array([np.median(hist_all[:,b]) for b in range(len(mbins))])
        std = np.array([np.std(hist_all[:,b]) for b in range(len(mbins))])
        ax[0][i].fill_between(mbins, median-std, median+std,color='orange',label=label)
        ax[1][i].fill_between(mbins, median-std, median+std,color='orange',label=label)
        ax[0][i].fill_between(mbins, median-2*std, median+2*std,color='orange', alpha=0.5)
        ax[1][i].fill_between(mbins, median-2*std, median+2*std,color='orange', alpha=0.5)
        ax[0][i].set_title(r'$\mathrm{\sigma_{sm} }=$'+str(sigma[i])+'cMpc/h', fontsize=30)
        ax[0][i].plot(mbins, histl, 'k',linestyle='dashed',label='LATIS '+str(z_range[0])+'<z<'+str(z_range[1]))
        ax[1][i].plot(mbins, histl, 'k',linestyle='dashed',label='LATIS '+str(z_range[0])+'<z<'+str(z_range[1]))
        ax[0][i].set_yscale('log')
        ax[1][i].set_yscale('linear')
        #ax[0][i].legend(loc='lower center')
        ax[0][i].grid('True')
        ax[1][i].grid('True')

        if over_sigma:
            ax[1][i].set_xlabel(r'$\mathrm{\delta_F / \sigma_{LATIS}}$')
            ax[0][i].set_xlabel(r'$\mathrm{\delta_F / \sigma_{LATIS}}$')
            ax[0][i].set_xlim(-6,4)
            ax[1][i].set_xlim(-6,4)
            ax[0][i].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0][i].tick_params(which='both', width=2, direction='in')
            ax[0][i].tick_params(which='major', length=7, labelsize=30,direction='in')
            ax[0][i].tick_params(which='minor', length=4,direction='in')
            
            ax[0][i].set_ylim(1e-6,0.7)

        else :
            ax[1][i].set_xlabel(r'$\mathrm{\delta_F} $')
            ax[0][i].set_xlim(-0.8,0.5)
            ax[1][i].set_xlim(-.6,.8)
            ax[0][i].set_ylim(1e-6,10)

            
        ax[0][0].set_ylabel('Pdf- logscale')
        ax[1][0].set_ylabel('Pdf')
    plt.tight_layout()


def _plot_mock_pdf(filename, color, ind, ax_main, ax_res, hist_latis, sigma, dim, bins):
    """ A helper method to plot the flux PDF and the residual for mocks """
    hist_all = []
    #m_true = get_true_map('spectra_Base2_z2.3_true.hdf5')
    #if sigma is not None :
    #    m_true = convolve(m_true, sigma=sigma)
    #hist_true = np.histogram(m_true, bins=bins, density=True)[0] 
    snap_range = range(1,20)
    if filename[0:3] == 'TNG':
        dim = [205, 205, 205]
    else :
        dim = [60, 60, 60]
    for j in snap_range:
            #(mbin, hist) = get_pdf_Wiener_filtered(dim=dim, map_file=filename+'_n'+str(j), bins=bins, sigma=sigma)
            (mbin, hist) = get_pdf_Wiener_filtered(dim=dim, map_file='TNG_z2.6_n'+str(j)+'_no_thresh', bins=bins, sigma=sigma)
            mbin=mbin[ind]
            hist=hist[ind]
            hist_all.append(hist)
    hist_all = np.array(hist_all)
    y_1=[]
    y_2=[]
    y_3=[]
    y_4=[]
    for b in range(0,np.shape(hist_all)[1]):
        mean = np.mean(hist_all[:,b])
        std = np.sqrt(np.var(hist_all[:,b]))
        #y_1.append(mean+*std)
        y_1.append(np.max(hist_all[:,b]))
        y_2.append(np.min(hist_all[:,b]))
        y_3.append(mean+2*std)
        #y_2.append(mean-*std)
        y_4.append(mean-2*std)

    #ax_main.fill_between(mbin, y_1, y_2, label=label, color=color, alpha=0.4)
    ax_main.plot(mbin, y_1, color=color, alpha=0.7)
    ax_main.plot(mbin, y_2, color=color, linestyle='dashed',alpha=0.7)
    #ax_main.plot(mbin , hist_true[ind], color = 'k', alpha=0.7, label='true map')
    #ax_main.fill_between(mbin, y_3, y_4, color=color, alpha=0.2)
    #ax_res.fill_between(mbin, y_1/hist_latis - 1, y_2/hist_latis - 1, color=color, alpha=0.4)
    ax_res.plot(mbin, y_1/hist_latis - 1, color = color, alpha = 0.7)
    ax_res.plot(mbin, y_2/hist_latis - 1, linestyle='dashed',color = color, alpha = 0.7)
    #ax_res.plot(mbin, hist_true[ind]/hist_latis - 1, color='k', alpha=0.7 )
    #ax_res.fill_between(mbin, (y_3/hist_latis -1), (y_4/hist_latis -1), linestyle='dotted',color=color, alpha=0.5)
    return y_1/hist_latis, y_2/hist_latis




def plot_pdf_Wiener_filtered(z_range, linear=True,sigma= None,mapfile=[], color=[], label=[],bins=np.arange(-1.0, 1.0, 0.02), figname='pdf_map_maskedp5.png'):
    
    plt.figure(figsize=(10,5))
    ax_res = plt.axes([0.1, 0.1, 0.8,0.5])
    ax_main = plt.axes([0.1, 0.6+0.005, 0.8, 0.3])
    if linear:
        ax_main.set_yscale('linear')
    else :
        ax_main.set_yscale('log')
    #ax_main.set_ylim((0,4))
    ax_res.set_xlabel(r'$\delta_F$',fontsize=20)
    ax_res.set_xlim((-0.5, 0.5))
    ax_res.set_ylim((-1.5,1.5))
    ax_main.set_xlim((-.5,0.5))
    
    ax_main.set_ylabel(r'$ pdf \ of \ voxels$',fontsize=14)
    ax_res.set_ylabel('(Sim - LATIS) / LATIS', fontsize=14)


    mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
    
    (mbin, hist_latis) = get_pdf_Wiener_filtered(map_file='obs', bins=bins, z_range=np.around(z_range, decimals=1), sigma=sigma)
    ind = hist_latis > 0.0
    hist_latis = hist_latis[ind]
    mbin = mbin[ind]
    ax_main.plot(mbin, hist_latis, label='LATIS '+str(np.around(z_range[0], decimals=1))+'< z <'+ str(np.around(z_range[1],decimals=1)), linestyle='dashed', color= 'b')
    #ax_main.plot(mbin, hist_latis, label='LATIS ', linestyle='dashed', color= 'b')
    ax_res.plot(mbin, np.zeros_like(mbin), linestyle='dashed', color='b')
    y1 = []
    y2 = []
    for i in range(np.size(mapfile)):
        if mapfile[i][0:4] == 'Base':
            dim = np.array([60,60,60]).astype(int)
        if mapfile[i][0:3] == 'TNG':
            dim = np.array([205,205,205]).astype(int)
        dim = np.array([205,205,205]).astype(int)
        y1t, y2t = _plot_mock_pdf(sigma=sigma, dim=dim,filename = mapfile[i], color='orange', ind=ind, ax_main=ax_main, ax_res=ax_res, hist_latis=hist_latis, bins=bins)
        y1.append(y1t)
        y2.append(y2t)

    if (sigma == 4)*(z_range == [2.4,2.6]):
        # A vertical line shwoing the region related to protoclusters
        ax_res.axvline(x=-0.3, color='c')
        ax_res.axvline(x=-0.05, color='c')
    #if linear :
    plt.legend(fontsize=13, loc='upper left')
    #else :
    #    plt.legend(fontsize=13, loc='center')
    #plt.title(r'$\sigma_{smoothing} \ = \ $'+str(sigma))
    plt.savefig(figname)
    #return y1, y2, mbin

    ftrue = h5py.File('spectra/maps/map_TNG_true_0.5Mpc_z2.6.hdf5','r')
    #m = convolve(ftrue['map'][:],4)
    m = gaussian_filter(ftrue['map'][:],4, mode='wrap')
    bins=np.arange(-1.0, 1.0, 0.02)
    mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
    hist = np.histogram(m, bins=bins, density=True)[0]
    ax_main.plot(mbin[ind], hist[ind], label='true map',color='g')


    return ax_main, ax_res

def get_latis_map(latis_file = '../spectra/maps/mapsv13/dcv13map.dat', map_version = 13, z_range=[2.2, 2.8], sigma=None):
    """
    A helper function to load the LATIS map. It is masking the edges and un-observed regions
    latis_file : Path to latis map
    map_verison : It is needed since different versions have different un-observed regions
    z_rage : The redshift of the LATIS you want
    sigma : if None, smooth the map with sigma
    """
    if map_version == 9 :
        m = np.fromfile(latis_file).reshape((63, 51, 483))
    if (map_version == 11) or (map_version==13):
        m = np.fromfile(latis_file).reshape((93, 51, 483))
    # Tansform redshift range  to h^-1 cMpc
    r_range = z_bin.radial_distance(z_range).astype(int)
    # Smooth the map
    if sigma is not None :
        m = gaussian_filter(m, sigma=sigma)
    mask=np.zeros(m.shape).astype(np.bool)
    # excluding the neighborhood of the observational boundary
    if r_range[0] < 4 :
        r_range[0]=4
    if r_range[1] > 479:
        r_range[1]=479
        
    mask[4:-4,4:-4,r_range[0]:r_range[1]+1]=True
    if map_version == 9 :
        mask[:34,:28,:]=False
    if map_version == 11 :
        mask[60:,:26,:]=False
    
    print('LATIS mean deltaF  at z='+str(z_range)+' is ', np.mean(m[mask]) )
    return m[mask]

def get_pdf_Wiener_filtered(dim=None ,map_file='120_1024',bins=np.arange(-1.0, 1.0, 0.02), z_range=None, sigma=None, map_version=13):
    
    
    mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])

    if map_file=='obs':
        m = get_latis_map(map_version=map_version, z_range=z_range, sigma=sigma)
        hist = np.histogram(m, bins=bins, density=True)[0]
    else :
        m = np.fromfile('./spectra/maps/map_'+map_file+'.dat').reshape((dim[0], dim[1], dim[2]))
        # Smoothing the map
        if sigma is not None:
            m =  gaussian_filter(m, sigma=sigma, mode='wrap')
        hist = np.histogram(m, bins=bins, density=True)[0]

    return mbin, hist

def find_boundary_curves(hist_all):
    """ a method to find the boundary curves so can fill the region in between later 
    takes, hist_all: a list of all curves
    returns, (y_1, y_2): The min and max values of curves in each bin
    """
    rows_size = np.shape(hist_all)[0]
    columns_size = np.shape(hist_all)[1]
    all_curves = np.zeros(shape=(rows_size, columns_size))
    y_1 = np.zeros(shape=(columns_size,))
    y_2 = np.zeros(shape=(columns_size,))

    for i in range(rows_size):
        all_curves[i,:] = hist_all[i]
    for i in range(columns_size):
        y_2[i] = np.max(all_curves[:,i])
        y_1[i] = np.min(all_curves[:,i])

    return(y_1, y_2)


def plot_pdf_spectra(spectra_file=['randspectra_120_1024_res_123.hdf5','spectra_As_1_29.hdf5','spectra_As_2_24.hdf5'], bins=np.arange(-1.0, 1.0, 0.02)):
    mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
    plt.figure(figsize=(15,13))
    plt.title('flux along sightlines')
    # for LATIS
    deltaF = latis.spec[:,4]
    pdf_LATIS = np.histogram(deltaF, bins =bins, density=True)[0]

    for i in spectra_file:
        #f =h5py.File('./spectra/'+i,'r')
        #lines = np.size(f['spectra']['axis'])
        #f.close()
        lines=2448
        #deltaF = get_deltaF(savefile='./spectra/'+i, lines=lines, mean_flux_corr=True, noisy=True)
        #pdf = np.histogram(deltaF, bins=bins, density=True)[0]
        #plt.semilogy(mbin, pdf, label='MF True, noise True')
        CNR = get_CNR(lines=lines)
        CE = get_CE(CNR)
        """
        deltaF = get_deltaF(savefile='./spectra/'+i, lines=lines, CNR=CNR, CE=CE)
        pdf = np.histogram(deltaF, bins=bins, density=True)[0]
        plt.semilogy(mbin, pdf, label='CE, CNR ON')
        """
        """
        deltaF = get_deltaF(savefile='./spectra/'+i, lines=lines)
        pdf = np.histogram(deltaF, bins=bins, density=True)[0]
        plt.semilogy(mbin, pdf, label='No Noise')
        """
        #plt.semilogy(mbin, pdf_LATIS, label='LATIS', color='k')

        deltaF = get_deltaF(savefile='./spectra/'+i, CNR=10*CNR, CE=0.1*CE)
        pdf = np.histogram(deltaF, bins=bins, density=True)[0]
        plt.semilogy(mbin, pdf, label='CE, CNR ON, 1/10 noise', linestyle='dotted')
        """
        deltaF = get_deltaF(savefile='./spectra/'+i, lines=lines, CNR=10*CNR, CE=0.1*CE)
        pdf = np.histogram(deltaF, bins=bins, density=True)[0]
        plt.semilogy(mbin, pdf, label='CE, CNR ON, 1/10 noise', linestyle='dotted')
        """

    plt.semilogy(mbin, pdf_LATIS, label='LATIS', color='k')




    plt.xlabel(r'$\delta_F$')
    plt.ylabel(r'$pdf$')
    plt.legend(loc='center')

    
    plt.savefig('pdf_spectra_noise.png')

def plot_noisy_spectrum(flux=False, noise = True, spec_num=320, xlims=(-1500,1500), savefile='randspectra_120.hdf5', savefig='flux.png', color='blue', lines=10):

    
    if noise ==True :
        CNR = get_CNR(lines=lines)
        ## Get CE for eontinuum error
        CE = get_CE(CNR)
    else :

        CNR = None
        CE = None

    ps = PlottingSpectra(num = 1, base='./', savedir='./', savefile=savefile, res = 127, spec_res = 145)

    ps.plot_spectrum(elem='H', ion=1, line=1215, spec_num=spec_num, xlims =xlims,flux=flux, color=color)
    #plt.title(savefileno noise)
    #plt.savefig('spectrum'+str(spec_num)+'_flux_'+str(flux)+'_noise_'+str(noise)+'_spec_res_added_120.png')
    #plt.savefig(savefig)


def write_input_dachshund(res, spec_res,savefile= 'randspectra_120_1024_res_123.hdf5', output_file='pix_MF_ON_120_1024_NO_CE_FULL_noise.dat',output_file_id='pix_MF_ON_120_102', floor_correction=True, noise=True, add_CE=None, add_CNR=None ,MF_corr=True):
    """ Write a binary file of [X, y, z, sigma_deltaF, deltaF] of each pixel.
        Each row would be info for a single pixel
        Arguments:
        res : pixel resolution in km/s
        spec_res : spectral resolution in km/s
        savefile: the fake spectra file
        output_file : the file interpretable by dachshund

    """
    # A binary file file to write the result in
    out = open(output_file, 'wb') 
    ps_no_noise = PS(num = 1, res=res, spec_res=spec_res,base='./', savedir='', savefile=savefile)
    lines = np.size(ps_no_noise.axis)
    #CNR *= np.ones(shape=(lines,))
    #CE *= np.ones(shape=(lines,))
    # Pick 2% of them as Quasars
    np.random.seed(94)
    QSO = np.random.randint(0, lines, size=int(0.02*lines))
    # Drew's implementtion
    #QSO = np.array([])
    ## Calculate total noise
    #if np.all(CNR==np.inf) :
        #CNR = 2*get_CNR(lines=lines, z=ps_no_noise.red, QSO=QSO)
        #CE = get_CE(CNR)
    #else :
    CNR = get_CNR(lines=lines, z=ps_no_noise.red, QSO=QSO)
    #    if np.all(CE!=0):
    CE = get_CE(CNR)
    #if noise ==False:
    #    CNR=100*CNR
    #    CE = get_CE(CNR)
    if add_CNR is not None:
        add_CNR = CNR
    if add_CE is not None :
        add_CE = CE
    (deltaF, current_mean_flux, mask) =  get_deltaF(res, spec_res, savefile= savefile, CE= add_CE, CNR= add_CNR,MF_corr=MF_corr)
    deltaF = deltaF.reshape(lines, int(np.size(deltaF)/lines))
    ## noise for pixels along each spectrum
    #if CE is not None:
    tot_noise = np.sqrt(CE**2 + (1/CNR)**2)/current_mean_flux

    #else:
    #    tot_noise = (1./(1.*CNR))/current_mean_flux
    ## Baed on LATIS paper page 22, I add a flooe for total estimated noise sigma_delta > 0.2
    if floor_correction :
        #if np.all(CNR==np.inf)*np.all(CE==0):
        #    tot_noise = 1.0*np.ones_like(tot_noise)
        #else :
        ind_low_noise = np.where(tot_noise < 0.2)[0]
        tot_noise[ind_low_noise] = 0.2
    ## ZERO noise
    #tot_noise= np.zeros(shape=(lines,))
    ## factor find the position of the pixels along the spectrum
    # the size of each pixel in h^-1 cMpc
    dy = ps_no_noise.box / ps_no_noise.nbins
    for i in range(lines):
        for j in range(int(np.size(deltaF)/lines)):
            if ~mask[i,j]:
                array.array('d', [ps_no_noise.cofm[i][0]/1000.,  ps_no_noise.cofm[i][1]/1000.,  dy*j*1.0/1000. , tot_noise[i], deltaF[i,j]]).tofile(out)
    out.close()

def write_input_dachshund_v2(spec_res, addpix=10, savefile= 'randspectra_120_1024_res_123.hdf5', output_file='pix_MF_ON_120_1024_NO_CE_FULL_noise.dat', floor_correction=True, add_CE=None, add_CNR=None ,MF_corr=True, domask=True, seed=14):
    """ Write a binary file of [X, y, z, sigma_deltaF, deltaF] of each pixel (In this new version, the spectra is averaged along z
    to mock what happens in observations.)
        Each row would be info for a single pixel
        Arguments:
        spec_res : spectral resolution in km/s
        addpix : How many adjacent pixels along the spectra should be avergaed to get
        the detector's pixel resolution
        savefile: the fake spectra output
        output_file : the file interpretable by dachshund
        floor_correction : Inpout the noise floor adopted in LATIS
        add_CNR, add_CE : If each True, add noise and continuum fitting error respectively
        MF_corr : If True, correct the mean flux with get_mean_flux() method
        domask : If True, mask stronger absorbers with Ew > 5 A with the same window of 5 A
    """
    # A binary file to write the result in
    out = open(output_file, 'wb') 
    ps = PS(res= None, num = 1, base='./', savedir='', savefile=savefile)
    num_spectra = np.size(ps.axis)
    # Pick 2% of them as Quasars
    np.random.seed(94)
    QSO = np.random.randint(0, num_spectra, size=int(0.02*num_spectra))
    CNR = get_CNR(num_spectra=num_spectra, z=ps.red, QSO=QSO, seed=seed)
    CE = get_CE(CNR)
    if add_CNR is not None:
        add_CNR = CNR
    if add_CE is not None :
        add_CE = CE
    (deltaF, current_mean_flux, mask) =  get_deltaF_v2(specfile=savefile, res=None, spec_res=spec_res/ps.dvbin, addpix=addpix, CE= add_CE, CNR= add_CNR,MF_corr=MF_corr, domask=domask)
    deltaF = deltaF.reshape(num_spectra, int(np.size(deltaF)/num_spectra))
    ## noise for pixels along each spectrum
    tot_noise = np.sqrt(CE**2 + (1/CNR)**2)/current_mean_flux

    ## Bawed on LATIS paper page 22, I add a floor for total estimated noise sigma_delta > 0.2
    if floor_correction :
        ind_low_noise = np.where(tot_noise < 0.2)[0]
        tot_noise[ind_low_noise] = 0.2
    ## factor find the position of the pixels along the spectrum
    # the size of each pixel in h^-1 cMpc
    #dy = ps.box / ps.nbins
    points = 0
    dy = ps.box/deltaF.shape[1]
    for i in range(num_spectra):
        for j in range(int(np.size(deltaF)/num_spectra)):
            if ~mask[i,j]:
                points+=1
                array.array('d', [ps.cofm[i][0]/1000.,  ps.cofm[i][1]/1000.,  dy*j*1.0/1000. , tot_noise[i], deltaF[i,j]]).tofile(out)
    out.close()


def get_deltaF(spec_res, savefile='randspectra_120.hdf5', CNR=None, CE= None, MF_corr=True):
    """Calculates deltaF = (F/F_average) - 1   for each peixel
    res : pixel resolution, it should match the one used in fake_spectra to genrate the spectra
    spec_res : spectral resolution, a gaussian with this correlation length will be convolved
    CNR, CE, MF_corr : random, continumm noise and Mean Flux correction to be activated or not
    """
    ps = PS(num = 1, base='./', savedir='', savefile=savefile)
    tau = ps.get_tau(elem='H', ion=1, line=1215)
    mean_flux_desired = get_mean_flux(z=ps.red)
    print('initial mean flux =', np.mean(np.exp(-tau)))
    print('True MF:', mean_flux_desired)
    if MF_corr :
        NHI = np.sum(ps.get_col_density(elem='H', ion=1), axis=1)
        ind = np.where(NHI < 10**19.)
        flux = correct_mean_flux(tau=tau, mean_flux_desired=mean_flux_desired, ind=ind)
    else:
        flux=np.exp(-tau)
    print('after corre:', np.mean(flux)) 
    #Spectra with noise 
    if CE is not None:
        if CNR is not None:
            # the order below is important
            (flux, delta) = ps.add_cont_error(CE=CE, flux=flux)
            #t = np.clip(1/CNR, 0.2*np.mean(np.ravel(flux)), np.inf)
            #CNR = 1/t
            (flux,noise_array) = ps.add_noise(snr=CNR, flux=flux)
        else:
            (flux, delta) = ps.add_cont_error(CE=CE, flux=flux)
    else:
        if CNR is not None:
            (flux, noise_array) = ps.add_noise(snr=CNR, flux=flux)
    # Smoothing the spectra by spectral resolution
    flux = spec_utils.res_corr(flux, ps.dvbin, spec_res)
    mask = np.zeros_like(flux,dtype=bool)
    current_mean_flux = np.mean(np.ravel(flux))
    for i in range(flux.shape[0]):
        mask[i,:] = mask_strong_absb_v2(deltav=ps.dvbin, Fnorm=flux[i,:], maxdv=1000, Fm=current_mean_flux, ewmin=5)
    flux = np.ravel(flux)
    print('mean flux after noise =', current_mean_flux)
    print ("*** Error on mean flux :*** ", current_mean_flux-mean_flux_desired)

    # over flux for each pixel
    #deltaF = (flux/(1.0*np.mean(flux))) - 1
    deltaF = (flux/current_mean_flux) - 1

    return (deltaF, current_mean_flux, mask)

def get_deltaF_v2(specfile, res, spec_res, addpix=int(10), CNR=None, CE=None, MF_corr=True, domask=True):
    """Get the optical depth and return the realistic mock spectra 
    specfile : Address to the spectra. It should be in the foramt as the fale_spectra outputs
    spec_res : spectral resolution in units of voxels along the spectrum
    addpix : make a coarser spectrum by averaging this number of consecutive pixels along the line-of-sight
    CNR : Continuum to Nosie ratio
    CE : Continumm error
    MF_corr : If true, correct the mean flux of the spectra
    domask : If true, mask strong absorbtions along the spectrum
    """
    ps = PS(res=res, num = 1, base='./', savedir='', savefile=specfile)
    spec_file = h5py.File(specfile, 'r')
    if MF_corr:
        try :
            # If HI density is recorded, do not use the high column density 
            # sightlines for fixing the mean flux. 
            NHI = spec_file['colden/H/1'][:]
            ind = np.where(np.sum(NHI,axis=1)<10**19)
        except (KeyError, np.AxisError, AttributeError):
            # It is only for FGPA spectra, as we do not know the exact HI density
            ind = np.ones_like(spec_file['tau/H/1/1215'][:], dtype=bool)
        mean_flux_desired = get_mean_flux(z=spec_file['Header'].attrs['redshift'])
        flux = correct_mean_flux(tau=spec_file['tau/H/1/1215'][:], mean_flux_desired=mean_flux_desired, ind=ind)
    
    from scipy.ndimage.filters import gaussian_filter1d
    flux = gaussian_filter1d(flux, spec_res, axis=-1, mode='wrap')
    L = np.shape(flux)[1]
    # Check if the last pixel is fixed
    t = np.arange(0,L+1,addpix)
    new_flux = np.zeros(shape=(np.shape(flux)[0], t.size-1))
    #new_NHI = np.zeros(shape=(np.shape(NHI)[0], t.size))
    # Averaging over the flux within a pixel
    for i in range(t.size-1) :
        new_flux[:,i] = (np.sum(flux[:,t[i]:t[i+1]], axis=1))/addpix
    if CE is not None:
        if CNR is not None:
            # the order below is important
            (new_flux, delta) = ps.add_cont_error(CE=CE, flux=new_flux)
            # A bit of hack, solve it later
            ps.nbins = int(L/addpix)
            (new_flux,noise_array) = ps.add_noise(snr=CNR, flux=new_flux)
        else:
            (new_flux, delta) = ps.add_cont_error(CE=CE, flux=new_flux)
    else:
        if CNR is not None:
            ps.nbins = int(L/addpix)
            (new_flux, noise_array) = ps.add_noise(snr=CNR, flux=new_flux)
    if domask :
        mask = np.zeros_like(new_flux,dtype=bool)
        for i in range(new_flux.shape[0]):
            mask[i,:] = mask_strong_absb_v2(deltav=addpix*ps.dvbin, Fnorm=new_flux[i,:], CNR=CNR[i]*np.ones(shape=(new_flux.shape[1],)), maxdv=1000, Fm=np.mean(new_flux), ewmin=5)
    else :
        mask = np.zeros(shape=new_flux.shape, dtype=bool)
    new_flux = np.ravel(new_flux)
    current_mean_flux = np.mean(np.ravel(new_flux))
    print('mean flux after noise =', current_mean_flux)
    print ("*** Error on mean flux :*** ", current_mean_flux-mean_flux_desired)
    # flux contrast for each pixel
    #deltaF = (new_flux/(1.0*np.mean(new_flux))) - 1
    deltaF = (new_flux/current_mean_flux) - 1

    return (deltaF, current_mean_flux, mask)


def get_mean_flux(z, metal=False) :
    """ get the mean flux used in LATIS Faucher-Giguere 2008"""
    if metal :
        # The below is not good for HI absorption as includes the effect of metals
        return np.exp(-0.001845*(1+z)**3.924)
    else :
        # The below is good for only HI absorptions, does not include metal absorption
        return np.exp(-0.001330*(1+z)**4.094)

def correct_mean_flux(tau, mean_flux_desired, ind=None):
    """ returns the corrected flux to have a desired mean flux
    arguments:
    tau : optical depth BEFORE adding noise to it
    ind: indices to spectra with NHI < 10^19. Only them being used for finding the scale. This is according to 
    Faucher-Giguere et al 2008. mean flux without including any metals. 
    mean_flux_desired:
    returns: The flux after scaling the optical depth to have mean_flux_desired = <e^(-scale * tau)>

    """
    if ind is not None:
        scale = fstat.mean_flux(tau[ind], mean_flux_desired)
    else :
        scale = fstat.mean_flux(tau, mean_flux_desired)
        
    flux = np.exp(-scale * tau)

    return flux

def input_MDPL2(read_file='/lustre/scratch/mqezlou/MDPL2/000truefluxsm.sav', save_file='pix_MDPL2.dat', sigma=2, noise=True, CE=True, rand_seed=23):
    """ Wite the imput for Dachshund for Multi Dark Matter Planck 2 simulations 
    arguments: 
    sigma: which smoothed field is desired, options 2, 3 and 4 h^-1 cMpc
    """
    out = open(save_file, 'wb')
    mock = MDPL2(read_file = read_file, sigma=sigma, rand_seed=rand_seed)
    num_spec = np.shape(mock.spec)[0]
    # Correct the mean flux
    print('mean flux : \n before correction: ', np.mean(mock.spec))
    print('desired = ', get_mean_flux(z=2.535))
    mock.spec = mock.spec.astype(np.float64)
    mock.spec = correct_mean_flux(-np.log(mock.spec), mean_flux_desired= get_mean_flux(z=2.535))
    print('after correction=', np.mean(mock.spec))
    # calculate total noise
    CNR = get_CNR(lines=num_spec)
    CE = get_CE(CNR)
    mock.add_noise(CNR=CNR, CE=CE)
    print('after noise addition: ', np.mean(mock.spec))
    mock.spec = mock.spec - np.mean(mock.spec)
    if noise == False:
       CNR=100*CNR
       CE = get_CE(CNR)

    if CE is not None:
       tot_noise = np.sqrt(CE**2 + (1/CNR)**2)
    else:
       tot_noise = 1./(1.*CNR)
     
    # Force a floor for noise 
    ind_low_noise = np.where(tot_noise < 0.2)[0]
    tot_noise[ind_low_noise] = 0.2
     
    for i in range(num_spec):
        for j in range(mock.L_z):
            array.array('d', [mock.x[i]*mock.voxel_size[2], mock.y[i]*mock.voxel_size[1], j*mock.voxel_size[0], tot_noise[i], mock.spec[i,j]]).tofile(out)

    out.close()

def get_flux_power_1D_MDPL2(read_file='/lustre/scratch/mqezlou/MDPL2/000truefluxsm.sav', rand_seed=23, sigma=2):
    mock = MDPL2(read_file = read_file, sigma=sigma, rand_seed=rand_seed)
    mock.spec = mock.spec.astype(np.float64)
    mock.spec = correct_mean_flux(-np.log(mock.spec), mean_flux_desired= get_mean_flux(z=2.535))

    (kf, avg_flux_power) = fstat.flux_power(tau=-np.log(mock.spec)[:,92], vmax=12825.45, spec_res=None, window=False)

    return(kf, avf_flux_power)


def get_CNR(num_spectra, z, QSO=[], DCV13_model=True, seed=14):
   """ Calculate Continuum to noise ratio (signal to noise ratio) modeled in LATIS
   QSO contains the sightline number of quasars.
   """
   CNR = np.zeros(num_spectra)
   np.random.seed(seed)

   for ii in range(num_spectra):
       if ii in QSO:
           if DCV13_model:
               CNR[ii] = np.exp(np.random.normal(2.3, 1.2))

       else:
           mean = 0.84 + 0.99 * (z - 2.5)- 1.82*(z - 2.5)**2
           CNR[ii] = np.exp(np.random.normal(mean, .43))
        

   return CNR


def get_CE(CNR) :
    """ Calculate Continuum noise for each spectra modeled in LATIS"""
    CE = 0.24*CNR**(-0.86)
    CE[np.where(CE < 0.05)] = 0.05

    return CE

def mask_strong_absb(deltav, Fnorm, maxdv=1000, Fm=None, ewmin=5):
    """ Borrowed from Drew Newman
    Fnorm: is a 1D spectrum normalized to the continuum level. 
    Fm is the mean flux, so <Fnorm> ~ Fm.
    deltav: is the velocity interval per pixel in km/s. 
    ewmin is the threshold rest-frame equivalent width
    to make a line. 
    A boolean mask is returned that is True within lines exceeding ewmin.
    """
    minimum = (Fnorm < np.roll(Fnorm, 1)) & (Fnorm < np.roll(Fnorm, -1))
    minimum[0], minimum[-1] = True, True
    mask = np.zeros(len(Fnorm), dtype=bool)
    abovemean = (Fnorm >= Fm)
    L = len(Fnorm)
    parr = np.arange(L)
    varr = parr * deltav
    dlam = deltav / 299792. * 1215.67 # rest-frame A per pixel
    for j in np.where(minimum)[0]:
        # Find endpoints of line
        frst = np.where((parr < parr[j]) & abovemean)[0]
        lst = np.where((parr > parr[j]) & abovemean)[0]
        if frst.size==0: 
            frst = np.where((parr < np.max(parr)) & abovemean)[0][-1]
            pf = -1*(np.max(parr)-parr[frst])
        else :
            frst = frst[-1]
            pf = parr[frst]
        if lst.size==0: 
            lst = np.where((parr > 0) & abovemean)[0][0]
        pl = parr[lst]
        #frst, lst = frst[-1], lst[0]
        # Above is not exactly like the data. We allow lines that begin/end at the boundaries
        #vcen = (varr[frst] + varr[lst]) / 2.0
        pcen = int((pf + pl)/2.0)%np.max(parr)
        #inwin = np.abs(varr - vcen) < maxdv
        b = pcen - int(1000/deltav)
        e = pcen + int(1000/deltav)
        inwin = np.zeros(L,dtype=bool)
        if b < e : inwin[b:e+1]=True
        else : inwin[0:e+1], inwin[b,-1] = True, True
        # Note here the EW is defined relative to the unabsorbed continuum, not the mean flux.
        # It doesn't really matter much, but this is consistent with the data treatment.
        ew = np.sum(1 - Fnorm[inwin]) * dlam
        if ew > ewmin:
            if frst < lst :mask[frst:lst+1] = True
            else : mask[0:lst+1], mask[frst:-1] = True, True
    return mask

def mask_strong_absb_v2(deltav, Fnorm, CNR, maxdv=1000, Fm=None, ewmin=5):
    """
    Masking the strong absorbers with EW > 5 A
    deltav : The velocity interval per pixel in km/s
    Fnorm : The 1D spectrum normalized to the continuum level.
    CNR = Continuum to Noise ratio in each pixel along the sightline,
    it should be the same for all pixels on that spectrum for LATIS mock spectra
    Fm : the mean flux, so <Fnorm>~Fm
    maxdv: the velocity width over which we integrate to calculate the EW
    ewmin : The minimum EW of an absorber to be masked
    returns : A boolean array with size of the spectra, True in pixels
    need to be masked.
    """
    L = len(Fnorm)
    dlam = deltav / 299792. * 1215.67 # rest-frame A per pixel
    mask = np.zeros(L, dtype=bool)
    from scipy.ndimage import label
    Fnorm = np.convolve(Fnorm, np.ones(5)/5)[2:-2]
    lspec, _ = label((Fnorm < Fm).astype(int))
    # If the whole spectrum is under the mean flux mask all of it
    if np.where(lspec)[0].size==L :
        mask = np.ones(L, dtype=bool)
        return mask
    # Periodic Boundary Condition
    bc = False
    if (lspec[0])*(lspec[-1]):
        bc=True
        #lspec[lspec==lspec[-1]] = lspec[lspec==lspec[0]][0]
    if bc:
        l = np.unique(lspec).size - 2
    else:
        l = np.unique(lspec).size - 1
    for j in range(1,l+1):
        if j==1:
            if bc:
                ind = np.where(lspec==1)[0]
                ind0 = ind[-1]
                lastisland = np.unique(lspec).size - 1
                ind = np.where(lspec==lastisland)[0]
                ind1 = ind[0]
                cen = int((ind0 + -1*(L-ind1))/2)%L
                lspec[lspec==lspec[-1]] = lspec[lspec==lspec[0]][0]
            else :
                ind = np.where(lspec==j)[0]
                cen = int((ind[0]+ind[-1])/2)
        else :
            ind = np.where(lspec==j)[0]
            cen = int((ind[0]+ind[-1])/2)
        b, e = (cen-int(1000/deltav))%L, (cen+int(1000/deltav))%L
        inwin = np.zeros(shape=(L,),dtype=bool)
        if b<e : inwin[b:e] = True
        else : inwin[b:-1], inwin[0:e] = True, True
        # Note here the EW is defined relative to the unabsorbed continuum, not the mean flux.
        # It doesn't really matter much, but this is consistent with the data treatment.
        ew = np.sum(1 - Fnorm[inwin])*dlam
        # Only mask absorbers more sifnificant than the noise 
        dew = np.sqrt(np.sum(1/CNR[inwin]**2))*dlam
        if ew > ewmin and dew/ew < 5: mask[lspec==j]=True
    return mask

def get_sightline_num(pixfile='./spectra/maps/mapsv13/dcv13pix.dat', mapfile='./spectra/maps/mapsv13/dcv13map.dat',idsfile='./spectra/maps/mapsv13/dcv13ids.dat', latis_dim=[93,51,483]):
    from .latis import Latis
    """Finds number of sightlines in LATIS within the observed redshift span, BORROWED From Drew Newman
    pixfile, mapfile, idsfile : The LATIS data.
    latis_dim : The shape of the volume in LATIS.
    """
    latis = Latis(pixfile, mapfile, idsfile, shapemap=latis_dim)
    zpos = np.sort(np.unique(latis.spec[:,2]))
    redshifts = z_bin.cmpch_to_redshift(zpos)
    sightlines = np.array([np.sum(latis.spec[:,2]==zp) for zp in zpos])
    return (redshifts, sightlines)


def get_mock_sightline_number(z_range, pixfile='./spectra/maps/mapsv13/dcv13pix.dat', mapfile='./spectra/maps/mapsv13/dcv13map.dat', idsfile='./spectra/maps/mapsv13/dcv13ids.dat', latis_dim=[93,51,483], mock_dim=[205,205] ,unobs=None):
    """ Calculate the sigh-line number needed for mock to resemble latis 
    arguments:
    z_range is the redshift range we are interesyed in
    pixfile, mapfile, idsfile : The LATIS data
    latis_dim : The shape of the volume in LATIS.
    unob: is the size of the unobserved box
    mock_dim: x,y dimension of the mock
    returns:
    numbers of the spectra needed in mock

    """
    from .latis import Latis
    latis = Latis()
    redshifts, sightlines = get_sightline_num(pixfile, mapfile, idsfile, latis_dim)
    # Take the average sightline num within z_range
    ind = (redshifts >= z_range[0])*(redshifts <= z_range[1])
    num = np.mean(sightlines[ind])
    # excluding the unobserved part
    [xmin, ymin, zmin] = [np.min(latis.spec[:,0]), np.min(latis.spec[:,1]), np.min(latis.spec[:,2])]
    [xmax, ymax, zmax] = [np.max(latis.spec[:,0]), np.max(latis.spec[:,1]), np.max(latis.spec[:,2])]
    
    mid_z = (z_range[0] + z_range [1]) / 2
    zend_obs = redshifts[-1]
    # We need to correct the latis_area as the map's size is for high-z end of the observed box
    if unobs is not None:
        latis_area = (((xmax - xmin)*(ymax - ymin) - ((unob[0]-xmin)*(unob[1]-ymin))) * (cosmo.comoving_transverse_distance(mid_z)/cosmo.comoving_transverse_distance(zend_obs))**2).value()
    else :
        latis_area = (((xmax - xmin)*(ymax - ymin)) * (cosmo.comoving_transverse_distance(mid_z)/cosmo.comoving_transverse_distance(zend_obs))**2).value

    mock_area = mock_dim[0]*mock_dim[1]

    return num*mock_area/latis_area
    #return (num/latis_area)*(67.7**2)
    #return 0.2*mock_area

def get_spec_res(z=2.2, spec_res=2.06, pix_size=1.8):
    """ Calculates the pixel size (pix_size) and spectral resolution (spec_res) in 
    km/s for the MOCK SPECTRA.
    arguments: z, redshift. spec_res, spectral resoloution in Angst.  pixel_size
    in sngst.
    returns:
    (pixel_size, spec_res) in km/s
    """
    # conversion factor from Angstrom to km/s at any redshift
    conv_fac = 3e5*0.000823/(1+z)

    return(pix_size*conv_fac, spec_res*conv_fac)

def _get_flux_noiseless(specfile, addpix):
    """ A helper for get_true_map_v2() 
    It is almost identical to get_deltaF_v2() for mock maps
    specfile : spectra file for the true spectra
    addpix : number of consecutive pixels need to be summed
    to get the desired resolution
    """
    ps = PS(num = 1, base='./', savedir='', savefile=specfile, res=None)
    spec_file = h5py.File(specfile, 'r')
    NHI = spec_file['colden/H/1'][:]
    ind = np.where(np.sum(NHI,axis=1)<10**19)
    mean_flux_desired = get_mean_flux(z=spec_file['Header'].attrs['redshift'])
    flux = correct_mean_flux(tau=spec_file['tau/H/1/1215'][:], mean_flux_desired=mean_flux_desired, ind=ind)
    # Check if the last pixel is fixed
    L = np.shape(flux)[1]
    t = np.arange(0,L+1,addpix)
    new_flux = np.zeros(shape=(np.shape(flux)[0], t.size-1))
    new_NHI = np.zeros(shape=(np.shape(NHI)[0], t.size-1))
    # Averaging over the flux within a pixel
    for i in range(t.size-1) : 
        new_flux[:,i] = (np.sum(flux[:,t[i]:t[i+1]], axis=1))/addpix
        new_NHI[:,i] = np.sum(new_NHI[:,t[i]:t[i+1]], axis=1)
    new_flux = np.ravel(new_flux)
    new_NHI = np.ravel(new_NHI)
    current_mean_flux = np.mean(np.ravel(new_flux))
    print('mean flux after noise =', current_mean_flux)
    print ("*** Error on mean flux :*** ", current_mean_flux-mean_flux_desired)
    
    return new_flux, new_NHI


def get_noiseless_uniform_grid_map(specfile, addpix, savefile, boxsize=205, trans_sep=1.0):
    """Write a 3D matrix for the noiseless map, with averaing consecutive pixels along each specttum
    - specfile : The raw spectra generated with fake_spectra
    - addpix : the number of consecutive pixels to be averaged over
    - savefile : The hdf5 file name to save the result in
    - trans_sep : the transverse separation between sightlines in cMpc/h
    """
    f = h5py.File(specfile,'r')
    flux, NHI = _get_flux_noiseless(specfile,addpix)
    tdim= int(boxsize/trans_sep)
    flux = np.ravel(flux)
    NHI_map = np.ravel(NHI)
    m = (flux/np.mean(flux))-1
    flux = flux.reshape(tdim,tdim,int(flux.size/(tdim**2)))
    m = m.reshape(tdim,tdim,int(m.size/(tdim**2)))
    NHI_map = np.ravel(NHI)
    NHI_map = NHI.reshape(tdim, tdim, int(NHI.size/(tdim**2)))
    # I  am not sure why, but plotting the map says I need an (x,y) transpose
    for z in range(0, flux.shape[2]):
        flux[:,:,z] = flux[:,:,z].T
        m[:,:,z] = m[:,:,z].T
        NHI_map[:,:,z] = NHI_map[:,:,z].T
    if savefile is not None:
        with h5py.File(savefile, 'w') as ftrue :
            ftrue['flux'] = flux
            ftrue['map'] = m
            ftrue['redshift'] = f['Header'].attrs['redshift']
            ftrue['NHI'] = NHI_map
    else:
        return m


def get_vel_res(vmax, pixres):
    """Get the velocity bin needed to run the fake_spectra with,
    The certain number of these will be averaged to get the flux
    in each detector's pixel. """

    #vmax = (box*cosmo.H(z))/(cosmo.h*(1+z)) # Box size  (physical km/s)

    numpix = np.around(vmax/pixres)
    pixres = vmax/numpix

    return pixres/10

def make_pure_noise_map(z, L=205, l_delta=-.6, u_delta=.6):
    """Making a (L,L,L) oure noise map with LATIS parameters"""
    np.random.seed(88)
    # select sightlines to be QSO
    QSO = np.random.randint(0, L*L, size=int(0.02*L*L))
    CNR = get_CNR(lines=L*L, z=z, QSO=QSO)
    CE = get_CE(CNR)
    flux = np.ones(shape=(L*L,L))
    # ADD Continuum error, copied from fake_Spectra.spectra.add_cont_error()
    delta = np.empty(L*L)
    for ii in range(L*L):
        np.random.seed(2*ii)
        delta[ii] = np.random.normal(0, CE[ii])
        while (delta[ii] < l_delta) or (delta[ii] > u_delta):
            delta[ii] = np.random.normal(0, CE[ii])
            flux[ii,:] /= (1.0 + delta[ii])        
    
    # Add random noise
    noise_array = np.array([])
    for ii in range(L*L):
        np.random.seed(ii)
        noise = np.random.normal(0, 1./CNR[ii], L)
        noise_array = np.append(noise_array, noise)
        flux[ii]+= noise
    
    return flux.reshape(L,L,L)
    


    
