import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter as gf
from . import spectra_mocking as sm
from . import  minima
from . import load_data

class Paint:
    """ A class to paint stoughs on flux map """
    def __init__(self, m_dach, dim, sigma, thresh, linking_contour=-2, load_BH=False, load_halos=False, load_progen_DM=False, load_progen_Gas=False, load_protoc=True, pc=None, periodic_bound=True):

        self.m_dach = m_dach
        self.dim = dim
        self.sigma = sigma
        self.thresh = thresh
        self.linking_contour = linking_contour
        if load_BH :
            self.data_BH = load_data.load_BH()
        if load_halos :
            self.halos_coords = load_data.load_halos()
        if load_progen_DM :
            self.DM_coords = load_data.load_progen_DM()
        if load_progen_Gas:
            self.Gas_coords = load_data.load_progen_Gas()
        if load_protoc :
            if pc is None :
                m_conv = gf(self.m_dach, self.sigma, mode='wrap')
                pc = minima.find_extrema_v3(mapconv = m_conv, thresh=self.thresh, linking_contour=-self.linking_contour, periodic_bound=periodic_bound)
            pc_coords = np.empty(shape=(pc['x'].size, 3))
            # Switch x and y. As we want to overplot it on plt.imshow map.(converting position on matrix m to plt.imshow map) -- I switched back to normal
            pc_coords[:,0], pc_coords[:,1], pc_coords[:,2]  =  pc['x'], pc['y'], pc['z']
            self.pc_coords = pc_coords
            self.pc = pc
            """
            # This "if" is used for the selected peaks in true map, pass the indices to pc_ind argument
            if pc_ind.size > 0 :
                self.pc_ind = pc_ind
            # if pc_indis empty, then we overlay all the peaks
            else :
                self.pc_ind = np.arange(pc['x'].size)
            """

    def _within_linking_length(self, coords, z, deltaz=3):
        """Getting the objects within +- b parameter of FOF"""
        mask = np.ones((coords.shape[0],), dtype=bool)
        #mask *= coords[:,2] == z 
        mask *= coords[:,2] >= z-deltaz
        mask *= coords[:,2] <= z+deltaz

        return coords[mask].astype(int), mask


    def circle(self, ax, x0, y0, r, c):
        """Draw a circle on axis ax"""
        box_size=self.m_dach.shape[0]
        x0, y0, r = np.array(x0), np.array(y0), np.array(r)
        for i in range(x0.size):
            x, y = np.array([]),np.array([])
            for t in np.arange(0,2*np.pi,0.1):
                x = np.append(x, (x0[i] + r[i]*np.cos(t))%(box_size) )
                y = np.append(y,  (y0[i] + r[i]*np.sin(t))%(box_size) )
            mask =  (x>0)*(x<self.m_dach.shape[0])*(y>0)*(y<self.m_dach.shape[0])
            x, y =x[mask], y[mask]
            ax.scatter(x,y,c=c, s=5, alpha=0.9)

    def _plot_Rsig_circle(self, ax, z, c, pc_ind):

        d = np.abs(self.pc_coords[:,2][pc_ind] - z)
        ind = np.where( self.pc['Rsig'][pc_ind]-d >=0 )
        r  = np.sqrt(self.pc['Rsig'][pc_ind][ind]**2 - d[ind]**2)
        self.circle(ax=ax, x0=self.pc_coords[pc_ind][ind][:,1], y0=self.pc_coords[pc_ind][ind][:,0], r=r, c=c)

    def plot_2D_flux(self, ax, z, true_map=False, contour_color='k', only_contour=False, plot_Rsig=True,levels=[-7, -6, -5, -4, -3, -2], cmap_reversed=True, color_bar=True, pc_ind=np.array([])):
        """plots the flux map at coordinate z = z on the axis passed to the function"""
        # Get the location of protoclusters and smoothed map
        #if m is None :
        #   m = np.fromfile('./spectra/maps/map_'+map_file+'.dat').reshape((dim[0], dim[1], dim[2]))
        m = gf(self.m_dach, self.sigma, mode='wrap')
        m = m/np.std(m)
        if cmap_reversed:
            cmap = plt.get_cmap('jet').reversed()
        else :
            cmap = plt.get_cmap('jet')
        m_2D = m[:,:,z]
        if true_map :
            m_2D = m_2D.T
        if not only_contour:
            im = ax.imshow(m_2D, cmap=cmap, origin='lower', extent=[0, self.dim[0], 0, self.dim[1]], interpolation='bilinear', vmin=-4, vmax=3)
        ax.contour(m_2D, levels= levels, origin='lower', colors=contour_color)
        try :
            DM_coords, _ = self._within_linking_length(self.DM_coords, z=z)
            ax.scatter(DM_coords[:,1], DM_coords[:,0], color='b', marker='*', s=10)
        except AttributeError: pass
        try :
            g_coords, _ = self._within_linking_length(self.Gas_coords, z=z)
            ax.scatter(g_coords[:,1], g_coords[:,0], color='m', marker='.', s=5)
        except AttributeError: pass
        # Overplot protoclusters
        try :
            # pc_ind is not empty only if we want to plot the closest peaks in true map to the peaks in mock map
            if pc_ind.size == 0 :
                pc_ind = np.arange(self.pc['x'].size)
                
            p_coords, mask = self._within_linking_length(self.pc_coords[pc_ind,:], z=z)
            # all protoclusters in a single z
            ax.scatter(p_coords[:,1], p_coords[:,0], color=contour_color, marker='*', s=50)
            if plot_Rsig:
                self._plot_Rsig_circle(ax=ax, z=z, c=contour_color, pc_ind=pc_ind)

        except AttributeError: pass
        try :
          BH_coords, _ = self._within_linking_length(self.data_BH['coords'], z=z)                    
          # all BHs in a single z                                                      
          ax.scatter(BH_coords[:,1], BH_coords[:,0], color='k', marker='*', s=50)     
        except AttributeError: pass
        try :
          h_coords, _ = self._within_linking_length(self.halos_coords, z=z)                         
          ax.scatter(h_coords[:,1], h_coords[:,0], color='w',  alpha=0.5 ,s=50)       
        except AttributeError: pass

        ax.set_title('z_coordonate = '+str(z)+r'$h^-1 cMpc$')
        ax.set_xlim(0,self.dim[0])                                                               
        ax.set_ylim(0,self.dim[1])
        if not only_contour :
            return ax, im
        else :
            return ax


    
