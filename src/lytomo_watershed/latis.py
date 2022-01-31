import numpy as np


class Latis:

    def __init__(self, pixfile='./spectra/maps/mapsv13/dcv13pix.dat', mapfile='./spectra/maps/mapsv13/dcv13map.dat',idsfile='./spectra/maps/mapsv13/dcv13ids.dat', shapemap=[93,51,483]):
        self.map = np.fromfile(mapfile).reshape(shapemap)
        self.spec = np.fromfile(pixfile)
        self.spec = self.spec.reshape(int(self.spec.shape[0]/5), 5)
        self.spec_id = np.fromfile(idsfile).astype(int)
        self.spec_num = np.max(self.spec_id)+1
        
        def get_missed_spectra():
            # There exist some missed spectra (no pixel corresponds to them
            a=[]
            for i in range(self.spec_num):
                ind = np.where(self.spec_id == i)
                if np.size(ind) == 0:
                    a.append(i)
            return a
        self.missed_spectra = np.array(get_missed_spectra())
        # An rray of spectra excluding missed ones
        self.avail_spec = np.delete(np.arange(0,self.spec_num), self.missed_spectra)

    def get_deltaF_spec(self):
        """ read deltaF on each pixel along all the spectra

	"""
        ind = np.where(self.spec_id == 1)
        spec_len = []
        deltaF_ordered = np.empty(shape=(np.size(self.avail_spec),), dtype=np.ndarray)
        for (i,j) in enumerate(self.avail_spec):
            ind = np.where(self.spec_id == j)
            deltaF_ordered[i]= self.spec[:,4][ind]

        return deltaF_ordered




