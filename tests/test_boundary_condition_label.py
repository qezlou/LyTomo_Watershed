import numpy as np
import scipy
from scipy.ndimage import label
import minima
def test_2D():

    a = np.zeros(shape=(5,5,5))
    a[0,0,0], a[0,1,0], a[1,0,0] = 1, 1, 1
    a[4,4,0], a[4,3,0], a[3,4,0] = 2, 2, 2
    a[3,2,0], a[3,2,4] = 5, 7
    return mass_calcs.boundary_condition_label(a)
