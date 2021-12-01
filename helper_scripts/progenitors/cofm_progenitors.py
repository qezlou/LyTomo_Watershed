import numpy as np
import h5py
from scipy.ndimage import label

def get_cofm(savefile='cofm_progenitors.hdf5', L=205):
  """ A code to find the Center of Mass of each individual cluster progenitor """
  X, Y, Z, cluster_id = np.array([]), np.array([]), np.array([]),np.array([])
  with h5py.File('./clusters_TNG300-1.hdf5','r') as f:
      ind = np.where((f['Mass'][:]>10**3.75) * (f['Mass'][:]<10**4.0))[0]
  # interate over cluster progenitors
  a = np.append(np.arange(248),ind[1::])
  for j in a:
    new_num_features = 0
    with h5py.File('./prog_maps/map_PC_prog_R200_cluster'+str(j)+'.hdf5','r') as f:
        m = (f['map'][:]*f['num_parts'][()]/(205**3))
    labeled_array, num_features = label(m[:] > 0 )
    xcm, ycm, zcm, num_parts = np.array([]), np.array([]), np.array([]), np.array([])
    # interate over all islands a cluster progenitors is spread over (Peridoc boundary condition
    # breaks cluster progenitor into pieces
    for i in range(num_features):
      indp = np.where(labeled_array==i+1)
      num_parts= np.append(num_parts, np.sum(m[indp]))
      xcm= np.append(xcm, np.sum(indp[0]*m[indp])/num_parts[-1])
      ycm= np.append(ycm, np.sum(indp[1]*m[indp])/num_parts[-1])
      zcm= np.append(zcm, np.sum(indp[2]*m[indp])/num_parts[-1])
    # Take care of periodic boundary condition to get Center Of Mass
    if num_features > 1:
      if np.any(xcm > L/2)*np.any(xcm < L/2):
         ind = np.where(xcm < L/2)
         xcm[ind] += L
      if np.any(ycm > L/2)*np.any(ycm < L/2):
         ind = np.where(ycm < L/2)
         ycm[ind] += L
      if np.any(zcm > L/2)*np.any(zcm < L/2):
         ind = np.where(zcm < L/2)
         zcm[ind] += L
    # Store the center of mass for entire inividual cluster progenitors
    X = np.append(X, (np.sum(xcm*num_parts)/np.sum(num_parts))%L)
    Y = np.append(Y, (np.sum(ycm*num_parts)/np.sum(num_parts))%L)
    Z = np.append(Z, (np.sum(zcm*num_parts)/np.sum(num_parts))%L)
    cluster_id = np.append(cluster_id, j)
  with h5py.File(savefile, 'w') as fw:
    fw['x'] = X
    fw['y'] = Y
    fw['z'] = Z
    fw['cluster_id'] = cluster_id




         
      
           
             
             
               
            
