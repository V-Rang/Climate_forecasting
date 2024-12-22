import torch
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

def ClusterDetermine(in_arr : torch.tensor) -> torch.tensor:
    
    '''
    input: (a, b, c)
    output: cluster labels (a, b)
    i.e. for each of the 'a' batches, get the label array corresp.
    to the 'b' samples, each of length 'c'.

    for now, using KMeans from sklearn.cluster with 5 centers, function needs to be made extensible to
    other clustering methods.
    '''

    in_arr = in_arr.detach().numpy()
    k = 5
    scaler = StandardScaler()
    labels = np.zeros((in_arr.shape[0], in_arr.shape[1]), dtype = int)

    for i in range(in_arr.shape[0]):
        batch_data = in_arr[i] 
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(batch_data) # normalizing each (l, v, s) sample.        
        kmeans = KMeans(n_clusters = k, random_state = 7)
        labels[i] = kmeans.fit_predict(normalized_data)

    return torch.from_numpy(labels)
