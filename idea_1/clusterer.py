# a test code to cluster a sub-section of NA (part of East Coast) based on u10m values:
# latitude range: 30, 20 
# longitude range: 260, 275
# time range: '2019-01-01T00:00:00.000000000', '2019-01-05T23:00:00.000000000'

import numpy as np
import xarray as xr

obs_path = "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/"
data = xr.open_zarr(obs_path)
date_picked = ['2019-01-01T00:00:00.000000000', '2019-01-05T23:00:00.000000000'] # 30 day tr, 10 day te.

lats = data['latitude'].values # 721
lons = data['longitude'].values # 1440

# lat_range = [90., 45.] # only focus on NA region
# long_range = [200, 250]

lat_range = [30., 20.] # only focus on NA region # 41
long_range = [260, 275] # ideal for East Coast # 61
 
data_u10 = data['10m_u_component_of_wind']
data_u10 = data_u10.sel(time = slice(*date_picked))
data_u10 = data_u10.sel(latitude = slice(*lat_range), longitude = slice(*long_range)) # 120, 41, 61

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

data_matrix = data_u10.to_numpy().reshape(data_u10.shape[0], data_u10.shape[1] * data_u10.shape[2]).T #(41*61, 120)
print(data_matrix.shape)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_matrix)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(normalized_data)
print(labels.shape)

trunc_lats = lats[(lats <= lat_range[0]) & (lats >= lat_range[1])]
trunc_lons = lons[(lons >= long_range[0]) & (lons <= long_range[1])]
grid_labels = labels.reshape(len(trunc_lats), len(trunc_lons))


import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
axs.coastlines()
lons_2d, lats_2d = np.meshgrid(trunc_lons, trunc_lats)
sc1 = axs.pcolormesh(lons_2d, lats_2d, grid_labels, cmap='Spectral_r', vmin=grid_labels.min(), vmax=grid_labels.max(), transform=ccrs.PlateCarree())
plt.colorbar(sc1, ax=axs, label='Z-surface')

unique_labels = np.unique(grid_labels)
for label in unique_labels:
    # Find indices of the grid points belonging to this cluster
    cluster_indices = np.argwhere(grid_labels == label)

    # Compute the mean longitude and latitude for this cluster
    cluster_lons = lons_2d[grid_labels == label]
    cluster_lats = lats_2d[grid_labels == label]
    center_lon = cluster_lons.mean()
    center_lat = cluster_lats.mean()

    # Annotate the cluster center with the label
    axs.text(
        center_lon, center_lat, str(label),
        color='black', fontsize=10, fontweight='bold',
        ha='center', va='center',
        transform=ccrs.PlateCarree()
    )

# Add a title
axs.set_title("k-means Clustering with 5 centers")
plt.savefig('u10m_lat_range_30_20_long_range_260_275_time_5days_5_centers.png')
plt.show()

np.save('labels_lat_range_30_20_long_range_260_275_time_5days_5_centers.npy', grid_labels)


