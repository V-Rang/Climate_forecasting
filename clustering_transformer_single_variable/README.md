Clustered attention transformer for a single variable data (Nt x Ny x Nx) stored in a numpy array. Allows for:
1. attention masking (zero out attention for non-cluster points).
2. time encoding (additional *query* and *key* corresponding to time-instance of variable value).
3. wavelet transform (Wavelet Transformation for data pre-processing using [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)).