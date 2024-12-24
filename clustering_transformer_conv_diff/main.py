import numpy as np
import torch
import torch.nn as nn
# make clusters, rollling  5 days each.

import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
# from data_provider.dataset_maker import DatasetCreate
from models.clustered_transformer import Model
import gcsfs
from torch import optim
from experiments.exp_template import Exp

# refine how you supply arguments needed:
# make a dictionary:
input_settings = {}
input_settings['model_type'] = 'clustered_transformer'
# input_settings['obs_path'] = "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/"
input_settings['obs_path'] = "snapshots_CD.npy"
input_settings['seq_len'] = 40 # t-instances fed to model  
# input_settings['d_model'] = 4 * input_settings['seq_len']
input_settings['d_model'] = 512 # could experiment with this too.
input_settings['pred_len'] = 30 # t-instances output by model 
input_settings['batch_size'] = 4 # vary as  2, 4, 8, 16, 32, ....
input_settings['num_epochs'] = 100    
input_settings['learning_rate'] = 1e-3
input_settings['e_layers'] = 4 # num. of encoding layers (could experiment with this)
input_settings['checkpoints'] = './checkpoints/' # model checkpoint locations
# no. of successive validation batch runs with worsening loss 
# post which the model training for the current epoch will be stopped
# done to prevent overfitting on training data.
input_settings['patience'] = 3 

# the model will take in (b, l, v, s) shaped input.
# three choices:
# 1. normalize all 'b' samples in the batch.
# 2. normalize samples individually (b * s).
# 3. No normalization.

input_settings['normalization_flag'] = 'batch' # batch, sample or None

# input_settings['use_norm_batch'] = False
# input_settings['use_norm_sample'] = True

# eg input = 24 (1 day of data), output = 1 (1 hour in the future)
# data_set, data_loader = DataLoaderCreate(input_settings)
# print(len(data_set))

# import torch
# print(torch.cuda.is_available())  # True
# print(torch.cuda.device_count())  # 2
# print(torch.cuda.current_device())  # 0
# print(torch.cuda.get_device_name(0))  # NVIDIA A100 80GB PCIe

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# inputs, labels = inputs.to(device), labels.to(device)


exp = Exp(input_settings)

exp_name = '{}_{}-sl{}-dm{}-pl{}-bs{}-ne{}-lr{}-el{}_with_attention'.format(
    input_settings['model_type'],
    input_settings['obs_path'][42:-1],
    input_settings['seq_len'],
    input_settings['d_model'],
    input_settings['pred_len'],
    input_settings['batch_size'],
    input_settings['num_epochs'],
    input_settings['learning_rate'],
    input_settings['e_layers'],   
)

exp.train(exp_name)
exp.test(exp_name)