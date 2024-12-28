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
input_settings['obs_path'] = "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/"
input_settings['training_period'] = ['2019-01-01T00:00:00.000000000', '2019-02-05T23:00:00.000000000'] # 240 total instances.
input_settings['testing_period'] = ['2019-02-10T00:00:00.000000000', '2019-02-14T23:00:00.000000000'] 
input_settings['validation_period'] = ['2019-03-20T00:00:00.000000000', '2019-03-24T23:00:00.000000000'] 
input_settings['lat_range'] = [30., 20.] # only focus on NA region # 41
input_settings['long_range'] = [260, 275] # ideal for East Coast # 61
input_settings['variables_list'] = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
# input_settings['flag'] = 'train' # train, test or val
input_settings['seq_len'] = 12 # t-instances fed to model  
input_settings['d_model'] = 4 * input_settings['seq_len'] # could experiment with this.
input_settings['pred_len'] = 2 # t-instances output by model 
input_settings['batch_size'] = 4
input_settings['num_epochs'] = 3    
input_settings['learning_rate'] = 1e-3
input_settings['e_layers'] = 3 # num. of encoding layers
input_settings['checkpoints'] = './checkpoints/' # model checkpoint locations
# no. of successive validation batch runs with worsening loss 
# post which the model training for the current epoch will be stopped
# done to prevent overfitting on training data.
input_settings['patience'] = 3 
input_settings['time_enc'] = 0 # 0 or 1.    

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

# model = Model(
#     model_params
# )

exp = Exp(input_settings)
exp_name = '{}_{}-la_{}-{}-lo_{}-{}-nv-{}-sl{}-dm{}-pl{}-bs{}-ne{}-lr{}-el{}'.format(
    input_settings['model_type'],
    input_settings['obs_path'][42:-1],
    input_settings['lat_range'][0],
    input_settings['lat_range'][1],
    input_settings['long_range'][0],
    input_settings['long_range'][1],
    len(input_settings['variables_list']),
    input_settings['seq_len'],
    input_settings['d_model'],
    input_settings['pred_len'],
    input_settings['batch_size'],
    input_settings['num_epochs'],
    input_settings['learning_rate'],
    input_settings['e_layers'],   
)


#look into this, what does this do?
# fix_seed = 2023
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

exp.train(exp_name)
exp.test(exp_name)