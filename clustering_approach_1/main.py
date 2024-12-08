import numpy as np
import torch
import torch.nn as nn
# make clusters, rollling  5 days each.

import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
# from data_provider.dataset_maker import DatasetCreate
from data_provider.data_loader import DataLoaderCreate
from models.clustered_transformer import Model
import gcsfs
from torch import optim

# refine how you supply arguments needed:
# make a dictionary:
input_settings = {}
input_settings['obs_path'] = "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/"
input_settings['date_picked'] = ['2019-01-01T00:00:00.000000000', '2019-01-10T23:00:00.000000000'] # 240 total instances.
input_settings['lat_range'] = [30., 20.] # only focus on NA region # 41
input_settings['long_range'] = [260, 275] # ideal for East Coast # 61
input_settings['variables_list'] = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
input_settings['flag'] = 'train'
input_settings['seq_len'] = 24 # t-instances fed to model  
input_settings['pred_len'] = 1 # t-instances output by model 
input_settings['train_batch_size'] = 3
input_settings['num_epochs'] = 1
input_settings['learning_rate'] = 1e-3
# eg input = 24 (1 day of data), output = 1 (1 hour in the future)
data_set, data_loader = DataLoaderCreate(input_settings)

model_params = {'seq_len': input_settings['seq_len'],
                'pred_len': input_settings['pred_len'],
                'num_vars': len(input_settings['variables_list'])}

model = Model(
    model_params
)

# loss criterion
criterion = nn.MSELoss()
# optimizer
model_optim = optim.Adam(model.parameters(), lr=input_settings['learning_rate'])

# for epoch in range(input_settings['num_epochs']):
train_loss = []
model.train()
for i, (inp_list_arr, out_list_arr) in enumerate(data_loader):
    model_optim.zero_grad()
    inp_list_arr = inp_list_arr.float() # (b, v, s_len, lat, lon)
    out_list_arr = out_list_arr.float() # (b, v, p_len, lat, lon)
    train_steps = len(data_loader)
    batch_size, num_var, s_len, num_lats, num_lons = inp_list_arr.shape
    
    # inp_list_arr = inp_list_arr.reshape((batch_size, num_var * s_len, num_lats * num_lons))
    # inp_list_arr = inp_list_arr.permute((0, 2, 1))
    
    pred_output = model(inp_list_arr)
    
    # output: (b, lat*lon, v*p_len)
    pred_output = pred_output.reshape(batch_size, num_lats, num_lons, num_var, input_settings['pred_len'])
    pred_output = pred_output.permute((0, 3, 4, 1, 2)) #(b, v, p_len, lat, lon)
    
    pred_output = pred.detach()
    out_list_arr  = out_list_arr.detach()

    loss = criterion(pred_output, out_list_arr)
    train_loss.append(loss.item())

    if (i + 1) % 10 == 0:
        print("\titers: {0}, epoch: 0 | loss: {1:.7f}".format(i+1, loss.item()))
    
    loss.backward()
    model_optim.step()
    train_loss = np.average(train_loss)

    print("Steps: {0} | Train Loss: {1:.7f}".format(train_steps, train_loss))

    # best_model_path = path + '/' + 'checkpoint.pth'
    # model.load_state_dict(torch.load(best_model_path))

    # input_list: (b, v, s_len, lat, long)
    # output_list: (b, v, p_len, lat, long)
    # print(inp_list.shape,":", out_list.shape)
        
    # for each input (b, v, s_len, lat, lon) -> label-array (b, lat*lon)

    # then based on label-array, divide lat*lon points into clusters for each batch
    # b {(ncl)_i, v * s_len} (normalized from above step or here separately)

    # model:
    # loop through each batch, each set of (ncl)_i within it,
    # get output:
    # (ncl)_i , v*s_len -> model -> (ncl)_i, v*p_len
    # Question: how to design model that takes varying number of spatial points??

    # for each batch: collect the outputs from the model back to get
    # (lat*lon, v* p_len)
    # end with: (b, lat*lon, v_*p_len) -> (b, v, p_len, lat, lon)
    # loss = difference between this output and out_list.    
    break


