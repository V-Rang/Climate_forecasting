# return dataset and dataloader using 
# inputs:
# 1. file path
# 2. time durations
# 3. variable names
# 4. latitude, longitude ranges
# 5. sequence length (input number of t-steps), prediciton length (output number of t-steps)

from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import torch

class DatasetCreate(Dataset):
    def __init__(self, settings, flag) -> np.array:
        
        # assert(flag == 'train' or flag == 'test' or flag == 'val')
        self.seq_len = settings['seq_len']
        self.pred_len = settings['pred_len']       
        # self.data = data
        # self.len_data = len(data)
    
        glb_data = np.load(settings['obs_path']) #(time instances, num points)
        len_glb_data = glb_data.shape[0]
        n_train = int(0.8 * len_glb_data)
        n_val = int(0.1 * len_glb_data)
        n_test = len_glb_data - n_train - n_val
        
        left_limit = [0, n_train, len_glb_data - n_test]
        right_limit = [n_train, n_train + n_val, len_glb_data]

        set_type = {'train' : 0, 'val': 1, 'test': 2}

        dl, dr = left_limit[set_type[flag]], right_limit[set_type[flag]]
        
        self.loc_data = glb_data[dl: dr]

        self.len_data = self.loc_data.shape[0]

        self.time_enc_data = np.zeros(dr - dl)

        # print(len(self.time_enc_data), self.loc_data.shape[0])
        
        if(settings['time_enc']):
            total_time_vals = np.linspace(0,1,len_glb_data)
            self.time_enc_data = total_time_vals[dl: dr]

        
    def __len__(self):
        return self.len_data - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        inp_start_index = index
        inp_end_index = inp_start_index + self.seq_len
        
        out_start_index = inp_end_index
        out_end_index = out_start_index + self.pred_len
        
        input_data_list = self.loc_data[inp_start_index: inp_end_index]
        output_data_list = self.loc_data[out_start_index: out_end_index]

        return torch.tensor(input_data_list), torch.tensor(self.time_enc_data[inp_start_index: inp_end_index]), torch.tensor(output_data_list),  torch.tensor(self.time_enc_data[out_start_index: out_end_index])