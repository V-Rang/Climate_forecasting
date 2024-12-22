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
    def __init__(self, 
    data: np.array,
    seq_len: int,
    pred_len: int) -> np.array:
        
        # assert(flag == 'train' or flag == 'test' or flag == 'val')
        self.seq_len = seq_len
        self.pred_len = pred_len       
        self.data = data
        self.len_data = len(data)

    def __len__(self):
        return self.len_data - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        inp_start_index = index
        inp_end_index = inp_start_index + self.seq_len
        
        out_start_index = inp_end_index
        out_end_index = out_start_index + self.pred_len
        
        input_data_list = self.data[inp_start_index: inp_end_index]
        output_data_list = self.data[out_start_index: out_end_index]

        return torch.tensor(input_data_list), torch.tensor(output_data_list)

    
        # input_data_list, output_data_list = [], []
        
        # for i in range(self.len_data):
        #     input_data_list.append(self.data[i][inp_start_index:inp_end_index])
        #     output_data_list.append(self.data[i][out_start_index: out_end_index])
        
        # return torch.tensor(np.array(input_data_list)), torch.tensor(np.array(output_data_list))
