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
    dataset_path: str, 
    time_duration: list, 
    variable_list: list, 
    latitude_range: list, 
    longitude_range: list,
    seq_len: int,
    pred_len: int) -> np.array:
        
        # assert(flag == 'train' or flag == 'test' or flag == 'val')
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        data = xr.open_zarr(dataset_path)
        
        # data for all vars. correps. to all t-instances (train + test).
        self.data_total = [] #each element repr. a xarr DataArray of data (train + test) corresp. to each variable to be modeled.
    
        for i in range(len(variable_list)):
            data_var = data[variable_list[i]]
            data_var = data_var.sel(time = slice(*time_duration))
            data_var = data_var.sel(latitude = slice(*latitude_range), longitude = slice(*longitude_range))
            self.data_total.append(data_var)
        
        self.data_total = torch.tensor(np.array(self.data_total)) #(v, t, la, lo)
        self.data_total = self.data_total.reshape((self.data_total.shape[0], self.data_total.shape[1], -1)) #(v, t, la * lo)
        self.data_total = self.data_total.permute(2,0,1) #(l,v,t)
        # type_map = {'train':0, 'test': 1, 'val' : 2}
        # self.set_type = type_map[flag]
        
        # self.__read_data__()
    
    # def __read_data__(self):
        # len_dataset = self.data_total[0].shape[0] #total no. of time instances
        # n_train = int(0.8 * len_dataset)
        # glb_start_indices = [0, n_train]
        # glb_end_indices = [n_train, len_dataset]
        # loc_index_range = [glb_start_indices[self.set_type], glb_end_indices[self.set_type]]
        
        # data for all vars. corresp. to flag (train OR test only.)
        # self.data_x = []
        
        # for i in range(len(self.data_total)): #i.e. corresp. to each variable in variables_list.
        #     # self.data_x.append(self.data_total[i][loc_index_range[0] : loc_index_range[1]])
        #     self.data_x.append(self.data_total[i][:])
        # return self.data_total

    def __len__(self):
        return self.data_total.shape[2] - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        inp_start_index = index
        inp_end_index = inp_start_index + self.seq_len

        out_start_index = inp_end_index
        out_end_index = out_start_index + self.pred_len
                
        # return (l,v,s) and (l,v,p).

        return self.data_total[:, :, inp_start_index: inp_end_index], self.data_total[:, :, out_start_index: out_end_index] 
            
        # inp, out = self.data_total[:, :, inp_start_index: inp_end_index], self.data_total[:, :, out_start_index: out_end_index] 
        # print(inp.shape,":", out.shape)
        # return inp, out

        # input_data_list, output_data_list = [], []
        # make it faster -- don't use loop.
        # for i in range(len(self.data_total)):  #i.e. corresp. to each variable in variables_list.
        #     input_data_list.append(self.data_total[i][inp_start_index:inp_end_index])
        #     output_data_list.append(self.data_total[i][out_start_index: out_end_index])
        
        # return torch.tensor(np.array(input_data_list)), torch.tensor(np.array(output_data_list)) # np.arrays of xarray.DataArrays






