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
import pywt as pw
import h5py

class DatasetCreate(Dataset):
    def __init__(self, args, flag) -> np.array:
        
        # assert(flag == 'train' or flag == 'test' or flag == 'val')
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len       
        # self.data = data
        # self.len_data = len(data)

        with h5py.File(args.data_file, "r") as f:
            first_key = next(iter(f.keys()))
            glb_data = f[first_key] 
            glb_data = glb_data[next(iter(glb_data.keys()))][:] #(T, Ny, Nx, v)
            
        # glb_data = np.load(settings['obs_path']) #(time instances, num points_y, num points_x)
        # settings['height'], settings['width'] = glb_data.shape[1], glb_data.shape[2]

        len_glb_data = glb_data.shape[0]
        n_train = int(0.8 * len_glb_data)
        n_val = int(0.1 * len_glb_data)
        n_test = len_glb_data - n_train - n_val
        
        left_limit = [0, n_train, len_glb_data - n_test]
        right_limit = [n_train, n_train + n_val, len_glb_data]

        set_type = {'train' : 0, 'val': 1, 'test': 2}

        dl, dr = left_limit[set_type[flag]], right_limit[set_type[flag]]
        
        self.loc_data = glb_data[dl: dr] #(t, Ny, Nx, v)        
        self.len_data = self.loc_data.shape[0]
        self.time_enc_data = np.zeros((1, args.num_vars, len_glb_data)) #(1, v, t)

        # print(len(self.time_enc_data), self.loc_data.shape[0])
        
        if(args.time_enc):
            total_time_vals = torch.linspace(args.time_lower_limit, args.time_upper_limit,len_glb_data)
            self.time_enc_data = total_time_vals.unsqueeze(0).expand((args.num_vars,-1)).unsqueeze(0) #(1, v, t)

    def __len__(self):
        return self.len_data - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        inp_start_index = index
        inp_end_index = inp_start_index + self.seq_len
        
        out_start_index = inp_end_index
        out_end_index = out_start_index + self.pred_len

        input_data_list = self.loc_data[inp_start_index: inp_end_index] 
        output_data_list = self.loc_data[out_start_index: out_end_index]

        # pre-processing using wavelet transform:
        # data_2d_decomposed_coeffs = pw.wavedec2(input_data_list, 'db1', axes = (-1,-2), level = 1)
        # data_2d_coeff1, (data_2d_coeff2, data_2d_coeff3, data_2d_coeff4) = data_2d_decomposed_coeffs 
        # data_2d_frames = np.concatenate(((data_2d_coeff1, data_2d_coeff2, data_2d_coeff3, data_2d_coeff4)), axis=2) #(slen , 24, 96)
        # data_init_model = data_2d_frames.reshape((data_2d_frames.shape[0], -1)) #(slen, 24 * 96) = (slen, 2304) = (slen, 48 * 48)
        # return data_init_model, self.time_enc_data[inp_start_index: inp_end_index], output_data_list,  self.time_enc_data[out_start_index: out_end_index]

        return input_data_list, self.time_enc_data[:,:,inp_start_index: inp_end_index], output_data_list,  self.time_enc_data[:,:, out_start_index: out_end_index]