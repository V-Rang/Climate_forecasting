from torch.utils.data import Dataset, DataLoader
from data_provider.dataset_maker import DatasetCreate
import numpy as np

def DataLoaderCreate(settings, flag):
    data = np.load(settings['obs_path']) #(time instances, num points)
    len_data = data.shape[0]
    n_train = int(0.8 * len_data)
    n_val = int(0.1 * len_data)
    n_test = len_data - n_train - n_val
    
    left_limit = [0, n_train, len_data - n_test]
    right_limit = [n_train, n_train + n_val, len_data]

    set_type = {'train' : 0, 'val': 1, 'test': 2}

    dl, dr = left_limit[set_type[flag]], right_limit[set_type[flag]]

    data_set = DatasetCreate( data[dl : dr], settings['seq_len'], settings['pred_len'])
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsize = 1 for test    
    
    else: # train and val.
        shuffle_flag = True
        drop_last = True
        batch_size = settings['batch_size']


    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)

    return data_set, data_loader
