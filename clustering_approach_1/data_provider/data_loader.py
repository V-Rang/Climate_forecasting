from torch.utils.data import Dataset, DataLoader
from data_provider.dataset_maker import DatasetCreate

def DataLoaderCreate(settings, flag):
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsize = 1 for test

        data_set = DatasetCreate(
            settings['obs_path'],
            settings['testing_period'],
            settings['variables_list'],
            settings['lat_range'],
            settings['long_range'],
            settings['seq_len'],
            settings['pred_len'])
 
    elif flag == 'val':
        shuffle_flag = True
        drop_last = True
        batch_size = settings['batch_size']  # bsize for train and val

        data_set = DatasetCreate(
            settings['obs_path'],
            settings['validation_period'],
            settings['variables_list'],
            settings['lat_range'],
            settings['long_range'],
            settings['seq_len'],
            settings['pred_len'])
 
    else: # train
        shuffle_flag = True
        drop_last = True
        batch_size = settings['batch_size']  # bsize for train and val

        data_set = DatasetCreate(
            settings['obs_path'],
            settings['training_period'],
            settings['variables_list'],
            settings['lat_range'],
            settings['long_range'],
            settings['seq_len'],
            settings['pred_len'])
    
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)

    return data_set, data_loader
