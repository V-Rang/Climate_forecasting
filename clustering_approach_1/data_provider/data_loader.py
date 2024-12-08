from torch.utils.data import Dataset, DataLoader
from data_provider.dataset_maker import DatasetCreate

def DataLoaderCreate(settings):
    if settings['flag'] == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation

    else: # train
        shuffle_flag = True
        drop_last = True
        batch_size = settings['train_batch_size']  # bsz for train and valid

    data_set = DatasetCreate(
        settings['obs_path'],
        settings['date_picked'],
        settings['variables_list'],
        settings['lat_range'],
        settings['long_range'],
        settings['seq_len'],
        settings['pred_len'],
        settings['flag'])
 
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)

    return data_set, data_loader