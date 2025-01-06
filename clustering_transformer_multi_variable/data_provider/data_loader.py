from torch.utils.data import Dataset, DataLoader
from data_provider.dataset_maker import DatasetCreate

def DataLoaderCreate(args, flag):

    data_set = DatasetCreate(args, flag)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsize = 1 for test 
    else: # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsize for train and val

    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)

    return data_set, data_loader
