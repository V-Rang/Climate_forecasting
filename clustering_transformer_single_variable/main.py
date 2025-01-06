import numpy as np
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
# from data_provider.dataset_maker import DatasetCreate
from models.clustered_transformer import Model
import gcsfs
from torch import optim
from experiments.exp_template import Exp
import argparse

parser = argparse.ArgumentParser(description='clustered_attention_transformer')
parser.add_argument('--model_type', type = str, default = 'clustered_transformer')
parser.add_argument('--obs_path', type = str, default = 'u_vals.npy')
parser.add_argument('--seq_len', type = int, default = 40)
parser.add_argument('--pred_len', type = int, default = 10)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--num_epochs', type = int, default = 100)
parser.add_argument('--learning_rate', type = float, default = 1e-3)
parser.add_argument('--e_layers', type = int, default = 4)
parser.add_argument('--d_model', type = int, default = 512)
parser.add_argument('--checkpoints', type = str, default = './checkpoints/', help='location of model checkpoints')
parser.add_argument('--patience', type = int, default = 3)
parser.add_argument('--normalization_flag', type = str, default = 'batch')

parser.add_argument('--attention_masking', type = int, default = 0) # to restrict attention to cluster specific points only.
parser.add_argument('--time_enc', type = int, default = 0)
parser.add_argument('--wavelet_transformation', type = int, default = 0)


args = parser.parse_args()


exp = Exp(args)

exp_name = '{}_{}-sl{}-dm{}-pl{}-bs{}-ne{}-lr{}-el{}-attn{}-timenc{}-wt{}'.format(
    args.model_type,
    args.obs_path[:-3],
    args.seq_len,
    args.d_model,
    args.pred_len,
    args.batch_size,
    args.num_epochs,
    args.learning_rate,
    args.e_layers,     
    args.attention_masking,     
    args.time_enc,     
    args.wavelet_transformation,     
)

exp.train(exp_name)
exp.test(exp_name)

# from data_provider.data_loader import DataLoaderCreate
# from layers.wavelet_transform import wave_dec, wave_rec
# data_set, data_loader = DataLoaderCreate(input_settings, flag = 'train')
# for i, (t1, t2, t3, t4) in enumerate(data_loader):
#     print(t1.shape,":",t2.shape ,":", t3.shape, ":", t4.shape)
#     break

