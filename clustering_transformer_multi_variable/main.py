import numpy as np
import torch
import torch.nn as nn
# make clusters, rollling  5 days each.

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
parser.add_argument('--data_file', type = str, default = '2d_diff_react_32_32_10k.h5')
parser.add_argument('--num_vars', type = int, default = 2)
parser.add_argument('--d_model', type = int, default = 4)
parser.add_argument('--seq_len', type = int, default = 40)
parser.add_argument('--pred_len', type = int, default = 10)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--num_epochs', type = int, default = 100)
parser.add_argument('--learning_rate', type = float, default = 1e-3)
parser.add_argument('--e_layers', type = int, default = 4)
parser.add_argument('--checkpoints', type = str, default = './checkpoints/', help='location of model checkpoints')
parser.add_argument('--patience', type = int, default = 3)
parser.add_argument('--normalization_flag', type = str, default = 'batch')
parser.add_argument('--attention_masking', type = int, default = 0) # to restrict attention to cluster specific points only.
parser.add_argument('--time_enc', type = int, default = 0)
parser.add_argument('--wavelet_transformation', type = int, default = 0)

args = parser.parse_args()
exp = Exp(args)

# the model will take in (b, l, v, s) shaped input.
# three choices:
# 1. normalize all 'b' samples in the batch.
# 2. normalize samples individually (b * s).
# 3. No normalization.

# input_settings['use_norm_batch'] = False
# input_settings['use_norm_sample'] = True

# eg input = 24 (1 day of data), output = 1 (1 hour in the future)
# data_set, data_loader = DataLoaderCreate(input_settings)
# print(len(data_set))

# import torch
# print(torch.cuda.is_available())  # True
# print(torch.cuda.device_count())  # 2
# print(torch.cuda.current_device())  # 0
# print(torch.cuda.get_device_name(0))  # NVIDIA A100 80GB PCIe

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# inputs, labels = inputs.to(device), labels.to(device)

exp_name = '{}_{}-sl{}-dm{}-pl{}-bs{}-ne{}-lr{}-el{}-attn{}-timenc{}-wt{}'.format(
    args.model_type,
    args.data_file[:-3],
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


# from data_provider.data_loader import DataLoaderCreate
# # from layers.wavelet_transform import wave_dec, wave_rec
# data_set, data_loader = DataLoaderCreate(input_settings, flag = 'train')
# for i, (t1, t2, t3, t4) in enumerate(data_loader):
#     print(t1.shape,":",t2.shape ,":", t3.shape, ":", t4.shape)
#     break

exp.train(exp_name)
exp.test(exp_name)