import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.clustered_attention import ClusteredAttention
from utils.cluster_tools import ClusterDetermine

class Model(nn.Module):
    def __init__(self, model_settings):
        super(Model, self).__init__()
        self.seq_len = model_settings['seq_len']
        self.pred_len = model_settings['pred_len']
        self.num_vars = model_settings['num_vars']
        self.l1 = nn.Linear(self.num_vars * self.seq_len, 3)
        self.l2 = nn.Linear(3, self.num_vars * self.seq_len)
        # self.attention = ClusteredAttention(

        # )

    def forward(self, input_arr):
        # inputs:
        # input_arr: (b, lat*lon, v*s_len)
        # info about the (b, lat*lon points), which point belongs to which cluster region.

        # output = self.l2(self.l1(input_arr))
        # output = nn.Linear(self.num_vars * self.seq_len, self.num_vars * self.seq_len)(output)
        # return output[:,:,:self.num_vars * self.pred_len]

        # return input_arr[:,: :self.num_vars * self.p_len]
        #outputs:
        # output: (b, lat*lon, v*p_len)

        # input_arr: (b, v, s_len, lat, lon)
        
        bsize, num_vars, seq_len, num_lats, num_lons = input_arr.shape
        # input_arr = input_arr.reshape(bsize, num_vars * seq_len, num_lats * num_lons)
        # input_arr = input_arr.permute((0, 2, 1)) #(b, lat*lon, v*s_len)

        labels = ClusterDetermine(input_arr.reshape(bsize, num_vars * seq_len, num_lats * num_lons).permute((0, 2, 1))) 
        # (b, lat * lon)
        print(labels.shape)
        # for each sample (v, lat*lon, s_len) implement clustered attention using the (lat * lon) labels, 
        # zeroing out values for points not in the same cluster.

        # output_arr = self.attention(   )


        output = self.l2(self.l1(input_arr))
        return output[:,:,:self.num_vars * self.pred_len]

        
