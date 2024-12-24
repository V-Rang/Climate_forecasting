import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.clustered_attention import ClusteredAttention
from utils.cluster_tools import ClusterDetermine
from layers.Embed import TseriesEmbed
from layers.Enc_Dec import EncoderLayer
from layers.Enc_Dec import Encoder

class Model(nn.Module):
    def __init__(self, model_settings):
        super(Model, self).__init__()
        
        self.seq_len = model_settings['seq_len']
        self.pred_len = model_settings['pred_len']
        self.d_model = model_settings['d_model']
        self.e_layers = model_settings['e_layers']
        self.norm_flag = model_settings['norm_flag']
        self.device = model_settings['device']
        self.attention = ClusteredAttention(output_attention=True)
        
        self.encod_embedding = TseriesEmbed(self.seq_len, self.d_model)

        self.encoder = Encoder(
            [EncoderLayer(self.attention, self.d_model) for l in range(self.e_layers)],
            norm_layer = nn.LayerNorm(self.d_model)
        )

        self.decoder = nn.Linear(self.d_model, self.pred_len, bias=True)

    #(b, v, s, lat, lon) -> model -> (b, v, p, lat, lon)
    def forward(self, input_arr):

        assert(self.norm_flag in ('batch', 'sample', 'None')  )
        
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
        
        bsize, seq_len, num_spt_points = input_arr.shape
        
        # input_arr = input_arr.reshape(bsize, num_vars * seq_len, num_lats * num_lons)
        # input_arr = input_arr.permute((0, 2, 1)) #(b, lat*lon, v*s_len)
        
        #in: b, la * lo, v, s
        # input_arr = input_arr.reshape(bsize, num_vars, seq_len, num_lats * num_lons).permute((0, 3, 1, 2))
        input_arr = input_arr.permute((0, 2, 1)) # (b, l, s)

        # labels = ClusterDetermine(input_arr.reshape(bsize, num_vars * seq_len, num_lats * num_lons).permute((0, 2, 1))) 
        labels = ClusterDetermine(input_arr) #(b, l)
        labels = labels.to(self.device) # move to gpu.

        if self.norm_flag  == 'batch':
            means = input_arr.mean((0,2), keepdim = True).detach() # (1,l,1)
            input_arr -=  means 
            stdev = torch.sqrt(torch.var(input_arr, dim=(0,2), keepdim=True, correction=False) + 1e-5)
            input_arr /= stdev 

        elif self.norm_flag == 'sample':
            means = input_arr.mean((2), keepdim = True).detach() # (b,l,1)
            input_arr -=  means
            stdev = torch.sqrt(torch.var(input_arr, dim=(2), keepdim=True, correction=False) + 1e-5)
            input_arr /= stdev 
    
        # (b, lat *lon, v * s)
        # input_arr = input_arr.reshape((input_arr.shape[0], input_arr.shape[1], -1))
        # print(labels.shape) # (b, lat * lon)

        # for each sample (lat*lon, v,  s_len) implement clustered attention using the (lat * lon) labels, 
        # zeroing out values for points not in the same cluster.
        #testing self.attention
        
        encode_out = self.encod_embedding(input_arr) #(b,l, s) -> (b,l, d_model)

        # is it possible to process entire batch at once instead of one sample at a time as below - yes, done.
        # del_x, scores = self.attention(input_arr, input_arr, input_arr, labels)                
        encode_out, attention_vals = self.encoder(encode_out, labels)

        # (b, l, v, d_model), n_elayers, (b, l, v, l)
        # print(encode_out.shape, ":", len(attention_vals), ":", attention_vals[0].shape)

        dec_out = self.decoder(encode_out) # (b, l, v, p)

        # normalization??
        if self.norm_flag  == 'batch' or self.norm_flag == 'sample':
            dec_out *= stdev            
            dec_out += means
        
        dec_out = dec_out.permute(0,2, 1) # (b, p, l)

        
        # dec_out = dec_out.reshape((list(dec_out.shape[:-1]) + [num_lats, num_lons]   )) # (b, v, p, la, lo)
        return dec_out # (b, p, l)

        # dummy output to get code to run for now.
        # output = self.l2(self.l1(input_arr))
        # output = output[:,:,:,:self.pred_len] # b, l, v, p
        # return output.permute((0, 2, 3, 1)).reshape((bsize, num_vars, self.pred_len, num_lats, num_lons )) # b, v, p, la, lo