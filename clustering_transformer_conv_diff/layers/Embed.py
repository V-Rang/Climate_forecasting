import torch
import torch.nn as nn
import math
import numpy as np

class TseriesEmbed(nn.Module):
    def __init__(self, seq_len, d_model):
        super(TseriesEmbed, self).__init__()
        self.embed_layer = nn.Linear(seq_len, d_model, bias = False)
    
    def forward(self, x, x_enc):
        '''
        x -> b l v s
        output -> b l v d_model
        '''
        # print(x.shape, ":", x_enc[0].shape)
        # print(x_enc.dtype) #float64
        test_arr = torch.zeros(len(x_enc[0]), dtype = x_enc[0].dtype).to(x_enc.device)

        # if(x_enc[0] == torch.zeros(len(x_enc[0]) )): # no time encoding
        # if(torch.isclose(x_enc[0], test_arr , atol = 1e-5)): # no time encoding
        if(torch.allclose(x_enc[0], test_arr)): # no time encoding
            x = self.embed_layer(x) # (b,l, s)
        else:
            '''
            x -> (b,l,s)
            x_enc -> (b, s)
            So, x_enc -> (b,1, s), then concat to x (axis = 1) -> (b, l+1, s)
            '''
            x = self.embed_layer(torch.cat([x, x_enc.unsqueeze(1)], axis = 1)) #(b,l+1,s)        
        
        return x  
