import torch
import torch.nn as nn
import math

class TseriesEmbed(nn.Module):
    def __init__(self, seq_len, d_model):
        super(TseriesEmbed, self).__init__()
        self.embed_layer = nn.Linear(seq_len, d_model, bias = False)
    
    def forward(self, x):
        '''
        x -> b l v s
        output -> b l v d_model
        '''
        return self.embed_layer(x)

