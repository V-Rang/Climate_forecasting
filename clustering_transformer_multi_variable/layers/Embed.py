import torch
import torch.nn as nn
import math

class TseriesEmbed(nn.Module):
    def __init__(self, seq_len, d_model):
        super(TseriesEmbed, self).__init__()
        self.embed_layer = nn.Linear(seq_len, d_model, bias = False)
    
    def forward(self, x, x_enc):
        
        '''
        x -> b l v s
        x_enc -> b 4 v s
        output -> b l v d_model
        '''

        # test_arr = torch.zeros(len(x_enc[0]), dtype = x_enc[0].dtype).to(x_enc.device)
        test_arr = torch.zeros_like(x_enc[0]).to(x_enc.device)        
        if (torch.allclose(x_enc[0], test_arr)): # no time encoding
            x = self.embed_layer(x) # (b,l, v, d_model)
        else:
            '''
            x -> (b, l, v, s)
            x_enc -> (b, 4, v, s)
            concatenate: (b, l+4, v, s)
            '''
            x = self.embed_layer(torch.cat([x, x_enc], axis = 1)) #(b, l+4, v, d_model)        
        
        test_arr = test_arr.cpu()
        del test_arr

        return x

