import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import time

class ClusteredAttention(nn.Module):
    def __init__(self, scale = None, attention_dropout = 0.1, output_attention = False, attention_masking = 0, time_enc = 0):
        super(ClusteredAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.scale = scale
        self.attention_masking = attention_masking
        self.time_enc = time_enc

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, label_arr: np.array) -> torch.tensor:        
        
        '''
        query, key, value: (b, l, d_model) 
        label_arr -> (b, l) # l-1 if time_enc is True.
        scores -> (b, l, l)
        '''
                    
        b, l, s  = query.shape
        scale = self.scale or 1./ sqrt(s)

        scores = torch.einsum("ble,bse->bls", query, key)
        
        if self.attention_masking: #if true, restrict attention to only cluster specific points
            mask = label_arr[:, :, None] != label_arr[:, None, :] 
            # if time enc is true, modify label_arr such that every 'query' has to pay attention
            # the corresp. 'key'.
            if self.time_enc:
                time_enc_mask = torch.zeros((b,1,label_arr.shape[1]), dtype=torch.bool, device = query.device) #(b, 1, l - 1)
                mask = torch.cat((mask, time_enc_mask), dim = 1) # (b, l, l-1)
                time_enc_mask = torch.zeros((b,label_arr.shape[1]+1, 1), dtype=torch.bool, device = query.device) #(b, 1, l - 1)
                mask = torch.cat((mask, time_enc_mask), dim = 2) # (b, l, l)            
            
            scores[mask] = float('-inf')

            mask = mask.cpu()
            del mask
            
            label_arr = label_arr.cpu()
            del label_arr

            time_enc_mask = time_enc_mask.cpu()
            del time_enc_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        V = torch.einsum("bls,bsd->bld", A, value)

        # test_attenion_scores = A.detach().cpu().numpy()
        # test_label_arr = label_arr.detach().cpu().numpy()
        # np.save('test_label_arr.npy', test_label_arr)
        # np.save('test_attention_scores.npy', test_attenion_scores)
        
        # V = ( value.unsqueeze(2) * A.permute(0,3,2,1).transpose(-1, -2).unsqueeze(-1)).permute(0,2,1,3,4) # (b, l, l, v, s)
        # V = V.sum(dim = 2) # (b, l, l, v, s) -> (b, l, v, s)
        # ( b,l,v,s) can be added to the query (b, l, v, s).

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)