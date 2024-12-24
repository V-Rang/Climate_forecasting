import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import time

class ClusteredAttention(nn.Module):
    def __init__(self, scale = None, attention_dropout = 0.1, output_attention = False ):
        super(ClusteredAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.scale = scale

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, label_arr: np.array) -> torch.tensor:        
        
        '''
        query, key, value: (b, l, d_model) 
        label_arr -> (b, l)
        scores -> (b, l, l)
        '''
        b, l, s  = query.shape
        scale = self.scale or 1./ sqrt(s)

        scores = torch.einsum("ble,bse->bls", query, key)
        
        # applying mask.
        mask = label_arr[:, :, None] != label_arr[:, None, :] 
        scores[mask] = float('-inf')

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

        # b, l, v, s  = query.shape
        # scale = self.scale or 1./ sqrt(s)

        # scores = torch.zeros((b, l, v, l))

        # label_mask = label_arr.unsqueeze(2) == label_arr.unsqueeze(1)  # Shape: (b, l, l)

        # sum_tot_vec = key.sum(dim = 2)  # Summing over the `k1` dimension, Shape: (b, l, s)

        # scores = -torch.inf * torch.ones((b, l, v, l), device=query.device)
        
        # for k in range(v): # 3 variables only for now.
        #     q1 = query[:,:,k,:].unsqueeze(2)
        #     q2 = sum_tot_vec.unsqueeze(1).transpose(-1, -2).squeeze(2)
        #     q3 = (q1 @ q2).squeeze(2)
        #     scores[:,:,k, :] = torch.where(label_mask, q3, scores[:, :, k, :])

        # A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # # A: (b, l, v, l) l_1 -> queries, l_3 -> keys
        # # values: (b, l, v, s) 

        # # values -> (b,l, 1, v, s)
        # # A -> (b,l,v,l) -> (b, l, l, v) -> (b,l, l, v, 1)
        # # V -> (b, l, l, v, s) l_1 -> key, l_2 -> query

        # V = ( value.unsqueeze(2) * A.permute(0,3,2,1).transpose(-1, -2).unsqueeze(-1)).permute(0,2,1,3,4) # (b, l, l, v, s)
        # V = V.sum(dim = 2) # (b, l, l, v, s) -> (b, l, v, s)
        # # (b,l,v,s) can be added to the query (b, l, v, s).

        # if self.output_attention:
        #     return (V.contiguous(), A)
        # else:
        #     return (V.contiguous(), None)