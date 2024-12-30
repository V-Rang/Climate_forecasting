import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import time

class ClusteredAttention(nn.Module):
    def __init__(self, scale = None, attention_dropout = 0.1, output_attention = False, time_enc = 0):
        super(ClusteredAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.scale = scale
        self.time_enc = time_enc

    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, label_arr: np.array) -> torch.tensor:        
        
        '''
        query, key, value: (b, l, v, d_model) or (b, l+4, v, d_model) (if time_enc.) 
        label_arr -> (b, l)
        scores -> (b, l, v, l) or (b, l+4, v, l+4) (time_enc = 1).
        '''

        b, l, v, s  = query.shape
        scale = self.scale or 1./ sqrt(s)

        # scores = torch.zeros((b, l, v, l))

        label_mask = label_arr.unsqueeze(2) == label_arr.unsqueeze(1)  # Shape: (b, l, l)
        # if time_enc: label_mask: (b, l, l) -> (b, l+4, l+4)
        if self.time_enc:
            time_enc_mask = torch.ones((b,4,label_arr.shape[1]), dtype=torch.bool, device = query.device) #(b, 4, l)
            label_mask = torch.cat((label_mask, time_enc_mask), dim = 1) # (b, l+4, l)
            time_enc_mask = torch.ones((b,label_arr.shape[1]+4, 4), dtype=torch.bool, device = query.device) #(b, l+4, 4)
            label_mask = torch.cat((label_mask, time_enc_mask), dim = 2) # (b, l + 4, l + 4)

        sum_tot_vec = key.sum(dim = 2)  # Summing over the `k1` dimension, Shape: (b, l, s)

        key = key.cpu()
        del key

        # ***************************************        
        # scores = torch.randn((b, l, v, l), device=query.device)
        # scores = torch.einsum('', query, sum_tot_vec)

        scores = -torch.inf * torch.ones((b, l, v, l), device=query.device)

        for k in range(v): # 3 variables only for now.
            # q1 = query[:,:,k,:].unsqueeze(2)
            # q2 = sum_tot_vec.unsqueeze(1).transpose(-1, -2)
            # q2 = sum_tot_vec.unsqueeze(1).transpose(-1, -2).squeeze(2)
            # q3 = (query[:,:,k,:].unsqueeze(2) @ sum_tot_vec.unsqueeze(1).transpose(-1, -2)).squeeze(2)
            # scores[:,:,k, :] = q3   

            # w/o attention- remove mask code above loop.
            # q3 = (query[:,:,k,:].unsqueeze(2) @ sum_tot_vec.unsqueeze(1).transpose(-1, -2)).squeeze(2) # (b,l,l) or (b,l +4, l+4)
            # scores[:,:,k, :] = q3

            # with attention:
            q3 = (query[:,:,k,:].unsqueeze(2) @ sum_tot_vec.unsqueeze(1).transpose(-1, -2)).squeeze(2) # (b,l,l) or (b,l +4, l+4)
            # wherever labels match (i.e. same cluster), q3, else -inf.
            scores[:,:,k, :] = torch.where(label_mask, q3, scores[:, :, k, :])
        # ***************************************        
        
        # to clear gpu memory:
        query = query.cpu()
        del query

        # label_mask = label_mask.cpu()
        # del label_mask

        sum_tot_vec = sum_tot_vec.cpu()
        del sum_tot_vec

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) #gpu

        scores = scores.cpu()
        del scores

        # print(A.get_device(), ":", query.get_device())
        # A: (b, l, v, l) l_1 -> queries, l_3 -> keys
        # values: (b, l, v, s) 

        # values -> (b,l, 1, v, s)
        # A -> (b,l,v,l) -> (b, l, l, v) -> (b,l, l, v, 1)
        # V -> (b, l, l, v, s) l_1 -> key, l_2 -> query

        # ***************************************        
        # V = ( value.unsqueeze(2) * A.permute(0,3,2,1).transpose(-1, -2).unsqueeze(-1)).permute(0,2,1,3,4) # (b, l, l, v, s)

        V = ( value.unsqueeze(2) * A.permute(0,3,2,1).transpose(-1, -2).unsqueeze(-1)).transpose(1,2) # (b, l, l, v, s)        
        V = V.sum(dim = 2) # (b, l, l, v, s) -> (b, l, v, s) 
        
        # (b,l,v,s) can be added to the query (b, l, v, s).
        # ***************************************        
        # V = torch.randn((b,l,v,s), device = query.device)
        value = value.cpu()
        del value

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

        # scores = torch.zeros((b, l, v, l))

        # num_runs = 1
        # total_time = 0.
        
        # for _ in range(num_runs):
        #     start_time = time.time()
        #     for b_ind in range(b): # for each batch, is it possible to remove this loop?? process all samples in the batch at once??
        #         for i in range(l): 
        #             for j in range(l):
        #                 if(label_arr[b_ind][i] == label_arr[b_ind][j]): # compute attention only if i, j in same cluster region, else set to -np.inf for all vars.
        #                     for k in range(v):
        #                         sum_tot_vec = torch.zeros((s))
        #                         for k1 in range(v):
        #                             sum_tot_vec += key[b_ind, j, k1 , :]
        #                         scores[b_ind, i, k, j] = query[b_ind, i, k, :] @ sum_tot_vec
        #                 else:
        #                     scores[b_ind, i, :, j] = -np.inf * torch.ones((v))
        #     end_time = time.time()
        #     total_time += end_time - start_time
        
        # print(f'Method 1: average time = {total_time/num_runs} seconds.')

        # total_time = 0.
        # for _ in range(num_runs):
        #     start_time = time.time()
        #     for b_ind in range(b):            
        #         for i in range(l): # loop through each key (v, s), calculate sum of all the 'v' 's' length sub-keys, and then multiply with each query (sub-queries) to compute corresp. score.
        #             sub_tot_vec = torch.zeros((s))
        #             for j in range(v):
        #                 sub_tot_vec += key[b_ind, i, j, :]
                    
        #             for k1 in range(l): # loop through each query.
        #                 if(label_arr[b_ind][k1] == label_arr[b_ind][i]):
        #                     for k2 in range(v):
        #                         scores[b_ind, k1, k2, i] = query[b_ind, k1, k2, :] @ sub_tot_vec
        #                 else:
        #                     scores[b_ind, k1, : ,i] = -np.inf * torch.ones(v)        
        #     end_time = time.time()
        #     total_time += end_time - start_time
        
        # print(f'Method 2: average time = {total_time/num_runs} seconds.')


        # return scores
