import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention_mech, d_model, d_exp = None, dropout = 0.1, activation="relu"):
        super(EncoderLayer,self).__init__()
        self.attention =  attention_mech
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        d_exp = d_exp or 2 * d_model
        self.conv1 = nn.Conv2d(in_channels = d_model, out_channels = d_exp, kernel_size = 1)
        self.conv2 = nn.Conv2d(in_channels = d_exp, out_channels = d_model, kernel_size = 1)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, label_arr):
        '''
        x -> b,l,v, d_model or b, l+4, v, d_model
        label_arr -> b,l
        
        output:
        b,l,v, d_model or b,l+4, v, d_model
        '''
        
        new_x, attention = self.attention(
            x, x, x, label_arr
        )

        x = x + self.dropout(new_x)
        #gpu memory purposes:
        new_x = new_x.cpu()
        del new_x

        y = x = self.norm1(x) #(b, l, v, d_model)
        y = self.dropout(self.activation(self.conv1(y.permute(0,3,1,2))))
        y = self.dropout(self.conv2(y).permute(0,2,3,1))
        z = self.norm2(x + y)

        y = y.cpu()
        x = x.cpu()
        del x, y

        return z, attention


class Encoder(nn.Module):
    def __init__(self, attention_layers, norm_layer = None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.norm = norm_layer

    def forward(self, x, label_arr):
        attention_list = []
        for attention_layer in self.attention_layers:
            x, attn = attention_layer(x, label_arr)
            attention_list.append(attn)
    
        if self.norm is not None:
            x = self.norm(x)

        return x, attention_list



