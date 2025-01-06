import torch
import numpy as np
import pywt as pw
import torch.nn.functional as F

def wave_dec(input_arr : torch.tensor) -> np.array:
    '''
    input_arr -> (b,v,s,la,lo)
    output -> (b,l,v,s)
    '''
    
    data_2d_decomposed_coeffs = pw.wavedec2(input_arr, 'db1', axes = (-2,-1), level = 1) # 4 x (b,v,s, la/2,lo/2)
    data_2d_coeff1, (data_2d_coeff2, data_2d_coeff3, data_2d_coeff4) = data_2d_decomposed_coeffs
    data_2d_frames = np.concatenate(((data_2d_coeff1, data_2d_coeff2, data_2d_coeff3, data_2d_coeff4)), axis=-1) #(b,v,s, la/2,2*lo)
    data_dec = data_2d_frames.reshape(( data_2d_frames.shape[0], data_2d_frames.shape[1], data_2d_frames.shape[2], int(data_2d_frames.shape[3]*2), int(data_2d_frames.shape[4 ]/2))) #(b,v,s, la, lo)
    data_dec = data_dec.reshape(data_dec.shape[0], data_dec.shape[1], data_dec.shape[2], data_dec.shape[3] * data_dec.shape[4]) #(b,v,s,l)
    data_dec = np.array(torch.tensor(data_dec).permute(0,3,1,2)) # (b,v,s,l) -> (b,l,v,s)
    return data_dec

def wave_rec(input_arr: np.array, height: int, width: int) -> np.array:
    
    '''
    input_arr -> (b,l,v,p)
    output_arr -> (b,l,v,p)
    '''

    input_arr = np.array(torch.tensor(input_arr).permute(0,2,3,1)) # (b,l,v,p) -> (b,v,p,l)
    output = input_arr.reshape((input_arr.shape[0], input_arr.shape[1], input_arr.shape[2],  height, width)) #(b,v,p,la,lo)
    output = output.reshape((output.shape[0], output.shape[1],output.shape[2], int(output.shape[3]/2), int(output.shape[4]*2) )) #(b,v,p,la/2, 2*lo)
    coeffs1, coeffs2, coeffs3, coeffs4 = np.split(output, 4, axis = -1) #each (b,v,p,la/2,lo/2)
    output = pw.waverec2([coeffs1,(coeffs2, coeffs3, coeffs4)],'db1') #(b, v, p, la, lo)
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2],  int(height * width) ) # (b,v,p,l)
    output = np.array(torch.tensor(output).permute(0,3,1,2))  #(b,l,v,p)

    return output