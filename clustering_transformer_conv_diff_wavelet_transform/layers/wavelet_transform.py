import pywt as pw
import torch
import numpy as np

def wave_dec(input_arr : np.array) -> np.array:
    '''
    input_arr -> (b,s,h,w)
    output -> (b,s,h*w)
    '''

    data_2d_decomposed_coeffs = pw.wavedec2(input_arr, 'db1', axes = (-1,-2), level = 1)
    data_2d_coeff1, (data_2d_coeff2, data_2d_coeff3, data_2d_coeff4) = data_2d_decomposed_coeffs
    # print(data_2d_coeff1.shape,":", data_2d_coeff2.shape, ":", data_2d_coeff3.shape, ":", data_2d_coeff4.shape)
    data_2d_frames = np.concatenate(((data_2d_coeff1, data_2d_coeff2, data_2d_coeff3, data_2d_coeff4)), axis=-1) #(slen , 24, 96)
    # print(data_2d_frames.shape)
    data_dec = data_2d_frames.reshape((data_2d_frames.shape[0],data_2d_frames.shape[1], -1)) #(slen, 24 * 96) = (slen, 2304) = (slen, 48 * 48)
    return data_dec

def wave_rec(input_arr: np.array, height: int, width: int) -> np.array:
    '''
    input_arr -> (b,p,l)
    output_arr -> (b,p,h*w)
    '''

    output = input_arr.reshape((input_arr.shape[0], input_arr.shape[1], height, width)) #(b,p,h,w)
    output = output.reshape((output.shape[0], output.shape[1], int(height/2), int(width*2) )) #(b,p,h/2, 2*w)
    coeffs1, coeffs2, coeffs3, coeffs4 = np.split(output, 4, axis = -1) #each (b,p,h/2,w/2)
    output = pw.waverec2([coeffs1,(coeffs2, coeffs3, coeffs4)],'db1') #(b, p, h, w)
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[2])

    return output
