import torch
import torch.nn as nn
import numpy as np
from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
# from einops import rearrange


class ClusteredAttention(nn.Module):
    def __init__(self, ):
        super(ClusteredAttention, self).__init__()
        