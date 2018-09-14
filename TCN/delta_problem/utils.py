import numpy as np
import torch
from torch.autograd import Variable

def data_generator(T, b_size):
    """
    Generate data for the delta task

    :param T: The total blank time length
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    x = torch.rand(b_size, T)
    y = torch.cat((torch.zeros(b_size, T - 1), (x[:, -1] - x[:, 0]).unsqueeze(1)), 1)
    
    x, y = Variable(x), Variable(y)
    return x, y
