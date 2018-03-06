"""
Utility functions for PyTorch.
"""

# Torch imports
import torch
from torch.autograd import Variable

# Torch-numpy conversion functions
np_to_torch_cpu = lambda x: Variable(torch.from_numpy(x))
np_to_torch_gpu = lambda x: Variable(torch.from_numpy(x)).cuda()
np_to_torch = None
torch_to_np = lambda x: x.cpu().data.numpy()

# Construction of zero-initialized tensor variable
torch_zeros_cpu = lambda *size: Variable(torch.FloatTensor(*size).zero_())
torch_zeros_gpu = lambda *size: Variable(torch.cuda.FloatTensor(*size).zero_())
torch_zeros = None

_cuda = False
def set_cuda(c):
    global _cuda, np_to_torch, torch_zeros
    _cuda = c
    np_to_torch = np_to_torch_gpu if c else np_to_torch_cpu
    torch_zeros = torch_zeros_gpu if c else torch_zeros_cpu