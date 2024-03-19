import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

t0 = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]])
t1 = torch.randn(9) 
t2 = torch.ones(7,3)

print(t0[0].size())