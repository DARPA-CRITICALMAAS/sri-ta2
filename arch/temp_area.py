
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim


import util.perm_inv as perm_inv
import util.einsum as einsum

class new(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=perm_inv.MLP(32,32,32,1)
    
    def forward(self,X):
        return F.log_softmax(X[:,0],dim=-1)

