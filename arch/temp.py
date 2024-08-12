
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
        B,N,K=X.shape
        T=0.003
        
        #First compute a per-doc average
        if K%1000==0:
            h=X.view(B,N,-1,1000)
        else:
            h=X.view(B,N,1,-1)
        
        h=h.mean(dim=-1)
        
        #Do a per-doc temperature softmax
        h=F.softmax(h/T,dim=-2)
        h=h.mean(dim=-1)
        h=torch.log(h+1e-20) #BxN
        return h

