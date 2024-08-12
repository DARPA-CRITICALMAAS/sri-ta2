
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import math

import util.perm_inv as perm_inv
import util.einsum as einsum

pool=perm_inv.einpool_multihead('X_ZabH',order=5,equivariant=True)

class einnet(nn.Module):
    def __init__(self,ninput=100,nh=32,noutput=100,nstacks=1,nh0=8,nlayers=1):
        super().__init__()
        nheads=pool.nheads()
        nterms=len(pool.eqs)
        
        self.t=nn.ModuleList()
        self.t.append(perm_inv.MLP(ninput,nh,nh0*nheads,nlayers))
        for i in range(nstacks-1):
            self.t.append(perm_inv.MLP(nh0*nterms,nh,nh0*nheads,nlayers))
        
        self.t.append(perm_inv.MLP(nh0*nterms,nh,noutput,nlayers))
        
        self.nheads=nheads
        self.nterms=nterms
    
    def forward(self,X):
        nstacks=len(self.t)-1
        h=X
        for i in range(nstacks):
            skip=h
            h=self.t[i](h)
            h=h.view(*h.shape[:-1],-1,self.nheads)
            h=torch.sin(h)
            h=pool(h)
            h=h.view(*h.shape[:-2],-1)
            if i>0:
                h+=skip
        
        h=self.t[nstacks](h)
        return h


class new(nn.Module):
    def __init__(self,nh=32,nh0=8,nstacks=2,nlayers=2):
        super().__init__()
        self.encoder=einnet(4,nh,nh,nstacks,nh0,nlayers)
        self.t=perm_inv.MLP(nh,nh,1,2)
        self.t0=perm_inv.MLP(nh,nh,1,2)
        #self.w=nn.Parameter(torch.Tensor(1).fill_(0))
        #self.b=nn.Parameter(torch.Tensor(1).fill_(0))
    
    def forward(self,X,low_mem=False):
        #normalization
        B,K,N=X.shape
        assert B==1
        X=X[0]
        X=F.log_softmax(X,dim=-1)
        X0=X[0:1]
        X0=X0.repeat(len(X),1)
        X=torch.stack((X0,X),dim=-1) #K,N+1,2
        X0=X[:,-1:,:].repeat(1,N-1,1)
        X=torch.cat((X[:,:-1,:],X0),dim=-1)
        
        h=X.unsqueeze(dim=0) # 1 K N 4
        h=self.encoder(h)
        
        h=h.mean(dim=1) #1 N h
        h0=h.mean(dim=1)# 1,h
        h=self.t(h).squeeze(-1)
        h0=self.t0(h0)
        
        h=torch.cat([h,h0],dim=-1)
        return h

