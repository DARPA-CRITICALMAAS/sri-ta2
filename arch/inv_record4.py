
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import math

import util.perm_inv as perm_inv
import util.einsum as einsum

pool=perm_inv.einpool_multihead('X_ZabH',order=3,equivariant=True)

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
        self.encoder=einnet(1,nh,nh,nstacks,nh0,nlayers)
        self.t=perm_inv.MLP(nh,nh,1,2)
        self.t0=perm_inv.MLP(nh,nh,1,2)
        #self.w=nn.Parameter(torch.Tensor(1).fill_(0))
        #self.b=nn.Parameter(torch.Tensor(1).fill_(0))
    
    def forward(self,X,low_mem=False):
        #normalization
        B,N,K=X.shape
        assert B==1
        X=F.pad(X,(0,math.ceil(K/200)*200-K))
        
        if low_mem:
            n=30
        else:
            n=60
        
        if X.shape[-1]%200==0:
            X=X.view(N,-1,200)
        else:
            X=X.view(N,1,-1)
        
        if X.shape[1]>=n:
            ind=torch.randperm(X.shape[1])[:n].to(X.device)
            X=X[:,ind,:].contiguous()
        
        X=X.permute(1,0,2).contiguous() #para N 200
        
        h=X.unsqueeze(dim=-1)
        h=self.encoder(h)
        
        h0=h.mean(dim=[0,1,2]).unsqueeze(0)
        h=h.mean(dim=[0,2])
        h=self.t(h).squeeze(-1).unsqueeze(0)
        h0=self.t0(h0)
        h=torch.cat([h,h0],dim=-1)
        #h=torch.tanh(torch.exp(self.w*10)*h+self.b)*8
        #print(X.shape,h.shape)
        return h

