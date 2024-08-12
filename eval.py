import time
import os
import math
import sys
import pandas
import json
import importlib
import gzip

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager

def default_params():
    params=smartparse.obj();
    params.root='predictions/scores_qa_gpt_4o_mini'
    params.split='index/splits/eval_joint.csv'
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);



#metrics
def eval_cls(scores,gt):
    sgt=scores.gather(-1,gt.view(-1,1))
    r=scores.ge(sgt).long().sum(dim=-1)
    mrr=(1/r.float()).mean()
    top1=r.eq(1).float().mean()
    top5=r.le(5).float().mean()
    return float(top1),float(top5),float(mrr)

def eval_ret(scores,gt):
    #Compute P/R for top K=50 most frequent deposit types
    #For reliable estimates (>100 occurence out of 30000)
    K=50
    prior=[0 for i in range(max(gt.view(-1).tolist())+1)]
    for x in gt.view(-1).tolist():
        prior[x]+=1
    
    prior=torch.LongTensor(prior)
    _,r=prior.sort(dim=0,descending=True)
    
    #AP and P@R=50
    ap=[]
    p50=[]
    for i,ind in enumerate(r.view(-1).tolist()[:K]):
        s=scores[:,ind]
        _,rank=s.sort(dim=0,descending=True)
        _,rank=rank.sort(dim=0)
        rank_gt=rank[gt.eq(ind)]+1
        #print(rank_gt)
        rank_gt,_=rank_gt.sort(dim=0)
        ap_i=torch.arange(1,len(rank_gt)+1).to(rank_gt.device)/rank_gt
        ind_0=math.floor(len(ap_i)/2)
        ind_1=math.ceil(len(ap_i)/2)
        p50_i=(ap_i[ind_0]+ap_i[ind_1])/2
        ap_i=ap_i.mean()
        
        ap.append(float(ap_i))
        p50.append(float(p50_i))
    
    return sum(ap)/len(ap),sum(p50)/len(p50)


#Load CMMI types and GT
t0=time.time()
cmmi=pandas.read_csv('taxonomy/cmmi_options_full_gpt4_number.csv',encoding='latin')
cmmi=list(cmmi['Deposit type'])

#Data
index=pandas.read_csv(params.split)
labels=list(index['deposit_type'])
paths=list(index['path'])
gt=[cmmi.index(x) for x in labels]
labels=torch.LongTensor(gt)

#Compute deposit type prior in split
prior=[0 for i in range(len(cmmi))]
for x in gt:
    prior[x]+=1

prior=torch.Tensor(prior)
prior=prior/prior.sum()
logprior=torch.log(prior+1e-20)

#Load scores
class loader:
    def __init__(self,paths,root):
        self.paths=paths
        self.root=root
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,i):
        path=self.paths[i]
        path=os.path.join(self.root,path.replace('.json','.gz'))
        f=gzip.open(path,'rb')
        scores=torch.load(f,map_location='cpu').float().data
        f.close()
        return scores


data=DataLoader(loader(paths,params.root),batch_size=1,num_workers=32)

scores=[]
for i,s in enumerate(data):
    print('%d/%d    '%(i,len(index)),end='\r')
    scores.append(s.data.clone())

scores=torch.cat(scores,dim=0)

#Evaluate performance
tracker=session_manager.loss_tracker()
scores=F.log_softmax(scores,dim=-1)
scores_with_prior=scores[:,:len(cmmi)]+logprior.view(1,-1)

loss=F.cross_entropy(scores_with_prior,labels)
top1,top5,mrr=eval_cls(scores_with_prior,labels)
ap,p50=eval_ret(scores_with_prior,labels)
tracker.add(loss_test=loss,top1=top1,top5=top5,mrr=mrr,ap=ap,p50=p50)
print('Eval, %s, time %.2f'%(tracker.str(),time.time()-t0))
