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

import util.smartparse as smartparse
import util.session_manager as session_manager
import util.metrics as metrics

def default_params():
    params=smartparse.obj();
    params.root='predictions/scores_llama3-8b-ft'
    params.arch='arch.inv_record'
    params.base='arch.temp'
    params.split='index/splits/train_joint.csv'
    params.split_eval='index/splits/eval_joint.csv'
    params.load=''
    params.lr=1e-3
    params.batch=16
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);

class Dataset:
    def __init__(self,split,options,root):
        self.path=list(split['path'])
        self.label=list(split['deposit_type'])
        self.root=root
        self.gt=[options.index(x) for x in self.label]
        
        prior=[0 for i in range(len(options))]
        for x in self.gt:
            prior[x]+=1
        
        prior=torch.Tensor(prior)
        prior=prior/prior.sum()
        logprior=torch.log(prior+1e-20)
        self.prior=prior
        self.logprior=logprior
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,i):
        path=self.path[i]
        path=os.path.join(self.root,path.replace('.json','.gz'))
        try:
            scores=torch.load(gzip.open(path,'rb'),map_location='cpu').float().data
        except:
            print('error',path)
            scores=torch.load(gzip.open(path,'rb'),map_location='cpu').float().data
        
        return scores,self.gt[i],self.logprior

dataset_train=pandas.read_csv(params.split)
dataset_test=pandas.read_csv(params.split_eval)
cmmi=pandas.read_csv('../science/dataset/taxonomy/cmmi_options_full_gpt4_number.csv',encoding='latin')
options=list(cmmi['Deposit type'])

dataset_train=Dataset(dataset_train,options,params.root)
dataset_test=Dataset(dataset_test,options,params.root)

data_train=DataLoader(dataset_train,batch_size=1,shuffle=True,num_workers=32)
data_test=DataLoader(dataset_test,batch_size=1,shuffle=True,num_workers=32)

#Network
arch=importlib.import_module(params.arch)
temperature=importlib.import_module(params.base)
net=arch.new().cuda()
baseline=temperature.new().cuda()
if not params.load=='':
    checkpoint=torch.load(params.load)
    net.load_state_dict(checkpoint['net'],strict=False)

t0=time.time()
opt=optim.Adam(net.parameters(),lr=params.lr)
for epoch in range(1000000):
    tracker=session_manager.loss_tracker()
    if epoch>=0:
        _=net.eval()
        with torch.no_grad():
            scores=[]
            scores_base=[]
            labels=[]
            for i,(s,gt,logprior) in enumerate(data_test):
                s,gt,logprior=s.cuda(),gt.cuda(),logprior.cuda()
                pred=net(s)
                scores.append(pred)
                pred_base=baseline(s)
                scores_base.append(pred_base)
                labels.append(gt)
                print('%d/%d   '%(i,len(data_test)),end='\r')
                #if i>=3000:
                #    break
            
            scores=torch.cat(scores,dim=0)
            labels=torch.cat(labels,dim=0)
            scores=F.log_softmax(scores,dim=-1)
            
            scores_base=torch.cat(scores_base,dim=0)
            scores_base=F.log_softmax(scores_base,dim=-1)
            
            loss=F.cross_entropy(scores+logprior,labels)
            top1,top5,mrr=metrics.eval_cls(scores+logprior,labels)
            ap,p50=metrics.eval_ret(scores+logprior,labels)
            tracker.add(loss_test=loss,top1=top1,top5=top5,mrr=mrr,ap=ap,p50=p50)
            
            top1,top5,mrr=metrics.eval_cls(scores_base+logprior,labels)
            ap,p50=metrics.eval_ret(scores_base+logprior,labels)
            tracker.add(top1_b=top1,top5_b=top5,mrr_b=mrr,ap_b=ap,p50_b=p50)
        
        #torch.save({'scores_base':scores_base,'scores':scores,'labels':labels,'logprior':logprior},'scores.pt')
        session.log('Epoch %d eval, %s, time %.2f     '%(epoch,tracker.str(),time.time()-t0))
        torch.save({'net':net.state_dict(),'params':smartparse.obj2dict(params)},session.file('model','%d.pt'%epoch))
    
    tracker=session_manager.loss_tracker()
    opt.zero_grad()
    _=net.train()
    for i,(s,gt,logprior) in enumerate(data_train):
        s,gt,logprior=s.cuda(),gt.cuda(),logprior.cuda()
        pred=net(s,low_mem=True)
        #ce
        loss=F.cross_entropy(pred+logprior,gt)
        loss.backward()
        
        tracker.add(loss=loss)
        if (i+1)%params.batch==0:
            opt.step()
            opt.zero_grad()
            print('%d/%d, %s, time %.2f    '%(i,len(data_train),tracker.str(),time.time()-t0),end='\r')
    
    session.log('Epoch %d train, %s, time %.2f     '%(epoch,tracker.str(),time.time()-t0))
