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

def default_params():
    params=smartparse.obj();
    params.arch='arch.inv_record'
    params.load='model_checkpoints/model_score.pt'
    params.split='index/splits/eval.csv'
    params.out='predictions/scores_agg_llama3-8b-ft'
    params.root='predictions/scores_llama3-8b-ft'
    params.override=False
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
#session=session_manager.create_session(params);


class Dataset:
    def __init__(self,split,root):
        self.path=list(split['path'])
        self.root=root
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,i):
        path=self.path[i]
        path=os.path.join(self.root,path.replace('.json','.gz'))
        scores=torch.load(gzip.open(path,'rb'),map_location='cpu').float().data
        return scores,self.path[i]

dataset=pandas.read_csv(params.split,low_memory=False)
dataset={k:list(dataset[k]) for k in dataset.keys()}
dataset=Dataset(dataset,params.root)
data=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=32)


#Network
arch=importlib.import_module(params.arch)
net=arch.new().cuda()
if not params.load=='':
    checkpoint=torch.load(params.load)
    net.load_state_dict(checkpoint['net'],strict=True)

#Run network
t0=time.time()
_=net.eval()
with torch.no_grad():
    for i,(s,path) in enumerate(data):
        s=s.cuda()
        if not isinstance(path,str):
            path=path[0]
        
        path_out=os.path.join(params.out,path.replace('.json','.gz'))
        if params.override or not os.path.exists(path_out):
            pred=net(s).data.cpu().view(-1)
            os.makedirs(os.path.dirname(path_out),exist_ok=True)
            torch.save(pred,gzip.open(path_out,'wb'))
        
        print('%d/%d   '%(i,len(data)),end='\r')

