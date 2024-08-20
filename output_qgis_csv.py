
import time
import gzip
import os
import math
import sys
import copy
import pandas
import json

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager

import util.smartparse as smartparse
import util.session_manager as session_manager

def default_params():
    params=smartparse.obj();
    
    params.score_threshold=0.2
    params.split='index/demo_sites.csv'
    params.scores='predictions/scores_qa_gpt-4o-mini'
    params.out='predictions/SRI_deptype_qgis_gpt-4o-mini_demo.csv'
    
    params.override=False
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
#session=session_manager.create_session(params);


def group_scores(s,v):
    vidx=sorted(list(set(v)))
    s2=[[] for i in vidx]
    for i,si in enumerate(s):
        ind=vidx.index(v[i])
        s2[ind].append(si)
    
    s2=[max(x) for x in s2]
    s2=[x/(sum(s2)+1e-12) for x in s2]
    return vidx,s2

labels=pandas.read_csv('taxonomy/cmmi_options_full_gpt4_number.csv', encoding ='latin1')
options=list(labels['Deposit type'])
group=list(labels['Deposit group'])
environment=list(labels['Deposit environment'])
#options=[clean_str(x) for x in options]
#group=[clean_str(x) for x in group]
#environment=[clean_str(x) for x in environment]

data=pandas.read_csv(params.split,low_memory=False)
#data=db.Table({x:list(data[x]) for x in ['path','latitude','longitude','url','deposit_type','name']})
data=db.Table({x:list(data[x]) for x in data.keys()})
bias=torch.log(torch.Tensor([1/len(options) for i in range(len(options))]))

records=[]
t0=time.time()
for i in range(len(data['path'])):
    print('%d/%d    '%(i,len(data)),end='\r')
    path=data['path'][i]
    dataset=path.split('/')[-2]
    id=path.split('/')[-1].split('.')[0] #infer deposit id from filename
    if 'url' in data.d:
        url=data['url'][i]
    else:
        url=''
    
    long=data['longitude'][i]
    lat=data['latitude'][i]
    name=data['name'][i]
    sme_label=data['deposit_type'][i]
    
    #load scores
    fname=os.path.join(params.scores,path.replace('.json','.gz'))
    if not os.path.exists(fname):
        continue;
    
    scores=torch.load(gzip.open(fname,'rb'),map_location='cpu')
    if isinstance(scores,dict):
        justification=scores['justification']
        scores=scores['scores']
    else:
        justification=''
    
    scores=F.softmax(scores[:len(options)]+bias,dim=-1)
    scores[~torch.isfinite(scores)]=1/len(options)
    s,ind=scores.sort(dim=0,descending=True)
    if s[0]<=params.score_threshold:
        continue
    
    group_idx,sg=group_scores(scores.tolist(),group)
    env_idx,senv=group_scores(scores.tolist(),environment)
    
    #Prediction text
    pred=''
    for j in range(5):
        pred+='%d. %s, p=%.4f\n'%(j+1,options[ind[j]],s[j])
    
    
    
    record={}
    record['url']=url
    record['source']=dataset
    record['id']=id
    record['name']=name
    record['x']=long
    record['y']=lat
    record['algorithm']='SRI deposit classifier v2'
    record['prediction']=pred
    record['justification']=justification
    record['relevant_data']=''
    record['vis_label']=options[ind[0]]
    s,ind=torch.Tensor(senv).sort(dim=0,descending=True)
    record['vis_color']='#%s'%env_idx[ind[0]]
    
    record['sme_label']=sme_label
    
    #Top5 deposit type
    s,ind=scores.sort(dim=0,descending=True)
    for j in range(5):
        record['top_type_%d'%(j+1)]=options[ind[j]]
        record['top_type_%d_p'%(j+1)]=float(s[j])
    
    #Top 3 deposit group
    s,ind=torch.Tensor(sg).sort(dim=0,descending=True)
    for j in range(3):
        record['top_group_%d'%(j+1)]=group_idx[ind[j]]
        record['top_group_%d_p'%(j+1)]=float(s[j])
    
    #Top 1 deposit environment
    s,ind=torch.Tensor(senv).sort(dim=0,descending=True)
    for j in range(1):
        record['top_system_%d'%(j+1)]=env_idx[ind[j]]
        record['top_system_%d_p'%(j+1)]=float(s[j])
    
    #Per-type scores
    for j in range(len(options)):
        record['score_type_%s'%(options[j])]=float(scores[j])
    
    #Per-group scores
    for j in range(len(group_idx)):
        record['score_group_%s'%(group_idx[j])]=float(sg[j])
    
    #Per-environment scores
    for j in range(len(env_idx)):
        record['score_system_%s'%(env_idx[j])]=float(senv[j])
    
    records.append(record)


pandas.DataFrame.from_records(records).to_csv(params.out)


#Add CA and MRDS GT labels

