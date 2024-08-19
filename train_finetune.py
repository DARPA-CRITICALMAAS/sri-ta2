
import time
import os
import gc
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
from torch.utils.data import DataLoader

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers import pipeline
from torch.optim import Optimizer


import util.helper_lm as helper


default_params=smartparse.obj();
default_params.root='dataset/zinc/NI_43-101_US-Zn_OCR/'
#default_params.lm='NousResearch/Llama-2-7b-hf'
default_params.lm='google/gemma-2b'
default_params.data_test='dataset/ft_v0/data_holdout.pt'

default_params.L=300
default_params.T=0.01
default_params.r=64
default_params.bsz=16
default_params.lora_dropout=0.00
default_params.load=''

params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv;
session=session_manager.create_session(params);


from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=64,
    use_rslora=True,
    #target_modules=["q", "v"],
    lora_dropout=params.lora_dropout,
    bias="none",
)

model,tokenizer=helper.load_lm(params.lm,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2") # max_mem=(0.05,0.3),
model = get_peft_model(model, lora_config)

if not params.load=='':
    print('loading checkpoint %s'%params.load)
    checkpoint=torch.load(params.load)
    model.load_state_dict(checkpoint['net'])


class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().clone() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad.data.float()
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_( grad, grad,value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # print(type(fp32_p))
                fp32_p.addcdiv_(exp_avg, denom,value=-step_size)
                p.data = fp32_p.type(p.data.dtype)
        
        return loss

def clear_json(x):
    if isinstance(x,list):
        return [clear_json(v) for v in x]
    elif isinstance(x,dict):
        return {k:clear_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
    else:
        return x


prefix_template=[]
prefix_template.append('Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references.\n\n{description} For example, here is an example MRDS json record for a {option} type mineral deposit:\n\n')
prefix_template.append('{description} For example, here is an example MRDS json record for a {option} type mineral deposit:\n\n')
prefix_template.append('Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references.\n\nFor example, here is an example MRDS json record for a {option} type mineral deposit:\n\n')
prefix_template.append('For example, here is an example MRDS json record for a {option} type mineral deposit:\n\n')
prefix_template.append('Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references. {description}For example, here is an example MRDS json record for a {option} type mineral deposit:')
prefix_template.append('{description}For example, here is an example MRDS json record for a {option} type mineral deposit:')
prefix_template.append('Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references. For example, here is an example MRDS json record for a {option} type mineral deposit:')
prefix_template.append('For example, here is an example MRDS json record for a {option} type mineral deposit:')

Ls=[300]

class Dataset:
    def __init__(self,data,L=200,test=False):
        self.test=test
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,ii):
        #Random crop
        def crop(text,L):
            tokens_body=tokenizer(text,add_special_tokens = False)['input_ids']
            if len(tokens_body)>L:
                i=int(torch.LongTensor(1).random_(0,len(tokens_body)-L+1))
                tokens_body=tokens_body[i:i+L]
            
            return tokens_body
        
        def prompt(tokens_body,prefix,p,L):
            pad=0#tokenizer.pad_token_id
            tokens_prefix=tokenizer(prefix)['input_ids']
            #tokens_prefix=tokens_prefix[-1:]
            '''
            tokens_body=tokenizer(text,add_special_tokens = False)['input_ids']
            if len(tokens_body)>L:
                i=int(torch.LongTensor(1).random_(0,len(tokens_body)-L+1))
                tokens_body=tokens_body[i:i+L]
            
            '''
            assert len(tokens_prefix)>=1
            input_ids=tokens_prefix+tokens_body
            #print(tokenizer.decode(tokens_prefix))
            #print('')
            target_ids=[pad for i in tokens_prefix[:-1]]+tokens_body+[pad] #assuming 0 is not a common dictionary token
            attention_mask=[1 for i in input_ids]
            #print('input %d'%ii,tokenizer.decode(input_ids))
            #print('target %d'%ii,tokenizer.decode(target_ids[len(tokens_prefix)-1:]))
            return {'input_ids':input_ids,'target_ids':target_ids,'attention_mask':attention_mask,'prior':torch.Tensor([p])}
        
        def stack(data):
            pad=0#tokenizer.pad_token_id
            Lmax=max([len(x['input_ids']) for x in data])
            attention_mask=[x['attention_mask']+[0]*(Lmax-len(x['attention_mask'])) for x in data]
            input_ids=[x['input_ids']+[pad]*(Lmax-len(x['input_ids'])) for x in data]
            target_ids=[x['target_ids']+[pad]*(Lmax-len(x['target_ids'])) for x in data]
            p=torch.cat([x['prior'] for x in data])
            return {'input_ids':input_ids,'attention_mask':attention_mask,'target_ids':target_ids,'prior':p}
        
        
        if not self.test:
            L=Ls[int(torch.LongTensor(1).random_(len(Ls)))]
        else:
            L=params.L
        
        if not self.test:
            template=prefix_template[int(torch.LongTensor(1).random_(len(prefix_template)))]
        else:
            template=prefix_template[0]
        
        item=self.data[ii]
        text=item['data']
        tokens_body=crop(text,L)
        data=[]
        for i in range(min(len(item['options']),3)):
            data.append(prompt(tokens_body,template.format(option=item['options'][i],description=item['descriptions'][i]),p=item['prior'][i],L=L))
        
        data=stack(data)
        input_ids=torch.LongTensor(data['input_ids'])#;print(input_ids.shape)
        attention_mask=torch.LongTensor(data['attention_mask'])#;print(attention_mask.shape)
        target_ids=torch.LongTensor(data['target_ids'])#;print(target_ids.shape)
        prior=data['prior']
        return input_ids,attention_mask,target_ids,prior

def collate(stuff):
    pad=0#tokenizer.pad_token_id
    Lmax=max([x[0].shape[-1] for x in stuff])
    input_ids=torch.stack([F.pad(x[0],(0,Lmax-x[0].shape[-1]),value=pad) for x in stuff],dim=0)
    attention_mask=torch.stack([F.pad(x[1],(0,Lmax-x[1].shape[-1]),value=0) for x in stuff],dim=0)
    target_ids=torch.stack([F.pad(x[2],(0,Lmax-x[2].shape[-1]),value=pad) for x in stuff],dim=0)
    prior=torch.stack([x[3] for x in stuff],dim=0)
    return input_ids,attention_mask,target_ids,prior

dataset_train=torch.load('dataset/ft_v0/data_train.pt')
dataset_test=torch.load(params.data_test)
dataset_train=Dataset(dataset_train,test=False)
dataset_test=Dataset(dataset_test,test=True)

data_train=DataLoader(dataset_train,batch_size=1,collate_fn=collate,shuffle=True,num_workers=8,drop_last=True)
data_test=DataLoader(dataset_test,batch_size=1,collate_fn=collate,shuffle=True,num_workers=8)

pad=0#tokenizer.pad_token_id
def forward(model,input_ids,attention_mask,target_ids,prior,test=False):
    B,K,L=input_ids.shape
    #print(input_ids.shape)
    input_ids=input_ids.view(B*K,L).cuda()
    attention_mask=attention_mask.view(B*K,L).cuda()
    target_ids=target_ids.view(B*K,L).cuda()
    prior=prior.cuda()
    
    logits=model(input_ids=input_ids)['logits'] #,attention_mask=attention_mask
    logits=F.log_softmax(logits,dim=-1)
    logits=logits.gather(-1,target_ids.unsqueeze(-1)).squeeze(-1)
    mask=target_ids.ne(pad).float()
    n=mask.sum(-1)
    s=(logits*mask).sum(-1)/(n+1e-20)
    #print(s)
    s=s.view(B,K)
    
    pred=F.log_softmax(s/params.T+torch.log(prior),dim=-1)
    loss=-pred[:,0].mean()
    
    return loss,pred,s

t0=time.time()
opt=Adam16(model.parameters(),lr=1e-5)
for epoch in range(50):
    tracker=session_manager.loss_tracker()
    if (epoch)%1==0:
        with torch.no_grad():
            model.eval()
            for i,(input_ids,attention_mask,target_ids,prior) in enumerate(data_test):
                #input_ids,attention_mask,target_ids=permute(input_ids,attention_mask,target_ids)
                loss,pred,s=forward(model,input_ids,attention_mask,target_ids,prior,test=True)
                tracker.add(loss_test=loss)
                print('%d/%d, %s'%(i,len(data_test),tracker.str()),end='\r')
                if (i+1)%1000==0:
                    break;
        
        session.log('epoch %d, %s, time %.2f'%(epoch,tracker.str(),time.time()-t0))
        if epoch>0:
            torch.save({'net':model.state_dict()},session.file('%03d.pt'%epoch))
    
    model.train()
    opt.zero_grad()
    for i,(input_ids,attention_mask,target_ids,prior) in enumerate(data_train):
        loss,pred,s=forward(model,input_ids,attention_mask,target_ids,prior)
        loss.backward()
        tracker.add(loss=loss)
        print('%d/%d, %s'%(i,len(data_train),tracker.str()),end='\r')
        if (i+1)%params.bsz==0:
            opt.step()
            opt.zero_grad()
        
        if (i+1)%10000==0:
            break;
    
    session.log('epoch %d, %s, time %.2f'%(epoch,tracker.str(),time.time()-t0))
    
    