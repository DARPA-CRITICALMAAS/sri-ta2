
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

import util.db as db
import util.smartparse as smartparse
import util.session_manager as session_manager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers import pipeline
import gzip


import helper


default_params=smartparse.obj();
default_params.root='dataset/'
default_params.lm='meta-llama/Meta-Llama-3-8B'
default_params.hf_token='your_token'
default_params.load='./llama3-8b_ft_011-014.pt'
default_params.options='../science-ft-2/dataset/taxonomy/cmmi_options_full_gpt4_number.csv'
default_params.split='index/all_sites.csv'
default_params.out='predictions/scores_llama3-8b-ft/'

default_params.L=1000
default_params.bsz=3

default_params.world_size=1
default_params.rank=0

params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv;
session=session_manager.create_session(params);


model,tokenizer=helper.load_lm(params.lm,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",token=params.hf_token)
if not params.load=='':
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=64,
        use_rslora=True,
        #target_modules=["q", "v"],
        lora_dropout=0.00,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print('loading checkpoint %s'%params.load)
    checkpoint=torch.load(params.load,map_location='cpu')
    model.load_state_dict(checkpoint['net'])

model.eval()

try:
    labels=pandas.read_csv(params.options)
except:
    labels=pandas.read_csv(params.options,encoding='latin1')

data=pandas.read_csv(params.split)
fnames=list(data['path'])
options=list(labels['Deposit type'])
descriptions=list(labels['Description'])
descriptions=[x if type(x)==str else '' for x in descriptions]



#Chunk multiple text into fixed length chunks, 50% overlap. Also keep index to original document
#Make the last chunk as long as possible by increasing overlap
def chunk(tokenizer,texts,L=700):
    tokens=[tokenizer(t,add_special_tokens = False)['input_ids'] for t in texts]
    chunks=[]
    source_id=[]
    for j,t in enumerate(tokens):
        for i in range(0,len(t),L//2):
            r=min(i+L,len(t))
            if r==len(t):
                i=max(r-L,0)
            
            chunks.append(t[i:r])
            source_id.append(j)
            if r==len(t):
                break;
    
    return chunks,source_id

t0=time.time();
def run(model,tokenizer,texts,options,descriptions,L=1000):
    bsz=params.bsz
    chunks,source_id=chunk(tokenizer,texts,L)
    
    #Run chunk perplexity through each option
    pad=tokenizer.eos_token_id
    scores=[[[] for i in range(len(options))] for t in texts]
    with torch.no_grad():
        for i,option in enumerate(options):
            prompt='Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references.\n\n%s For example, here is an example MRDS json record for a %s type mineral deposit:\n\n'%(descriptions[i],option)
            
            tokens_prompt=tokenizer(prompt)['input_ids']
            
            last_prompt_token=tokens_prompt[-1]
            input_ids=[[last_prompt_token]+x+[pad]*(L-len(x)) for x in chunks]
            target_ids=[x+[pad]*(L+1-len(x)) for x in chunks]
            input_ids=torch.LongTensor(input_ids).cuda()
            target_ids=torch.LongTensor(target_ids).cuda()
            
            cache=model(torch.LongTensor([tokens_prompt[:-1]]).cuda(),use_cache=True)['past_key_values']
            scores_i=[]
            for j in range(0,len(input_ids),bsz):
                print('%d/%d, %d/%d, time %.2f'%(i,len(options),j,len(input_ids),time.time()-t0),end='\r')
                r=min(j+bsz,len(input_ids))
                #Make bigger cache to match tokens1
                cache_i=[[z.repeat(r-j,1,1,1) for z in x] for x in cache]
                logp_i=model(input_ids[j:r],past_key_values=cache_i)['logits']
                #print(tokenizer.decode(input_ids[j:r].tolist()[0]),logp_i.keys()) 
                logp_i=F.log_softmax(logp_i,dim=-1)
                logp_i=logp_i.gather(-1,target_ids[j:r].unsqueeze(-1)).squeeze(-1)
                #print(logp_i.shape)
                
                for k in range(r-j):
                    ind=source_id[j+k]
                    s=logp_i[k][target_ids[k].ne(pad)].cpu().tolist()
                    scores[ind][i]+=s
    
    scores=[torch.Tensor(x).bfloat16() for x in scores]
    #print([len(s) for i,s in scores])
    return scores


def clear_json(x):
    if isinstance(x,list):
        return [clear_json(v) for v in x]
    elif isinstance(x,dict):
        return {k:clear_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
    else:
        return x

#Compute some scores
queue=[]
t0=time.time()
queue_size=20
for i,fname in enumerate(fnames):
    print('%d/%d, time %.2f '%(i,len(fnames),time.time()-t0))
    fname_out=os.path.join(params.out,fname.replace('.json','.gz'))
    if i%params.world_size==params.rank and not os.path.exists(fname_out):
        text=json.load(open(os.path.join(params.root,fname),'r'))
        text=json.dumps(clear_json(),indent=2)
        queue.append((i,text,fname_out))
    
    #Run and clear queue when queue is filled
    if len(queue)>=queue_size or i==len(fnames)-1:
        scores=run(model,tokenizer,[x[1] for x in queue],options,descriptions,L=params.L)
        for j in range(len(queue)):
            fname_out=queue[j][2]
            os.makedirs(os.path.dirname(fname_out),exist_ok=True)
            torch.save(scores[j],gzip.open(fname_out,'wb'))
        
        queue=[]

