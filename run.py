
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
import torch.nn as nn

import helper
import ocr
import util.smartparse as smartparse
import util.session_manager as session_manager


def default_params():
    params=smartparse.obj();
    params.pdf=''
    params.json=''
    
    params.options='taxonomy/cmmi_options_full_gpt4_number.csv'
    params.out='tmp.csv'
    
    params.hf_token=''
    params.openai_key=''
    params.lm='NousResearch/Llama-2-7b-hf'
    
    
    #params.topk=20
    params.L=1000
    #params.topn=50
    #params.topn_nparas=5
    params.T=0.005
    params.bsz=5
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);

t0=time.time()

#Load LLM in 16-bit precision
model,tokenizer=helper.load_lm(params.lm,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2", token= params.hf_token)
print('Load LLMs time %.2f'%(time.time()-t0))

#Extract text from incoming document
def chunk(tokenizer,texts,L=700):
    #Chunking
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

if not params.pdf=='':
    data=ocr.ocr_parallel(params.pdf,num_workers=8)
    text=''.join(data)
    lat=(0,0)
    id=params.pdf
    prompt_template='This area has been identified as a {deposit} deposit. {description}\n\n...'
elif not params.json=='':
    def clean_json(x):
        if isinstance(x,list):
            return [clean_json(v) for v in x]
        elif isinstance(x,dict):
            return {k:clean_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
        else:
            return x
    
    def get_lat(x):
        try:
            if 'geometry' in x:
                coord=x['geometry']['coordinates']
                if isinstance(coord[0],list):
                    assert isinstance(coord[0][0],float) or isinstance(coord[0][0],int) 
                    assert isinstance(coord[0][1],float) or isinstance(coord[0][1],int) 
                    return coord[0][0],coord[0][1]
                else:
                    assert isinstance(coord[0],float) or isinstance(coord[0],int) 
                    assert isinstance(coord[1],float) or isinstance(coord[1],int) 
                    return coord[0],coord[1]
            else:
                return 0,0
        except:
            return 0,0
    
    def get_id(item,fname):
        try:
            if not 'id' in item and 'deposits' in item:
                i=item['deposits']['dep_id']
                i='https://mrdata.usgs.gov/mrds/record/%s'%i
            else:
                i=item['id']
        except:
            return fname
        
        return i
    
    data=json.load(open(params.json,'r'))
    data=clean_json(data)
    text=json.dumps(data,indent=2)
    lat=get_lat(data)
    id=get_id(data,params.json)
    prompt_template='Mineral Resources Data System (MRDS) is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references.\n\n{description} For example, here is an example MRDS json record for a {deposit} type mineral deposit:\n\n'
else:
    print('Specify an input document with --pdf or --json')
    a=0/0

chunks,_=chunk(tokenizer,[text],L=params.L)

#Prepare deposit type options
labels=pandas.read_csv(params.options, encoding ='latin1')
options=list(labels['Deposit type'])
group=list(labels['Deposit group'])
environment=list(labels['Deposit environment'])
descriptions=[x if isinstance(x,str) else '' for x in list(labels['Description'])]


if not params.pdf=='':
    retrieval_prompt='With these information, researchers believe that the mineral deposit in this area follows the type of'
    options_desc=[prompt_template.format(deposit=option,description=descriptions[i]) for i,option in enumerate(options)]
    scores_agg,paras,doc_short,scores=helper.RAPMC(model,tokenizer,text,options_desc,retrieval_prompt,params)
    s,ind=scores_agg.sort(dim=-1,descending=True)
    #Organize predictions into text
    pred=''
    for j in range(5):
        pred+='%d. %s, p=%.4f\n'%(j+1,options[ind[j]],s[j])

    #Check top relevant paragraphs
    s_para,ind_para=scores[:,ind[0]].sort(dim=0,descending=True)
    relevant=tokenizer.decode(paras[int(ind_para[0])])
else:
    #Compute scores on document
    t0=time.time();
    L=params.L
    bsz=params.bsz
    pad=tokenizer.pad_token_id

    #Run chunk perplexity through each option
    scores=[]
    with torch.no_grad():
        for i,option in enumerate(options):
            prompt=prompt_template.format(deposit=option,description=descriptions[i])
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
                logp_i=F.log_softmax(logp_i,dim=-1)
                logp_i=logp_i.gather(-1,target_ids[j:r].unsqueeze(-1)).squeeze(-1)
                n=target_ids[j:r].ne(pad).sum(dim=-1)
                s=(logp_i*target_ids[j:r].ne(pad)).sum(dim=-1)
                s=s/(n+1e-20)
                scores_i.append(s)
            
            scores_i=torch.cat(scores_i,dim=0) #ntext
            scores.append(scores_i.cpu())
        
        scores=torch.stack(scores,dim=-1)
    
    
    #Compute type/group/environment predictions
    scores=F.softmax(scores/params.T,dim=-1)
    scores_agg=scores.mean(dim=0)
    
    s,ind=scores_agg.sort(dim=-1,descending=True)
    #Organize predictions into text
    pred=''
    for j in range(5):
        pred+='%d. %s, p=%.4f\n'%(j+1,options[ind[j]],s[j])

    #Check top relevant paragraphs
    s_para,ind_para=scores[:,ind[0]].sort(dim=0,descending=True)
    relevant=tokenizer.decode(chunks[int(ind_para[0])])
    relevant_topk=[tokenizer.decode(chunks[int(ind_para[i])]) for i in range(min(5,len(ind_para)))]
    doc_short='\n\n...'.join(relevant_topk)



def aggregate(s,v):
    vidx=sorted(list(set(v)))
    s2=[[] for i in vidx]
    for i,si in enumerate(s):
        ind=vidx.index(v[i])
        s2[ind].append(si)
    
    s2=[max(x) for x in s2]
    s2=[x/(sum(s2)+1e-12) for x in s2]
    return vidx,s2

group_idx,sg=aggregate(scores_agg.tolist(),group)
env_idx,senv=aggregate(scores_agg.tolist(),environment)

#Get justification
if not params.openai_key=='':
    
    import backoff 
    import openai 
    client = openai.OpenAI(
      api_key=params.openai_key,
    )
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)
    
    options_gpt=['%d. %s. %s'%(i+1,options[ind[i]],descriptions[ind[i]]) for i in range(5)]
    options_gpt='\n'.join(options_gpt)
    
    msg0=[{"role": "system", "content": "You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context document and answer the multiple-choice question about the context. There is only one option that is correct. Start your answer with which option you'd like to pick, and then show your reasoning."}]
    msg='Context: %s\nWhich of the following mineral deposit types best fits the area that the document describes?\n%s'%(doc_short,options_gpt)
    msg=msg0+[{"role": "user", "content": msg}]
    
    openai.api_key = params.openai_key
    completion = completions_with_backoff(model="gpt-4-turbo", messages=msg)
    explanation=completion.choices[0].message.content
else:
    explanation=''

record={}
record['id']=id
record['x']=lat[0]
record['y']=lat[1]
record['algorithm']='SRI deposit type classification, v1, 20240426'
record['prediction']=pred
record['relevant_data']=relevant
record['vis_label']=options[ind[0]]
s,ind=torch.Tensor(senv).sort(dim=0,descending=True)
record['vis_color']='#%s'%env_idx[ind[0]]
record['justification']=explanation


#Top5 deposit type
s,ind=scores_agg.sort(dim=0,descending=True)
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
    record['score_type_%s'%(options[j])]=float(scores_agg[j])

#Per-group scores
for j in range(len(group_idx)):
    record['score_group_%s'%(group_idx[j])]=float(sg[j])

#Per-environment scores
for j in range(len(env_idx)):
    record['score_system_%s'%(env_idx[j])]=float(senv[j])

pandas.DataFrame.from_records([record]).to_csv(params.out)