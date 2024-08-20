
import torch
import os
import time
import sys
import json
import pandas
import math
import util.db as db

import gzip
import util.smartparse as smartparse
import util.session_manager as session_manager


default_params=smartparse.obj();
default_params.root='dataset/'
default_params.lm='gpt-4o-mini'
default_params.openai_api_key='your_key'
default_params.options='taxonomy/cmmi_options_full_description_with_number.csv'
default_params.split='index/demo_reports.csv'
default_params.out='predictions/scores_qa_gpt-4o-mini/'

default_params.world_size=1
default_params.rank=0

params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv;
print(smartparse.obj2dict(params))


data=pandas.read_csv(params.split)
fnames=list(data['path'])

#Compile options
try:
    labels=pandas.read_csv(params.options)
except:
    labels=pandas.read_csv(params.options,encoding='latin1')

options=list(labels['Deposit type'])
descriptions=list(labels['Description'])
descriptions=[x if type(x)==str else '' for x in descriptions]

list_of_options=''
for i in range(len(options)):
    list_of_options+='a%03d. %s. %s\n'%(i+1,options[i],descriptions[i])


#OpenAI
import backoff 
import openai
client = openai.OpenAI(
  api_key=params.openai_api_key,
)
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

#Prompting approach
def run(text,max_retries=20):
    #Limit text length to 100k
    #Longer documents will not fit within the API
    if len(text)>200000:
        text=text[:200000]
    
    #Compose query
    msg=[]
    msg.append({"role": "system", "content": "You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context description and PDF report and answer the multiple-choice question about the PDF report."})
    msg+=[{"role": "user", "content": "Context: NI 43-101 applies broadly to companies both public and private, and to a variety of disclosures types including mineral exploration reports, reporting of resources and reserves, presentations, oral comments, and websites. NI 43-101 covers mineral products such, precious metals and solid energy commodities as well as bulk minerals, dimension stone, precious stones and mineral sands commodities. The following is an NI 43-101 report describing a mineral resource. Please read and answer the question below.\n\n```json\n{json_data}\n```\nQuestion: Which of the following mineral deposit types best fits the area that the context PDF report describes? Options:\n{list_of_options}\n\nPlease select the mineral deposit type that best fits the area that the context PDF report describes. Please choose only 1 most likely option. Answer the question with only the 4-letter alpha-numeric id (a***) of the most likely option and nothing else.".format(json_data=text,list_of_options=list_of_options)}]
    
    for i in range(max_retries):
        response=completions_with_backoff(model=params.lm, messages=msg,logprobs=True,top_logprobs=5).choices[0]
        
        #Verify that top answer is like 'a000' as instructed, otherwise retry
        #Most importantly, the first token needs to be 'a'
        p0=response.logprobs.content[0].logprob
        t0=response.logprobs.content[0].token
        p1=response.logprobs.content[1].logprob
        t1=response.logprobs.content[1].token
        if not (t0=='a' and len(t1)==3):
            continue
        
        #Obtain top5 answers and their logprobs
        pred=[]
        for x in response.logprobs.content[1].top_logprobs:
            try:
                pred_x=(int(x.token)-1,x.logprob)
                assert pred_x[0]<len(options)
                assert pred_x[0]>=0
                pred.append(pred_x)
            except:
                pass
        
        if len(pred)==0:
            continue
        
        #Compose probability distribution
        p=torch.zeros(len(options))+math.log(1e-20)
        for x in pred:
            p[x[0]]=x[1]
        
        return p
    
    #Didn't get valid answer in 20 tries.
    return torch.zeros(len(options))+math.log(1e-20)

def run_explanation(text):
    #Limit text length to 100k
    #Longer documents will not fit within the API
    if len(text)>200000:
        text=text[:200000]
    
    #Compose query
    msg=[]
    msg.append({"role": "system", "content": "You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context description and PDF report and answer the multiple-choice question about the PDF report."})
    msg+=[{"role": "user", "content": "Context: NI 43-101 applies broadly to companies both public and private, and to a variety of disclosures types including mineral exploration reports, reporting of resources and reserves, presentations, oral comments, and websites. NI 43-101 covers mineral products such, precious metals and solid energy commodities as well as bulk minerals, dimension stone, precious stones and mineral sands commodities. The following is an NI 43-101 report describing a mineral resource. Please read and answer the question below.\n\n```json\n{json_data}\n```\nQuestion: Which of the following mineral deposit types best fits the area that the context PDF report describes? Options:\n{list_of_options}\n\nPlease select the mineral deposit type that best fits the area that the context PDF report describes. Please choose only 1 most likely option. Answer the question with the most likely option and explain your reasoning.".format(json_data=text,list_of_options=list_of_options)}]
    
    response=completions_with_backoff(model=params.lm, messages=msg).choices[0]
    explanation=response.message.content
    return explanation


#Compute some scores
t0=time.time()
for i,fname in enumerate(fnames):
    text=json.load(open(os.path.join(params.root,fname),'r'))
    if isinstance(text,list):
        text='\n'.join(text)
    
    print('%d/%d, time %.2f '%(i,len(fnames),time.time()-t0))
    fname_out=os.path.join(params.out,fname.replace('.json','.gz'))
    
    if i%params.world_size==params.rank and not os.path.exists(fname_out):
        scores=run(text)
        explanation=run_explanation(text)
        os.makedirs(os.path.dirname(fname_out),exist_ok=True)
        torch.save({'scores':scores,'justification':explanation},gzip.open(fname_out,'wb'))

