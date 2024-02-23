
'''
pip install transformers==4.31
pip install accelerate
pip install tokenizers
pip install sentencepiece
pip install protobuf==3.19
pip install openpyxl


pip install pdf2image
pip install pytesseract
pip install opencv-python
conda install tesseract
conda install poppler

pip install openai
pip install accelerate
pip install backoff

'''
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

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel

import helper
import ocr
import util.smartparse as smartparse
import util.session_manager as session_manager


def default_params():
    params=smartparse.obj();
    params.json='test.json'
    params.options='labels_type.csv'
    params.lm='NousResearch/Llama-2-7b-hf'
    params.openai_key=''
    
    params.T=0.01
    params.bsz=2
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);

t0=time.time()

#Load the json file 
with open(params.json,'r') as f:
    data=json.load(f)

#Recursively remove uninformative fields (Specifically designed for MRDS records)
def clean_json(x):
    if isinstance(x,list):
        return [clean_json(v) for v in x]
    elif isinstance(x,dict):
        return {k:clean_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
    else:
        return x

data=clean_json(data)


#Prepare deposit type options
def clean_str(s):
    s=s.replace('\n','')
    s=s.replace('  ',' ')
    s=s.replace('- ','-')
    s=s.replace(' -','-')
    return s

labels=pandas.read_csv(params.options)
options=list(labels['label'])
#options_aug=list(labels['Description']) #not used in current pipeline due to lack of evaluation. But potentially better perf
print('Load labels time %.2f'%(time.time()-t0))

mrds_info='Mineral Resources Data System (MRDS)\n\nMRDS is a collection of reports describing metallic and nonmetallic mineral resources throughout the world. Included are deposit name, location, commodity, deposit description, geologic characteristics, production, reserves, resources, and references. It subsumes the original MRDS and MAS/MILS. MRDS is large, complex, and somewhat problematic. This service provides a subset of the database comprised of those data fields deemed most useful and which most frequently contain some information, but full reports of most records are available as well.\n\nCurrent status: As of 2011, USGS has ceased systematic updates to MRDS, and is working to create a new database, focused primarily on the conterminous US. For locations outside the United States, MRDS remains the best collection of reports that USGS has available. For locations in Alaska, the Alaska Resource Data File remains the most coherent collection of such reports and is in continuing development.\n\nResource descriptions here include an indication of the overall quantity and diversity of information they contain. Many records in this database are simple reports of commodity at some location, but some records provide substantial detail of the geological setting and industrial exploitation of the resource. To help users find these more thorough records, a map interface and search form are provided that rank results by overall quality, records graded A having more information about more aspects of the resource, records graded D having only summary information about the resource. Records graded B and C are intermediate between these, and records graded E generally lack bibliographic references.\n\n'
options_desc=[mrds_info+'Here is an example MRDS json record for a %s type mineral deposit:\n\n'%(clean_str(options[i])) for i in range(len(options))]


#Load LLM
model,tokenizer=helper.load_lm(params.lm,dtype=torch.half)
print('Load LLMs time %.2f'%(time.time()-t0))

#Perform QA
scores,paras,doc_short,scores_raw=helper.RAPMC_json(model,tokenizer,data,options_desc,params=params)
session.log('')
s,ind=scores.sort(dim=-1,descending=True)
ind=ind.tolist()
for i in range(10):
    o=options[int(ind[i])]
    session.log('Rank %d\t%.4f\t%s'%(i+1,s[i],o.replace('\n','')))

session.log('')
df=pandas.DataFrame({'label':[options[ind[i]] for i in range(len(ind))],'score':s.tolist()})
df.to_csv(session.file('predictions.csv'))

#Check top relevant paragraphs
scores_raw=F.softmax(scores_raw/params.T,dim=-1)
s_para,ind_para=scores_raw[:,ind[0]].sort(dim=0,descending=True)
df=pandas.DataFrame({'paragraph':[tokenizer.decode(paras[ind_para[i]]) for i in range(len(ind_para))],'score':s_para.tolist()})
df.to_csv(session.file('relevant_paragraphs.csv'))
print('Prediction time %.2f'%(time.time()-t0))

#Generate explanations
if not params.openai_key=='':
    doc_short='\n\n...'.join([tokenizer.decode(x) for x in doc_short])
    import backoff 
    import openai 
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)
    
    options_gpt=['%s. %s'%(clean_str(options[ind[i]]),clean_str(options_aug[ind[i]])) for i in range(5)]
    msg0=[{"role": "system", "content": "You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context document and answer the multiple-choice question about the context. There is only one option that is correct. Start your answer with which option you'd like to pick, before showing your reasoning."}]
    msg='Context: %s\nWhich of the following mineral deposit types best fits the area that the document describes?\nA.%s\nB.%s\nC.%s\nD.%s\nE.%s\n'%(doc_short,*[clean_str(x) for x in [options_gpt[i] for i in range(5)]])
    msg=msg0+[{"role": "user", "content": msg}]
    
    openai.api_key = params.openai_key
    completion = completions_with_backoff(model="gpt-4-1106-preview", messages=msg)
    session.log('GPT-4 justification')
    session.log('SYSTEM: %s'%msg[0]['content'][:300])
    session.log('USER: %s'%msg[1]['content'][:300])
    session.log('ASSISTANT: %s'%completion.choices[0].message.content)
    with open(session.file('explanation.json'),'w') as f:
        json.dump({'explanation':completion.choices[0].message.content},f)
    
