
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
import backoff 
import openai 
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

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

import util.smartparse as smartparse
import util.session_manager as session_manager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers import pipeline


import helper
import ocr


def default_params():
    params=smartparse.obj();
    params.pdf='4530692_GSTEH9EJ_HC6MXVK4.pdf'
    params.options='labels_type.csv'
    params.lm='NousResearch/Llama-2-7b-hf'
    #params.lm='openlm-research/open_llama_3b_v2'
    params.topk=50
    params.style='v0'
    params.chunk=300
    params.T=1e-2
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);

t0=time.time()

data=ocr.ocr_parallel(params.pdf)
print('OCR time %.2f'%(time.time()-t0))


model_embed,tokenizer_embed=helper.load_embedding('BAAI/bge-large-en-v1.5')
model,tokenizer=helper.load_lm(params.lm,dtype=torch.half)
print('Load LLMs time %.2f'%(time.time()-t0))


labels=pandas.read_csv(params.options)
options=list(labels['label'])
options_aug=list(labels['Description'])
print('Load labels time %.2f'%(time.time()-t0))

#Prepare QA templates
style=params.style
if style=='v2':
    q_template='DEPOSIT MODEL OF {option}\n{option} deposits have been spotted in the example below. {option_aug}\n\nEXAMPLE OF {option} deposit\n"...'
    a_template='{para}'
    sep='</s><s>'
elif style=='v1':
    q_template='DEPOSIT MODEL OF {option}\n{option} deposits have been spotted in the example below. {option_aug}\n\nEXAMPLE OF {option} deposit\n"...'
    a_template='{para}'
    sep='\n\n'
elif style=='v0':
    q_template='This area has been identified as a {option} deposit. {option_aug}\n\n...'
    a_template='{para}'
    sep='\n\n'
elif style=='title':
    q_template='This area has been identified as a {option} deposit. \n\n...'
    a_template='{para}'
    sep='\n\n'
elif style=='qa':
    q_template='Answer the yes/no question based on the context below with only "Yes" or "No", or "Unsure" if the context does not provide enough information.\n\nContext: ...{para}...\n\nQuestion: Is the mineral deposit in this area a {option} deposit?\n\n Answer:'
    a_template='Yes'
    sep='\n\n'
elif style=='qav2':
    q_template=helper.qa_template('Context: ...{para}...\n\nQuestion: Is the mineral deposit in this area a {option} deposit?',sys='You are a friendly geologist chatbot who always responds to user questions accurately. Answer the yes/no question based on the context below with only "Yes" or "No", or "Unsure" if the context does not provide enough information.',style='zephyr')
    a_template='Yes'
    sep='\n\n'
elif style=='qa_aug':
    q_template='Answer the yes/no question based on the context below with only "Yes" or "No", or "Unsure" if the context does not provide enough information.\n\nContext: ...{para}...\n\nQuestion: Is the mineral deposit in this area a {option} deposit? {option_aug}\n\n Answer:'
    a_template='Yes'
    sep='\n\n'
else:
    a=0/0

def qa_few_shot(q_template,a_template,sep,examples):
    q=''
    a=''
    for i,ex in enumerate(examples):
        qi=q_template.format(**ex)
        ai=a_template.format(**ex)
        q+=qi
        if i<len(examples)-1:
            q+=ai
            q+=sep
        else:
            a+=ai
    
    return q,a




s=params.chunk
all_data=''.join(data)
data_chunks=[all_data[i:i+s] for i in range(0,len(all_data),s//2)]

#Filter text, return top 10 most relevant chunks
ref='Detailed studies reveal that this area contains a %s deposit.'
ref=[ref%o for o in options]
data2,s_embed=helper.filter_embedding(model_embed,tokenizer_embed,data_chunks,ref,topk=50)
print('Retrieve relevant paragraphs time %.2f'%(time.time()-t0))

s=F.softmax(s_embed/params.T,dim=-1).mean(dim=0)
s,ind=s.sort(dim=-1,descending=True)
options=[options[i] for i in ind[:50].tolist()]
options_aug=[options_aug[i] for i in ind[:50].tolist()]
print('Subselect options time %.2f'%(time.time()-t0))
data2=data2[:params.topk]

#Scores
scores=[]
for i,para in enumerate(data2):
    for j,o in enumerate(options):
        q,a=qa_few_shot(q_template,a_template,sep,[{'para':para,'option':o,'option_aug':options_aug[j]}])
        s0,s1=helper.perplexity2(model,tokenizer,q,a)
        scores.append(float(s1.mean()))
        print('QA %d/%d, %d/%d    %f, time %.2f '%(i,len(data2),j,len(options),s1.mean(),time.time()-t0),end='\r')

scores=torch.Tensor(scores).view(len(data2),len(options))
s=F.softmax(scores/params.T,dim=-1).mean(dim=0)
print('QA scores time %.2f'%(time.time()-t0))

torch.save(s,'scores.pt')
s,ind=s.sort(dim=-1,descending=True)
ind=ind.tolist()
for i in range(10):
    o=options[int(ind[i])]
    print('Rank %d\t%.4f\t%s'%(i+1,s[i],o.replace('\n','')))


print('')

#Organize chunks
def concat(data2,data):
    mark=[]
    for j in range(30):
        ind=data.index(data2[j])
        mark.append(max(ind-1,0))
        mark.append(ind)
        mark.append(min(ind+1,len(data2)))
    
    mark=sorted(list(set(mark)))
    
    para=''
    prev=-1
    for j in mark:
        if j==prev+1:
            para+=data[j]
        else:
            para+='...'+data[j]
    
    return para

msg0=[{"role": "system", "content": "You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context document and answer the multiple-choice question about the context, and show your reasoning."}]
msg='Context: %s\n Question: does this document mention any of the following deposits? A.%s. %s\nB.%s. %s\nC.%s. %s\nD.%s. %s\nE.%s. %s\n'%(concat(data2,data_chunks),*[x.replace('\n','') for x in [options[ind[0]],options_aug[ind[0]],options[ind[1]],options_aug[ind[1]],options[ind[2]],options_aug[ind[2]],options[ind[3]],options_aug[ind[3]],options[ind[4]],options_aug[ind[4]]]])

openai.api_key = "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"
completion = completions_with_backoff(model="gpt-4", messages=msg0+[{"role": "user", "content": msg}])
print('GPT-4 justification: %s'%completion.choices[0].message.content)
