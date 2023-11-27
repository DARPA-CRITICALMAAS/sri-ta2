
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
    params.pdf='4530692_GSTEH9EJ_HC6MXVK4.pdf'
    params.options='labels_type.csv'
    params.lm='NousResearch/Llama-2-7b-hf'
    #params.lm='openlm-research/open_llama_3b_v2'
    params.openai_key=''
    params.style='v0'
    
    params.topk=20
    params.L=200
    params.topn=50
    params.topn_nparas=5
    params.T=0.01
    params.bsz=5
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);

t0=time.time()

#OCR the incoming document
session.log('%s'%params.pdf)
session.log('')
data=ocr.ocr_parallel(params.pdf,num_workers=16)
with open(session.file('ocr.json'),'w') as f:
    json.dump(data,f)

print('OCR time %.2f'%(time.time()-t0))
doc=''.join(data)


#Prepare deposit type options
def clean_str(s):
    s=s.replace('\n','')
    s=s.replace('  ',' ')
    s=s.replace('- ','-')
    s=s.replace(' -','-')
    return s

labels=pandas.read_csv(params.options)
options=list(labels['label'])
options_aug=list(labels['Description'])
print('Load labels time %.2f'%(time.time()-t0))

#Prepare QA templates
style=params.style
if style=='v0':
    options_desc=['This area has been identified as a %s deposit. %s\n\n...'%(clean_str(options[i]),clean_str(options_aug[i])) for i in range(len(options))]
elif style=='title':
    options_desc=['This area has been identified as a %s deposit.\n\n...'%(clean_str(options[i])) for i in range(len(options))]

retrieval_prompt='With these information, researchers believe that the mineral deposit in this area follows the type of'


#Load LLM
model,tokenizer=helper.load_lm(params.lm,dtype=torch.half)
print('Load LLMs time %.2f'%(time.time()-t0))

#Perform QA
scores,paras,doc_short,scores_raw=helper.RAPMC(model,tokenizer,doc,options_desc,retrieval_prompt,params)
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
    
