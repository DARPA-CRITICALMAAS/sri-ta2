import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers import pipeline



def device_profile():
    n=torch.cuda.device_count()
    return [torch.cuda.get_device_properties(i).total_memory for i in range(n)]

#Obtain embedding from embedding model
def run_embedding(model,tokenizer,sentences,bsz=4):
    e=[]
    for i in range(0,len(sentences),bsz):
        r=min(i+bsz,len(sentences))
        encoded_input = tokenizer(sentences[i:r], padding=True, truncation=True, return_tensors='pt').to(model.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        e.append(sentence_embeddings)
    
    e=torch.cat(e,dim=0)
    return e

#Filter by similarity
def filter_embedding(model,tokenizer,text,ref,topk=10):
    with torch.no_grad():
        e=run_embedding(model,tokenizer,text)
        e0=run_embedding(model,tokenizer,ref)
        s=torch.mm(e,e0.t())
        _,ind=s.max(dim=-1)[0].sort(dim=0,descending=True)
    return [text[i] for i in ind[:topk]],s[ind[:topk]]



def load_embedding(model_name='BAAI/bge-large-en-v1.5',dtype=torch.float,max_mem=0.8):
    mem=device_profile()
    kwargs = dict(
        #device_map="cuda:0",
        #max_memory={i:int(max_mem*m) for i,m in enumerate(mem)},
        torch_dtype=dtype,
        offload_folder='cache/',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval();
    model=model.cuda()
    
    return model,tokenizer

def load_lm(model_name,dtype=torch.float,max_mem=0.8):
    mem=device_profile()
    kwargs = dict(
        device_map="auto",
        max_memory={i:int(max_mem*m) for i,m in enumerate(mem)},
        torch_dtype=dtype,
        offload_folder='cache/',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True, **kwargs)
    model.eval();
    
    return model,tokenizer


#Perplexity for 1 sentence
def perplexity(model,tokenizer,text,IGNORE_INDEX=-100):
    with torch.no_grad():
        x=tokenizer(text,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        if 'token_type_ids' in x:
            _=x.pop('token_type_ids')
        
        x={k:v.to(model.device) for k,v in x.items()}
        
        y=x['input_ids'].data.clone()
        y[:,:-1]=y[:,1:].clone()
        y[:,-1]=IGNORE_INDEX
        
        
        pred=model(**x,labels=y)
        a=F.log_softmax(pred.logits.squeeze(0),dim=-1)
        logp=a.gather(1,y.clamp(min=0).view(-1,1)).view(-1)
        loss=logp[:-1].clone()
    
    return loss


#Perplexity for 2 sentences concatenated
def perplexity2(model,tokenizer,text,option,IGNORE_INDEX=-100):
    with torch.no_grad():
        x=tokenizer(text,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        z=tokenizer(option,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        
        #Try to cut tokens to the length of just the text
        s=tokenizer.decode(z['input_ids'][0])
        if not s==option and s.find(option)>0: #Check if there's bos padding
            z={k:v[:,1:] for k,v in z.items()}
            s=tokenizer.decode(z['input_ids'][0])
        
        if not s==option and s.find(option)==0: #Check if there's eos padding
            z={k:v[:,:-1] for k,v in z.items()}
            x={k:v[:,:-1] for k,v in x.items()}
            s=tokenizer.decode(z['input_ids'][0])
        
        if not s==option: #Not sure what's going on
            print('Warning',s,option,z)
            a=0/0
        
        #Synthesize input/output with option
        #Append option after input
        xz={k:torch.cat((x[k],z[k]),dim=-1) for k,v in x.items()}
        if 'token_type_ids' in xz:
            _=xz.pop('token_type_ids')
        
        xz={k:v.to(model.device) for k,v in xz.items()}
        y=xz['input_ids'].data.clone()
        y[:,:-1]=y[:,1:].clone()
        y[:,-1]=IGNORE_INDEX
        
        '''
        s=tokenizer.decode(y[0,:-1])
        print(s,y[0,-8:-1],x['input_ids'][0,-5:],z['input_ids'][0,:])
        a=0/0
        '''
        
        pred=model(**xz,labels=y)
        #Check CE
        logp=F.log_softmax(pred.logits.squeeze(0),dim=-1)
        logp=logp.gather(1,y.clamp(min=0).view(-1,1)).view(-1)
        
        #print(logp.shape,x['input_ids'][0].shape,z['input_ids'][0].shape,xz['input_ids'][0].shape,)
        loss_q=logp[:x['input_ids'].shape[-1]-1].clone()
        loss_a=logp[x['input_ids'].shape[-1]-1:-1].clone()
    
    #print(loss_q.shape,loss_a.shape)
    return loss_q,loss_a


def perplexity2b(model,tokenizer,text,option,IGNORE_INDEX=-100):
    with torch.no_grad():
        x=tokenizer(text,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        z=tokenizer(option,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        
        #Try to cut tokens to the length of just the text
        s=tokenizer.decode(z['input_ids'][0])
        if not s==option and s.find(option)>0: #Check if there's bos padding
            z={k:v[:,1:] for k,v in z.items()}
            s=tokenizer.decode(z['input_ids'][0])
        
        if not s==option and s.find(option)==0: #Check if there's eos padding
            z={k:v[:,:-1] for k,v in z.items()}
            x={k:v[:,:-1] for k,v in x.items()}
            s=tokenizer.decode(z['input_ids'][0])
        
        if not s==option: #Not sure what's going on
            print('Warning',s,option,z)
            a=0/0
        
        #Synthesize input/output with option
        #Append option after input
        xz={k:torch.cat((x[k],z[k]),dim=-1) for k,v in x.items()}
        if 'token_type_ids' in xz:
            _=xz.pop('token_type_ids')
        
        xz={k:v.to(model.device) for k,v in xz.items()}
        
        y=xz['input_ids'].data.clone()
        y[:,:-1]=y[:,1:].clone()
        y[:,-1]=IGNORE_INDEX
        
        pred=model(**xz,labels=y)
        #Check CE
        logp=F.log_softmax(pred.logits.squeeze(0),dim=-1)
        logp=logp.gather(1,y.clamp(min=0).view(-1,1)).view(-1)
        
        loss_q=logp[:x['input_ids'].shape[-1]-1].clone()
        loss_a=logp[x['input_ids'].shape[-1]-1:-1].clone()
    
    return loss_q,loss_a


#Generation
def generate(model,tokenizer,text,max_new_tokens=500,**kwargs):
    with torch.no_grad():
        x=tokenizer(text,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        y=model.generate(input_ids=x['input_ids'].cuda(),max_new_tokens=500,**kwargs)
        out=tokenizer.decode(y[0].tolist())
        out2=tokenizer.decode(y[0,x['input_ids'].shape[-1]:].tolist())
    
    return out,out2

#LLAMA2 example
'<s>'# padded by tokenizer
'[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nQ '
'[/INST] ans </s><s>'
'[INST] Q '
'[/INST] ans </s><s>'
'[INST] Q '
'[/INST]'

#alpaca example
'<s>'# padded by tokenizer
'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
'### Instruction: Q\n\n'
'### Response: ans</s>' #(One round of dialog, no follow ups)
'### Instruction: Q\n\n'
'### Response:'


#vicuna example
'<s>'# padded by tokenizer
"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
'USER: Q '
'ASSISTANT: ans</s>' #(One round of dialog, no follow ups)
'USER: Q '
'ASSISTANT:'


def qa_template(q,sys=None,style='llama2'):
    if style=='llama2':
        if sys is None:
            sys="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            #prompt='[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]'%(sys,q)
            #prompt='[INST] %s [/INST]'%(q)
            prompt='[INST] %s %s [/INST]'%(sys,q)
        else:
            prompt='[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]'%(sys,q)
    elif style=='alpaca':
        if sys is None:
            prompt='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: %s\n\n### Response:'%(q)
        else:
            prompt='%s\n\n### Instruction: %s\n\n### Response:'%(sys,q)
    elif style=='vicuna':
        if sys is None:
            prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: %s ASSISTANT:"%(q)
        else:
            prompt="%s\n\nUSER: %s ASSISTANT:"%(sys,q)
    elif style=='zephyr':
        if sys is None:
            prompt="<|system|>\nYou are a friendly geologist chatbot who always responds to user's question accurately.</s>\n<|user|>\n%s</s>\n<|assistant|>\n"%(q)
        else:
            prompt="<|system|>\n%s</s>\n<|user|>\n%s</s>\n<|assistant|>\n"%(sys,q)
    
    return prompt