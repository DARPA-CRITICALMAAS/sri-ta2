import torch
import json
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
        cache_dir='cache/',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    model=model.cuda()
    
    return model,tokenizer

def load_lm(model_name,dtype=None,max_mem=0.8,**kwargs2):
    mem=device_profile()
    kwargs = dict(
        device_map="auto",
        max_memory={i:int(max_mem*m) for i,m in enumerate(mem)},
        offload_folder='cache/',
        cache_dir='cache/',
    )
    if not dtype is None:
        kwargs['torch_dtype']=dtype
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,**kwargs2)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True, **kwargs,**kwargs2)
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


#Perplexity for 2 sentences concatenated, but with tokens
def perplexity2_tokens(model,tokens0,tokens1,IGNORE_INDEX=-100):
    with torch.no_grad():
        tokens0=torch.LongTensor(tokens0).cuda()
        tokens1=torch.LongTensor(tokens1).cuda()
        input_ids=torch.cat((tokens0,tokens1),dim=-1)
        logp=model(input_ids)['logits']
        logp=F.log_softmax(logp,dim=-1)
        logp=logp[:,:-1,:].gather(-1,input_ids[:,1:].unsqueeze(-1)).sum(-1)
        logp0=logp[:,:tokens0.shape[-1]-1].clone().cpu()
        logp1=logp[:,tokens0.shape[-1]-1:].clone().cpu()
    
    return logp0,logp1

#Use when tokens0 remains being the same thing throughout
#Use_cache to accelerate perplexity computation
def perplexity2_tokens_cache(model,tokens0,tokens1,IGNORE_INDEX=-100,bsz=5):
    with torch.no_grad():
        t=tokens0[-1]
        tokens0=torch.LongTensor(tokens0[:-1]).unsqueeze(0).cuda()
        tokens1=torch.LongTensor([[t]+x for x in tokens1]).cuda()
        cache=model(tokens0,use_cache=True)['past_key_values']
        logp1=[]
        for i in range(0,len(tokens1),bsz):
            r=min(i+bsz,len(tokens1))
            #Make bigger cache to match tokens1
            cache_i=[[z.repeat(r-i,1,1,1) for z in x] for x in cache]
            logp_i=model(tokens1[i:r],past_key_values=cache_i)['logits']
            logp_i=F.log_softmax(logp_i,dim=-1)
            logp1_i=logp_i[:,:-1,:].gather(-1,tokens1[i:r,1:].unsqueeze(-1)).sum(-1).cpu()
            logp1.append(logp1_i)
        
        logp1=torch.cat(logp1,dim=0)
    return logp1

#Use when tokens0 remains being the same thing throughout
#Use_cache to accelerate perplexity computation
#Pad short tokens1 sequences with -1
def perplexity2_tokens_v3(model,tokens0,tokens1,IGNORE_INDEX=-100,bsz=5):
    with torch.no_grad():
        t=tokens0[-1]
        tokens0=torch.LongTensor(tokens0[:-1]).unsqueeze(0).cuda()
        L=max([len(s) for s in tokens1])
        tokens1=[s+[-1]*(L-len(s)) for s in tokens1]
        mask=torch.LongTensor(tokens1).cuda().ge(0).float()
        tokens1=torch.LongTensor([[t]+x for x in tokens1]).cuda()
        tokens1=tokens1.clamp(min=0)
        
        
        cache=model(tokens0,use_cache=True)['past_key_values']
        logp1=[]
        for i in range(0,len(tokens1),bsz):
            r=min(i+bsz,len(tokens1))
            #Make bigger cache to match tokens1
            cache_i=[[z.repeat(r-i,1,1,1) for z in x] for x in cache]
            logp_i=model(tokens1[i:r],past_key_values=cache_i)['logits']
            logp_i=F.log_softmax(logp_i,dim=-1)
            logp1_i=logp_i[:,:-1,:].gather(-1,tokens1[i:r,1:].unsqueeze(-1)).sum(-1).cpu()
            logp1.append(logp1_i)
        
        logp1=torch.cat(logp1,dim=0)
    return logp1,mask.data.cpu()

def chunker_json(model=None,tokenizer=None,json_data=None,chunk_nrows=20,chunk_max_chars=300):
    def chunk(s):
        rows=s.split('\n')
        rows_=[]
        #Split long rows into multiple rows
        for l in rows:
            if len(l)>chunk_max_chars:
                for i in range(0,len(l),chunk_max_chars):
                    r=min(i+chunk_max_chars,len(l))
                    rows_.append(l[i:r])
            else:
                rows_.append(l)
        
        rows=rows_
        
        #Divide into chunks 
        chunks=[]
        step=chunk_nrows//2
        for i in range(0,len(rows),step):
            r=min(i+chunk_nrows,len(rows))
            if not(i>0 and r-i<step-1):
                chunks.append('\n'.join(rows[i:r]))
        
        return chunks
    
    
    #Works only for the LLaMA family
    text=json.dumps(json_data, indent=2)
    chunks=chunk(text)
    data=[tokenizer(x)['input_ids'][1:] for x in chunks]
    return data,text


#Return text does not have BOS
def filter_lm(model,tokenizer,document,prompt='',topk=50,L=200,bsz=5):
    #Works only for the LLaMA family
    bos=tokenizer('a.')['input_ids'][0]
    stop=tokenizer('a.')['input_ids'][-1]
    #Find appropriate places for inserting this sentence
    all_tokens=tokenizer(document)['input_ids'][1:]
    q=tokenizer(prompt)['input_ids'][1:]
    
    N=L
    step=L//2
    
    #Find possible places for inserting the sentence
    stops=[0]+[i for i,t in enumerate(all_tokens) if t==stop]
    tprev=0
    stops2=[]
    for t in stops:
        if t>=N-1 and t-tprev>=step:
            s=min([i for i in stops if i>=t-N+1])
            #stops2.append((s+1,t+1))
            stops2.append((t-N+1,t+1))
            tprev=t
    
    data=[[bos]+all_tokens[s:t] for s,t in stops2]
    #print(stops2)
    
    #Batch inferencing to cap GPU util
    #Leverages the fact that all chunks are the same num.tokens
    batch=bsz
    scores=[]
    for i in range(0,len(data),batch):
        print('filter %d/%d   '%(i,len(data)),end='\r')
        data_i=data[i:i+batch]
        _,logp=perplexity2_tokens(model,data_i,[q for i in range(len(data_i))])
        scores.append(logp.mean(dim=-1))
    
    scores=torch.cat(scores,dim=-1)
    _,ind=scores.sort(dim=0,descending=True)
    pos=ind[:topk].tolist()
    
    out=[]
    for i in pos:
        s,t=stops2[i]
        #Move the start position to a full stop
        #Leave the end position dangling
        s=min([i for i in stops if i>=t-N+1])
        if s+N+1>len(all_tokens):
            s=max([i for i in stops if i+N+1<=len(all_tokens)])
        
        out.append((s+1,s+N+1))
    
    doc=[all_tokens[x[0]:x[1]] for x in sorted(out,key=lambda x:x[0])]
    out=[all_tokens[x[0]:x[1]] for x in out]
    return out,doc #[data[i][1:] for i in pos] if we want start position dangling

#Prefix-based multiple-choice
def prefix_mc(model,paras,options,T=0.01,bsz=5):
    scores=[]
    for i,o in enumerate(options):
        s=perplexity2_tokens_cache(model,o,paras,bsz=bsz)
        scores.append(s.mean(dim=-1))
        print('Prefix MC %d/%d'%(i,len(options)),end='\r')
    
    scores=torch.cat(scores,dim=0).view(len(options),len(paras)).t().contiguous()
    scores_avg=F.softmax(scores/T,dim=-1).mean(dim=0)
    return scores_avg,scores

#Retrieval + prefix-based multiple-choice
def RAPMC(model,tokenizer,document,options,retrieval_prompt,params=None):
    import util.smartparse as smartparse
    default_params=smartparse.obj()
    default_params.topk=20
    default_params.L=200
    default_params.T=0.01
    default_params.bsz=5
    default_params.short_list=True
    default_params.topn=50
    default_params.topn_nparas=5
    params = smartparse.merge(params, default_params)
    
    #Retrieve relevant paragraphs (in tokens)
    paras,doc=filter_lm(model,tokenizer,document,prompt=retrieval_prompt,topk=params.topk,L=params.L,bsz=params.bsz)
    
    #Run question answering on top 5 paragraphs to narrow down options
    options_tok=[tokenizer(o)['input_ids'] for o in options]
    if params.short_list:
        scores,_=prefix_mc(model,paras[:params.topn_nparas],options_tok,T=params.T,bsz=params.bsz)
        _,ind=scores.sort(dim=-1,descending=True)
        ind=ind[:params.topn].tolist()
    else:
        ind=list(range(len(options_tok)))
    
    options_tok_short=[options_tok[i] for i in ind]
    
    #Run full question answering on short list
    scores_short,scores_short_raw=prefix_mc(model,paras,options_tok_short,T=params.T,bsz=params.bsz)
    
    #Map short list probs to original list
    scores=torch.Tensor(len(options)).fill_(0)
    scores[ind]=scores_short.float()
    
    scores_raw=torch.Tensor(len(paras),len(options)).fill_(-1e10)
    scores_raw[:,ind]=scores_short_raw.float()
    return scores,paras,doc,scores_raw

def RAPMC_json(model,tokenizer,json_data,options,params=None):
    import util.smartparse as smartparse
    default_params=smartparse.obj()
    default_params.bsz=2
    default_params.T=0.01
    params = smartparse.merge(params, default_params)
    
    paras,doc=chunker_json(model,tokenizer,json_data)
    options=[tokenizer(o)['input_ids'] for o in options]
    
    scores=[]
    for i,o in enumerate(options):
        s,mask=perplexity2_tokens_v3(model,o,paras,bsz=params.bsz)
        savg=(s*mask).sum(dim=-1)/(mask.sum(dim=-1)+1e-20)
        scores.append(savg)
        print('Prefix MC %d/%d'%(i,len(options)),end='\r')
    
    scores_raw=torch.cat(scores,dim=0).view(len(options),len(paras)).t().contiguous()
    scores=F.softmax(scores_raw/params.T,dim=-1).mean(0)
    return scores,paras,doc,scores_raw


#Generation
def generate(model,tokenizer,text,max_new_tokens=500,**kwargs):
    with torch.no_grad():
        x=tokenizer(text,return_tensors='pt',padding='longest',max_length=tokenizer.model_max_length)
        y=model.generate(input_ids=x['input_ids'].cuda(),max_new_tokens=max_new_tokens,**kwargs)
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