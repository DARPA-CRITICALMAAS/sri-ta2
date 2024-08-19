import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
from transformers import pipeline


def device_profile():
    n=torch.cuda.device_count()
    return [torch.cuda.get_device_properties(i).total_memory for i in range(n)]

def load_lm(model_name,dtype=None,max_mem=0.8,**kwargs2):
    mem=device_profile()
    kwargs = dict(
        device_map="auto",
        max_memory={i:int(max_mem*m) for i,m in enumerate(mem)},
        offload_folder='offload/',
        cache_dir='cache/',
    )
    if not dtype is None:
        kwargs['torch_dtype']=dtype
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,**kwargs2)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True, **kwargs,**kwargs2)
    #model=torch.compile(model)
    model.eval();
    return model,tokenizer



#High efficiency token-level perplexity
#Leveraging cache -- might not be compatible with everyone?
#    full: returns one probability for each token in tokens1. Default false, returns average probability
#Tokens0: BOS yes, EOS no.
#Tokens1: BOS no, EOS no.
#Tokens1 should try to be iso-length to avoid issues with pad token and masks
def perplexity2(model,tokenizer,tokens0,tokens1,bsz=5,full=False,verbose=True):
    pad=tokenizer.pad_token_id
    L=max([len(x) for x in tokens1])
    scores=[]
    with torch.no_grad():
        t0=time.time()
        for i,prompt in enumerate(tokens0):
            last_prompt_token=prompt[-1]
            cache=model(torch.LongTensor([tokens_prompt[:-1]]).cuda(),use_cache=True)['past_key_values']
            
            scores_i=[]
            for j in range(0,len(tokens1),bsz):
                if verbose:
                    print('%d/%d, %d/%d, time %.2f'%(i,len(tokens0),j,len(tokens1),time.time()-t0),end='\r')
                
                r=min(j+bsz,len(input_ids))
                
                input_ids=[[last_prompt_token]+x+[pad]*(L-len(x)) for x in tokens1[i:r]]
                target_ids=[x+[pad]*(L+1-len(x)) for x in tokens1[i:r]]
                input_ids=torch.LongTensor(input_ids).cuda()
                target_ids=torch.LongTensor(target_ids).cuda()
                
                #Make bigger cache to match tokens1
                cache_i=[[z.repeat(r-j,1,1,1) for z in x] for x in cache]
                #Get logits for each word
                logp_i=model(input_ids,past_key_values=cache_i)['logits']
                logp_i=F.log_softmax(logp_i,dim=-1)
                logp_i=logp_i.gather(-1,target_ids.unsqueeze(-1)).squeeze(-1)
                s=(logp_i*target_ids.ne(pad))
                
                if not full:
                    #Normalize scores 
                    n=target_ids.ne(pad).sum(dim=-1)
                    s=s.sum(dim=-1)/(n+1e-20)
                
                scores_i.append(s)
            
            scores_i=torch.cat(scores_i,dim=0) #len(tokens1) x ?
            scores.append(scores_i.cpu())
        
        scores=torch.stack(scores,dim=1) #len(tokens1) x len(tokens2) x ?
    
    return scores




#MRDS utils
def mrds_clean_json(x):
    if isinstance(x,list):
        return [clean_json(v) for v in x]
    elif isinstance(x,dict):
        return {k:clean_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
    else:
        return x


def mrds_get_name(d):
    if 'properties' in d:
        d=d['properties']
    
    if not 'name' in d:
        print('cannot find name',d)
        return None
    
    name=d['name']
    if isinstance(name,str):
        return name
    elif isinstance(name,list):
        names=[x['name'] for x in name if x['status']=='Current']
        assert len(names)==1
        return names[0]
    elif isinstance(name,dict) and 'name' in name and isinstance(name['name'],str):
        return name['name']
    else:
        print('cannot find name',name)
        return None


def mrds_get_lat(x):
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


def mrds_get_link(item):
    if not 'id' in item and 'deposits' in item:
        i=item['deposits']['dep_id']
        i='https://mrdata.usgs.gov/mrds/record/%s'%i
    else:
        i=item['id']
    
    return i

def mrds_get_id(item):
    if not 'id' in item and 'deposits' in item:
        i=item['deposits']['dep_id']
        i=int(i)
    else:
        i=int(item['id'].split('/')[-1])
    
    return i


def mrds_get_commodity(d):
    if 'properties' in d:
        d=d['properties']
    
    if not 'commodity' in d:
        #print('cannot find commodity',d)
        return []
    
    commod=d['commodity']
    #print(commod)
    if not isinstance(commod,list):
        commod=[commod]
    
    commod_=[]
    for x in commod:
        if isinstance(x,str):
            commod_.append(x)
        elif 'commod' in x:
            commod_.append(x['commod'])
        elif 'name' in x:
            commod_.append(x['name'])
        elif 'value' in x:
            commod_.append(x['value'])
        else:
            a=0/0
    
    commod=commod_
    return commod

#Onlys seen in mrds
def mrds_get_rank(d):
    if "properties" in d:
        d=d['properties']
    
    if 'grade' in d:
        return d['grade']
    else:
        return None

#seen in mrds and ardf
def mrds_get_type(d):
    if "properties" in d:
        d=d['properties']
    
    if 'rec_tp' in d:
        return d['rec_tp']
    elif 'class' in d:
        return d['class']
    elif 'type' in d and not d['type']=='Feature':
        return d['type']
    else:
        return None



#Compute edit distance between two strings
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


