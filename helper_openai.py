
import backoff 
import openai
import tiktoken
import math

@backoff.on_exception(backoff.expo, openai.RateLimitError)


class new:
    def __init__(self,params):
        if not params.azure_api_endpoint=='':
            if params.azure_lm=='':
                params.azure_lm=params.lm
                self.tokenizer=tiktoken.encoding_for_model(params.lm)
            elif params.lm=='':
                self.tokenizer=tiktoken.encoding_for_model(params.azure_lm)
            else:
                self.tokenizer=tiktoken.encoding_for_model(params.lm)
            
            self.lm=params.azure_lm
            self.client = openai.AzureOpenAI(api_key=params.openai_api_key,api_version=params.azure_api_version,azure_endpoint=params.azure_api_endpoint)
        else:
            self.client = openai.OpenAI(api_key=params.openai_api_key)
            self.lm=params.lm
            self.tokenizer=tiktoken.encoding_for_model(params.lm)
        
        self.token_count=0
        self.params=params #lm=model openai_key=openai_api_key
    
    def completions_with_backoff(self,**kwargs):
        return self.client.chat.completions.create(**kwargs)
    
    def qa(self,system,user):
        msg=[]
        msg+=[{"role": "system", "content": system}]
        msg+=[{"role": "user", "content": user}]
        response=self.completions_with_backoff(model=self.lm, messages=msg).choices[0]
        self.token_count+=len(self.tokenizer.encode(system+user))
        return response.message.content
    
    #options are a001~a{noptions}. 3 digits max
    #returns a probability distribution
    def multiple_choice(self,system,user,noptions,max_retries=5):
        assert noptions<1000
        
        msg=[]
        msg+=[{"role": "system", "content": system}]
        msg+=[{"role": "user", "content": user}]
        
        for i in range(max_retries):
            self.token_count+=len(self.tokenizer.encode(system+user))
            response=self.completions_with_backoff(model=self.lm, messages=msg,logprobs=True,top_logprobs=5).choices[0]
            
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
                    assert pred_x[0]<noptions
                    assert pred_x[0]>=0
                    pred.append(pred_x)
                except:
                    pass
            
            if len(pred)==0:
                continue
            
            #Compose probability distribution
            logp=[math.log(1e-20) for i in range(noptions)] # add 1e-20 so logprobs show something
            for x in pred:
                logp[x[0]]=x[1]
            
            return logp,response.message.content
        
        return [math.log(1e-20) for i in range(noptions)],response.message.content
    
    def chunk(self,text,L=80000):
        tokens=self.tokenizer.encode(text)
        tokens=[tokens[i:i+L] for i in range(0,len(tokens),L)] 
        chunks=[self.tokenizer.decode(t) for t in tokens]
        return chunks
    
