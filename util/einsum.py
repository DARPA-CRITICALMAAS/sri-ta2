#Einsum pooling and simplification
import math
import numpy
import torch
import itertools


def parse_line(line):
    return [x for x in line.split('\n')[-1].split(' ') if not x=='']

def parse_einpath(path):
    p=path[0][1:]
    rows=path[1].split('\n')
    flops=float(parse_line(rows[4])[-1])
    memory=float(parse_line(rows[6])[-2])
    cmd=[parse_line(row)[1] for row in rows[10:]]
    return flops,memory,list(zip(cmd,p))

#Optimize an einsum operation
#Input einsum string, operand size
#Output optimized einsum steps, flops, memory size
def einopt(s,sz,prepro=True):
    s_=s.split('-')
    lhs,rhs=s_[0],s_[1][1:]
    lhs=lhs.split(',')
    
    #Map letters to size
    sz_c={}
    for term in lhs:
        for i,ch in enumerate(term):
            sz_c[ch]=sz[i]
    
    ops=[]
    #First eliminate singletons that does not appear in other terms
    #Because numpy einsum_path doesn't do that
    alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lhs_simplified=[]
    for term in lhs:
        input=''.join([alphabet[i] for i,ch in enumerate(term)])
        out=''.join([alphabet[i] for i,ch in enumerate(term) if s.count(ch)>1])
        out2=''.join([ch for i,ch in enumerate(term) if s.count(ch)>1])
        assert len(out)>0 #Fix your einsum strings. Add batch or sth
        ops.append((input+'->'+out,(0,)))
        lhs_simplified.append(out2)
    
    #Optimize the remaining with numpy.einsum_path
    s_simplified=','.join(lhs_simplified)+'->'+rhs
    arr=[numpy.ones([sz_c[ch] for ch in term]) for term in lhs_simplified]
    path=numpy.einsum_path(s_simplified,*arr,optimize='optimal')
    flops,memory,cmd=parse_einpath(path)
    if prepro:
        return ops+cmd
    else:
        return cmd

#Approximate einsum
import time
def do_einsum(s,*args,approximate=False,order=4,N=10000):
    lhs=s.split('-')[0]
    terms=lhs.split(',')
    rhs=s.split('>')[1]
    if approximate and len(terms)>=order:
        t0=time.time()
        device=args[0].device
        #Use a ray-tracing like method to do large einsums
        #Figure out the literals
        #Figure out procedural dimensions (related to output) and free dimensions
        ch=[c for c in rhs]+list(set([c for c in lhs if c.isalpha() and not c in rhs]))
        
        #Figure out range for each literal
        sz={}
        for i,t in enumerate(terms):
            for j,sz_i in enumerate(args[i].shape):
                sz[t[j]]=sz_i
        
        sz=[sz[c] for c in ch]
        
        #Conduct sampling
        sz_N=sz[:len(rhs)]+[N]
        idx=[]
        #procedural dimensions
        for i,c in enumerate(rhs):
            idx_i=torch.arange(0,sz[i],device=device).long().view(-1,*([1]*(len(rhs)-i))).expand(*sz_N)
            idx.append(idx_i)
        
        #random dimensions
        for i,c in enumerate(ch[len(rhs):]):
            idx_i=torch.LongTensor(*sz_N).to(device).random_(sz[i+len(rhs)])
            idx.append(idx_i)
        
        #Compute
        data=None
        scale=1
        for i,t in enumerate(terms):
            idx_i=[idx[ch.index(c)] for c in t]
            
            data_i=args[i].__getitem__(idx_i)
            if i==0:
                data=data_i
            else:
                data*=data_i
        
        #Aggregate
        data=data.mean(dim=-1)
        scale=numpy.prod(sz[len(rhs):])
        #print('mine %.4f'%(time.time()-t0))
        #v=torch.einsum(s,*args)
        #print('orig %.4f'%(time.time()-t0))
        return data*scale
    else:
        return torch.einsum(s,*args)

#x=torch.rand(3,5)
#a=do_einsum('ab,cb->ac',x,x,approximate=True)


#Concerns: 
#    Repetitive work when multiple terms are fired together
#    Eats a lot of memory to store intermediate variables

#Wrapper for einsum path optimization
known_paths={}
'''
def scale_down(sz):
    return tuple([min(max(round(x**0.5),2),100) for x in sz])

import torch.utils.checkpoint
def einsum(s,*args,approximate=False):
    #Prepare RHS for equivariant operations
    # whose dimensions collapse
    lhs=s.split('>')[0]
    rhs=s.split('>')[-1]
    rhs2=''.join([c for c in rhs if c in lhs])
    s=lhs+'>'+rhs2
    s2=rhs+','+rhs2+'->'+rhs
    
    nterms=len(s.split(','))
    if len(args)!=nterms and len(args)==1:
        args=[args[0] for i in range(nterms)]
    
    
    if approximate:
        v=do_einsum(s,*args,approximate=True)
    else:
        #Compute einsum optimization
        
        sz=tuple(scale_down(args[0].shape))
        if not (s,sz) in known_paths:
            path=einopt(s,sz)
            known_paths[(s,sz)]=path
        else:
            path=known_paths[(s,sz)]
        
        #Perform optimized einsum
        #v=torch.utils.checkpoint.checkpoint(einsum_with_path,path,*args)
        v=einsum_with_path(path,*args)
    
    #Equivariant collapse handling
    if not rhs==rhs2:
        v=torch.einsum(s2,args[0].data.clone().fill_(1),v)
    return v

def einsum_with_path(path,*args):
    X=[v for v in args]
    for (s,ind) in path:
        X_i=[X[i] for i in ind[::-1]]
        for i in sorted(list(ind),reverse=True):
            X.pop(i)
        
        y=torch.einsum(s,*X_i)
        X.append(y);
    
    
    return y

def cost_path(s,sz):
    lhs=s.split('-')[0]
    rhs=s.split('>')[1]
    terms=lhs.split(',')
    rhs2=''.join([c for c in rhs if c in lhs])
    s=lhs+'->'+rhs2
    
    
    #print(tuple(scale_down(sz)))
    path=einopt(s,tuple(scale_down(sz)),prepro=False)
    
    #Map letters to size
    sz_c={}
    for term in terms:
        for i,ch in enumerate(term):
            sz_c[ch]=sz[i]
    
    flops_=[]
    mem_=[]
    print(path,tuple(scale_down(sz)))
    for s,ind in path:
        lhs=s.split('-')[0]
        rhs=s.split('>')[1]
        terms=lhs.split(',')
        if len(terms)==1:
            flops=numpy.prod([sz_c[c] for c in set(terms[0])])
            mem=numpy.prod([sz_c[c] for c in set(terms[0])])
            flops_.append(flops)
            mem_.append(mem)
        else:
            x0=set(terms[0])
            x1=set(terms[1])
            xrhs=set(rhs)
            
            x=x0.union(x1)
            x_gone=x0.intersection(x1).difference(xrhs)
            
            flops=numpy.prod([sz_c[c] for c in x])
            mem=numpy.prod([sz_c[c] for c in x if not c in x_gone])
            flops_.append(flops)
            mem_.append(mem)
    
    return sum(flops_),max(mem_)

'''
'''
'''
def powerset(iterable):
    xs = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(xs,n) for n in range(1,len(xs))))

einsum_path_cache={}
def einsum_path(s,sz,verbose=False):
    lhs=s.split('-')[0]
    rhs=s.split('>')[1]
    assert len(set(rhs))==len(rhs) #Does not deal with unsqueezing/diagonal filling
    terms=lhs.split(',')
    #Check a special case where all inputs are same size
    if not (isinstance(sz[0],list) or isinstance(sz[0],tuple) or isinstance(sz[0],torch.Size)):
        sz=[sz]*len(terms)
    
    #Check whether results are cached
    if (s,tuple([tuple(s) for s in sz])) in einsum_path_cache:
        return einsum_path_cache[(s,tuple([tuple(s) for s in sz]))]
    
    #Map letters to size
    sz_c={}
    for i,term in enumerate(terms):
        for j,ch in enumerate(term):
            sz_c[ch]=sz[i][j]
    
    
    if len(terms)==1:
        #Only one term, calculate mem & flops
        input_flops=numpy.prod([float(sz_c[c]) for c in set(terms[0])])
        input_size=numpy.prod([float(sz_c[c]) for c in terms[0]])
        output_size=numpy.prod([float(sz_c[c]) for c in rhs])
        mem=max(input_size,output_size)
        flops=max(input_flops,output_size)
        path=[(s,(0,))]
        best_solution=(flops,mem,path)
    elif len(terms)==2:
        #Two terms, first run simplification, and then call einsum
        ch0=set(terms[0])
        ch1=set(terms[1])
        o0=''.join([c for c in sz_c if c in ch0 and (c in ch1 or c in rhs)])
        o1=''.join([c for c in sz_c if c in ch1 and (c in ch0 or c in rhs)])
        s0=terms[0]+'->'+o0
        s1=terms[1]+'->'+o1
        s2=o1+','+o0+'->'+rhs
        
        flops_0,mem_0,path_0=einsum_path(s0,[sz[0]])
        flops_1,mem_1,path_1=einsum_path(s1,[sz[1]])
        path_1=[(path_1[0][0],(1,))]
        flops_2=numpy.prod([float(sz_c[c]) for c in sz_c if c in o0 or c in o1])
        mem_2=numpy.prod([float(sz_c[c]) for c in rhs])
        
        path=path_0+path_1+[(s2,(-1,-1))]
        flops=flops_0+flops_1+flops_2
        mem=max([mem_0,mem_1,mem_2])
        best_solution=(flops,mem,path)
    else:
        best_cost=1e100 #something impossibly big
        best_solution=None
        #Make the best cut
        x=powerset(range(len(terms)))
        y=[[i for i in range(len(terms)) if not i in v] for v in x]
        for i in range(len(x)):
            #Compose sub ops
            ch0=set(''.join([terms[j] for j in x[i]]))
            ch1=set(''.join([terms[j] for j in y[i]]))
            o0=''.join([c for c in sz_c if c in ch0 and (c in ch1 or c in rhs)])
            o1=''.join([c for c in sz_c if c in ch1 and (c in ch0 or c in rhs)])
            
            s0=','.join([terms[j] for j in x[i]])+'->'+o0
            sz0=[sz[j] for j in x[i]]
            s1=','.join([terms[j] for j in y[i]])+'->'+o1
            sz1=[sz[j] for j in y[i]]
            s2=o1+','+o0+'->'+rhs
            sz2=[[sz_c[c] for c in o1],[sz_c[c] for c in o0]]
            
            flops_0,mem_0,path_0=einsum_path(s0,sz0)
            flops_1,mem_1,path_1=einsum_path(s1,sz1)
            flops_2,mem_2,_=einsum_path(s2,sz2)
            
            flops_i=flops_0+flops_1+flops_2
            mem_i=max([mem_0,mem_1,mem_2])
            cost_i=0.5*math.log(flops_i)+math.log(mem_i)
            
            path_i=[]
            #Path variable remap
            for si,vi in path_0:
                vi_=[]
                for v in vi:
                    if v>=0:
                        vi_.append(x[i][v])
                    else:
                        vi_.append(v)
                
                path_i.append((si,tuple(vi_)))
            
            for si,vi in path_1:
                vi_=[]
                for v in vi:
                    if v>=0:
                        vi_.append(y[i][v])
                    else:
                        vi_.append(v)
                
                path_i.append((si,tuple(vi_)))
            
            path_i.append((s2,(-1,-1)))
            
            if cost_i<best_cost:
                best_cost=cost_i
                best_solution=(flops_i,mem_i,path_i)
    
    einsum_path_cache[(s,tuple([tuple(s) for s in sz]))]=best_solution
    if verbose:
        print(s,sz,best_solution)
    return best_solution


def einsum_expand(s,x,target_sz):
    #We couldn't allow each character to appear more than twice
    #since pytorch doesn't have diag_embed for 3+ diags
    lhs=s.split('-')[0]
    rhs=s.split('>')[1]
    
    s1=[c for c in lhs if rhs.count(c)==1]
    s2=[c for c in lhs if rhs.count(c)==2]
    s01=[c for c in set(rhs) if not c in lhs and rhs.count(c)==1]
    s02=[c for c in set(rhs) if not c in lhs and rhs.count(c)==2]
    
    if len(s01)>0 or len(s02)>0:
        sz_c={c:target_sz[i] for i,c in enumerate(rhs)}
    
    #process diagonal dimensions
    #pull all s2 to end
    h=x
    h=h.permute(*[lhs.index(c) for c in s1],*[lhs.index(c) for c in s2])
    for c in s2:
        h=torch.diag_embed(h,dim1=0,dim2=1)
    
    #Unsqueeze new dimensions
    for c in s01:
        h=h.unsqueeze(-1).expand(*h.shape,sz_c[c])
    
    #Unsqueeze new dimensions and diag_embed
    for c in s02:
        h=h.unsqueeze(-1).expand(*h.shape,sz_c[c])
        h=torch.diag_embed(h,dim1=0,dim2=1)
    
    #Order dimensions as in rhs
    ind=[]
    for i in range(len(rhs)):
        c=rhs[i]
        if c in s01:
            ind.append(len(s02)*2+len(s2)*2+len(s1)+s01.index(c))
        elif c in s1:
            ind.append(len(s02)*2+len(s2)*2+s1.index(c))
        elif c in s02 and c in rhs[:i]:
            ind.append(2*s02.index(c)+1)
        elif c in s02 and not c in rhs[:i]:
            ind.append(2*s02.index(c))
        elif c in s2 and c in rhs[:i]:
            ind.append(len(s02)*2+2*s2.index(c)+1)
        elif c in s2 and not c in rhs[:i]:
            ind.append(len(s02)*2+2*s2.index(c))
        else:
            a=0/0
    
    h=h.permute(ind)
    return h



def einsum_with_path(path,*args,M=False,verbose=False):
    if verbose:
        print(path,len(args))
    
    X=[v for v in args]
    for (s,ind) in path:
        X_i=[]
        for i in ind:
            if i>=0:
                X_i.append(X[i])
            else:
                X_i.append(X.pop(i))
        
        #check num. terms and synthesize new Meq
        if M is True:
            lhs=s.split('-')[0]
            rhs=s.split('>')[1]
            terms=lhs.split(',')
            if len(terms)==1:
                s=terms[0]+'AB'+'->'+rhs+'AB'
            elif len(terms)==2:
                s=terms[0]+'AB,'+terms[1]+'BC->'+rhs+'AC'
            else:
                a=0/0
        
        y=torch.einsum(s,*X_i)
        X.append(y);
    
    return y

import torch.utils.checkpoint
def einsum(s,*args,M=False,checkpoint=False,verbose=False):
    if verbose:
        print(s)
    
    #Analyze the equation to deal with special cases
    lhs=s.split('-')[0]
    rhs=s.split('>')[-1]
    rhs2=''.join([c for c in sorted(list(set(rhs))) if c in lhs])
    s=lhs+'->'+rhs2
    
    #When all variables are the same, allows only one input
    nterms=len(s.split(','))
    if verbose:
        print(len(args),nterms)
    if len(args)!=nterms and len(args)==1:
        args=[args[0] for i in range(nterms)]
    
    #Compute einsum path optimization
    #print(s)
    if M is True:
        flops,mem,path=einsum_path(s,[tuple(a.shape[:-2]) for a in args],verbose=verbose)
    else:
        flops,mem,path=einsum_path(s,[tuple(a.shape) for a in args],verbose=verbose)
    
    #Perform optimized einsum
    if checkpoint:
        v=torch.utils.checkpoint.checkpoint(einsum_with_path,path,*args,use_reentrant=False,verbose=verbose)
    else:
        v=einsum_with_path(path,*args,M=M,verbose=verbose)
    
    #Equivariant collapse handling
    if not rhs==rhs2:
        if M is True:
            rhs2=rhs2+'AB'
            rhs=rhs+'AB'
        
        s2=rhs2+'->'+rhs
        v=einsum_expand(s2,v,tuple(args[0].shape))
    
    return v

#Test cases for einsum

'''
#Test cases for expansion
x=torch.rand(10,5,5,9)
print(einsum('Zaac->Zaad',x).shape)
print(einsum('Zaac->Zbbc',x).shape)
print(einsum('Zaac->Zcca',x).shape)
'''

'''
flops,mem,path=einsum_path('Zac,Zad,Zbc,Zbd->Z',[10,5,5])
print(flops)
print(mem)
print(path)
'''

'''
path=einopt('Zab,Zcd,Zac->Zbd',[5,5,5])
x=torch.rand(3,5,5)
einsum_with_path(path,x,x,x)

einsum('Zab,Zcd,Zac->Zbd',x,x,x)
'''