import itertools
import util.einsum as einsum


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    known_contractions=torch.load('perm_inv_cache.pt')
except:
    known_contractions={}

#X_bijklm   [-1,0,1,0,2,-1] means that ik are co-invariant, j and l are invariant independently. b and m are batch-like dimensions and should be ignored.
#Currently does not support non-invariant dimensions beyond batch-like
#Would like to support n-complex numbers in the future

#Dependency graph is a list of edges [(0,1),(1,2)] means group 0 contraction requires group 1 contraction


#Generate unique coloring of N items with up to N colors
def coloring(N):
    if N==1:
        return [[0]]
    else:
        v=coloring(N-1)
        out=[]
        for x in v:
            out+=[x+[i] for i in range(max(x)+2)]
        
        return out

#Get an adj matrix encoding for invariant assignment graph
def adj(v,r):
    K=len(v)
    code=[[[] for o2 in range(K)] for o1 in range(K)]
    for o1 in range(K):
        for o2 in range(K):
            i=0
            for n in r:
                for j in range(i,i+n):
                    code[o1][o2]+=[int(v[o1][j]==v[o2][k]) for k in range(i,i+n)]
                
                i+=n
    
    
    for o1 in range(K):
        for o2 in range(K):
            code[o1][o2]=tuple(code[o1][o2])
    
    return code

#Generate unique contractions for invariant operations
#This in general is a graph isomorphism problem. Not sure if it has a good solution
#    Order is not expected to be large, so enumeration might not be too bad
#    At this point I prefer less bugs over speed
#r: a vector indicating the rank of each node (How many co-invariant dimensions for each invariant group)
#Algorithm is incrementally adding new invariant parameter groups
#    It's intuitive that's feasible
def contractions(r,orders=[1]):
    if (tuple(r),tuple(orders)) in known_contractions:
        return known_contractions[(tuple(r),tuple(orders))]
    
    if len(r)==1:
        n=r[0]
        order=sum(orders)
        v=coloring(n*order)
        comps=[[x[i*n:(i+1)*n] for i in range(order)] for x in v]
    else:
        prev=contractions(r[:-1],orders)
        order=sum(orders)
        v=coloring(r[-1]*order)
        n=r[-1]
        #allcomb and remove duplicates
        comps=[]
        for x in prev:
            for y in v:
                comps.append([x[i]+y[i*n:(i+1)*n] for i in range(order)])
    
    i=0
    perms=[]
    for order in orders:
        perm_i=list(itertools.permutations(range(i,i+order)))
        i+=order
        perms.append(perm_i)
    
    perms=list(itertools.product(*perms))
    perms=[list(itertools.chain(*x)) for x in perms]
    #print(perms)
    
    comps_by_code={}
    for c in comps:
        A=adj(c,r)
        k=[]
        for ind in perms:
            x=[[A[i][j] for j in ind] for i in ind]
            x=tuple([tuple(row) for row in x])
            k.append(x)
        
        k=frozenset(k)
        if not k in comps_by_code:
            comps_by_code[k]=c
    
    #print(r)
    #print('ncomps',len(comps))
    #print('%d %d terms'%(len(r),len(comps_by_code)))
    #for k in comps_by_code:
    #    print(k,comps_by_code[k])
    
    out=[comps_by_code[k] for k in comps_by_code]
    known_contractions[(tuple(r),tuple(orders))]=out
    try:
        torch.save(known_contractions,'perm_inv_cache.pt')
    except:
        pass
    return [comps_by_code[k] for k in comps_by_code]


#Find terms that can be broken into disjoint parts
#Only check dimensions specified in r
#If the use case is equivariance and the last entry in r corresponds to the RHS, pass in r[:-1]

#power set without full and empty
def powerset(iterable):
    xs = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(xs,n) for n in range(1,len(xs))))

#Find terms that can be broken into parts that share terms up to only a input size
def filter_breakable_equivariance(comps,r,orders):
    filtered=[]
    for c in comps:
        i=0
        breakable=False
        for n in orders:
            full=set(list(range(i,i+n)))
            layers_x=powerset(range(i,i+n))
            layers_y=[[x for x in full if not x in v] for v in layers_x]
            for l in range(len(layers_x)):
                j=0
                breakable_l=True
                for m in r:
                    idx_x=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_x[l]]+[c[-1][j:j+m]])))
                    idx_x2=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_x[l]])))
                    idx_y=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_y[l]]+[c[-1][j:j+m]])))
                    
                    #print(idx_x,idx_x2,idx_y,)
                    if not(len(idx_x.intersection(idx_y))<=m or (len(layers_x[l])>1 and len(idx_x2.intersection(idx_y))<=m)):
                        breakable_l=False
                        break
                    
                    j+=m
                
                if breakable_l:
                    breakable=True
                    #print(c)
                    break
            
            i+=n
            if breakable:
                break
        
        if not breakable:
            filtered.append(c)
    
    return filtered

#c=[[0,0],[0,1],[1,0],[1,1]]
#filter_breakable_equivariance([c],[1,1],[3])

#Find terms that can be broken into parts that share a constant
def filter_breakable_equivariance_v0(comps,r,orders):
    filtered=[]
    for c in comps:
        i=0
        breakable=False
        for n in orders:
            full=set(list(range(i,i+n)))
            layers_x=powerset(range(i,i+n))
            layers_y=[[x for x in full if not x in v] for v in layers_x]
            for l in range(len(layers_x)):
                j=0
                breakable_l=True
                for m in r:
                    idx_x=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_x[l]]+[c[-1][j:j+m]])))
                    idx_y=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_y[l]])))
                    bridge=idx_x.intersection(idx_y)
                    if len(bridge)>0:
                        breakable_l=False
                        break
                    
                    j+=m
                
                if breakable_l:
                    breakable=True
                    #print(c)
                    break
            
            i+=n
            if breakable:
                break
        
        if not breakable:
            filtered.append(c)
    
    return filtered



#Find terms that can be broken into parts that share a constant
def filter_breakable(comps,r,orders):
    filtered=[]
    for c in comps:
        i=0
        breakable=False
        for n in orders:
            full=set(list(range(i,i+n)))
            layers_x=powerset(range(i,i+n))
            layers_y=[[x for x in full if not x in v] for v in layers_x]
            for l in range(len(layers_x)):
                j=0
                breakable_l=True
                for m in r:
                    idx_x=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_x[l]])))
                    idx_y=set(list(itertools.chain.from_iterable([c[x][j:j+m] for x in layers_y[l]])))
                    if len(idx_x.intersection(idx_y))>0:
                        breakable_l=False
                        break
                    
                    j+=m
                
                if breakable_l:
                    breakable=True
                    #print(c)
                    break
            
            i+=n
            if breakable:
                break
        
        if not breakable:
            filtered.append(c)
    
    return filtered


#Serve graphs, check compatibility with graph
#For any two layers, super node rep different => sub node rep different
#Order is tricky, need to know whether two terms are the same thing or not. Sometimes orders mean X and Y where things are supposed to contract, more often times it's X->X where you can still keep
def filter_dependency(comps,r,deps):
    filtered=[]
    for c in comps:
        passed=True
        for e in deps:
            s0=sum(r[:e[0]])
            s1=s0+r[e[0]]
            t0=sum(r[:e[1]])
            t1=t0+r[e[1]]
            
            for o1 in range(len(c)):
                for o2 in range(o1+1,len(c)):
                    if not c[o1][s0:s1]==c[o2][s0:s1]:
                        if len(set(c[o1][t0:t1]).intersection(set(c[o2][t0:t1])))>0:
                            passed=False
                            break
                
                if not passed:
                    break
            
            if not passed:
                break
        
        if passed:
            filtered.append(c)
        else:
            pass
            #print(c)
    
    return filtered



'''
c=contractions([2,1],[2,1])
c2=filter_breakable(c,[2,1],[2,1])
c3=filter_dependency(c2,[2,1],[[0,1]])
'''

#Serve graph using above utilities

#X_bijklm   [-1,0,1,0,2,-1] means that ik are co-invariant, j and l are invariant independently. b and m are batch-like dimensions and should be ignored.
#Currently does not support non-invariant dimensions beyond batch-like
#Would like to support n-complex numbers in the future

#Dependency graph is a list of edges [(0,1),(1,2)] means group 0 contraction requires group 1 contraction
#form is a string starting with X_ and then a list of subscripts. Capital letters are non-invariant dimensions, lower case are invariant. Lower case with the same character are co-invariant. Deps are based on the lower case.
def invariance_terms(form='X_BanbnaH',order=1,deps=('a->b','b->c')):
    s=form.split('_')[1]
    ch=sorted(list(set([ch for ch in s if ch.islower()])))
    r=[s.count(c) for c in ch]
    deps_=[]
    for x in deps:
        v=x.split('-')
        deps_.append((ch.index(v[0][0]),ch.index(v[1][1])))
    
    #print(deps_)
    
    comps=[]
    for o in range(1,order+1):
        comps_i=contractions(r,[o])
        comps_i=filter_breakable(comps_i,r,[o])
        comps_i=filter_dependency(comps_i,r,deps_)
        print('order %d, %d terms'%(o,len(comps_i)))
        comps+=comps_i
    
    #translate comps into einsum equations
    lower='abcdefghijklmnopqrstuvwxyz'
    upper='ZYXWVUTSRQPONMLKJIHGFEDCBA'
    
    eqs=[]
    for c in comps:
        #Augment comp representation with invariant node name
        # node - order - id  => node - color mapping
        node_map={}
        c2=[[] for x in c]
        for i in range(len(r)):
            for j in range(len(c)):
                for k in range(sum(r[:i]),sum(r[:i+1])):
                    node_map[(i,j,k-sum(r[:i]))]=(i,c[j][k])
        
        id_map=sorted(list(set([node_map[x] for x in node_map])))
        id_map2=[c for c in s if c.isupper()]
        #Compose einsum terms
        terms=[]
        for i in range(len(c)):
            t=[]
            for j,c in enumerate(s):
                if c.islower():
                    id=s[:j+1].count(c)-1
                    node=ch.index(c)
                    id=id_map.index(node_map[(node,i,id)])
                    t.append(lower[id])
                else:
                    id=id_map2.index(c)
                    t.append(upper[id])
            
            terms.append(''.join(t))
        
        lhs=','.join(terms)
        rhs=''.join([upper[id_map2.index(c)] for c in s if c.isupper()])
        eqs.append(lhs+'->'+rhs)
    
    return eqs


def equivariance_terms(form='X_BanbnaH',order=1,deps=('a->b','b->c')):
    s=form.split('_')[1]
    ch=sorted(list(set([ch for ch in s if ch.islower()])))
    r=[s.count(c) for c in ch]
    deps_=[]
    for x in deps:
        v=x.split('-')
        deps_.append((ch.index(v[0][0]),ch.index(v[1][1])))
    
    #print(deps_)
    
    comps=[]
    for o in range(1,order+1):
        comps_i=contractions(r,[o,1])
        comps_i=filter_breakable_equivariance(comps_i,r,[o])
        comps_i=filter_dependency(comps_i,r,deps_)
        print('order %d, %d terms'%(o,len(comps_i)))
        comps+=comps_i
    
    #translate comps into einsum equations
    lower='abcdefghijklmnopqrstuvwxyz'
    upper='ZYXWVUTSRQPONMLKJIHGFEDCBA'
    
    eqs=[]
    for c in comps:
        #Augment comp representation with invariant node name
        # node - order - id  => node - color mapping
        node_map={}
        c2=[[] for x in c]
        for i in range(len(r)):
            for j in range(len(c)):
                for k in range(sum(r[:i]),sum(r[:i+1])):
                    node_map[(i,j,k-sum(r[:i]))]=(i,c[j][k])
        
        id_map=sorted(list(set([node_map[x] for x in node_map])))
        id_map2=[c for c in s if c.isupper()]
        #Compose einsum terms
        terms=[]
        for i in range(len(c)):
            t=[]
            for j,c in enumerate(s):
                if c.islower():
                    id=s[:j+1].count(c)-1
                    node=ch.index(c)
                    id=id_map.index(node_map[(node,i,id)])
                    t.append(lower[id])
                else:
                    id=id_map2.index(c)
                    t.append(upper[id])
            
            terms.append(''.join(t))
        
        lhs=','.join(terms[:-1])
        rhs=terms[-1]
        eqs.append(lhs+'->'+rhs)
    
    return eqs

'''
for t in equivariance_terms(form='X_aaa',order=5,deps=[]):
    pass
    #print(t)
'''

#Einsum-based pooling network
class einpool(nn.Module):
    def __init__(self,form='X_Bab',order=5,deps=[],eqs=None,equivariant=False,M=False):
        super().__init__()
        if eqs is None:
            if equivariant:
                self.eqs=equivariance_terms(form=form,order=order,deps=deps)
            else:
                self.eqs=invariance_terms(form=form,order=order,deps=deps)
        else:
            self.eqs=eqs
        
        self.M=M
        print('%d terms'%len(self.eqs))
        
    
    def optimize(self,sz,flops=1e12,mem=1e9):
        flops_=[]
        mem_=[]
        eq_filtered=[]
        for eq in self.eqs:
            flops_i,mem_i,_=einsum.einsum_path(eq,sz)
            if flops_i<=flops and mem_i<=mem:
                eq_filtered.append(eq)
                flops_.append(flops_i)
                mem_.append(mem_i)
        
        print('Compute %.3f TF Mem %.3f GB'%(sum(flops_)/1e12,sum(mem_)/1e9*4))
        self.eqs=eq_filtered
        return
    
    
    def forward(self,X,mask=None,to_cpu=False):
        h=[]
        n=[]
        if mask is None:
            mask=X.data.clone().fill_(1)
        else:
            mask=mask.expand_as(X)
            X=X*mask
        
        for s in self.eqs:
            #print(s)
            h_i=einsum.einsum(s,X,M=self.M)
            if to_cpu:
                h_i=h_i.cpu()
            
            n_i=einsum.einsum(s,mask,M=self.M)
            if to_cpu:
                n_i=n_i.cpu()
            
            h.append(h_i)
            n.append(n_i)
        
        h=torch.stack(h,dim=-1)
        n=torch.stack(n,dim=-1)
        return h/n.clamp(min=1e-12)

#Einsum-based pooling network
class einpool_multihead(nn.Module):
    def __init__(self,form='X_Bab',order=5,deps=[],eqs=None,equivariant=False,M=False,checkpoint=False,normalize=True):
        super().__init__()
        if eqs is None:
            if equivariant:
                self.eqs=equivariance_terms(form=form,order=order,deps=deps)
            else:
                self.eqs=invariance_terms(form=form,order=order,deps=deps)
        else:
            self.eqs=eqs
        
        self.M=M
        self.checkpoint=checkpoint
        self.normalize=normalize
        print('%d terms'%len(self.eqs))
        
    
    def nheads(self):
        n=sum([len(s.split('-')[0].split(',')) for s in self.eqs])
        return n
    
    def forward(self,X,mask=None,to_cpu=False,verbose=False):
        assert X.shape[-1]==self.nheads()
        
        h=[]
        n=[]
        if mask is None:
            mask=X.data.clone().fill_(1)
        else:
            mask=mask.expand_as(X)
            X=X*mask
        
        X=X.split(1,dim=-1)
        mask=mask.split(1,dim=-1)
        
        i=0
        for j,s in enumerate(self.eqs):
            k=len(s.split('-')[0].split(','))
            h_i=einsum.einsum(s,*[x.squeeze(-1) for x in X[i:i+k]],M=self.M,checkpoint=self.checkpoint,verbose=verbose)
            if to_cpu:
                h_i=h_i.cpu()
            
            n_i=einsum.einsum(s,*[x.squeeze(-1) for x in mask[i:i+k]],M=self.M,checkpoint=self.checkpoint,verbose=verbose)
            if to_cpu:
                n_i=n_i.cpu()
            
            h.append(h_i)
            n.append(n_i)
            #if verbose and j==36:
            #    print(s,j,i,i+k)
            #    print(s,j,i,X[i].abs().sum(),i+k,X[i+1].abs().sum())
            i+=k
        
        h=torch.stack(h,dim=-1)
        n=torch.stack(n,dim=-1)
        if self.normalize:
            return h/n.clamp(min=1e-12)
        else:
            return h


class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        return;
    
    def forward(self,x):
        h=x;
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.gelu(h);
        
        h=self.layers[-1](h);
        return h

class einnet(nn.Module):
    def __init__(self,nhinput,nh0,nh,nstacks=2,nlayers=2,form='X_Bab',order=5,deps=[]):
        super().__init__()
        self.pool=einpool(form='X_Bab',order=order,deps=deps,equivariant=True)
        n=len(self.pool.eqs)
        
        self.t=nn.ModuleList()
        self.t0=MLP(nhinput,nh,nh0,nlayers)
        
        for i in range(nstacks):
            self.t.append(MLP(n*nh0,nh,nh0,nlayers))
    
    
    def forward(self,X,mask=None):
        h=self.t0(X)
        if mask is None:
            mask=X.data.mean(dim=-1,keepdim=True)*0+1
        
        for layer in self.t:
            h=h*mask
            skip=h
            h=pool(torch.sin(h),mask).view(*h.shape[:-1],-1)
            h=self.t1(h)
            h=h*mask
            h+=skip
        
        return h


'''

eqs=invariance_terms(form='X_Bab',order=5,deps=[])
for eq in eqs:
    print(eq)

print('%d total'%len(eqs))
'''
'''
net=einpool('X_Bab',order=3,equivariance=True)
x=torch.rand(3,5,7)
y=net(x)
print(y.shape)
'''
'''
def adj_code(v,r):
    code=adj(v,r)
    code_rows=[frozenset(row) for row in code]
    code_rows=frozenset(code_rows)
    
    code_T=[[code[i][j] for i in range(len(code))] for j in range(len(code[0]))]
    code_cols=[frozenset(col) for col in code_T]
    code_cols=frozenset(code_cols)
    return (code_rows,code_cols)
'''
