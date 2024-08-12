import torch
import math

#metrics
def eval_cls(scores,gt):
    sgt=scores.gather(-1,gt.view(-1,1))
    r=scores.ge(sgt).long().sum(dim=-1)
    mrr=(1/r.float()).mean()
    top1=r.eq(1).float().mean()
    top5=r.le(5).float().mean()
    return float(top1),float(top5),float(mrr)

def eval_ret(scores,gt,K=50):
    #Compute P/R for top K=50 most frequent deposit types
    #For reliable estimates (>100 occurence out of 30000)
    prior=[0 for i in range(max(gt.view(-1).tolist())+1)]
    for x in gt.view(-1).tolist():
        prior[x]+=1
    
    prior=torch.LongTensor(prior)
    _,r=prior.sort(dim=0,descending=True)
    
    #AP and P@R=50
    ap=[]
    p50=[]
    for i,ind in enumerate(r.view(-1).tolist()[:K]):
        s=scores[:,ind]
        _,rank=s.sort(dim=0,descending=True)
        _,rank=rank.sort(dim=0)
        rank_gt=rank[gt.eq(ind)]+1
        #print(rank_gt)
        rank_gt,_=rank_gt.sort(dim=0)
        ap_i=torch.arange(1,len(rank_gt)+1).to(rank_gt.device)/rank_gt
        ind_0=math.floor(len(ap_i)/2)
        ind_1=math.ceil(len(ap_i)/2)
        p50_i=(ap_i[ind_0]+ap_i[ind_1])/2
        ap_i=ap_i.mean()
        
        ap.append(float(ap_i))
        p50.append(float(p50_i))
    
    return sum(ap)/len(ap),sum(p50)/len(p50)
