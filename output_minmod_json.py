import json
import sys
import os
import pandas
import copy
import torch
import torch.nn.functional as F
import gzip

import util.smartparse as smartparse
import util.session_manager as session_manager

import util.helper as helper

def default_params():
    params=smartparse.obj();
    
    params.score_threshold=0.2
    params.split='index/demo_sites.csv'
    params.scores='predictions/scores_qa_gpt-4o-mini'
    params.json='dataset/'
    params.out='minmod/demo/SRI_deptype_minmod_gpt-4o-mini.json'
    
    params.override=False
    return params

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
#session=session_manager.create_session(params);


minmod_types=pandas.read_csv('minmod/deposit_type.csv')
types_cmmi=pandas.read_csv('taxonomy/cmmi_options_full_gpt4_number.csv', encoding = "latin1")
#Merge minmod types into types_cmmi
print('Mapping CMMI to minmod types')
options=list(types_cmmi['Deposit type'])
options_minmod_name=[]
options_minmod_id=[]
for x in options:
    d=[]
    for y in list(minmod_types['Deposit type']):
        d.append(helper.levenshteinDistance(x,y))
    
    v,ind=torch.Tensor(d).min(dim=0)
    v=float(v)
    ind=int(ind)
    if v>=2:
        print(v,x,minmod_types['Deposit type'][ind])
    
    options_minmod_id.append(minmod_types['Minmod ID'][ind])
    options_minmod_name.append(minmod_types['Deposit type'][ind])

def parse_gt_type(s):
    if s in options:
        return [options.index(s)]
    if s in options_minmod_name:
        return [options_minmod_name.index(s)]
    
    try:
        s=json.loads(s)
        if isinstance(s,list):
            data=[]
            for x in s:
                data+=parse_gt_type(x)
            
            print(data)
            return data
        else:
            return []
    except:
        return []


'''
example={
    'MineralSite':
    [
        {
            'source_id':'https://mrdata.usgs.gov/mrds',
            'record_id':10079610,
            'site_rank':'',
            'deposit_type_candidate':[
                {
                    'observed_name': 'Abyssal pegmatite REE',
                    'source': 'SME',
                    'confidence': 0.5,
                    'normalized_uri': 'https://minmod.isi.edu/resource/Q469',
                    
                },
                {
                    'observed_name': 'Albitite-hosted uranium',
                    'source': 'SME',
                    'confidence': 0.5,
                    'normalized_uri': 'https://minmod.isi.edu/resource/Q398',
                    
                },
            ],
        },
    ]
}
'''

commodity2id=pandas.read_csv('minmod/commodity.csv')
commodity2id=dict(zip(list(commodity2id['CommodityinMRDS']),list(commodity2id['minmod_id'])))
missing_commodities=set()
def mineral_inventory(names,source):
    data=[]
    for name in names:
        if name in commodity2id:
            data_i={'commodity':{},'reference':{'document':{'uri':source}}}
            data_i['commodity']['observed_name']=name
            data_i['commodity']['normalized_uri']='https://minmod.isi.edu/resource/%s'%commodity2id[name]
            data_i['commodity']['source']='SRI database agent v0'
            data_i['commodity']['confidence']=1.0
            data.append(data_i)
        else:
            data_i={'commodity':{},'reference':{'document':{'uri':source}}}
            data_i['commodity']['observed_name']=name
            data_i['commodity']['source']='SRI database agent v0'
            data_i['commodity']['confidence']=1.0
            data.append(data_i)
            missing_commodities.add(name)
    
    return data

def deposit_type_candidate_gt(inds):
    if len(inds)>0:
        p=float(1/len(inds))
    
    data=[]
    for i in inds:
        data_i={}
        data_i['observed_name']=options_minmod_name[i]
        data_i['confidence']=p
        data_i['normalized_uri']='https://minmod.isi.edu/resource/%s'%options_minmod_id[i]
        data_i['source']='algorithm predictions, SRI crosswalk agent v0'
        data.append(data_i)
    
    return data

def deposit_type_candidate_scores(inds,ps):
    data=[]
    for i,j in enumerate(inds):
        data_i={}
        data_i['observed_name']=options_minmod_name[j]
        data_i['confidence']=ps[i]
        data_i['normalized_uri']='https://minmod.isi.edu/resource/%s'%options_minmod_id[j]
        data_i['source']='algorithm predictions, SRI deposit type classification, v2, 20240710'
        data.append(data_i)
    
    return data

class minmod_writer:
    def __init__(self,N=5000,name='SRI_MRDS_v1'):
        self.data=[]
        self.N=N
        self.i=0
        self.name='.'.join(name.split('.')[:-1]) #Remove '.json'
        self.suffix=name.split('.')[-1]
    
    def append(self,site):
        self.data.append(site)
        if len(self.data)>=self.N:
            self.dump()
    
    def dump(self):
        if len(self.data)==0:
            return;
        
        
        if self.i==0:
            fname='%s.%s'%(self.name,self.suffix)
            os.makedirs(os.path.dirname(fname),exist_ok=True)
            json.dump({'MineralSite':self.data},open(fname,'w'),indent=2)
        else:
            fname='%s_part%02d.%s'%(self.name,self.i,self.suffix)
            os.makedirs(os.path.dirname(fname),exist_ok=True)
            json.dump({'MineralSite':self.data},open(fname,'w'),indent=2)
        
        self.i+=1
        self.data=[]
    
    def __len__(self):
        return self.i*self.N+len(self.data)

'''
example2={"MineralSite": [
        {
            "deposit_type_candidate": [],
            "source_id": "https://mrdata.usgs.gov/sedexmvt",
            "record_id": 139,
            "name": "Maramungee",
            "mineral_inventory": [],
        }]}
'''

#GT data
index=pandas.read_csv(params.split,low_memory=False)
index = index.where(pandas.notnull(index), None)
index={k:list(index[k]) for k in index.keys()}
index_type=[parse_gt_type(x) for x in list(index['deposit_type'])]
index['type']=index_type
root_scores=params.scores
root_json=params.json

#Predictions
def get_source(path):
    dataset=path.split('/')[1]
    if dataset=='ardf':
        return 'https://doi.org/10.5066/P96MMRFD'
    elif dataset=='ofr20051294':
        return 'https://mrdata.usgs.gov/major-deposits'
    else:
        return 'https://mrdata.usgs.gov/%s'%dataset

def get_record_id(path):
    dataset=path.split('/')[1]
    i=path.split('/')[-1].split('.')[0]
    '''
    try:
        i=int(i)
    except:
        pass
    '''
    return i
    

records=minmod_writer(name=params.out)
threshold=params.score_threshold
for i in range(len(index['path'])):
    print('%d      '%(len(records)),end='\r')
    #Load prediction if exists
    path=index['path'][i]
    top_p=[]
    path_scores=os.path.join(root_scores,path.replace('.json','.gz'))
    if os.path.exists(path_scores):
        s=torch.load(gzip.open(path_scores,'rb'),map_location='cpu')
        s=F.softmax(s,dim=-1)[:len(options)]
        s,ind=s.sort(dim=-1,descending=True)
        s=s.tolist()
        ind=ind.tolist()
        top_p=[(ind[j],s[j]) for j in range(len(s)) if s[j]>=threshold]
    
    
    if not (len(index['type'][i])>0 or len(top_p)>0):
        continue
    
    d=json.load(open(os.path.join(root_json,index['path'][i]),'r'))
    
    source_id=get_source(path)
    record_id=get_record_id(path)
    name=index['name'][i]
    #Cleaning
    if not isinstance(name,str) or name.find('[{')>=0:
        name=None
    
    if name is None:
        print('Missing name',path)
        print(index['type'][i])
        print(top_p)
        #name=''
        #a=0/0
    
    commodity=helper.mrds_get_commodity(d)
    commodity=mineral_inventory(commodity,source_id)
    
    deptype=deposit_type_candidate_gt(index['type'][i])
    if len(top_p)>0:
        deptype+=deposit_type_candidate_scores([x[0] for x in top_p],[x[1] for x in top_p])
    
    site_rank=helper.mrds_get_rank(d)
    site_type=helper.mrds_get_type(d)
    
    #compose record
    record={}
    record['deposit_type_candidate']=deptype
    record['mineral_inventory']=commodity
    record['source_id']=source_id
    record['record_id']=record_id
    if not name is None:
        record['name']=name
    
    if not site_rank is None:
        record['site_rank']=site_rank
    
    if not site_type is None:
        record['site_type']=site_type
    
    records.append(record)

records.dump()


print('missing minmod commodities',list(missing_commodities))

