import httpx
import requests
import math
import pandas
import json
from datetime import datetime

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


class writer:
    def __init__(self,params):
        self.params=params
        minmod_types=pandas.read_csv('minmod/deposit_type.csv')
        minmod_id=list(minmod_types['Minmod ID'])
        minmod_types=list(minmod_types['Deposit type'])
        try:
            cmmi=list(pandas.read_csv(params.taxonomy)['Deposit type'])
        except:
            cmmi=list(pandas.read_csv(params.taxonomy,encoding='latin1')['Deposit type'])
        
        minmod_mapping={}
        for i,x in enumerate(minmod_types):
            best=None
            d=1e10
            for j,y in enumerate(cmmi):
                dj=levenshteinDistance(x,y)
                if dj<d:
                    best=y
                    d=dj
            
            print('CMMI - Minmod mapping: %s - %s'%(best,x),end='\r')
            minmod_mapping[best]={'deposit_type':x,'id':minmod_id[i]}
        
        self.minmod_mapping=minmod_mapping
        self.cmmi=cmmi
        self.threshold=params.confidence_threshold
    
    def deposit_type_candidate(self,candidates,explanation):
        data=[]
        for i,(cmmi,p) in enumerate(candidates):
            data_i={}
            data_i['observed_name']=explanation
            data_i['confidence']=p*0.9
            data_i['normalized_uri']='https://minmod.isi.edu/resource/%s'%self.minmod_mapping[cmmi]['id']
            data_i['source']='algorithm predictions, SRI deposit type classification, v2, 20240710'
            data.append(data_i)
        
        return data
    
    def mineral_site_cdr(self,cdr_id,logp,explanation=''):
        modified_at=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        predictions=list(zip(self.cmmi,logp[:len(self.cmmi)]))
        predictions=[(x[0],math.exp(x[1])) for x in predictions if math.exp(x[1])>=self.threshold]
        if len(predictions)==0:
            return None
        
        predictions=sorted(predictions,reverse=True,key=lambda x:x[1])
        deptype=self.deposit_type_candidate(predictions,explanation)
        
        #compose record
        record={}
        record['deposit_type_candidate']=deptype
        record['source_id']='mining-report::https://api.cdr.land/v1/docs/documents'
        record['record_id']=cdr_id
        record['site_rank']=""
        record['reference']=[{"document": {"uri": "https://api.cdr.land/v1/docs/documents/%s"%cdr_id}}]
        record['created_by']=["https://minmod.isi.edu/users/s/sri"]
        record['modified_at']=modified_at
        return record



class API:
    def __init__(self,minmod_username,minmod_password):
        self.username=minmod_username
        self.password=minmod_password
        self.endpoint='https://dev.minmod.isi.edu/api/v1'
        self.cookies=None
    
    def create_site(self,site_record):
        endpoint = f"{self.endpoint}/mineral-sites"
        params=site_record
        response = httpx.post(endpoint,json=params,cookies=self.cookies,timeout=None)
        if json.dumps(response.json()).find('exists')>=0:
            print(response.json())
            return {}
        
        response.raise_for_status()
        print(response.json())
        return response.json()
    
    def update_site(self,cdr_id,site_record):
        site_id=self.get_id(cdr_id)
        
        endpoint = f"{self.endpoint}/mineral-sites"
        params={'site_id':site_id}
        response = httpx.put(endpoint,params=params,json=site_record,cookies=self.cookies,timeout=None)
        if json.dumps(response.json()).find('exists')>=0:
            print(response.json())
            return {}
        
        response.raise_for_status()
        print(response.json())
        return response.json()
    
    def get_id(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies,timeout=None)
        #print(response.json())
        response.raise_for_status()
        url=response.json()
        return url
    
    def link_to_site(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies,timeout=None)
        #print(response.json())
        response.raise_for_status()
        url=response.json()
        url=url.replace('minmod','dev.minmod')
        return url
    
    def login(self):
        endpoint = f"{self.endpoint}/login"
        params={'username':self.username,'password':self.password}
        response = httpx.post(endpoint,json=params,timeout=None)
        response.raise_for_status()
        self.cookies=response.cookies
        return response.json()
    
    def whoami(self):
        endpoint = f"{self.endpoint}/whoami"
        response = httpx.get(endpoint,cookies=self.cookies,timeout=None)
        response.raise_for_status()
        return response.json()

