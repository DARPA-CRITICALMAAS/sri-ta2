import httpx
import requests
import math
import pandas
import json
import copy
from datetime import datetime

from minmodapi import MinModAPI

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
            data_i['observed_name']='Prediction: %s. %s'%(cmmi,explanation)
            data_i['confidence']=p*0.9
            if cmmi in self.minmod_mapping:
                data_i['normalized_uri']='https://minmod.isi.edu/resource/%s'%self.minmod_mapping[cmmi]['id']
            
            data_i['source']=self.params.minmod_algorithm_string
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
        record['source_id']=self.params.minmod_cdr_source
        record['record_id']=cdr_id
        record['site_rank']=""
        record['reference']=[{"document": {"uri": "https://api.cdr.land/v1/docs/documents/%s"%cdr_id}}]
        record['created_by']=["https://minmod.isi.edu/users/s/sri"]
        record['modified_at']=modified_at
        return record

def update_func(existing_site: dict, new_site: dict):
    # override everything except for deposit_type_candidate and created_by
    print(existing_site)
    for k in new_site:
        # ignore empty items
        if new_site[k] is None:
            continue
        
        if k == "deposit_type_candidate":
            if not k in existing_site:
                existing_site[k] = new_site[k]
            else:
                existing_site[k] += new_site[k]
            
            # Remove identical records to prevent pollution
            deposit_type_predictions = existing_site[k]
            deposit_type_predictions = {
                json.dumps(x,sort_keys=True): x for x in deposit_type_predictions
            }
            deposit_type_predictions = [
                deposit_type_predictions[x] for x in deposit_type_predictions
            ]
            existing_site[k] = deposit_type_predictions
        elif k == "created_by":
            pass
        else:
            existing_site[k] = new_site[k]
    
    return existing_site

class API(MinModAPI):
    def __init__(self,endpoint,minmod_username,minmod_password):
        super().__init__(endpoint)
        self.username=minmod_username
        self.password=minmod_password
    
    def link_to_site(self,site_id):
        #site_id=self.get_id(cdr_id)
        url = f"{self.endpoint}/api/v1/mineral-sites/{site_id}"
        return url
        #url = f"{self.endpoint_root}/resource/{site_id}"
        #return url
        '''
        url=f"https://minmod.isi.edu/resource/"
        
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies,timeout=None)
        #print(response.json())
        response.raise_for_status()
        url=response.json()
        #url=url.replace('https://minmod.isi.edu',self.endpoint_root)
        return url
        '''
    
    def update_site_safe(self,site_data):
        return super().upsert_mineral_site(site_data,update_func)
    
    def login(self):
        return super().login(self.username,self.password)
    