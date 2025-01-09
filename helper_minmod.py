import httpx
import requests
import math
import pandas
import json
import copy
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
        record['source_id']='mining-report::https://api.cdr.land/v1/docs/documents'
        record['record_id']=cdr_id
        record['site_rank']=""
        record['reference']=[{"document": {"uri": "https://api.cdr.land/v1/docs/documents/%s"%cdr_id}}]
        record['created_by']=["https://minmod.isi.edu/users/s/sri"]
        record['modified_at']=modified_at
        return record



class API:
    def __init__(self,endpoint,api_version,minmod_username,minmod_password):
        self.username=minmod_username
        self.password=minmod_password
        self.endpoint=endpoint+api_version
        self.endpoint_root=endpoint
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
        
        endpoint = f"{self.endpoint}/mineral-sites/{site_id}"
        response = httpx.put(endpoint,json=site_record,cookies=self.cookies,timeout=None)
        
        print(response.json())
        response.raise_for_status()
        return response.json()
    
    def get_id(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies,timeout=None)
        #print(response.json())
        response.raise_for_status()
        url=response.json()
        id=url.split('/')[-1]
        return id
    
    def get_site(self,cdr_id):
        site_id=self.get_id(cdr_id)
        
        endpoint = f"{self.endpoint}/mineral-sites/{site_id}"
        response = httpx.get(endpoint,cookies=self.cookies,timeout=None) #params=params,
        if json.dumps(response.json()).find('does not exist')>=0:
            print(response.json())
            return {}
        
        response.raise_for_status() 
        return response.json()
    
    def merge(self,old_record,new_record):
        #override everything except for deposit_type_candidate and created_by
        merged_record=copy.deepcopy(old_record)
        for k in new_record:
            #ignore empty items
            if new_record[k] is None:
                continue
            
            if k=='deposit_type_candidate':
                if not k in merged_record:
                    merged_record[k]=new_record[k]
                else:
                    merged_record[k]+=new_record[k]
                
                #Remove identical records to prevent pollution
                deposit_type_predictions=merged_record[k]
                deposit_type_predictions={json.dumps(x):x for x in deposit_type_predictions}
                deposit_type_predictions=[deposit_type_predictions[x] for x in deposit_type_predictions]
                merged_record[k]=deposit_type_predictions
            elif k=='created_by':
                pass
            else:
                merged_record[k]=new_record[k]
        
        return merged_record
    
    def update_site_safe(self,cdr_id,site_record):
        #Get site
        result=self.create_site(site_record)
        if len(result)==0:
            old_record=self.get_site(cdr_id)
            new_record=self.merge(old_record,site_record)
            result=self.update_site(cdr_id,new_record)
        
        #response.raise_for_status()
        return result
    
    def link_to_site(self,cdr_id):
        site_id=self.get_id(cdr_id)
        url = f"{self.endpoint}/mineral-sites/{site_id}"
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

