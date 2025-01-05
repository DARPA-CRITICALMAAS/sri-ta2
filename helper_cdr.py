import httpx
import requests

class new:
    def __init__(self,endpoint,api_version,cdr_key):
        self.cdr_key=cdr_key
        self.endpoint=endpoint+api_version
    
    def list_documents(self,i,N=1000):
        endpoint = f"{self.endpoint}/docs/documents"
        params={'size':N,'page':i}
        headers = {"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        response = httpx.get(endpoint,params=params,headers=headers)
        response.raise_for_status()
        return response.json()
    
    def query_documents_provenance(self,kw,i=0,N=1000):
        endpoint = f"{self.endpoint}/docs/documents/q/provenance"
        params={'size':N,'page':i,'pattern':kw}
        headers = {"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        response = httpx.post(endpoint,params=params,headers=headers)
        response.raise_for_status()
        return response.json()
    
    def query_document_metadata(self,doc_id):
        endpoint = f"{self.endpoint}/docs/document/meta/{doc_id}"
        headers = {"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        response = httpx.get(endpoint,headers=headers)
        response.raise_for_status()
        return response.json()
    
    def add_document_metadata(self,doc_id,source_name,source_url,source_id=""):
        endpoint = f"{self.endpoint}/docs/documents/provenance/{doc_id}"
        params={"external_system_name": source_name,"external_system_id": source_id,"external_system_url": source_url}
        headers={"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        response = httpx.post(endpoint,json=params,headers=headers)
        response.raise_for_status()
        return response
    
    #kw is understood as a regex
    def query_documents_title(self,kw='Bisie Project in Africa',i=0,N=1000):
        endpoint = f"{self.endpoint}/docs/documents/q/title"
        params={'size':N,'page':i,'pattern':kw}
        headers = {"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        response = httpx.post(endpoint,params=params,headers=headers)
        response.raise_for_status()
        return response.json()
    
    def download_document(self,doc_id,local_fname):
        endpoint = f"{self.endpoint}/docs/document/%s"%doc_id
        headers = {"Authorization": "Bearer %s"%self.cdr_key, "accept": "application/json"}
        #response = requests.get(endpoint,headers=headers, stream=True)
        with requests.get(endpoint,headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(local_fname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
        return local_fname
