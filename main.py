
#Load config
import os
import sys
import json
import time
from datetime import datetime
import util.smartparse as smartparse
import util.session_manager as session_manager
params=json.load(open('config.json','r'))


def default_params():
    params=smartparse.obj()
    params.cdr_key=""
    params.openai_api_key=""
    params.azure_lm=''
    params.azure_api_version='2024-07-01-preview'
    params.azure_api_endpoint=''
    params.lm="gpt-4o"
    params.lm_context_window=128000
    params.ocr_num_threads=12
    params.dir_cache_pdf="cache/docs_PDF"
    params.dir_cache_ocr="cache/docs_ocr"
    params.dir_predictions="predictions"
    params.dir_mineral_sites="sri/mineral_sites"
    params.taxonomy="taxonomy/cmmi_full_num_v2.csv"
    params.cdr_query_interval=30
    params.confidence_threshold=0.2
    params.minmod_username=""
    params.minmod_password=""
    return params


params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv
session=session_manager.create_session(params)


class DTC_APP:
    def __init__(self,minmod_api,minmod_writer,cdr,depqa,session,params):
        self.queue=set()
        self.cdr=cdr
        self.depqa=depqa
        self.minmod_api=minmod_api
        self.minmod_writer=minmod_writer
        self.session=session
        self.params=params
    
    def process(self):
        self.query_docs()
        for i in self.queue:
            print(i)
            url=self.update_kg(i)
            if not (url is None or url=='Exists'):
                t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session.log('%s Processed %s, at %s'%(t,i,url))
        
        return
    
    #Query docs
    def query_docs(self):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s Querying CDR for docs'%t)
        cdr=self.cdr
        docs=[]
        for i in range(100000):
            try:
                data=cdr.query_documents_title(kw=r'*[Mineral Site]*',i=i,N=1000)
                docs+=data
                if len(data)<=0:
                    break
            
            except KeyboardInterrupt:
                a=0/0
            except:
                break
        
        #Check new uploads
        unique_ids=sorted(list(set([doc['id'] for j,doc in enumerate(docs)])))
        self.queue=self.queue | set(unique_ids)
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s %d documents in queue'%(t,len(self.queue)))
        return
    
    def update_kg(self,cdr_id):
        params=self.params
        cdr=self.cdr
        minmod_api=self.minmod_api
        
        fname_out=os.path.join(params.dir_mineral_sites,'%s.json'%cdr_id)
        if os.path.exists(fname_out):
            return 'Exists'#json.load(open(fname_out,'r'))
        
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s Attempting to update KG of document %s'%(t,cdr_id))
        
        pred=self.get_prediction(cdr_id)
        if pred is None:
            return None
        
        try:
            site_data=minmod_writer.mineral_site_cdr(cdr_id,pred['scores'],pred['justification'])
            minmod_api.login()
            try:
                minmod_api.create_site(site_data)
            except:
                minmod_api.update_site(cdr_id,site_data)
            
            url=minmod_api.link_to_site(cdr_id)
            meta=cdr.query_document_metadata(cdr_id)
            if not any([x['external_system_name']==minmod_api.endpoint for x in meta['provenance']]):
                cdr.add_document_metadata(doc_id=cdr_id,source_name=minmod_api.endpoint,source_url=url)
            
            os.makedirs(params.dir_mineral_sites,exist_ok=True)
            json.dump(site_data,open(fname_out,'w'),indent=2)
            return url
        except KeyboardInterrupt:
            a=0/0
        except Exception as error:
            print("Delivery issue with %s "%cdr_id, type(error).__name__, "–", error)
            pass
        
        return None
    
    def get_prediction(self,cdr_id):
        params=self.params
        depqa=self.depqa
        
        
        fname_out=os.path.join(params.dir_predictions,'%s.json'%cdr_id)
        if os.path.exists(fname_out):
            return json.load(open(fname_out,'r'))
        
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s Attempting to make predictions on document %s'%(t,cdr_id))
        
        text=self.get_ocr(cdr_id)
        if text is None:
            return None
        
        try:
            if isinstance(text,list):
                text='\n'.join(text)
            
            scores,justification=depqa.run(text)
            data={'scores':scores,'justification':justification}
            os.makedirs(params.dir_predictions,exist_ok=True)
            json.dump(data,open(fname_out,'w'),indent=2)
            return data
        except KeyboardInterrupt:
            a=0/0
        except Exception as error:
            print("Prediction issue with %s "%cdr_id, type(error).__name__, "–", error)
            pass
        
        return None
    
    def get_ocr(self,cdr_id):
        params=self.params
        fname_out=os.path.join(params.dir_cache_ocr,'%s.json'%cdr_id)
        if os.path.exists(fname_out):
            return json.load(open(fname_out,'r'))
        
        
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s Attempting to run OCR on document %s'%(t,cdr_id))
        
        fname_in=self.get_pdf(cdr_id)
        if fname_in is None:
            return None
        
        try:
            fname_in=os.path.join(params.dir_cache_pdf,'%s.pdf'%cdr_id)
            data=OCR.ocr(fname_in,num_workers=params.ocr_num_threads)
            os.makedirs(params.dir_cache_ocr,exist_ok=True)
            json.dump(data,open(fname_out,'w'),indent=2)
            return data
        except KeyboardInterrupt:
            a=0/0
        except Exception as error:
            print("OCR issue with %s "%cdr_id, type(error).__name__, "–", error)
            pass
        
        return None
    
    def get_pdf(self,cdr_id):
        params=self.params
        cdr=self.cdr
        
        fname_out=os.path.join(params.dir_cache_pdf,'%s.pdf'%cdr_id)
        if os.path.exists(fname_out):
            return fname_out
        
        
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session.log('%s Attempting to download document from CDR %s'%(t,cdr_id))
        
        try:
            os.makedirs(params.dir_cache_pdf,exist_ok=True)
            cdr.download_document('%s'%cdr_id,os.path.join(params.dir_cache_pdf,'%s.pdf'%cdr_id))
            return fname_out
        except KeyboardInterrupt:
            a=0/0
        except Exception as error:
            print("Error downloading %s "%cdr_id, type(error).__name__, "–", error)
            pass
        
        return None



if __name__ == "__main__":
    #Initialize modules
    import helper_cdr as CDR
    cdr=CDR.new(cdr_key=params.cdr_key)
    import helper_openai as LLM
    llm=LLM.new(params=params)
    import helper_deptype as DEPQA
    depqa=DEPQA.new(llm=llm,params=params)
    import helper_ocr as OCR
    import helper_minmod as minmod
    minmod_writer=minmod.writer(params)
    minmod_api=minmod.API(params.minmod_username,params.minmod_password)
    
    
    app=DTC_APP(minmod_api,minmod_writer,cdr,depqa,session,params)

    while True:
        t0=time.time()
        try:
            app.process()
        except KeyboardInterrupt:
            a=0/0
        except Exception as error:
            print("Exception", type(error).__name__, "–", error)
            pass
        
        
        #Check every so often
        t1=time.time()
        delta=min(params.cdr_query_interval-(t1-t0),params.cdr_query_interval)
        if delta>0:
            time.sleep(delta)
        





