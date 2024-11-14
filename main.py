
#Load config
import os
import json
import time
from datetime import datetime
import util.smartparse as smartparse
import util.session_manager as session_manager
params=json.load(open('config.json','r'))
params=smartparse.dict2obj(params)
session=session_manager.create_session(params)


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
    
    while True:
        t0=time.time()
        try:
            #Query docs from CDR
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
            needs_processing=[i for j,i in enumerate(unique_ids) if not os.path.exists(os.path.join(params.dir_mineral_sites,'%s.json'%i))]
            
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session.log('%s\tCDR update: %d/%d new docs'%(t,len(needs_processing),len(unique_ids)))
            
            #Download report
            for j,i in enumerate(needs_processing):
                if not os.path.exists(os.path.join(params.dir_cache_pdf,'%s.pdf'%i)) and all([not os.path.exists(os.path.join(dir,'%s.json'%i)) for dir in [params.dir_predictions,params.dir_cache_ocr,params.dir_mineral_sites]]):
                    try:
                        os.makedirs(params.dir_cache_pdf,exist_ok=True)
                        cdr.download_document('%s'%i,os.path.join(params.dir_cache_pdf,'%s.pdf'%i))
                        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        session.log('%s\tDownloaded %d/%d %s'%(t,j+1,len(needs_processing),i))
                    except KeyboardInterrupt:
                        a=0/0
                    except Exception as error:
                        print("Error downloading %s "%i, type(error).__name__, "–", error)
                        pass
            
            #Run OCR
            for j,i in enumerate(needs_processing):
                if os.path.exists(os.path.join(params.dir_cache_pdf,'%s.pdf'%i)) and all([not os.path.exists(os.path.join(dir,'%s.json'%i)) for dir in [params.dir_predictions,params.dir_mineral_sites,params.dir_cache_ocr]]):
                    try:
                        os.makedirs(params.dir_cache_ocr,exist_ok=True)
                        fname_in=os.path.join(params.dir_cache_pdf,'%s.pdf'%i)
                        fname_out=os.path.join(params.dir_cache_ocr,'%s.json'%i)
                        data=OCR.ocr(fname_in,num_workers=params.ocr_num_threads)
                        json.dump(data,open(fname_out,'w'),indent=2)
                        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        session.log('%s\tOCR %d/%d %s'%(t,j+1,len(needs_processing),i))
                    except KeyboardInterrupt:
                        a=0/0
                    except Exception as error:
                        print("OCR issue with %s "%i, type(error).__name__, "–", error)
                        pass
            
            #Run prediction
            for j,i in enumerate(needs_processing):
                if os.path.exists(os.path.join(params.dir_cache_ocr,'%s.json'%i)) and all([not os.path.exists(os.path.join(dir,'%s.json'%i)) for dir in [params.dir_predictions,params.dir_mineral_sites]]):
                    try:
                        os.makedirs(params.dir_predictions,exist_ok=True)
                        fname_in=os.path.join(params.dir_cache_ocr,'%s.json'%i)
                        fname_out=os.path.join(params.dir_predictions,'%s.json'%i)
                        text=json.load(open(fname_in,'r'))
                        if isinstance(text,list):
                            text='\n'.join(text)
                        
                        scores,justification=depqa.run(text)
                        data={'scores':scores,'justification':justification}
                        json.dump(data,open(fname_out,'w'),indent=2)
                        
                        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        session.log('%s\tClassify %d/%d %s,tokens so far: %d'%(t,j+1,len(needs_processing),i,llm.token_count))
                    except KeyboardInterrupt:
                        a=0/0
                    except Exception as error:
                        print("Prediction issue with %s "%i, type(error).__name__, "–", error)
                        pass
            
            #Generate mineral site data
            minmod_api.login()
            for i in needs_processing:
                if os.path.exists(os.path.join(params.dir_predictions,'%s.json'%i)) and not os.path.exists(os.path.join(params.dir_mineral_sites,'%s.json'%i)):
                    try:
                        os.makedirs(params.dir_mineral_sites,exist_ok=True)
                        fname_in=os.path.join(params.dir_predictions,'%s.json'%i)
                        fname_out=os.path.join(params.dir_mineral_sites,'%s.json'%i)
                        data=json.load(open(fname_in,'r'))
                        data=minmod_writer.mineral_site_cdr(i,data['scores'],data['justification'])
                        
                        
                        minmod_api.create_site(data)
                        url=minmod_api.link_to_site(i)
                        meta=cdr.query_document_metadata(i)
                        if not any([x['external_system_name']==minmod_api.endpoint for x in meta['provenance']]):
                            cdr.add_document_metadata(doc_id=i,source_name=minmod_api.endpoint,source_url=url)
                        
                        json.dump(data,open(fname_out,'w'),indent=2)
                        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        session.log('%s\Knowledge graph data %d/%d %s'%(t,j+1,len(needs_processing),i))
                    except KeyboardInterrupt:
                        a=0/0
                    except Exception as error:
                        print("Delivery issue with %s "%i, type(error).__name__, "–", error)
                        pass
            
                    
                    
            
            
        except KeyboardInterrupt:
            a=0/0
        except:
            print("Exception", type(error).__name__, "–", error)
            pass
        
        
        '''
        minmod_api.login()
        minmod_api.query_site("02c46db47a8a1b2d2705a2b2190951a2eab55b6909f845cc607f561a85852a1822")
        data=json.load(open('sri/mineral_sites/02c46db47a8a1b2d2705a2b2190951a2eab55b6909f845cc607f561a85852a1822.json','r'))
        minmod_api.create_site(data)
        '''
        
        #Check every so often
        t1=time.time()
        delta=min(params.cdr_query_interval-(t1-t0),params.cdr_query_interval)
        if delta>0:
            time.sleep(delta)
        





