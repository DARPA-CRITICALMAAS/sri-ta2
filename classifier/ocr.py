

'''
pip install pdf2image
pip install pytesseract
pip install opencv-python
conda install tesseract
'''

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output
import cv2
import time

from multiprocessing.pool import ThreadPool
import copy

tessdata_dir_config = r'--tessdata-dir ./tessdata/'
def ocr(fname):
    t0=time.time()
    pdf_pages = convert_from_path(fname)
    data=[]
    for pid,img in enumerate(pdf_pages):
        print('Reading %s page %d/%d, time %.2f'%(fname,pid,len(pdf_pages),time.time()-t0),end='\r')
        #print('Reading %s page %d/%d'%(fname,pid,len(pdf_pages)))
        d = pytesseract.image_to_data(copy.deepcopy(img), output_type=Output.DICT,config=tessdata_dir_config)
        blks=set(d['block_num'])
        #cleaning
        for block_id in blks:
            data_i=[]
            for i in range(len(d['block_num'])):
                if d['block_num'][i]==block_id:
                    if d['text'][i]=='':
                        data_i.append('\n')
                    else:
                        data_i.append(d['text'][i])
            
            s=' '.join(data_i)
            s=s.replace('- \n ','')
            s=s.replace('-\n ','')
            s=s.replace(' \n ',' ')
            #s=s.replace('\n',' ')
            data.append(s)
    
    '''
    for s in data:
        print(s)
    '''
    
    return data

def ocr_parallel(fname,num_workers=16):
    def parse_page(img):
        d = pytesseract.image_to_data(copy.deepcopy(img), output_type=Output.DICT,config=tessdata_dir_config)
        blks=set(d['block_num'])
        #cleaning
        data=[]
        for block_id in blks:
            data_i=[]
            for i in range(len(d['block_num'])):
                if d['block_num'][i]==block_id:
                    if d['text'][i]=='':
                        data_i.append('\n')
                    else:
                        data_i.append(d['text'][i])
            
            s=' '.join(data_i)
            s=s.replace('- \n ','')
            s=s.replace('-\n ','')
            s=s.replace(' \n ',' ')
            #s=s.replace('\n',' ')
            data.append(s)
        
        return data
    
    pdf_pages = convert_from_path(fname)
    pool = ThreadPool(processes=num_workers)
    results=pool.map(parse_page,pdf_pages)
    data=[]
    for x in results:
        data+=x
    
    return data

#stuff=parse('Anderson + Macqueen 1982.pdf')
def bulk_ocr(fnames,num_workers=2):
    pool = ThreadPool(processes=num_workers)
    results=pool.map(ocr,fnames)
    return results








