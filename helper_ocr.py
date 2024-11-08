

'''
pip install pdf2image
pip install pytesseract
pip install opencv-python
conda install tesseract
'''

import pytesseract
from pdf2image import pdfinfo_from_path, convert_from_path
from PIL import Image
from pytesseract import Output
#import cv2
#import time

from multiprocessing.pool import ThreadPool
import copy

#tessdata_dir_config = r'--tessdata-dir ./tessdata/'
tessdata_dir_config = r''#--tessdata-dir ./tessdata/'

def ocr(fname,num_workers=16):
    def parse_page(stuff):
        pdf_file,page_id=stuff
        data=[]
        try:
            imgs=convert_from_path(pdf_file, dpi=200, first_page=page_id, last_page = page_id)
            for img in imgs:
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
        except:
            pass
        
        return data
    
    
    info = pdfinfo_from_path(fname, userpw=None, poppler_path=None)
    maxPages = info["Pages"]
    
    pdf_pages = [(fname,i) for i in range(maxPages+1)]
    pool = ThreadPool(processes=num_workers)
    results=pool.map(parse_page,pdf_pages)
    data=[]
    for x in results:
        data+=x
    
    return data








