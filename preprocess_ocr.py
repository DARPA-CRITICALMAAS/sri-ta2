
import torch
import os
import time
import sys
import json
import pandas
import math
import util.db as db

import gzip
import util.smartparse as smartparse
import util.session_manager as session_manager
import util.ocr as ocr


default_params=smartparse.obj();
default_params.pdf='dataset/NI_43-101'
default_params.out='dataset/NI_43-101_json'
default_params.threads=8

default_params.world_size=1
default_params.rank=0

params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv;
print(smartparse.obj2dict(params))

fnames=os.listdir(params.pdf)

t0=time.time()
fnames=sorted(fnames)
for i,fname in enumerate(fnames):
    print('%d/%d, time %.2f '%(i,len(fnames),time.time()-t0))
    fname_out=os.path.join(params.out,fname.replace('.pdf','.json'))
    
    if i%params.world_size==params.rank and not os.path.exists(fname_out):
        result=ocr.ocr_parallel(os.path.join(params.pdf,fname),num_workers=params.threads)
        os.makedirs(os.path.dirname(fname_out),exist_ok=True)
        json.dump(result,open(fname_out,'w'))

