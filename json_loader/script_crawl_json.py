import os
import urllib.request
import json
import pandas

def update_json_from_url(url,fname):
    if not os.path.exists(fname):
        try:
            with urllib.request.urlopen(url) as f:
                data = json.load(f)
            
            with open(fname,'w') as f2:
                json.dump(data,f2)
        except:
            print('error processing %s'%url)


root='mrdata_links'
root_out='mrdata_json'
fnames=os.listdir('mrdata_links')

for fname in fnames:
    data=pandas.read_csv(os.path.join(root,fname))
    url=list(data['url'])
    depid=list(data['deposit_id'])
    
    folder_out='.'.join(fname.split('.')[:-1])
    try:
        os.mkdir(os.path.join(root_out,folder_out))
    except:
        pass
    
    for i in range(len(url)):
        update_json_from_url(url[i],os.path.join(root_out,folder_out,'%s.json'%str(depid[i])))

