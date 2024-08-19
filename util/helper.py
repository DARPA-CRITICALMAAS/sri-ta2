import time
import torch
import torch.nn.functional as F


#MRDS utils
def mrds_clean_json(x):
    if isinstance(x,list):
        return [clean_json(v) for v in x]
    elif isinstance(x,dict):
        return {k:clean_json(x[k]) for k in x if not k in ['inserted_by','updated_by','insert_date','update_date','recid']}
    else:
        return x


def mrds_get_name(d):
    if 'properties' in d:
        d=d['properties']
    
    if not 'name' in d:
        print('cannot find name',d)
        return None
    
    name=d['name']
    if isinstance(name,str):
        return name
    elif isinstance(name,list):
        names=[x['name'] for x in name if x['status']=='Current']
        assert len(names)==1
        return names[0]
    elif isinstance(name,dict) and 'name' in name and isinstance(name['name'],str):
        return name['name']
    else:
        print('cannot find name',name)
        return None


def mrds_get_lat(x):
    if 'geometry' in x:
        coord=x['geometry']['coordinates']
        if isinstance(coord[0],list):
            assert isinstance(coord[0][0],float) or isinstance(coord[0][0],int) 
            assert isinstance(coord[0][1],float) or isinstance(coord[0][1],int) 
            return coord[0][0],coord[0][1]
        else:
            assert isinstance(coord[0],float) or isinstance(coord[0],int) 
            assert isinstance(coord[1],float) or isinstance(coord[1],int) 
            return coord[0],coord[1]
    else:
        return 0,0


def mrds_get_link(item):
    if not 'id' in item and 'deposits' in item:
        i=item['deposits']['dep_id']
        i='https://mrdata.usgs.gov/mrds/record/%s'%i
    else:
        i=item['id']
    
    return i

def mrds_get_id(item):
    if not 'id' in item and 'deposits' in item:
        i=item['deposits']['dep_id']
        i=int(i)
    else:
        i=int(item['id'].split('/')[-1])
    
    return i


def mrds_get_commodity(d):
    if 'properties' in d:
        d=d['properties']
    
    if not 'commodity' in d:
        #print('cannot find commodity',d)
        return []
    
    commod=d['commodity']
    #print(commod)
    if not isinstance(commod,list):
        commod=[commod]
    
    commod_=[]
    for x in commod:
        if isinstance(x,str):
            commod_.append(x)
        elif 'commod' in x:
            commod_.append(x['commod'])
        elif 'name' in x:
            commod_.append(x['name'])
        elif 'value' in x:
            commod_.append(x['value'])
        else:
            a=0/0
    
    commod=commod_
    return commod

#Onlys seen in mrds
def mrds_get_rank(d):
    if "properties" in d:
        d=d['properties']
    
    if 'grade' in d:
        return d['grade']
    else:
        return None

#seen in mrds and ardf
def mrds_get_type(d):
    if "properties" in d:
        d=d['properties']
    
    if 'rec_tp' in d:
        return d['rec_tp']
    elif 'class' in d:
        return d['class']
    elif 'type' in d and not d['type']=='Feature':
        return d['type']
    else:
        return None



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


