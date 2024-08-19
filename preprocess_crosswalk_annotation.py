import os
import pandas
import util.agent as agent
import random


import sys
import util.smartparse as smartparse
import util.session_manager as session_manager
default_params=smartparse.obj();
default_params.openai_api_key='your_key'
params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv
agent.set_openai_key(params.openai_api_key)


# Crosswalk deposit type annotations
# Use USGS deposit code to CMMI mapping provided by Graham
table_usgs_cmmi=pandas.read_csv('taxonomy/usgs_num_cmmi_crosswalk.csv',encoding='latin1')
table_usgs_cmmi={'Deposit type':table_usgs_cmmi['Deposit type'],'USGS_Model':table_usgs_cmmi['USGS_Model']}
table_usgs=pandas.read_csv('taxonomy/usgs_num.csv',encoding='latin1')
table_usgs={'model_name':table_usgs['model_name'],'usgs_num':table_usgs['usgs_num']}

table_usgs_cmmi=agent.markdown_table(table_usgs_cmmi)

table_examples={}
table_examples['28-APR-03, Massive sulfide, kuroko, 12-MAR-02, 28a']="'28a'"
table_examples['Carbonatite, 28-APR-03, 12-MAR-02']="'10'"
table_examples['Mississippi Valley, S.E. Missouri Pb-Zn, E12, 32a']="'32a'"
table_examples['Basaltic Cu (Cox and Singer, 1986; model 23).']="'23'"
table_examples['Volcanogenic Mn? (Cox and Singer, 1986; model 24c?).']="'24c'"
table_examples['Disseminated, gold-bearing sulfide mineralization in calcareous metasedimentary schist; simple Sb deposits; low sulfide, Au-quartz vein? (Cox and Singer, 1986; model 27d and 36a).']="['27d','36a']"
table_examples['Massive sulfide, Besshi (Japanese deposits), woodruff, 17-OCT-2003 07:58:41, 12-MAR-2002 00:00:00']="'24b'"
table_examples={'deposit_type':list(table_examples.keys()),'usgs_num':[table_examples[k] for k in table_examples]}


#Slice crosswalked labeled data for MRDS, MRDS-CA and other datasets
# MRDS
input_file='index/sites/mrds.csv'
fname_out='index/annotations/mrds.csv'
df_out=agent.run_df_agent("I have collected a data frame of mineral site records in {input_file}, where some sites have been annotated with a deposit_type field. Please follow this table to extract the USGS model number:\n {table_usgs}\nAs you notice, there are two ways to get the deposit number. One is inferring the deposit number from the deposit type. The other is just reading the model number off the string. Here are some example input and outputs:\n{table_examples}\nPlease first filter for records with valid deposit_type annotations, and then add a new column `usgs_num` for the extracted USGS model number. Return a data frame with path, name, longitude, latitude, deposit_type and usgs_num columns. Notice that when inferring deposit number from the deposit type name, the deposit names can also have commas in them, so try to process the whole string. Also the deposit names can have overlaps, for example 'Placer Au-PGE' deposits are not 'Placer Au', so you'll need to return the longer string match 'Placer Au-PGE' which is '39a' and not 'Placer Au' or '17a'. ".format(input_file=input_file,table_usgs=agent.markdown_table(table_usgs),table_examples=agent.markdown_table(table_examples)))
#validate
print(len([x for x in list(df_out['usgs_num']) if not x is None]))
df_all.to_csv(fname_out)


# MRDS-CA

# Custom string matchers
import pandas
import re
import json

data=pandas.read_csv('index.csv')

rules=[]
rules.append({'dataset':'sir20105090z','name':'Porphyry copper','cmmi':['Porphyry copper ± gold']})
rules.append({'dataset':'sir20105090z','name':'Porphyry molybdenum-copper','cmmi':['Porphyry copper-molybdenum']})
rules.append({'dataset':'sir20105090z','name':'Porphyry gold','cmmi':['Porphyry gold ± copper']})
rules.append({'dataset':'porcu','name':'17','cmmi':['Porphyry copper ± gold']})
rules.append({'dataset':'porcu','name':'21a','cmmi':['Porphyry copper-molybdenum']})
rules.append({'dataset':'porcu','name':'20c','cmmi':['Porphyry gold ± copper​']})
rules.append({'dataset':'nicrpge','name':None,'cmmi':['Komatiite nickel-copper-PGE','U-M layered intrusion chromium','U-M layered intrusion nickel-copper-PGE','U-M layered intrusion PGE','U-M conduit nickel-copper-PGE','Arc U-M intrusion nickel-copper-PGE','Ophiolite chromium']})
rules.append({'dataset':'sedexmvt','name':'MVT','cmmi':['MVT zinc-lead']})


rules.append({'dataset':'carbonatite','name':'10','cmmi':['Carbonatite REE']})
rules.append({'dataset':'ree','name':'alk-ig','cmmi':['Peralkaline igneous HFSE-REE']})
rules.append({'dataset':'ree','name':'carb','cmmi':['Carbonatite REE']})
rules.append({'dataset':'ofr20151121','name':'carbonatite','cmmi':['Carbonatite REE']})
rules.append({'dataset':'ofr20151121','name':'igneous','cmmi':['Peralkaline igneous HFSE-REE']})
rules.append({'dataset':'ofr20151121','name':'Carbonatite','cmmi':['Carbonatite REE']})




#ardf
def get_dataset(fname):
    return fname.split('/')[1]

ardf_mapping={
    '17': 'Porphyry copper ± gold',
    '21a': 'Porphyry copper-molybdenum',
    '20c': 'Porphyry gold ± copper',
    '14a': 'Skarn tungsten ± Mo',
    '14b': 'Skarn tin ± copper ± Mo',
    '18f': 'Skarn gold ± copper ± tungsten',
    '6a': 'Komatiite nickel-copper-PGE',
    '2a': 'U-M layered intrusion chromium',
    '5a': 'U-M layered intrusion nickel-copper-PGE',
    '1': 'U-M layered intrusion PGE',
    '3': 'U-M intrusion nickel-copper-PGE',  # Assuming this mapping takes precedence over the duplicate
    '5b': 'U-M conduit nickel-copper-PGE',
    '9': 'Arc U-M intrusion nickel-copper-PGE',
    '7b': 'Anorthosite massif titanium',
    '31b': 'Siliciclastic-mafic barite',
    '31a': 'Siliciclastic-carbonate zinc-lead',
    '32c': 'Kipushi-type sediment-hosted copper-zinc-lead',
    '32a': 'MVT zinc-lead',
    '32f': 'MVT barite',
    '32d': 'MVT fluorspar',
    '25ob': 'Lacustrine zeolite (± Li, B)',
    '25lc': 'Lacustrine clay lithium',
    '35bm': 'Lacustrine brine lithium',
    '10': 'Carbonatite REE',
    '11': 'Peralkaline igneous HFSE-REE'
}

def extract_model_numbers(text):
    # Define the regex pattern to match the model numbers
    # This pattern matches numbers followed by optional sequences of letters or punctuation
    pattern = r'\bmodel(?:s)?\s*([\d\w\?\.\, ]+)\b'
    
    # Find all matches for the pattern
    matches = re.findall(pattern, text)
    
    # If matches are found, process them to extract individual model numbers
    if matches:
        result = []
        for match in matches:
            # Split by commas, 'or', and remove any extraneous characters such as '?' or spaces
            models = re.split(r'[,\sor]+', match.strip())
            for model in models:
                model = model.strip(' ?.,')
                if model:  # Make sure the model number is not empty
                    result.append(model)
        return result if len(result) > 1 else result[0] if result else None
    return None


ann=[]
for i in range(len(data)):
    ds=get_dataset(data['path'][i])
    deposit=data['deposit_type'][i]
    if ds=='ardf':
        if not isinstance(deposit,str):
            continue
        
        usgs_num=extract_model_numbers(deposit)
        if usgs_num is None:
            continue
        if isinstance(usgs_num,str):
            usgs_num=[usgs_num]
        
        dep_types=[ardf_mapping[x] for x in usgs_num if x in ardf_mapping]
        if len(dep_types)>0:
            print(i,dep_types)
            ann.append((i,dep_types))
    else:
        for r in rules:
            if r['dataset']==ds and (r['name']==deposit or r['name'] is None):
                ann.append((i,r['cmmi']))


data=pandas.read_csv('index.csv',low_memory=False)

deposit_type=list(data['deposit_type'])
for x in ann:
    deposit_type[x[0]]=json.dumps(x[1])

data['deposit_type']=deposit_type
data.to_csv('index2.csv')






















#  Generate run/train/eval splits
#  run: not redacted + redacted & has gt
#  eval: redacted, has gt, lat%5>2.5
#  eval2: not redacted, otherwise same as eval
#  train: redacted, has gt, lat%5<2.5 
#  train2: not redacted, otherwise same as eval
#Used in tasks
#  Select LLM: test on eval
#  Finetune LLM: train on train+train2, test on eval, eval2
#  Score aggregation: train on train+train2, test on eval, eval2
#  Area aggregation: train on train+train2 with run-train-train2-eval-eval2 as support, test on eval, eval2

df_out_cat=agent.run_df_agent("Load dataframes from `index.csv` and all `collection/usmin*.csv` (* is a wild card), and return the concatenated dataframe.")


df_out_cat=agent.run_df_agent("Load dataframes from `collection_v1.csv`, `collection/mrds.csv`, and return the concatenated dataframe.")

index=pandas.read_csv('index.csv',low_memory=False)
cmmi=pandas.read_csv('../science/dataset/taxonomy/cmmi_options_full_gpt4_number.csv',encoding='latin')
cmmi=list(cmmi['Deposit type'])

def deposit_type_matcher(ann,cmmi):
    if isinstance(ann,list):
        if ann in cmmi:
            for x in ann:
                label=deposit_type_matcher(x,cmmi)
                if not label is None:
                    return label
            
            return None
        else:
            return None
    elif isinstance(ann,str):
        if ann in cmmi:
            return ann
        else:
            return None
    else:
        return None

ann=[deposit_type_matcher(x,cmmi) for x in list(index['deposit_type'])]
ind_train=[i for i in range(len(ann)) if not ann[i] is None and index['latitude'][i]!=0 and index['latitude'][i]%5>2.5]
random.shuffle(ind_train)
ind_eval=[i for i in range(len(ann)) if not ann[i] is None and index['latitude'][i]!=0 and index['latitude'][i]%5<=2.5]
random.shuffle(ind_eval)




#  Create redacted json records: remove deposit type information from data
#  Used for split generation
root='dataset/mrdata_json'
root_out='dataset/mrdata_json_redacted'
df_out_redacted=agent.run_df_agent("I have provided a database of mineral deposit resources under directory {folder}. The database consists of folders of json files, where each json file is a database record as a nested dictionary-list hybrid. Please help me redact these records of mineral deposit type information, returning a record without them. Deposit type information is often found in fields `properties.deptype`, `properties.dep_type`, `properties.sub_type`, `properties.subtype`,`properties.mintype`, `deptype`, `dep_type`, `properties.deposit_model`,`min_type`, `mintype`, `depk10km`, `depk5km`, `dep10km`, `dep5km`, `properties.deposit_model.model_code`,`properties.dep10km`, `properties.dep5km`, `model_type`, `properties.model`, `properties.deptext`, `model`, `properties.model_code`, `properties.model_name`, `model_code`, `model_name`, `properties.model_type`, `deposits.dep_tp`, `properties.deposits.dep_tp`, `properties.type_detail`, `type_detail`, `properties.deptypea`, `properties.deptypeb`, `properties.deptypec`, `properties.comments`, `properties.prevtype`, `dep_subtype`, `properties.dep_model`,`properties.dpmd_nonm`. Because these json records may not have consistent fields, please check the record structure for each record. For nested fields, I've been writing them in the form of `parent.child.grandchild. ...`, etc. Some keys may point to list of dictionaries, so please go into each dictionary and try to find the keys in question. Please return a dataframe with 2 columns:\npath: path to the json file as a string, in the form of `mrdata_json/xxxx/yyyy.json`.\nredacted_json: redacted json record as a string, generated using json.dumps. ".format(folder=root))

path=[x.replace('mrdata_json','mrdata_json_redacted') for x in df_out_redacted['path']]
for i in range(len(df_out_redacted)):
    if not os.path.exists(path[i]):
        os.makedirs(os.path.dirname(path[i]), exist_ok=True)
        json.dump(json.loads(df_out_redacted['redacted_json'][i]),open(path[i],'w'))




#Generate splits
fnames_train=[index['path'][i] for i in ind_train]
labels_train=[ann[i] for i in ind_train]
fnames_train2=[index['path'][i].replace('mrdata_json','mrdata_json_redacted') for i in ind_train]
fnames_eval=[index['path'][i] for i in ind_eval]
labels_eval=[ann[i] for i in ind_eval]
fnames_eval2=[index['path'][i].replace('mrdata_json','mrdata_json_redacted') for i in ind_eval]
fnames_run=fnames_eval2+fnames_train2+list(index['path'])

#Save splits
def save_split(fnames,labels=None,fname_out=''):
    if labels is None:
        return pandas.DataFrame({'path':fnames}).to_csv(fname_out)
    else:
        return pandas.DataFrame({'path':fnames,'labels':labels}).to_csv(fname_out)

save_split(fnames_run,fname_out='splits/run.csv')
save_split(fnames_train,labels_train,fname_out='splits/train.csv')
save_split(fnames_train2,labels_train,fname_out='splits/train2.csv')
save_split(fnames_eval,labels_eval,fname_out='splits/eval.csv')
save_split(fnames_eval2,labels_eval,fname_out='splits/eval2.csv')



