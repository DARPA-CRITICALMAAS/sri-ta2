import os
import pandas
import util.agent as agent
import random
import copy

if False:
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



#Load all annotations
index=pandas.read_csv('index/annotations/mrds_with_ca.csv',low_memory=False)
index={k:list(index[k]) for k in index}
index_redacted=copy.deepcopy(index)
index_redacted['path']=[fname.replace('mrdata_json','mrdata_json_redacted') for fname in index_redacted['path']]

ind_train=[i for i in range(len(index['latitude'])) if index['latitude'][i]!=0 and index['latitude'][i]%5>2.5]
ind_eval=[i for i in range(len(index['latitude'])) if index['latitude'][i]!=0 and index['latitude'][i]%5<=2.5]


data_train={k:[index[k][i] for i in ind_train] for k in index}
data_eval={k:[index[k][i] for i in ind_eval] for k in index}
data_train_redacted={k:[index_redacted[k][i] for i in ind_train] for k in index_redacted}
data_eval_redacted={k:[index_redacted[k][i] for i in ind_eval] for k in index_redacted}

data_train=pandas.DataFrame(data_train)
data_eval=pandas.DataFrame(data_eval)

data_train_redacted=pandas.DataFrame(data_train_redacted)
data_eval_redacted=pandas.DataFrame(data_eval_redacted)

data_train.to_csv('index/splits/train.csv')
data_eval.to_csv('index/splits/eval.csv')
data_train_redacted.to_csv('index/splits/train_redacted.csv')
data_eval_redacted.to_csv('index/splits/eval_redacted.csv')

pandas.concat([data_train,data_train_redacted],axis=0).to_csv('index/splits/train_joint.csv')
pandas.concat([data_eval,data_eval_redacted],axis=0).to_csv('index/splits/eval_joint.csv')

