import os
import pandas
import util.agent as agent
import random

#  Generate an index file with {'path','name','latitude','longitude','deposit_type'} columns for each database 
#  Used for crosswalking and split generation
root='dataset/mrdata_json'
root_out='index/sites'
root_links='index/mrdata_links'
for database in os.listdir(root):
    fname_out=os.path.join(root_out,'%s.csv'%database)
    if not os.path.exists(fname_out):
        df_collection_i=agent.run_df_agent("I have provided a database of mineral deposit resources under directory {folder}. The database consists of (folders of) json files, where each json file is a database record as a nested hybrid dictionary-list. Please help me index these records, returning a dataframe with 7 columns:\npath: path to the json file as a string, in the form of `{fname_pattern}`. Do not add anything before it.\ndeposit_id: the id of the mineral deposit which is the name of the json file excluding path to the folder and without suffix '.json'. \nname: name of the mineral deposit site as a string. Often found under properties.depname, properties.dep_name, properties.altname, dep_name, depname, properties.name, properties.ftr_name, name. If multiple conflicting values are present for each field, return a list of them.\nlongitude: longitude of the site as a floating point number, often found in longitude, geometry.coordinates, properties.londeg, properties.lonmin, properties.lonsec, properties.longitude, properties.geo_coordinates, geometry.coordinates. Please convert deg-min-sec into a single degree as a float.\nlatitude: latitude of the site as a floating point number, often found in latitude, geometry.coordinates, properties.latdeg, properties.latmin, properties.latsec, properties.latitude, properties.geo_coordinates, geometry.coordinates. Please convert deg-min-sec into a single degree as a float.\ndeposit_type: the deposit type information as a string. Deposit type information is often found in fields `properties.deptype`, `properties.dep_type`, `properties.sub_type`, `properties.subtype`,`properties.mintype`, `deptype`, `dep_type`, `properties.deposit_model`,`min_type`, `mintype`, `properties.deposit_model.model_code`, `model_type`, `properties.model`, `properties.deptext`, `model`, `properties.model_code`, `properties.model_name`, `model_code`, `model_name`, `properties.model_type`, `deposits.dep_tp`, `properties.deposits.dep_tp`, `properties.type_detail`, `type_detail`, `properties.deptypea`, `properties.deptypeb`, `properties.deptypec`, `properties.prevtype`, `dep_subtype`,`properties.dpmd_nonm`,`properties.dep_model.dpmd_nonm`. When multiple deposit type annotations are present, please keep a list of strings for all annotated deposit types.\nurl: http link to the record as a string that looks like `https://mrdata...`. The url can be found by loading another pandas data frame at {mrdata_links}, and query using deposit_id as the key. So load that dataframe and join through the deposit_id column. Use fast algorithms, since the data frame is large. \nThese json records may not have consistent fields, so please check the database field available in each record. For nested fields, I've been writing them in the form of `parent.child.grandchild. ...`, etc. Some keys may point to list of dictionaries, so please go into each dictionary and try to find the keys in question. ".format(folder=os.path.join(root,database),mrdata_links=os.path.join(root_links,'%s.csv'%database),fname_pattern=os.path.join('mrdata_json',database,'xxxx.json')))
        df_collection_i.to_csv(fname_out)

#Combine individual index files to a single index file for human annotation extraction and splitting
root='index/sites'
fname_out='index/all_sites.csv'
df_all=agent.run_df_agent("I have provided a number of csv files under directory {folder}. All csv files have the same fields: path, deposit_id, name, longitude, latitude, deposit_type and url. Please concatenate these csv data frames by alphabetical order based on their file name and return the concatenated data frame.".format(folder=root))
df_all.to_csv(fname_out)

