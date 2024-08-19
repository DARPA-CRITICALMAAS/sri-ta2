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


#  Generate an index file with {'path','name','latitude','longitude','deposit_type'} columns for each database 
#  Used for crosswalking and split generation
root='dataset/mrdata_json'
root_out='index/sites'
root_links='index/mrdata_links'
for database in os.listdir(root):
    fname_out=os.path.join(root_out,'%s.csv'%database)
    if not os.path.exists(fname_out):
        df_collection_i=agent.run_df_agent("I have provided a database of mineral deposit resources under directory {folder}. The database consists of (folders of) json files, where each json file is a database record as a nested hybrid dictionary-list. Please help me index these records, returning a dataframe with 7 columns:\npath: partial path to the json file as a string, must be in the form of `{fname_pattern}`.\ndeposit_id: the id of the mineral deposit which is the name of the json file excluding path to the folder and without suffix '.json'. Can be a string or an integer.\nname: name of the mineral deposit site as a string. Often found under properties.depname, properties.dep_name, properties.altname, dep_name, depname, properties.name, properties.ftr_name, properties.site_name, name. If multiple conflicting values are present for each field, return a list of them.\nlongitude: longitude of the site as a floating point number. Must be present. Often found in longitude, geometry.coordinates, properties.londeg, properties.lonmin, properties.lonsec, properties.longitude, properties.geo_coordinates or geometry.coordinates so search exhaustively. Please convert deg-min-sec into a single degree as a float, must not be just the integer degree.\nlatitude: latitude of the site as a floating point number. Must be present. Often found in latitude, geometry.coordinates, properties.latdeg, properties.latmin, properties.latsec, properties.latitude, properties.geo_coordinates or geometry.coordinates so search exhaustively. Please convert deg-min-sec into a single degree as a float, must not be just the integer degree.\ndeposit_type: the deposit type information as a string. Deposit type information is often found in fields `properties.deptype`, `properties.dep_type`, `properties.sub_type`, `properties.subtype`,`properties.mintype`, `deptype`, `dep_type`, `properties.deposit_model`,`min_type`, `mintype`, `properties.deposit_model.model_code`, `model_type`, `properties.model`, `properties.deptext`, `model`, `properties.model_code`, `properties.model_name`, `model_code`, `model_name`, `properties.model_type`, `deposits.dep_tp`, `properties.deposits.dep_tp`, `properties.type_detail`, `type_detail`, `properties.deptypea`, `properties.deptypeb`, `properties.deptypec`, `properties.prevtype`, `dep_subtype`,`properties.dpmd_nonm` or `properties.dep_model.dpmd_nonm`. When multiple deposit type annotations are present, please keep a list of strings for all annotated deposit types.\nurl: http link to the record as a string that looks like `https://mrdata...`. Must be present. The url can be found by loading another pandas data frame at {mrdata_links}, and query using deposit_id as the key. So load that dataframe and join through the deposit_id column. The database is large, so use efficient algorithms for lookup. Consider the type ambiguity of deposit_id since it can be a string or an integer. \nThese json records may not have consistent fields, so please check the database field available in each record. For nested fields, I've been writing them in the form of `parent.child.grandchild. ...`, etc. Some keys may point to list of dictionaries, so please go into each dictionary and try to find the keys in question. Parallelize work on 16 threads to make things run faster.".format(folder=os.path.join(root,database),mrdata_links=os.path.join(root_links,'%s.csv'%database),fname_pattern=os.path.join('mrdata_json',database,'xxxx.json')))
        df_collection_i.to_csv(fname_out)


#Combine individual index files to a single index file for human annotation extraction and splitting
root='index/sites'
fname_out='index/all_sites.csv'
df_all=agent.run_df_agent("I have provided a number of csv files under directory {folder}. All csv files have the same fields: path, deposit_id, name, longitude, latitude, deposit_type and url. Please concatenate these csv data frames by alphabetical order based on their file name and return the concatenated data frame.".format(folder=root))
df_all.to_csv(fname_out)

