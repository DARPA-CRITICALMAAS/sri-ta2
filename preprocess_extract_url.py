#Dependencies
#geopandas
#fastkml
#sqlalchemy

import os
import pandas
import util.agent as agent

import sys
import util.smartparse as smartparse
import util.session_manager as session_manager
default_params=smartparse.obj();
default_params.openai_api_key='your_key'
params = smartparse.parse()
params = smartparse.merge(params, default_params)
params.argv=sys.argv
agent.set_openai_key(params.openai_api_key)


idmap={'mrds': 'dep_id', 'usmin': 'site_id', 'porcu': 'rec_id', 'sedznpb': 'rec_id', 'sedexmvt': 'rec_id', 'sedcu': 'rec_id', 'nicrpge': 'site_id', 'carbonatite': 'rec_id', 'ree': 'rec_id', 'laterite': 'rec_id', 'ardf': 'ardf_num', 'sir20105090z': 'gmrap_id', 'pp577': 'id', 'ofr20051294': 'gid', 'pp1802': 'gid', 'vms': 'rec_id', 'asbestos': 'rec_id', 'phosphate': 'rec_no', 'ofr20151121': 'gid', 'podchrome': 'rec_id', 'sedau': 'recno', 'potash': 'rec_no'}

#porcu etc.
for dataset in ['porcu','sedznpb','sedexmvt','sedcu','carbonatite','ree','laterite','pp577','ofr20051294','vms','asbestos','phosphate','ofr20151121','podchrome','sedau','potash','nicrpge','ardf','mrds']:
    if not os.path.exists('index/mrdata_links/%s.csv'%dataset):
        id=idmap[dataset]
        url_pattern='https://mrdata.usgs.gov/to-json.php?db={dataset}&id={id}&labno={{{id}}}'.format(dataset=dataset,id=id)
        df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `dataset/mrdata_raw/{dataset}/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have a {id} field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 6 columns: \ndeposit_id: the record id in the database as in {id}\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.\nurl: an url to the deposit record. The url can be derived from the id field as {url_pattern}.".format(dataset=dataset,id=id,url_pattern=url_pattern),max_iter=20)
        df_out.to_csv('index/mrdata_links/%s.csv'%dataset)



#sir20105090z
#Slightly different URL pattern
if not os.path.exists('index/mrdata_links/sir20105090z.csv'):
    dataset='sir20105090z'
    id=idmap[dataset]
    url_pattern='https://mrdata.usgs.gov/{dataset}/json/{{{id}}}'.format(dataset=dataset,id=id)
    df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `dataset/mrdata_raw/{dataset}/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have a {id} field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 6 columns: \ndeposit_id: the record id in the database as in {id}\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.\nurl: an url to the deposit record. The url can be derived from the id field as {url_pattern}.".format(dataset=dataset,id=id,url_pattern=url_pattern),max_iter=20)
    
    df_out.to_csv('index/mrdata_links/sir20105090z.csv')

#usmin
usmin_commodity_mapping={}
usmin_commodity_mapping['USGS_Tungsten_ver2_CSV']='W'
usmin_commodity_mapping['USGS_REE_US_CSV']='REE'
usmin_commodity_mapping['USGS_Cobalt_US_CSV']='Co'
usmin_commodity_mapping['USGS_Indium_US_CSV']='In'
usmin_commodity_mapping['USGS_Graphite_US_CSV']='Graphite'
usmin_commodity_mapping['USGS_Gallium_US_CSV']='Ga'
usmin_commodity_mapping['USGS_Sn_US_CSV']='Sn'
usmin_commodity_mapping['USGS_Rhenium_US_CSV']='Re'
usmin_commodity_mapping['USGS_Germanium_US_CSV']='Ga'
usmin_commodity_mapping['USGS_Lithium_US_CSV']='Li'
usmin_commodity_mapping['USGS_Niobium_US_CSV']='Nb'
usmin_commodity_mapping['USGS_Tellurium_US_CSV']='Te'
usmin_commodity_mapping['USGS_Tantalum_US_CSV']='Ta'

#usmin does not have lat-long in the same file as deposit id & deposit name.
#For this database we request the core ID and name fields only.
for folder in usmin_commodity_mapping:
    commodity=usmin_commodity_mapping[folder]
    if not os.path.exists('index/mrdata_links/usmin_%s.csv'%commodity):
        url_pattern="https://mrdata.usgs.gov/deposit/json/{commodity}-{{{id}}}".format(commodity=commodity,id='deposit_id')
        df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `dataset/mrdata_raw/usmin/{folder}`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have a `Ftr_ID` or `Ftr_ID *`  field that looks something like `Mo00620` or `Mf00401`, which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 3 columns: \ndeposit_id: the record id (`Ftr_ID`) in the database\ndeposit_name: name of the mineral deposit. Leave it as NA if such annotations do not exist. To find the correct fields to look at, please print all fields that are present in those databases to console.\nurl: an url to the deposit record. The url can be derived from the Ftr_ID field as {url_pattern}".format(folder=folder,url_pattern=url_pattern),max_iter=20)
        df_out.to_csv('index/mrdata_links/usmin_%s.csv'%commodity)

