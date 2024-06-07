#Dependencies
#geopandas
#fastkml
#sqlalchemy

import os
import pandas
import agent

tmp=pandas.read_csv('mapping.csv')
idmap=dict(zip(list(tmp['db']),list(tmp['id'])))


#porcu
for dataset in ['porcu','sedznpb','sedexmvt','sedcu','carbonatite','ree','laterite','pp577','ofr20051294','vms','asbestos','phosphate','ofr20151121','podchrome','sedau','potash','nicrpge']:
    if not os.path.exists('mrdata_links/%s.csv'%dataset):
        id=idmap[dataset]
        df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `mrdata/{dataset}/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have a {id} field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 5 columns: \ndeposit_id: the record id in the database as in {id}\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.".format(dataset=dataset,id=id),max_iter=20)
        url=['https://mrdata.usgs.gov/to-json.php?db=%s&id=%s&labno=%s'%(dataset,id,str(i)) for i in df_out['deposit_id']]
        df_out['url']=url
        df_out.to_csv('mrdata_links/%s.csv'%dataset)



#sir20105090z
if not os.path.exists('mrdata_links/sir20105090z.csv'):
    df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `mrdata/sir20105090z/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have a gmrap_id field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 5 columns: \ndeposit_id: the record id in the database\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.",max_iter=20)
    url=['https://mrdata.usgs.gov/sir20105090z/json/%s'%str(i) for i in df_out['deposit_id']]
    df_out['url']=url
    df_out.to_csv('mrdata_links/sir20105090z.csv')

#ardf
if not os.path.exists('mrdata_links/ardf.csv'):
    df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `mrdata/ardf/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats. The records in those databases should have an `ardf_num` field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 5 columns: \ndeposit_id: the record id in the database\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.",max_iter=20)
    url=['https://mrdata.usgs.gov/to-json.php?db=ardf&id=ardf_num&labno=%s'%str(i) for i in df_out['deposit_id']]
    df_out['url']=url
    df_out.to_csv('mrdata_links/ardf.csv')


#usmin
if not os.path.exists('mrdata_links/usmin.csv'):
    df_out=agent.run_df_agent("I have provided a database of mineral deposit resources under directory `mrdata/usmin/`. The database consists of multiple geodatabase, gpkg, shp, kml or csv files with potentially different formats, distributed in multiple folders under `mrdata/usmin/`. The records in those databases should have an `Site_ID` field which corresponds to the record id. \nRequest: please extract all mineral deposits in my database that are available in those geodatabase, gpkg, shp, kml or csv files. Please return a data frame of 5 columns: \ndeposit_id: the record id (`Site_ID`) in the database\ndeposit_name: name of the mineral deposit\nlatitude\nlongitude\ndeposit_model (optional): mineral deposit model in text as if annotated in the database. Leave it as NA if such annotations do not exist.",max_iter=20)
    url=['https://mrdata.usgs.gov/deposit/json/%s'%str(i) for i in df_out['deposit_id']]
    df_out['url']=url
    df_out.to_csv('mrdata_links/usmin.csv')

