
import tensorflow as ts
import pandas as pd
import os
import seaborn as sns
import scipy.stats as stats
import math
import numpy as np
import pandasql as ps
import pandas as pd
import sqlite3
import pandas.io.sql as psql
import ast
import re
import datetime
import seaborn as sb
import sklearn
import camelot
from pandasql import sqldf
from pandasql import *
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
##import featuretools as ft
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ps = lambda q: sqldf(q, globals())
from scipy.stats import pearsonr
sns.set(style='white', font_scale=1.2)

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))

pysqldf = lambda q: sqldf(q, globals())

## Nationwide Data

Nationwidemaster11102021_share1 = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/NationwideEmailSegement02032022.csv',encoding= 'iso-8859-1')

Nationwidemaster11102021_share1.columns = Nationwidemaster11102021_share1.columns.str.replace(' ', '')

con = sqlite3.connect("Nationwidemaster11102021_share1.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Nationwidemaster11102021_share1.to_sql("Nationwidemaster11102021_share1", con, if_exists='replace')

Nationwidemaster11102021_share1.info()

## RocketReach with Agency Data

rocketreachcontactsAppointedAdvisrosWithAgencyName = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/rocketreachcontactsAppointedAdvisrosWithAgencyName.csv',encoding= 'iso-8859-1')

rocketreachcontactsAppointedAdvisrosWithAgencyName.columns = rocketreachcontactsAppointedAdvisrosWithAgencyName.columns.str.replace(' ', '')

con = sqlite3.connect("rocketreachcontactsAppointedAdvisrosWithAgencyName.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
rocketreachcontactsAppointedAdvisrosWithAgencyName.to_sql("rocketreachcontactsAppointedAdvisrosWithAgencyName", con, if_exists='replace')

## RocketReach Without Agency Data (Non Agency Data)

rocketreachcontactsAppointedAdvisrosWithAgencyNull = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/rocketreachcontactsAppointedAdvisrosWithAgencyNull.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("rocketreachcontactsAppointedAdvisrosWithAgencyNull.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
rocketreachcontactsAppointedAdvisrosWithAgencyNull.to_sql("rocketreachcontactsAppointedAdvisrosWithAgencyNull", con, if_exists='replace')

rocketreachcontactsAppointedAdvisrosWithAgencyNull.info()

q6 = """SELECT a.ContactID18, a.NationwideStreet, a.NationwideCity, a.NationwideState, a.HomeStreet, a.HomeCity, a.HomeState_Province,  a.HomeZip_PostalCode, b.* FROM Nationwidemaster11102021_share1 a INNER JOIN rocketreachcontactsAppointedAdvisrosWithAgencyName b on (a.ContactID18 =b.ID);"""      

Nationwidemaster11102021_share1Append =  pysqldf(q6)

Nationwidemaster11102021_share1.info()

q7 = """SELECT a.ContactID18, a.NPN, a.AgentCRD, a.NationwideStreet, a.NationwideCity, a.NationwideState, a.HomeStreet, a.HomeCity, a.HomeState_Province,  a.HomeZip_PostalCode, b.* FROM Nationwidemaster11102021_share1 a INNER JOIN rocketreachcontactsAppointedAdvisrosWithAgencyNull b on (a.ContactID18 =b.ID);"""      

Nationwidemaster11102021_share2Append =  pysqldf(q7)

### Do an Outer Join with Nationwide with Agency and Natiowide witiout Agency

DataAppendRR= pd.merge(Nationwidemaster11102021_share1Append, Nationwidemaster11102021_share2Append, on="ID", how='outer')

out = DataAppendRR.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\DataAppendRR.csv', index = None, header=True) 

con = sqlite3.connect("DataAppendRR.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR.to_sql("DataAppendRR", con, if_exists='replace')

### This is the common segment NW Appointed and Agency and Non Agency of Rocket Reach

q6 = """SELECT a.ContactID18, a.NationwideStreet, a.NationwideCity, a.NationwideState, a.HomeStreet, a.HomeCity, a.HomeState_Province, a.HomeZip_PostalCode, b.ID FROM Nationwidemaster11102021_share1 a INNER JOIN DataAppendRR b on (a.ContactID18 =b.ID);"""      

Common =  pysqldf(q6)

### This is the excludsive Nationwide no overlap possible with Rocker Reach

q9= """SELECT * FROM Nationwidemaster11102021_share1 WHERE ContactID18 NOT IN (SELECT ContactID18 FROM Common);"""

RaminingNWApp =  pysqldf(q9)

out = RaminingNWApp.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\RaminingNWApp.csv', index = None, header=True) 

RaminingNWApp['ID'] = RaminingNWApp['ContactID18']

### The full Nationwide Appointed Sample


DataAppendRR_Fin= pd.merge(DataAppendRR, RaminingNWApp, on="ID", how='outer')

out = DataAppendRR_Fin.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\DataAppendRR_Fin.csv', index = None, header=True) 

### Now bring back the clean data into this.. This is same as DataAppendRR_Fin with less variables

DataAppendRR_FinCleaned = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/DataAppendRR_FinCleaned.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("DataAppendRR_FinCleaned.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR_FinCleaned.to_sql("DataAppendRR_FinCleaned", con, if_exists='replace')

DataAppendRR_FinCleaned.info()

DataFr_Email21 = DataAppendRR_FinCleaned[["ID", "Email21"]]

DataFr_Email21= DataFr_Email21.loc[pd.notnull(DataFr_Email21.Email21)]

DataFr_Email21['Email'] = DataFr_Email21['Email21']

DataFr_Email21.info()



DataFr_Email22 = DataAppendRR_FinCleaned[["ID", "Email22"]]

DataFr_Email22= DataFr_Email22.loc[pd.notnull(DataFr_Email22.Email22)]

DataFr_Email22['Email'] = DataFr_Email22['Email22']

DataFr_Email22.info()

con = sqlite3.connect("DataFr_Email21.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email21.to_sql("DataFr_Email21", con, if_exists='replace')

con = sqlite3.connect("DataFr_Email22.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email22.to_sql("DataFr_Email22", con, if_exists='replace')

q3  = """SELECT a.* FROM DataFr_Email21 a INNER JOIN DataFr_Email22 b on (a.Email =b.Email);"""

Common21_22 =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_Email21 WHERE Email NOT IN (SELECT Email FROM Common21_22);"""
        
DataFr_Email21Exclusive=  pysqldf(p23)

Ch1= pd.merge(DataFr_Email22, DataFr_Email21Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ch1.db")

Ch1.to_sql("Ch1", con, if_exists='replace')

### Bring Email23

DataFr_Email23 = DataAppendRR_FinCleaned[["ID", "Email23"]]

DataFr_Email23= DataFr_Email23.loc[pd.notnull(DataFr_Email23.Email23)]

DataFr_Email23['Email'] = DataFr_Email23['Email23']

DataFr_Email23.info()

con = sqlite3.connect("DataFr_Email23.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email23.to_sql("DataFr_Email23", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch1 a INNER JOIN DataFr_Email23 b on (a.Email =b.Email);"""

ComCh1_23 =  pysqldf(q3) 

p24  = """SELECT * FROM DataFr_Email23 WHERE Email NOT IN (SELECT Email FROM ComCh1_23);"""
        
DataFr_Email23Exclusive=  pysqldf(p24)

Ch2= pd.merge(Ch1, DataFr_Email23Exclusive, on="Email", how='outer')



Ch2['ID1']= Ch2['ID_x']

Ch2['ID2']= Ch2['ID_y']

Ch2.info()

Ch2['ID1_x']= Ch2['ID_x']

Ch2['ID2_x']= Ch2['ID_y']


con = sqlite3.connect("Ch2.db")

Ch2.to_sql("Ch2", con, if_exists='replace')

DataFr_Email24 = DataAppendRR_FinCleaned[["ID", "Email24"]]

DataFr_Email24= DataFr_Email24.loc[pd.notnull(DataFr_Email24.Email24)]

DataFr_Email24['Email'] = DataFr_Email24['Email24']

DataFr_Email24.info()

con = sqlite3.connect("DataFr_Email24.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email24.to_sql("DataFr_Email24", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch2 a INNER JOIN DataFr_Email24 b on (a.Email =b.Email);"""

ComCh2_24 =  pysqldf(q3) 

p24  = """SELECT * FROM DataFr_Email24 WHERE Email NOT IN (SELECT Email FROM ComCh2_24);"""
        
DataFr_Email24Exclusive=  pysqldf(p24)

Ch3= pd.merge(Ch2, DataFr_Email24Exclusive, on="Email", how='outer')

out = Ch3.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ch3.csv', index = None, header=True) 

### Now bring back the clean data into this.. This is same as DataAppendRR_Fin with less variables

Ch3_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ch3_U.csv',encoding= 'iso-8859-1')

Ch3_U.info()

con = sqlite3.connect("Ch3_U.db")

Ch3_U.to_sql("Ch3_U", con, if_exists='replace')

DataFr_Email25 = DataAppendRR_FinCleaned[["ID", "Email25"]]

DataFr_Email25= DataFr_Email25.loc[pd.notnull(DataFr_Email25.Email25)]

DataFr_Email25['Email'] = DataFr_Email25['Email25']

DataFr_Email25.info()

con = sqlite3.connect("DataFr_Email25.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email25.to_sql("DataFr_Email25", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch3_U a INNER JOIN DataFr_Email25 b on (a.Email =b.Email);"""

ComCh3_25 =  pysqldf(q3) 

p25  = """SELECT * FROM DataFr_Email25 WHERE Email NOT IN (SELECT Email FROM ComCh3_25);"""
        
DataFr_Email25Exclusive=  pysqldf(p25)

Ch4= pd.merge(Ch3, DataFr_Email25Exclusive, on="Email", how='outer')

out = Ch4.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ch4.csv', index = None, header=True) 

Ch4_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ch4_U.csv',encoding= 'iso-8859-1')

DataAppendRR_FinCleanedU = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/DataAppendRR_FinCleanedU.csv',encoding= 'iso-8859-1')

DataAppendRR_FinCleanedU.info()

DataAppendRR1_Email1_y = DataAppendRR_FinCleanedU[["ID", "Email1_y"]]

DataAppendRR1_Email1_y= DataAppendRR1_Email1_y.loc[pd.notnull(DataAppendRR1_Email1_y.Email1_y)]

DataAppendRR1_Email1_y['Email'] = DataAppendRR1_Email1_y['Email1_y']

con = sqlite3.connect("DataAppendRR1_Email1_y.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR1_Email1_y.to_sql("DataAppendRR1_Email1_y", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch4_U a INNER JOIN DataAppendRR1_Email1_y b on (a.Email =b.Email);"""

ComCh4_Email1_y =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email1_y WHERE Email NOT IN (SELECT Email FROM ComCh4_Email1_y);"""
        
DataAppendRR1_Email1_yExclusive=  pysqldf(p25)

Ch4_1= pd.merge(Ch4_U, DataAppendRR1_Email1_yExclusive, on="Email", how='outer')

DataAppendRR1_Email3_y = DataAppendRR_FinCleanedU[["ID", "Email3_y"]]

DataAppendRR1_Email3_y= DataAppendRR1_Email3_y.loc[pd.notnull(DataAppendRR1_Email3_y.Email3_y)]

DataAppendRR1_Email3_y['Email'] = DataAppendRR1_Email3_y['Email3_y']

con = sqlite3.connect("DataAppendRR1_Email3_y.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR1_Email3_y.to_sql("DataAppendRR1_Email3_y", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch4_1 a INNER JOIN DataAppendRR1_Email3_y b on (a.Email =b.Email);"""

ComCh4_1Email3_y =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email3_y WHERE Email NOT IN (SELECT Email FROM ComCh4_1Email3_y);"""
        
DataAppendRR1_Email3_yExclusive=  pysqldf(p25)

Ch4_2= pd.merge(Ch4_1, DataAppendRR1_Email3_yExclusive, on="Email", how='outer')

DataAppendRR1_Email = DataAppendRR_FinCleanedU[["ID", "Email"]]

DataAppendRR1_Email= DataAppendRR1_Email.loc[pd.notnull(DataAppendRR1_Email.Email)]

DataAppendRR1_Email['Email'] = DataAppendRR1_Email['Email']

con = sqlite3.connect("DataAppendRR1_Email.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR1_Email.to_sql("DataAppendRR1_Email", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch4_2 a INNER JOIN DataAppendRR1_Email b on (a.Email =b.Email);"""

ComCh4_2Email =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email WHERE Email NOT IN (SELECT Email FROM ComCh4_2Email);"""
        
DataAppendRR1_EmailExclusive=  pysqldf(p25)

Ch4_3= pd.merge(Ch4_2, DataAppendRR1_EmailExclusive, on="Email", how='outer')

DataAppendRR1_AtheneEmail = DataAppendRR_FinCleanedU[["ID", "AtheneEmail"]]

DataAppendRR1_AtheneEmail= DataAppendRR1_AtheneEmail.loc[pd.notnull(DataAppendRR1_AtheneEmail.AtheneEmail)]

DataAppendRR1_AtheneEmail['Email'] = DataAppendRR1_AtheneEmail['AtheneEmail']

con = sqlite3.connect("DataAppendRR1_AtheneEmail.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataAppendRR1_AtheneEmail.to_sql("DataAppendRR1_AtheneEmail", con, if_exists='replace')

q3  = """SELECT a.* FROM Ch4_3 a INNER JOIN DataAppendRR1_AtheneEmail b on (a.Email =b.Email);"""

ComCh4_3Email =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_AtheneEmail WHERE Email NOT IN (SELECT Email FROM ComCh4_3Email);"""
        
DataAppendRR1_AtheneEmailExclusive=  pysqldf(p25)

Ch4_4= pd.merge(Ch4_3, DataAppendRR1_AtheneEmailExclusive, on="Email", how='outer')

out = Ch4_4.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ch4_4.csv', index = None, header=True) 

Ch4_4U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ch4_4U.csv',encoding= 'iso-8859-1')

###
con = sqlite3.connect("Ch4_4U.db")

Ch4_4U.to_sql("Ch4_4U", con, if_exists='replace')

###### Start with RocketReach End now


DataAppendRR1= DataAppendRR_FinCleaned

DataAppendRR1.info()

DataAppendRR1_Email1 = DataAppendRR1[["ID", "Email1"]]


DataAppendRR1_Email1= DataAppendRR1_Email1.loc[pd.notnull(DataAppendRR1_Email1.Email1)]

DataAppendRR1_Email1['Email'] = DataAppendRR1_Email1['Email1']

### DB

con = sqlite3.connect("DataAppendRR1_Email1.db")

DataAppendRR1_Email1.to_sql("DataAppendRR1_Email1", con, if_exists='replace')

DataAppendRR1_Email2 = DataAppendRR1[["ID", "Email2"]]

DataAppendRR1_Email2= DataAppendRR1_Email2.loc[pd.notnull(DataAppendRR1_Email2.Email2)]

DataAppendRR1_Email2['Email'] = DataAppendRR1_Email2['Email2']


### DB

con = sqlite3.connect("DataAppendRR1_Email2.db")

DataAppendRR1_Email2.to_sql("DataAppendRR1_Email2", con, if_exists='replace')

q3  = """SELECT a.* FROM DataAppendRR1_Email1 a INNER JOIN DataAppendRR1_Email2 b on (a.Email =b.Email);"""

ComEm1_Em2 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email2 WHERE Email NOT IN (SELECT Email FROM ComEm1_Em2);"""
        
Em2_Exclusive=  pysqldf(p25)

Ah1= pd.merge(DataAppendRR1_Email1, Em2_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah1.db")

Ah1.to_sql("Ah1", con, if_exists='replace')

### 
DataAppendRR1_Email3 = DataAppendRR1[["ID", "Email3"]]

DataAppendRR1_Email3= DataAppendRR1_Email3.loc[pd.notnull(DataAppendRR1_Email3.Email3)]

DataAppendRR1_Email3['Email'] = DataAppendRR1_Email3['Email3']

con = sqlite3.connect("DataAppendRR1_Email3.db")

DataAppendRR1_Email3.to_sql("DataAppendRR1_Email3", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah1 a INNER JOIN DataAppendRR1_Email3 b on (a.Email =b.Email);"""

ComAh1_Em3 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email3 WHERE Email NOT IN (SELECT Email FROM ComAh1_Em3);"""
        
Em3_Exclusive=  pysqldf(p25)

Ah2= pd.merge(Ah1, Em3_Exclusive, on="Email", how='outer')

Ah2.info()


con = sqlite3.connect("Ah2.db")

Ah2.to_sql("Ah2", con, if_exists='replace')

DataAppendRR1_Email4 = DataAppendRR1[["ID", "Email4"]]

DataAppendRR1_Email4= DataAppendRR1_Email4.loc[pd.notnull(DataAppendRR1_Email4.Email4)]

DataAppendRR1_Email4['Email'] = DataAppendRR1_Email4['Email4']

con = sqlite3.connect("DataAppendRR1_Email4.db")

DataAppendRR1_Email4.to_sql("DataAppendRR1_Email4", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah2 a INNER JOIN DataAppendRR1_Email4 b on (a.Email =b.Email);"""

ComAh2_Em4 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email4 WHERE Email NOT IN (SELECT Email FROM ComAh2_Em4);"""
        
Em4_Exclusive=  pysqldf(p25)

Ah3= pd.merge(Ah2, Em4_Exclusive, on="Email", how='outer')

out = Ah3.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah3.csv', index = None, header=True) 

Ah3_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah3_U.csv',encoding= 'iso-8859-1')


Ah3_U.info()

con = sqlite3.connect("Ah3_U.db")

Ah3_U.to_sql("Ah3_U", con, if_exists='replace')

DataAppendRR1_Email5 = DataAppendRR1[["ID", "Email5"]]

DataAppendRR1_Email5= DataAppendRR1_Email5.loc[pd.notnull(DataAppendRR1_Email5.Email5)]

DataAppendRR1_Email5['Email'] = DataAppendRR1_Email5['Email5']

### DB

con = sqlite3.connect("DataAppendRR1_Email5.db")

DataAppendRR1_Email5.to_sql("DataAppendRR1_Email5", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah3_U a INNER JOIN DataAppendRR1_Email5 b on (a.Email =b.Email);"""

ComAh3_Em5 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email5 WHERE Email NOT IN (SELECT Email FROM ComAh3_Em5);"""
        
Em5_Exclusive=  pysqldf(p25)

Ah4= pd.merge(Ah3_U, Em5_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah4.db")

Ah4.to_sql("Ah4", con, if_exists='replace')

DataAppendRR1_UnverifiedEmails  = DataAppendRR1[["ID", "UnverifiedEmails"]]

DataAppendRR1_UnverifiedEmails= DataAppendRR1_UnverifiedEmails.loc[pd.notnull(DataAppendRR1_UnverifiedEmails.UnverifiedEmails)]

DataAppendRR1_UnverifiedEmails['Email'] = DataAppendRR1_UnverifiedEmails['UnverifiedEmails']

### DB

con = sqlite3.connect("DataAppendRR1_UnverifiedEmails.db")

DataAppendRR1_UnverifiedEmails.to_sql("DataAppendRR1_UnverifiedEmails", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah4 a INNER JOIN DataAppendRR1_UnverifiedEmails b on (a.Email =b.Email);"""

ComAh4_Unv =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_UnverifiedEmails WHERE Email NOT IN (SELECT Email FROM ComAh4_Unv);"""
        
EmUnv_Exclusive=  pysqldf(p25)

Ah5= pd.merge(Ah4, EmUnv_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah5.db")

Ah5.to_sql("Ah5", con, if_exists='replace')

DataAppendRR1_RecommendedPersonalEmail = DataAppendRR1[["ID", "RecommendedPersonalEmail"]]

DataAppendRR1_RecommendedPersonalEmail= DataAppendRR1_RecommendedPersonalEmail.loc[pd.notnull(DataAppendRR1_RecommendedPersonalEmail.RecommendedPersonalEmail)]

DataAppendRR1_RecommendedPersonalEmail['Email'] = DataAppendRR1_RecommendedPersonalEmail['RecommendedPersonalEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedPersonalEmail.db")

DataAppendRR1_RecommendedPersonalEmail.to_sql("DataAppendRR1_RecommendedPersonalEmail", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah5 a INNER JOIN DataAppendRR1_RecommendedPersonalEmail b on (a.Email =b.Email);"""

ComAh5_RPE =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_RecommendedPersonalEmail WHERE Email NOT IN (SELECT Email FROM ComAh5_RPE);"""
        
EmRPF_Exclusive=  pysqldf(p25)

Ah6= pd.merge(Ah5, EmRPF_Exclusive, on="Email", how='outer')

out = Ah6.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah6.csv', index = None, header=True) 

Ah6_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah6_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Ah6_U.db")

Ah6_U.to_sql("Ah6_U", con, if_exists='replace')


DataAppendRR1_RecommendedEmail = DataAppendRR1[["ID", "RecommendedEmail"]]

DataAppendRR1_RecommendedEmail= DataAppendRR1_RecommendedEmail.loc[pd.notnull(DataAppendRR1_RecommendedEmail.RecommendedEmail)]

DataAppendRR1_RecommendedEmail['Email'] = DataAppendRR1_RecommendedEmail['RecommendedEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedEmail.db")

DataAppendRR1_RecommendedEmail.to_sql("DataAppendRR1_RecommendedEmail", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah6_U a INNER JOIN DataAppendRR1_RecommendedEmail b on (a.Email =b.Email);"""

ComAh6_RE =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_RecommendedEmail WHERE Email NOT IN (SELECT Email FROM ComAh6_RE);"""
        
EmRE_Exclusive=  pysqldf(p25)

Ah7= pd.merge(Ah6_U, EmRE_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah7.db")

Ah7.to_sql("Ah7", con, if_exists='replace')

DataAppendRR1_RecommendedWorkEmail = DataAppendRR1[["ID", "RecommendedWorkEmail"]]

DataAppendRR1_RecommendedWorkEmail= DataAppendRR1_RecommendedWorkEmail.loc[pd.notnull(DataAppendRR1_RecommendedWorkEmail.RecommendedWorkEmail)]

DataAppendRR1_RecommendedWorkEmail['Email'] = DataAppendRR1_RecommendedWorkEmail['RecommendedWorkEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedWorkEmail.db")

DataAppendRR1_RecommendedWorkEmail.to_sql("DataAppendRR1_RecommendedWorkEmail", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah7 a INNER JOIN DataAppendRR1_RecommendedWorkEmail b on (a.Email =b.Email);"""

ComAh7_RWE =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_RecommendedEmail WHERE Email NOT IN (SELECT Email FROM ComAh7_RWE);"""
        
EmRWE_Exclusive=  pysqldf(p25)

Ah8= pd.merge(Ah7, EmRWE_Exclusive, on="Email", how='outer')

DataAppendRR1.info()

con = sqlite3.connect("Ah8.db")

Ah8.to_sql("Ah8", con, if_exists='replace')

DataAppendRR1_PastEmails = DataAppendRR1[["ID", "PastEmails"]]

DataAppendRR1_PastEmails= DataAppendRR1_PastEmails.loc[pd.notnull(DataAppendRR1_PastEmails.PastEmails)]

DataAppendRR1_PastEmails['Email'] = DataAppendRR1_PastEmails['PastEmails']

### DB

con = sqlite3.connect("DataAppendRR1_PastEmails.db")

DataAppendRR1_PastEmails.to_sql("DataAppendRR1_PastEmails", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah8 a INNER JOIN DataAppendRR1_PastEmails b on (a.Email =b.Email);"""

ComAh8_PE =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_RecommendedEmail WHERE Email NOT IN (SELECT Email FROM ComAh8_PE);"""
        
EmPE_Exclusive=  pysqldf(p25)

Ah9= pd.merge(Ah8, EmPE_Exclusive, on="Email", how='outer')

out = Ah9.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah9.csv', index = None, header=True) 

Ah9_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah9_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Ah9_U.db")

Ah9_U.to_sql("Ah9_U", con, if_exists='replace')

### Try Email21 until Email25

DataAppendRR1_Email21 = DataAppendRR1[["ID", "Email21"]]

DataAppendRR1_Email21= DataAppendRR1_Email21.loc[pd.notnull(DataAppendRR1_Email21.Email21)]

DataAppendRR1_Email21['Email'] = DataAppendRR1_Email21['Email21']

### DB

con = sqlite3.connect("DataAppendRR1_Email21.db")

DataAppendRR1_Email21.to_sql("DataAppendRR1_Email21", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah9_U a INNER JOIN DataAppendRR1_Email21 b on (a.Email =b.Email);"""

ComAh9_Em21 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email21 WHERE Email NOT IN (SELECT Email FROM ComAh9_Em21);"""
        
Em21_Exclusive=  pysqldf(p25)

Ah10= pd.merge(Ah9_U, Em21_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah10.db")

Ah10.to_sql("Ah10", con, if_exists='replace')

DataAppendRR1_Email22 = DataAppendRR1[["ID", "Email22"]]

DataAppendRR1_Email22= DataAppendRR1_Email22.loc[pd.notnull(DataAppendRR1_Email22.Email22)]

DataAppendRR1_Email22['Email'] = DataAppendRR1_Email22['Email22']

con = sqlite3.connect("DataAppendRR1_Email22.db")

DataAppendRR1_Email22.to_sql("DataAppendRR1_Email22", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah10 a INNER JOIN DataAppendRR1_Email22 b on (a.Email =b.Email);"""

ComAh10_Em22 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email22 WHERE Email NOT IN (SELECT Email FROM ComAh10_Em22);"""
        
Em22_Exclusive=  pysqldf(p25)

Ah11= pd.merge(Ah10, Em22_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah11.db")

Ah11.to_sql("Ah11", con, if_exists='replace')

DataAppendRR1_Email23 = DataAppendRR1[["ID", "Email23"]]

DataAppendRR1_Email23= DataAppendRR1_Email23.loc[pd.notnull(DataAppendRR1_Email23.Email23)]

DataAppendRR1_Email23['Email'] = DataAppendRR1_Email23['Email23']

### DB

con = sqlite3.connect("DataAppendRR1_Email23.db")

DataAppendRR1_Email23.to_sql("DataAppendRR1_Email23", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah11 a INNER JOIN DataAppendRR1_Email23 b on (a.Email =b.Email);"""

ComAh11_Em23 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email23 WHERE Email NOT IN (SELECT Email FROM ComAh11_Em23);"""
        
Em23_Exclusive=  pysqldf(p25)


Ah12= pd.merge(Ah11, Em23_Exclusive, on="Email", how='outer')

out = Ah12.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah12.csv', index = None, header=True) 

Ah12_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah12_U.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("Ah12_U.db")

Ah12_U.to_sql("Ah12_U", con, if_exists='replace')

DataAppendRR1_Email24 = DataAppendRR1[["ID", "Email24"]]

DataAppendRR1_Email24= DataAppendRR1_Email24.loc[pd.notnull(DataAppendRR1_Email24.Email24)]

DataAppendRR1_Email24['Email'] = DataAppendRR1_Email24['Email24']

### DB

con = sqlite3.connect("DataAppendRR1_Email24.db")

DataAppendRR1_Email24.to_sql("DataAppendRR1_Email24", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah12_U a INNER JOIN DataAppendRR1_Email24 b on (a.Email =b.Email);"""

ComAh12_Em24 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email23 WHERE Email NOT IN (SELECT Email FROM ComAh12_Em24);"""
        
Em24_Exclusive=  pysqldf(p25)


Ah13= pd.merge(Ah12_U, Em24_Exclusive, on="Email", how='outer')

con = sqlite3.connect("Ah13.db")

Ah13.to_sql("Ah13", con, if_exists='replace')

DataAppendRR1_Email25 = DataAppendRR1[["ID", "Email25"]]

DataAppendRR1_Email25= DataAppendRR1_Email25.loc[pd.notnull(DataAppendRR1_Email25.Email25)]

DataAppendRR1_Email25['Email'] = DataAppendRR1_Email25['Email25']

### DB

con = sqlite3.connect("DataAppendRR1_Email25.db")

DataAppendRR1_Email25.to_sql("DataAppendRR1_Email25", con, if_exists='replace')

q3  = """SELECT a.* FROM Ah13 a INNER JOIN DataAppendRR1_Email25 b on (a.Email =b.Email);"""

ComAh13_Em25 =  pysqldf(q3) 

p25  = """SELECT * FROM DataAppendRR1_Email25 WHERE Email NOT IN (SELECT Email FROM ComAh13_Em25);"""
        
Em25_Exclusive=  pysqldf(p25)


Ah14= pd.merge(Ah13, Em25_Exclusive, on="Email", how='outer')


Ah15= pd.merge(Ah14, Ch4_U, on="Email", how='outer')

out = Ah15.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah15.csv', index = None, header=True) 

Ah15_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah15_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Ah15_U.db")

Ah15_U.to_sql("Ah15_U", con, if_exists='replace')


#######

Nationwidemaster11102021_share1.info()

NW_PrimaryEm = Nationwidemaster11102021_share1[["ContactID18", "Email"]]

NW_PrimaryEm_Email= NW_PrimaryEm.loc[pd.notnull(NW_PrimaryEm.Email)]

Ah16= pd.merge(NW_PrimaryEm_Email, Ah15_U, on="Email", how='outer')

out = Ah16.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah16.csv', index = None, header=True) 

Ah16_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah16_U.csv',encoding= 'iso-8859-1')

Ah17_A= pd.merge(Ch4_4U, Ah16_U, on="Email", how='outer')

out = Ah17_A.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Ah17_A.csv', index = None, header=True) 

Ah17_AU = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Ah17_AU.csv',encoding= 'iso-8859-1')



#### Phone number starts
Nationwidemaster11102021_share2 = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Nationwidemaster11102021_share2.csv',encoding= 'iso-8859-1')

Nationwidemaster11102021_share2.info()

Aff_NW = Nationwidemaster11102021_share2[["ContactID18", "Phone1Updated"]]

Aff_NW.info()

Aff_NW['Phone1Updated']=Aff_NW['Phone1Updated'].astype(float).round(0)


Aff_NW= Aff_NW.loc[pd.notnull(Aff_NW.Phone1Updated)]

Aff_NW['PhoneUpdated']= Aff_NW['Phone1Updated']


Aff_NW1 = Nationwidemaster11102021_share2[["ContactID18", "Phone2Updated"]]

Aff_NW1.info()

Aff_NW1= Aff_NW1.loc[pd.notnull(Aff_NW1.Phone2Updated)]

Aff_NW1['PhoneUpdated']= Aff_NW1['Phone2Updated']

Aff_NW2 = Nationwidemaster11102021_share2[["ContactID18", "Phone3Updated"]]

Aff_NW2= Aff_NW2.loc[pd.notnull(Aff_NW2.Phone3Updated)]


Aff_NW2['PhoneUpdated']= Aff_NW2['Phone3Updated']


con = sqlite3.connect("Aff_NW.db")

Aff_NW.to_sql("Aff_NW", con, if_exists='replace')


con = sqlite3.connect("Aff_NW1.db")

Aff_NW1.to_sql("Aff_NW1", con, if_exists='replace')

con = sqlite3.connect("Aff_NW2.db")

Aff_NW2.to_sql("Aff_NW2", con, if_exists='replace')

q34  = """SELECT a.* FROM Aff_NW a LEFT JOIN Aff_NW1 b on (a.Phone1Updated =b.Phone2Updated) where b.Phone2Updated is NULL;"""

Phone1Ex =  pysqldf(q34) 

Phase1= pd.merge(Aff_NW, Phone1Ex, on="PhoneUpdated", how='outer')

out = Phase1.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Phase1.csv', index = None, header=True) 

Phase1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Phase1_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Phase1_U.db")

Phase1_U.to_sql("Phase1_U", con, if_exists='replace')

q34  = """SELECT a.* FROM Phase1_U a LEFT JOIN Aff_NW2 b on (a.PhoneUpdated =b.Phone3Updated) where b.Phone3Updated is NULL;"""

Phone2Ex =  pysqldf(q34) 

Phase2= pd.merge(Phase1_U, Phone2Ex, on="PhoneUpdated", how='outer')

out = Phase2.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Phase2.csv', index = None, header=True) 

### Phase2 and NW_EmailFinal

Phase2_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Phase2_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Phase2_U.db")

Phase2_U.to_sql("Phase2_U", con, if_exists='replace')

DataAppendRRUpdate= pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/DataAppendRRUpdate.csv',encoding= 'iso-8859-1')
 
DataAppendRRUpdate.info()

Aa1 = DataAppendRRUpdate[["ID", "Phone_x"]]

Aa1= Aa1.loc[pd.notnull(Aa1.Phone_x)]

Aa1['PhoneUpdated']= Aa1['Phone_x']

con = sqlite3.connect("Aa1.db")

Aa1.to_sql("Aa1", con, if_exists='replace')

q34  = """SELECT a.* FROM Phase2_U a INNER JOIN Aa1 b on (a.PhoneUpdated =b.PhoneUpdated);"""

CmPh1 =  pysqldf(q34) 

p25  = """SELECT * FROM Aa1 WHERE PhoneUpdated NOT IN (SELECT PhoneUpdated FROM CmPh1);"""
        
Phone_x_Exclusive=  pysqldf(p25)

out = Phone_x_Exclusive.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Phone_x_Exclusive.csv', index = None, header=True) 

Phone_x_Exclusive_U= pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Phone_x_Exclusive_U.csv',encoding= 'iso-8859-1')
 
### Phase2 and NW_EmailFinal

Phase2_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Phase2_U.csv',encoding= 'iso-8859-1')

Phase2_U.info()

Phone_x_Exclusive_U.info()

Bb= pd.merge(Phase2_U, Phone_x_Exclusive_U, on="PhoneUpdated", how='outer')

out = Bb.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Bb.csv', index = None, header=True) 

Bb_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Bb_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Bb_U.db")

Bb_U.to_sql("Bb_U", con, if_exists='replace')

Aa2 = DataAppendRRUpdate[["ID", "OtherPhones"]]

Aa2= Aa2.loc[pd.notnull(Aa2.OtherPhones)]

Aa2['PhoneUpdated']= Aa2['OtherPhones']

con = sqlite3.connect("Aa2.db")

Aa2.to_sql("Aa2", con, if_exists='replace')

q34  = """SELECT a.* FROM Bb_U a INNER JOIN Aa2 b on (a.PhoneUpdated =b.PhoneUpdated);"""

CmOther =  pysqldf(q34) 

p25  = """SELECT * FROM Aa2 WHERE PhoneUpdated NOT IN (SELECT PhoneUpdated FROM CmOther);"""
        
Phone_Other_Exclusive=  pysqldf(p25)

Bb1= pd.merge(Bb_U, Phone_Other_Exclusive, on="ID", how='outer')

out = Bb1.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Bb1.csv', index = None, header=True) 

Bb1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Bb1_U.csv',encoding= 'iso-8859-1')


Bb1_U.info()

con = sqlite3.connect("Bb1_U.db")

Bb1_U.to_sql("Bb1_U", con, if_exists='replace')

Aa3 = DataAppendRRUpdate[["ID", "OfficePhone"]]

Aa3= Aa3.loc[pd.notnull(Aa3.OfficePhone)]

Aa3['PhoneUpdated']= Aa3['OfficePhone']

con = sqlite3.connect("Aa3.db")

Aa3.to_sql("Aa3", con, if_exists='replace')

q34  = """SELECT a.* FROM Bb1_U a INNER JOIN Aa3 b on (a.PhoneUpdated =b.PhoneUpdated);"""

CmOffice =  pysqldf(q34) 

p25  = """SELECT * FROM Aa3 WHERE PhoneUpdated NOT IN (SELECT PhoneUpdated FROM CmOffice);"""
        
Phone_Office_Exclusive=  pysqldf(p25)

Bb2= pd.merge(Bb1_U, Phone_Office_Exclusive, on="ID", how='outer')

out = Bb2.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Bb2.csv', index = None, header=True) 

Bb2_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Bb2_U.csv',encoding= 'iso-8859-1')

Ah177= pd.merge(Ah17_AU, Bb2_U, on="ID", how='outer')

Nationwidemaster11102021_share1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Nationwidemaster11102021_share3.csv',encoding= 'iso-8859-1')

Nationwidemaster11102021_share1_U.info()

con = sqlite3.connect("Nationwidemaster11102021_share1_U.db")

Nationwidemaster11102021_share1_U.to_sql("Nationwidemaster11102021_share1_U", con, if_exists='replace')

con = sqlite3.connect("Ah177.db")

Ah177.to_sql("Ah177", con, if_exists='replace')

Ah177.info()

Nationwidemaster11102021_share1_U.info()

q34  = """SELECT a.*, b.NPN, b.ID, b.FirstName, b.LastName, b.NationwideStreet, b.NationwideCity, b.NationwideState, b.NationwidePostalCode, b.HomeStreet, b.HomeCity, b.HomeState_Province, b.HomeZip_PostalCode FROM Ah177 a INNER JOIN Nationwidemaster11102021_share1_U b on ((a.ID =b.ID) or (a.Email= b.Email1) or(a.PhoneUpdated=b.Phone1Updated));"""

aabb1 =  pysqldf(q34) 

aabb = pd.merge(Ah17, Nationwidemaster11102021_share1_U, left_on="ID", right_on="ID", how="left", validate="m:1")

out = aabb.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Athene\aabb.csv', index = None, header=True) 

out = aabb1.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\FinalShare\aabb1.csv', index = None, header=True) 

DataAppendRR1.info()

con = sqlite3.connect("aabb.db")

aabb.to_sql("aabb", con, if_exists='replace')

aabb.info()

q34  = """SELECT  distinct(Email), distinct(PhoneUpdated), ID from aabb group by ID;"""

aabb_distinct =  pysqldf(q34)


out = DataAppendRR1.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\DataAppendRR1.csv', index = None, header=True) 
