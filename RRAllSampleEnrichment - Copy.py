
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

q32  = """SELECT a.*, b.Email FROM DataFr_Email21 a LEFT JOIN DataFr_Email22 b on (a.Email =b.Email);"""

Passsss =  pysqldf(q32) 

### Validation in a single step
q33  = """SELECT a.* FROM DataFr_Email21 a LEFT JOIN DataFr_Email22 b on (a.Email =b.Email) where b.Email is NULL;"""

DataFr_Email21Exclusive1 =  pysqldf(q33) 

Check1= pd.merge(DataFr_Email21, DataFr_Email21Exclusive1, on="Email", how='outer')

out = Check1.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Check1.csv', index = None, header=True) 

Check1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Check1_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Check1_U.db")

Check1_U.to_sql("Check1_U", con, if_exists='replace')

### Bring Email23

DataFr_Email23 = DataAppendRR_FinCleaned[["ID", "Email23"]]

DataFr_Email23= DataFr_Email23.loc[pd.notnull(DataFr_Email23.Email23)]

DataFr_Email23['Email'] = DataFr_Email23['Email23']

DataFr_Email23.info()

con = sqlite3.connect("DataFr_Email23.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email23.to_sql("DataFr_Email23", con, if_exists='replace')

q34  = """SELECT a.* FROM Check1_U  a LEFT JOIN DataFr_Email23 b on (a.Email =b.Email) where b.Email is NULL;"""

DataFr_Email23Exclusive1 =  pysqldf(q34) 

Check2= pd.merge( Check1_U, DataFr_Email23Exclusive1, on="Email", how='outer')

### Need a cleanup of check2

con = sqlite3.connect("Check2.db")

Check2.to_sql("Check2", con, if_exists='replace')

q34  = """SELECT  distinct(Email), ID_x from Check2 group by ID_x;"""

Check2_AA =  pysqldf(q34)

Check2_AA1= pd.merge( Check1_U, Check2_AA, on="Email", how='outer')

out = Check2_AA1.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Check2_AA1.csv', index = None, header=True) 

Check2_AA1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Check2_AA1_U.csv',encoding= 'iso-8859-1')

### Bring Email24

DataFr_Email24 = DataAppendRR_FinCleaned[["ID", "Email24"]]

DataFr_Email24= DataFr_Email24.loc[pd.notnull(DataFr_Email24.Email24)]

DataFr_Email24['Email'] = DataFr_Email24['Email24']

DataFr_Email24.info()

con = sqlite3.connect("DataFr_Email24.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email24.to_sql("DataFr_Email24", con, if_exists='replace')

q34  = """SELECT a.* FROM Check2_AA1_U a LEFT JOIN DataFr_Email24 b on (a.Email =b.Email) where b.Email is NULL;"""

DataFr_Email24Exclusive1 =  pysqldf(q34) 

Check3= pd.merge(Check2_AA1_U, DataFr_Email24Exclusive1, on="Email", how='outer')

out = Check3.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Check3.csv', index = None, header=True) 

Check3.info()
Check3.to_sql("Check2", con, if_exists='replace')

q34  = """SELECT  distinct(Email), ID_x from Check2 group by ID_x;"""

Check3_AA =  pysqldf(q34)


### Bring Email25

DataFr_Email25 = DataAppendRR_FinCleaned[["ID", "Email25"]]

DataFr_Email25= DataFr_Email25.loc[pd.notnull(DataFr_Email25.Email25)]

DataFr_Email25['Email'] = DataFr_Email25['Email25']

DataFr_Email25.info()

con = sqlite3.connect("DataFr_Email25.db")
# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DataFr_Email25.to_sql("DataFr_Email25", con, if_exists='replace')

q34  = """SELECT a.* FROM Check3_AA  a LEFT JOIN DataFr_Email25 b on (a.Email =b.Email) where b.Email is NULL;"""

DataFr_Email25Exclusive1 =  pysqldf(q34) 

Check4= pd.merge( Check3_AA, DataFr_Email25Exclusive1, on="Email", how='outer')

Check4.info()


q34  = """SELECT  distinct(Email), ID_x_x from Check4 group by ID_x_x;"""

Check4_AA =  pysqldf(q34)

out = Check4_AA.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Check4_AA.csv', index = None, header=True) 


### Rocket Reach Starts Here

### Now same process need to be followed for Rocket Reach part

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

q34  = """SELECT a.* FROM DataAppendRR1_Email1 a LEFT JOIN DataAppendRR1_Email2 b on (a.Email =b.Email) where b.Email is NULL;"""

DataAppendRR1_Email1Exclusive =  pysqldf(q34) 

### Validation step

q34  = """SELECT a.* FROM DataAppendRR1_Email1 a INNER JOIN DataAppendRR1_Email2 b on (a.Email =b.Email);"""

Common_check =  pysqldf(q34) 

q9= """SELECT * FROM DataAppendRR1_Email2 WHERE Email NOT IN (SELECT Email FROM Common_check);"""

RaminingDataAppendRR1_Email2 =  pysqldf(q9)

Pass1_x_check= pd.merge(DataAppendRR1_Email1, RaminingDataAppendRR1_Email2, on="Email", how='outer')

out = Pass1_x_check.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass1_x_check.csv', index = None, header=True) 

Pass1_x_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass1_x_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass1_x_U.db")

Pass1_x_U.to_sql("Pass1_x_U", con, if_exists='replace')

### 
DataAppendRR1_Email3 = DataAppendRR1[["ID", "Email3"]]

DataAppendRR1_Email3= DataAppendRR1_Email3.loc[pd.notnull(DataAppendRR1_Email3.Email3)]

DataAppendRR1_Email3['Email'] = DataAppendRR1_Email3['Email3']

con = sqlite3.connect("DataAppendRR1_Email3.db")

DataAppendRR1_Email3.to_sql("DataAppendRR1_Email3", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass1_x_U a LEFT JOIN DataAppendRR1_Email3 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass1_x_UExclusive =  pysqldf(q34) 

Pass2= pd.merge(Pass1_x_U, Pass1_x_UExclusive, on="Email", how='outer')

out = Pass2.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass2.csv', index = None, header=True) 

Pass2_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass2_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass2_U.db")

Pass2_U.to_sql("Pass2_U", con, if_exists='replace')

### DB

DataAppendRR1_Email4 = DataAppendRR1[["ID", "Email4"]]

DataAppendRR1_Email4= DataAppendRR1_Email4.loc[pd.notnull(DataAppendRR1_Email4.Email4)]

DataAppendRR1_Email4['Email'] = DataAppendRR1_Email4['Email4']

con = sqlite3.connect("DataAppendRR1_Email4.db")

DataAppendRR1_Email4.to_sql("DataAppendRR1_Email4", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass2_U a LEFT JOIN DataAppendRR1_Email4 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass2_x_UExclusive =  pysqldf(q34) 

Pass3= pd.merge(Pass2_U, Pass2_x_UExclusive, on="Email", how='outer')

out = Pass3.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass3.csv', index = None, header=True) 

Pass3_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass3_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass3_U.db")

Pass3_U.to_sql("Pass3_U", con, if_exists='replace')

DataAppendRR1_Email5 = DataAppendRR1[["ID", "Email5"]]

DataAppendRR1_Email5= DataAppendRR1_Email5.loc[pd.notnull(DataAppendRR1_Email5.Email5)]

DataAppendRR1_Email5['Email'] = DataAppendRR1_Email5['Email5']

### DB

con = sqlite3.connect("DataAppendRR1_Email5.db")

DataAppendRR1_Email5.to_sql("DataAppendRR1_Email5", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass3_U a LEFT JOIN DataAppendRR1_Email5 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass3_x_UExclusive =  pysqldf(q34) 

Pass4= pd.merge(Pass3_U, Pass3_x_UExclusive, on="Email", how='outer')

out = Pass4.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass4.csv', index = None, header=True) 

Pass4_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass4_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass4_U.db")

Pass4_U.to_sql("Pass4_U", con, if_exists='replace')

DataAppendRR1_UnverifiedEmails  = DataAppendRR1[["ID", "UnverifiedEmails"]]

DataAppendRR1_RecommendedPersonalEmail = DataAppendRR1[["ID", "RecommendedPersonalEmail"]]

DataAppendRR1_RecommendedPersonalEmail= DataAppendRR1_RecommendedPersonalEmail.loc[pd.notnull(DataAppendRR1_RecommendedPersonalEmail.RecommendedPersonalEmail)]

DataAppendRR1_RecommendedPersonalEmail['Email'] = DataAppendRR1_RecommendedPersonalEmail['RecommendedPersonalEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedPersonalEmail.db")

DataAppendRR1_RecommendedPersonalEmail.to_sql("DataAppendRR1_RecommendedPersonalEmail", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass4_U a LEFT JOIN DataAppendRR1_RecommendedPersonalEmail b on (a.Email =b.Email) where b.Email is NULL;"""

Pass4_x_UExclusive =  pysqldf(q34) 

Pass5= pd.merge(Pass4_U, Pass4_x_UExclusive, on="Email", how='outer')

out = Pass5.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass5.csv', index = None, header=True) 

Pass5_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass5_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass5_U.db")

Pass5_U.to_sql("Pass5_U", con, if_exists='replace')

DataAppendRR1_RecommendedEmail = DataAppendRR1[["ID", "RecommendedEmail"]]

DataAppendRR1_RecommendedEmail= DataAppendRR1_RecommendedEmail.loc[pd.notnull(DataAppendRR1_RecommendedEmail.RecommendedEmail)]

DataAppendRR1_RecommendedEmail['Email'] = DataAppendRR1_RecommendedEmail['RecommendedEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedEmail.db")

DataAppendRR1_RecommendedEmail.to_sql("DataAppendRR1_RecommendedEmail", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass5_U a LEFT JOIN DataAppendRR1_RecommendedEmail b on (a.Email =b.Email) where b.Email is NULL;"""

Pass5_x_UExclusive =  pysqldf(q34) 

Pass6= pd.merge(Pass5_U, Pass5_x_UExclusive, on="Email", how='outer')

out = Pass6.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass6.csv', index = None, header=True) 

Pass6_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass6_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass6_U.db")

Pass6_U.to_sql("Pass6_U", con, if_exists='replace')


DataAppendRR1_RecommendedWorkEmail = DataAppendRR1[["ID", "RecommendedWorkEmail"]]

DataAppendRR1_RecommendedWorkEmail= DataAppendRR1_RecommendedWorkEmail.loc[pd.notnull(DataAppendRR1_RecommendedWorkEmail.RecommendedWorkEmail)]

DataAppendRR1_RecommendedWorkEmail['Email'] = DataAppendRR1_RecommendedWorkEmail['RecommendedWorkEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedWorkEmail.db")

DataAppendRR1_RecommendedWorkEmail.to_sql("DataAppendRR1_RecommendedWorkEmail", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass6_U a LEFT JOIN DataAppendRR1_RecommendedWorkEmail b on (a.Email =b.Email) where b.Email is NULL;"""

Pass6_x_UExclusive =  pysqldf(q34) 

Pass7= pd.merge(Pass6_U, Pass6_x_UExclusive, on="Email", how='outer')

Pass7.info()

q34  = """SELECT  distinct(Email), ID_x, ID_y from Pass7 group by ID_x;"""

Pass7_Update =  pysqldf(q34)

Pass7_Update1= pd.merge(Pass6_U, Pass7_x_UExclusive, on="Email", how='outer')

out = Pass7_Update1.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass7_Update1.csv', index = None, header=True) 

Pass7_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass7_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass7_U.db")

Pass7_U.to_sql("Pass7_U", con, if_exists='replace')

DataAppendRR1_RecommendedPersonalEmail = DataAppendRR1[["ID", "RecommendedPersonalEmail"]]

DataAppendRR1_RecommendedPersonalEmail= DataAppendRR1_RecommendedPersonalEmail.loc[pd.notnull(DataAppendRR1_RecommendedPersonalEmail.RecommendedPersonalEmail)]

DataAppendRR1_RecommendedPersonalEmail['Email'] = DataAppendRR1_RecommendedPersonalEmail['RecommendedPersonalEmail']

### DB

con = sqlite3.connect("DataAppendRR1_RecommendedPersonalEmail.db")

DataAppendRR1_RecommendedPersonalEmail.to_sql("DataAppendRR1_RecommendedPersonalEmail", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass7_U a LEFT JOIN DataAppendRR1_RecommendedPersonalEmail b on (a.Email =b.Email) where b.Email is NOT NULL;"""

Pass7_x_UExclusive =  pysqldf(q34) 

Pass8= pd.merge(Pass7_U, Pass7_x_UExclusive, on="Email", how='outer')


Pass8.info()

q34  = """SELECT  distinct(Email), ID_x, ID_y from Pass8 group by ID_x;"""

Pass8_Update =  pysqldf(q34)

Pass8_Update1= pd.merge(Pass7_U, Pass8_Update, on="Email", how='outer')

out = Pass8_Update1.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass8_Update1.csv', index = None, header=True) 

Pass8_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass8_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass8_U.db")

Pass8_U.to_sql("Pass8_U", con, if_exists='replace')


DataAppendRR1_Email21 = DataAppendRR1[["ID", "Email21"]]

DataAppendRR1_Email21= DataAppendRR1_Email21.loc[pd.notnull(DataAppendRR1_Email21.Email21)]

DataAppendRR1_Email21['Email'] = DataAppendRR1_Email21['Email21']

### DB

con = sqlite3.connect("DataAppendRR1_Email21.db")

DataAppendRR1_Email21.to_sql("DataAppendRR1_Email21", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass8_U a LEFT JOIN DataAppendRR1_Email21 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass8_x_UExclusive =  pysqldf(q34) 

Pass9= pd.merge(Pass8_U, Pass8_x_UExclusive, on="Email", how='outer')

out = Pass9.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass9.csv', index = None, header=True) 

Pass9_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass9_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass9_U.db")

Pass9_U.to_sql("Pass9_U", con, if_exists='replace')

### DB


DataAppendRR1_Email22 = DataAppendRR1[["ID", "Email22"]]

DataAppendRR1_Email22= DataAppendRR1_Email22.loc[pd.notnull(DataAppendRR1_Email22.Email22)]

DataAppendRR1_Email22['Email'] = DataAppendRR1_Email22['Email22']

con = sqlite3.connect("DataAppendRR1_Email22.db")

DataAppendRR1_Email22.to_sql("DataAppendRR1_Email22", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass9_U a LEFT JOIN DataAppendRR1_Email22 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass9_x_UExclusive =  pysqldf(q34) 

Pass10= pd.merge(Pass9_U, Pass9_x_UExclusive, on="Email", how='outer')

Pass10.info()

q34  = """SELECT  distinct(Email), ID_x_x from Pass10 group by ID_x_x;"""

Pass10_Update =  pysqldf(q34)

Pass10_Update1= pd.merge(Pass9_U, Pass10_Update, on="Email", how='outer')

out = Pass10_Update1.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass10_Update1.csv', index = None, header=True) 

Pass10_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass10_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass10_U.db")

Pass10_U.to_sql("Pass10_U", con, if_exists='replace')

### DB

DataAppendRR1_Email23 = DataAppendRR1[["ID", "Email23"]]

DataAppendRR1_Email23= DataAppendRR1_Email23.loc[pd.notnull(DataAppendRR1_Email23.Email23)]

DataAppendRR1_Email23['Email'] = DataAppendRR1_Email23['Email23']

### DB

con = sqlite3.connect("DataAppendRR1_Email23.db")

DataAppendRR1_Email23.to_sql("DataAppendRR1_Email23", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass10_U a LEFT JOIN DataAppendRR1_Email23 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass10_x_UExclusive =  pysqldf(q34) 

Pass11= pd.merge(Pass10_U, Pass10_x_UExclusive, on="Email", how='outer')

out = Pass11.to_csv (r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass11.csv', index = None, header=True) 

Pass11_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass11_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass11_U.db")

Pass11_U.to_sql("Pass11_U", con, if_exists='replace')

DataAppendRR1_Email24 = DataAppendRR1[["ID", "Email24"]]

DataAppendRR1_Email24= DataAppendRR1_Email24.loc[pd.notnull(DataAppendRR1_Email24.Email24)]

DataAppendRR1_Email24['Email'] = DataAppendRR1_Email24['Email24']

### DB

con = sqlite3.connect("DataAppendRR1_Email24.db")

DataAppendRR1_Email24.to_sql("DataAppendRR1_Email24", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass11_U a LEFT JOIN DataAppendRR1_Email24 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass11_x_UExclusive =  pysqldf(q34) 

Pass12= pd.merge(Pass11_U, Pass11_x_UExclusive, on="Email", how='outer')


out = Pass12.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass12.csv', index = None, header=True) 


Pass12_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass12_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Pass12_U.db")

Pass12_U.to_sql("Pass12_U", con, if_exists='replace')


DataAppendRR1_Email25 = DataAppendRR1[["ID", "Email25"]]

DataAppendRR1_Email25= DataAppendRR1_Email25.loc[pd.notnull(DataAppendRR1_Email25.Email25)]

DataAppendRR1_Email25['Email'] = DataAppendRR1_Email25['Email25']

### DB

con = sqlite3.connect("DataAppendRR1_Email25.db")

DataAppendRR1_Email25.to_sql("DataAppendRR1_Email25", con, if_exists='replace')

q34  = """SELECT a.* FROM Pass12_U a LEFT JOIN DataAppendRR1_Email25 b on (a.Email =b.Email) where b.Email is NULL;"""

Pass12_x_UExclusive =  pysqldf(q34) 

Pass14= pd.merge(Pass12_U, Pass12_x_UExclusive, on="Email", how='outer')

Pass14.info()

q34  = """SELECT  distinct(Email), ID_x from Pass14 group by ID_x;"""

Pass14_Update =  pysqldf(q34)

Pass14_Update1= pd.merge(Pass12_U, Pass14_Update, on="Email", how='outer')

out = Pass14_Update1.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Pass14_Update1.csv', index = None, header=True) 



### Lets use the cross polination of Check4_AA_U and Pass14_Update1

Pass14_Update_Final = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Pass14_Update_Final.csv',encoding= 'iso-8859-1')

Check4_AA_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Check4_AA_U.csv',encoding= 'iso-8859-1')

Final_DF= pd.merge(Pass14_Update_Final, Check4_AA_U, on="Email", how='outer')

Nationwidemaster11102021_share1.info()

NW_PrimaryEm = Nationwidemaster11102021_share1[["ContactID18", "Email"]]

NW_PrimaryEm_Email= NW_PrimaryEm.loc[pd.notnull(NW_PrimaryEm.Email)]

NW_EmailFinal= pd.merge(NW_PrimaryEm_Email, Final_DF, on="Email", how='outer')

out = NW_EmailFinal.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\NW_EmailFinal.csv', index = None, header=True) 

Nationwidemaster11102021_share1.info()

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


NW_EmailFinal['ID'] = NW_EmailFinal['ContactID18']

con = sqlite3.connect("Phase2.db")

Phase2_U.to_sql("Phase2_U", con, if_exists='replace')

con = sqlite3.connect("NW_EmailFinal.db")

NW_EmailFinal.to_sql("NW_EmailFinal", con, if_exists='replace')

Phase3= pd.merge(NW_EmailFinal, Phase2_U, on="ID", how='outer')

out = Phase3.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\Phase3.csv', index = None, header=True) 

Phase3_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Phase3_U.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Phase3_U.db")

Phase3_U.to_sql("Phase3_U", con, if_exists='replace')

Nationwidemaster11102021_share1_U = pd.read_csv('C:/Users/test/Documents/RocketReachTest_AllSample/RockerReachAllAppointed/Nationwidemaster11102021_share1_U.csv',encoding= 'iso-8859-1')

Nationwidemaster11102021_share1_U.info()

con = sqlite3.connect("Nationwidemaster11102021_share1_U.db")

Nationwidemaster11102021_share1_U.to_sql("Nationwidemaster11102021_share1_U", con, if_exists='replace')

Nationwidemaster11102021_share1_U.info()

q34  = """SELECT a.*, b.FirstName, b.LastName FROM Phase3_U a INNER JOIN Nationwidemaster11102021_share1_U b on (a.ID =b.ID) ;"""

aa1 =  pysqldf(q34) 

### One to Many Join

aa = pd.merge(Phase3_U, Nationwidemaster11102021_share1_U, left_on="ID", right_on="ID", how="left", validate="m:1")

aa2= aa[aa.FirstName.isnull()]

out = aa2.to_csv(r'C:\Users\test\Documents\RocketReachTest_AllSample\RockerReachAllAppointed\aa2.csv', index = None, header=True) 


con = sqlite3.connect("aa.db")

aa.to_sql("aa", con, if_exists='replace')

con = sqlite3.connect("aa1.db")

aa1.to_sql("aa1", con, if_exists='replace')






