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
from operator import attrgetter
from sklearn.metrics import silhouette_samples
import featuretools as ft
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

### Nationwide

NWLast2Years = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/SubmitReport01012019_02222021.csv',encoding= 'iso-8859-1')

NWLast2Years.columns = NWLast2Years.columns.str.replace(' ', '')

NWLast2Years.columns = NWLast2Years.columns.str.lstrip()

NWLast2Years.columns = NWLast2Years.columns.str.rstrip()

NWLast2Years.columns = NWLast2Years.columns.str.strip()

NWLast2Years['SubmitDate'] = pd.to_datetime(NWLast2Years['SubmitDate'])

con = sqlite3.connect("NWLast2Years.db")

NWLast2Years.to_sql("NWLast2Years", con, if_exists='replace')

NWLast2Years.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ProductType, ProductTerm, ProductCodeDetailed, ContractIssueState, ContactState, AdvisorContactHomeState_Province, AdvisorContactMailingAddress,AdvisorContactMailingStreet,AdvisorContactMailingCity,AdvisorContactMailingState_Province,AdvisorContactMailingZip_PostalCode,AdvisorContactCurrentNationwideIDC, max(SubmitDate) as LastSubmitDate  
      FROM NWLast2Years group by AdvisorContactIDText, AdvisorName;"""
      
NWLast2YearsGrBy1 =  pysqldf(q3)  

con = sqlite3.connect("NWLast2YearsGrBy.db")

NWLast2YearsGrBy.to_sql("NWLast2YearsGrBy", con, if_exists='replace')

NWLast3Months = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/SubmitReport11152020_02222021.csv',encoding= 'iso-8859-1')

NWLast3Months.columns = NWLast3Months.columns.str.replace(' ', '')

NWLast3Months.columns = NWLast3Months.columns.str.lstrip()

NWLast3Months.columns = NWLast3Months.columns.str.rstrip()

NWLast3Months.columns = NWLast3Months.columns.str.strip()

con = sqlite3.connect("NWLast3Months.db")

NWLast3Months.to_sql("NWLast3Months", con, if_exists='replace')

NWLast3Months.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM NWLast3Months group by AdvisorContactIDText, AdvisorName;"""
      
NWLast3MonthsGrBy =  pysqldf(q3)  

con = sqlite3.connect("NWLast3MonthsGrBy.db")

NWLast3MonthsGrBy.to_sql("NWLast3MonthsGrBy", con, if_exists='replace')

### Let's take the last 90 days data

vm1 = """SELECT a.* FROM NWLast2YearsGrBy1 a INNER JOIN NWLast3MonthsGrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

Commonlast90daysvs20192020 =  pysqldf(vm1)

p23  = """SELECT * FROM NWLast2YearsGrBy1 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Commonlast90daysvs20192020);"""
        
FallenAngels90daysvs20192020=  pysqldf(p23)

### Let's validate the appointed from the Email Data

NWAppointedAdv = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/EmailJasonForAppointedValidation.csv',encoding= 'iso-8859-1')

NWAppointedAdv.columns = NWAppointedAdv.columns.str.replace(' ', '')

NWAppointedAdv.columns = NWAppointedAdv.columns.str.lstrip()

NWAppointedAdv.columns = NWAppointedAdv.columns.str.rstrip()

NWAppointedAdv.columns = NWAppointedAdv.columns.str.strip()

NWAppointedAdv.info()

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table

con = sqlite3.connect("NWAppointedAdv.db")

NWAppointedAdv.to_sql("NWAppointedAdv", con, if_exists='replace')

qqq2  = """SELECT a.*,  b.NWLastSubmitDate, b.NWSalesYTD, b.NationwideStreet, b.NationwideCity, b.NationwideState, b.NationwidePostalCode, b.NationwideCountry  FROM FallenAngels90daysvs20192020 a INNER JOIN NWAppointedAdv b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
NWFinalFallenAngelsApp =  pysqldf(qqq2) 

NWFinalFallenAngelsApp.info()

Out4 = NWFinalFallenAngelsApp.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/NWFinalFallenAngelsApp.csv', index = None, header=True)

### Lets look into DW...

### Let's bring the paid data from 2018, 2019 and 2020

DWPaid2018_2019_2020= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/DWPaid2018_2019_2020.csv',encoding= 'iso-8859-1')

DWPaid2018_2019_2020.columns = DWPaid2018_2019_2020.columns.str.replace(' ', '')

DWPaid2018_2019_2020.columns = DWPaid2018_2019_2020.columns.str.lstrip()

DWPaid2018_2019_2020.columns = DWPaid2018_2019_2020.columns.str.rstrip()

DWPaid2018_2019_2020.columns = DWPaid2018_2019_2020.columns.str.strip()

DWPaid2018_2019_2020.info()

con = sqlite3.connect("DWPaid2018_2019_2020.db")

DWPaid2018_2019_2020.to_sql("DWPaid2018_2019_2020", con, if_exists='replace')

q3  = """SELECT distinct(AdvisorKey), FullName, count(PolicyNumber) as PolicyCnt, sum(TP) as PremiumAmt, ProductCode,ApplicationSignedJurisdictionStateCode,ContractOwnerResidenceJurisdictionStateCode,SFContactId  
      FROM DWPaid2018_2019_2020 group by AdvisorKey;"""
      
NWPaidLast3YearsGrBy =  pysqldf(q3)  

### Let's bring the paid data from 2019 and 2020

DWPaid11152020_02222021= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/DWPaid11152020_02222021.csv',encoding= 'iso-8859-1')

DWPaid11152020_02222021.columns = DWPaid11152020_02222021.columns.str.replace(' ', '')

DWPaid11152020_02222021.columns = DWPaid11152020_02222021.columns.str.lstrip()

DWPaid11152020_02222021.columns = DWPaid11152020_02222021.columns.str.rstrip()

DWPaid11152020_02222021.columns = DWPaid11152020_02222021.columns.str.strip()

DWPaid11152020_02222021.info()

con = sqlite3.connect("DWPaid11152020_02222021.db")

DWPaid11152020_02222021.to_sql("DWPaid11152020_02222021", con, if_exists='replace')

q3  = """SELECT distinct(AdvisorKey), FullName, count(PolicyNumber) as PolicyCnt, sum(TP) as PremiumAmt, ProductCode,ApplicationSignedJurisdictionStateCode,ContractOwnerResidenceJurisdictionStateCode,SFContactId  
      FROM DWPaid11152020_02222021 group by AdvisorKey, FullName,ProductCode,ApplicationSignedJurisdictionStateCode,ContractOwnerResidenceJurisdictionStateCode;"""
      
NWPaidLast3MonthsGrBy =  pysqldf(q3)  

### Let's take the last 90 days data

vm1 = """SELECT a.* FROM NWPaidLast3YearsGrBy a INNER JOIN NWPaidLast3MonthsGrBy b on ((a.AdvisorKey =b.AdvisorKey));"""      

CommonPaid90days =  pysqldf(vm1)

p23  = """SELECT * FROM NWPaidLast3YearsGrBy WHERE AdvisorKey NOT IN (SELECT AdvisorKey FROM CommonPaid90days);"""
        
FallenAngelsPaid90daysvs201820192020=  pysqldf(p23)

con = sqlite3.connect("FallenAngelsPaid90daysvs201820192020.db")

FallenAngelsPaid90daysvs201820192020.to_sql("FallenAngelsPaid90daysvs201820192020", con, if_exists='replace')


#### Now Bring the Submit Data from 1/1/2018 until 11/15/2020

SubmitReport01012018_11152020 = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/SubmitReport01012018_11152020.csv',encoding= 'iso-8859-1')

SubmitReport01012018_11152020.columns = SubmitReport01012018_11152020.columns.str.replace(' ', '')

SubmitReport01012018_11152020.columns = SubmitReport01012018_11152020.columns.str.lstrip()

SubmitReport01012018_11152020.columns = SubmitReport01012018_11152020.columns.str.rstrip()

SubmitReport01012018_11152020.columns = SubmitReport01012018_11152020.columns.str.strip()

SubmitReport01012018_11152020['SubmitDate'] = pd.to_datetime(SubmitReport01012018_11152020['SubmitDate'])

con = sqlite3.connect("SubmitReport01012018_11152020.db")

SubmitReport01012018_11152020.to_sql("SubmitReport01012018_11152020", con, if_exists='replace')

SubmitReport01012018_11152020.info()

q3  = """SELECT AdvisorContactAgentKey, AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastNWSubmitDateCal,ProductTerm,ProductCode_Detailed,AdvisorContactMailingAddress,AdvisorContactMailingStreet,AdvisorContactMailingCity,AdvisorContactMailingState_Province,AdvisorContactMailingZip_PostalCode    
      FROM SubmitReport01012018_11152020 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitReport01012018_11152020GrBy =  pysqldf(q3)  

SubmitReport01012018_11152020GrBy.info()

con = sqlite3.connect("SubmitReport01012018_11152020GrBy.db")

SubmitReport01012018_11152020GrBy.to_sql("SubmitReport01012018_11152020GrBy", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM FallenAngelsPaid90daysvs201820192020 a INNER JOIN SubmitReport01012018_11152020GrBy b on ((a.AdvisorKey =b.AdvisorContactAgentKey ));"""      

Check12 =  pysqldf(vm1)

con = sqlite3.connect("Check12.db")

Check12.to_sql("Check12", con, if_exists='replace')

Check12.info()

### Let's validate with Jason' Appointed Segment

qqq2  = """SELECT a.*,  b.NWLastSubmitDate, b.NWSalesYTD, b.NationwideStreet, b.NationwideCity, b.NationwideState, b.NationwidePostalCode, b.NationwideCountry  FROM Check12 a INNER JOIN NWAppointedAdv b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
NWFinalFallenAngelsPaidApp =  pysqldf(qqq2) 

Out4 = NWFinalFallenAngelsPaidApp.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/NWFinalFallenAngelsPaidApp.csv', index = None, header=True)

### Look for the States where NH 9 approved vs. not NH 12 approved..

StatesNH9Approved= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/StatesNH9Approved.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("StatesNH9Approved.db")

StatesNH9Approved.to_sql("StatesNH9Approved", con, if_exists='replace')

StatesNH12Approved= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/StatesNH12Approved.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("StatesNH12Approved.db")

StatesNH12Approved.to_sql("StatesNH12Approved", con, if_exists='replace')

vm1 = """SELECT a.* FROM StatesNH9Approved a INNER JOIN StatesNH12Approved b on ((a.State =b.State));"""      

CommonStatesNH9andNH12 =  pysqldf(vm1)

con = sqlite3.connect("CommonStatesNH9andNH12.db")

CommonStatesNH9andNH12.to_sql("CommonStatesNH9andNH12", con, if_exists='replace')

### States only NH9 approved..

p23  = """SELECT * FROM StatesNH9Approved WHERE State NOT IN (SELECT State FROM CommonStatesNH9andNH12);"""
        
OnlyNH9Approved=  pysqldf(p23)

con = sqlite3.connect("OnlyNH9Approved.db")

OnlyNH9Approved.to_sql("OnlyNH9Approved", con, if_exists='replace')

### Now check the overlap of OnlyNH9Approved and NWFinalFallenAngelsPaidApp

con = sqlite3.connect("NWFinalFallenAngelsPaidApp.db")

NWFinalFallenAngelsPaidApp.to_sql("NWFinalFallenAngelsPaidApp", con, if_exists='replace')

NWFinalFallenAngelsPaidApp.info()

### qqq2  = """SELECT a.*  FROM NWFinalFallenAngelsPaidApp a INNER JOIN OnlyNH9Approved b on (((a.ApplicationSignedJurisdictionStateCode=b.State) and (a.ContractOwnerResidenceJurisdictionStateCode=b.State)) and ((NationwideState=b.State) and (a.AdvisorContactMailingState_Province=b.State))) ;"""

qqq2  = """SELECT a.*  FROM NWFinalFallenAngelsPaidApp a INNER JOIN OnlyNH9Approved b on (((a.ApplicationSignedJurisdictionStateCode=b.State) and (a.ContractOwnerResidenceJurisdictionStateCode=b.State)) and ((NationwideState=b.State) and (a.AdvisorContactMailingState_Province=b.State))) ;"""
        
ListforNH9OnlyStates =  pysqldf(qqq2)  
 
ListforNH9OnlyStates =  pysqldf(qqq2) 

Out4 = ListforNH9OnlyStates.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/ListforNH9OnlyStates.csv', index = None, header=True)

### All States with NH 12 approved

qqq2  = """SELECT a.*  FROM NWFinalFallenAngelsPaidApp a INNER JOIN StatesNH12Approved b on (((a.ApplicationSignedJurisdictionStateCode=b.State) and (a.ContractOwnerResidenceJurisdictionStateCode=b.State)) and ((NationwideState=b.State) and (a.AdvisorContactMailingState_Province=b.State))) ;"""
        
ListforNH12Approved =  pysqldf(qqq2)  

Out4 = ListforNH12Approved.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/ListforNH12Approved.csv', index = None, header=True)


##qqq2  = """SELECT a.*  FROM NWFinalFallenAngelsPaidApp a INNER JOIN StatesNH12Approved b on (((a.ApplicationSignedJurisdictionStateCode=b.State) and (a.ContractOwnerResidenceJurisdictionStateCode=b.State)) and ((NationwideState=b.State))) ;"""
        
## ListforNH12Approved1 =  pysqldf(qqq2)  

##qqq2  = """SELECT a.*  FROM NWFinalFallenAngelsPaidApp a INNER JOIN StatesNH12Approved b on ((a.ApplicationSignedJurisdictionStateCode=b.State) and (NationwideState=b.State)) ;"""
        
## ListforNH12Approved2 =  pysqldf(qqq2)  


### Business Name Append

##Nh 9

ListforNH9OnlyStatesFinal_NShareFinal= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH9OnlyStatesFinal_NShareFinal.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ListforNH9OnlyStatesFinal_NShareFinal.db")

ListforNH9OnlyStatesFinal_NShareFinal.to_sql("ListforNH9OnlyStatesFinal_NShareFinal", con, if_exists='replace')

ListforNH9OnlyStatesFinal_NShareFinal.info()

NWAppointedAdv.info()

qqq2  = """SELECT a.*, b.CurrentNWIDCName FROM ListforNH9OnlyStatesFinal_NShareFinal a INNER JOIN NWAppointedAdv b on a.SFContactId=b.ContactID18;"""
        
ListforNH9OnlyStatesFinal_NShareFinalBusiApp =  pysqldf(qqq2)  

Out4 = ListforNH9OnlyStatesFinal_NShareFinalBusiApp.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH9OnlyStatesFinal_NShareFinalBusiApp.csv', index = None, header=True)

## NH 12

ListforNH12Approved_NShare_AAA= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH12Approved_NShare_AAA.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ListforNH12Approved_NShare_AAA.db")

ListforNH9OnlyStatesFinal_NShareFinal.to_sql("ListforNH12Approved_NShare_AAA", con, if_exists='replace')

ListforNH12Approved_NShare_AAA.info()

NWAppointedAdv.info()

qqq2  = """SELECT a.*, b.CurrentNWIDCName FROM ListforNH12Approved_NShare_AAA a INNER JOIN NWAppointedAdv b on a.AdvisorContactIDText=b.ContactID18;"""
        
ListforNH12Approved_NShare_AAABusiApp =  pysqldf(qqq2)  

Out4 = ListforNH12Approved_NShare_AAABusiApp.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH12Approved_NShare_AAABusiApp.csv', index = None, header=True)

#####

ListforNH12Approved_NShare_AAA= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH12Approved_NShare_AAA.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ListforNH12Approved_NShare_AAA.db")

ListforNH9OnlyStatesFinal_NShareFinal.to_sql("ListforNH12Approved_NShare_AAA", con, if_exists='replace')

ListforNH12Approved_NShare_AAA.info()

NWAppointedAdv.info()

qqq2  = """SELECT a.*, b.CurrentNWIDCName FROM ListforNH12Approved_NShare_AAA a INNER JOIN NWAppointedAdv b on a.AdvisorContactIDText=b.ContactID18;"""
        
ListforNH12Approved_NShare_AAABusiApp =  pysqldf(qqq2) 

#### Another Validation

List1234= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ListforNH12Approved_NShare_Final_Data.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List1234.db")

List1234.to_sql("List1234", con, if_exists='replace')

List1234.info()

####

SubmitReport11152020_02222021= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/SubmitReport11152020_02222021.csv',encoding= 'iso-8859-1')

SubmitReport11152020_02222021.columns = SubmitReport11152020_02222021.columns.str.replace(' ', '')

##SubmitReport11152020_02222021.columns = SubmitReport11152020_02222021.str.lstrip()

SubmitReport11152020_02222021.columns = SubmitReport11152020_02222021.columns.str.rstrip()

SubmitReport11152020_02222021.columns = SubmitReport11152020_02222021.columns.str.strip()

SubmitReport11152020_02222021['SubmitDate'] = pd.to_datetime(SubmitReport11152020_02222021['SubmitDate'])

con = sqlite3.connect("SubmitReport11152020_02222021.db")

SubmitReport11152020_02222021.to_sql("SubmitReport11152020_02222021", con, if_exists='replace')

SubmitReport11152020_02222021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate    
      FROM SubmitReport11152020_02222021 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitReport11152020_02222021GrBy =  pysqldf(q3)  

SubmitReport11152020_02222021GrBy.info()

qqq2  = """SELECT a.* FROM List1234 a INNER JOIN SubmitReport11152020_02222021GrBy b on a.AdvisorContactIDText=b.AdvisorContactIDText;"""
        
CommonList1=  pysqldf(qqq2) 

con = sqlite3.connect("CommonList1.db")

CommonList1.to_sql("CommonList1", con, if_exists='replace')

p23  = """SELECT * FROM List1234 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonList1);"""
        
RemainingNW=  pysqldf(p23)

Out4 = RemainingNW.to_csv(r'C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/RemainingNW.csv', index = None, header=True)


ABNH= pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SelectLaunch/FinalShare/ABNH.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ABNH.db")

ABNH.to_sql("ABNH", con, if_exists='replace')

qqq2  = """SELECT a.* FROM ABNH a INNER JOIN SubmitReport11152020_02222021GrBy b on a.SFContactId=b.AdvisorContactIDText;"""
        
CommonList11=  pysqldf(qqq2) 


 
