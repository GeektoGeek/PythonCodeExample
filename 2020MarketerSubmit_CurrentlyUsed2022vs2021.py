
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

#### Remaining States for This Year

### Let's get the data first three months of the data i.e. this years producers for non-competitive producers

#2022
Submit2022 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Marketer2022AnnuityU.csv',encoding= 'iso-8859-1')

Submit2022.columns = Submit2022.columns.str.replace(' ', '')

con = sqlite3.connect("Submit2022.db")

Submit2022.to_sql("Submit2022", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName,MarketerContactCurrentIDCName as MarketerContactCurrentIDCName_2022, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount, Max(SubmitDate) as LastSubmitDate2022
      FROM Submit2022 group by MarketerContactIDText;"""
      
Submit2022GrBy =  pysqldf(q3)  

con = sqlite3.connect("Submit2022GrBy.db")

Submit2022GrBy.to_sql("Submit2022GrBy", con, if_exists='replace')

##2021

Submit2021 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Marketer2021Annuity.csv',encoding= 'iso-8859-1')

Submit2021.columns = Submit2021.columns.str.replace(' ', '')

con = sqlite3.connect("Submit2021.db")

Submit2021.to_sql("Submit2021", con, if_exists='replace')

Submit2021.info()

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, MarketerContactCurrentIDCName as MarketerContactCurrentIDCName_2021, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount, Max(SubmitDate) as LastSubmitDate2021
      FROM Submit2021 group by MarketerContactIDText;"""
      
Submit2021GrBy =  pysqldf(q3)  

export_csv = Submit2021GrBy.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\Submit2021GrBy.csv', index = None, header=True)


con = sqlite3.connect("Submit2021GrBy.db")

Submit2021GrBy.to_sql("Submit2021GrBy", con, if_exists='replace')

### Common Marketers

q6 = """SELECT a.* FROM Submit2022GrBy a INNER JOIN Submit2021GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommonMarketer =  pysqldf(q6)

con = sqlite3.connect("CommonMarketer.db")

CommonMarketer.to_sql("CommonMarketer", con, if_exists='replace')

export_csv = CommonMarketer.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\CommonMarketer.csv', index = None, header=True)

### Fallen Marketers

q7= """SELECT * FROM Submit2021GrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommonMarketer);"""

FallenMarketer =  pysqldf(q7)

export_csv = FallenMarketer.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\FallenMarketer.csv', index = None, header=True)

### New Marketers

q7= """SELECT * FROM Submit2022GrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommonMarketer);"""

NewMarketer2022 =  pysqldf(q7)

export_csv = NewMarketer2022.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\NewMarketer2022.csv', index = None, header=True)


### 2022 groupby Marketer

Submit2022['SubmitDate'] = pd.to_datetime(Submit2022['SubmitDate'])

Submit2022['date_month'] = Submit2022['SubmitDate'].dt.strftime('%Y-%m')

Submit2022_A= Submit2022.groupby(['date_month', 'MarketerContactIDText'], as_index=False)['SubmitAmount'].sum()

Submit2022_A= Submit2022_A.groupby(['MarketerContactIDText', 'date_month'], as_index=False)['SubmitAmount'].sum()

### Changing it to month over month using Pivot function

Dff1=  (Submit2022_A.pivot(index='MarketerContactIDText', columns='date_month', values='SubmitAmount').add_prefix('P_').reset_index())

#con = sqlite3.connect("Dff1.db")

#Dff1.to_sql("Dff1", con, if_exists='replace')

q6 = """SELECT a.*, b.MarketerName FROM Dff1 a INNER JOIN Submit2022GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Dff1_A =  pysqldf(q6)

export_csv = Dff1_A.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\Dff1_A.csv', index = None, header=True)


### 2021 groupby Marketer

Submit2021['SubmitDate'] = pd.to_datetime(Submit2021['SubmitDate'])

Submit2021['date_month'] = Submit2021['SubmitDate'].dt.strftime('%Y-%m')

Submit2021_A= Submit2021.groupby(['date_month', 'MarketerContactIDText'], as_index=False)['SubmitAmount'].sum()

Submit2021_A= Submit2021_A.groupby(['MarketerContactIDText', 'date_month'], as_index=False)['SubmitAmount'].sum()


### Changing it to month over month using Pivot function

Dff2=  (Submit2021_A.pivot(index='MarketerContactIDText', columns='date_month', values='SubmitAmount').add_prefix('P_').reset_index())

con = sqlite3.connect("Dff2.db")

Dff2.to_sql("Dff2", con, if_exists='replace')

q6 = """SELECT a.*, b.MarketerName FROM Dff2 a INNER JOIN Submit2021GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Dff2_A =  pysqldf(q6)

q6 = """SELECT a.*, b.* FROM Dff1_A  a INNER JOIN Dff2_A b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Dff1_2 =  pysqldf(q6)

export_csv = Dff1_2.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\Dff1_2.csv', index = None, header=True)

### Combine the Dff1 and Dff2 

#con = sqlite3.connect("Dff1.db")

#Dff1.to_sql("Dff1", con, if_exists='replace')

#con = sqlite3.connect("Dff2.db")

#Dff2.to_sql("Dff2", con, if_exists='replace')


q6 = """SELECT a.*, b.* FROM Dff1 a INNER JOIN Dff2 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Dff3 =  pysqldf(q6)

q6 = """SELECT a.*, b.MarketerName FROM Dff3 a INNER JOIN Submit2021GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Dff3_A =  pysqldf(q6)

export_csv = Dff3_A.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\Dff3_A.csv', index = None, header=True)



### Find the proportion of Product marketers


## SubmittedBusiness
Submit2022.info()

Submit2022_sub = Submit2022[["MarketerContactIDText", "SubmitAmount","Carrier"]]

values = ['Nationwide', 'Athene', 'NorthAmerican','AIG']

df_out = (Submit2022_sub.pivot_table(index='MarketerContactIDText', columns='Carrier',
                         values='SubmitAmount', aggfunc='sum',
                         fill_value=0)
 .reindex(columns=values, fill_value=0)
 .add_suffix('_sum'))

## App Count
Submit2022_sub1 = Submit2022[["MarketerContactIDText", "SubmitDate","Carrier"]]

values = ['Nationwide', 'Athene', 'NorthAmerican','AIG']

df_out1 = (Submit2022_sub1.pivot_table(index='MarketerContactIDText', columns='Carrier',
                         values='SubmitDate', aggfunc='count',
                         fill_value=0)
 .reindex(columns=values, fill_value=0)
 .add_suffix('_appcount'))

df_out1.info()

## Bring Submits and App Count together

q6 = """SELECT a.*, b.Nationwide_appcount, b.Athene_appcount, b.NorthAmerican_appcount, b.AIG_appcount FROM df_out a INNER JOIN df_out1 b on (a.MarketerContactIDText=b.MarketerContactIDText);"""      

df_out2 =  pysqldf(q6)

## Reset the index and make the marketerID a column in the dataframe

df_out2.reset_index(inplace=True)

df_out2.info()

## Append Marketer Name
q6 = """SELECT a.*, b.MarketerName FROM df_out2 a INNER JOIN Submit2022GrBy b on (a.MarketerContactIDText=b.MarketerContactIDText);"""      

df_out3 =  pysqldf(q6)

export_csv = df_out3.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\df_out3.csv', index = None, header=True)

####

Group3 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Group3.csv',encoding= 'iso-8859-1')

Group3.columns = Group3.columns.str.replace(' ', '')
con = sqlite3.connect("Group3.db")


Group3.to_sql("Group3", con, if_exists='replace')

Group3.info()

####

ActiveMarketerBreakdown09162022 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/ActiveMarketerBreakdown09162022.csv',encoding= 'iso-8859-1')

ActiveMarketerBreakdown09162022.columns = ActiveMarketerBreakdown09162022.columns.str.replace(' ', '')

con = sqlite3.connect("ActiveMarketerBreakdown09162022.db")

ActiveMarketerBreakdown09162022.to_sql("ActiveMarketerBreakdown09162022", con, if_exists='replace')

ActiveMarketerBreakdown09162022.info()

## Append Marketer Name
q6 = """SELECT a.*, b.MarketerName, b.Sum_2021, b.Sum_2022, b.Delta FROM ActiveMarketerBreakdown09162022 a INNER JOIN Group3 b on (a.MarketerContactIDText=b.MarketerContactIDText);"""      

df_out4 =  pysqldf(q6)

export_csv = df_out4.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\df_out4.csv', index = None, header=True)

####

Group1 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Group1.csv',encoding= 'iso-8859-1')

Group1.columns = Group1.columns.str.replace(' ', '')

con = sqlite3.connect("Group1.db")

Group1.to_sql("Group1", con, if_exists='replace')

Group1.info()

## Append Marketer Name
q6 = """SELECT a.*, b.MarketerName, b.Sum_2021, b.Sum_2022, b.Delta FROM ActiveMarketerBreakdown09162022 a INNER JOIN Group1 b on (a.MarketerContactIDText=b.MarketerContactIDText);"""      

df_out5 =  pysqldf(q6)

export_csv = df_out5.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\df_out5.csv', index = None, header=True)

###

### Nationwide Submit Advisors


### This needs matching via NPN, Email, FullName and Phone etc..

NewSubmitAdvisor2022 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Nationwide/NewSubmitAdvisor2022.csv',encoding= 'iso-8859-1')

NewSubmitAdvisor2022.columns = NewSubmitAdvisor2022.columns.str.replace(' ', '')

con = sqlite3.connect("NewSubmitAdvisor2022.db")

NewSubmitAdvisor2022.to_sql("NewSubmitAdvisor2022", con, if_exists='replace')

NewSubmitAdvisor2022.info()

q3  = """SELECT AdvisorContactAgentKey, AdvisorName1, AccountName,  count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount, Max(SubmitDate) as LastSubmitDate2021
      FROM NewSubmitAdvisor2022 group by AdvisorName1;"""
      
NewSubmitAdvisor2022GrBy =  pysqldf(q3)  

NewSubmitAdvisor2022GrBy.info()


NW2022Appointed = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Nationwide/NW2022Appointed.csv',encoding= 'iso-8859-1')

NW2022Appointed.columns = NW2022Appointed.columns.str.replace(' ', '')

con = sqlite3.connect("NW2022Appointed.db")

NW2022Appointed.to_sql("NW2022Appointed", con, if_exists='replace')

NW2022Appointed.info()

NewSubmitAdvisor2022GrBy.info()

## Append Marketer Name
q6 = """SELECT a.*, b.BeginDate FROM NewSubmitAdvisor2022GrBy a INNER JOIN NW2022Appointed b on ((a.AdvisorContactAgentKey=b.AdvisorKey) or (a.AdvisorName1=b.FullName));"""      

NewSubmitAdvisorApp =  pysqldf(q6)


### Amp Producers..

AmpAdvisors = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/AmpAdvisors.csv',encoding= 'iso-8859-1')

AmpAdvisors.columns = AmpAdvisors.columns.str.replace(' ', '')

con = sqlite3.connect("AmpAdvisors.db")

AmpAdvisors.to_sql("AmpAdvisors", con, if_exists='replace')

AmpAdvisors.info()

Submit2022 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmitAnalysis2022/Marketer2022AnnuityU.csv',encoding= 'iso-8859-1')

Submit2022.columns = Submit2022.columns.str.replace(' ', '')

con = sqlite3.connect("Submit2022.db")

Submit2022.to_sql("Submit2022", con, if_exists='replace')

Submit2022.info()

q3  = """SELECT AdvisorContactIDText, AdvisorName, AccountName, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitAmount, Max(SubmitDate) as LastSubmitDate2022
      FROM Submit2022 group by AdvisorContactIDText;"""
      
Submit2022GrBy_adv =  pysqldf(q3)  

con = sqlite3.connect("Submit2022GrBy_adv.db")

Submit2022GrBy_adv.to_sql("Submit2022GrBy_adv", con, if_exists='replace')

q6 = """SELECT a.* FROM Submit2022GrBy_adv a INNER JOIN AmpAdvisors b on (a.AdvisorContactIDText =b.ContactID18);"""      

CommonAdvisorwithAmp =  pysqldf(q6)

con = sqlite3.connect("CommonAdvisorwithAmp.db")

CommonAdvisorwithAmp.to_sql("CommonAdvisorwithAmp", con, if_exists='replace')

q9= """SELECT * FROM Submit2022GrBy_adv WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonAdvisorwithAmp);"""

NonAmpAdv=  pysqldf(q9)

export_csv = NonAmpAdv.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerSubmitAnalysis2022\NonAmpAdv.csv', index = None, header=True)

####