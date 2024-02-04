
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


Submit2021_Main = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2021Submit_Main.csv',encoding= 'iso-8859-1')


Submit2021_Main.columns = Submit2021_Main.columns.str.replace(' ', '')

Submit2021_Main.info()

Submit2021_Main['SubmitDate'] = pd.to_datetime(Submit2021_Main['SubmitDate'])

Submit2021_Main['date_month'] = Submit2021_Main['SubmitDate'].dt.strftime('%Y-%m')

Submit2021_Main1= Submit2021_Main.groupby(['date_month', 'MarketerName'], as_index=False)['SubmitAmount'].sum()

Submit2021_Main12= Submit2021_Main.groupby([ 'MarketerName', 'date_month'], as_index=False)['SubmitAmount'].sum()

### Changing it to month over month using Pivot function

DFF1=  (Submit2021_Main12.pivot(index='MarketerName', columns='date_month', values='SubmitAmount').add_prefix('P_').reset_index())

export_csv = Submit2021_Main1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2021_Main1.csv', index = None, header=True)

export_csv = DFF1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\DFF1.csv', index = None, header=True)


con = sqlite3.connect("Submit2021_Main.db")

Submit2021_Main.to_sql("Submit2021_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2021_Main group by MarketerContactIDText;"""
      
Submit2021_MainGrBy =  pysqldf(q3)  

### Lets roll it up by Week




Submits2020Pulled01052021 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2020SubmitsPulled01052021.csv',encoding= 'iso-8859-1')

Submits2020Pulled01052021.columns = Submits2020Pulled01052021.columns.str.replace(' ', '')


Submits2020Pulled01052021['SubmitDate'] = pd.to_datetime(Submits2020Pulled01052021['SubmitDate'])

Submits2020Pulled01052021['date_month'] = Submits2020Pulled01052021['SubmitDate'].dt.strftime('%Y-%m')

Submits2020Pulled01052021_Main1= Submits2020Pulled01052021.groupby(['date_month', 'MarketerName'], as_index=False)['SubmitAmount'].sum()

export_csv = Submits2020Pulled01052021_Main1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submits2020Pulled01052021_Main1.csv', index = None, header=True)


con = sqlite3.connect("Submits2020Pulled01052021.db")

Submits2020Pulled01052021.to_sql("Submits2020Pulled01052021", con, if_exists='replace')


q4  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2020, sum(SubmitAmount) as SubmittedBusiness_2020
      FROM Submits2020Pulled01052021 group by MarketerContactIDText;"""
      
Submits2020Pulled01052021GrBy =  pysqldf(q4) 


DFFF= pd.merge(Submit2021_MainGrBy, Submits2020Pulled01052021GrBy, on="MarketerContactIDText", how='outer')

DFFF.info()

### Common Marketers' present in 2020 and 2021

CommonMarketers_2020_2021 = DFFF[DFFF['MarketerName_x'] >= DFFF['MarketerName_y']]

export_csv = CommonMarketers_2020_2021.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommonMarketers_2020_2021.csv', index = None, header=True)


### New Marketers' in 2021
NewMarketer_2021 = DFFF[DFFF['MarketerName_y'].isnull()].drop("MarketerName_y", axis=1)



### Fallen Marketers' in 2021
FallenMarketer_2020 = DFFF[DFFF['MarketerName_x'].isnull()].drop("MarketerName_x", axis=1)


export_csv = FallenMarketer_2020.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\FallenMarketer_2020.csv', index = None, header=True)

### New Marketer in a True Sense

con = sqlite3.connect("NewMarketer_2021.db")

NewMarketer_2021.to_sql("NewMarketer_2021", con, if_exists='replace')

### All Marketers from 2016 until 2020

SubmitMarketer2016_2020 = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/SubmitMarketer2016_2020.csv',encoding= 'iso-8859-1')

SubmitMarketer2016_2020.columns = SubmitMarketer2016_2020.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitMarketer2016_2020.db")

SubmitMarketer2016_2020.to_sql("SubmitMarketer2016_2020", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount2016_2021, sum(SubmitAmount) as SubmittedBusiness2016_2021
      FROM SubmitMarketer2016_2020 group by MarketerContactIDText;"""
      
SubmitMarketer2016_2020GrBy =  pysqldf(q3)

con = sqlite3.connect("SubmitMarketer2016_2020GrBy.db")

SubmitMarketer2016_2020GrBy.to_sql("SubmitMarketer2016_2020GrBy", con, if_exists='replace')

q6 = """SELECT a.* FROM NewMarketer_2021 a INNER JOIN SubmitMarketer2016_2020GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommonMarketer =  pysqldf(q6)

con = sqlite3.connect("CommonMarketer.db")

CommonMarketer.to_sql("CommonMarketer", con, if_exists='replace')

export_csv = CommonMarketer.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommonMarketer.csv', index = None, header=True)


q7= """SELECT * FROM NewMarketer_2021 WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommonMarketer);"""

TrueNewMarketer =  pysqldf(q7)


q34  = """SELECT a.* FROM  NewMarketer_2021  a LEFT JOIN SubmitMarketer2016_2020GrBy b on (a.MarketerContactIDText =b.MarketerContactIDText) where b.MarketerContactIDText is NULL;"""

TrueNewMarketer2021_X =  pysqldf(q34) 

export_csv = TrueNewMarketer.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\TrueNewMarketer.csv', index = None, header=True)


### Where the Fallen Marketers Fall into (which group)

SubmitRank2020Marketers= pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/SubmitRank2020Marketers.csv',encoding= 'iso-8859-1')

SubmitRank2020Marketers.columns = SubmitRank2020Marketers.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitRank2020Marketers.db")

SubmitRank2020Marketers.to_sql("SubmitRank2020Marketers", con, if_exists='replace')

SubmitRank2020Marketers.info()

con = sqlite3.connect("FallenMarketer_2020.db")

FallenMarketer_2020.to_sql("FallenMarketer_2020", con, if_exists='replace')

q6 = """SELECT a.* FROM SubmitRank2020Marketers a INNER JOIN FallenMarketer_2020 b on (a.MarketerName =b.MarketerName_y);"""      

FallenMarketer_CumGroup =  pysqldf(q6)

out = FallenMarketer_CumGroup.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\FallenMarketer_CumGroup.csv', index = None, header=True)


### Where the New Marketers Fall into (which group)

SubmitRank2021Marketers= pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/SubmitRank2021Marketers.csv',encoding= 'iso-8859-1')

SubmitRank2021Marketers.columns = SubmitRank2021Marketers.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitRank2021Marketers.db")

SubmitRank2021Marketers.to_sql("SubmitRank2021Marketers", con, if_exists='replace')

SubmitRank2021Marketers.info()

TrueNewMarketer2021_X.info()

con = sqlite3.connect("TrueNewMarketer2021_X.db")

TrueNewMarketer2021_X.to_sql("TrueNewMarketer2021_X", con, if_exists='replace')

q6 = """SELECT a.* FROM SubmitRank2021Marketers a INNER JOIN TrueNewMarketer2021_X b on (a.MarketerName =b.MarketerName_x);"""      

NewTrue2021Marketer_CumGroup =  pysqldf(q6)

out = NewTrue2021Marketer_CumGroup.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\NewTrue2021Marketer_CumGroup.csv', index = None, header=True)



###
### Submit2021_Main
Submit2021_Main.info()

con = sqlite3.connect("Submit2021_Main.db")

Submit2021_Main.to_sql("Submit2021_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, ProductCode, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2021_Main group by MarketerContactIDText,ProductCode;"""
      
Submit2021_MainGrByProductCode =  pysqldf(q3) 

out = Submit2021_MainGrByProductCode.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2021_MainGrByProductCode.csv', index = None, header=True)

UniqueProduct_byMarketer= Submit2021_MainGrByProductCode.pivot_table(index="MarketerContactIDText", values=["ProductCode"], aggfunc=lambda x: ", ".join(x))
 
## change the pivot to a dataframe
UniqueProduct_byMarketer1 = UniqueProduct_byMarketer.reset_index()

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(distinct(ProductCode)) as Prod_Code, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2021_Main group by MarketerContactIDText;"""
      
Submit2021_MainGrByProductCode1 =  pysqldf(q3)  

out = Submit2021_MainGrByProductCode1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2021_MainGrByProductCode1.csv', index = None, header=True)


Submit2021_MainGrByProductCode1.info()

con = sqlite3.connect("Submit2021_MainGrByProductCode1.db")

Submit2021_MainGrByProductCode1.to_sql("Submit2021_MainGrByProductCode1", con, if_exists='replace')


con = sqlite3.connect("UniqueProduct_byMarketer1.db")

UniqueProduct_byMarketer1.to_sql("UniqueProduct_byMarketer1", con, if_exists='replace')

Submit2021_MainGrByProductCode1.info()

UniqueProduct_byMarketer1.info()

q6 = """SELECT a.*, b.ProductCode as Product_Code_Aggregae FROM Submit2021_MainGrByProductCode1 a INNER JOIN UniqueProduct_byMarketer1 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

NewTrue2021Marketer_CumGroup_Ch =  pysqldf(q6)


out = NewTrue2021Marketer_CumGroup_Ch.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\NewTrue2021Marketer_CumGroup_Ch.csv', index = None, header=True)

### 2020 Marketers 

Submits2020Pulled01052021.info()

con = sqlite3.connect("Submits2020Pulled01052021.db")

Submits2020Pulled01052021.to_sql("Submits2020Pulled01052021", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, ProductCode, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submits2020Pulled01052021 group by MarketerContactIDText,ProductCode;"""
      
Submits2020Pulled01052021GrByProductCode =  pysqldf(q3) 

out = Submits2020Pulled01052021GrByProductCode.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submits2020Pulled01052021GrByProductCode.csv', index = None, header=True)

UniqueProduct_byMarketer2020= Submits2020Pulled01052021GrByProductCode.pivot_table(index="MarketerContactIDText", values=["ProductCode"], aggfunc=lambda x: ", ".join(x))
 
## change the pivot to a dataframe
UniqueProduct_byMarketer1_2020 = UniqueProduct_byMarketer2020.reset_index()

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(distinct(ProductCode)) as Prod_Code, count(SubmitDate) as AppCount_2020, sum(SubmitAmount) as SubmittedBusiness_2020
      FROM Submits2020Pulled01052021 group by MarketerContactIDText;"""
      
Submits2020Pulled01052021GrByProductCode1 =  pysqldf(q3)  

out = Submits2020Pulled01052021GrByProductCode1.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submits2020Pulled01052021GrByProductCode1.csv', index = None, header=True)

Submits2020Pulled01052021GrByProductCode1.info()

con = sqlite3.connect("Submits2020Pulled01052021GrByProductCode1.db")

Submits2020Pulled01052021GrByProductCode1.to_sql("Submits2020Pulled01052021GrByProductCode1", con, if_exists='replace')


con = sqlite3.connect("UniqueProduct_byMarketer1_2020.db")

UniqueProduct_byMarketer1_2020.to_sql("UniqueProduct_byMarketer1_2020", con, if_exists='replace')

Submits2020Pulled01052021GrByProductCode1.info()

UniqueProduct_byMarketer1_2020.info()

q6 = """SELECT a.*, b.ProductCode as Product_Code_Aggregae FROM Submits2020Pulled01052021GrByProductCode1 a INNER JOIN UniqueProduct_byMarketer1_2020 b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

NewTrue2020Marketer_CumGroup_Ch =  pysqldf(q6)

out = NewTrue2020Marketer_CumGroup_Ch.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\NewTrue2020Marketer_CumGroup_Ch.csv', index = None, header=True)

### Lets see how the common group performed..

con = sqlite3.connect("NewTrue2021Marketer_CumGroup_Ch.db")

NewTrue2021Marketer_CumGroup_Ch.to_sql("NewTrue2021Marketer_CumGroup_Ch", con, if_exists='replace')

NewTrue2021Marketer_CumGroup_Ch.info()

con = sqlite3.connect("NewTrue2020Marketer_CumGroup_Ch.db")

NewTrue2020Marketer_CumGroup_Ch.to_sql("NewTrue2020Marketer_CumGroup_Ch", con, if_exists='replace')

NewTrue2020Marketer_CumGroup_Ch.info()

q6 = """SELECT a.MarketerContactIDText, a.MarketerName, a.AppCount_2021, a.SubmittedBusiness_2021, a.AccountName, a.Prod_Code, a.Product_Code_Aggregae, b.AppCount_2020 as AppCount_2020, b.SubmittedBusiness_2020 as SubmittedBusiness_2020, b.Prod_Code as Prod_Code_2020, b.Product_Code_Aggregae as Product_Code_Aggregae_2020  FROM NewTrue2021Marketer_CumGroup_Ch a INNER JOIN NewTrue2020Marketer_CumGroup_Ch b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

Common_2020_2021_x =  pysqldf(q6)

out = Common_2020_2021_x.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Common_2020_2021_x.csv', index = None, header=True)

### Lets bring the active marketers based on status from DW

ActiveMarketersStatusFromDWCleaned = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/ActiveMarketersStatusFromDWCleaned.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ActiveMarketersStatusFromDWCleaned.db")

ActiveMarketersStatusFromDWCleaned.to_sql("ActiveMarketersStatusFromDWCleaned", con, if_exists='replace')

ActiveMarketersStatusFromDWCleaned.info()

### For 2021 Marketers status

Submit2021_MainGrBy.info()

q6 = """SELECT a.*, b.status  FROM Submit2021_MainGrBy a INNER JOIN ActiveMarketersStatusFromDWCleaned b on (a.MarketerContactIDText =b.SFContactId);"""      

MarketersStatus2021 =  pysqldf(q6)

con = sqlite3.connect("MarketersStatus2021.db")

MarketersStatus2021.to_sql("MarketersStatus2021", con, if_exists='replace')

q9= """SELECT * FROM Submit2021_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM MarketersStatus2021);"""

SubmitMarketers2021Status_Inactive =  pysqldf(q9)

con = sqlite3.connect("SubmitMarketers2021Status_Inactive.db")

SubmitMarketers2021Status_Inactive.to_sql("SubmitMarketers2021Status_Inactive", con, if_exists='replace')

## crosschek this..

MarketesInactiveDW = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketesInactiveDW.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("MarketesInactiveDW.db")

MarketesInactiveDW.to_sql("MarketesInactiveDW", con, if_exists='replace')

q6 = """SELECT a.*, b.status  FROM SubmitMarketers2021Status_Inactive a INNER JOIN MarketesInactiveDW b on (a.MarketerContactIDText =b.SFContactId);"""      

MarketesInactiveDW_2021 =  pysqldf(q6)



out = MarketesInactiveDW_2021.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketesInactiveDW_2021.csv', index = None, header=True)


### For 2020 Marketers status

q6 = """SELECT a.*, b.status  FROM Submits2020Pulled01052021GrBy a INNER JOIN ActiveMarketersStatusFromDWCleaned b on (a.MarketerContactIDText =b.SFContactId);"""      

MarketersStatus2020 =  pysqldf(q6)

con = sqlite3.connect("MarketersStatus2020.db")

MarketersStatus2020.to_sql("MarketersStatus2020", con, if_exists='replace')

q9= """SELECT * FROM Submits2020Pulled01052021GrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM MarketersStatus2020);"""

SubmitMarketers2020Status_Inactive =  pysqldf(q9)

out = SubmitMarketers2020Status_Inactive.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\SubmitMarketers2020Status_Inactive.csv', index = None, header=True)


### 2019

Submit2019_Main = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketersSubmit2019.csv',encoding= 'iso-8859-1')

Submit2019_Main.columns = Submit2019_Main.columns.str.replace(' ', '')

Submit2019_Main.info()

con = sqlite3.connect("Submit2019_Main.db")

Submit2019_Main.to_sql("Submit2019_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2019, sum(SubmitAmount) as SubmittedBusiness_2019
      FROM Submit2019_Main group by MarketerContactIDText;"""
      
Submit2019_MainGrBy =  pysqldf(q3)  

out = Submit2019_MainGrBy.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2019_MainGrBy.csv', index = None, header=True)


con = sqlite3.connect("Submit2019_MainGrBy.db")

Submit2019_MainGrBy.to_sql("Submit2019_MainGrBy", con, if_exists='replace')

q6 = """SELECT a.* FROM Submits2020Pulled01052021GrBy a INNER JOIN Submit2019_MainGrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommomMarketer2019_2020 =  pysqldf(q6)

out = CommomMarketer2019_2020.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommomMarketer2019_2020.csv', index = None, header=True)



con = sqlite3.connect("CommomMarketer2019_2020.db")

CommomMarketer2019_2020.to_sql("CommomMarketer2019_2020", con, if_exists='replace')

q9= """SELECT * FROM Submit2019_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2019_2020);"""

MarketerFrom2019Fellin2020=  pysqldf(q9)

out = MarketerFrom2019Fellin2020.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerFrom2019Fellin2020.csv', index = None, header=True)

### How many new marketers added in 2020

### 2020 New Marketers added

q9= """SELECT * FROM Submits2020Pulled01052021GrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2019_2020);"""

MarketerNewAddedin2020=  pysqldf(q9)

#### 2018


Submit2018_Main = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/MarketerSubmit2018.csv',encoding= 'iso-8859-1')

Submit2018_Main.columns = Submit2018_Main.columns.str.replace(' ', '')

Submit2018_Main.info()

con = sqlite3.connect("Submit2018_Main.db")

Submit2018_Main.to_sql("Submit2018_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2018_Main group by MarketerContactIDText;"""
      
Submit2018_MainGrBy =  pysqldf(q3)  

out = Submit2018_MainGrBy.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2018_MainGrBy.csv', index = None, header=True)


con = sqlite3.connect("Submit2018_MainGrBy.db")

Submit2018_MainGrBy.to_sql("Submit2018_MainGrBy", con, if_exists='replace')

q6 = """SELECT a.* FROM Submit2019_MainGrBy a INNER JOIN Submit2018_MainGrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommomMarketer2018_2019 =  pysqldf(q6)

out = CommomMarketer2018_2019.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommomMarketer2018_2019.csv', index = None, header=True)



con = sqlite3.connect("CommomMarketer2018_2019.db")

CommomMarketer2018_2019.to_sql("CommomMarketer2018_2019", con, if_exists='replace')

q9= """SELECT * FROM Submit2018_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2018_2019);"""

MarketerFrom2018Fellin2019=  pysqldf(q9)

out = MarketerFrom2018Fellin2019.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerFrom2018Fellin2019.csv', index = None, header=True)


### How many new marketers added in 2020

### 2019 New Marketers added

q9= """SELECT * FROM Submit2018_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2018_2019);"""

MarketerNewAddedin2019=  pysqldf(q9)

out = MarketerNewAddedin2019.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerNewAddedin2019.csv', index = None, header=True)



#### 2017

Submit2017_Main = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2017Submit.csv',encoding= 'iso-8859-1')

Submit2017_Main.columns = Submit2017_Main.columns.str.replace(' ', '')

Submit2017_Main.info()

con = sqlite3.connect("Submit2017_Main.db")

Submit2017_Main.to_sql("Submit2017_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2017_Main group by MarketerContactIDText;"""
      
Submit2017_MainGrBy =  pysqldf(q3)  

out = Submit2017_MainGrBy.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2017_MainGrBy.csv', index = None, header=True)

con = sqlite3.connect("Submit2017_MainGrBy.db")

Submit2017_MainGrBy.to_sql("Submit2017_MainGrBy", con, if_exists='replace')

q6 = """SELECT a.* FROM Submit2017_MainGrBy a INNER JOIN Submit2018_MainGrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommomMarketer2017_2018 =  pysqldf(q6)

out = CommomMarketer2017_2018.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommomMarketer2017_2018.csv', index = None, header=True)

con = sqlite3.connect("CommomMarketer2017_2018.db")

CommomMarketer2017_2018.to_sql("CommomMarketer2017_2018", con, if_exists='replace')

q9= """SELECT * FROM Submit2017_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2017_2018);"""

MarketerFrom2017Fellin2018=  pysqldf(q9)

out = MarketerFrom2017Fellin2018.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerFrom2017Fellin2018.csv', index = None, header=True)

#### 2016

Submit2016_Main = pd.read_csv('C:/Users/test/Documents/MarketerSubmitAnalysis/2016Submit.csv',encoding= 'iso-8859-1')

Submit2016_Main.columns = Submit2016_Main.columns.str.replace(' ', '')

Submit2016_Main.info()

con = sqlite3.connect("Submit2016_Main.db")

Submit2016_Main.to_sql("Submit2016_Main", con, if_exists='replace')

q3  = """SELECT MarketerContactIDText, MarketerName, AccountName, count(SubmitDate) as AppCount_2021, sum(SubmitAmount) as SubmittedBusiness_2021
      FROM Submit2016_Main group by MarketerContactIDText;"""
      
Submit2016_MainGrBy =  pysqldf(q3)  

out = Submit2016_MainGrBy.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\Submit2016_MainGrBy.csv', index = None, header=True)

con = sqlite3.connect("Submit2016_MainGrBy.db")

Submit2016_MainGrBy.to_sql("Submit2016_MainGrBy", con, if_exists='replace')

q6 = """SELECT a.* FROM Submit2016_MainGrBy a INNER JOIN Submit2017_MainGrBy b on (a.MarketerContactIDText =b.MarketerContactIDText);"""      

CommomMarketer2016_2017 =  pysqldf(q6)

out = CommomMarketer2016_2017.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\CommomMarketer2016_2017.csv', index = None, header=True)

con = sqlite3.connect("CommomMarketer2016_2017.db")

CommomMarketer2016_2017.to_sql("CommomMarketer2016_2017", con, if_exists='replace')

q9= """SELECT * FROM Submit2016_MainGrBy WHERE MarketerContactIDText NOT IN (SELECT MarketerContactIDText FROM CommomMarketer2016_2017);"""

MarketerFrom2016Fellin2017=  pysqldf(q9)

out = MarketerFrom2016Fellin2017.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\MarketerFrom2016Fellin2017.csv', index = None, header=True)

### Number of Submit-Advisors under each submit marketer in a year

Submit2021_Main.info()

q3  = """SELECT count(distinct(AdvisorContactIDText)), MarketerName FROM Submit2021_Main group by MarketerContactIDText;"""
      
AdvisorCount_Marketer2021 =  pysqldf(q3)  

out = AdvisorCount_Marketer2021.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\AdvisorCount_Marketer2021.csv', index = None, header=True)


### Number of Submit-Advisors under each submit marketer in a year

Submits2020Pulled01052021.info()

q3  = """SELECT count(distinct(AdvisorContactIDText)), MarketerName FROM Submits2020Pulled01052021 group by MarketerContactIDText;"""
      
AdvisorCount_Marketer2020 =  pysqldf(q3)  

out = AdvisorCount_Marketer2020.to_csv (r'C:\Users\test\Documents\MarketerSubmitAnalysis\AdvisorCount_Marketer2020.csv', index = None, header=True)
