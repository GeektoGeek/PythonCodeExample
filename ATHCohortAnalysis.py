
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
import datetime as dt
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
from datetime import datetime 
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

#### Let's bring 


####

### Athene Appointment data from 2019

Appointment2019 = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/CohortAnalysisDansMethod/Appointment2019Raw.csv',quotechar='"', encoding= 'iso-8859-1')

Submit2019 = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/CohortAnalysisDansMethod/SubmitAnalysis20192020.csv',quotechar='"', encoding= 'iso-8859-1')

Appointment2019.columns = Appointment2019.columns.str.replace(' ', '')

Appointment2019.columns = Appointment2019.columns.str.lstrip()

Appointment2019.columns = Appointment2019.columns.str.rstrip()

Appointment2019.columns = Appointment2019.columns.str.strip()

Appointment2019.info()



##Appointment2019.shape()

#### Remember that Appointment data is unique.
### It does not require any roll up

Appointment2019['CurrentAtheneAdvApptStartDate1'] = Appointment2019['CurrentAtheneAdvApptStartDate1'].astype('datetime64[ns]')

Appointment2019['AtheneAppMonthYear'] = Appointment2019['CurrentAtheneAdvApptStartDate1'].apply(lambda x: x.strftime('%Y-%m'))

AtheneAppJan2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-01')]
AtheneAppFeb2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-02')]
AtheneAppMar2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-03')]
AtheneAppApr2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-04')]
AtheneAppMay2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-05')]
AtheneAppJune2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-06')]
AtheneAppJuly2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-07')]
AtheneAppAug2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-08')]
AtheneAppSep2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-09')]
AtheneAppOct2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-10')]
AtheneAppNov2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-11')]
AtheneAppDec2019 = Appointment2019[(Appointment2019['AtheneAppMonthYear'] == '2019-12')]

con = sqlite3.connect("AtheneAppJan2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AtheneAppJan2019.to_sql("AtheneAppJan2019", con, if_exists='replace')

Submit2019.info()

##### Take the Submit data
Submit2019['SubmitDate1']= Submit2019['SubmitDate']

Submit2019['SubmitDate1'] = Submit2019['SubmitDate'].astype('datetime64[ns]')

Submit2019['SubmitDate1'] = Submit2019['SubmitDate1'].apply(lambda x: x.strftime('%Y-%m'))

Submit2019Jan2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-01')]
Submit2019Feb2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-02')]
Submit2019Mar2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-03')]
Submit2019Apr2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-04')]
Submit2019May2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-05')]
Submit2019June2019 =Submit2019[(Submit2019['SubmitDate1'] == '2019-06')]
Submit2019July2019 =Submit2019[(Submit2019['SubmitDate1'] == '2019-07')]
Submit2019Aug2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-08')]
Submit2019Sep2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-09')]
Submit2019Oct2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-10')]
Submit2019Nov2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-11')]
Submit2019Dec2019 = Submit2019[(Submit2019['SubmitDate1'] == '2019-12')]
Submit2020Jan2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-01')]
Submit2020Feb2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-02')]
Submit2020Mar2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-03')]
Submit2020Apr2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-04')]
Submit2020May2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-05')]
Submit2020Jun2020 = Submit2019[(Submit2019['SubmitDate1'] == '2020-06')]

Submit2019Jan2019.info()

con = sqlite3.connect("AtheneAppJan2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AtheneAppJan2019.to_sql("AtheneAppJan2019", con, if_exists='replace')

con = sqlite3.connect("Submit2019Jan2019.db")

Submit2019Jan2019.to_sql("Submit2019Jan2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Jan2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubJan2019 =  pysqldf(q2)  

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table

### Jan 2019 Appointed Advisors Feb Submit

con = sqlite3.connect("Submit2019Feb2019.db")

Submit2019Feb2019.to_sql("Submit2019Feb2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Feb2019 b on b.ContactID18=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubFeb2019 =  pysqldf(q2)  

Submit2019Feb2019.info()

Appointment2019.info()

### ### Jan 2019 Appointed Advisors Mar Submit

con = sqlite3.connect("Submit2019Mar2019.db")

Submit2019Mar2019.to_sql("Submit2019Mar2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Mar2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubMar2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors Apr Submit

con = sqlite3.connect("Submit2019Apr2019.db")

Submit2019Apr2019.to_sql("Submit2019Apr2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Apr2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubApr2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors May Submit

con = sqlite3.connect("Submit2019May2019.db")

Submit2019May2019.to_sql("Submit2019May2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019May2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubMay2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors June Submit

con = sqlite3.connect("Submit2019June2019.db")

Submit2019June2019.to_sql("Submit2019June2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019June2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubMay2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors June Submit

con = sqlite3.connect("Submit2019June2019.db")

Submit2019June2019.to_sql("Submit2019June2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019June2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubJune2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors July Submit

con = sqlite3.connect("Submit2019July2019.db")

Submit2019July2019.to_sql("Submit2019July2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019July2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubJuly2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors Aug Submit

con = sqlite3.connect("Submit2019Aug2019.db")

Submit2019Aug2019.to_sql("Submit2019Aug2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Aug2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubAug2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors Sep Submit

con = sqlite3.connect("Submit2019Sep2019.db")

Submit2019Sep2019.to_sql("Submit2019Sep2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Sep2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubSep2019 =  pysqldf(q2)  

### ### Jan 2019 Appointed Advisors Oct Submit

con = sqlite3.connect("Submit2019Oct2019.db")

Submit2019Oct2019.to_sql("Submit2019Oct2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Oct2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubOct2019 =  pysqldf(q2) 

### ### Jan 2019 Appointed Advisors Nov Submit

con = sqlite3.connect("Submit2019Nov2019.db")

Submit2019Nov2019.to_sql("Submit2019Nov2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Nov2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubNov2019 =  pysqldf(q2) 

### ### Jan 2019 Appointed Advisors Dec Submit

con = sqlite3.connect("Submit2019Dec2019.db")

Submit2019Dec2019.to_sql("Submit2019Dec2019", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2019Dec2019 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubDec2019 =  pysqldf(q2) 

### Let's take the Submit 2020 data


### ### Jan 2019 Appointed Advisors Jan 2020 Submit

con = sqlite3.connect("Submit2020Jan2020.db")

Submit2020Jan2020.to_sql("Submit2020Jan2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020Jan2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubJan2020 =  pysqldf(q2) 

### ### Jan 2019 Appointed Advisors Feb 2020 Submit

con = sqlite3.connect("Submit2020Feb2020.db")

Submit2020Feb2020.to_sql("Submit2020Feb2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020Feb2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubFeb2020 =  pysqldf(q2) 

### ### Jan 2019 Appointed Advisors Mar 2020 Submit

con = sqlite3.connect("Submit2020Mar2020.db")

Submit2020Mar2020.to_sql("Submit2020Mar2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020Mar2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubMar2020 =  pysqldf(q2) 

### ### Jan 2019 Appointed Advisors Apr 2020 Submit

con = sqlite3.connect("Submit2020Apr2020.db")

Submit2020Apr2020.to_sql("Submit2020Apr2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020Apr2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubApr2020 =  pysqldf(q2) 


### ### Jan 2019 Appointed Advisors May 2020 Submit

con = sqlite3.connect("Submit2020May2020.db")

Submit2020May2020.to_sql("Submit2020May2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020May2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubMay2020 =  pysqldf(q2) 


### ### Jan 2019 Appointed Advisors June 2020 Submit

con = sqlite3.connect("Submit2020Jun2020.db")

Submit2020Jun2020.to_sql("Submit2020Jun2020", con, if_exists='replace')

q2  = """SELECT b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmount FROM AtheneAppJan2019 a INNER JOIN Submit2020Jun2020 b on b.AdvisorContactIDText=a.ContactID18 group by AdvisorContactIDText """

AdvAppSubJune2020 =  pysqldf(q2) 


## AppSubmit_Jan2019 = pd.merge(left=AtheneAppJan2019, right=Submit2019Jan2019, on=['ContactID18','AdvisorContactIDText'], how= 'inner')


df['OrderPeriod'] = df['SubmitDate'].apply(lambda x: x.strftime('%Y-%m'))

Submit2019.columns = Submit2019.columns.str.replace(' ', '')

Submit2019.columns = Submit2019.columns.str.lstrip()

Submit2019.columns = Submit2019.columns.str.rstrip()

Submit2019.columns = Submit2019.columns.str.strip()

SubmitRaw = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/CohortAnalysisDansMethod/SubmitRaw.csv',quotechar='"', encoding= 'iso-8859-1')

SubmitRaw.columns = SubmitRaw.columns.str.replace(' ', '')

SubmitRaw.columns = SubmitRaw.columns.str.lstrip()

SubmitRaw.columns = SubmitRaw.columns.str.rstrip()

SubmitRaw.columns = SubmitRaw.columns.str.strip()

con = sqlite3.connect("SubmitRaw.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitRaw.to_sql("SubmitRaw", con, if_exists='replace')

SubmitRaw.info()

SubmitRaw['SubmitDate'] = SubmitRaw['SubmitDate'].astype('datetime64[ns]')

q2  = """SELECT AdvisorContactIDText,  AdvisorName, ProductCode, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmount, SubmitDate, min(SubmitDate) as FirstSubmitDate, max(SubmitDate) as LastSubmitDate,  AdvisorContactFirstAtheneAdvApptStartDate2 from SubmitRaw group by AdvisorContactIDText;"""
      
SubmitRawGrBy =  pysqldf(q2)  


con = sqlite3.connect("SubmitRawGrBy.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitRawGrBy.to_sql("SubmitRawGrBy", con, if_exists='replace')


AppointmentRaw = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07182020/AppointmentRaw.csv',quotechar='"',encoding= 'iso-8859-1')

AppointmentRaw.columns = AppointmentRaw.columns.str.replace(' ', '')

AppointmentRaw.columns = AppointmentRaw.columns.str.lstrip()

AppointmentRaw.columns = AppointmentRaw.columns.str.rstrip()

AppointmentRaw.columns = AppointmentRaw.columns.str.strip()

AppointmentRaw['CurrentAtheneAdvApptStartDate1'] = AppointmentRaw['CurrentAtheneAdvApptStartDate1'].astype('datetime64[ns]')

AppointmentRaw['FirstAtheneAdvApptStartDate2 '] = AppointmentRaw['FirstAtheneAdvApptStartDate2'].astype('datetime64[ns]')

con = sqlite3.connect("AppointmentRaw.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AppointmentRaw.to_sql("AppointmentRaw", con, if_exists='replace')

AppointmentRaw.info()

q3  = """SELECT a.*, b.ContactID18, b.CurrentAtheneAdvApptStartDate1, b.FirstAtheneAdvApptStartDate2 from SubmitRawGrBy a LEFT JOIN AppointmentRaw b on a.AdvisorContactIDText =b.ContactID18  group by AdvisorContactIDText;"""
      
SubmitApp2019_2020 =  pysqldf(q3) 

Out4 = SubmitApp2019_2020.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\CohortAnalysis\SubmitApp2019_2020.csv', index = None, header=True) 

##### Let's Try to build the chart

ChortData= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/CohortAnalysis/Submit2019OnwardsApp2018Final.csv',encoding= 'iso-8859-1')

ChortData.columns = ChortData.columns.str.replace(' ', '')

ChortData.columns = ChortData.columns.str.lstrip()

ChortData.columns = ChortData.columns.str.rstrip()

ChortData.columns = ChortData.columns.str.strip()

ChortData.info()

ChortData['CurrentAtheneAdvApptStartDate1'] = ChortData['CurrentAtheneAdvApptStartDate1'].astype('datetime64[ns]')

ChortData['SubmitDate'] = ChortData['SubmitDate'].astype('datetime64[ns]')

##ChortData['CurrentAtheneAdvApptStartDate1']= pd.to_datetime(ChortData['CurrentAtheneAdvApptStartDate1']).dt.date

##ChortData['CurrentAtheneAdvApptStartDate1']= pd.to_datetime(ChortData['CurrentAtheneAdvApptStartDate1']).dt.date

ChortData.info()

df= ChortData

df.head()

Out4 = df.to_csv (r'C:\Users\test\Documents\Athene90DaysProducersAndFA\Output\df.csv', index = None, header=True)


##def parse_dmy(s):
  
##    month, day, year = s.split('/')
 ##   return datetime(int(year), int(month), int(day))

df['OrderDate'] = df['CurrentAtheneAdvApptStartDate1'].apply(lambda x: x.strftime('%Y-%m'))

df['OrderPeriod'] = df['SubmitDate'].apply(lambda x: x.strftime('%Y-%m'))

df.head()

# convert each user id into their cohort group

df = df.set_index('AdvisorContactIDText')
df['CohortGroup'] = (df.groupby(level = 0)['CurrentAtheneAdvApptStartDate1'].min().apply(lambda x: x.strftime('%Y-%m')))
df = df.reset_index()
df.head()

grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

grouped.head(20)

# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg({'AdvisorContactIDText': pd.Series.nunique,'SubmitCnt': np.sum,'SubmitAmount': np.sum })

cohorts.head(10)

# make the column names more meaningful
renaming = {'AdvisorContactIDText': 'TotalUsers'}
cohorts = cohorts.rename(columns = renaming)
cohorts.head()

def cohort_period(df):
    """
    Creates a `CohortPeriod` column, 
    which is the Nth period based on the user's first purchase.
    """
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level = 'CohortGroup').apply(cohort_period)
cohorts.head(10)

# convert the CohortPeriod as indices instead of OrderPeriod
cohorts = cohorts.reset_index()
cohorts = cohorts.set_index(['CohortGroup', 'CohortPeriod'])
cohorts.head()

# create a Series holding the total size of each CohortGroup
cohorts_size = cohorts['TotalUsers'].groupby(level = 'CohortGroup').first()
cohorts_size.head()

ChortData.info()

# convert the CohortPeriod as indices instead of OrderPeriod
cohorts = cohorts.reset_index()
cohorts = cohorts.set_index(['CohortGroup', 'CohortPeriod'])
cohorts.head()

# create a Series holding the total size of each CohortGroup
cohorts_size = cohorts['TotalUsers'].groupby(level = 'CohortGroup').first()
cohorts_size.head()

# applying it 
user_retention = (cohorts['TotalUsers'].
                  unstack('CohortGroup').
                  divide(cohorts_size, axis = 1))

user_retention.head()

# change default figure and font size
plt.rcParams['figure.figsize'] = 10, 8
plt.rcParams['font.size'] = 12

user_retention[['2019-01', '2019-02', '2019-03','2019-04','2019-05']].plot()
plt.title('Cohorts: User Retention')
plt.xticks(range(1, 13))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Submitting')
plt.show()

ns.set(style = 'white')

plt.figure(figsize = (12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T,
            cmap = plt.cm.Blues,
            mask = user_retention.T.isnull(),  # data will not be shown where it's True
            annot = True,  # annotate the text on top
            fmt = '.0%')  # string formatting when annot is True
plt.show()

"""

Submit0318_0605['SubmitDay'] = Submit0318_0605['SubmitDate'].astype('datetime64[ns]')

Submit0318_0605['SubmitDay'] = pd.to_datetime(Submit0318_0605['SubmitDay'])

# Getting week number
Submit0318_0605['WeekNumber'] = Submit0318_0605['SubmitDay'].dt.week
# Getting year. Weeknum is common across years to we need to create unique index by using year and weeknum
Submit0318_0605['Month'] = Submit0318_0605['SubmitDay'].dt.month

con = sqlite3.connect("Submit0318_0605.db")


# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit0318_0605.to_sql("Submit0318_0605", con, if_exists='replace')

Submit0318_0605.info()

Submit0318_0605W= pd.read_sql("SELECT  count(SubmitDate ) as SubmitCount, SubmitAmount, SubmitDay FROM Submit0318_0605 group by ProductionNo",con)

Submit0318_0605W['SubmitDay'] = pd.to_datetime(Submit0318_0605W['SubmitDay'])

Submit0318_0605.info()

weekly_summary = Submit0318_0605W.resample('W', on='SubmitDay').sum()

weekly_summary['SubmitWeekStartDate'] = weekly_summary.index


Out4 = weekly_summary.to_csv (r'C:\Users\test\Documents\BCARateChange\weekly_summary.csv', index = None, header=True)

"""



