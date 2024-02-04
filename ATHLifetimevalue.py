
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

###NW Advisor 2020

Athene3months = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Lastthreemonthdata.csv',encoding= 'iso-8859-1')

Athene3months.columns = Athene3months.columns.str.replace(' ', '')

Athene3months.columns = Athene3months.columns.str.lstrip()

Athene3months.columns = Athene3months.columns.str.rstrip()

Athene3months.columns = Athene3months.columns.str.strip()

AtheneLast48months = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Last48monthsdataUpdated.csv',encoding= 'iso-8859-1')

AtheneLast48months.columns = AtheneLast48months.columns.str.replace(' ', '')

AtheneLast48months.columns = AtheneLast48months.columns.str.lstrip()

AtheneLast48months.columns = AtheneLast48months.columns.str.rstrip()

AtheneLast48months.columns = AtheneLast48months.columns.str.strip()

Athene3months['SubmitDate'] = Athene3months['SubmitDate'].astype('datetime64[ns]')

AtheneLast48months['SubmitDate'] = AtheneLast48months['SubmitDate'].astype('datetime64[ns]')

AtheneLast48months['AdvisorContactFirstAtheneAdvApptStartDate2'] = AtheneLast48months['AdvisorContactFirstAtheneAdvApptStartDate2'].astype('datetime64[ns]')

AtheneLast48months.info()

con = sqlite3.connect("AtheneLast48months.db")

AtheneLast48months.to_sql("AtheneLast48months", con, if_exists='replace')

AtheneLast48months.info()

## AtheneLast48moLastSubmitDate1= pd.read_sql("SELECT * FROM AtheneLast48months WHERE SubmitDate IN (SELECT max(SubmitDate) FROM AtheneLast48months group by AdvisorContactIDText)", con)

AtheneLast48moLastSubmitDate = pd.read_sql("SELECT distinct(AdvisorContactIDText), max(SubmitDate) as LastSubmitDate, min(SubmitDate) as FirsubmitDate, count(SubmitDate) as TotalSubmitCount, sum(SubmitAmount) as TotalSubmitAmt, SubmitDate, AdvisorContactFirstAtheneAdvApptStartDate2, AdvisorName, WeekNumberFromSubmitDate, WeekNumberFromAtheneAppDate FROM AtheneLast48months group by AdvisorContactIDText",con)

Out4 = AtheneLast48moLastSubmitDate.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\AtheneLast48moLastSubmitDate.csv', index = None, header=True)

## AtheneLast48moLastSubmitDate['LastSubmitDate'] = AtheneLast48moLastSubmitDate['LastSubmitDate'].astype('datetime64[ns]')

AtheneLast48moLastSubmitDate['LastSubmitDate1'] = pd.to_datetime(AtheneLast48moLastSubmitDate['LastSubmitDate']).dt.date

AtheneLast48moLastSubmitDate['FirsubmitDate1'] = pd.to_datetime(AtheneLast48moLastSubmitDate['FirsubmitDate']).dt.date

AtheneLast48moLastSubmitDate['AtheneAppDate1'] = pd.to_datetime(AtheneLast48moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2']).dt.date

AtheneLast48moLastSubmitDate.info()


AtheneLast48moLastSubmitDate['LastSubmitDate1'] = AtheneLast48moLastSubmitDate['LastSubmitDate1'].astype('datetime64[ns]')

##AtheneLast48moLastSubmitDate['Weeknumber_lastSubmitDate'] = pd.to_datetime(AtheneLast48moLastSubmitDate['LastSubmitDate1']).weekday()

AtheneLast48moLastSubmitDate['Weeknumber_lastSubmitDate'] = AtheneLast48moLastSubmitDate['LastSubmitDate1'].dt.week

con = sqlite3.connect("AtheneLast48moLastSubmitDate.db")

AtheneLast48moLastSubmitDate.to_sql("AtheneLast48moLastSubmitDate", con, if_exists='replace')

Exit = pd.read_sql("SELECT  count(LastSubmitDate1) as CountExit, WeekNumberFromSubmitDate as WeekNum FROM AtheneLast48moLastSubmitDate group by WeekNumberFromSubmitDate",con)

Entrance = pd.read_sql("SELECT  count(AtheneAppDate1) as CountEntrance, WeekNumberFromAtheneAppDate as WeekNum FROM AtheneLast48moLastSubmitDate group by WeekNumberFromAtheneAppDate",con)

### AtheneLast48moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2'] = pd.to_datetime(AtheneLast48moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2']).dt.date

con = sqlite3.connect("Exit.db")

Exit.to_sql("Exit", con, if_exists='replace')

Exit.info()

Entrance.info()

con = sqlite3.connect("Entrance.db")

Entrance.to_sql("Entrance", con, if_exists='replace')

qqq1  = """SELECT a.WeekNum, a.CountExit, b.CountEntrance FROM Exit a LEFT JOIN Entrance b on a.WeekNum = b.WeekNum;"""
        
MergeEntranceExist=  pysqldf(qqq1) 

MergeEntranceExist.info()

Out4 = MergeEntranceExist.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\MergeEntranceExist.csv', index = None, header=True)

# multiple line plot
plt.plot( 'WeekNum', 'CountEntrance', data=MergeEntranceExist, marker='', markerfacecolor='blue', markersize=12, color='blue', linewidth=4)
plt.plot( 'WeekNum', 'CountExit', data=MergeEntranceExist, marker='', color='olive', linewidth=2)
plt.legend()

### Let's look into the cohort analysis

AtheneLast48moLastSubmitDate.describe().transpose()

AtheneLast48moLastSubmitDate.info()

AtheneLast48months.info()

DF_Athene = AtheneLast48moLastSubmitDate[['AdvisorContactIDText', 'LastSubmitDate1']].drop_duplicates()

AtheneLast48moLastSubmitDate['Submit_month'] = AtheneLast48moLastSubmitDate['LastSubmitDate1'].dt.to_period('M')

AtheneLast48moLastSubmitDate['cohort'] = AtheneLast48moLastSubmitDate.groupby('AdvisorContactIDText')['LastSubmitDate1'] \
                 .transform('min') \
                 .dt.to_period('M') 
                 
AtheneLast48moLastSubmitDate_cohort = AtheneLast48moLastSubmitDate.groupby(['cohort', 'Submit_month']) \
              .agg(n_customers=('AdvisorContactIDText', 'nunique')) \
              .reset_index(drop=False) 

AtheneLast48moLastSubmitDate_cohort['period_number'] = (AtheneLast48moLastSubmitDate_cohort.Submit_month - AtheneLast48moLastSubmitDate_cohort.cohort).apply(attrgetter('n'))


cohort_pivot = AtheneLast48moLastSubmitDate_cohort.pivot_table(index = 'cohort',
                                     columns = 'period_number',
                                     values = 'n_customers')   


grouped = AtheneLast48moLastSubmitDate.groupby(['cohort', 'Submit_month'])

cohorts = grouped.agg({'AdvisorContactIDText': pd.Series.nunique,
                       'TotalSubmitCount': pd.Series.nunique,
                       'TotalSubmitAmt': np.sum})

cohorts.head()

#### Restart...

df= AtheneLast48moLastSubmitDate

##df1=AtheneLast48months


DF2 = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Athene2019_2020Updated2.csv',encoding= 'iso-8859-1')

DF2.columns = DF2.columns.str.replace(' ', '')

DF2.columns = DF2.columns.str.lstrip()

DF2.columns = DF2.columns.str.rstrip()

DF2.columns = DF2.columns.str.strip()



DF2['SubmitDate'] = DF2['SubmitDate'].astype('datetime64[ns]')

DF2['AdvisorContactFirstAtheneAdvApptStartDate2'] = DF2['AdvisorContactFirstAtheneAdvApptStartDate2'].astype('datetime64[ns]')




DF2['AdvisorContactFirstAtheneAdvApptStartDate2'] = DF2['AdvisorContactFirstAtheneAdvApptStartDate2'].astype('datetime64[ns]')

df1=DF2

df1.info()
### Perhaps create the cohort based on their joining date

### Join Period is essentially is the cohort group

df1['joinPeriod'] = df1.AdvisorContactFirstAtheneAdvApptStartDate2.apply(lambda x: x.strftime('%Y-%m'))
df1.head()

### Submit Period is essentially is the Order group

df1['SubmitPeriod'] = df1.SubmitDate .apply(lambda x: x.strftime('%Y-%m'))

###df['SubmitPeriod'] = df.FirsubmitDate1.apply(lambda x: x.strftime('%Y-%m'))
df1.head()

grouped1 = df1.groupby(['SubmitPeriod','joinPeriod'])

grouped1.head()

# count the unique users, orders, and total revenue per Group + Period
cohorts1 = grouped1.agg({'AdvisorContactIDText': pd.Series.nunique,
                       'ProductionTotalID': pd.Series.nunique,
                       'SubmitAmount': np.sum})

cohorts1.rename(columns={'AdvisorContactIDText': 'TotalAdvisors','ProductionTotalID': 'TotalSubmits'}, inplace=True)

cohorts1.head()

def cohort1_period(df1):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime', inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df1['CohortPeriod'] = np.arange(len(df1)) + 1
    return df1

cohorts1 = cohorts1.groupby(level=0).apply(cohort1_period)
cohorts1.head()

### Testing
"""
x = df1[(df1.joinPeriod == '2016-06') & (df1.SubmitPeriod == '2016-06')]
y = cohorts1.ix[('2016-06', '2016-06')]

assert(x['AdvisorContactIDText'].nunique() == y['TotalAdvisors'])
assert(x['SubmitAmount'].sum().round(2) == y['SubmitAmount'].round(2))
assert(x['ProductionTotalID'].nunique() == y['TotalSubmits'])
"""

"""
x = df1[(df1.joinPeriod == '2016-06') & (df1.SubmitPeriod == '2016-06')]
y = cohorts1.ix[('2016-06', '2016-06')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-05') & (df.OrderPeriod == '2009-09')]
y = cohorts.ix[('2009-05', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders']) 

"""

cohorts1.reset_index(inplace=True)
cohorts1.set_index(['joinPeriod', 'SubmitPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort1_group_size = cohorts1['TotalAdvisors'].groupby(level=0).first()
cohort1_group_size.head()

cohorts1['TotalAdvisors'].head()

cohorts1['TotalAdvisors'].unstack(0).head()

Advisor_retention = cohorts1['TotalAdvisors'].unstack(0).divide(cohort1_group_size, axis=1)
Advisor_retention.head(10)

Advisor_retention[['2019-01', '2019-02', '2019-03']].plot(figsize=(10,5))
plt.title('Cohorts: Advisors Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Submitting');

plt.figure(figsize=(20, 12))
plt.title('Cohorts: User Retention')
sb.heatmap(Advisor_retention.T, mask=Advisor_retention.T.isnull(), annot=True, fmt='.0%')

#### 18 Months

### Last Submit--> Exit

AtheneSubmit18monthWeeknumber = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07152020/AtheneSubmit18monthWeeknumber.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneSubmit18monthWeeknumber.db")

AtheneSubmit18monthWeeknumber.to_sql("AtheneSubmit18monthWeeknumber", con, if_exists='replace')

AtheneSubmit18monthWeeknumber.info()

Exit_18months = pd.read_sql("SELECT  count(LastSubmitDate) as CountExit, WeekNumber FROM AtheneSubmit18monthWeeknumber group by WeekNumber",con)

sum1= Exit_18months['CountExit'].sum()

print (sum1)

Out1 = Exit_18months.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\Exit_18months.csv', index = None, header=True)


### Appointment--> Entrance

AtheneAppointment18monthWeeknumber = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07152020/AtheneAppointment18monthWeeknumber.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneAppointment18monthWeeknumber.db")

AtheneAppointment18monthWeeknumber.to_sql("AtheneAppointment18monthWeeknumber", con, if_exists='replace')

AtheneAppointment18monthWeeknumber.info()

Entrance_18months = pd.read_sql("SELECT  count(WeeknumberAtheneApp) as CountEntrance, WeeknumberAtheneApp FROM AtheneAppointment18monthWeeknumber group by WeeknumberAtheneApp",con)

sum2= Entrance_18months['CountEntrance'].sum()

print (sum2)

Out2 = Entrance_18months.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\Entrance_18months.csv', index = None, header=True)

con = sqlite3.connect("Exit_18months.db")

Exit_18months.to_sql("Exit_18months", con, if_exists='replace')

Exit_18months.info()

Entrance_18months.info()

con = sqlite3.connect("Entrance_18months.db")

Entrance_18months.to_sql("Entrance_18months", con, if_exists='replace')

qqq1  = """SELECT a.WeekNumber, a.CountExit, b.CountEntrance FROM Exit_18months a LEFT JOIN Entrance_18months b on a.WeekNumber = b.WeeknumberAtheneApp;"""
        
MergeEntranceExist_18month=  pysqldf(qqq1) 

MergeEntranceExist_18month.info()

##Out4 = MergeEntranceExist.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\MergeEntranceExist.csv', index = None, header=True)

# multiple line plot
plt.plot( 'WeekNumber', 'CountEntrance', data=MergeEntranceExist_18month, marker='', markerfacecolor='blue', markersize=12, color='blue', linewidth=4)
plt.plot( 'WeekNumber', 'CountExit', data=MergeEntranceExist_18month, marker='', color='olive', linewidth=2)
plt.legend()

#### 48 Months

### Last Submit--> Exit

AtheneSubmit48monthWeeknumber = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07152020/AtheneSubmit48monthWeeknumber.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneSubmit48monthWeeknumber.db")

AtheneSubmit48monthWeeknumber.to_sql("AtheneSubmit48monthWeeknumber", con, if_exists='replace')

AtheneSubmit48monthWeeknumber.info()

Exit_48months = pd.read_sql("SELECT  count(LastSubmitDate) as CountExit, WeekNumber FROM AtheneSubmit48monthWeeknumber group by WeekNumber",con)

sum3= Exit_48months['CountExit'].sum()

print (sum3)

Out1 = Exit_48months.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\Exit_48months.csv', index = None, header=True)


### Appointment--> Entrance

AtheneAppointment48monthWeeknumber = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07152020/AtheneAppointment48monthWeeknumber.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneAppointment48monthWeeknumber.db")

AtheneAppointment48monthWeeknumber.to_sql("AtheneAppointment48monthWeeknumber", con, if_exists='replace')

AtheneAppointment48monthWeeknumber.info()

Entrance_48months = pd.read_sql("SELECT  count(AtheneAppWeek) as CountEntrance, AtheneAppWeek FROM AtheneAppointment48monthWeeknumber group by AtheneAppWeek",con)

sum4= Entrance_48months['CountEntrance'].sum()

print (sum4)

Out2 = Entrance_48months.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\Entrance_48months.csv', index = None, header=True)

con = sqlite3.connect("Exit_18months.db")

Exit_18months.to_sql("Exit_18months", con, if_exists='replace')

Exit_18months.info()

Entrance_18months.info()

con = sqlite3.connect("Entrance_18months.db")

Entrance_18months.to_sql("Entrance_18months", con, if_exists='replace')

qqq1  = """SELECT a.WeekNumber, a.CountExit, b.CountEntrance FROM Exit_18months a LEFT JOIN Entrance_18months b on a.WeekNumber = b.WeeknumberAtheneApp;"""
        
MergeEntranceExist_18month=  pysqldf(qqq1) 

MergeEntranceExist_18month.info()

##Out4 = MergeEntranceExist.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\MergeEntranceExist.csv', index = None, header=True)

# multiple line plot
plt.plot( 'WeekNumber', 'CountEntrance', data=MergeEntranceExist_18month, marker='', markerfacecolor='blue', markersize=12, color='blue', linewidth=4)
plt.plot( 'WeekNumber', 'CountExit', data=MergeEntranceExist_18month, marker='', color='olive', linewidth=2)
plt.legend()


#### New vs. Existing Advisors

### Let's look into 2019

SubmitAdv2019= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/Advisors2019SubmitRollUp.csv',encoding= 'iso-8859-1')

SubmitAdv2019.columns = SubmitAdv2019.columns.str.replace(' ', '')

SubmitAdv2019.columns = SubmitAdv2019.columns.str.lstrip()

SubmitAdv2019.columns = SubmitAdv2019.columns.str.rstrip()

SubmitAdv2019.columns = SubmitAdv2019.columns.str.strip()

SubmitAdv2019.info()

con = sqlite3.connect("SubmitAdv2019.db")

SubmitAdv2019.to_sql("SubmitAdv2019", con, if_exists='replace')

Oct2018_2019AtheneAppointedAdvisors= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/Oct2018_2019AtheneAppointedAdvisors.csv',encoding= 'iso-8859-1')


Oct2018_2019AtheneAppointedAdvisors.columns = Oct2018_2019AtheneAppointedAdvisors.columns.str.replace(' ', '')

Oct2018_2019AtheneAppointedAdvisors.columns = Oct2018_2019AtheneAppointedAdvisors.columns.str.lstrip()

Oct2018_2019AtheneAppointedAdvisors.columns = Oct2018_2019AtheneAppointedAdvisors.columns.str.rstrip()

Oct2018_2019AtheneAppointedAdvisors.columns = Oct2018_2019AtheneAppointedAdvisors.columns.str.strip()

Oct2018_2019AtheneAppointedAdvisors.info()

con = sqlite3.connect("Oct2018_2019AtheneAppointedAdvisors.db")

Oct2018_2019AtheneAppointedAdvisors.to_sql("Oct2018_2019AtheneAppointedAdvisors", con, if_exists='replace')

qqq1  = """SELECT a.*, b.* FROM SubmitAdv2019 a INNER JOIN Oct2018_2019AtheneAppointedAdvisors b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
Common =  pysqldf(qqq1) 

#### 

### Let's look into 2018

SubmitAdv2018= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/2018/2018AdvSubmitsGrBy.csv',encoding= 'iso-8859-1')

SubmitAdv2018.columns = SubmitAdv2018.columns.str.replace(' ', '')

SubmitAdv2018.columns = SubmitAdv2018.columns.str.lstrip()

SubmitAdv2018.columns = SubmitAdv2018.columns.str.rstrip()

SubmitAdv2018.columns = SubmitAdv2018.columns.str.strip()

SubmitAdv2018.info()

con = sqlite3.connect("SubmitAdv2018.db")

SubmitAdv2018.to_sql("SubmitAdv2018", con, if_exists='replace')

Oct2017_2018AtheneAppointedAdvisors= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/2018/2017Oct2018AtheneAppAdvisors.csv',encoding= 'iso-8859-1')


Oct2017_2018AtheneAppointedAdvisors.columns = Oct2017_2018AtheneAppointedAdvisors.columns.str.replace(' ', '')

Oct2017_2018AtheneAppointedAdvisors.columns = Oct2017_2018AtheneAppointedAdvisors.columns.str.lstrip()

Oct2017_2018AtheneAppointedAdvisors.columns = Oct2017_2018AtheneAppointedAdvisors.columns.str.rstrip()

Oct2017_2018AtheneAppointedAdvisors.columns = Oct2017_2018AtheneAppointedAdvisors.columns.str.strip()

Oct2017_2018AtheneAppointedAdvisors.info()

con = sqlite3.connect("Oct2017_2018AtheneAppointedAdvisors.db")

Oct2017_2018AtheneAppointedAdvisors.to_sql("Oct2017_2018AtheneAppointedAdvisors", con, if_exists='replace')

qqq2  = """SELECT a.*, b.* FROM SubmitAdv2018 a INNER JOIN Oct2017_2018AtheneAppointedAdvisors b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
Common2018 =  pysqldf(qqq2) 

##### Let's look into 2020

SubmitAdv2020= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/2020/2020AdvSubmitsGroupBy.csv',encoding= 'iso-8859-1')

SubmitAdv2020.columns = SubmitAdv2020.columns.str.replace(' ', '')

SubmitAdv2020.columns = SubmitAdv2020.columns.str.lstrip()

SubmitAdv2020.columns = SubmitAdv2020.columns.str.rstrip()

SubmitAdv2020.columns = SubmitAdv2020.columns.str.strip()

SubmitAdv2020.info()

con = sqlite3.connect("SubmitAdv2020.db")

SubmitAdv2020.to_sql("SubmitAdv2020", con, if_exists='replace')

Sep2019_2020AtheneAppAdvisors= pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/NewVsExistingProducers2019/2020/2020_Sep2019AtheneAppAdvisors.csv',encoding= 'iso-8859-1')


Sep2019_2020AtheneAppAdvisors.columns = Sep2019_2020AtheneAppAdvisors.columns.str.replace(' ', '')

Sep2019_2020AtheneAppAdvisors.columns = Sep2019_2020AtheneAppAdvisors.columns.str.lstrip()

Sep2019_2020AtheneAppAdvisors.columns = Sep2019_2020AtheneAppAdvisors.columns.str.rstrip()

Sep2019_2020AtheneAppAdvisors.columns = Sep2019_2020AtheneAppAdvisors.columns.str.strip()

Sep2019_2020AtheneAppAdvisors.info()

con = sqlite3.connect("Sep2019_2020AtheneAppAdvisors.db")

Sep2019_2020AtheneAppAdvisors.to_sql("Sep2019_2020AtheneAppAdvisors", con, if_exists='replace')

qqq3  = """SELECT a.*, b.* FROM SubmitAdv2020 a INNER JOIN Sep2019_2020AtheneAppAdvisors b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
Common2020 =  pysqldf(qqq3) 



#### Rolling up the Submit Data to overlay Rate Change Events

SubmitRaw = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07182020/SubmitRaw.csv',encoding= 'iso-8859-1')

SubmitRaw.columns = SubmitRaw.columns.str.replace(' ', '')

SubmitRaw.columns = SubmitRaw.columns.str.lstrip()

SubmitRaw.columns = SubmitRaw.columns.str.rstrip()

SubmitRaw.columns = SubmitRaw.columns.str.strip()


con = sqlite3.connect("SubmitRaw.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitRaw.to_sql("SubmitRaw", con, if_exists='replace')

SubmitRaw.info()

SubmitRaw['SubmitDate'] = SubmitRaw['SubmitDate'].astype('datetime64[ns]')

q2  = """SELECT AdvisorContactIDText, AdvisorName, ProductCode, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmount, SubmitDate, min(SubmitDate) as FirstSubmitDate, max(SubmitDate) as LastSubmitDate,  AdvisorContactFirstAtheneAdvApptStartDate2 from SubmitRaw group by AdvisorContactIDText;"""
      
SubmitRawGrBy =  pysqldf(q2)  


con = sqlite3.connect("SubmitRawGrBy.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitRawGrBy.to_sql("SubmitRawGrBy", con, if_exists='replace')


AppointmentRaw = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Data07182020/AppointmentRaw.csv',encoding= 'iso-8859-1')

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
      
SubmitApp =  pysqldf(q3) 

Out4 = SubmitApp.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\SubmitApp.csv', index = None, header=True) 

##### Let's Try to build the chart

Submit0318_0605= pd.read_csv('C:/Users/test/Documents/BCARateChange/Submit03292019_06052020.csv',encoding= 'iso-8859-1')

Submit0318_0605.columns = Submit0318_0605.columns.str.replace(' ', '')

Submit0318_0605.columns = Submit0318_0605.columns.str.lstrip()

Submit0318_0605.columns = Submit0318_0605.columns.str.rstrip()

Submit0318_0605.columns = Submit0318_0605.columns.str.strip()

Submit0318_0605['SubmitDate'] = Submit0318_0605['SubmitDate'].astype('datetime64[ns]')

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

