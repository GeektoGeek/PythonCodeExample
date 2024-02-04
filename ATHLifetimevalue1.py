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

###NW Advisor 2020

Athene3months = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/Lastthreemonthdata.csv',encoding= 'iso-8859-1')

Athene3months.columns = Athene3months.columns.str.replace(' ', '')

Athene3months.columns = Athene3months.columns.str.lstrip()

Athene3months.columns = Athene3months.columns.str.rstrip()

Athene3months.columns = Athene3months.columns.str.strip()

AtheneLast18months = pd.read_csv('C:/Users/test/Documents/AtheneLifetimeValue/AtheneLast18Months.csv',encoding= 'iso-8859-1')

AtheneLast18months.columns = AtheneLast18months.columns.str.replace(' ', '')

AtheneLast18months.columns = AtheneLast18months.columns.str.lstrip()

AtheneLast18months.columns = AtheneLast18months.columns.str.rstrip()

AtheneLast18months.columns = AtheneLast18months.columns.str.strip()

Athene3months['SubmitDate'] = Athene3months['SubmitDate'].astype('datetime64[ns]')

AtheneLast18months['SubmitDate'] = AtheneLast18months['SubmitDate'].astype('datetime64[ns]')

AtheneLast18months['AdvisorContactFirstAtheneAdvApptStartDate2'] = AtheneLast18months['AdvisorContactFirstAtheneAdvApptStartDate2'].astype('datetime64[ns]')

AtheneLast18months.info()

con = sqlite3.connect("AtheneLast18months.db")

AtheneLast18months.to_sql("AtheneLast18months", con, if_exists='replace')

AtheneLast18months.info()

## AtheneLast48moLastSubmitDate1= pd.read_sql("SELECT * FROM AtheneLast48months WHERE SubmitDate IN (SELECT max(SubmitDate) FROM AtheneLast48months group by AdvisorContactIDText)", con)

AtheneLast18moLastSubmitDate = pd.read_sql("SELECT distinct(AdvisorContactIDText), max(SubmitDate) as LastSubmitDate, min(SubmitDate) as FirsubmitDate, count(SubmitDate) as TotalSubmitCount, sum(SubmitAmount) as TotalSubmitAmt, SubmitDate, AdvisorContactFirstAtheneAdvApptStartDate2, AdvisorName, WeekNumberFromSubmitDate, WeekNumberFromAtheneAppDate FROM AtheneLast18months group by AdvisorContactIDText",con)

Out4 = AtheneLast18moLastSubmitDate.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\AtheneLast18moLastSubmitDate.csv', index = None, header=True)

## AtheneLast48moLastSubmitDate['LastSubmitDate'] = AtheneLast48moLastSubmitDate['LastSubmitDate'].astype('datetime64[ns]')

AtheneLast18moLastSubmitDate['LastSubmitDate1'] = pd.to_datetime(AtheneLast18moLastSubmitDate['LastSubmitDate']).dt.date

AtheneLast18moLastSubmitDate['FirsubmitDate1'] = pd.to_datetime(AtheneLast18moLastSubmitDate['FirsubmitDate']).dt.date

AtheneLast18moLastSubmitDate['AtheneAppDate1'] = pd.to_datetime(AtheneLast18moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2']).dt.date

AtheneLast18moLastSubmitDate.info()


AtheneLast18moLastSubmitDate['LastSubmitDate1'] = AtheneLast18moLastSubmitDate['LastSubmitDate1'].astype('datetime64[ns]')

##AtheneLast48moLastSubmitDate['Weeknumber_lastSubmitDate'] = pd.to_datetime(AtheneLast48moLastSubmitDate['LastSubmitDate1']).weekday()

AtheneLast18moLastSubmitDate['Weeknumber_lastSubmitDate'] = AtheneLast18moLastSubmitDate['LastSubmitDate1'].dt.week

con = sqlite3.connect("AtheneLast18moLastSubmitDate.db")

AtheneLast18moLastSubmitDate.to_sql("AtheneLast18moLastSubmitDate", con, if_exists='replace')

Exit_18mon = pd.read_sql("SELECT  count(LastSubmitDate1) as CountExit, WeekNumberFromSubmitDate as WeekNum FROM AtheneLast18moLastSubmitDate group by WeekNumberFromSubmitDate",con)

Entrance_18mo = pd.read_sql("SELECT  count(AtheneAppDate1) as CountEntrance, WeekNumberFromAtheneAppDate as WeekNum FROM AtheneLast18moLastSubmitDate group by WeekNumberFromAtheneAppDate",con)

### AtheneLast48moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2'] = pd.to_datetime(AtheneLast48moLastSubmitDate['AdvisorContactFirstAtheneAdvApptStartDate2']).dt.date

con = sqlite3.connect("Exit_18mon.db")



Exit_18mon.to_sql("Exit_18mon", con, if_exists='replace')

Exit_18mon.info()

Entrance_18mo.info()

con = sqlite3.connect("Entrance_18mo.db")

Entrance_18mo.to_sql("Entrance_18mo", con, if_exists='replace')

qqq11  = """SELECT a.WeekNum, a.CountExit, b.CountEntrance FROM Exit_18mon a LEFT JOIN Entrance_18mo b on a.WeekNum = b.WeekNum;"""
        
Merge18moEntranceExist=  pysqldf(qqq11) 

Merge18moEntranceExist.info()

Out4 = Merge18moEntranceExist.to_csv (r'C:\Users\test\Documents\AtheneLifetimeValue\Output\Merge18moEntranceExist.csv', index = None, header=True)

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