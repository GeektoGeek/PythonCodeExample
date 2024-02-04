import tensorflow as ts
import pandas as pd
import seaborn as sns
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

### Email Data

Email_DF = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/ListSupression/IDCDeliverabilityResearch20191014b-Master.csv',encoding= 'iso-8859-1')

Email_DF.info()

Email_DF['SendDateTimestampC'] = Email_DF['SendDateTimestamp']

Email_DF['OpenDateC'] = Email_DF['OpenDate'] 

Email_DF['ClickDateC'] = Email_DF['ClickDate']

Email_DF.info()


Email_DF['SendDateTimestamp'] = Email_DF['SendDateTimestamp'].astype('datetime64[ns]')

Email_DF['Sendtime_hour'] = Email_DF['SendDateTimestamp'].dt.hour

Email_DF['OpenDate'] = Email_DF['OpenDate'].astype('datetime64[ns]')

Email_DF['Opentime_hour'] = Email_DF['OpenDate'].dt.hour

Email_DF['ClickDate'] = Email_DF['ClickDate'].astype('datetime64[ns]')

Email_DF['Clicktime_hour'] = Email_DF['ClickDate'].dt.hour


con = sqlite3.connect("Email_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_DF.to_sql("Email_DF", con, if_exists='replace')

Email_DF.info()


EmailSummary = pd.read_sql("SELECT distinct(Name), SendDate, Account_Name__c, sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged  FROM Email_DF group by Name, Account_Name__c",con)

EmailSummary['SendDate'] = pd.to_datetime(EmailSummary['SendDate'])

EmailSummary['SendDate'] = pd.to_datetime(EmailSummary['SendDate']).dt.normalize()

EmailSummary.info()

### Submit Data
Submit_DF= pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/ListSupression/May8th18-Oct14th19.csv',encoding= 'iso-8859-1')

Submit_DF.info()

Submit_DF['SubmitDate'] = pd.to_datetime(Submit_DF['SubmitDate'])

Submit_DF.info()

con = sqlite3.connect("Submit_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit_DF.to_sql("Submit_DF", con, if_exists='replace')

SubmitSummaryAdv = pd.read_sql("SELECT AdvisorName, Account_AccountName, SubmitDate, sum(SubmitDate) as SubmitCount, SubmitAmount FROM Submit_DF group by AdvisorName, Account_AccountName",con)

SubmitSummaryMar = pd.read_sql("SELECT Marketer_MarketerName, SubmitDate, Account_AccountName, sum(SubmitDate) as SubmitCount, SubmitAmount FROM Submit_DF group by Marketer_MarketerName, Account_AccountName",con)

q  = """SELECT * FROM EmailSummary a
        JOIN SubmitSummaryAdv b on a.Name = b.AdvisorName where ((julianday(b.SubmitDate)-julianday(a.SendDate) >= 0) and (julianday(b.SubmitDate)-julianday(a.SendDate) < 181));"""
        
Adv_Sub_Email =  pysqldf(q)  

q1  = """SELECT * FROM EmailSummary a
        JOIN SubmitSummaryMar b on (a.Name = b.Marketer_MarketerName) where ((julianday(b.SubmitDate)-julianday(a.SendDate) >= 0) and (julianday(b.SubmitDate)-julianday(a.SendDate) < 181));"""
        
Mar_Sub_Email =  pysqldf(q1)  

### Illustration data

Illu_DF= pd.read_csv('C:/Users/test/Documents/SingleContract-MultiContract_Pre_Post/Multi-Contract-Brian/Multi-Contract-Rider.csv',encoding= 'iso-8859-1')

Illu_DF.info()



Illu_DF['PreparationDate'] = pd.to_datetime(Illu_DF['PreparationDate'])

con = sqlite3.connect("Illu_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Illu_DF.to_sql("Illu_DF", con, if_exists='replace')

Illus_Count = pd.read_sql("SELECT PreparationDate, count(PreparationDate) as Illus_Count, Name, PreparedBy, Imo, Role from Illu_DF where (PreparationDate >= '2018-05-08') and (PreparationDate >= '2019-10-15') group by Name",con)

q2  = """SELECT * FROM EmailSummary c
        JOIN Illus_Count d on (c.Name = d.Name) where ((julianday(d.PreparationDate)-julianday(c.SendDate) >= 0) and (julianday(d.PreparationDate)-julianday(c.SendDate) < 181));"""
        
Mar_Illus_Email =  pysqldf(q2)  

### Illustration Submit and Email
        
 
q3  = """SELECT * FROM Mar_Illus_Email e
        JOIN SubmitSummaryMar f on (e.Name = f.Marketer_MarketerName);"""

Mar_Illus_Email_Sub =  pysqldf(q3) 

export_csv = Mar_Illus_Email.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\Mar_Illus_Email.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = Mar_Sub_Email.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\Mar_Sub_Email.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = Mar_Illus_Email_Sub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\Mar_Illus_Email_Sub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Send Time Optimization

## Take the base dataset Email_DF

Email_DF1= Email_DF

Email_DF1.info()

##Email_DF1['SendDateTimestamp'] = pd.to_datetime(Email_DF1['SendDateTimestamp'])

# Set datetime precision to 'day'

##Email_DF1['SendDateTimestamp'] = Email_DF1['SendDateTimestamp'].astype('datetime64[D]')

##Email_DF1['ClickExist'].plot(kind='line', grid=True, title='Open of Email Based on Send Dates, 2018-2019')
## Email_DF1.head()

### Lets process the Send Date
Email_DF1['Sendmonth'] = Email_DF1['SendDateTimestamp'].dt.month

Email_DF1['Sendday'] = Email_DF1['SendDateTimestamp'].dt.day

Email_DF1['Sendquarter'] = Email_DF1['SendDateTimestamp'].dt.quarter

Email_DF1['Sendsemester'] = np.where(Email_DF1.Sendquarter.isin([1,2]),1,2)

Email_DF1['Senddayofweek'] = Email_DF1['SendDateTimestamp'].dt.dayofweek

Email_DF1['Senddayofweek_name'] = Email_DF1['SendDateTimestamp'].dt.weekday_name

Email_DF1['is_Send_weekend'] = np.where(Email_DF1['Senddayofweek_name'].isin(['Sunday','Saturday']),1,0)

Email_DF1['Sendtime_hour'] = Email_DF1.SendDateTimestamp.apply(lambda x: x.hour)

con = sqlite3.connect("Email_DF1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_DF1.to_sql("Email_DF1", con, if_exists='replace')

Email_DF1.info()

EmailSendDayFre = pd.read_sql("SELECT count(Senddayofweek_name) as Freq, Senddayofweek_name as Send_Day  FROM Email_DF1 group by Senddayofweek_name order by Freq desc",con)


export_csv = EmailSendDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailSendDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailSendMonthDayFre = pd.read_sql("SELECT count(Sendday) as Freq, Sendday as Sendmonth_Day FROM Email_DF1 group by Sendday order by Sendday",con)

export_csv = EmailSendMonthDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailSendMonthDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
### Lets process the Open Date


Email_DF1['OpenDate'] = pd.to_datetime(Email_DF1['OpenDate'])

# Set datetime precision to 'day'

Email_DF1['OpenDate'] = Email_DF1['OpenDate'].astype('datetime64[D]')


Email_DF1['Openmonth'] = Email_DF1['OpenDate'].dt.month

Email_DF1['Openday'] = Email_DF1['OpenDate'].dt.day

Email_DF1['Openquarter'] = Email_DF1['OpenDate'].dt.quarter

Email_DF1['Opensemester'] = np.where(Email_DF1.Sendquarter.isin([1,2]),1,2)

Email_DF1['Opendayofweek'] = Email_DF1['OpenDate'].dt.dayofweek

Email_DF1['Opendayofweek_name'] = Email_DF1['OpenDate'].dt.weekday_name

Email_DF1['is_Open_weekend'] = np.where(Email_DF1['Opendayofweek_name'].isin(['Sunday','Saturday']),1,0)

Email_DF1.info()

con = sqlite3.connect("Email_DF1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_DF1.to_sql("Email_DF1", con, if_exists='replace')

Email_DF1.info()

EmailOpenDayFre = pd.read_sql("SELECT count(Openday) as Freq, Openday  as Openmonth_Day FROM Email_DF1 group by Opendayofweek_name  order by Freq desc",con)

export_csv = EmailOpenDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailOpenDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailOpenMonthDayFre = pd.read_sql("SELECT count(Openday) as Freq, Openday as Openmonth_Day FROM Email_DF1 group by Openday order by Openday",con)

export_csv = EmailOpenMonthDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailOpenMonthDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Lets process the Click Date

Email_DF1['ClickDate'] = pd.to_datetime(Email_DF1['ClickDate'])

# Set datetime precision to 'day'

Email_DF1['ClickDate'] = Email_DF1['ClickDate'].astype('datetime64[D]')

Email_DF1['Clickmonth'] = Email_DF1['ClickDate'].dt.month

Email_DF1['Clickday'] = Email_DF1['ClickDate'].dt.day

Email_DF1['Clickquarter'] = Email_DF1['ClickDate'].dt.quarter

Email_DF1['Clicksemester'] = np.where(Email_DF1.Sendquarter.isin([1,2]),1,2)

Email_DF1['Clickdayofweek'] = Email_DF1['ClickDate'].dt.dayofweek

Email_DF1['Clickdayofweek_name'] = Email_DF1['ClickDate'].dt.weekday_name

Email_DF1['is_Click_weekend'] = np.where(Email_DF1['Clickdayofweek_name'].isin(['Sunday','Saturday']),1,0)

con = sqlite3.connect("Email_DF1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_DF1.to_sql("Email_DF1", con, if_exists='replace')

Email_DF1.info()

EmailClickDayFre = pd.read_sql("SELECT count(Clickdayofweek_name ) as Freq, Clickdayofweek_name  as Click_Day FROM Email_DF1 group by Clickdayofweek_name  order by Freq desc",con)

export_csv = EmailClickDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailClickDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailClickMonthDayFre = pd.read_sql("SELECT count(Clickday) as Freq, Clickday as Clickmonth_Day FROM Email_DF1 group by Clickday order by Clickday",con)

export_csv = EmailClickMonthDayFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailClickMonthDayFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#######

EmailSendHourFre = pd.read_sql("SELECT count(Sendtime_hour) as Freq, Sendtime_hour as Sendhour FROM Email_DF1 group by Sendtime_hour order by Sendtime_hour desc",con)

export_csv = EmailSendHourFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\Sendtime_hourFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailOpenHourFre = pd.read_sql("SELECT count(Opentime_hour) as Freq, Opentime_hour  as Opentime_hour FROM Email_DF1 group by Opentime_hour order by Opentime_hour desc",con)

export_csv = EmailOpenHourFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailOpenHourFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

EmailClickHourFre = pd.read_sql("SELECT count(Clicktime_hour) as Freq, Clicktime_hour as Clicktime_hour FROM Email_DF1 group by Clicktime_hour order by Clicktime_hour",con)

export_csv = EmailClickHourFre.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\EmailClickHourFre.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
