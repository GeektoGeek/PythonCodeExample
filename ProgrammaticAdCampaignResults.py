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

### AIG List 5 & List 6 Quick Quote Tool comparison started 12/09

### Bring the Submit Data For AIG Only from 12/05 until 2/1

AIGSubmit12092020_012722021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGSubmit12092020_012722021.csv',encoding= 'iso-8859-1')

AIGSubmit12092020_012722021.columns = AIGSubmit12092020_012722021.columns.str.replace(' ', '')

AIGSubmit12092020_012722021.columns = AIGSubmit12092020_012722021.columns.str.lstrip()

AIGSubmit12092020_012722021.columns = AIGSubmit12092020_012722021.columns.str.rstrip()

AIGSubmit12092020_012722021.columns = AIGSubmit12092020_012722021.columns.str.strip()

AIGSubmit12092020_012722021['SubmitDate'] = pd.to_datetime(AIGSubmit12092020_012722021['SubmitDate'])

con = sqlite3.connect("AIGSubmit12092020_012722021.db")

AIGSubmit12092020_012722021.to_sql("AIGSubmit12092020_012722021", con, if_exists='replace')

AIGSubmit12092020_012722021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AIGSubmit12092020_012722021 group by AdvisorContactIDText, AdvisorName;"""

AIGSubmit12092020_01272202GrBy =  pysqldf(q3) 

### Lets Bring the list 5

### List 5 is the Fallen Angels of 3 carriers over 90 days

List5 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/List5.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List5.db")

List5.to_sql("List5", con, if_exists='replace')

List5.info()

vm11 = """SELECT a.*, b.FullName FROM AIGSubmit12092020_01272202GrBy a INNER JOIN List5 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

ConversionAIGList5 =  pysqldf(vm11)

Out4 = ConversionAIGList5.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ConversionAIGList5.csv', index = None, header=True)

### Lets Bring the List 6

### List 6: AIG Appointed Advisors

List6 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/List6.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List6.db")

List6.to_sql("List6", con, if_exists='replace')

List6.info()

vm11 = """SELECT a.*, b.FullName FROM AIGSubmit12092020_01272202GrBy a INNER JOIN List6 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

ConversionAIGList6 =  pysqldf(vm11)

Out4 = ConversionAIGList6.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ConversionAIGList6.csv', index = None, header=True)

### Let's Filter out the common advisors from List 5 and List 6

con = sqlite3.connect("ConversionAIGList5.db")

ConversionAIGList5.to_sql("ConversionAIGList5", con, if_exists='replace')

ConversionAIGList5.info()

con = sqlite3.connect("ConversionAIGList6.db")

ConversionAIGList6.to_sql("ConversionAIGList6", con, if_exists='replace')

vm11 = """SELECT a.* FROM ConversionAIGList5 a INNER JOIN ConversionAIGList6 b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonConversion =  pysqldf(vm11)

con = sqlite3.connect("CommonConversion.db")

CommonConversion.to_sql("CommonConversion", con, if_exists='replace')

p23  = """SELECT * FROM ConversionAIGList5 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonConversion);"""
        
ExternalC=  pysqldf(p23)

p23  = """SELECT * FROM ConversionAIGList6 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonConversion);"""
        
ExternalC1=  pysqldf(p23)

Out4 = ConversionAIGList5.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ConversionAIGList5.csv', index = None, header=True)

Out4 = CommonConversion.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/CommonConversion.csv', index = None, header=True)

Out4 = ExternalC.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ExternalC.csv', index = None, header=True)

Out4 = ExternalC1.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ExternalC1.csv', index = None, header=True)

### Previous 9 months AIG Submitted business

AIGSubmit03162020_012082020 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGSubmit03162020_012082020.csv',encoding= 'iso-8859-1')

AIGSubmit03162020_012082020.columns = AIGSubmit03162020_012082020.columns.str.replace(' ', '')

AIGSubmit03162020_012082020.columns = AIGSubmit03162020_012082020.columns.str.lstrip()

AIGSubmit03162020_012082020.columns = AIGSubmit03162020_012082020.columns.str.rstrip()

AIGSubmit03162020_012082020.columns = AIGSubmit03162020_012082020.columns.str.strip()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AIGSubmit03162020_012082020 group by AdvisorContactIDText, AdvisorName;"""

AIGSubmit03162020_012082020GrBy =  pysqldf(q3) 

### Lets review the overlap betwwen ExternalC vs. AIGSubmit03162020_012082020GrBy 

con = sqlite3.connect("ExternalC.db")

ExternalC.to_sql("ExternalC", con, if_exists='replace')

con = sqlite3.connect("AIGSubmit03162020_012082020GrBy.db")

AIGSubmit03162020_012082020GrBy.to_sql("AIGSubmit03162020_012082020GrBy", con, if_exists='replace')

vm11 = """SELECT a.*, b.SubmitCnt as Past9moSubmitCnt, b.SubmitAmt as Past9moSubmitAmt FROM ExternalC a INNER JOIN AIGSubmit03162020_012082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

commonAIGExternalC =  pysqldf(vm11)

Out4 = commonAIGExternalC.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/commonAIGExternalC.csv', index = None, header=True)

### Lets review the overlap betwwen CommonConversion vs. AIGSubmit03162020_012082020GrBy 

AIGSubmit03162020_012082020GrBy.info()

con = sqlite3.connect("CommonConversion.db")

CommonConversion.to_sql("CommonConversion", con, if_exists='replace')

vm11 = """SELECT a.*, b.SubmitCnt as Past9moSubmitCnt, b.SubmitAmt as Past9moSubmitAmt FROM CommonConversion a INNER JOIN AIGSubmit03162020_012082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonConversionAAA =  pysqldf(vm11)

Out4 = CommonConversionAAA.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/CommonConversionAAA.csv', index = None, header=True)

#### Let's understand the 29 Advisors contribution vs. 11 Advisors contribution

### Lets review the overlap betwwen ExternalC1 vs. AIGSubmit03162020_012082020GrBy 

con = sqlite3.connect("ExternalC1.db")

ExternalC1.to_sql("ExternalC1", con, if_exists='replace')

vm11 = """SELECT a.*, b.SubmitCnt as Past9moSubmitCnt, b.SubmitAmt as Past9moSubmitAmt FROM ExternalC1 a INNER JOIN AIGSubmit03162020_012082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

commonAIGExternalC1 =  pysqldf(vm11)

Out4 = commonAIGExternalC1.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/commonAIGExternalC1.csv', index = None, header=True)


### 22 Current producres--> List 5. Fallen angels over 90 days for 3 carriers including AIG, all appointed AIG producer
### 20 of them sold AIG between 03/16 and 12/08
### We picked up 2 new producers who are Fallen Angels from other carriers 

### 10 Fallen Angels
### 1 of them sold AIG between 03/16 and 12/08 
## 9 of them are true Fallen Angles activated...

### Bring the Email Data

EmailData = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailDataWithColumnExtended.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("EmailData.db")

EmailData.to_sql("EmailData", con, if_exists='replace')

EmailData.info()

q2  = """SELECT  sum(SendExist) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClick, sum(EngagedFinal) as Engaged, SubscriberKey, Name, EmailName FROM EmailData group by SubscriberKey;"""
      
EmailDataGroupBy =  pysqldf(q2)

q2  = """SELECT SubscriberKey, sum(SendExist) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClick, sum(EngagedFinal) as Engaged, SubscriberKey, Name, EmailName FROM EmailData group by SubscriberKey;"""
      
EmailDataGroupBy1 =  pysqldf(q2)

###Lets breakout the email data into carrier

con = sqlite3.connect("EmailDataGroupBy.db")

EmailDataGroupBy.to_sql("EmailDataGroupBy", con, if_exists='replace')

EmailDataGroupBy.info()

q2  = """SELECT * FROM EmailDataGroupBy where EmailName like '%ATH%' ;"""
      
EmailDataGroupByATH =  pysqldf(q2)

q2  = """SELECT * FROM EmailDataGroupBy where EmailName like '%NH%' ;"""
      
EmailDataGroupByNH =  pysqldf(q2)

q2  = """SELECT * FROM EmailDataGroupBy where EmailName like '%AIG%' ;"""
      
EmailDataGroupByAIG =  pysqldf(q2)

ExternalC.info()

EmailDataGroupByAIG.info()

### Email overlap with ExternalC

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ExternalC a INNER JOIN EmailDataGroupBy1 b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap =  pysqldf(vm11)

Out4 = EmailOverlap.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlap.csv', index = None, header=True)

### Email overlap with CommonConversion

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM CommonConversion a INNER JOIN EmailDataGroupBy1 b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap1 =  pysqldf(vm11)

Out4 = EmailOverlap1.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlap1.csv', index = None, header=True)

### Email overlap with ExternalC1

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ExternalC1 a INNER JOIN EmailDataGroupBy1 b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap2 =  pysqldf(vm11)

Out4 = EmailOverlap2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlap2.csv', index = None, header=True)

### Lets calculate the same with the AIG Segment

con = sqlite3.connect("EmailDataGroupByAIG.db")

EmailDataGroupByAIG.to_sql("EmailDataGroupByAIG", con, if_exists='replace')

### With ExternalC

index = ExternalC.index
number_of_rows = len(index)
print(number_of_rows)

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ExternalC a INNER JOIN EmailDataGroupByAIG b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlapAIG =  pysqldf(vm11)

Out4 = EmailOverlapAIG.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlapAIG.csv', index = None, header=True)

### With CommonConversion 

index1 = CommonConversion.index
number_of_rows1 = len(index1)
print(number_of_rows1)

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM CommonConversion a INNER JOIN EmailDataGroupByAIG b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap1AIG1 =  pysqldf(vm11)

Out4 = EmailOverlap1AIG1.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlap1AIG1.csv', index = None, header=True)

index2 = ExternalC1.index
number_of_rows2 = len(index2)
print(number_of_rows2)

##ExternalC1 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/ExternalC1.csv',encoding= 'iso-8859-1')

###con = sqlite3.connect("ExternalC1.db")

### ExternalC1.to_sql("ExternalC1", con, if_exists='replace')

index2 = ExternalC1.index
number_of_rows2 = len(index2)
print(number_of_rows2)

vm12 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ExternalC1 a INNER JOIN EmailDataGroupByAIG b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap1AIG2 =  pysqldf(vm12)

Out4 = EmailOverlap1AIG2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/EmailOverlap1AIG2.csv', index = None, header=True)

###########
###########
################# Do the same process for Nationwide ############################################

NWSubmit12092020_012722021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/NWSubmit12092020_01312021.csv',encoding= 'iso-8859-1')

NWSubmit12092020_012722021.columns = NWSubmit12092020_012722021.columns.str.replace(' ', '')

NWSubmit12092020_012722021.columns = NWSubmit12092020_012722021.columns.str.lstrip()

NWSubmit12092020_012722021.columns = NWSubmit12092020_012722021.columns.str.rstrip()

NWSubmit12092020_012722021.columns = NWSubmit12092020_012722021.columns.str.strip()

NWSubmit12092020_012722021['SubmitDate'] = pd.to_datetime(NWSubmit12092020_012722021['SubmitDate'])

con = sqlite3.connect("NWSubmit12092020_012722021.db")

NWSubmit12092020_012722021.to_sql("NWSubmit12092020_012722021", con, if_exists='replace')

NWSubmit12092020_012722021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM NWSubmit12092020_012722021 group by AdvisorContactIDText, AdvisorName;"""

NWSubmit12092020_012722021GrBy =  pysqldf(q3) 

### Lets Bring the list 2

List2 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/List2.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List2.db")

List2.to_sql("List2", con, if_exists='replace')

List2.info()

vm11 = """SELECT a.*, b.FullName FROM NWSubmit12092020_012722021GrBy a INNER JOIN List2 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

ConversionNWList2 =  pysqldf(vm11)

Out4 = ConversionNWList2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/ConversionNWList2.csv', index = None, header=True)

#### Let's look into previous one years NW submit and overlap between it

con = sqlite3.connect("ConversionNWList2.db")

ConversionNWList2.to_sql("ConversionNWList2", con, if_exists='replace')

NWSubmit12092019_12082020 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/NWSubmit12092019_12082020.csv',encoding= 'iso-8859-1')

NWSubmit12092019_12082020.columns = NWSubmit12092019_12082020.columns.str.replace(' ', '')

NWSubmit12092019_12082020.columns = NWSubmit12092019_12082020.columns.str.lstrip()

NWSubmit12092019_12082020.columns = NWSubmit12092019_12082020.columns.str.rstrip()

NWSubmit12092019_12082020.columns = NWSubmit12092019_12082020.columns.str.strip()

NWSubmit12092019_12082020['SubmitDate'] = pd.to_datetime(NWSubmit12092019_12082020['SubmitDate'])

con = sqlite3.connect("NWSubmit12092019_12082020.db")

NWSubmit12092019_12082020.to_sql("NWSubmit12092019_12082020", con, if_exists='replace')

NWSubmit12092020_012722021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM NWSubmit12092019_12082020 group by AdvisorContactIDText, AdvisorName;"""

NWSubmit12092019_12082020GrBy =  pysqldf(q3) 

con = sqlite3.connect("NWSubmit12092019_12082020GrBy.db")

NWSubmit12092019_12082020GrBy.to_sql("NWSubmit12092019_12082020GrBy", con, if_exists='replace')

### Now lets evaluate how many of the current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM ConversionNWList2 a INNER JOIN NWSubmit12092019_12082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcomp =  pysqldf(vm11)

##Out4 = CurrentvsPastcomp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/CurrentvsPastcomp.csv', index = None, header=True)


### As list 2 was current and non producers appointed with NW so that assume these were non producers

### Let's look into the last 4 years of the Any carrier data and check the overlap of the these submitted advisors during the ad campaign

Submit4YearsAllCarrier = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/Submit4YearsAllCarrier1.csv',encoding= 'iso-8859-1')

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.replace(' ', '')

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.lstrip()

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.rstrip()

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.strip()

Submit4YearsAllCarrier['SubmitDate'] = pd.to_datetime(Submit4YearsAllCarrier['SubmitDate'])

con = sqlite3.connect("Submit4YearsAllCarrier.db")

Submit4YearsAllCarrier.to_sql("Submit4YearsAllCarrier", con, if_exists='replace')

NWSubmit12092020_012722021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submit4YearsAllCarrier group by AdvisorContactIDText, AdvisorName;"""

Submit4YearsAllCarrierGrBy =  pysqldf(q3) 

con = sqlite3.connect("Submit4YearsAllCarrierGrBy.db")

Submit4YearsAllCarrierGrBy.to_sql("Submit4YearsAllCarrierGrBy", con, if_exists='replace')

vm11 = """SELECT a.* FROM ConversionNWList2 a INNER JOIN Submit4YearsAllCarrierGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcomp4yearallcarrier =  pysqldf(vm11)

#### true Non Producers

con = sqlite3.connect("CurrentvsPastcomp4yearallcarrier.db")

CurrentvsPastcomp4yearallcarrier.to_sql("CurrentvsPastcomp4yearallcarrier", con, if_exists='replace')

p23  = """SELECT * FROM ConversionNWList2 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CurrentvsPastcomp4yearallcarrier);"""
        
NWNonProducerEver=  pysqldf(p23)

Out4 = NWNonProducerEver.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/NWNonProducerEver.csv', index = None, header=True)

### CurrentvsPastcomp is 199 producers

con = sqlite3.connect("CurrentvsPastcomp.db")

CurrentvsPastcomp.to_sql("CurrentvsPastcomp", con, if_exists='replace')

### Subtracting 199 from the full converted list2 234 producers gives us the list of Fallen Angels and Non Producers activated

p23  = """SELECT * FROM ConversionNWList2 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CurrentvsPastcomp);"""
        
NWNonProducer_NotProdPast12m=  pysqldf(p23)

Out4 = NWNonProducer_NotProdPast12m.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/NWNonProducer_NotProdPast12m.csv', index = None, header=True)

### Lets substract the Non Producers from NWNonProducer_NotProdPast12m to get Fallen Angels

con = sqlite3.connect("NWNonProducerEver.db")

NWNonProducerEver.to_sql("NWNonProducerEver", con, if_exists='replace')

con = sqlite3.connect("NWNonProducer_NotProdPast12m.db")

NWNonProducer_NotProdPast12m.to_sql("NWNonProducer_NotProdPast12m", con, if_exists='replace')

p23  = """SELECT * FROM NWNonProducer_NotProdPast12m WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM NWNonProducerEver);"""
        
NWFallenAngelsActivation_12months=  pysqldf(p23)

Out4 = NWFallenAngelsActivation_12months.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/NWFallenAngelsActivation_12months.csv', index = None, header=True)


### Lets look into the email overlap

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ConversionNWList2 a INNER JOIN EmailDataGroupBy1 b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlapNW =  pysqldf(vm11)

Out4 = EmailOverlapNW.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/EmailOverlapNW.csv', index = None, header=True)


### Lets look into the email overlap with NW emails only

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ConversionNWList2 a INNER JOIN EmailDataGroupByNH b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap1NW_NWEmail =  pysqldf(vm11)

Out4 = EmailOverlap1NW_NWEmail.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/EmailOverlap1NW_NWEmail.csv', index = None, header=True)

#### Athene Performance

################# Do the same process for Athene ############################################

AtheneSubmitReport01112021_02182021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/SubmitReport01112021_02182021.csv',encoding= 'iso-8859-1')

AtheneSubmitReport01112021_02182021.columns = AtheneSubmitReport01112021_02182021.columns.str.replace(' ', '')

AtheneSubmitReport01112021_02182021.columns = AtheneSubmitReport01112021_02182021.columns.str.lstrip()

AtheneSubmitReport01112021_02182021.columns = AtheneSubmitReport01112021_02182021.columns.str.rstrip()

AtheneSubmitReport01112021_02182021.columns = AtheneSubmitReport01112021_02182021.columns.str.strip()

AtheneSubmitReport01112021_02182021['SubmitDate'] = pd.to_datetime(AtheneSubmitReport01112021_02182021['SubmitDate'])

con = sqlite3.connect("AtheneSubmitReport01112021_02182021.db")

AtheneSubmitReport01112021_02182021.to_sql("AtheneSubmitReport01112021_02182021", con, if_exists='replace')

AtheneSubmitReport01112021_02182021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmitReport01112021_02182021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmitReport01112021_02182021GrBy =  pysqldf(q3) 

### Lets Bring the list 3 for Athene

List3 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/List3.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List3.db")

List3.to_sql("List3", con, if_exists='replace')

List3.info()

vm11 = """SELECT a.*, b.FullName FROM AtheneSubmitReport01112021_02182021GrBy a INNER JOIN List3 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

ConversionAtheneList3 =  pysqldf(vm11)

Out4 = ConversionAtheneList3.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/ConversionAtheneList3.csv', index = None, header=True)

#### Let's look into previous one years Athene submit and overlap between it

con = sqlite3.connect("ConversionAtheneList3.db")

ConversionAtheneList3.to_sql("ConversionAtheneList3", con, if_exists='replace')

AtheneSubmitReport01102020_01102021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/SubmitReport01102020_01102021.csv',encoding= 'iso-8859-1')

AtheneSubmitReport01102020_01102021.columns = AtheneSubmitReport01102020_01102021.columns.str.replace(' ', '')

AtheneSubmitReport01102020_01102021.columns = AtheneSubmitReport01102020_01102021.columns.str.lstrip()

AtheneSubmitReport01102020_01102021.columns = AtheneSubmitReport01102020_01102021.columns.str.rstrip()

AtheneSubmitReport01102020_01102021.columns = AtheneSubmitReport01102020_01102021.columns.str.strip()

AtheneSubmitReport01102020_01102021['SubmitDate'] = pd.to_datetime(AtheneSubmitReport01102020_01102021['SubmitDate'])

con = sqlite3.connect("AtheneSubmitReport01102020_01102021.db")

AtheneSubmitReport01102020_01102021.to_sql("AtheneSubmitReport01102020_01102021", con, if_exists='replace')

AtheneSubmitReport01102020_01102021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmitReport01102020_01102021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmitReport01102020_01102021GrBy =  pysqldf(q3) 

con = sqlite3.connect("AtheneSubmitReport01102020_01102021GrBy.db")

AtheneSubmitReport01102020_01102021GrBy.to_sql("AtheneSubmitReport01102020_01102021GrBy", con, if_exists='replace')

### Now lets evaluate how many of the Athene current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM ConversionAtheneList3 a INNER JOIN AtheneSubmitReport01102020_01102021GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcompAthene =  pysqldf(vm11)

Out4 = CurrentvsPastcompAthene.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/CurrentvsPastcompAthene.csv', index = None, header=True)

### As list 3 was used with Atehne so that assume these were non producers

### Let's look into the last 4 years of the Any carrier data and check the overlap of the these submitted advisors during the ad campaign

## Becasuse this part was already processed we just need to take the data and 
vm11 = """SELECT a.* FROM ConversionAtheneList3 a INNER JOIN Submit4YearsAllCarrierGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcomp4yearallcarrierAthene =  pysqldf(vm11)

### Subtracting 220 from the full converted list3 240 producers gives us the list of Fallen Angels activated

con = sqlite3.connect("CurrentvsPastcompAthene.db")

CurrentvsPastcompAthene.to_sql("CurrentvsPastcompAthene", con, if_exists='replace')

p23  = """SELECT * FROM ConversionAtheneList3 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CurrentvsPastcompAthene);"""
        
AtheneFallenAngelsActivated=  pysqldf(p23)

Out4 = AtheneFallenAngelsActivated.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/AtheneFallenAngelsActivated.csv', index = None, header=True)


### Lets validate with Athene 4 years of data

AtheneSubmitReport01012016_01102021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/AtheneSubmitReport01012016_01102021.csv',encoding= 'iso-8859-1')

AtheneSubmitReport01012016_01102021.columns = AtheneSubmitReport01012016_01102021.columns.str.replace(' ', '')

AtheneSubmitReport01012016_01102021.columns = AtheneSubmitReport01012016_01102021.columns.str.lstrip()

AtheneSubmitReport01012016_01102021.columns = AtheneSubmitReport01012016_01102021.columns.str.rstrip()

AtheneSubmitReport01012016_01102021.columns = AtheneSubmitReport01012016_01102021.columns.str.strip()

AtheneSubmitReport01012016_01102021['SubmitDate'] = pd.to_datetime(AtheneSubmitReport01012016_01102021['SubmitDate'])

con = sqlite3.connect("AtheneSubmitReport01012016_01102021.db")

AtheneSubmitReport01012016_01102021.to_sql("AtheneSubmitReport01012016_01102021", con, if_exists='replace')

AtheneSubmitReport01012016_01102021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmitReport01012016_01102021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmitReport01012016_01102021GrBy =  pysqldf(q3) 

con = sqlite3.connect("AtheneSubmitReport01012016_01102021.db")

AtheneSubmitReport01012016_01102021GrBy.to_sql("AtheneSubmitReport01012016_01102021GrBy", con, if_exists='replace')

vm11 = """SELECT a.* FROM ConversionAtheneList3 a INNER JOIN AtheneSubmitReport01012016_01102021GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

ValidationAthene =  pysqldf(vm11)

con = sqlite3.connect("AtheneFallenAngelsActivated.db")

AtheneFallenAngelsActivated.to_sql("AtheneFallenAngelsActivated", con, if_exists='replace')

vm11 = """SELECT a.* FROM AtheneFallenAngelsActivated a INNER JOIN AtheneSubmitReport01012016_01102021GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

ValidationAthene1 =  pysqldf(vm11)

### Let's check the Athene Email overlap


vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ConversionAtheneList3 a INNER JOIN EmailDataGroupBy1 b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlapAthene =  pysqldf(vm11)

Out4 = EmailOverlapAthene.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/EmailOverlapAthene.csv', index = None, header=True)


### Lets look into the email overlap with NW emails only

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.TotalSend, b.TotalOpen, b.TotalClick, b.Engaged FROM ConversionAtheneList3 a INNER JOIN EmailDataGroupByNH b on ((a.AdvisorContactIDText=b.SubscriberKey));"""      

EmailOverlap1Athene_AtheneEmail =  pysqldf(vm11)

Out4 = EmailOverlap1Athene_AtheneEmail.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/EmailOverlap1Athene_AtheneEmail.csv', index = None, header=True)

#### Programmatic Ad Results

### List from Nathalea after removing IUl and NW appointed

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.csv',encoding= 'iso-8859-1')

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns = List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns.str.replace(' ', '')

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns = List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns.str.lstrip()

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns = List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns.str.rstrip()

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns = List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.columns.str.strip()

con = sqlite3.connect("List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.db")

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.to_sql("List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2", con, if_exists='replace')

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.info()

### AIG Email Appointed

AIGAppointedEmails = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/AIGAppointedEmails.csv',encoding= 'iso-8859-1')

AIGAppointedEmails.columns = AIGAppointedEmails.columns.str.replace(' ', '')

con = sqlite3.connect("AIGAppointedEmails.db")

AIGAppointedEmails.to_sql("AIGAppointedEmails", con, if_exists='replace')

AIGAppointedEmails.info()

vm11 = """SELECT a.*, b.* FROM List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 a INNER JOIN AIGAppointedEmails b on ((a.NPN= b.NPN) or (a.FullName=b.FullName));"""      

CommonAPN =  pysqldf(vm11)

## Submit Data

Submits12092020_03032021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/Submits12092020_03032021.csv',encoding= 'iso-8859-1')

Submits12092020_03032021.columns = Submits12092020_03032021.columns.str.replace(' ', '')

Submits12092020_03032021.info()

con = sqlite3.connect("Submits12092020_03032021.db")

Submits12092020_03032021.to_sql("Submits12092020_03032021", con, if_exists='replace')

Submits12092020_03032021.info()

q4  = """SELECT AdvisorName, AdvisorContactIDText, AdvisorContactAgentKey, AdvisorContactNPN, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submits12092020_03032021 group by AdvisorContactIDText, AdvisorName;"""

Submits12092020_03032021GrBy =  pysqldf(q4) 

con = sqlite3.connect("Submits12092020_03032021GrBy.db")

Submits12092020_03032021GrBy.to_sql("Submits12092020_03032021GrBy", con, if_exists='replace')

Submits12092020_03032021GrBy.info()

vm11 = """SELECT a.*, b.* FROM List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 a INNER JOIN Submits12092020_03032021GrBy b on ((a.NPN= b.AdvisorContactNPN) or (a.FullName=b.AdvisorName));"""      

CommonAPN1 =  pysqldf(vm11)

Out4 =CommonAPN1 .to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CommonAPN1.csv', index = None, header=True)


### App Data DW after 12092020

AppointedDataDWAfter12092020 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/AppointedDataDWAfter12092020.csv',encoding= 'iso-8859-1')

AppointedDataDWAfter12092020.columns = AppointedDataDWAfter12092020.columns.str.replace(' ', '')

con = sqlite3.connect("AppointedDataDWAfter12092020.db")

AppointedDataDWAfter12092020.to_sql("AppointedDataDWAfter12092020", con, if_exists='replace')

AppointedDataDWAfter12092020.info()

Submits12092020_03032021GrBy.info()

con = sqlite3.connect("CommonAPN1.db")

CommonAPN1.to_sql("CommonAPN1", con, if_exists='replace')

### This retrun null

## vm11 = """SELECT a.*, b.* FROM List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 a INNER JOIN AppointedDataDWAfter12092020 b on (a.FullName=b.FullName);"""      

###CommonAPN3 =  pysqldf(vm11)

vm11 = """SELECT a.*, b.* FROM CommonAPN1 a INNER JOIN AppointedDataDWAfter12092020 b on (a.AdvisorContactAgentKey =b.AdvisorKey );"""      

Common123 =  pysqldf(vm11)

### AIG AppointedFrom Landing

AIGAPpointedFromLanding = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/AIGAppointedfromDW03162021.csv',encoding= 'iso-8859-1')

AIGAPpointedFromLanding.columns = AIGAPpointedFromLanding.columns.str.replace(' ', '')

con = sqlite3.connect("AIGAPpointedFromLanding.db")

AIGAPpointedFromLanding.to_sql("AIGAPpointedFromLanding", con, if_exists='replace')

AIGAPpointedFromLanding.info()

vm11 = """SELECT a.*, b.* FROM List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 a INNER JOIN AIGAPpointedFromLanding b on ((a.NPN= b.PartyNPN));"""      

CommonAPN1111 =  pysqldf(vm11)

### Lets bring the original list 10

List10_A = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CopyofList10CleanedAIG_NonProtected_bw_JL.csv',encoding= 'iso-8859-1')

List10_A.columns = List10_A.columns.str.replace(' ', '')

con = sqlite3.connect("List10_A.db")

List10_A.to_sql("List10_A", con, if_exists='replace')

List10_A.info()

##vm11 = """SELECT a.*, b.* FROM List10_A a INNER JOIN AIGAPpointedFromLanding b on ((a.NPN= b.PartyNPN));"""      

##CommonAPN11112 =  pysqldf(vm11)

###Out4 =CommonAPN11112.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CommonAPN11112.csv', index = None, header=True)


### Replicate the method with Justins file

List10CleanedAIG_NonProtected_bw_JL= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CopyofList10CleanedAIG_NonProtected_bw_JL.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List10CleanedAIG_NonProtected_bw_JL.db")

List10CleanedAIG_NonProtected_bw_JL.to_sql("List10CleanedAIG_NonProtected_bw_JL", con, if_exists='replace')

List10CleanedAIG_NonProtected_bw_JL.info()

## Data1 

Data1 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/RequestedList_DiscoveryData_Athene_AllianzAppointedNonAnnexusAIGBDprotected.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Data1.db")

Data1.to_sql("Data1", con, if_exists='replace')

Data1.info()

qqq1  = """SELECT a.*, b.NPN FROM List10CleanedAIG_NonProtected_bw_JL a LEFT JOIN Data1 b on (a.Hash1=b.Hash1);"""
  
##qqq1  = """SELECT a.*, b.NPN FROM List10CleanedAIG_NonProtected_bw_JL a INNER JOIN List10_A b on (a.Primary_Address1=b.Primary_Address1 and ) ;"""
        
DataAppend2=  pysqldf(qqq1) 

Out4 =DataAppend2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/DataAppend2.csv', index = None, header=True)

con = sqlite3.connect("DataAppend2.db")

DataAppend2.to_sql("DataAppend2", con, if_exists='replace')

vm11 = """SELECT a.*, b.* FROM DataAppend2 a INNER JOIN AIGAPpointedFromLanding b on ((a.NPN= b.PartyNPN));"""      

CommonLeadAppAIG =  pysqldf(vm11)

Out4 =CommonLeadAppAIG.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CommonLeadAppAIG.csv', index = None, header=True)

List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2.info()

vm11 = """SELECT a.*, b.* FROM List10CleanedAIG_NonProtected_b_Appointmentsremoved_v2 a INNER JOIN AIGAPpointedFromLanding b on ((a.NPN= b.PartyNPN));"""      

CommonLeadAppAIG1 =  pysqldf(vm11)

### Lets bring the clean appointed and validate with Sunmits

CleanedAppointmentValidated= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CleanedAppointmentValidated.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CleanedAppointmentValidated.db")

CleanedAppointmentValidated.to_sql("CleanedAppointmentValidated", con, if_exists='replace')

vm11 = """SELECT b.* FROM CleanedAppointmentValidated a INNER JOIN Submits12092020_03032021GrBy b on ((a.PartyNPN= b.AdvisorContactNPN));"""      

CleanedAppointmentValidated_Sub =  pysqldf(vm11)

Out4= CleanedAppointmentValidated_Sub.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/CleanedAppointmentValidated_Sub.csv', index = None, header=True)

#### jason Email activation

AIGEmail02122020 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/AIGEmail02122020.csv',encoding= 'iso-8859-1')

AIGEmail02122020.columns = AIGEmail02122020.columns.str.replace(' ', '')

con = sqlite3.connect("AIGEmail02122020.db")

AIGEmail02122020.to_sql("AIGEmail02122020", con, if_exists='replace')

AIGEmail02122020.info()

Submit45daysAllCarriers = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/Submit45daysAllCarriers.csv',encoding= 'iso-8859-1')

Submit45daysAllCarriers.columns = Submit45daysAllCarriers.columns.str.replace(' ', '')

con = sqlite3.connect("Submit45daysAllCarriers.db")

Submit45daysAllCarriers.to_sql("Submit45daysAllCarriers", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt,  max(SubmitDate) as LastSubmitDate  
      FROM Submit45daysAllCarriers group by AdvisorContactIDText, AdvisorName;"""
      
Submit45daysAllCarriersgrBy =  pysqldf(q3)  

Submit45daysAllCarriersgrBy.info()

vm11 = """SELECT a.*, b.* FROM AIGEmail02122020 a INNER JOIN Submit45daysAllCarriersgrBy b on ((a.SubscriberKey= b.AdvisorContactIDText));"""      

EmailOverlapSubmit =  pysqldf(vm11)

Out4= EmailOverlapSubmit.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/EmailOverlapSubmit.csv', index = None, header=True)


Submit45daysAIG = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/Submit45daysAlG.csv',encoding= 'iso-8859-1')

Submit45daysAIG.columns = Submit45daysAIG.columns.str.replace(' ', '')

con = sqlite3.connect("Submit45daysAIG.db")

Submit45daysAIG.to_sql("Submit45daysAIG", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt,  max(SubmitDate) as LastSubmitDate  
      FROM Submit45daysAIG group by AdvisorContactIDText, AdvisorName;"""
      
Submit45daysAIGgrBy =  pysqldf(q3)  

Submit45daysAIGgrBy.info()

vm11 = """SELECT a.*, b.* FROM AIGEmail02122020 a INNER JOIN Submit45daysAIGgrBy b on ((a.SubscriberKey= b.AdvisorContactIDText));"""      

EmailOverlapSubmitAIG =  pysqldf(vm11)

Out4= EmailOverlapSubmitAIG.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AIGleads/EmailOverlapSubmitAIG.csv', index = None, header=True)

###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#############--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
####################-------------------------------------------------------------------------------------------------------------------------------------------------------------------

NWSubmits12092021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/Submits12092021_03012021.csv',encoding= 'iso-8859-1')

NWSubmits12092021.columns = NWSubmits12092021.columns.str.replace(' ', '')

NWSubmits12092021.columns = NWSubmits12092021.columns.str.lstrip()

NWSubmits12092021.columns = NWSubmits12092021.columns.str.rstrip()

NWSubmits12092021.columns = NWSubmits12092021.columns.str.strip()

NWSubmits12092021['SubmitDate'] = pd.to_datetime(NWSubmits12092021['SubmitDate'])

con = sqlite3.connect("NWSubmits12092021.db")

NWSubmits12092021.to_sql("NWSubmits12092021", con, if_exists='replace')

NWSubmits12092021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM NWSubmits12092021 group by AdvisorContactIDText, AdvisorName;"""
      

NWSubmits12092021GrBy =  pysqldf(q3) 

vm11 = """SELECT a.*, b.FullName FROM NWSubmits12092021GrBy a INNER JOIN List2 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

ConversionNWList2_ite2 =  pysqldf(vm11)

Out4 = ConversionNWList2_ite2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/ConversionNWList2_ite2.csv', index = None, header=True)

### Let's look into one year overlap

NWSubmit12092019_12082020GrBy.to_sql("NWSubmit12092019_12082020GrBy", con, if_exists='replace')

### Now lets evaluate how many of the current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM ConversionNWList2_ite2 a INNER JOIN NWSubmit12092019_12082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPast12mocomp =  pysqldf(vm11)

Out4 = CurrentvsPast12mocomp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/CurrentvsPast12mocomp.csv', index = None, header=True)

### As list 2 was current and non producers appointed with NW so that assume these were non producers

### Let's look into the last 4 years of the Any carrier data and check the overlap of the these submitted advisors during the ad campaign

Submit4YearsAllCarrier = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW/Submit4YearsAllCarrier1.csv',encoding= 'iso-8859-1')

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.replace(' ', '')

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.lstrip()

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.rstrip()

Submit4YearsAllCarrier.columns = Submit4YearsAllCarrier.columns.str.strip()

Submit4YearsAllCarrier['SubmitDate'] = pd.to_datetime(Submit4YearsAllCarrier['SubmitDate'])

con = sqlite3.connect("Submit4YearsAllCarrier.db")

Submit4YearsAllCarrier.to_sql("Submit4YearsAllCarrier", con, if_exists='replace')

NWSubmit12092020_012722021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submit4YearsAllCarrier group by AdvisorContactIDText, AdvisorName;"""

Submit4YearsAllCarrierGrBy =  pysqldf(q3) 

con = sqlite3.connect("Submit4YearsAllCarrierGrBy.db")

Submit4YearsAllCarrierGrBy.to_sql("Submit4YearsAllCarrierGrBy", con, if_exists='replace')

vm11 = """SELECT a.* FROM ConversionNWList2_ite2 a INNER JOIN Submit4YearsAllCarrierGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcomp4yearallcarrier_ite2 =  pysqldf(vm11)

#### true Non Producers

con = sqlite3.connect("CurrentvsPastcomp4yearallcarrier_ite2.db")

CurrentvsPastcomp4yearallcarrier_ite2.to_sql("CurrentvsPastcomp4yearallcarrier_ite2", con, if_exists='replace')

p23  = """SELECT * FROM ConversionNWList2_ite2 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CurrentvsPastcomp4yearallcarrier_ite2);"""
        
NWNonProducerEver_ite2=  pysqldf(p23)

Out4 = NWNonProducerEver_ite2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/NWNonProducerEver_ite2.csv', index = None, header=True)

### CurrentvsPastcomp is 199 producers

con = sqlite3.connect("CurrentvsPast12mocomp.db")

CurrentvsPast12mocomp.to_sql("CurrentvsPast12mocomp", con, if_exists='replace')

### Subtracting 265 from the full converted list2 234 producers gives us the list of Fallen Angels and Non Producers activated

p23  = """SELECT * FROM ConversionNWList2_ite2 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CurrentvsPast12mocomp);"""
        
NWNonProducer_NotProdPast12mfull=  pysqldf(p23)

Out4 = NWNonProducer_NotProdPast12mfull.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/NWNonProducer_NotProdPast12mfull.csv', index = None, header=True)

### Lets substract the Non Producers from NWNonProducer_NotProdPast12m to get Fallen Angels

con = sqlite3.connect("NWNonProducerEver_ite2.db")

NWNonProducerEver_ite2.to_sql("NWNonProducerEver_ite2", con, if_exists='replace')

con = sqlite3.connect("NWNonProducer_NotProdPast12mfull.db")

NWNonProducer_NotProdPast12mfull.to_sql("NWNonProducer_NotProdPast12mfull", con, if_exists='replace')

p23  = """SELECT * FROM NWNonProducer_NotProdPast12mfull WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM NWNonProducerEver_ite2);"""
        
NWFallenAngelsActivation_12monthsfull=  pysqldf(p23)

Out4 = NWFallenAngelsActivation_12monthsfull.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/NWFallenAngelsActivation_12monthsfull.csv', index = None, header=True)

####

####### Campign New Heights Select Launch (something big is coming) >> 9 (the wait is over)

######

Submit03012021_03082021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/Submit03012021_03082021.csv',encoding= 'iso-8859-1')

Submit03012021_03082021.columns = Submit03012021_03082021.columns.str.replace(' ', '')

Submit03012021_03082021.columns = Submit03012021_03082021.columns.str.lstrip()

Submit03012021_03082021.columns = Submit03012021_03082021.columns.str.rstrip()

Submit03012021_03082021.columns = Submit03012021_03082021.columns.str.strip()

Submit03012021_03082021['SubmitDate'] = pd.to_datetime(Submit03012021_03082021['SubmitDate'])

con = sqlite3.connect("Submit03012021_03082021.db")

Submit03012021_03082021.to_sql("Submit03012021_03082021", con, if_exists='replace')

Submit03012021_03082021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submit03012021_03082021 group by AdvisorContactIDText, AdvisorName;"""

NWSubmit03012021_03082021GrBy =  pysqldf(q3) 

vm11 = """SELECT a.*, b.FullName FROM NWSubmit03012021_03082021GrBy a INNER JOIN List2 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

Somethingbigiscoming  =  pysqldf(vm11)

Out4 = Somethingbigiscoming.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/Somethingbigiscoming.csv', index = None, header=True)

### Let's look into one year overlap

NWSubmit12092019_12082020GrBy.to_sql("NWSubmit12092019_12082020GrBy", con, if_exists='replace')

### Now lets evaluate how many of the current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM Somethingbigiscoming a INNER JOIN NWSubmit12092019_12082020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

SomethingbigiscomingCurrvsPast12mocomp =  pysqldf(vm11)

Out4 = SomethingbigiscomingCurrvsPast12mocomp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/SomethingbigiscomingCurrvsPast12mocomp.csv', index = None, header=True)

### Let's look into the last 4 years of the Any carrier data and check the overlap of the these submitted advisors during the ad campaign

vm11 = """SELECT a.* FROM Somethingbigiscoming a INNER JOIN Submit4YearsAllCarrierGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

Somethingbigiscoming4Years =  pysqldf(vm11)

#### true Non Producers

p23  = """SELECT * FROM Somethingbigiscoming WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Somethingbigiscoming4Years);"""
        
SomethingbigiscomingNWNonProducer=  pysqldf(p23)

### CurrentvsPastcomp is 199 producers

con = sqlite3.connect("SomethingbigiscomingCurrvsPast12mocomp.db")

SomethingbigiscomingCurrvsPast12mocomp.to_sql("SomethingbigiscomingCurrvsPast12mocomp", con, if_exists='replace')

### Subtracting 265 from the full converted list2 234 producers gives us the list of Fallen Angels and Non Producers activated

p23  = """SELECT * FROM Somethingbigiscoming WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM SomethingbigiscomingCurrvsPast12mocomp);"""
        
SomethingbigiscomingNWNonProducer_NotProdPast12mfull=  pysqldf(p23)

Out4 = SomethingbigiscomingNWNonProducer_NotProdPast12mfull.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/SomethingbigiscomingNWNonProducer_NotProdPast12mfull.csv', index = None, header=True)

### Lets substract the Non Producers from NWNonProducer_NotProdPast12m to get Fallen Angels

con = sqlite3.connect("SomethingbigiscomingNWNonProducer.db")

SomethingbigiscomingNWNonProducer.to_sql("SomethingbigiscomingNWNonProducer", con, if_exists='replace')

con = sqlite3.connect("SomethingbigiscomingNWNonProducer_NotProdPast12mfull.db")

SomethingbigiscomingNWNonProducer_NotProdPast12mfull.to_sql("SomethingbigiscomingNWNonProducer_NotProdPast12mfull", con, if_exists='replace')

p23  = """SELECT * FROM SomethingbigiscomingNWNonProducer_NotProdPast12mfull WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM SomethingbigiscomingNWNonProducer);"""
        
SomethingbigiscomingNWFallenAngelsActivation_12monthsfull=  pysqldf(p23)

Out4 = SomethingbigiscomingNWFallenAngelsActivation_12monthsfull.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NW04052021/SomethingbigiscomingNWFallenAngelsActivation_12monthsfull.csv', index = None, header=True)

##### NW Life

Submit02082020_041402020 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NWLife04052021/Submit02082020_041402020.csv',encoding= 'iso-8859-1')

Submit02082020_041402020.columns = Submit02082020_041402020.columns.str.replace(' ', '')

Submit02082020_041402020.columns = Submit02082020_041402020.columns.str.lstrip()

Submit02082020_041402020.columns = Submit02082020_041402020.columns.str.rstrip()

Submit02082020_041402020columns = Submit02082020_041402020.columns.str.strip()

Submit02082020_041402020['SubmitDate'] = pd.to_datetime(Submit02082020_041402020['SubmitDate'])

con = sqlite3.connect("Submit02082020_041402020.db")

Submit02082020_041402020.to_sql("Submit02082020_041402020", con, if_exists='replace')

Submit02082020_041402020.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submit02082020_041402020 group by AdvisorContactIDText, AdvisorName;"""
      

Submit02082020_041402020GrBy =  pysqldf(q3) 

###List 8 was all appointed Life Advisors

List8 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NWLife04052021/List8.csv',encoding= 'iso-8859-1')

List8.columns = List8.columns.str.replace(' ', '')

con = sqlite3.connect("List8.db")

List8.to_sql("List8", con, if_exists='replace')

List8.info()

vm11 = """SELECT a.*, b.FullName FROM Submit02082020_041402020GrBy a INNER JOIN List8 b on (a.AdvisorContactIDText=b.ContactID18);"""      

ConversionNWIULList8 =  pysqldf(vm11)

ConversionNWIULList8.info()

Out4 = ConversionNWIULList8.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NWLife04052021/ConversionNWIULList8.csv', index = None, header=True)

### Let's look into previous one years NW_IUL submit and overlap between it

con = sqlite3.connect("ConversionNWIULList8.db")

ConversionNWIULList8.to_sql("ConversionNWIULList8", con, if_exists='replace')

NWIULSubmit11012019_02072021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NWLife04052021/Submit11012019_02072021.csv',encoding= 'iso-8859-1')

NWIULSubmit11012019_02072021.columns = NWIULSubmit11012019_02072021.columns.str.replace(' ', '')

NWIULSubmit11012019_02072021.columns = NWIULSubmit11012019_02072021.columns.str.lstrip()

NWIULSubmit11012019_02072021.columns = NWIULSubmit11012019_02072021.columns.str.rstrip()

NWIULSubmit11012019_02072021.columns = NWIULSubmit11012019_02072021.columns.str.strip()

NWIULSubmit11012019_02072021['SubmitDate'] = pd.to_datetime(NWIULSubmit11012019_02072021['SubmitDate'])

con = sqlite3.connect("NWIULSubmit11012019_02072021.db")

NWIULSubmit11012019_02072021.to_sql("NWIULSubmit11012019_02072021", con, if_exists='replace')

NWIULSubmit11012019_02072021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM NWIULSubmit11012019_02072021 group by AdvisorContactIDText, AdvisorName;"""

NWIULSubmit11012019_02072021GrBy =  pysqldf(q3) 

con = sqlite3.connect("NWIULSubmit11012019_02072021GrBy.db")

NWIULSubmit11012019_02072021GrBy.to_sql("NWIULSubmit11012019_02072021GrBy", con, if_exists='replace')

NWIULSubmit11012019_02072021GrBy.info()

### Now lets evaluate how many of the Athene current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM ConversionNWIULList8 a INNER JOIN NWIULSubmit11012019_02072021GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CurrentvsPastcompNWIUL =  pysqldf(vm11)

Out4 = CurrentvsPastcompNWIUL.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/NWLife04052021/CurrentvsPastcompNWIUL.csv', index = None, header=True)


##### Athene 

#### There were three campaigns that were run

#### 10 Bonus Campaign- List 3 1/11/2021 (Ongoing)
#### Industry Leading- List 3  1/11/2021 (Ongoing)
#### 14 Bonus Campaign- List 3 1/11/2021 (Ongoing)

#### All the above ads were modified on 2/1/2021

#### Athene Performance

################# Do the same process for Athene ############################################

#### 10 Bonus Campaign- List 3 1/11/2021 (Ongoing)

AtheneSubmitReport04022021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene04052021/AtheneSubmits01112021_04022021.csv',encoding= 'iso-8859-1')

AtheneSubmitReport04022021.columns = AtheneSubmitReport04022021.columns.str.replace(' ', '')

AtheneSubmitReport04022021.columns = AtheneSubmitReport04022021.columns.str.lstrip()

AtheneSubmitReport04022021.columns = AtheneSubmitReport04022021.columns.str.rstrip()

AtheneSubmitReport04022021.columns = AtheneSubmitReport04022021.columns.str.strip()

AtheneSubmitReport04022021['SubmitDate'] = pd.to_datetime(AtheneSubmitReport04022021['SubmitDate'])

con = sqlite3.connect("AtheneSubmitReport04022021.db")

AtheneSubmitReport04022021.to_sql("AtheneSubmitReport04022021", con, if_exists='replace')

AtheneSubmitReport04022021.info()

q4  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM AtheneSubmitReport04022021 group by AdvisorContactIDText;"""

AtheneSubmitReport04022021GrBy =  pysqldf(q4) 

### Lets Bring the list 3 for Athene

List3 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene/List3.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("List3.db")

List3.to_sql("List3", con, if_exists='replace')

List3.info()

vm11 = """SELECT a.*, b.FullName FROM AtheneSubmitReport04022021GrBy a INNER JOIN List3 b on (a.AdvisorContactIDText=b.ContactID18);"""      

ConversionAtSubmit04022021List3 =  pysqldf(vm11)

Out4 = ConversionAtSubmit04022021List3.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene04052021/ConversionAtSubmit04022021List3.csv', index = None, header=True)

### Now lets evaluate how many of the Athene current producers also produced in the past 12 months..

vm11 = """SELECT a.*, b.SubmitCnt as earSubmitCnt, b.SubmitAmt as earSubmitAmt FROM ConversionAtSubmit04022021List3 a INNER JOIN AtheneSubmitReport01102020_01102021GrBy  b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

ConAtSubmit04022021CurvsPastcompAthene =  pysqldf(vm11)

Out4 = ConAtSubmit04022021CurvsPastcompAthene.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene04052021/ConAtSubmit04022021CurvsPastcompAthene.csv', index = None, header=True)

### As list 3 was used with Atehne so that assume these were non producers

### Let's look into the last 4 years of the Any carrier data and check the overlap of the these submitted advisors during the ad campaign

## Becasuse this part was already processed we just need to take the data and 

vm11 = """SELECT a.* FROM ConversionAtSubmit04022021List3 a INNER JOIN Submit4YearsAllCarrierGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText) and ((a.AdvisorName=b.AdvisorName)));"""      

CurrvsPastcomp4yearallcarAtSubmit04022021_AAA =  pysqldf(vm11)

Out4 = CurrvsPastcomp4yearallcarAtSubmit04022021_AAA.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene04052021/CurrvsPastcomp4yearallcarAtSubmit04022021_AAA.csv', index = None, header=True)

### Subtracting 385 from the full converted list3 345 producers gives us the list of Fallen Angels activated

con = sqlite3.connect("ConAtSubmit04022021CurvsPastcompAthene.db")

ConAtSubmit04022021CurvsPastcompAthene.to_sql("ConAtSubmit04022021CurvsPastcompAthene", con, if_exists='replace')

p23  = """SELECT * FROM ConversionAtSubmit04022021List3 WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM ConAtSubmit04022021CurvsPastcompAthene);"""
        
AtheneFallenAngelsActivated_April4th=  pysqldf(p23)

Out4 = AtheneFallenAngelsActivated_April4th.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/Athene04052021/AtheneFallenAngelsActivated_April4th.csv', index = None, header=True)

######## 

### Ron and Don's request to go deeper into the Fallen Angels and Non Producers

### Lets' bring the Non Producers

NWNonProducerEver = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWNonProducerEverOri.csv',encoding= 'iso-8859-1')

NWNonProducerEver.columns = NWNonProducerEver.columns.str.replace(' ', '')

NWNonProducerEver.columns = NWNonProducerEver.columns.str.lstrip()

NWNonProducerEver.columns = NWNonProducerEver.columns.str.rstrip()

NWNonProducerEver.columns = NWNonProducerEver.columns.str.strip()

NWNonProducerEver['LastSubmitDate'] = pd.to_datetime(NWNonProducerEver['LastSubmitDate'])

con = sqlite3.connect("NWNonProducerEver.db")

NWNonProducerEver.to_sql("NWNonProducerEver", con, if_exists='replace')

### Lets' bring the Appointed Advisors

AppointedAdvisors = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/AppointedAdvisors.csv',encoding= 'iso-8859-1')

AppointedAdvisors.columns = AppointedAdvisors.columns.str.replace(' ', '')

AppointedAdvisors.columns = AppointedAdvisors.columns.str.lstrip()

con = sqlite3.connect("AppointedAdvisors.db")

AppointedAdvisors.to_sql("AppointedAdvisors", con, if_exists='replace')

AppointedAdvisors.info()

vm11 = """SELECT a.*, b.CurrentNWIDCName, b.NPN, b.AnnuityMarketerFullName, b.FirstNWAdvApptStartDate2  FROM NWNonProducerEver a LEFT JOIN AppointedAdvisors b on ((a.AdvisorContactIDText=b.ContactID18));"""      

NWNonProducerEverIDCAp =  pysqldf(vm11)

Out4 = NWNonProducerEverIDCAp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWNonProducerEverIDCAp.csv', index = None, header=True)

### Lets' bring the Fallen Angels

NWFallenAngelOri = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWFallenAngelOri.csv',encoding= 'iso-8859-1')

NWFallenAngelOri.columns = NWFallenAngelOri.columns.str.replace(' ', '')

NWFallenAngelOri.columns = NWFallenAngelOri.columns.str.lstrip()

NWFallenAngelOri.columns = NWFallenAngelOri.columns.str.rstrip()

NWFallenAngelOri.columns = NWFallenAngelOri.columns.str.strip()

NWFallenAngelOri['LastSubmitDate'] = pd.to_datetime(NWFallenAngelOri['LastSubmitDate'])

con = sqlite3.connect("NWFallenAngelOri.db")

NWFallenAngelOri.to_sql("NWFallenAngelOri", con, if_exists='replace')

vm11 = """SELECT a.*, b.CurrentNWIDCName, b.NPN, b.AnnuityMarketerFullName, b.FirstNWAdvApptStartDate2  FROM NWFallenAngelOri a LEFT JOIN AppointedAdvisors b on ((a.AdvisorContactIDText=b.ContactID18));"""      

NWFallenAngelOriIDCAp =  pysqldf(vm11)

Out4 = NWFallenAngelOriIDCAp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWFallenAngelOriIDCAp.csv', index = None, header=True)

### Let's Bring the Paid Data From NW for the Ad Period 12/9/2020 and 3/1/2021 are reported

PaidData = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/PaidDataFrom12092020_03012021.csv',encoding= 'iso-8859-1')

PaidData.columns = PaidData.columns.str.replace(' ', '')

PaidData.info()

con = sqlite3.connect("PaidData.db")

PaidData.to_sql("PaidData", con, if_exists='replace')

vm11 = """SELECT SFContactId, FullName, count(PolicyNumber4) as NumberofPolicy, sum(TP) as TotalPremium, Max(SFNationwideLastSubmitDate) as LastSubmitDate, Max(CarrierReceivedDate) as LastCarrierReceivedDate, Max(IssueProcessedDate) as LastIssueProcessedDate, ProductName, ProductCode, ProductDescription FROM PaidData group by SFContactId;"""      

PaidDataGrBy =  pysqldf(vm11)

### Lets look into the overlap between this group with NonProducer NWNonProducerEverIDCAp

con = sqlite3.connect("PaidDataGrBy.db")

PaidDataGrBy.to_sql("PaidDataGrBy", con, if_exists='replace')

con = sqlite3.connect("NWNonProducerEverIDCAp.db")

NWNonProducerEverIDCAp.to_sql("NWNonProducerEverIDCAp", con, if_exists='replace')

vm11 = """SELECT a.*, b.*  FROM NWNonProducerEverIDCAp a INNER JOIN PaidDataGrBy b on ((a.AdvisorContactIDText=b.SFContactId));"""      

NWNonProducerEverIDCApWithPaid =  pysqldf(vm11)

Out4 = NWNonProducerEverIDCApWithPaid.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWNonProducerEverIDCApWithPaid.csv', index = None, header=True)

### Lets look into the overlap between this group with NWFallenAngels NWFallenAngelOriIDCAp

con = sqlite3.connect("NWFallenAngelOriIDCAp.db")

NWFallenAngelOriIDCAp.to_sql("NWFallenAngelOriIDCAp", con, if_exists='replace')

vm11 = """SELECT a.*, b.*  FROM NWFallenAngelOriIDCAp a INNER JOIN PaidDataGrBy b on ((a.AdvisorContactIDText=b.SFContactId));"""      

NWFallenAngelOriIDCAppWithPaid =  pysqldf(vm11)

Out4 = NWFallenAngelOriIDCAppWithPaid.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AdResutsRonDonFollowU/NWFallenAngelOriIDCAppWithPaid.csv', index = None, header=True)


### List 8 modifications
### "the life list is that the majority were not appointed to sell life, so it they have sold New Heights annuities in the last 3 years they should not be on the life list."

Submit01012018_07112021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/Submit01012018_07112021.csv',encoding= 'iso-8859-1')

Submit01012018_07112021.columns = Submit01012018_07112021.columns.str.replace(' ', '')

Submit01012018_07112021.columns = Submit01012018_07112021.columns.str.lstrip()

Submit01012018_07112021.columns = Submit01012018_07112021.columns.str.rstrip()

Submit01012018_07112021.columns = Submit01012018_07112021.columns.str.strip()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Submit01012018_07112021 group by AdvisorContactIDText, AdvisorName;"""

Submit01012018_07112021GrBy =  pysqldf(q3) 

List8= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/List8.csv',encoding= 'iso-8859-1')

List8.columns = List8.columns.str.replace(' ', '')

List8.columns = List8.columns.str.lstrip()

List8.columns = List8.columns.str.rstrip()

List8.columns = List8.columns.str.strip()

con = sqlite3.connect("Submit01012018_07112021GrBy.db")

Submit01012018_07112021GrBy.to_sql("Submit01012018_07112021GrBy", con, if_exists='replace')

con = sqlite3.connect("List8.db")

List8.to_sql("List8", con, if_exists='replace')

List8.info()

vm11 = """SELECT a.*, b.*  FROM Submit01012018_07112021GrBy a INNER JOIN List8 b on ((a.AdvisorContactIDText=b.ID));"""      

OverlapbetweenNHandLife =  pysqldf(vm11)

con = sqlite3.connect("OverlapbetweenNHandLife.db")

OverlapbetweenNHandLife.to_sql("OverlapbetweenNHandLife", con, if_exists='replace')

p23  = """SELECT * FROM List8 WHERE ID NOT IN (SELECT AdvisorContactIDText FROM OverlapbetweenNHandLife);"""
        
RemovedNWSubmitAdvFromList8=  pysqldf(p23)

con = sqlite3.connect("RemovedNWSubmitAdvFromList8.db")

RemovedNWSubmitAdvFromList8.to_sql("RemovedNWSubmitAdvFromList8", con, if_exists='replace')


Out4 = RemovedNWSubmitAdvFromList8.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/RemovedNWSubmitAdvFromList8.csv', index = None, header=True)

JasonsEmailSegmentLifeAppointed= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/JasonsEmailSegmentLifeAppointed.csv',encoding= 'iso-8859-1')

JasonsEmailSegmentLifeAppointed.columns = JasonsEmailSegmentLifeAppointed.columns.str.replace(' ', '')

JasonsEmailSegmentLifeAppointed.columns = JasonsEmailSegmentLifeAppointed.columns.str.lstrip()

JasonsEmailSegmentLifeAppointed.columns = JasonsEmailSegmentLifeAppointed.columns.str.rstrip()

JasonsEmailSegmentLifeAppointed.columns = JasonsEmailSegmentLifeAppointed.columns.str.strip()

con = sqlite3.connect("JasonsEmailSegmentLifeAppointed.db")

JasonsEmailSegmentLifeAppointed.to_sql("JasonsEmailSegmentLifeAppointed", con, if_exists='replace')

JasonsEmailSegmentLifeAppointed.info()

vm11 = """SELECT a.*, b.CurrentNWLifeAdvApptStartDate1  FROM RemovedNWSubmitAdvFromList8 a INNER JOIN JasonsEmailSegmentLifeAppointed b on ((a.ID=b.ContactID18));"""      

RemovedNWSubmitAdvFromList8StillApp =  pysqldf(vm11)

Out4 = RemovedNWSubmitAdvFromList8StillApp.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/RemovedNWSubmitAdvFromList8StillApp.csv', index = None, header=True)


### Athene result refresh 09/11/2021

#### Performance of 10 Bonus Campaign, Industry Leading Campaing and 14 Bonus Campaing

### These three campaigns started on 01/11/2021 and finished on 07/23/2021


AtheneSubmit01112021_07312021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmit01112021_07312021.csv',encoding= 'iso-8859-1')

AtheneSubmit01112021_07312021.columns = AtheneSubmit01112021_07312021.columns.str.replace(' ', '')

AtheneSubmit01112021_07312021.columns = AtheneSubmit01112021_07312021.columns.str.lstrip()

AtheneSubmit01112021_07312021.columns = AtheneSubmit01112021_07312021.columns.str.rstrip()

AtheneSubmit01112021_07312021.columns = AtheneSubmit01112021_07312021.columns.str.strip()

AtheneSubmit01112021_07312021['SubmitDate'] = pd.to_datetime(AtheneSubmit01112021_07312021['SubmitDate'])

con = sqlite3.connect("AtheneSubmit01112021_07312021.db")

AtheneSubmit01112021_07312021.to_sql("AtheneSubmit01112021_07312021", con, if_exists='replace')

AtheneSubmit01112021_07312021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmit01112021_07312021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmit01112021_07312021GrBy =  pysqldf(q3) 

### Lets Bring the list 3 for Athene

con = sqlite3.connect("AtheneSubmit01112021_07312021GrBy.db")

AtheneSubmit01112021_07312021GrBy.to_sql("AtheneSubmit01112021_07312021GrBy", con, if_exists='replace')

vm11 = """SELECT a.*, b.FullName FROM AtheneSubmit01112021_07312021GrBy a INNER JOIN List3 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AtheneSubmit01112021_07312021GrBy_List3 =  pysqldf(vm11)

Out4 = AtheneSubmit01112021_07312021GrBy_List3.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmit01112021_07312021GrBy_List3.csv', index = None, header=True)

### List 4

List4 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/ListSizeAfter10142020/FinalShareChrisShare/List4.csv',encoding= 'iso-8859-1')

List4.columns = List4.columns.str.replace(' ', '')

List4.columns = List4.columns.str.lstrip()

List4.columns = List4.columns.str.rstrip()

vm11 = """SELECT a.*, b.FullName FROM AtheneSubmit01112021_07312021GrBy a INNER JOIN List4 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AtheneSubmit01112021_07312021GrBy_List4 =  pysqldf(vm11)

Out4 = AtheneSubmit01112021_07312021GrBy_List4.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmit01112021_07312021GrBy_List4.csv', index = None, header=True)

### What is the overlap between these two segment

con = sqlite3.connect("AtheneSubmit01112021_07312021GrBy_List3.db")

AtheneSubmit01112021_07312021GrBy_List3.to_sql("AtheneSubmit01112021_07312021GrBy_List3", con, if_exists='replace')

AtheneSubmit01112021_07312021GrBy_List3.info()

con = sqlite3.connect("AtheneSubmit01112021_07312021GrBy_List4.db")

AtheneSubmit01112021_07312021GrBy_List4.to_sql("AtheneSubmit01112021_07312021GrBy_List4", con, if_exists='replace')

vm11 = """SELECT a.* FROM AtheneSubmit01112021_07312021GrBy_List3 a INNER JOIN AtheneSubmit01112021_07312021GrBy_List4 b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

AtheneSubmit01112021_07312021GrBy_List3_List4 =  pysqldf(vm11)

AtheneSubmit01112021_07312021GrBy_List4.info()

######

### This is realted to 

AtheneSubmits05192021_06102021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmits05192021_06102021.csv',encoding= 'iso-8859-1')

AtheneSubmits05192021_06102021.columns = AtheneSubmits05192021_06102021.columns.str.replace(' ', '')

AtheneSubmits05192021_06102021.columns = AtheneSubmits05192021_06102021.columns.str.lstrip()

AtheneSubmits05192021_06102021.columns = AtheneSubmits05192021_06102021.columns.str.rstrip()

AtheneSubmits05192021_06102021.columns = AtheneSubmits05192021_06102021.columns.str.strip()

AtheneSubmits05192021_06102021['SubmitDate'] = pd.to_datetime(AtheneSubmits05192021_06102021['SubmitDate'])

con = sqlite3.connect("AtheneSubmits05192021_06102021.db")

AtheneSubmits05192021_06102021.to_sql("AtheneSubmits05192021_06102021", con, if_exists='replace')

AtheneSubmits05192021_06102021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmits05192021_06102021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmits05192021_06102021GrBy =  pysqldf(q3) 

### Lets Bring the list 3 for Athene

con = sqlite3.connect("AtheneSubmits05192021_06102021GrBy.db")

AtheneSubmits05192021_06102021GrBy.to_sql("AtheneSubmits05192021_06102021GrBy", con, if_exists='replace')

vm11 = """SELECT a.*, b.FullName FROM AtheneSubmits05192021_06102021GrBy a INNER JOIN List4 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AtheneSubmits05192021_06102021GrBy_List4 =  pysqldf(vm11)

Out4 = AtheneSubmits05192021_06102021GrBy_List4.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmits05192021_06102021GrBy_List4.csv', index = None, header=True)

#### Now lets measure the overlap

con = sqlite3.connect("AtheneSubmits05192021_06102021GrBy_List4.db")

AtheneSubmits05192021_06102021GrBy_List4.to_sql("AtheneSubmits05192021_06102021GrBy_List4", con, if_exists='replace')

con = sqlite3.connect("AtheneSubmit01112021_07312021GrBy_List4.db")

AtheneSubmit01112021_07312021GrBy_List4.to_sql("AtheneSubmit01112021_07312021GrBy_List4", con, if_exists='replace')

vm11 = """SELECT a.*, b.* FROM AtheneSubmits05192021_06102021GrBy_List4 a INNER JOIN AtheneSubmit01112021_07312021GrBy_List4 b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonFrom_List4 =  pysqldf(vm11)

Out4 = CommonFrom_List4.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/CommonFrom_List4.csv', index = None, header=True)

### Shiller Allocator 6 

AtheneSubmit07232021_09302021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmit07232021_09302021.csv',encoding= 'iso-8859-1')

AtheneSubmit07232021_09302021.columns = AtheneSubmit07232021_09302021.columns.str.replace(' ', '')

AtheneSubmit07232021_09302021.columns = AtheneSubmit07232021_09302021.columns.str.lstrip()

AtheneSubmit07232021_09302021.columns = AtheneSubmit07232021_09302021.columns.str.rstrip()

AtheneSubmit07232021_09302021.columns = AtheneSubmit07232021_09302021.columns.str.strip()

AtheneSubmit07232021_09302021['SubmitDate'] = pd.to_datetime(AtheneSubmit07232021_09302021['SubmitDate'])

con = sqlite3.connect("AtheneSubmit07232021_09302021.db")

AtheneSubmit07232021_09302021.to_sql("AtheneSubmit07232021_09302021", con, if_exists='replace')

AtheneSubmit07232021_09302021.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM AtheneSubmit07232021_09302021 group by AdvisorContactIDText, AdvisorName;"""

AtheneSubmit07232021_09302021GrBy =  pysqldf(q3) 

con = sqlite3.connect("AtheneSubmit07232021_09302021GrBy.db")

AtheneSubmit07232021_09302021GrBy.to_sql("AtheneSubmit07232021_09302021GrBy", con, if_exists='replace')

vm11 = """SELECT a.*, b.* FROM AtheneSubmit07232021_09302021GrBy a INNER JOIN List2 b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AtheneSubmit07232021_09302021GrBy_List2 =  pysqldf(vm11)

Out4 = AtheneSubmit07232021_09302021GrBy_List2.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/AtheneSubmit07232021_09302021GrBy_List2.csv', index = None, header=True)


### Check on Fer and FerMax over time

Fer_FerMaxUniquePolicy2021 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMaxUniquePolicy2021.csv',encoding= 'iso-8859-1')

Fer_FerMaxUniquePolicy2021.columns = Fer_FerMaxUniquePolicy2021.columns.str.replace(' ', '')

Fer_FerMaxUniquePolicy2021.columns = Fer_FerMaxUniquePolicy2021.columns.str.lstrip()

Fer_FerMaxUniquePolicy2021.columns = Fer_FerMaxUniquePolicy2021.columns.str.rstrip()

Fer_FerMaxUniquePolicy2021.columns = Fer_FerMaxUniquePolicy2021.columns.str.strip()

con = sqlite3.connect("Fer_FerMaxUniquePolicy2021.db")

Fer_FerMaxUniquePolicy2021.to_sql("Fer_FerMaxUniquePolicy2021", con, if_exists='replace')

Fer_FerMaxUniquePolicy2021.info()


q3  = """SELECT FullName, Gender, SFContactId, count(PolicyNumber) as TotalPolicyCnt, sum(TP) as TotalPremium, max(IssueDate) as LastIssueDate, ProductFamilyName, ProductName,ProductCode, ProductDescription, ProductTypeCode, UnderlyingSecurityName, Carrier_Rider_Description, FundValueAmount, FeePercentage, GuaranteedInterestRate, FundPercentage, RiderType, RiderFamily, Bonus, Joint,SFExportRiderMarketingName,  ServiceFeatureProductCode  
      FROM Fer_FerMaxUniquePolicy2021 group by SFContactId, FullName;"""

Fer_FerMaxUniquePolicy2021GrBy =  pysqldf(q3) 

Fer_FerMaxUniquePolicy2021GrBy1 = Fer_FerMaxUniquePolicy2021GrBy[Fer_FerMaxUniquePolicy2021GrBy['SFContactId'].notnull()]

AtheneSubmit01112021_07312021GrBy_List3.info()
 
con = sqlite3.connect("AtheneSubmit01112021_07312021GrBy_List3.db")

AtheneSubmit01112021_07312021GrBy_List3.to_sql("AtheneSubmit01112021_07312021GrBy_List3", con, if_exists='replace')

con = sqlite3.connect("Fer_FerMaxUniquePolicy2021GrBy1.db")

Fer_FerMaxUniquePolicy2021GrBy1.to_sql("Fer_FerMaxUniquePolicy2021GrBy1", con, if_exists='replace')

AtheneSubmit01112021_07312021GrBy_List3.info()

q3  = """SELECT a.*, b.SubmitCnt, b.SubmitAmt, b.LastSubmitDate FROM Fer_FerMaxUniquePolicy2021GrBy1 a INNER JOIN AtheneSubmit01112021_07312021GrBy_List3 b on (a.SFContactId=b.AdvisorContactIDText);"""

Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit =  pysqldf(q3) 

Out4 = Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit.csv', index = None, header=True)

##### Fallen Angel Activations from Fer and FerMax

Fer_FerMax2020June_Dec= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMax2020June_Dec.csv',encoding= 'iso-8859-1')

Fer_FerMax2020June_Dec.columns = Fer_FerMax2020June_Dec.columns.str.replace(' ', '')

Fer_FerMax2020June_Dec.columns = Fer_FerMax2020June_Dec.columns.str.lstrip()

Fer_FerMax2020June_Dec.columns = Fer_FerMax2020June_Dec.columns.str.rstrip()

Fer_FerMax2020June_Dec.columns = Fer_FerMax2020June_Dec.columns.str.strip()

con = sqlite3.connect("Fer_FerMax2020June_Dec.db")

Fer_FerMax2020June_Dec.to_sql("Fer_FerMax2020June_Dec", con, if_exists='replace')

Fer_FerMax2020June_Dec.info()

q3  = """SELECT FullName, Gender, SFContactId, count(PolicyNumber) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate, ProductFamilyName, ProductName, ProductTypeCode, UnderlyingSecurityName, Carrier_Rider_Description, FundValueAmount, FeePercentage, GuaranteedInterestRate, FundPercentage   
      FROM Fer_FerMax2020June_Dec group by SFContactId, FullName;"""

Fer_FerMax2020June_DecGrBy =  pysqldf(q3)

#### 

Fer_FerMax2018_2019_2020JanJune= pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMax2018_2019_2020JanJune.csv',encoding= 'iso-8859-1')

Fer_FerMax2018_2019_2020JanJune.columns = Fer_FerMax2018_2019_2020JanJune.columns.str.replace(' ', '')

Fer_FerMax2018_2019_2020JanJune.columns = Fer_FerMax2018_2019_2020JanJune.columns.str.lstrip()

Fer_FerMax2018_2019_2020JanJune.columns = Fer_FerMax2018_2019_2020JanJune.columns.str.rstrip()

Fer_FerMax2018_2019_2020JanJune.columns = Fer_FerMax2018_2019_2020JanJune.columns.str.strip()

con = sqlite3.connect("Fer_FerMax2018_2019_2020JanJune.db")

Fer_FerMax2018_2019_2020JanJune.to_sql("Fer_FerMax2018_2019_2020JanJune", con, if_exists='replace')

Fer_FerMax2018_2019_2020JanJune.info()

q3  = """SELECT FullName, Gender, SFContactId, count(PolicyNumber) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate, ProductFamilyName, ProductName, ProductTypeCode, UnderlyingSecurityName, Carrier_Rider_Description, FundValueAmount, FeePercentage, GuaranteedInterestRate, FundPercentage   
      FROM Fer_FerMax2018_2019_2020JanJune group by SFContactId, FullName;"""

Fer_FerMax2018_2019_2020JanJuneGrBy =  pysqldf(q3)


### Now see the overlap these two groups


con = sqlite3.connect("Fer_FerMax2020June_DecGrBy.db")

Fer_FerMax2020June_DecGrBy.to_sql("Fer_FerMax2020June_DecGrBy", con, if_exists='replace')


con = sqlite3.connect("Fer_FerMax2018_2019_2020JanJuneGrBy.db")

Fer_FerMax2018_2019_2020JanJuneGrBy.to_sql("Fer_FerMax2018_2019_2020JanJuneGrBy", con, if_exists='replace')

vm11 = """SELECT a.* FROM Fer_FerMax2020June_DecGrBy a INNER JOIN Fer_FerMax2018_2019_2020JanJuneGrBy b on ((a.SFContactId=b.SFContactId));"""      

Common_ex =  pysqldf(vm11)

con = sqlite3.connect("Common_ex.db")

Common_ex.to_sql("Common_ex", con, if_exists='replace')

p23  = """SELECT * FROM Fer_FerMax2018_2019_2020JanJuneGrBy WHERE SFContactId NOT IN (SELECT SFContactId FROM Common_ex);"""
        
Fer_FerMaxFallenAngels=  pysqldf(p23)



con = sqlite3.connect("Fer_FerMaxFallenAngels.db")

Fer_FerMaxFallenAngels.to_sql("Fer_FerMaxFallenAngels", con, if_exists='replace')

### AtheneSubmit01112021_07312021GrBy_List3

vm11 = """SELECT a.*,b.* FROM AtheneSubmit01112021_07312021GrBy_List3 a INNER JOIN Fer_FerMaxFallenAngels b on (a.AdvisorContactIDText=b.SFContactId);"""      

Fer_FerMaxFallenAngels_reactivated =  pysqldf(vm11)

Out4 = Fer_FerMaxFallenAngels_reactivated.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMaxFallenAngels_reactivated.csv', index = None, header=True)


con = sqlite3.connect("Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit.db")

Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit.to_sql("Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit", con, if_exists='replace')

vm11 = """SELECT a.* FROM Fer_FerMaxUniquePolicy2021GrBy1_overlappedSubmit a INNER JOIN Fer_FerMaxFallenAngels b on (a.SFContactId=b.SFContactId);"""      

Fer_FerMaxFallenAngels_reactivated1 =  pysqldf(vm11)

Out4 = Fer_FerMaxFallenAngels_reactivated1.to_csv(r'C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdResults/AtheneResultsRefresh/Fer_FerMaxFallenAngels_reactivated1.csv', index = None, header=True)
