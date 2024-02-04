"""
"""
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

### FedEx Select9 

NH9Fedexmailing1 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/NH9FedexmailingAfterReturnProcessedByMR.csv',encoding= 'iso-8859-1')

NH9Fedexmailing1.columns = NH9Fedexmailing1.columns.str.replace(' ', '')

con = sqlite3.connect("NH9Fedexmailing1.db")

NH9Fedexmailing1.to_sql("NH9Fedexmailing1", con, if_exists='replace')

NH9Fedexmailing1.info()

### Lets do this

ListforNH9OnlyStates_NShare_Final_DanOri = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/ListforNH9OnlyStates_NShare_Final_DanOri.csv',encoding= 'iso-8859-1')

ListforNH9OnlyStates_NShare_Final_DanOri.columns = ListforNH9OnlyStates_NShare_Final_DanOri.columns.str.replace(' ', '')

con = sqlite3.connect("ListforNH9OnlyStates_NShare_Final_DanOri.db")

ListforNH9OnlyStates_NShare_Final_DanOri.to_sql("ListforNH9OnlyStates_NShare_Final_DanOri", con, if_exists='replace')

ListforNH9OnlyStates_NShare_Final_DanOri.info()

vm2 = """SELECT a.*,b.SFContactId FROM NH9Fedexmailing1 a INNER JOIN ListforNH9OnlyStates_NShare_Final_DanOri b on ((a.Hash= b.Hash));"""      

NHMailingListAdIDAppend =  pysqldf(vm2)

con = sqlite3.connect("NHMailingListAdIDAppend.db")

NHMailingListAdIDAppend.to_sql("NHMailingListAdIDAppend", con, if_exists='replace')

Out4 = NHMailingListAdIDAppend.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/NH9MailingListAdIDAppend.csv', index = None, header=True)

### Now this list contact SFID that Megan and Jason missed

NW04122021_06232021= pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/Submit04122021_06232021SeectFiltered.csv',encoding= 'iso-8859-1')

NW04122021_06232021.columns = NW04122021_06232021.columns.str.replace(' ', '')

q3  = """SELECT AdvisorName, AdvisorContactIDText, AccountName, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM NW04122021_06232021 group by AdvisorContactIDText, AdvisorName;"""     

SubmitRepor3monGrBy =  pysqldf(q3)  

SubmitRepor3monGrBy.info()

con = sqlite3.connect("SubmitRepor3monGrBy.db")

SubmitRepor3monGrBy.to_sql("SubmitRepor3monGrBy", con, if_exists='replace')

SubmitRepor3monGrBy.info()

vm2 = """SELECT a.*, b.* FROM NHMailingListAdIDAppend a INNER JOIN SubmitRepor3monGrBy b on ((a.SFContactId= b.AdvisorContactIDText) or (a.AdvisorName=b.AdvisorName));"""      

NH9FedexmailingActivation1 =  pysqldf(vm2)

Out4 = NH9FedexmailingActivation1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/NH9FedexmailingActivation1.csv', index = None, header=True)

#####

### Lets Validate the data with Webinar Attendance

#Marketer
WebinarMarketerData1 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210303MarketerWebinarSelectLaunchAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarMarketerData1.columns = WebinarMarketerData1.columns.str.replace(' ', '')

con = sqlite3.connect("WebinarMarketerData1.db")

WebinarMarketerData1.to_sql("WebinarMarketerData1", con, if_exists='replace')

ListofReenagedAdvisor_13 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/NH9FedexmailingActivation1.csv',encoding= 'iso-8859-1')

ListofReenagedAdvisor_13.columns = ListofReenagedAdvisor_13.columns.str.replace(' ', '')

con = sqlite3.connect("ListofReenagedAdvisor_13.db")

ListofReenagedAdvisor_13.to_sql("ListofReenagedAdvisor_13", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarMarketerData1 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.MarketerName));"""      

OverlapMarListofReengAdv =  pysqldf(vm1)


### First Advisor Webinar

WebinarData1 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210305AdvisorNationalSelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarData1.columns = WebinarData1.columns.str.replace(' ', '')

con = sqlite3.connect("WebinarData1.db")

WebinarData1.to_sql("WebinarData1", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarData1 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.AdvisorName));"""      

OverlapListofReengAdvData1 =  pysqldf(vm1)


### Second Advisor Webinar

WebinarData2 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210309AdvisorNationalSelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarData2.columns = WebinarData2.columns.str.replace(' ', '')

con = sqlite3.connect("WebinarData2.db")

WebinarData2.to_sql("WebinarData2", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarData2 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.AdvisorName));"""      

OverlapListofReengAdvData2 =  pysqldf(vm1)

### Third Advisor Webinar

WebinarData3 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210311AdvisorNationalSelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarData3.columns = WebinarData3.columns.str.replace(' ', '')

con = sqlite3.connect("WebinarData3.db")

WebinarData3.to_sql("WebinarData3", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarData3 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.AdvisorName));"""      

OverlapListofReengAdvData3 =  pysqldf(vm1)

### Fourth Advisor Webinar

WebinarData4 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210312AdvisorNationalSelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarData4.columns = WebinarData4.columns.str.replace(' ', '')

con = sqlite3.connect("WebinarData4.db")

WebinarData4.to_sql("WebinarData4", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarData4 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.AdvisorName));"""      

OverlapListofReengAdvData4 =  pysqldf(vm1)

### Select Marketer Analysis

WebinarData5Marketer = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210303MarketerWebinarSelectLaunchAttendeeReport.csv',encoding= 'iso-8859-1')

WebinarData5Marketer.columns = WebinarData5Marketer.columns.str.replace(' ', '')

WebinarData5Marketer.info()

con = sqlite3.connect("WebinarData5Marketer.db")

WebinarData5Marketer.to_sql("WebinarData5Marketer", con, if_exists='replace')

vm1 = """SELECT a.* FROM WebinarData5Marketer a INNER JOIN ListofReenagedAdvisor_13 b on ((a.MarketerName= b.AdvisorName));"""      

OverlapListofReengAdvData5 =  pysqldf(vm1)

### CA Marketer Webinar

CAMarketerWebinarData5 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210421MarketerWebinarCALaunchAttendeeReport.csv',encoding= 'iso-8859-1')

CAMarketerWebinarData5.columns = CAMarketerWebinarData5.columns.str.replace(' ', '')

con = sqlite3.connect("CAMarketerWebinarData5.db")

CAMarketerWebinarData5.to_sql("CAMarketerWebinarData5", con, if_exists='replace')

vm1 = """SELECT a.* FROM CAMarketerWebinarData5 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.CAMarketerName));"""      

OverlapCAMarketerWebinarData5 =  pysqldf(vm1)

### CA Advisor Webinar

CAWebinarData6 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210428AdvisorCASelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

CAWebinarData6.columns = CAWebinarData6.columns.str.replace(' ', '')

con = sqlite3.connect("CAWebinarData6.db")

CAWebinarData6.to_sql("CAWebinarData6", con, if_exists='replace')

vm1 = """SELECT a.* FROM CAWebinarData6 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.CAAdvisorName));"""      

OverlapCAWebinarData6 =  pysqldf(vm1)

### CA Advisor Webinar

CAWebinarData7 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210430AdvisorCASelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

CAWebinarData7.columns = CAWebinarData7.columns.str.replace(' ', '')

con = sqlite3.connect("CAWebinarData7.db")

CAWebinarData7.to_sql("CAWebinarData7", con, if_exists='replace')

vm1 = """SELECT a.* FROM CAWebinarData7 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.CAAdvisorName));"""      

OverlapCAWebinarData7 =  pysqldf(vm1)

### CA Advisor Webinar

CAWebinarData8 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Webinar/210506AdvisorCASelectLaunchWebinarAttendeeReport.csv',encoding= 'iso-8859-1')

CAWebinarData8.columns = CAWebinarData8.columns.str.replace(' ', '')

con = sqlite3.connect("CAWebinarData8.db")

CAWebinarData8.to_sql("CAWebinarData8", con, if_exists='replace')

vm1 = """SELECT a.* FROM CAWebinarData8 a INNER JOIN ListofReenagedAdvisor_13 b on ((b.AdvisorName= a.CAAdvisorName));"""      

OverlapCAWebinarData8 =  pysqldf(vm1)

### Let's look into Jason's Email data

EmailDataSelect = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Email/NWSelectLaunchEmailSends20210520aColumnExtended.csv',encoding= 'iso-8859-1')

EmailDataSelect.columns = EmailDataSelect.columns.str.replace(' ', '')

con = sqlite3.connect("EmailDataSelect.db")

EmailDataSelect.to_sql("EmailDataSelect", con, if_exists='replace')

### Now group by subscriberID

q2  = """SELECT  sum(SendExist) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClick, sum(EngagedFinal) as Engaged, SubscriberKey, Name, EmailName FROM EmailDataSelect group by SubscriberKey;"""
      
EmailDataSelectGroupBy =  pysqldf(q2)

ListofReenagedAdvisor_13.info()

con = sqlite3.connect("EmailDataSelectGroupBy.db")

EmailDataSelectGroupBy.to_sql("EmailDataSelectGroupBy", con, if_exists='replace')

vm1 = """SELECT a.*, b.TotalSend, b.TotalOpen,b.TotalClick, b.Engaged FROM ListofReenagedAdvisor_13 a INNER JOIN EmailDataSelectGroupBy b on ((a.AdvisorContactIDText= b.SubscriberKey));"""      

ListofReenagedAdvisor_13Email1 =  pysqldf(vm1)

Out4 = ListofReenagedAdvisor_13Email1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Email/ListofReenagedAdvisor_13Email1.csv', index = None, header=True)

### FedEx Select12

NH12Fedexmailing1 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select12/NH12OnlyStates_Final_MR deliveredUpdatedFinalWithRetRem.csv',encoding= 'iso-8859-1')

NH12Fedexmailing1.columns = NH12Fedexmailing1.columns.str.replace(' ', '')

con = sqlite3.connect("NH12Fedexmailing1.db")

NH12Fedexmailing1.to_sql("NH12Fedexmailing1", con, if_exists='replace')

NH12Fedexmailing1.info() 

vm1 = """SELECT a.*, b.SubmitCnt, b.SubmitAmt, b.LastSubmitDate FROM NH12Fedexmailing1 a INNER JOIN SubmitRepor3monGrBy b on ((a.AdvisorContactIDText = b.AdvisorContactIDText));"""      

NH12Fedexmailing1Activation =  pysqldf(vm1)

Out4 = NH12Fedexmailing1Activation.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/NH12Fedexmailing1Activation.csv', index = None, header=True)

### Lets check the Webinar 

con = sqlite3.connect("NH12Fedexmailing1Activation.db")

NH12Fedexmailing1Activation.to_sql("NH12Fedexmailing1Activation", con, if_exists='replace')

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN WebinarData1 b on ((b.AdvisorName= a.AdvisorName));"""      

NH12Fedexmailing1ActivationData1 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN WebinarData2 b on ((b.AdvisorName= a.AdvisorName));"""      

NH12Fedexmailing1ActivationData2 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN WebinarData3 b on ((b.AdvisorName= a.AdvisorName));"""      

NH12Fedexmailing1ActivationData3 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN WebinarData4  b on ((b.AdvisorName= a.AdvisorName));"""      

NH12Fedexmailing1ActivationData4 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN CAMarketerWebinarData5 b on ((a.AdvisorName= b.CAMarketerName));"""      

NH12Fedexmailing1ActivationData5 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN CAWebinarData6 b on ((a.AdvisorName= b.CAAdvisorName));"""      
  
NH12Fedexmailing1ActivationData6 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN CAWebinarData7 b on ((a.AdvisorName= b.CAAdvisorName));"""      
  
NH12Fedexmailing1ActivationData7 =  pysqldf(vm1)

vm1 = """SELECT a.* FROM NH12Fedexmailing1Activation a INNER JOIN CAWebinarData8 b on ((a.AdvisorName= b.CAAdvisorName));"""      
  
NH12Fedexmailing1ActivationData8 =  pysqldf(vm1)

#### Email overlap with Select 12

vm1 = """SELECT a.*, b.TotalSend, b.TotalOpen,b.TotalClick, b.Engaged FROM NH12Fedexmailing1Activation a INNER JOIN EmailDataSelectGroupBy b on ((a.AdvisorContactIDText= b.SubscriberKey));"""      

NH12Fedexmailing1ActivationEmail =  pysqldf(vm1)

Out4 = NH12Fedexmailing1ActivationEmail.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Email/NH12Fedexmailing1ActivationEmail.csv', index = None, header=True)

### FedEx CA Select Campaign #########################################################################################################################################################

CASelectFedexmailing = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/CASelect/CAListMailedAfterReturnedExcludedfromMegan.csv',encoding= 'iso-8859-1')

CASelectFedexmailing.columns = CASelectFedexmailing.columns.str.replace(' ', '')

con = sqlite3.connect("CASelectFedexmailing.db")

CASelectFedexmailing.to_sql("CASelectFedexmailing", con, if_exists='replace')

CASelectFedexmailing.info()

Submitreport05030201_06282021 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/Submitreport05030201_06282021.csv',encoding= 'iso-8859-1')

Submitreport05030201_06282021.columns = Submitreport05030201_06282021.columns.str.replace(' ', '')

con = sqlite3.connect("Submitreport05030201_06282021.db")

Submitreport05030201_06282021.to_sql("Submitreport05030201_06282021", con, if_exists='replace')

Submitreport05030201_06282021.info()

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate, ContactState, AdvisorContactMailingStateProvince, AdvisorContactHomeStateProvince  
      FROM Submitreport05030201_06282021 group by AdvisorContactIDText, AdvisorName;"""
      
Submitreport05030201_06282021GrBy=  pysqldf(q3)  

con = sqlite3.connect("Submitreport05030201_06282021GrBy.db")

Submitreport05030201_06282021GrBy.to_sql("Submitreport05030201_06282021GrBy", con, if_exists='replace')

vm1 = """SELECT a.NWLastSubmitDate, b.* FROM CASelectFedexmailing a INNER JOIN Submitreport05030201_06282021GrBy b on ((a.ContactID18= b.AdvisorContactIDText));"""      

CASelectMailingActivation =  pysqldf(vm1)

Out4 = CASelectMailingActivation.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/CASelectMailingActivation.csv', index = None, header=True)


## WebinarData1 --> 03/03

Submit03032021_06302021 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/Submit03032021_06302021.csv',encoding= 'iso-8859-1')

Submit03032021_06302021.columns = Submit03032021_06302021.columns.str.replace(' ', '')

con = sqlite3.connect("Submit03032021_06302021.db")

Submit03032021_06302021.to_sql("Submit03032021_06302021", con, if_exists='replace')

Submit03032021_06302021.info()

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate, ContactState, AdvisorContactMailingStateProvince, AdvisorContactHomeStateProvince  
      FROM Submit03032021_06302021 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitRepor3monGrBy1=  pysqldf(q3)  

con = sqlite3.connect("SubmitRepor3monGrBy1.db")

SubmitRepor3monGrBy1.to_sql("SubmitRepor3monGrBy1", con, if_exists='replace')

vm1 = """SELECT a.AdvisorName as AdvsiorName2, a.Attended, b.* FROM WebinarData1 a INNER JOIN SubmitRepor3monGrBy1 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Webinar1Submit1 =  pysqldf(vm1)

Out4 = Webinar1Submit1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Webinar/Webinar1Submit1.csv', index = None, header=True)

## WebinarData2 --> 03/05

vm1 = """SELECT a.AdvisorName as AdvsiorName2, a.Attended, b.* FROM WebinarData2 a INNER JOIN SubmitRepor3monGrBy1 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Webinar2Submit2 =  pysqldf(vm1)

Out4 = Webinar2Submit2.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Webinar/Webinar2Submit2.csv', index = None, header=True)

## WebinarData3 --> 03/11

vm1 = """SELECT a.AdvisorName as AdvsiorName2, a.Attended, b.* FROM WebinarData3 a INNER JOIN SubmitRepor3monGrBy1 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Webinar3Submit2 =  pysqldf(vm1)

Out4 = Webinar3Submit2.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Webinar/Webinar3Submit2.csv', index = None, header=True)

## WebinarData4 --> 03/12

vm1 = """SELECT a.AdvisorName as AdvsiorName2, a.Attended, b.* FROM WebinarData4 a INNER JOIN SubmitRepor3monGrBy1 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Webinar4Submit2 =  pysqldf(vm1)

Out4 = Webinar4Submit2.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Webinar/Webinar4Submit2.csv', index = None, header=True)

### Let's look into the overlap i.e. if Webinar1Submit1 attend Webinar 2

con = sqlite3.connect("Webinar1Submit1.db")

Webinar1Submit1.to_sql("Webinar1Submit1", con, if_exists='replace')

vm1 = """SELECT  b.* FROM Webinar1Submit1 a INNER JOIN WebinarData2 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Check1 =  pysqldf(vm1)

vm1 = """SELECT  b.* FROM Webinar1Submit1 a INNER JOIN WebinarData3 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Check2 =  pysqldf(vm1)

vm1 = """SELECT  b.* FROM Webinar1Submit1 a INNER JOIN WebinarData4 b on ((a.AdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

Check3 =  pysqldf(vm1)

### CA Advisor Webinar 6, 7 and 8

#CAWebinarData6, CAWebinarData7, CAWebinarData8

CASelectSubmit05032021_06182021 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Select06232021/CASelectSubmit05032021_06182021.csv',encoding= 'iso-8859-1')

CASelectSubmit05032021_06182021.columns = CASelectSubmit05032021_06182021.columns.str.replace(' ', '')

con = sqlite3.connect("CASelectSubmit05032021_06182021.db")

CASelectSubmit05032021_06182021.to_sql("CASelectSubmit05032021_06182021", con, if_exists='replace')

CASelectSubmit05032021_06182021.info()

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate, ContactState, AdvisorContactMailingStateProvince, AdvisorContactHomeStateProvince  
      FROM CASelectSubmit05032021_06182021 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitRepor3monGrBy3=  pysqldf(q3)  

con = sqlite3.connect("SubmitRepor3monGrBy3.db")

SubmitRepor3monGrBy3.to_sql("SubmitRepor3monGrBy3", con, if_exists='replace')


CAWebinarData6.info()

vm1 = """SELECT a.CAAdvisorName as AdvsiorName2, a.Attended, b.* FROM CAWebinarData6 a INNER JOIN SubmitRepor3monGrBy3 b on ((a.CAAdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

CAWebinar6Submit1 =  pysqldf(vm1)


vm1 = """SELECT a.CAAdvisorName as AdvsiorName2, a.Attended, b.* FROM CAWebinarData7 a INNER JOIN SubmitRepor3monGrBy3 b on ((a.CAAdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

CAWebinar7Submit1 =  pysqldf(vm1)

vm1 = """SELECT a.CAAdvisorName as AdvsiorName2, a.Attended, b.* FROM CAWebinarData8 a INNER JOIN SubmitRepor3monGrBy3 b on ((a.CAAdvisorName= b.AdvisorName)) where a.Attended='Yes' ;"""      

CAWebinar8Submit1 =  pysqldf(vm1)

### Email overlap with submits
EmailDataSelectGroupBy.info()

SubmitRepor3monGrBy.info()

vm1 = """SELECT a.*, b.* FROM EmailDataSelectGroupBy a INNER JOIN SubmitRepor3monGrBy1 b on ((a.SubscriberKey= b.AdvisorContactIDText));"""      

EmailDataSelectSubmit1 =  pysqldf(vm1)

Out4 = EmailDataSelectSubmit1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/Email/EmailDataSelectSubmit1.csv', index = None, header=True)

### Programmatic Ads

List2 = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/List2.csv',encoding= 'iso-8859-1')

List2.columns = List2.columns.str.replace(' ', '')

con = sqlite3.connect("List2.db")

List2.to_sql("List2", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM List2 a INNER JOIN SubmitRepor3monGrBy1 b on ((a.ContactID18 = b.AdvisorContactIDText));"""      

List2PrograSubmit =  pysqldf(vm1)

Out4 = List2PrograSubmit.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmit.csv', index = None, header=True)


##CA List

CAList = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AlexisResend/CAList.csv',encoding= 'iso-8859-1')

CAList.columns = CAList.columns.str.replace(' ', '')

con = sqlite3.connect("CAList.db")

List2.to_sql("CAList", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM CAList a INNER JOIN SubmitRepor3monGrBy3 b on ((a.ID = b.AdvisorContactIDText));"""      

CAListPrograSubmit =  pysqldf(vm1)

Out4 = CAListPrograSubmit.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/CAListPrograSubmit.csv', index = None, header=True)

### NonProducers

Year4SubmitData = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/4YearSubmitData.csv',encoding= 'iso-8859-1')

Year4SubmitData.columns = Year4SubmitData.columns.str.replace(' ', '')

con = sqlite3.connect("Year4SubmitData.db")

Year4SubmitData.to_sql("Year4SubmitData", con, if_exists='replace')

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Year4SubmitData group by AdvisorContactIDText, AdvisorName;"""
      
Year4SubmitDataGrBy3=  pysqldf(q3) 

con = sqlite3.connect("List2PrograSubmit.db")

List2PrograSubmit.to_sql("List2PrograSubmit", con, if_exists='replace')

vm1 = """SELECT a.AdvisorName as AdvisorName2, b.* FROM Year4SubmitDataGrBy3 a INNER JOIN List2PrograSubmit b on ((a.AdvisorContactIDText = b.AdvisorContactIDText));"""      

List2PrograSubmitCommon =  pysqldf(vm1)

Out4 = List2PrograSubmitCommon.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmitNonProd.csv', index = None, header=True)

con = sqlite3.connect("List2PrograSubmitCommon.db")

List2PrograSubmitCommon.to_sql("List2PrograSubmitCommon", con, if_exists='replace')


p23  = """SELECT * FROM List2PrograSubmit WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM List2PrograSubmitCommon);"""
        
List2PrograSubmitNonProducerAct=  pysqldf(p23)

Out4 = List2PrograSubmitNonProducerAct.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmitNonProducerAct.csv', index = None, header=True)

SubmitData2019 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Submit2019.csv',encoding= 'iso-8859-1')

SubmitData2019.columns = SubmitData2019.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitData2019.db")

SubmitData2019.to_sql("SubmitData2019", con, if_exists='replace')

SubmitData2019.info()

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate,ContactState,AdvisorContactMailingStateProvince,AdvisorContactHomeStateProvince FROM SubmitData2019 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitData2019grBy=  pysqldf(q3) 

Out4 = SubmitData2019grBy.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/SubmitData2019grBy.csv', index = None, header=True)

con = sqlite3.connect("List2PrograSubmit.db")

List2PrograSubmit.to_sql("List2PrograSubmit", con, if_exists='replace')

con = sqlite3.connect("SubmitData2019grBy.db")

SubmitData2019grBy.to_sql("SubmitData2019grBy", con, if_exists='replace')

SubmitData2019grBy.info()

List2PrograSubmit.info()

vm1 = """SELECT a.*, b.SubmitCnt, b.SubmitAmt, b.ContactState, b.LastSubmitDate FROM List2PrograSubmit a LEFT JOIN SubmitData2019grBy b on ((a.AdvisorContactIDText = b.AdvisorContactIDText));"""      

List2PrograSubmit_2019SubmitApp =  pysqldf(vm1)

vm1 = """SELECT a.*, b.SubmitCnt, b.SubmitAmt, b.ContactState, b.LastSubmitDate FROM List2PrograSubmit a INNER JOIN SubmitData2019grBy b on ((a.AdvisorContactIDText = b.AdvisorContactIDText));"""      

List2PrograSubmit_2019SubmitApp1 =  pysqldf(vm1)


Out4 = List2PrograSubmit_2019SubmitApp1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmit_2019SubmitApp1.csv', index = None, header=True)


### Now we need to check on the overlap of mailing list and the submit segment from Programmatic-submit
### Mailing list predominantly was Fallen Angels

##NH9 List

vm11 = """SELECT a.* FROM List2PrograSubmit a INNER JOIN NHMailingListAdIDAppend b on ((a.AdvisorContactIDText = b.SFContactId ));"""      

List2PrograSubmit_NH9MailinglistOver =  pysqldf(vm11)

Out4 = List2PrograSubmit_NH9MailinglistOver.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmit_NH9MailinglistOver.csv', index = None, header=True)



###NH12 List

vm12 = """SELECT a.* FROM List2PrograSubmit a INNER JOIN NH12Fedexmailing1 b on ((a.AdvisorContactIDText = b.AdvisorContactIDText ));"""      

List2PrograSubmit_NH12MailinglistOver =  pysqldf(vm12)

Out4 = List2PrograSubmit_NH12MailinglistOver.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmit_NH12MailinglistOver.csv', index = None, header=True)


### 2020 Submit data

SubmitData2020 = pd.read_csv('C:/Users/test/Documents/FedExMailingCampaigns/Submit2020.csv',encoding= 'iso-8859-1')

SubmitData2020.columns = SubmitData2020.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitData2020.db")

SubmitData2020.to_sql("SubmitData2020", con, if_exists='replace')

SubmitData2020.info()

q3  = """SELECT AdvisorName, AccountName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate,ContactState,AdvisorContactMailingStateProvince,AdvisorContactHomeStateProvince FROM SubmitData2020 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitData2020grBy=  pysqldf(q3) 

con = sqlite3.connect("SubmitData2020grBy.db")

SubmitData2020grBy.to_sql("SubmitData2020grBy", con, if_exists='replace')


vm1 = """SELECT a.*, b.SubmitCnt, b.SubmitAmt, b.LastSubmitDate, b.ContactState, b.LastSubmitDate  FROM List2PrograSubmit a INNER JOIN SubmitData2020grBy b on ((a.AdvisorContactIDText = b.AdvisorContactIDText));"""      

List2PrograSubmit_2020SubmitApp1 =  pysqldf(vm1)

Out4 = List2PrograSubmit_2020SubmitApp1.to_csv(r'C:/Users/test/Documents/FedExMailingCampaigns/List2PrograSubmit_2020SubmitApp1.csv', index = None, header=True)






