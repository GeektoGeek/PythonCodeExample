
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


AtheneAppointed= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Atheneallappointed08082021.csv',encoding= 'iso-8859-1')

AtheneAppointed.columns = AtheneAppointed.columns.str.replace(' ', '')

con = sqlite3.connect("AtheneAppointed.db")

AtheneAppointed.to_sql("AtheneAppointed", con, if_exists='replace')

AtheneAppointed.info()

AtheneAppointed['AdvisorContactIDText']= AtheneAppointed['ContactID18']

AtheneAppointed['AtheneSalesYTD'] = AtheneAppointed['AnnuitySalesYTD']


AtheneAppointed['AthenePriorYearAnnuitySales'] = AtheneAppointed['PriorYearAnnuitySales']

AtheneAppointed.info()


NWAppointed= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NWallappointed08082021.csv',encoding= 'iso-8859-1')

NWAppointed.columns = NWAppointed.columns.str.replace(' ', '')

NWAppointed.info()

NWAppointed['NWSalesYTD'] = AtheneAppointed['AnnuitySalesYTD']


NWAppointed['NWPriorYearAnnuitySales'] = NWAppointed['PriorYearAnnuitySales']

NWAppointed.info()

con = sqlite3.connect("NWAppointed.db")

NWAppointed.to_sql("NWAppointed", con, if_exists='replace')

NWAppointed.info()

vm1 = """SELECT a.*, b.* FROM AtheneAppointed a INNER JOIN NWAppointed b on (a.ContactID18 = b.ContactID18);"""      

AtheneApp_NWApp =  pysqldf(vm1)

Out4 = AtheneApp_NWApp.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneApp_NWApp.csv', index = None, header=True)

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
from plotly.offline import plot

iris= px.data.iris()

AtheneApp_NWApp.info()

Test=AtheneApp_NWApp[['FullName', 'ContactID18', 'NWSalesYTD','NWPriorYearAnnuitySales']]
Test.reset_index(drop=True, inplace=True)
Test1= Test.iloc[:,[0,2,4,5]]

Test1.info()


##Test3= Test1.groupby(['NWSalesYTD','FullName']).sum()

Test3= Test1.groupby(['ContactID18','FullName'])['NWSalesYTD', 'NWPriorYearAnnuitySales'].sum().reset_index()

fig = px.scatter(Test3, x='FullName', y='NWSalesYTD')

fig = px.line(Test3, x='FullName', y=['NWSalesYTD', 'NWPriorYearAnnuitySales'])

#fig = px.line(df, x='Date', y=['AAPL.High', 'AAPL.Low'])
plot(fig)

"""
import plotly.graph_objs as go
import pandas as pd

fig = go.Figure([
go.Scatter(
        name='Upper Bound',
        x=Test3['FullName'],
        y=Test3['NWSalesYTD']+Test3['PriorYearNWSales'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=1),
        showlegend=False
    )
        ])
fig.update_layout(
    yaxis_title='Wind speed (m/s)',
    title='Continuous, variable value error bars',
    hovermode="x"
)
fig.show()

plot(fig)

"""

### Lets' bring the Athene paid data with Athene Appointed Advisors

AthenePaidData= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AthenePaid2015_2021UniquePol.csv',encoding= 'iso-8859-1')

AthenePaidData.columns = AthenePaidData.columns.str.replace(' ', '')

con = sqlite3.connect("AthenePaidData.db")

AthenePaidData.to_sql("AthenePaidData", con, if_exists='replace')

AthenePaidData.info()

q3  = """SELECT SFContactId, FullName, Gender, SFMailingState, SFAtheneLastSubmitIMO, ProductCode, ProductName, ProductDescription, ProductTypeDescription, ProductFamilyCode, PlanName, MarketingName, count(PolicyNumber4) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate FROM AthenePaidData group by SFContactId, FullName  ;"""     

AthenePaidDatagrby =  pysqldf(q3)  

con = sqlite3.connect("AthenePaidDatagrby.db")

AthenePaidDatagrby.to_sql("AthenePaidDatagrby", con, if_exists='replace')

AthenePaidDatagrby.info()

#SFContactId

vm1 = """SELECT  b.* FROM AtheneAppointed a INNER JOIN AthenePaidDatagrby b on (a.ContactID18 = b.SFContactId);"""      

AtheneApp_PaidHist =  pysqldf(vm1)

Out4 = AtheneApp_PaidHist.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneApp_PaidHist.csv', index = None, header=True)


### Lets' bring the Athene Submitted data with Athene Appointed Advisors

Athene2015_2021SubmittedBusiness= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athene2015_2021SubmittedBusiness.csv',encoding= 'iso-8859-1')

Athene2015_2021SubmittedBusiness.columns = Athene2015_2021SubmittedBusiness.columns.str.replace(' ', '')

con = sqlite3.connect("Athene2015_2021SubmittedBusiness.db")

Athene2015_2021SubmittedBusiness.to_sql("Athene2015_2021SubmittedBusiness", con, if_exists='replace')

Athene2015_2021SubmittedBusiness.info()

Athene2015_2021SubmittedBusiness['SubmitAmount'].sum()

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, AccountName, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate,  MarketerName FROM Athene2015_2021SubmittedBusiness group by AdvisorContactIDText;"""
      
Athene2015_2021SubmittedBusinessGrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Athene2015_2021SubmittedBusiness group by AdvisorContactIDText, AdvisorName;"""
      
Athene2015_2021SBG1=  pysqldf(q3) 

Out4 = Athene2015_2021SBG1.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athene2015_2021SBG1.csv', index = None, header=True)


Athene2015_2021SubmittedBusinessGrBy['SubmitAmt'].sum()

con = sqlite3.connect("Athene2015_2021SubmittedBusinessGrBy.db")

Athene2015_2021SubmittedBusinessGrBy.to_sql("Athene2015_2021SubmittedBusinessGrBy", con, if_exists='replace')

Athene2015_2021SubmittedBusinessGrBy.info()

Athene2015_2021SubmittedBusinessGrBy['SubmitAmt'].sum()

Athene2015_2021SubmittedBusinessGrBy['AdvisorContactIDText'].count()

Out4 = Athene2015_2021SubmittedBusinessGrBy.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athene2015_2021SubmittedBusinessGrBy.csv', index = None, header=True)

AtheneAppointed.info()

#SFContactId

vm1 = """SELECT a.CurrentAtheneAdvApptStartDate1, a.FirstAtheneAdvApptStartDate2, b.* FROM AtheneAppointed a INNER JOIN Athene2015_2021SubmittedBusinessGrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

Athene2015_2021SubmittedBusinessGrBy_Check =  pysqldf(vm1)

con = sqlite3.connect("Athene2015_2021SubmittedBusinessGrBy_Check.db")

Athene2015_2021SubmittedBusinessGrBy_Check.to_sql("Athene2015_2021SubmittedBusinessGrBy_Check", con, if_exists='replace')


p23  = """SELECT * FROM Athene2015_2021SubmittedBusinessGrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Athene2015_2021SubmittedBusinessGrBy_Check);"""
        
AtheneAdvisorPastSubmittedNoLongerApp=  pysqldf(p23)


Out4 = Athene2015_2021SubmittedBusinessGrBy.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athene2015_2021SubmittedBusinessGrBy.csv', index = None, header=True)


Out4 = AtheneAdvisorPastSubmittedNoLongerApp.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneAdvisorPastSubmittedNoLongerApp.csv', index = None, header=True)

### Lets try to get the full segment between Athene appointed and 

vm1 = """SELECT a.CurrentAtheneAdvApptStartDate1, a.FirstAtheneAdvApptStartDate2, b.* FROM AtheneAppointed a INNER JOIN Athene2015_2021SubmittedBusiness b on (a.ContactID18 = b.AdvisorContactIDText);"""      

Athene2015_2021SubmittedBusinessCheck1 =  pysqldf(vm1)

Out4 = Athene2015_2021SubmittedBusinessCheck1.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athene2015_2021SubmittedBusinessCheck1.csv', index = None, header=True)

AtheneAdvisorPastSubmittedNoLongerApp.info()

con = sqlite3.connect("AtheneAdvisorPastSubmittedNoLongerApp.db")

AtheneAdvisorPastSubmittedNoLongerApp.to_sql("AtheneAdvisorPastSubmittedNoLongerApp", con, if_exists='replace')

### Check with All Appinted table in the DW

AppointedTableDW=  pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AppointedTableDW.csv',encoding= 'iso-8859-1')

AppointedTableDW.columns = AppointedTableDW.columns.str.replace(' ', '')

con = sqlite3.connect("AppointedTableDW.db")

AppointedTableDW.to_sql("AppointedTableDW", con, if_exists='replace')

AppointedTableDW.info()

vm1 = """SELECT a.AdvisorContactIDText, a.AdvisorName, b.AppointmentStatusCode, b.AppointmentStatusReasonCode FROM AtheneAdvisorPastSubmittedNoLongerApp a LEFT JOIN AppointedTableDW b on (a.AdvisorContactIDText = b.SFAgentCId);"""      

AtheneAdvisorPastSubmittedNoLongerAppwithRea =  pysqldf(vm1)


###

AppointmentHistoryTableDWUpdated=  pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AppointmentHistoryTableDWUpdated.csv',encoding= 'iso-8859-1')

AppointmentHistoryTableDWUpdated.columns = AppointmentHistoryTableDWUpdated.columns.str.replace(' ', '')

con = sqlite3.connect("AppointmentHistoryTableDWUpdated.db")

AppointmentHistoryTableDWUpdated.to_sql("AppointmentHistoryTableDWUpdated", con, if_exists='replace')

AppointmentHistoryTableDWUpdated.info()

vm1 = """SELECT a.AdvisorContactIDText, a.AdvisorName, b.AppointmentStatusCode, b.AppointmentStatusReasonCode FROM AtheneAdvisorPastSubmittedNoLongerApp a INNER JOIN AppointmentHistoryTableDWUpdated b on (a.AdvisorContactIDText = b.SFAgentCId);"""      

AtheneAdvisorPastSubmittedNoLongerAppwithRea1 =  pysqldf(vm1)

### Let's look into NW Appointed 

Nationwide2015_2021SubmittedBusiness=  pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwide2015_2021SubmittedBusiness.csv',encoding= 'iso-8859-1')

Nationwide2015_2021SubmittedBusiness.columns = Nationwide2015_2021SubmittedBusiness.columns.str.replace(' ', '')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness.db")

Nationwide2015_2021SubmittedBusiness.to_sql("Nationwide2015_2021SubmittedBusiness", con, if_exists='replace')

Nationwide2015_2021SubmittedBusiness.info()

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusinessGrBy=  pysqldf(q3) 

con = sqlite3.connect("Nationwide2015_2021SubmittedBusinessGrBy.db")

Nationwide2015_2021SubmittedBusinessGrBy.to_sql("Nationwide2015_2021SubmittedBusinessGrBy", con, if_exists='replace')

Out4 = Nationwide2015_2021SubmittedBusinessGrBy.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwide2015_2021SubmittedBusinessGrBy.csv', index = None, header=True)


### Overlap of NW Appointed and Submitted NW Advisors..

NWAppointed.info()

vm1 = """SELECT a.CurrentNWAdvApptStartDate1, a.FirstNWAdvApptStartDate2, b.* FROM NWAppointed a INNER JOIN Nationwide2015_2021SubmittedBusinessGrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

Nationwide2015_2021SubmittedBusinessGrBy_Check =  pysqldf(vm1)

Out4 = Nationwide2015_2021SubmittedBusinessGrBy_Check.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwide2015_2021SubmittedBusinessGrBy_Check.csv', index = None, header=True)


con = sqlite3.connect("Nationwide2015_2021SubmittedBusinessGrBy_Check.db")

Nationwide2015_2021SubmittedBusinessGrBy_Check.to_sql("Nationwide2015_2021SubmittedBusinessGrBy_Check", con, if_exists='replace')


p23  = """SELECT * FROM Nationwide2015_2021SubmittedBusinessGrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Nationwide2015_2021SubmittedBusinessGrBy_Check);"""
        
NWAdvisorPastSubmittedNoLongerApp=  pysqldf(p23)

Out4 = NWAdvisorPastSubmittedNoLongerApp.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NWAdvisorPastSubmittedNoLongerApp.csv', index = None, header=True)

###  Take some of the paid data:

## Athene --> Athene Non Producer

## AtheneAppointed

AtheneAppointed.info()

Athene2015_2021SubmittedBusinessGrBy_Check.info()

### Athene2015_2021SubmittedBusinessGrBy_Check

## AtheneNonProducer

AtheneAppointed.info()

Athene2015_2021SBG1.info()


vm1 = """SELECT b.* FROM AtheneAppointed a INNER JOIN Athene2015_2021SBG1 b on (a.ContactID18 = b.AdvisorContactIDText);"""      

AthenAppointvsAtheneSubOperlap =  pysqldf(vm1)

con = sqlite3.connect("AthenAppointvsAtheneSubOperlap.db")

AthenAppointvsAtheneSubOperlap.to_sql("AthenAppointvsAtheneSubOperlap", con, if_exists='replace')


p23  = """SELECT * FROM AtheneAppointed WHERE ContactID18 NOT IN (SELECT AdvisorContactIDText FROM AthenAppointvsAtheneSubOperlap);"""
        
AtheneAdvisorNonProducer=  pysqldf(p23)

Out4 = AtheneAdvisorNonProducer.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneAdvisorNonProducer.csv', index = None, header=True)

Out4 = AthenAppointvsAtheneSubOperlap.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AthenAppointvsAtheneSubOperlap.csv', index = None, header=True)


## NWNonProducer
NWAppointed.info()

Nationwide2015_2021SubmittedBusinessGrBy.info()

vm1 = """SELECT b.* FROM NWAppointed a INNER JOIN Nationwide2015_2021SubmittedBusinessGrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

NWAppointvsNWSubOperlap =  pysqldf(vm1)

p23  = """SELECT * FROM NWAppointed WHERE ContactID18 NOT IN (SELECT AdvisorContactIDText FROM NWAppointvsNWSubOperlap);"""
        
NWAdvisorNonProducer=  pysqldf(p23)

### AtheneApp_NWApp

### Common Non Producers
### A group of advisors are non producers for both NW and Athene

AtheneApp_NWApp.info()

NWAtheneSubmitted01012018_07282021= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NWAtheneSubmitted01012018_07282021.csv',encoding= 'iso-8859-1')

NWAtheneSubmitted01012018_07282021.columns = NWAtheneSubmitted01012018_07282021.columns.str.replace(' ', '')

con = sqlite3.connect("NWAtheneSubmitted01012018_07282021.db")

NWAtheneSubmitted01012018_07282021.to_sql("NWAtheneSubmitted01012018_07282021", con, if_exists='replace')


q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM NWAtheneSubmitted01012018_07282021 group by AdvisorContactIDText;"""
      
NWAtheneSubmitted01012018_07282021GrBy=  pysqldf(q3) 

con = sqlite3.connect("NWAtheneSubmitted01012018_07282021GrBy.db")

NWAtheneSubmitted01012018_07282021GrBy.to_sql("NWAtheneSubmitted01012018_07282021GrBy", con, if_exists='replace')


vm1 = """SELECT b.* FROM AtheneApp_NWApp a INNER JOIN NWAtheneSubmitted01012018_07282021GrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

AtheneApp_NWAppvsSubmitOverlap =  pysqldf(vm1)


con = sqlite3.connect("AtheneApp_NWAppvsSubmitOverlap.db")

AtheneApp_NWAppvsSubmitOverlap.to_sql("AtheneApp_NWAppvsSubmitOverlap", con, if_exists='replace')

AtheneApp_NWAppvsSubmitOverlap.info()


AtheneApp_NWApp.info()

p23  = """SELECT * FROM AtheneApp_NWApp WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM AtheneApp_NWAppvsSubmitOverlap);"""
        
NWAtheneCommonAdvisorNonProducer=  pysqldf(p23)

### Does Nationwde Non Producers submitted business for Athene?

con = sqlite3.connect("NWAdvisorNonProducer.db")

NWAdvisorNonProducer.to_sql("NWAdvisorNonProducer", con, if_exists='replace')

NWAdvisorNonProducer.info()

Athene2015_2021SubmittedBusinessGrBy.info()

vm1 = """SELECT b.* FROM NWAdvisorNonProducer a INNER JOIN Athene2015_2021SubmittedBusinessGrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

NWAdvisorNonProducerwithAtheneSubmit =  pysqldf(vm1)

con = sqlite3.connect("NWAdvisorNonProducerwithAtheneSubmit.db")

NWAdvisorNonProducerwithAtheneSubmit.to_sql("NWAdvisorNonProducerwithAtheneSubmit", con, if_exists='replace')

NWAdvisorNonProducerwithAtheneSubmit.info()

p23  = """SELECT * FROM NWAdvisorNonProducer WHERE ContactID18 NOT IN (SELECT AdvisorContactIDText FROM NWAdvisorNonProducerwithAtheneSubmit);"""
        
NationwideTrueNonProducer=  pysqldf(p23)

### Does Athene Non Producers submitted business for Nationwide?

con = sqlite3.connect("AtheneAdvisorNonProducer.db")

AtheneAdvisorNonProducer.to_sql("AtheneAdvisorNonProducer", con, if_exists='replace')

AtheneAdvisorNonProducer.info()

Nationwide2015_2021SubmittedBusinessGrBy.info()

vm1 = """SELECT b.* FROM AtheneAdvisorNonProducer a INNER JOIN Nationwide2015_2021SubmittedBusinessGrBy b on (a.ContactID18 = b.AdvisorContactIDText);"""      

AtheneAdvisorNonProducerwithNWSubmit =  pysqldf(vm1)

con = sqlite3.connect("AtheneAdvisorNonProducerwithNWSubmit.db")

AtheneAdvisorNonProducerwithNWSubmit.to_sql("AtheneAdvisorNonProducerwithNWSubmit", con, if_exists='replace')

p23  = """SELECT * FROM AtheneAdvisorNonProducer WHERE ContactID18 NOT IN (SELECT AdvisorContactIDText FROM AtheneAdvisorNonProducerwithNWSubmit);"""
        
AtheneTrueNonProducer=  pysqldf(p23)

### Fallen Angels..

### Last 90 days Athene Submitted Producers segment

### Athene  

AtheneLast3months2021= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneLast3months2021.csv',encoding= 'iso-8859-1')

AtheneLast3months2021.columns = AtheneLast3months2021.columns.str.replace(' ', '')

con = sqlite3.connect("AtheneLast3months2021.db")

AtheneLast3months2021.to_sql("AtheneLast3months2021", con, if_exists='replace')

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM AtheneLast3months2021 group by AdvisorContactIDText;"""
      
AtheneLast3months2021GrBy=  pysqldf(q3) 

### AthenAppointvsAtheneSubOperlap 

AtheneLast3months2021GrBy.info()

AthenAppointvsAtheneSubOperlap.info() 

con = sqlite3.connect("AtheneLast3months2021GrBy.db")

AtheneLast3months2021GrBy.to_sql("AtheneLast3months2021GrBy", con, if_exists='replace')

Out4 = AtheneLast3months2021GrBy.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneLast3months2021GrBy.csv', index = None, header=True)


vm1 = """SELECT b.* FROM AtheneLast3months2021GrBy a INNER JOIN AthenAppointvsAtheneSubOperlap b on (a.AdvisorContactIDText = b.AdvisorContactIDText);"""      

OverlapCurrentAtheneAdvisorlast3month =  pysqldf(vm1)

con = sqlite3.connect("OverlapCurrentAtheneAdvisorlast3month.db")

OverlapCurrentAtheneAdvisorlast3month.to_sql("OverlapCurrentAtheneAdvisorlast3month", con, if_exists='replace')

p23  = """SELECT * FROM AthenAppointvsAtheneSubOperlap WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM OverlapCurrentAtheneAdvisorlast3month);"""
        
AthenFallenAngenOver90days=  pysqldf(p23)


### Last 90 days Nationwide Submitted Producers segment

### Nationwide  

Nationwidelast3months2021= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwidelast3months2021.csv',encoding= 'iso-8859-1')

Nationwidelast3months2021.columns = Nationwidelast3months2021.columns.str.replace(' ', '')

con = sqlite3.connect("Nationwidelast3months2021.db")

Nationwidelast3months2021.to_sql("Nationwidelast3months2021", con, if_exists='replace')

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwidelast3months2021 group by AdvisorContactIDText;"""
      
Nationwidelast3months2021GrBy=  pysqldf(q3) 

Out4 = Nationwidelast3months2021GrBy.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwidelast3months2021GrBy.csv', index = None, header=True)


### AthenAppointvsAtheneSubOperlap 

Nationwidelast3months2021GrBy.info()

NWAppointvsNWSubOperlap.info() 

con = sqlite3.connect("Nationwidelast3months2021GrBy.db")

Nationwidelast3months2021GrBy.to_sql("Nationwidelast3months2021GrBy", con, if_exists='replace')


vm1 = """SELECT b.* FROM Nationwidelast3months2021GrBy a INNER JOIN NWAppointvsNWSubOperlap b on (a.AdvisorContactIDText = b.AdvisorContactIDText);"""      

OverlapCurrentNWAdvisorlast3month =  pysqldf(vm1)

con = sqlite3.connect("OverlapCurrentNWAdvisorlast3month.db")

OverlapCurrentNWAdvisorlast3month.to_sql("OverlapCurrentNWAdvisorlast3month", con, if_exists='replace')

p23  = """SELECT * FROM NWAppointvsNWSubOperlap WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM OverlapCurrentNWAdvisorlast3month);"""
        
NationwideFallenAngenOver90days=  pysqldf(p23)


### Lets take the Nationwide Paid 

NWpaiddata= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NWPaiddata2015_2021uniquPol.csv',encoding= 'iso-8859-1')

NWpaiddata.columns = NWpaiddata.columns.str.replace(' ', '')

con = sqlite3.connect("NWpaiddataa.db")

NWpaiddata.to_sql("NWpaiddata", con, if_exists='replace')

NWpaiddata.info()

q3  = """SELECT SFContactId, FullName, Gender, SFMailingState, SFNationwideLastSubmitIMO, ProductCode, ProductName, ProductDescription, ProductTypeDescription, ProductFamilyCode, UnderlyingSecurityName, PlanName, MarketingName, count(PolicyNumber4) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate FROM NWpaiddata group by SFContactId, FullName  ;"""     

NWpaiddatagrby =  pysqldf(q3)  

con = sqlite3.connect("NWpaiddatagrby.db")

NWpaiddatagrby.to_sql("NWpaiddatagrby", con, if_exists='replace')

NWpaiddatagrby.info()

### Let's overlap the Paid group with the submits I reported in the earlier

### 9548 advisors submitted business Nationwide2015_2021SubmittedBusinessGrBy
Nationwide2015_2021SubmittedBusinessGrBy.info()

vm1 = """SELECT a.* FROM NWpaiddatagrby a INNER JOIN Nationwide2015_2021SubmittedBusinessGrBy b on (a.SFContactId = b.AdvisorContactIDText);"""      

ProductIndicatorNWSubmittedSegment =  pysqldf(vm1)

Out4 = ProductIndicatorNWSubmittedSegment.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/ProductIndicatorNWSubmittedSegment.csv', index = None, header=True)



### Lets look into the 2021 Submitted business group overlap

NationwideSubmits2021= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NationwideSubmits2021.csv',encoding= 'iso-8859-1')

NationwideSubmits2021.columns = NationwideSubmits2021.columns.str.replace(' ', '')

con = sqlite3.connect("NationwideSubmits2021.db")

NationwideSubmits2021.to_sql("NationwideSubmits2021", con, if_exists='replace')

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM NationwideSubmits2021 group by AdvisorContactIDText;"""
      
NationwideSubmits2021GrBy=  pysqldf(q3) 

NationwideSubmits2021GrBy.info()

vm1 = """SELECT a.* FROM NWpaiddatagrby a INNER JOIN NationwideSubmits2021GrBy b on (a.SFContactId = b.AdvisorContactIDText);"""      

ProductIndicatorNWSubmittedSegment2021 =  pysqldf(vm1)

Out4 = ProductIndicatorNWSubmittedSegment2021.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/ProductIndicatorNWSubmittedSegment2021.csv', index = None, header=True)


### Athene

### Lets take the Athene Paid 

## Athenpaiddata, Athenepaiddatagrby already exist

## 9548 advisors submitted business Nationwide2015_2021SubmittedBusinessGrBy
AthenePaidData.info()

Athene2015_2021SubmittedBusinessGrBy.info()

vm1 = """SELECT a.*,  b.ProductName, b.SFMailingState FROM Athene2015_2021SubmittedBusinessGrBy a INNER JOIN AthenePaidData b on ( a.AdvisorContactIDText= b.SFContactId);"""      

ProductIndicatorAtheneSubmittedSegment_check =  pysqldf(vm1)

Out4 = ProductIndicatorAtheneSubmittedSegment_check.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/ProductIndicatorAtheneSubmittedSegment_check.csv', index = None, header=True)


### Lets look into the 2021 Submitted business group overlap

AtheneSubmits2021= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Athen2021Submitdata.csv',encoding= 'iso-8859-1')

AtheneSubmits2021.columns = AtheneSubmits2021.columns.str.replace(' ', '')

con = sqlite3.connect("AtheneSubmits2021.db")

AtheneSubmits2021.to_sql("AtheneSubmits2021", con, if_exists='replace')

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM AtheneSubmits2021 group by AdvisorContactIDText;"""
      
AtheneSubmits2021GrBy=  pysqldf(q3) 


AtheneSubmits2021GrBy.info()

vm1 = """SELECT a.*, b.ProductName, b.SFMailingState FROM AtheneSubmits2021GrBy  a LEFT JOIN AthenePaidDatagrby b on (a.AdvisorContactIDText= b.SFContactId);"""      

ProductIndicatorAtheneSubmittedSegment2021_check=  pysqldf(vm1)

Out4 = ProductIndicatorAtheneSubmittedSegment2021_check.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/ProductIndicatorAtheneSubmittedSegment2021_check.csv', index = None, header=True)

#### 

## NationwideFallenAngenOver90days

## AthenFallenAngenOver90days

con = sqlite3.connect("NationwideFallenAngenOver90days.db")

NationwideFallenAngenOver90days.to_sql("NationwideFallenAngenOver90days", con, if_exists='replace')

con = sqlite3.connect("AthenFallenAngenOver90days.db")

AthenFallenAngenOver90days.to_sql("AthenFallenAngenOver90days", con, if_exists='replace')

vm1 = """SELECT a.*  FROM NationwideFallenAngenOver90days a INNER JOIN AthenFallenAngenOver90days b on ( a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

CommonFallenAngels =  pysqldf(vm1)

Out4 = CommonFallenAngels.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/CommonFallenAngels.csv', index = None, header=True)

NWAdvisorPastSubmittedNoLongerApp.info()

NWAdvisorPastSubmittedNoLongerApp.to_sql("NWAdvisorPastSubmittedNoLongerApp", con, if_exists='replace')

con = sqlite3.connect("NWAdvisorPastSubmittedNoLongerApp.db")


Appointment= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Appointment.csv',encoding= 'iso-8859-1')

Appointment.columns = Appointment.columns.str.replace(' ', '')

con = sqlite3.connect("Appointment.db")

Appointment.to_sql("Appointment", con, if_exists='replace')

Appointment.info()

vm1 = """SELECT a.*, b.SFAgentCId, b.AppointmentStatusCode, b.AppointmentStatusReasonCode, b.EndDate FROM NWAdvisorPastSubmittedNoLongerApp a INNER JOIN Appointment b on (a.AdvisorContactIDText= b.SFAgentCId);"""      

NWAdvisorPastSubmittedNoLongerApp_Appointment =  pysqldf(vm1)

AtheneAdvisorPastSubmittedNoLongerApp.info()


AtheneAdvisorPastSubmittedNoLongerApp.to_sql("AtheneAdvisorPastSubmittedNoLongerApp", con, if_exists='replace')

con = sqlite3.connect("AtheneAdvisorPastSubmittedNoLongerApp.db")

vm1 = """SELECT a.*, b.SFAgentCId, b.AppointmentStatusCode, b.AppointmentStatusReasonCode, b.EndDate FROM AtheneAdvisorPastSubmittedNoLongerApp a INNER JOIN Appointment b on (a.AdvisorContactIDText= b.SFAgentCId);"""      

AtheneAdvisorPastSubmittedNoLongerApp_Appointment =  pysqldf(vm1)

AtheneAdvisorPastSubmittedNoLongerApp.info()

### AdvisorID based on AgentKey

SubmitsData2015_2021AdvisorsID= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/SubmitsData2015_2021AdvisorsID.csv',encoding= 'iso-8859-1')

SubmitsData2015_2021AdvisorsID.columns = SubmitsData2015_2021AdvisorsID.columns.str.replace(' ', '')

con = sqlite3.connect("SubmitsData2015_2021AdvisorsID.db")

SubmitsData2015_2021AdvisorsID.to_sql("SubmitsData2015_2021AdvisorsID", con, if_exists='replace')

SubmitsData2015_2021AdvisorsID.info()

q3  = """SELECT AdvisorContactAgentKey, AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM SubmitsData2015_2021AdvisorsID group by AdvisorContactAgentKey;"""
      
SubmitsData2015_2021AdvisorsID_GrBy=  pysqldf(q3) 

vm1 = """SELECT a.*, b.AdvisorContactAgentKey FROM AtheneAdvisorPastSubmittedNoLongerApp a LEFT JOIN SubmitsData2015_2021AdvisorsID_GrBy  b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

AtheneAdvisorPastSubmittedNoLongerApp_AgentKey =  pysqldf(vm1)

con = sqlite3.connect("AtheneAdvisorPastSubmittedNoLongerApp_AgentKey.db")

AtheneAdvisorPastSubmittedNoLongerApp_AgentKey.to_sql("AtheneAdvisorPastSubmittedNoLongerApp_AgentKey", con, if_exists='replace')

######

CarrierAdvisorMapping= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/CarrierAdvisorMapping.csv',encoding= 'iso-8859-1')

CarrierAdvisorMapping.columns = CarrierAdvisorMapping.columns.str.replace(' ', '')

con = sqlite3.connect("CarrierAdvisorMapping.db")

CarrierAdvisorMapping.to_sql("CarrierAdvisorMapping", con, if_exists='replace')

CarrierAdvisorMapping.info()

vm1 = """SELECT a.*, b.AdvisorKey, b.EndDate FROM AtheneAdvisorPastSubmittedNoLongerApp_AgentKey a INNER JOIN CarrierAdvisorMapping  b on (a.AdvisorContactAgentKey= b.AdvisorKey);"""      

AtheneAdvisorPastSubmittedNoLongerApp_AgentKey_check =  pysqldf(vm1)

Out4 = AtheneAdvisorPastSubmittedNoLongerApp_AgentKey_check.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneAdvisorPastSubmittedNoLongerApp_AgentKey_check.csv', index = None, header=True)

###################

Advisor_Appointment= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Advisor_Appointment.csv',encoding= 'utf-8-sig')

Advisor_Appointment.columns = Advisor_Appointment.columns.str.replace(' ', '')

Advisor_Appointment.columns = Advisor_Appointment.columns.str.lstrip()

Advisor_Appointment.columns = Advisor_Appointment.columns.str.rstrip()

Advisor_Appointment.columns = Advisor_Appointment.columns.str.strip()

con = sqlite3.connect("Advisor_Appointment.db")

Advisor_Appointment.to_sql("Advisor_Appointment", con, if_exists='replace')

Advisor_Appointment_Athene = Advisor_Appointment[Advisor_Appointment['CarrierKey']==1]

Advisor_Appointment_Nationwide = Advisor_Appointment[Advisor_Appointment['CarrierKey']==3]

con = sqlite3.connect("Advisor_Appointment_Athene.db")

Advisor_Appointment_Athene.to_sql("Advisor_Appointment_Athene", con, if_exists='replace')

Advisor_Appointment_Athene.info()

AtheneAdvisorPastSubmittedNoLongerApp_AgentKey.info()

AtheneAdvisorPastSubmittedNoLongerApp.info()

vm1 = """SELECT a.*, b.AdvisorKey, b.EndDate, AppointmentStatusCode, b.AppointmentStatusReasonCode FROM AtheneAdvisorPastSubmittedNoLongerApp a LEFT JOIN Advisor_Appointment_Athene  b on (a.AdvisorContactIDText= b.SFContactId);"""      

AtheneAdvisorPastSubmittedNoLongerApp_today =  pysqldf(vm1)


Out4 = AtheneAdvisorPastSubmittedNoLongerApp_today.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/AtheneAdvisorPastSubmittedNoLongerApp_today.csv', index = None, header=True)


con = sqlite3.connect("Advisor_Appointment_Nationwide.db")

Advisor_Appointment_Nationwide.to_sql("Advisor_Appointment_Nationwide", con, if_exists='replace')

Advisor_Appointment_Nationwide.info()

NWAdvisorPastSubmittedNoLongerApp.info()

vm1 = """SELECT a.*, b.AdvisorKey, b.EndDate, AppointmentStatusCode, b.AppointmentStatusReasonCode FROM NWAdvisorPastSubmittedNoLongerApp a LEFT JOIN Advisor_Appointment_Athene  b on (a.AdvisorContactIDText= b.SFContactId);"""      

NationwideAdvisorPastSubmittedNoLongerApp_today =  pysqldf(vm1)

Out4 = NationwideAdvisorPastSubmittedNoLongerApp_today.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/NationwideAdvisorPastSubmittedNoLongerApp_today.csv', index = None, header=True)

#####

### Followup Brian and Joe Wanted advisors who fell off 2019 to 2020 and 2020 vs 2021 were they producing 3 years prior

## Nationwide2015_2021SubmittedBusinessGrBy

Nationwide2015_2021SubmittedBusiness_Year= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Nationwide2015_2021SubmittedBusiness_Year.csv',encoding= 'iso-8859-1')

Nationwide2015_2021SubmittedBusiness_Year.columns = Nationwide2015_2021SubmittedBusiness_Year.columns.str.replace(' ', '')

Nationwide2015_2021SubmittedBusiness_Year.info()

Nationwide2015_2021SubmittedBusiness_2015 = Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2015]

Nationwide2015_2021SubmittedBusiness_2016= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2016]

Nationwide2015_2021SubmittedBusiness_2017= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2017]

Nationwide2015_2021SubmittedBusiness_2018= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2018]

Nationwide2015_2021SubmittedBusiness_2019= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2019]

Nationwide2015_2021SubmittedBusiness_2020= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2020]

Nationwide2015_2021SubmittedBusiness_2021= Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear']==2021]

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2015.db")

Nationwide2015_2021SubmittedBusiness_2015.to_sql("Nationwide2015_2021SubmittedBusiness_2015", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2016.db")

Nationwide2015_2021SubmittedBusiness_2016.to_sql("Nationwide2015_2021SubmittedBusiness_2016", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2017.db")

Nationwide2015_2021SubmittedBusiness_2017.to_sql("Nationwide2015_2021SubmittedBusiness_2017", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2018.db")

Nationwide2015_2021SubmittedBusiness_2018.to_sql("Nationwide2015_2021SubmittedBusiness_2018", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2019.db")

Nationwide2015_2021SubmittedBusiness_2019.to_sql("Nationwide2015_2021SubmittedBusiness_2019", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2020.db")

Nationwide2015_2021SubmittedBusiness_2020.to_sql("Nationwide2015_2021SubmittedBusiness_2020", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2021.db")

Nationwide2015_2021SubmittedBusiness_2021.to_sql("Nationwide2015_2021SubmittedBusiness_2021", con, if_exists='replace')

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2015 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2015GrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2016 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2016GrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2017 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2017GrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2018 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2018GrBy=  pysqldf(q3) 


q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2019 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2019GrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2020 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2020GrBy=  pysqldf(q3) 

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2021 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2021GrBy=  pysqldf(q3) 

### Lets look into 

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2020GrBy.db")

Nationwide2015_2021SubmittedBusiness_2020GrBy.to_sql("Nationwide2015_2021SubmittedBusiness_2020GrBy", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2021GrBy.db")

Nationwide2015_2021SubmittedBusiness_2021GrBy.to_sql("Nationwide2015_2021SubmittedBusiness_2021GrBy", con, if_exists='replace')

## How many common in 2020 and 2021

vm1 = """SELECT a.* FROM Nationwide2015_2021SubmittedBusiness_2020GrBy a INNER JOIN Nationwide2015_2021SubmittedBusiness_2021GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2020_2021 =  pysqldf(vm1)

## How many Dropped off between 2020 and 2021

Common_2020_2021.to_sql("Common_2020_2021", con, if_exists='replace')

con = sqlite3.connect("Common_2020_2021.db")

p23  = """SELECT * FROM Nationwide2015_2021SubmittedBusiness_2020GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common_2020_2021);"""
        
AdvisorFalloff2020_2021=  pysqldf(p23)

## For this group lets look into if they sell in 205, 2016, 2017

Year_lable = ['2015','2016','2017']

Nationwide2015_2021SubmittedBusiness_2015_2016_2017 = Nationwide2015_2021SubmittedBusiness_Year[Nationwide2015_2021SubmittedBusiness_Year['SubmitYear'].isin(Year_lable)]

Nationwide2015_2021SubmittedBusiness_2015_2016_2017.to_sql("Nationwide2015_2021SubmittedBusiness_2015_2016_2017", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2015_2016_2017.db")

q3  = """SELECT AdvisorContactIDText, AdvisorName, AdvisorName1, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate FROM Nationwide2015_2021SubmittedBusiness_2015_2016_2017 group by AdvisorContactIDText;"""
      
Nationwide2015_2021SubmittedBusiness_2015_2016_2017GrBy=  pysqldf(q3) 

### Lets look into the overlap Common_2020_2021, AdvisorFalloff2020_2021

Common_2020_2021.to_sql("Common_2020_2021", con, if_exists='replace')

con = sqlite3.connect("Common_2020_2021.db")

AdvisorFalloff2020_2021.to_sql("AdvisorFalloff2020_2021", con, if_exists='replace')

con = sqlite3.connect("AdvisorFalloff2020_2021.db")

vm1 = """SELECT a.* FROM Common_2020_2021 a INNER JOIN Nationwide2015_2021SubmittedBusiness_2015_2016_2017GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2020_2021sold2015onwards =  pysqldf(vm1)

vm1 = """SELECT a.* FROM AdvisorFalloff2020_2021 a INNER JOIN Nationwide2015_2021SubmittedBusiness_2015_2016_2017GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

AdvisorFalloff2020_2021onwards =  pysqldf(vm1)

### Validation against Sarath's file

SarathListZibraMozaic4thYearContinutiy= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/SarathListZibraMozaic4thYearContinutiy_U.csv',encoding= 'iso-8859-1')

SarathListZibraMozaic4thYearContinutiy.columns = SarathListZibraMozaic4thYearContinutiy.columns.str.replace(' ', '')

con = sqlite3.connect("SarathListZibraMozaic4thYearContinutiy.db")

SarathListZibraMozaic4thYearContinutiy.to_sql("SarathListZibraMozaic4thYearContinutiy", con, if_exists='replace')

SarathListZibraMozaic4thYearContinutiy.info()

### Check the overlap with the bigger sample

vm1 = """SELECT a.* FROM Common_2020_2021 a INNER JOIN SarathListZibraMozaic4thYearContinutiy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2020_2021sold2015onwards_A =  pysqldf(vm1)

## Check the overlap with the onwards

con = sqlite3.connect("Common_2020_2021sold2015onwards.db")

Common_2020_2021sold2015onwards.to_sql("Common_2020_2021sold2015onwards", con, if_exists='replace')

vm1 = """SELECT a.* FROM Common_2020_2021sold2015onwards a INNER JOIN SarathListZibraMozaic4thYearContinutiy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2020_2021sold2015onwards_AA =  pysqldf(vm1)

Out4 = Common_2020_2021sold2015onwards_A.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_A.csv', index = None, header=True)

Out4 = Common_2020_2021sold2015onwards_AA.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_AA.csv', index = None, header=True)


### Dan Pulled the Data Again

DataDanPulledOutUsingSarhatQuery_U= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/2015_2016_2017DataDanPulledOutUsingSarhatQuery_U.csv',encoding= 'iso-8859-1')

DataDanPulledOutUsingSarhatQuery_U.columns = DataDanPulledOutUsingSarhatQuery_U.columns.str.replace(' ', '')

con = sqlite3.connect("DataDanPulledOutUsingSarhatQuery_U.db")

DataDanPulledOutUsingSarhatQuery_U.to_sql("DataDanPulledOutUsingSarhatQuery_U", con, if_exists='replace')

DataDanPulledOutUsingSarhatQuery_U.info()

vm1 = """SELECT a.* FROM Common_2020_2021 a INNER JOIN DataDanPulledOutUsingSarhatQuery_U b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2020_2021sold2015onwards_AAA =  pysqldf(vm1)

Out4 = Common_2020_2021sold2015onwards_AAA.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_AAA.csv', index = None, header=True)


#####
### 
#### Lets do the same exercise for 2019 and 2020

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2019GrBy.db")

Nationwide2015_2021SubmittedBusiness_2019GrBy.to_sql("Nationwide2015_2021SubmittedBusiness_2019GrBy", con, if_exists='replace')

con = sqlite3.connect("Nationwide2015_2021SubmittedBusiness_2020GrBy.db")

Nationwide2015_2021SubmittedBusiness_2020GrBy.to_sql("Nationwide2015_2021SubmittedBusiness_2020GrBy", con, if_exists='replace')

## How many common in 2020 and 2021

vm1 = """SELECT a.* FROM Nationwide2015_2021SubmittedBusiness_2019GrBy a INNER JOIN Nationwide2015_2021SubmittedBusiness_2020GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2019_2020 =  pysqldf(vm1)

## How many Dropped off between 2020 and 2021

Common_2019_2020.to_sql("Common_2019_2020", con, if_exists='replace')

con = sqlite3.connect("Common_2019_2020.db")

p23  = """SELECT * FROM Nationwide2015_2021SubmittedBusiness_2019GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common_2019_2020);"""
        
AdvisorFalloff2019_2020=  pysqldf(p23)

con = sqlite3.connect("Common_2019_2020.db")

Common_2020_2021.to_sql("Common_2019_2020", con, if_exists='replace')

con = sqlite3.connect("AdvisorFalloff2019_2020.db")

AdvisorFalloff2019_2020.to_sql("AdvisorFalloff2019_2020", con, if_exists='replace')

vm1 = """SELECT a.* FROM Common_2019_2020 a INNER JOIN Nationwide2015_2021SubmittedBusiness_2015_2016_2017GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2019_2020sold2015onwards =  pysqldf(vm1)

vm1 = """SELECT a.* FROM AdvisorFalloff2019_2020 a INNER JOIN Nationwide2015_2021SubmittedBusiness_2015_2016_2017GrBy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

AdvisorFalloff2019_2020onwards =  pysqldf(vm1)

### Check the overlap with bigger sample
vm1 = """SELECT a.* FROM Common_2019_2020 a INNER JOIN SarathListZibraMozaic4thYearContinutiy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2019_2020sold2015onwards_B =  pysqldf(vm1)

con = sqlite3.connect("Common_2019_2020sold2015onwards.db")

Common_2019_2020sold2015onwards.to_sql("Common_2019_2020sold2015onwards", con, if_exists='replace')

vm1 = """SELECT a.* FROM Common_2019_2020sold2015onwards a INNER JOIN SarathListZibraMozaic4thYearContinutiy b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2019_2020sold2015onwards_BB =  pysqldf(vm1)

### Clearly Common_2019_2020sold2015onwards_B and Common_2019_2020sold2015onwards_BB are not same but they are close

Out4 = Common_2019_2020sold2015onwards_B.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_B.csv', index = None, header=True)

Out4 = Common_2019_2020sold2015onwards_BB.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_BB.csv', index = None, header=True)

### Dan Pulled the Data Again

DataDanPulledOutUsingSarhatQuery_U= pd.read_csv('C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/2015_2016_2017DataDanPulledOutUsingSarhatQuery_U.csv',encoding= 'iso-8859-1')

DataDanPulledOutUsingSarhatQuery_U.columns = DataDanPulledOutUsingSarhatQuery_U.columns.str.replace(' ', '')

con = sqlite3.connect("DataDanPulledOutUsingSarhatQuery_U.db")

DataDanPulledOutUsingSarhatQuery_U.to_sql("DataDanPulledOutUsingSarhatQuery_U", con, if_exists='replace')

DataDanPulledOutUsingSarhatQuery_U.info()

vm1 = """SELECT a.* FROM Common_2019_2020 a INNER JOIN DataDanPulledOutUsingSarhatQuery_U b on (a.AdvisorContactIDText= b.AdvisorContactIDText);"""      

Common_2019_2020sold2015onwards_BBB =  pysqldf(vm1)

Out4 = Common_2019_2020sold2015onwards_BBB.to_csv(r'C:/Users/test/Documents/Nationwide_AtheneOverviewAnalysis/Mozaic_ZebraFollowup/Common_2020_2021sold2015onwards_BBB.csv', index = None, header=True)














