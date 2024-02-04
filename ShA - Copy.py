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

AtheneBCAPaidData= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/AtheneBCA2020.csv',encoding= 'iso-8859-1')

#AtheneBCAPaidData= pd.read_csv('C:/Users/test/Documents/Schillar/ShillerAnalysis/AtheneBCA2020.csv',encoding= 'iso-8859-1')

AtheneBCAPaidData.info()
AtheneBCAPaidDataDedup = AtheneBCAPaidData.drop_duplicates(subset='PolicyNumber', keep="first")

export_csv = AtheneBCAPaidDataDedup.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\AtheneBCAPaidDataDedup.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Shiller Signed Book
Shiller_Signed= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/ShillerSignedwaddress.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Shiller_Signed.db")

Shiller_Signed.to_sql("Shiller_Signed", con, if_exists='replace')

Shiller_SignedAfterReturnRem= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/ShillerSignedwaddressAfterretrunremoval.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Shiller_SignedAfterReturnRem.db")

Shiller_SignedAfterReturnRem.to_sql("Shiller_SignedAfterReturnRem", con, if_exists='replace')

PostCampaingSubmit2Month= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport02032020_04052020_post2month.csv',encoding= 'iso-8859-1')

PreCampaingSubmit2Month= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport12012019_02022020_pre2month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("PostCampaingSubmit2Month.db")

PostCampaingSubmit2Month.to_sql("PostCampaingSubmit2Month", con, if_exists='replace')

PostCampaingSubmit2Month.info()

### Shiller Non Signed Book

### Let's look into the non-signed 

ShillerNonSigned1wAddress= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/ShillerNonSigned1wAddress.csv',encoding= 'iso-8859-1')

ShillerNonSigned1wAddress.info()

con = sqlite3.connect("ShillerNonSigned1wAddress.db")

ShillerNonSigned1wAddress.to_sql("ShillerNonSigned1wAddress", con, if_exists='replace')

v11 = """SELECT a.* FROM ShillerNonSigned1wAddress a INNER JOIN NonSignedBookReturned b on (a.AdvisorName=b.AdvisorFullName);"""      

CommonListShillerNonSigned1wAddress =  pysqldf(v11)

con = sqlite3.connect("CommonListShillerNonSigned1wAddress.db")

CommonListShillerNonSigned1wAddress.to_sql("CommonListShillerNonSigned1wAddress", con, if_exists='replace')

p22  = """SELECT * FROM ShillerNonSigned1wAddress WHERE AdvisorName NOT IN (SELECT AdvisorName FROM CommonListShillerNonSigned1wAddress);"""
        
RemainingNonSignedAdvisor =  pysqldf(p22)  

con = sqlite3.connect("RemainingNonSignedAdvisor.db")

RemainingNonSignedAdvisor.to_sql("RemainingNonSignedAdvisor", con, if_exists='replace')

#### Fallen Angels

FallenAngels= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/RemainingFallenAngelswaddress.csv',encoding= 'iso-8859-1')

FallenAngels.info()

con = sqlite3.connect("FallenAngels.db")

FallenAngels.to_sql("FallenAngels", con, if_exists='replace')

NonSignedBookReturned= pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/ShillerNonSignedBookreturned.csv',encoding= 'iso-8859-1')

NonSignedBookReturned.info()

con = sqlite3.connect("NonSignedBookReturned.db")

NonSignedBookReturned.to_sql("NonSignedBookReturned", con, if_exists='replace')

v1 = """SELECT a.* FROM FallenAngels a INNER JOIN NonSignedBookReturned b on (a.AdvisorName=b.AdvisorFullName);"""      

CommonList =  pysqldf(v1)

con = sqlite3.connect("CommonList.db")

CommonList.to_sql("CommonList", con, if_exists='replace')

p2  = """SELECT * FROM FallenAngels WHERE AdvisorName NOT IN (SELECT AdvisorName FROM CommonList);"""
        
RemainingFA =  pysqldf(p2)  

Out = RemainingFA.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\RemainingFA.csv', index = None, header=True)

### Let's look into Pre vs. Post data and examine the performance of this group

CampaignPostData = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport02032020_04052020_post2month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPostData.db")

CampaignPostData.to_sql("CampaignPostData", con, if_exists='replace')


CampaignPreData = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport12012019_02022020_pre2month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPreData.db")

CampaignPreData.to_sql("CampaignPreData", con, if_exists='replace')

RemainingFA.info()

CampaignPreData.info()

CampaignPostData.info()

#### Post and Pre data has to be processed 

###Post 2 Months

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPostData group by AdvisorName, AdvisorContactIDText;"""
      
SubmitbyAdvisorPost =  pysqldf(q3)  

con = sqlite3.connect("SubmitbyAdvisorPost.db")

SubmitbyAdvisorPost.to_sql("SubmitbyAdvisorPost", con, if_exists='replace')

SubmitbyAdvisorPost.info()

### Pre 2 Months

q4  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPreData group by AdvisorName, AdvisorContactIDText;"""
      
SubmitbyAdvisorPre =  pysqldf(q4)  

con = sqlite3.connect("SubmitbyAdvisorPre.db")

SubmitbyAdvisorPre.to_sql("SubmitbyAdvisorPre", con, if_exists='replace')

### Lets join the post 

l33 = """SELECT a.AdvisorName, a.SFContactId, b.AdvisorName, b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmt
      FROM RemainingFA a LEFT JOIN SubmitbyAdvisorPost b on (a.SFContactId= b.AdvisorContactIDText) or (a.AdvisorName=b.AdvisorName);"""
      
RemainingFAMatchPost2Month =  pysqldf(l33)

Out1 = RemainingFAMatchPost2Month.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\RemainingFAMatchPost2Month.csv', index = None, header=True)
  

### Lets join the pre

l34 = """SELECT a.AdvisorName, a.SFContactId, b.AdvisorName, b.AdvisorContactIDText, b.SubmitCnt, b.SubmitAmt
      FROM RemainingFA a LEFT JOIN SubmitbyAdvisorPre b on (a.SFContactId= b.AdvisorContactIDText) or (a.AdvisorName=b.AdvisorName);"""
      
RemainingFAMatchPre2Month =  pysqldf(l34) 

Out2 = RemainingFAMatchPre2Month.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\RemainingFAMatchPre2Month.csv', index = None, header=True)
  
### Pre vs. Post comparison

PreMatchComp=  pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/PreMatchComp.csv',encoding= 'iso-8859-1')

PostMatchComp=  pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/DataCamefromMegan/CSVs/PostMatchComp.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("PreMatchComp.db")

PreMatchComp.to_sql("PreMatchComp", con, if_exists='replace')

con = sqlite3.connect("PostMatchComp.db")

PostMatchComp.to_sql("PostMatchComp", con, if_exists='replace')

v11 = """SELECT a1.* FROM PostMatchComp a1 INNER JOIN PreMatchComp b1 on (a1.SFContactId=b1.SFContactId);"""      

CommonListPrePost =  pysqldf(v11)

con = sqlite3.connect("CommonListPrePost.db")

CommonListPrePost.to_sql("CommonListPrePost", con, if_exists='replace')

p22  = """SELECT * FROM PostMatchComp WHERE SFContactId NOT IN (SELECT SFContactId FROM CommonListPrePost);"""
        
RemainingFAPrepost =  pysqldf(p22)  

Out3 = RemainingFAPrepost.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\RemainingFAPrepost.csv', index = None, header=True)
  
### Let's do this another way to validate

### Compare Pre and Post Submit list SubmitbyAdvisorPre SubmitbyAdvisorPost
### Subtract the common from the post submit list

### Take the subtracted post list and match that with remaingFA group

SubmitbyAdvisorPost.info()

SubmitbyAdvisorPre.info()

v12 = """SELECT a2.* FROM SubmitbyAdvisorPost a2 INNER JOIN SubmitbyAdvisorPre b2 on (a2.AdvisorContactIDText=b2.AdvisorContactIDText);"""      

CommonListPrePost1 =  pysqldf(v12)

p23  = """SELECT * FROM SubmitbyAdvisorPost WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonListPrePost1);"""
        
RemainingFAPrepost1 =  pysqldf(p23)  

con = sqlite3.connect("RemainingFAPrepost1.db")

RemainingFAPrepost1.to_sql("RemainingFAPrepost1", con, if_exists='replace')

v188 = """SELECT a3.* FROM RemainingFAPrepost1 a3 INNER JOIN RemainingFA b3 on (a3.AdvisorContactIDText=b3.SFContactId);"""      

CommonListPrePostCheck =  pysqldf(v188)

Out4 = CommonListPrePostCheck.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonListPrePostCheck.csv', index = None, header=True)

### Let's look into pre and post one months

#### 

CampaignPreData1mo = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport01012020_02022020_Pre1month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPreData1mo.db")

CampaignPreData1mo.to_sql("CampaignPreData1mo", con, if_exists='replace')

q4  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPreData1mo group by  AdvisorContactIDText, AdvisorName;"""
      
SubmitbyAdvisorPre1mo =  pysqldf(q4)  

con = sqlite3.connect("SubmitbyAdvisorPre1mo.db")

SubmitbyAdvisorPre1mo.to_sql("SubmitbyAdvisorPre1mo", con, if_exists='replace')
  

CampaignPostData1mo = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport02032020_03032020_Post1month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPostData1mo.db")

CampaignPostData1mo.to_sql("CampaignPostData1mo", con, if_exists='replace')


q5  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPostData1mo group by AdvisorContactIDText, AdvisorName;"""
      
SubmitbyAdvisorPost1mo =  pysqldf(q5)

### We need to look into the three segments: Shiller_SignedAfterReturnRem, RemainingFA, RemainingNonSignedAdvisor

SubmitbyAdvisorPre1mo.info()

Shiller_SignedAfterReturnRem.info()

RemainingFA.info()

RemainingNonSignedAdvisor.info()

con = sqlite3.connect("Shiller_SignedAfterReturnRem.db")

Shiller_SignedAfterReturnRem.to_sql("Shiller_SignedAfterReturnRem", con, if_exists='replace')

v88 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM Shiller_SignedAfterReturnRem a LEFT JOIN SubmitbyAdvisorPre1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA1 =  pysqldf(v88)

Out = CommonAA1.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonAA1.csv', index = None, header=True)

v89 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM Shiller_SignedAfterReturnRem a LEFT JOIN SubmitbyAdvisorPost1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA2 =  pysqldf(v89)

Out = CommonAA2.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonAA2.csv', index = None, header=True)

con = sqlite3.connect("RemainingNonSignedAdvisor.db")

RemainingNonSignedAdvisor.to_sql("RemainingNonSignedAdvisor", con, if_exists='replace')

v90 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingNonSignedAdvisor a LEFT JOIN SubmitbyAdvisorPre1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA3 =  pysqldf(v90)

Out = CommonAA3.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonAA3.csv', index = None, header=True)

v91 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingNonSignedAdvisor a LEFT JOIN SubmitbyAdvisorPost1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA4 =  pysqldf(v91)

Out = CommonAA4.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonAA4.csv', index = None, header=True)

### Fallen Angels
con = sqlite3.connect("RemainingFA.db")

RemainingFA.to_sql("RemainingFA", con, if_exists='replace')

v92 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingFA a INNER JOIN SubmitbyAdvisorPre1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA5 =  pysqldf(v92)

v93 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingFA a INNER JOIN SubmitbyAdvisorPost1mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonAA6 =  pysqldf(v93)

con = sqlite3.connect("CommonAA5.db")

CommonAA5.to_sql("CommonAA5", con, if_exists='replace')

CommonAA5.info()

con = sqlite3.connect("CommonAA6.db")

CommonAA6.to_sql("CommonAA6", con, if_exists='replace')

v95 = """SELECT b.* FROM CommonAA5 a INNER JOIN CommonAA6 b on (a.SFContactId=b.SFContactId);"""      

CommonOneMo =  pysqldf(v95)

con = sqlite3.connect("CommonOneMo.db")

CommonOneMo.to_sql("CommonOneMo", con, if_exists='replace')

p26  = """SELECT * FROM CommonAA6 WHERE SFContactId NOT IN (SELECT SFContactId FROM CommonOneMo);"""
        
OnemonthGain =  pysqldf(p26)  

Out = OnemonthGain.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\OnemonthGain.csv', index = None, header=True)


### 3 months starts

### Pre
CampaignPreData3mo = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport11032019_02022020_pre3month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPreData3mo.db")

CampaignPreData3mo.to_sql("CampaignPreData3mo", con, if_exists='replace')


q14  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPreData3mo group by  AdvisorContactIDText, AdvisorName;"""
      
SubmitbyAdvisorPre3mo =  pysqldf(q14)  

con = sqlite3.connect("SubmitbyAdvisorPre3mo.db")

SubmitbyAdvisorPre3mo.to_sql("SubmitbyAdvisorPre3mo", con, if_exists='replace')

### Post

CampaignPostData3mo = pd.read_csv('C:/Users/test/Documents/SchillarBookCampaign/ShillerAnalysis/BCASubmitDatafromSubmitReport02032020_05032020_post3month.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("CampaignPostData3mo.db")


CampaignPostData3mo.to_sql("CampaignPostData3mo", con, if_exists='replace')

q15  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM CampaignPostData3mo group by AdvisorContactIDText, AdvisorName;"""
      
SubmitbyAdvisorPost3mo =  pysqldf(q15)

con = sqlite3.connect("SubmitbyAdvisorPost3mo.db")

SubmitbyAdvisorPost3mo.to_sql("SubmitbyAdvisorPost3mo", con, if_exists='replace')

### Pre vs Post Shiller Autograph

a88 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM Shiller_SignedAfterReturnRem a LEFT JOIN SubmitbyAdvisorPre3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB1 =  pysqldf(a88)

Out = CommonBB1.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonBB1.csv', index = None, header=True)


a89 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM Shiller_SignedAfterReturnRem a LEFT JOIN SubmitbyAdvisorPost3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB2 =  pysqldf(a89)

Out = CommonBB2.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonBB2.csv', index = None, header=True)

##### RemainingNonSignedAdvisor

a90 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingNonSignedAdvisor a LEFT JOIN SubmitbyAdvisorPre3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB3 =  pysqldf(a90)

Out = CommonBB3.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonBB3.csv', index = None, header=True)

a91 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingNonSignedAdvisor a LEFT JOIN SubmitbyAdvisorPost3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB4 =  pysqldf(a91)

Out = CommonBB4.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\CommonBB4.csv', index = None, header=True)


##### RemainingFA

a92 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingFA a INNER JOIN SubmitbyAdvisorPre3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB5 =  pysqldf(a92)


a93 = """SELECT a.*, b.SubmitAmt, b.SubmitCnt FROM RemainingFA a INNER JOIN SubmitbyAdvisorPost3mo b on (b.AdvisorContactIDText=a.SFContactId);"""      

CommonBB6 =  pysqldf(a93)

con = sqlite3.connect("CommonBB5.db")

CommonBB5.to_sql("CommonBB5", con, if_exists='replace')

con = sqlite3.connect("CommonBB6.db")

CommonBB6.to_sql("CommonBB6", con, if_exists='replace')

w95 = """SELECT b.* FROM CommonBB5 a INNER JOIN CommonBB6 b on (a.SFContactId=b.SFContactId);"""      

CommonThreeMo =  pysqldf(w95)

con = sqlite3.connect("CommonThreeMo.db")

CommonThreeMo.to_sql("CommonThreeMo", con, if_exists='replace')

l26  = """SELECT * FROM CommonBB6 WHERE SFContactId NOT IN (SELECT SFContactId FROM CommonThreeMo);"""
        
ThreemonthGain =  pysqldf(l26)  

Out = ThreemonthGain.to_csv (r'C:\Users\test\Documents\SchillarBookCampaign\ShillerAnalysis\DataCamefromMegan\CSVs\ThreemonthGain.csv', index = None, header=True)
