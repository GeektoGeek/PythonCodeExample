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

### Submits for the last 3 years

Years3SubmitNewHeightsUntilOct2017 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/3YearsSubmitNewHeightsUntilOct2017Sep2020.csv',encoding= 'iso-8859-1')

Years3SubmitNewHeightsUntilOct2017.columns = Years3SubmitNewHeightsUntilOct2017.columns.str.replace(' ', '')

Years3SubmitNewHeightsUntilOct2017.columns = Years3SubmitNewHeightsUntilOct2017.columns.str.lstrip()

Years3SubmitNewHeightsUntilOct2017.columns = Years3SubmitNewHeightsUntilOct2017.columns.str.rstrip()

Years3SubmitNewHeightsUntilOct2017.columns = Years3SubmitNewHeightsUntilOct2017.columns.str.strip()

con = sqlite3.connect("Years3SubmitNewHeightsUntilOct2017.db")

Years3SubmitNewHeightsUntilOct2017.to_sql("Years3SubmitNewHeightsUntilOct2017", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM Years3SubmitNewHeightsUntilOct2017 group by AdvisorContactIDText, AdvisorName;"""
      
Years3SubmitNewHeightsUntilOct2017GrBy =  pysqldf(q3) 

Years3SubmitNewHeightsUntilOct2017GrBy.info()

### Last 3 months..Sep 2020 until Dec 2020

SubmitSep2020_Dec2020 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitSep2020_Dec2020.csv',encoding= 'iso-8859-1')


SubmitSep2020_Dec2020.columns = SubmitSep2020_Dec2020.columns.str.replace(' ', '')

SubmitSep2020_Dec2020.columns = SubmitSep2020_Dec2020.columns.str.lstrip()

SubmitSep2020_Dec2020.columns = SubmitSep2020_Dec2020.columns.str.rstrip()

SubmitSep2020_Dec2020.columns = SubmitSep2020_Dec2020.columns.str.strip()

SubmitSep2020_Dec2020['SubmitDate'] = pd.to_datetime(SubmitSep2020_Dec2020['SubmitDate'])

con = sqlite3.connect("SubmitSep2020_Dec2020.db")

SubmitSep2020_Dec2020.to_sql("SubmitSep2020_Dec2020", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM SubmitSep2020_Dec2020 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitSep2020_Dec2020GrBy =  pysqldf(q3) 

SubmitSep2020_Dec2020.info()

SubmitSep2020_Dec2020['SubmitDate'] = pd.to_datetime(SubmitSep2020_Dec2020['SubmitDate'])

###Common 3 Yr vs last 3 months

con = sqlite3.connect("Years3SubmitNewHeightsUntilOct2017GrBy.db")

Years3SubmitNewHeightsUntilOct2017GrBy.to_sql("Years3SubmitNewHeightsUntilOct2017GrBy", con, if_exists='replace')

con = sqlite3.connect("SubmitSep2020_Dec2020GrBy.db")

SubmitSep2020_Dec2020GrBy.to_sql("SubmitSep2020_Dec2020GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr, b.AdvisorName FROM Years3SubmitNewHeightsUntilOct2017GrBy a INNER JOIN SubmitSep2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

Common3Yrvslast3Months =  pysqldf(vm11)

p23  = """SELECT * FROM Years3SubmitNewHeightsUntilOct2017GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common3Yrvslast3Months);"""
        
FallenAngels3Yrvslast3Months=  pysqldf(p23)

#### 
####Multiple Comparison

###Sep2017_Dec2017 vs. Sep2020_Dec2020

Sep2017_Dec2017=pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitSep2017_Dec2017.csv',encoding= 'iso-8859-1')

Sep2017_Dec2017.columns = Sep2017_Dec2017.columns.str.replace(' ', '')

Sep2017_Dec2017.columns = Sep2017_Dec2017.columns.str.lstrip()

Sep2017_Dec2017.columns = Sep2017_Dec2017.columns.str.rstrip()

Sep2017_Dec2017.columns = Sep2017_Dec2017.columns.str.strip()

con = sqlite3.connect("Sep2017_Dec2017.db")

Sep2017_Dec2017.to_sql("Sep2017_Dec2017", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, max(SubmitDate) as LastSubmitDate
      FROM Sep2017_Dec2017 group by AdvisorContactIDText, AdvisorName;"""
      
Sep2017_Dec2017GrBy =  pysqldf(q3) 

Sep2017_Dec2017GrBy.info()

con = sqlite3.connect("Sep2017_Dec2017GrBy.db")

Sep2017_Dec2017GrBy.to_sql("Sep2017_Dec2017GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Mon2017, a.SubmitAmt as SubmitAmt3Mon2017, b.AdvisorName FROM Sep2017_Dec2017GrBy a INNER JOIN SubmitSep2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

Common2017SepDec_2020SepDec =  pysqldf(vm11)

p23  = """SELECT * FROM Sep2017_Dec2017GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common2017SepDec_2020SepDec);"""
        
FallenAngels3Monlast3Months=  pysqldf(p23)

Out4 = FallenAngels3Monlast3Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngels3Monlast3Months.csv', index = None, header=True)

###June2017_Dec2017 vs Sep 2020_Dec2020

June2017_Dec2017=pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitJune2017_Dec2017.csv',encoding= 'iso-8859-1')

June2017_Dec2017.columns = June2017_Dec2017.columns.str.replace(' ', '')

June2017_Dec2017.columns = June2017_Dec2017.columns.str.lstrip()

June2017_Dec2017.columns = June2017_Dec2017.columns.str.rstrip()

June2017_Dec2017.columns = June2017_Dec2017.columns.str.strip()

con = sqlite3.connect("June2017_Dec2017.db")

June2017_Dec2017.to_sql("June2017_Dec2017", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM June2017_Dec2017 group by AdvisorContactIDText, AdvisorName;"""
      
June2017_Dec2017GrBy =  pysqldf(q3) 

June2017_Dec2017GrBy.info()

con = sqlite3.connect("June2017_Dec2017GrBy.db")

June2017_Dec2017GrBy.to_sql("June2017_Dec2017GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr, b.AdvisorName FROM June2017_Dec2017GrBy a INNER JOIN SubmitSep2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

Common2017JuneDec_2020SepDec =  pysqldf(vm11)

p23  = """SELECT * FROM June2017_Dec2017GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common2017JuneDec_2020SepDec);"""
        
FallenAngels3Monlast6Months=  pysqldf(p23)

Out4 = FallenAngels3Monlast6Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngels3Monlast6Months.csv', index = None, header=True)


###Oct2017_Mar2018 vs. Oct2020_Dec2020

SubmitOct2017_Mar2018 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitOct2017_Mar2018.csv',encoding= 'iso-8859-1')

SubmitOct2017_Mar2018.columns = SubmitOct2017_Mar2018.columns.str.replace(' ', '')

SubmitOct2017_Mar2018.columns = SubmitOct2017_Mar2018.columns.str.lstrip()

SubmitOct2017_Mar2018.columns = SubmitOct2017_Mar2018.columns.str.rstrip()

SubmitOct2017_Mar2018.columns = SubmitOct2017_Mar2018.columns.str.strip()

con = sqlite3.connect("SubmitOct2017_Mar2018.db")

SubmitOct2017_Mar2018.to_sql("SubmitOct2017_Mar2018", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitOct2017_Mar2018 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitOct2017_Mar2018GrBy =  pysqldf(q3) 

SubmitOct2017_Mar2018GrBy.info()

SubmitJun2020_Dec2020 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitJun2020_Dec2020.csv',encoding= 'iso-8859-1')

SubmitJun2020_Dec2020.columns = SubmitJun2020_Dec2020.columns.str.replace(' ', '')

SubmitJun2020_Dec2020.columns = SubmitJun2020_Dec2020.columns.str.lstrip()

SubmitJun2020_Dec2020.columns = SubmitJun2020_Dec2020.columns.str.rstrip()

SubmitJun2020_Dec2020.columns = SubmitJun2020_Dec2020.columns.str.strip()

con = sqlite3.connect("SubmitJun2020_Dec2020.db")

SubmitJun2020_Dec2020.to_sql("SubmitJun2020_Dec2020", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitJun2020_Dec2020 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitJun2020_Dec2020GrBy =  pysqldf(q3) 

SubmitJun2020_Dec2020.info()

con = sqlite3.connect("SubmitOct2017_Mar2018GrBy.db")

SubmitOct2017_Mar2018GrBy.to_sql("SubmitOct2017_Mar2018GrBy", con, if_exists='replace')

con = sqlite3.connect("SubmitJun2020_Dec2020GrBy.db")

SubmitJun2020_Dec2020GrBy.to_sql("SubmitJun2020_Dec2020GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr, b.AdvisorName as AdvName, b.SubmitCnt, b.SubmitAmt FROM SubmitOct2017_Mar2018GrBy a INNER JOIN SubmitJun2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

Common6Monlast6Months =  pysqldf(vm11)

p23  = """SELECT * FROM SubmitOct2017_Mar2018GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Common6Monlast6Months);"""
        
FallenAngels6Monthslast6Months=  pysqldf(p23)

Out4 = FallenAngels6Monthslast6Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngels6Monthslast6Months.csv', index = None, header=True)


### Oct2017-Mar 2018 VS. Sep 2020-Dec2020

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr FROM SubmitOct2017_Mar2018GrBy a INNER JOIN SubmitSep2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonActualTime6Monlast3Months =  pysqldf(vm11)

p23  = """SELECT * FROM SubmitOct2017_Mar2018GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonActualTime6Monlast3Months);"""
        
FallenAngelsActualTime6Monlast3Months= pysqldf(p23)

Out4 = FallenAngelsActualTime6Monlast3Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngelsActualTime6Monlast3Months.csv', index = None, header=True)

### Lets do A vs. B

con = sqlite3.connect("FallenAngels6Monthslast6Months.db")

FallenAngels6Monthslast6Months.to_sql("FallenAngels6Monthslast6Months", con, if_exists='replace')

con = sqlite3.connect("FallenAngelsActualTime6Monlast3Months.db")

FallenAngelsActualTime6Monlast3Months.to_sql("FallenAngelsActualTime6Monlast3Months", con, if_exists='replace')

vm11 = """SELECT a.* FROM FallenAngels6Monthslast6Months a INNER JOIN FallenAngelsActualTime6Monlast3Months b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonAAABBB =  pysqldf(vm11)

### Jan 2018-Mar 2018 VS. Sep 2020-Dec2020

SubmitJan2018_Mar2018 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitJan2018_Mar2018.csv',encoding= 'iso-8859-1')

SubmitJan2018_Mar2018.columns = SubmitJan2018_Mar2018.columns.str.replace(' ', '')

SubmitJan2018_Mar2018.columns = SubmitJan2018_Mar2018.columns.str.lstrip()

SubmitJan2018_Mar2018.columns = SubmitJan2018_Mar2018.columns.str.rstrip()

SubmitJan2018_Mar2018.columns = SubmitJan2018_Mar2018.columns.str.strip()

con = sqlite3.connect("SubmitJan2018_Mar2018.db")

SubmitJan2018_Mar2018.to_sql("SubmitJan2018_Mar2018", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitJan2018_Mar2018 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitJan2018_Mar2018GrBy =  pysqldf(q3) 

SubmitJan2018_Mar2018.info()

con = sqlite3.connect("SubmitJan2018_Mar2018GrBy.db")

SubmitJan2018_Mar2018GrBy.to_sql("SubmitJan2018_Mar2018GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCntQ12018, a.SubmitAmt as SubmitAmtQ12018, b.AdvisorName as Advisorname1, b.LastSubmitDate, b.SubmitCnt as SubmitCntQ42020, b.SubmitAmt as SubmitAmtQ12020 FROM SubmitJan2018_Mar2018GrBy a INNER JOIN SubmitSep2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonEarly2018last3Months =  pysqldf(vm11)

p23  = """SELECT * FROM SubmitJan2018_Mar2018GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonEarly2018last3Months);"""
        
FallenAngelsQ12018last3Months= pysqldf(p23)

Out4 = FallenAngelsQ12018last3Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngelsQ12018last3Months.csv', index = None, header=True)


#### Comparison of timeframe Oct 2017 Mar 2018 vs Nov 2020 and Dec 2020

SubmitNov2020_Dec2020 = pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitNov2020_Dec2020.csv',encoding= 'iso-8859-1')

SubmitNov2020_Dec2020.columns = SubmitNov2020_Dec2020.columns.str.replace(' ', '')

SubmitNov2020_Dec2020.columns = SubmitNov2020_Dec2020.columns.str.lstrip()

SubmitNov2020_Dec2020.columns = SubmitNov2020_Dec2020.columns.str.rstrip()

SubmitNov2020_Dec2020.columns = SubmitNov2020_Dec2020.columns.str.strip()

con = sqlite3.connect("SubmitNov2020_Dec2020.db")

SubmitNov2020_Dec2020.to_sql("SubmitNov2020_Dec2020", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitNov2020_Dec2020 group by AdvisorContactIDText, AdvisorName;"""
      
SubmitNov2020_Dec2020GrBy =  pysqldf(q3) 

SubmitNov2020_Dec2020.info()

con = sqlite3.connect("SubmitNov2020_Dec2020GrBy.db")

SubmitNov2020_Dec2020GrBy.to_sql("SubmitNov2020_Dec2020GrBy", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr, b.AdvisorName FROM SubmitOct2017_Mar2018GrBy a INNER JOIN SubmitNov2020_Dec2020GrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonOct17Mar18vsNovDec20 =  pysqldf(vm11)

p23  = """SELECT * FROM SubmitOct2017_Mar2018GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonOct17Mar18vsNovDec20);"""
        
FallenAngels20172018last2Months= pysqldf(p23)

Out4 = FallenAngels20172018last2Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/FallenAngels20172018last2Months.csv', index = None, header=True)

### Validation

con = sqlite3.connect("FallenAngels20172018last2Months.db")

FallenAngels20172018last2Months.to_sql("FallenAngels20172018last2Months", con, if_exists='replace')

con = sqlite3.connect("FallenAngelsActualTime6Monlast3Months.db")

FallenAngelsActualTime6Monlast3Months.to_sql("FallenAngelsActualTime6Monlast3Months", con, if_exists='replace')

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt as SubmitCnt3Yr, a.SubmitAmt as SubmitAmt3Yr FROM FallenAngelsActualTime6Monlast3Months a INNER JOIN FallenAngels20172018last2Months b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonNovDecvsPrevious4month =  pysqldf(vm11)

### Overlap between Fallen Angels period over period FallenAngelsQ12018last4Months vs. FallenAngels4Monlast4Months

con = sqlite3.connect("FallenAngelsQ12018last3Months.db")

FallenAngelsQ12018last3Months.to_sql("FallenAngelsQ12018last3Months", con, if_exists='replace')

con = sqlite3.connect("FallenAngels3Monlast3Months.db")

FallenAngels3Monlast3Months.to_sql("FallenAngels3Monlast3Months", con, if_exists='replace')

vm11 = """SELECT a.*, b.* FROM FallenAngelsQ12018last3Months a INNER JOIN FallenAngels3Monlast3Months b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonFallenAngels2017Q3and2018Q1vsQ42020 =  pysqldf(vm11)


### Overlap between Fallen Angels period over period FallenAngels3Monlast3Months vs. FallenAngels3Monlast6Months

con = sqlite3.connect("FallenAngels3Monlast3Months.db")

FallenAngels3Monlast3Months.to_sql("FallenAngels3Monlast3Months", con, if_exists='replace')

con = sqlite3.connect("FallenAngels3Monlast6Months.db")

FallenAngels3Monlast6Months.to_sql("FallenAngels3Monlast6Months", con, if_exists='replace')

vm11 = """SELECT  b.* FROM FallenAngels3Monlast3Months a INNER JOIN FallenAngels3Monlast6Months b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonFallenAngelsQ4217vsQ32020vsQ42020 =  pysqldf(vm11)

#### Overlap between Oct Mar and June Decemeber Common6Monlast6Months vs. CommonActualTime6Monlast3Months


### Lets look into the Fallen Angels with Appointed Data

### NW Advisors Appointed between Oct20_Mar31st

Last6MoAppointedData=pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/Last6MoAppointedData.csv',encoding= 'iso-8859-1')

Last6MoAppointedData.columns = Last6MoAppointedData.columns.str.replace(' ', '')

Last6MoAppointedData.columns = Last6MoAppointedData.columns.str.lstrip()

Last6MoAppointedData.columns = Last6MoAppointedData.columns.str.rstrip()

Last6MoAppointedData.columns = Last6MoAppointedData.columns.str.strip()

con = sqlite3.connect("Last6MoAppointedData.db")

Last6MoAppointedData.to_sql("Last6MoAppointedData", con, if_exists='replace') 

con = sqlite3.connect("FallenAngels6Monthslast6Months.db")

FallenAngels6Monthslast6Months.to_sql("FallenAngels6Monthslast6Months", con, if_exists='replace') 

Last6MoAppointedData.info()

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.* FROM  FallenAngels6Monthslast6Months a INNER JOIN Last6MoAppointedData b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AdvisorSubandRenewedby1231 =  pysqldf(vm11)

con = sqlite3.connect("Common6Monlast6Months.db")

Common6Monlast6Months.to_sql("Common6Monlast6Months", con, if_exists='replace') 


vm12 = """SELECT a.*, b.* FROM  Common6Monlast6Months a INNER JOIN Last6MoAppointedData b on ((a.AdvisorContactIDText=b.ContactID18));"""      

RenewalAdv11 =  pysqldf(vm12)


### Validated with 1 Year Appointment data

NWAppointedLast1year=pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/NWAppointedLast1year.csv',encoding= 'iso-8859-1')

NWAppointedLast1year.columns = NWAppointedLast1year.columns.str.replace(' ', '')

NWAppointedLast1year.columns = NWAppointedLast1year.columns.str.lstrip()

NWAppointedLast1year.columns = NWAppointedLast1year.columns.str.rstrip()

NWAppointedLast1year.columns = NWAppointedLast1year.columns.str.strip()

con = sqlite3.connect("NWAppointedLast1year.db")

NWAppointedLast1year.to_sql("NWAppointedLast1year", con, if_exists='replace') 

vm11 = """SELECT a.AdvisorName, a.AdvisorContactIDText, a.SubmitCnt, a.SubmitAmt, b.* FROM FallenAngels6Monthslast6Months a INNER JOIN NWAppointedLast1year b on ((a.AdvisorContactIDText=b.ContactID18));"""      

AdvisorSubandRenewedby4567 =  pysqldf(vm11)

con = sqlite3.connect("Common6Monlast6Months.db")

Common6Monlast6Months.to_sql("Common6Monlast6Months", con, if_exists='replace') 

vm12 = """SELECT a.*, b.* FROM  Common6Monlast6Months a INNER JOIN NWAppointedLast1year b on ((a.AdvisorContactIDText=b.ContactID18));"""      

RenewalAdv1111 =  pysqldf(vm12)

Out4 = CommonEarly2018last3Months.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/CommonEarly2018last3Months.csv', index = None, header=True)

CommonEarly2018last3Months.info()

### Append the Email and Phone

SubmitSep2020_Dec2020EmailPhAppend=pd.read_csv('C:/Users/test/Documents/ContractRenewalNewHeights/SubmitSep2020_Dec2020EmailPhAppend.csv',encoding= 'iso-8859-1')

SubmitSep2020_Dec2020EmailPhAppend.columns = SubmitSep2020_Dec2020EmailPhAppend.columns.str.replace(' ', '')

SubmitSep2020_Dec2020EmailPhAppend.columns = SubmitSep2020_Dec2020EmailPhAppend.columns.str.lstrip()

SubmitSep2020_Dec2020EmailPhAppend.columns = SubmitSep2020_Dec2020EmailPhAppend.columns.str.rstrip()

SubmitSep2020_Dec2020EmailPhAppend.columns = SubmitSep2020_Dec2020EmailPhAppend.columns.str.strip()

con = sqlite3.connect("SubmitSep2020_Dec2020EmailPhAppend.db")

SubmitSep2020_Dec2020EmailPhAppend.to_sql("SubmitSep2020_Dec2020EmailPhAppend", con, if_exists='replace') 

SubmitSep2020_Dec2020EmailPhAppend.info()

con = sqlite3.connect("CommonEarly2018last3Months.db")

CommonEarly2018last3Months.to_sql("CommonEarly2018last3Months", con, if_exists='replace') 

vm11 = """SELECT a.*, b.AccountName, b.ProductCode, b.MarketerName, b.MarketerContactIDText, b.AdvisorContactPhone,b.AdvisorContactOtherPhone, b.AdvisorEmailForReports,b.MarketerEMailForReports, b.AdvisorContactEmail FROM CommonEarly2018last3Months a INNER JOIN SubmitSep2020_Dec2020EmailPhAppend b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonEarly2018last3MonthsEmailPhAppend=  pysqldf(vm11)

Out4 = CommonEarly2018last3MonthsEmailPhAppend.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/CommonEarly2018last3MonthsEmailPhAppend.csv', index = None, header=True)

q3  = """SELECT AdvisorName, AdvisorContactIDText, AccountName, ProductCode, MarketerName, MarketerContactIDText, AdvisorContactPhone, AdvisorContactOtherPhone, AdvisorEmailForReports, MarketerEMailForReports, 
       AdvisorContactEmail FROM SubmitSep2020_Dec2020EmailPhAppend group by AdvisorContactIDText, AdvisorName;"""
      
SubmitSep2020_Dec2020EmailPhAppendGrBy =  pysqldf(q3) 


con = sqlite3.connect("SubmitSep2020_Dec2020EmailPhAppendGrBy.db")

SubmitSep2020_Dec2020EmailPhAppendGrBy.to_sql("SubmitSep2020_Dec2020EmailPhAppendGrBy", con, if_exists='replace') 


vm11 = """SELECT a.*, b.AccountName, b.ProductCode, b.MarketerName, b.MarketerContactIDText, b.AdvisorContactPhone,b.AdvisorContactOtherPhone, b.AdvisorEmailForReports,b.MarketerEMailForReports, b.AdvisorContactEmail FROM CommonEarly2018last3Months a INNER JOIN SubmitSep2020_Dec2020EmailPhAppendGrBy b on ((a.AdvisorContactIDText=b.AdvisorContactIDText));"""      

CommonEarly2018last3MonthsEmailPhAppend1=  pysqldf(vm11)

Out4 = CommonEarly2018last3MonthsEmailPhAppend1.to_csv(r'C:/Users/test/Documents/ContractRenewalNewHeights/CommonEarly2018last3MonthsEmailPhAppend1.csv', index = None, header=True)












