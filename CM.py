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

Annuity2020_AfterJan17 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Post2MonthAfterJan.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Annuity2020_AfterJan17.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Annuity2020_AfterJan17.to_sql("Annuity2020_AfterJan17", con, if_exists='replace')

Annuity2020_AfterJan17.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Annuity2020_AfterJan17 group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
SubmitbyAdvisor =  pysqldf(q3)  

con = sqlite3.connect("SubmitbyAdvisor.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitbyAdvisor.to_sql("SubmitbyAdvisor", con, if_exists='replace')

ChargerMailerName = pd.read_csv('C:/Users/test/Documents/ChargerMailer/RetentionMailer.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ChargerMailerName.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName.to_sql("ChargerMailerName", con, if_exists='replace')

MostRecentFileCameFromNick = pd.read_csv('C:/Users/test/Documents/ChargerMailer/MostRecentFileCameFromNick.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("MostRecentFileCameFromNick.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MostRecentFileCameFromNick.to_sql("MostRecentFileCameFromNick", con, if_exists='replace')

### Let's get the SFContactID appended with ChargerMailerName

q44 = """SELECT a.*, b.SFContactId
      FROM ChargerMailerName a LEFT JOIN MostRecentFileCameFromNick b on a.Name=b.Name;"""
      
ChargerMailerName1 =  pysqldf(q44)  

### Now lets make sure to join 

con = sqlite3.connect("ChargerMailerName1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName1.to_sql("ChargerMailerName1", con, if_exists='replace')


q4 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubmitbyAdvisor b on (a.SFContactId= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
MatchPost2Month =  pysqldf(q4)  

MatchResults = MatchPost2Month.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchPost2Month.csv', index = None, header=True)


#### Pre Campaign Datat 1 Month
     
Jan2020Dec2019 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Jan2020Dec2019.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Jan2020Dec2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Jan2020Dec2019.to_sql("Jan2020Dec2019", con, if_exists='replace')

q31  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Jan2020Dec2019 group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
SubbyAdJan2020Dec2019 =  pysqldf(q31)  

q48 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubbyAdJan2020Dec2019 b on (a.SFContactId= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
MatchJanDec =  pysqldf(q48)  

Match6 = MatchJanDec.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchJanDec.csv', index = None, header=True)
      
      
### Pre Campaign Datat 2 Month

Jan2020Nov2019 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Jan2020Nov2019.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Jan2020Nov2019.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Jan2020Nov2019.to_sql("Jan2020Nov2019", con, if_exists='replace')

q39  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Jan2020Nov2019 group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
SubbyAdJan2020Nov2019 =  pysqldf(q39)  

q488 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubbyAdJan2020Nov2019 b on (a.SFContactId= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
Match2MonthsPre =  pysqldf(q488)  

Match66 = Match2MonthsPre.to_csv (r'C:\Users\test\Documents\ChargerMailer\Match2MonthsPre.csv', index = None, header=True)

### Let's make sure to test the the numbers in Sep and Oct

### Sep
Sep2019SubmitData = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Sep2019SubmitData.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Sep2019SubmitData.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Sep2019SubmitData.to_sql("Sep2019SubmitData", con, if_exists='replace')

l9  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Sep2019SubmitData group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
SubbySep2019SubmitData =  pysqldf(l9)  

l488 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubbyAdJan2020Nov2019 b on (a.SFContactId= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
MatchSep2019 =  pysqldf(l488)  

Out11 = MatchSep2019.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchSep2019.csv', index = None, header=True)

### Oct

Oct2019SubmiteData = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Oct2019SubmiteData.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Oct2019SubmiteData.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Oct2019SubmiteData.to_sql("Oct2019SubmiteData", con, if_exists='replace')

ll9  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Oct2019SubmiteData group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
SubbyOct2019SubmiteData =  pysqldf(ll9)  

l588 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubbyOct2019SubmiteData b on (a.SFContactId= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
MatchOct2019 =  pysqldf(l588)  

Out12 = MatchOct2019.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchOct2019.csv', index = None, header=True)


### Let's do a t-test 

### Let's bring the data

PrePostMatch2monthsForttest = pd.read_csv('C:/Users/test/Documents/ChargerMailer/PrePostMatch2monthsForttest.csv',encoding= 'iso-8859-1')

PrePostMatch2monthsForttest.info()

stats.ttest_rel(a = PrePostMatch2monthsForttest['PostSubmitCnt_Avg'],b=PrePostMatch2monthsForttest['PreSubmitCnt_Avg'])

stats.ttest_rel(a = PrePostMatch2monthsForttest['PostSubmitAmt_Avg'],b=PrePostMatch2monthsForttest['PreSubmitAmt_Avg'])

#### Validation Based on the Balke's Question

#################################################

#####################################################

### In this case, Annuity2020_AfterJan17Srinival is the data I pulled from DW through SQL for 1/17/2020 to 3/17/2020 to check the match rate


Annuity2020_AfterJan17Srinival = pd.read_csv('C:/Users/test/Documents/ChargerMailer/DataValidationSriniteam.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Annuity2020_AfterJan17Srinival.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Annuity2020_AfterJan17Srinival.to_sql("Annuity2020_AfterJan17Srinival", con, if_exists='replace')

Annuity2020_AfterJan17Srinival.info()

q145  = """SELECT AdvisorName, sfcontactid, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt
      FROM Annuity2020_AfterJan17Srinival group by AdvisorName, sfcontactid;"""
      
SubmitbyAdvisorSrinival =  pysqldf(q145)  

con = sqlite3.connect("SubmitbyAdvisorSrinival.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
SubmitbyAdvisorSrinival.to_sql("SubmitbyAdvisorSrinival", con, if_exists='replace')

ChargerMailerName = pd.read_csv('C:/Users/test/Documents/ChargerMailer/RetentionMailer.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ChargerMailerName.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName.to_sql("ChargerMailerName", con, if_exists='replace')

MostRecentFileCameFromNick = pd.read_csv('C:/Users/test/Documents/ChargerMailer/MostRecentFileCameFromNick.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("MostRecentFileCameFromNick.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MostRecentFileCameFromNick.to_sql("MostRecentFileCameFromNick", con, if_exists='replace')

### Let's get the SFContactID appended with ChargerMailerName

q44 = """SELECT a.*, b.SFContactId
      FROM ChargerMailerName a LEFT JOIN MostRecentFileCameFromNick b on a.Name=b.Name;"""
      
ChargerMailerName1 =  pysqldf(q44)  

### Now lets make sure to join 

con = sqlite3.connect("ChargerMailerName1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName1.to_sql("ChargerMailerName1", con, if_exists='replace')


q4 = """SELECT a.Name, a.sfcontactid, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN SubmitbyAdvisor b on (a.sfcontactid= b.AdvisorContactIDText or a.Name=b.AdvisorName);"""
      
MatchPost2MonthSriniVal =  pysqldf(q4)  

MatchPost2MonthSriniVal = MatchPost2MonthSriniVal.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchPost2MonthSriniVal.csv', index = None, header=True)

### Bringing a PDF shared by Srini's team to a CSV using camelot package

tables = camelot.read_pdf('C:/Users/test/Documents/ChargerMailer/NWCargerMailerReportSrinisTeam.pdf') 

tables

tables.export(r'C:\Users\test\Documents\ChargerMailer\foo.csv', f='csv', compress=True)

tables[0]

tables[0].parsing_report

Copyoffoopage_1table1 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Copyoffoopage_1table1.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Copyoffoopage_1table1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Copyoffoopage_1table1.to_sql("Copyoffoopage_1table1", con, if_exists='replace')

Copyoffoopage_1table1.info()

Post2MonthToValidateSrinisReprt04032020 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Post2MonthToValidateSrinisReprt04032020.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Post2MonthToValidateSrinisReprt04032020.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Post2MonthToValidateSrinisReprt04032020.to_sql("Post2MonthToValidateSrinisReprt04032020", con, if_exists='replace')

Post2MonthToValidateSrinisReprt04032020.info()

q444 = """SELECT a.Advisor, a.NWSales0117_0328, b.Name, b.SubmitAmt
      FROM Copyoffoopage_1table1 a Inner JOIN Post2MonthToValidateSrinisReprt04032020 b on a.Advisor= b.Name;"""
      
Joinn =  pysqldf(q444)  

######## Test2 

#######

Submit01172020_03282020 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Submit01172020_03282020.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Submit01172020_03282020.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit01172020_03282020.to_sql("Submit01172020_03282020", con, if_exists='replace')

Submit01172020_03282020.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt, ContactState, AdvisorContactMailingState_Province 
      FROM Submit01172020_03282020 group by AdvisorName, AdvisorContactIDText,ContactState, AdvisorContactMailingState_Province;"""
      
Submit01172020_03282020byAdvisor =  pysqldf(q3)  

con = sqlite3.connect("Submit01172020_03282020byAdvisor.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit01172020_03282020byAdvisor.to_sql("Submit01172020_03282020byAdvisor", con, if_exists='replace')

ChargerMailerName = pd.read_csv('C:/Users/test/Documents/ChargerMailer/RetentionMailer.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("ChargerMailerName.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName.to_sql("ChargerMailerName", con, if_exists='replace')

MostRecentFileCameFromNick = pd.read_csv('C:/Users/test/Documents/ChargerMailer/MostRecentFileCameFromNick.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("MostRecentFileCameFromNick.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MostRecentFileCameFromNick.to_sql("MostRecentFileCameFromNick", con, if_exists='replace')

### Let's get the SFContactID appended with ChargerMailerName

q44 = """SELECT a.*, b.SFContactId
      FROM ChargerMailerName a LEFT JOIN MostRecentFileCameFromNick b on a.Name=b.Name;"""
      
ChargerMailerName1 =  pysqldf(q44)  

### Now lets make sure to join 

con = sqlite3.connect("ChargerMailerName1.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
ChargerMailerName1.to_sql("ChargerMailerName1", con, if_exists='replace')


q4 = """SELECT a.Name, a.SFContactId, a.State, a.ZipCode, b.SubmitCnt, b.SubmitAmt
      FROM ChargerMailerName1 a LEFT JOIN Submit01172020_03282020byAdvisor b on (a.SFContactId= b.AdvisorContactIDText);"""
      
MatchSubmit01172020_03282020Month =  pysqldf(q4)  

MatchResults1 = MatchSubmit01172020_03282020Month.to_csv (r'C:\Users\test\Documents\ChargerMailer\MatchSubmit01172020_03282020Month.csv', index = None, header=True)










