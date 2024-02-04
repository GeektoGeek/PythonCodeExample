
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
##import featuretools as ft
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

### Bring the Athene data last 90 days

NorthAppointedAppointmentSF = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/NorthAmericanAllAppointmentSF.csv',encoding= 'iso-8859-1')

NorthAppointedAppointmentSF.columns = NorthAppointedAppointmentSF.columns.str.replace(' ', '')

NorthAppointedAppointmentSF.info()


### Bring the Athene data from 2018

NorthAppointedAppointmentDW = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/NorthAppointedAppointmentDW.csv',encoding= 'iso-8859-1')

NorthAppointedAppointmentDW.columns = NorthAppointedAppointmentDW.columns.str.replace(' ', '')

NorthAppointedAppointmentDW.info()

##Common

q3  = """SELECT a.* FROM NorthAppointedAppointmentSF a INNER JOIN NorthAppointedAppointmentDW b on (a.ContactID18=b.SFContactId);"""

Check =  pysqldf(q3) 

out = Check.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\Check.csv', index = None, header=True)

Check.info()


DataFr_Email = Check[["ContactID18", "Email"]]

DataFr_Email= DataFr_Email.loc[pd.notnull(DataFr_Email.Email)]

DataFr_AlternateEmailaddress = Check[["ContactID18", "AlternateEmailaddress"]]

DataFr_AlternateEmailaddress= DataFr_AlternateEmailaddress.loc[pd.notnull(DataFr_AlternateEmailaddress.AlternateEmailaddress)]

DataFr_NationWideEmail = Check[["ContactID18", "NationWideEmail"]]

DataFr_NationWideEmail= DataFr_NationWideEmail.loc[pd.notnull(DataFr_NationWideEmail.NationWideEmail)]

DataFr_AtheneEmail = Check[["ContactID18", "AtheneEmail"]]

DataFr_AtheneEmail= DataFr_AtheneEmail.loc[pd.notnull(DataFr_AtheneEmail.AtheneEmail)]

DataFr_AIGEmail = Check[["ContactID18", "AIGEmail"]]

DataFr_AIGEmail= DataFr_AIGEmail.loc[pd.notnull(DataFr_AIGEmail.AIGEmail)]

DataFr_NorthAmericanEmail = Check[["ContactID18", "NorthAmericanEmail"]]

###Need to check why this field is blank

DataFr_NorthAmericanEmail= DataFr_NorthAmericanEmail.loc[pd.notnull(DataFr_NorthAmericanEmail.NorthAmericanEmail)]

DataFr_MinnLifeEmail = Check[["ContactID18", "MinnLifeEmail"]]

DataFr_MinnLifeEmail= DataFr_MinnLifeEmail.loc[pd.notnull(DataFr_MinnLifeEmail.MinnLifeEmail)]

q3  = """SELECT a.* FROM DataFr_Email a INNER JOIN DataFr_AlternateEmailaddress b on (a.Email=b.AlternateEmailaddress);"""

Email_AltEmail =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_AlternateEmailaddress WHERE AlternateEmailaddress NOT IN (SELECT Email FROM Email_AltEmail);"""
        
Exlsu_AltEmail=  pysqldf(p23)

Exlsu_AltEmail['Email'] = Exlsu_AltEmail['AlternateEmailaddress']

A1= pd.merge(DataFr_Email, Exlsu_AltEmail, on="Email", how='outer')

A1.info()

out = A1.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A1.csv', index = None, header=True)

### Start the next phase
A1modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A1modi.csv',encoding= 'iso-8859-1')

A1modi.info()

DataFr_NationWideEmail.info()

q3  = """SELECT a.* FROM A1modi a INNER JOIN DataFr_NationWideEmail b on (a.Email=b.NationWideEmail);"""

A1modi_NationWideEmail =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_NationWideEmail WHERE NationWideEmail NOT IN (SELECT Email FROM A1modi_NationWideEmail);"""
        
NWEmailExclu=  pysqldf(p23)

NWEmailExclu.info()

NWEmailExclu['Email'] = NWEmailExclu['NationWideEmail']

A2= pd.merge(A1modi, NWEmailExclu, on="Email", how='outer')

A2.info()

out = A2.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A2.csv', index = None, header=True)

## Start the next phase
A2modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A2modi.csv',encoding= 'iso-8859-1')

A2modi.info()

DataFr_AtheneEmail.info()

q3  = """SELECT a.* FROM A2modi a INNER JOIN DataFr_AtheneEmail b on (a.Email=b.AtheneEmail);"""

A2modi_AtheneEmail =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_AtheneEmail WHERE AtheneEmail NOT IN (SELECT Email FROM A2modi_AtheneEmail);"""
        
AtheneEmailExclu=  pysqldf(p23)

AtheneEmailExclu.info()

AtheneEmailExclu['Email'] = AtheneEmailExclu['AtheneEmail']

A3= pd.merge(A2modi, AtheneEmailExclu, on="Email", how='outer')

A3.info()

out = A3.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A3.csv', index = None, header=True)

## Start the next phase
A3modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A3modi.csv',encoding= 'iso-8859-1')

A3modi.info()

DataFr_AIGEmail.info()

q3  = """SELECT a.* FROM A3modi a INNER JOIN DataFr_AIGEmail b on (a.Email=b.AIGEmail);"""

A3modi_AIGEmail =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_AIGEmail WHERE AIGEmail NOT IN (SELECT Email FROM A3modi_AIGEmail);"""
        
AIGEmailExclu=  pysqldf(p23)

AIGEmailExclu.info()

AIGEmailExclu['Email'] = AIGEmailExclu['AIGEmail']

A4= pd.merge(A3modi, AIGEmailExclu, on="Email", how='outer')

A4.info()

out = A4.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A4.csv', index = None, header=True)

## Start the next phase
A4modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A4modi.csv',encoding= 'iso-8859-1')

A4modi.info()

DataFr_MinnLifeEmail.info()

q3  = """SELECT a.* FROM A4modi a INNER JOIN DataFr_MinnLifeEmail b on (a.Email=b.MinnLifeEmail);"""

A4modi_MinnLifeEmail =  pysqldf(q3) 

p23  = """SELECT * FROM DataFr_MinnLifeEmail WHERE MinnLifeEmail NOT IN (SELECT Email FROM A4modi_MinnLifeEmail);"""
        
MinnLifeEmailExclu=  pysqldf(p23)

MinnLifeEmailExclu.info()

MinnLifeEmailExclu['Email'] = MinnLifeEmailExclu['MinnLifeEmail']

A5= pd.merge(A4modi, MinnLifeEmailExclu, on="Email", how='outer')

A5.info()

out = A5.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A5.csv', index = None, header=True)

A5modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A5modi.csv',encoding= 'iso-8859-1')

A5modi.info()

Check.info()

###aabb = pd.merge(A5modi, Check, left_on="ContactID18", right_on="ContactID18", how="left", validate="m:1")

###out = aa.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\aa.csv', index = None, header=True)

####  Check with RR enriched last time


RRWithAgencyName = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/rocketreachcontactsAppointedAdvisrosWithAgencyName.csv',encoding= 'iso-8859-1')

RRWithAgencyName.columns = RRWithAgencyName.columns.str.replace(' ', '')

RRWithAgencyName.info()

RRWithAgencyName_RecoEmail = RRWithAgencyName['RecommendedEmail']

RRWithAgencyName_RecoEmail= RRWithAgencyName.loc[pd.notnull(RRWithAgencyName.RecommendedEmail)]

q3  = """SELECT a.* FROM A5modi a INNER JOIN RRWithAgencyName_RecoEmail b on (a.Email=b.RecommendedEmail);"""

A5modi_RRRecoEmail =  pysqldf(q3) 

p23  = """SELECT * FROM RRWithAgencyName_RecoEmail WHERE RecommendedEmail NOT IN (SELECT Email FROM A5modi_RRRecoEmail);"""
        
RRExclul=  pysqldf(p23)

RRExclul.info()

RRExclul['Email']= RRExclul['RecommendedEmail']

A6= pd.merge(A5modi, RRExclul, on="Email", how='outer')

out = A6.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A6.csv', index = None, header=True)

A6modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A6modi.csv',encoding= 'iso-8859-1')

A6modi.info()

Check.info()

NorthAmericanFinal= pd.merge(A6modi, Check,left_on="ContactID18", right_on="ContactID18", how="right", validate="m:m")

out = NorthAmericanFinal.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\NorthAmericanFinal.csv', index = None, header=True)

RRWithAgencyName.info()

RRWithAgencyName_PerEmail = RRWithAgencyName['RecommendedPersonalEmail']

RRWithAgencyName_PerEmail= RRWithAgencyName.loc[pd.notnull(RRWithAgencyName.RecommendedPersonalEmail)]

q3  = """SELECT a.* FROM A6modi a INNER JOIN RRWithAgencyName_PerEmail b on (a.Email=b.RecommendedPersonalEmail);"""

RRExclulPer =  pysqldf(q3) 

p23  = """SELECT * FROM RRWithAgencyName_PerEmail WHERE RecommendedPersonalEmail NOT IN (SELECT Email FROM RRExclulPer);"""
        
RRExclulPer1=  pysqldf(p23)

A78= pd.merge(A6modi, RRExclulPer1, on="ContactID18", how='outer')

NorthAmericanFinalU= pd.merge(A78, Check,left_on="ContactID18", right_on="ContactID18", how="right", validate="m:m")

out = A7.to_csv (r'C:\Users\test\Documents\SampleSizeExMarketingProgAds\AdlistsRefresh_10072022\NorthAmerican\A7.csv', index = None, header=True)

A7modi = pd.read_csv('C:/Users/test/Documents/SampleSizeExMarketingProgAds/AdlistsRefresh_10072022/NorthAmerican/A7modi.csv',encoding= 'iso-8859-1')

A7modi.info()

NorthAmericanFinal12= pd.merge(A7modi, Check,left_on="ContactID18", right_on="ContactID18", how="right", validate="m:m")







