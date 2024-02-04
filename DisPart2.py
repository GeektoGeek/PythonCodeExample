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
import tabula
from tabula import convert_into
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

### Let's look into Athene First

### Take Athene Segment From Discovery

AtheneAppDiscovery= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/AtheneAppointedAdvisorsinDiscovery.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneAppDiscovery.db")

AtheneAppDiscovery.to_sql("AtheneAppDiscovery", con, if_exists='replace')

AtheneAppDiscovery.info()

#### Take Athene Segment from Annexus SQL

AtheneSQLAdvisor= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/SQL_AtheneAdviros.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneAppDiscovery.db")

AtheneAppDiscovery.to_sql("AtheneAppDiscovery", con, if_exists='replace')

AtheneAppDiscovery.info()

AtheneSQLAdvisor.info()

qqq1  = """SELECT a.* FROM AtheneAppDiscovery a INNER JOIN AtheneSQLAdvisor b on a.NPN = b.NPN;"""
        
CommonAnnexus_Disco =  pysqldf(qqq1) 

#### Take Athene Segment from Annexus Jason Email List shared last week

AtheneAdvisorJasonEmail= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/All_AtheneAppointedAdvirosEmailSegment.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AtheneAdvisorJasonEmail.db")

AtheneAdvisorJasonEmail.to_sql("AtheneAdvisorJasonEmail", con, if_exists='replace')

AtheneAdvisorJasonEmail.info()

qqq2  = """SELECT a.* FROM AtheneAppDiscovery a INNER JOIN AtheneAdvisorJasonEmail b on a.NPN = b.NPN;"""
        
CommonAnnexusEmail_Disc =  pysqldf(qqq2) 

con = sqlite3.connect("CommonAnnexus_Disco.db")

CommonAnnexus_Disco.to_sql("CommonAnnexus_Disco", con, if_exists='replace')

### Athene Not Appointed Segment 1
p2  = """SELECT * FROM AtheneAppDiscovery WHERE NPN NOT IN (SELECT NPN FROM CommonAnnexus_Disco);"""
        
AtheneNotAppointedAnnexus1 =  pysqldf(p2)  

con = sqlite3.connect("AtheneNotAppointedAnnexus1.db")

AtheneNotAppointedAnnexus.to_sql("AtheneNotAppointedAnnexus1", con, if_exists='replace')

### Athene Not Appointed Segment 2
p3  = """SELECT * FROM AtheneAppDiscovery WHERE NPN NOT IN (SELECT NPN FROM CommonAnnexusEmail_Disc);"""
        
AtheneNotAppointedAnnexus2 =  pysqldf(p3)  

#### Let's do Allianz

AllianzAppAdvDiscovery= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/AllianzAppointementAdvisorsDiscovery.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AllianzAppAdvDiscovery.db")

AllianzAppAdvDiscovery.to_sql("AllianzAppAdvDiscovery", con, if_exists='replace')

AIG_AppointmentAdvisors= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/AIG_AppointmentAdvisors.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AIG_AppointmentAdvisors.db")

AIG_AppointmentAdvisors.to_sql("AIG_AppointmentAdvisors", con, if_exists='replace')

qqq4  = """SELECT a.* FROM AllianzAppAdvDiscovery a INNER JOIN AIG_AppointmentAdvisors b on a.NPN = b.PartyNPN;"""
        
CommonAllianz_AIG =  pysqldf(qqq4) 

con = sqlite3.connect("CommonAllianz_AIG.db")

CommonAllianz_AIG.to_sql("CommonAllianz_AIG", con, if_exists='replace')

p33  = """SELECT * FROM AllianzAppAdvDiscovery WHERE NPN NOT IN (SELECT NPN FROM CommonAllianz_AIG);"""
        
AIGNotAppointedAllianz =  pysqldf(p33)  

con = sqlite3.connect("AIGNotAppointedAllianz.db")

AIGNotAppointedAllianz.to_sql("AIGNotAppointedAllianz", con, if_exists='replace')


con = sqlite3.connect("AtheneNotAppointedAnnexus2.db")

AtheneNotAppointedAnnexus2.to_sql("AtheneNotAppointedAnnexus2", con, if_exists='replace')

qqq5  = """SELECT a.* FROM AIGNotAppointedAllianz a INNER JOIN AtheneNotAppointedAnnexus2 b on a.NPN = b.NPN;"""
        
Allianz_AtheneCommon =  pysqldf(qqq5) 

tables = camelot.read_pdf('C:/Users/test/Documents/DiscoveryNewProject05242020/AIG_Restricted_BD.pdf') 

print("Total tables extracted:", tables.n)

print(tables[0].df)

tables

tables.export(r'C:\Users\test\Documents\DiscoveryNewProject05242020\AIG_Restricted_BD.csv', f='csv', compress=True)

a=tables[0]

tables[0,1].parsing_report

Copyoffoopage_1table1 = pd.read_csv('C:/Users/test/Documents/ChargerMailer/Copyoffoopage_1table1.csv',encoding= 'iso-8859-1')

#### Lets get the data loaded from 

DiscoveryTest= pd.read_csv('C:/Users/test/Documents/DiscoveryNewProject05242020/DiscoveryTest.csv',encoding= 'iso-8859-1')

DiscoveryTest.info()

con = sqlite3.connect("DiscoveryTest.db")

DiscoveryTest.to_sql("DiscoveryTest", con, if_exists='replace')

qqq5  = """SELECT NPN, Email_PersonalType FROM DiscoveryTest;"""
        
EmailPersonal =  pysqldf(qqq5) 

EmailPersonal1 = EmailPersonal.dropna(how='any',axis=0) 

con = sqlite3.connect("EmailPersonal1.db")

EmailPersonal1.to_sql("EmailPersonal1", con, if_exists='replace')

qqq6  = """SELECT NPN, Email_BusinessType FROM DiscoveryTest;"""
        
EmailBusiness =  pysqldf(qqq6) 

EmailBusiness1 = EmailBusiness.dropna(how='any',axis=0) 

con = sqlite3.connect("EmailBusiness1.db")

EmailBusiness1.to_sql("EmailBusiness1", con, if_exists='replace')

qqq8  = """SELECT a.NPN, a.Email_PersonalType, b.Email_BusinessType FROM EmailPersonal1 a Inner JOIN EmailBusiness1 b on a.NPN=b.NPN;"""
        
EmailCommon =  pysqldf(qqq8) 

### Common is the common element between Business and Personal

### Remove the common from Business1 

con = sqlite3.connect("EmailCommon.db")

EmailCommon.to_sql("EmailCommon", con, if_exists='replace')

p23  = """SELECT * FROM EmailBusiness1 WHERE NPN NOT IN (SELECT NPN FROM EmailCommon);"""
        
EmailBusinessOnly =  pysqldf(p23)

p24  = """SELECT * FROM EmailPersonal1 WHERE NPN NOT IN (SELECT NPN FROM EmailCommon);"""
        
EmailPersonalOnly =  pysqldf(p24)

###Let's enrich the EmailBusinessOnly dataset

con = sqlite3.connect("EmailBusinessOnly.db")

EmailBusinessOnly.to_sql("EmailBusinessOnly", con, if_exists='replace')

EmailBusinessOnly.info()

qqq1  = """SELECT a.NPN as NPN1, a.Email_BusinessType as Email_Business, b.* FROM EmailBusinessOnly a LEFT JOIN DiscoveryTest b on a.NPN = b.NPN;"""
        
EmailBusinessOnlyAllAttr =  pysqldf(qqq1) 

export_csv1 = EmailBusinessOnlyAllAttr.to_csv(r'C:\Users\test\Documents\DiscoveryNewProject05242020\DataCamefromNathalea\FinalProcessing10012020\EmailBusinessOnlyAllAttr.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 

###Let's enrich the EmailPersonalOnly dataset

con = sqlite3.connect("EmailPersonalOnly.db")

EmailPersonalOnly.to_sql("EmailPersonalOnly", con, if_exists='replace')

EmailPersonalOnly.info()

qqq1  = """SELECT a.NPN as NPN1, a.Email_PersonalType as Email_Personal, b.* FROM EmailPersonalOnly a LEFT JOIN DiscoveryTest b on a.NPN = b.NPN;"""
        
EmailPersonalOnlyAllAttr =  pysqldf(qqq1) 

export_csv1 = EmailPersonalOnlyAllAttr.to_csv(r'C:\Users\test\Documents\DiscoveryNewProject05242020\DataCamefromNathalea\FinalProcessing10012020\EmailPersonalOnlyAllAttr.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 

### Now let's process the common group which contain both Business and Personal Email. This is the group 

EmailCommon.info()

qqq1  = """SELECT a.NPN as NPN1, a.Email_PersonalType as EmailPersonal, a.Email_BusinessType as Email_Business, b.* FROM EmailCommon a LEFT JOIN DiscoveryTest b on a.NPN = b.NPN;"""
        
EmailCommonAllAttr =  pysqldf(qqq1) 

export_csv1 = EmailCommonAllAttr.to_csv(r'C:\Users\test\Documents\DiscoveryNewProject05242020\DataCamefromNathalea\FinalProcessing10012020\EmailCommonAllAttr.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path 
