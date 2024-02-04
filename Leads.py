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

###What agents that wrote TA 5 yr are NOT writing Athene 6 year.  All of them should be leads. 

Athene6Year_PaidData2020_2021 = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/Athene6Year_PaidData2020_2021Updated.csv',encoding= 'iso-8859-1')

Athene6Year_PaidData2020_2021.columns = Athene6Year_PaidData2020_2021.columns.str.replace(' ', '')

con = sqlite3.connect("Athene6Year_PaidData2020_2021.db")

Athene6Year_PaidData2020_2021.to_sql("Athene6Year_PaidData2020_2021", con, if_exists='replace')

Athene6Year_PaidData2020_2021.info()

 
TA5YearsPaid2020_2021 = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/TA5YearsPaid2020_2021Updated.csv',encoding= 'iso-8859-1')

TA5YearsPaid2020_2021.columns = TA5YearsPaid2020_2021.columns.str.replace(' ', '')

con = sqlite3.connect("TA5YearsPaid2020_2021.db")

TA5YearsPaid2020_2021.to_sql("TA5YearsPaid2020_2021", con, if_exists='replace')

TA5YearsPaid2020_2021.info()

##SFContactId

vm1 = """SELECT a.* FROM TA5YearsPaid2020_2021 a INNER JOIN Athene6Year_PaidData2020_2021 b on (a.SFContactId= b.SFContactId);"""      

TAandAtheneCommon =  pysqldf(vm1)

con = sqlite3.connect("TAandAtheneCommon.db")

TAandAtheneCommon.to_sql("TAandAtheneCommon", con, if_exists='replace')


p23  = """SELECT * FROM TA5YearsPaid2020_2021 WHERE SFContactId NOT IN (SELECT SFContactId FROM TAandAtheneCommon);"""
        
leadsFinal=  pysqldf(p23)

Out4 = leadsFinal.to_csv(r'C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/leadsFinal.csv', index = None, header=True)


Athene6Year_PaidData2020_2021U1 = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/Athene6Year_PaidData2020_2021U1.csv',encoding= 'iso-8859-1')

Athene6Year_PaidData2020_2021U1.columns = Athene6Year_PaidData2020_2021U1.columns.str.replace(' ', '')

con = sqlite3.connect("Athene6Year_PaidData2020_2021U1.db")

Athene6Year_PaidData2020_2021U1.to_sql("Athene6Year_PaidData2020_2021U1", con, if_exists='replace')

Athene6Year_PaidData2020_2021U1.info()


q3  = """SELECT SFContactId, FullName, count(PolicyNumber4) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate FROM Athene6Year_PaidData2020_2021U1 group by SFContactId, FullName  ;"""     

Athene6Year_PaidData2020_2021U1grby =  pysqldf(q3)  

con = sqlite3.connect("Athene6Year_PaidData2020_2021U1grby.db")

Athene6Year_PaidData2020_2021U1grby.to_sql("Athene6Year_PaidData2020_2021U1grby", con, if_exists='replace')

TA5YearsPaid2020_2021U1 = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/TA5YearsPaid2020_2021U1.csv',encoding= 'iso-8859-1')

TA5YearsPaid2020_2021U1.columns = TA5YearsPaid2020_2021U1.columns.str.replace(' ', '')

con = sqlite3.connect("TA5YearsPaid2020_2021U1.db")

TA5YearsPaid2020_2021U1.to_sql("TA5YearsPaid2020_2021U1", con, if_exists='replace')

TA5YearsPaid2020_2021U1.info()

q3  = """SELECT SFContactId, FullName, count(PolicyNumber4) as PolicyCnt, sum(TP) as Premium, max(IssueDate) as LastIssueDate FROM TA5YearsPaid2020_2021U1 group by SFContactId, FullName  ;"""     

TA5YearsPaid2020_2021U1U1grby =  pysqldf(q3)  


con = sqlite3.connect("TA5YearsPaid2020_2021U1U1grby.db")

TA5YearsPaid2020_2021U1U1grby.to_sql("TA5YearsPaid2020_2021U1U1grby", con, if_exists='replace')


vm1 = """SELECT a.* FROM TA5YearsPaid2020_2021U1U1grby a INNER JOIN Athene6Year_PaidData2020_2021U1grby b on (a.SFContactId= b.SFContactId);"""      

TAandAtheneCommon1 =  pysqldf(vm1)

con = sqlite3.connect("TAandAtheneCommon1.db")

TAandAtheneCommon1.to_sql("TAandAtheneCommon1", con, if_exists='replace')

p23  = """SELECT * FROM TA5YearsPaid2020_2021U1U1grby WHERE SFContactId NOT IN (SELECT SFContactId FROM TAandAtheneCommon1);"""
        
leadsFinal1=  pysqldf(p23)

Out4 = leadsFinal1.to_csv(r'C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/leadsFinal1.csv', index = None, header=True)

### TA Submit data

TASubmit = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/Transamericasubmits0101202_06302021.csv',encoding= 'iso-8859-1')

TASubmit.columns = TASubmit.columns.str.replace(' ', '')

con = sqlite3.connect("TASubmit.db")

TASubmit.to_sql("TASubmit", con, if_exists='replace')

TASubmit.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, AccountName, count(SubmitDate) as SubmitCnt2020_2021, sum(SubmitAmount) as SubmitAmt2019, max(SubmitDate) as LastSubmitDate FROM TASubmit group by AdvisorContactIDText, AdvisorName  ;"""     

TASubmitGroupBy =  pysqldf(q3)  

TASubmitGroupBy.to_sql("TASubmitGroupBy", con, if_exists='replace')

TASubmitGroupBy.info()

leadsFinal1.to_sql("leadsFinal1", con, if_exists='replace')

leadsFinal1.info()

vm1 = """SELECT a.*, b.* FROM leadsFinal1 a INNER JOIN TASubmitGroupBy b on ((a.SFContactId = b.AdvisorContactIDText));"""      

leadsFinal1Submit =  pysqldf(vm1)

Out4 = leadsFinal1Submit.to_csv(r'C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/leadsFinal1Submit.csv', index = None, header=True)

### Take the TA Leads Share

leadsFinal_Share = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/leadsFinal_Share.csv',encoding= 'iso-8859-1')

leadsFinal_Share.columns = leadsFinal_Share.columns.str.replace(' ', '')

con = sqlite3.connect("leadsFinal_Share.db")

leadsFinal_Share.to_sql("TleadsFinal_Share", con, if_exists='replace')

leadsFinal_Share.info()

vm1 = """SELECT a.*, b.* FROM leadsFinal_Share a LEFT JOIN TASubmitGroupBy b on ((a.SFContactId = b.AdvisorContactIDText));"""      

leadsFinal_ShareSubmit =  pysqldf(vm1)

Out4 = leadsFinal_ShareSubmit.to_csv(r'C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/leadsFinal_ShareSubmit.csv', index = None, header=True)

##### Working on Brian's request

### TA Advisors appointed but not appointed with NW and Athene

TAAllAppointedSubmit = pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/TAAllAppointedEmailSegmentwith3yearsSubmit.csv',encoding= 'iso-8859-1')

TAAllAppointedSubmit.columns = TAAllAppointedSubmit.columns.str.replace(' ', '')

con = sqlite3.connect("TAAllAppointedSubmit.db")

TAAllAppointedSubmit.to_sql("TAAllAppointedSubmit", con, if_exists='replace')

TAAllAppointedSubmit.info()

AtheneEmailSegment= pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/AtheneEmailSegment.csv',encoding= 'iso-8859-1')

AtheneEmailSegment.columns = AtheneEmailSegment.columns.str.replace(' ', '')

con = sqlite3.connect("AtheneEmailSegment.db")

AtheneEmailSegment.to_sql("AtheneEmailSegment", con, if_exists='replace')


vm11 = """SELECT a.* FROM TAAllAppointedSubmit a INNER JOIN AtheneEmailSegment b on ((a.ContactID18 = b.ContactID18));"""      

CommonTAandAth =  pysqldf(vm11)

con = sqlite3.connect("CommonTAandAth.db")

CommonTAandAth.to_sql("CommonTAandAth", con, if_exists='replace')

p23  = """SELECT * FROM TAAllAppointedSubmit WHERE ContactID18 NOT IN (SELECT ContactID18 FROM CommonTAandAth);"""
        
RemainingTAAdvisor=  pysqldf(p23)

con = sqlite3.connect("RemainingTAAdvisor.db")

RemainingTAAdvisor.to_sql("RemainingTAAdvisor", con, if_exists='replace')

NWEmailSegment= pd.read_csv('C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/NWEmailSegment.csv',encoding= 'iso-8859-1')

NWEmailSegment.columns = NWEmailSegment.columns.str.replace(' ', '')

con = sqlite3.connect("NWEmailSegment.db")

NWEmailSegment.to_sql("NWEmailSegment", con, if_exists='replace')

vm11 = """SELECT a.* FROM TAAllAppointedSubmit a INNER JOIN RemainingTAAdvisor b on ((a.ContactID18 = b.ContactID18));"""      

CommonTAandNW =  pysqldf(vm11)

con = sqlite3.connect("CommonTAandNW.db")

CommonTAandNW.to_sql("CommonTAandNW", con, if_exists='replace')

p23  = """SELECT * FROM RemainingTAAdvisor WHERE ContactID18 NOT IN (SELECT ContactID18 FROM CommonTAandNW);"""
        
RemainingTAAdvisor1=  pysqldf(p23)

Out4 = RemainingTAAdvisor.to_csv(r'C:/Users/DSarkar/Documents/TA5YearsButNoAthene6Yr/RemainingTAAdvisor.csv', index = None, header=True)





