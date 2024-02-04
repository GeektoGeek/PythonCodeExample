
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
from pandasql import sqldf
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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

##Securian PendingandUnpaid
PendingandUnpaid09062022Securian1 = pd.read_csv('C:/Users/test/Documents/LifeAnalysis/PendingandUnpaid09062022Securian1.csv',encoding= 'iso-8859-1')

PendingandUnpaid09062022Securian1.columns = PendingandUnpaid09062022Securian1.columns.str.replace(' ', '')

PendingandUnpaid09062022Securian1.info()

con = sqlite3.connect("PendingandUnpaid09062022Securian1.db")

PendingandUnpaid09062022Securian1.to_sql("PendingandUnpaid09062022Securian1", con, if_exists='replace')

q3  = """SELECT BGA, sum(BaseFace) as BaseFace_PendUnpaid, sum(TgtPremium) as TgtPrm_PendUnpaid, count(Policy) as Pending_InpaidPolicy, sum(PlannedPremium) as PlannedPrem_PendUnpaid, sum(AtIssuePremium) as AtIssuePrem_PendUnpaid, ServiceRep FROM PendingandUnpaid09062022Securian1 group by ServiceRep;"""     

offenderbyServiceRep =  pysqldf(q3)  

Out4 = offenderbyServiceRep.to_csv(r'C:/Users/test/Documents/LifeAnalysis/offenderbyServiceRep.csv', index = None, header=True)

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm, sum(PlannedPremium) as PlannedPrem, sum(AtIssuePremium) as AtIssuePrem, ServiceRep, Policy FROM PendingandUnpaid09062022Securian1 group by ServiceRep,Policy;"""     

offenderbypolicynum =  pysqldf(q3)  

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm,sum(PlannedPremium) as PlannedPrem, sum(AtIssuePremium) as AtIssuePrem FROM PendingandUnpaid09062022Securian1 group by BGA;"""     

offenderbyBGA =  pysqldf(q3)  

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm, sum(PlannedPremium) as PlannedPrem, sum(AtIssuePremium) as AtIssuePrem, ServiceRep FROM PendingandUnpaid09062022Securian1 group by BGA,ServiceRep;"""     

offenderbyBGA_Servicerep =  pysqldf(q3)  

q3  = """SELECT distinct(ServiceRep) FROM PendingandUnpaid09062022Securian1;"""     

unique_Servicerep_pending =  pysqldf(q3)  


q3  = """SELECT distinct(Policy) FROM PendingandUnpaid09062022Securian1;"""     

unique_Policy_pending =  pysqldf(q3)  

#### Securian Paid

Paid09062022Securian1 = pd.read_csv('C:/Users/test/Documents/LifeAnalysis/Paid09062022.csv',encoding= 'iso-8859-1')

Paid09062022Securian1.columns = Paid09062022Securian1.columns.str.replace(' ', '')

Paid09062022Securian1.info()

con = sqlite3.connect("Paid09062022Securian1.db")

Paid09062022Securian1.to_sql("Paid09062022Securian1", con, if_exists='replace')

q3  = """SELECT BGA, sum(BaseFace) as BaseFace_Paid, count(Policy) as PaidPolicyCount, sum(TgtPremium) as TgtPrm_Paid, sum(PlannedPrem) as PlannedPrem_Paid, sum(AtIssuePrem) as AtIssuePrem_Paid, ServiceRep FROM Paid09062022Securian1 group by ServiceRep;"""     

PaidbyServiceRep =  pysqldf(q3)  

Out4 = PaidbyServiceRep.to_csv(r'C:/Users/test/Documents/LifeAnalysis/PaidbyServiceRep.csv', index = None, header=True)

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm, sum(PlannedPrem) as PlannedPrem, sum(AtIssuePrem) as AtIssuePrem, ServiceRep, Policy FROM Paid09062022Securian1 group by ServiceRep,Policy;"""     

Paidbypolicynum =  pysqldf(q3)  

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm, sum(PlannedPrem) as PlannedPrem, sum(AtIssuePrem) as AtIssuePrem FROM Paid09062022Securian1 group by BGA;"""     

PaidbyBGA =  pysqldf(q3)  

q3  = """SELECT BGA, sum(BaseFace) as BaseFace, sum(TgtPremium) as TgtPrm, sum(PlannedPrem) as PlannedPrem, sum(AtIssuePrem) as AtIssuePrem, ServiceRep FROM Paid09062022Securian1 group by BGA, ServiceRep;"""     

PaidbyBGA_ServiceRep =  pysqldf(q3)  

q3  = """SELECT distinct(ServiceRep) FROM Paid09062022Securian1;"""     

unique_Servicerep_paid =  pysqldf(q3)  


q3  = """SELECT distinct(Policy) FROM Paid09062022Securian1;"""     

unique_Policy_paid =  pysqldf(q3)  

### Paid to Pending ratio

## By Service Rep

con = sqlite3.connect("offenderbyServiceRep.db")

offenderbyServiceRep.to_sql("offenderbyServiceRep", con, if_exists='replace')

con = sqlite3.connect("PaidbyServiceRep.db")

PaidbyServiceRep.to_sql("PaidbyServiceRep", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM offenderbyServiceRep a LEFT JOIN PaidbyServiceRep b on ((a.ServiceRep =b.ServiceRep));"""      
 
offenderpending_Paid1=  pysqldf(vm1)

Out4 = offenderpending_Paid1.to_csv(r'C:/Users/test/Documents/LifeAnalysis/offenderpending_Paid1.csv', index = None, header=True)

## By BGA

con = sqlite3.connect("offenderbyBGA.db")

offenderbyBGA.to_sql("offenderbyBGA", con, if_exists='replace')

con = sqlite3.connect("PaidbyBGA.db")

PaidbyBGA.to_sql("PaidbyBGA", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM offenderbyBGA a LEFT JOIN PaidbyBGA b on ((a.BGA =b.BGA));"""      
 
BGApending_Paid=  pysqldf(vm1)

###

q3  = """SELECT BGA, count(Policy) as Paid_Policy_Count FROM Paid09062022Securian1 group by BGA;"""     

Paid_BGA =  pysqldf(q3)  


q3  = """SELECT BGA, count(Policy) as UnPaid_Policy_Count FROM PendingandUnpaid09062022Securian1 group by BGA;"""     

UnPaid_BGA =  pysqldf(q3)  


con = sqlite3.connect("Paid_BGA.db")

Paid_BGA.to_sql("Paid_BGA", con, if_exists='replace')

con = sqlite3.connect("UnPaid_BGA.db")

UnPaid_BGA.to_sql("UnPaid_BGA", con, if_exists='replace')

vm1 = """SELECT a.*, b.* FROM Paid_BGA a LEFT JOIN UnPaid_BGA b on ((a.BGA =b.BGA));"""      
 
check=  pysqldf(vm1)

Out4 = check.to_csv(r'C:/Users/test/Documents/LifeAnalysis/check.csv', index = None, header=True)

## By Nationwide

NHIUL_SarQuery= pd.read_csv('C:/Users/test/Documents/LifeAnalysis/NHIUL_SarQuery.csv',encoding= 'iso-8859-1')

NHIUL_SarQuery.columns = NHIUL_SarQuery.columns.str.replace(' ', '')

NHIUL_SarQuery.info()

NHIUL_SarQuery['submitamount'].sum()

con = sqlite3.connect("NHIUL_SarQuery.db")

NHIUL_SarQuery.to_sql("NHIUL_SarQuery", con, if_exists='replace')

q3  = """SELECT AdvisorName, ContactID1, IMOName, count(SubmitDate) as SubmitCnt2022, sum(SubmitAmount) as SubmitAmt2022, max(SubmitDate) as LastSubmitDate, MarketerName FROM NHIUL_SarQuery group by ContactID1, AdvisorName;"""     

NHIUL_SarQueryGrBy =  pysqldf(q3)  

con = sqlite3.connect("NHIUL_SarQueryGrBy.db")

NHIUL_SarQueryGrBy.to_sql("NHIUL_SarQueryGrBy", con, if_exists='replace')

NHIUL_SarQueryGrBy.info()

offenderbyServiceRep.info()


#vm1 = """SELECT a.* FROM NHIUL_09282022GrBy a INNER JOIN offenderbyServiceRep b on ((a.AdvisorName =b.ServiceRep));"""      
 
#Crossover=  pysqldf(vm1)

#PaidbyServiceRep.info()

#vm1 = """SELECT a.* FROM NHIUL_09282022GrBy a INNER JOIN PaidbyServiceRep b on ((a.AdvisorName =b.ServiceRep));"""      
 
#Crossover1=  pysqldf(vm1)

### Crossover between NH-IUL and Securian

MinnLifeSubmits= pd.read_csv('C:/Users/test/Documents/LifeAnalysis/MinnLife_SarQuery.csv',encoding= 'iso-8859-1')

MinnLifeSubmits.columns = MinnLifeSubmits.columns.str.replace(' ', '')

MinnLifeSubmits.info()

MinnLifeSubmits['submitamount'].sum()

con = sqlite3.connect("MinnLifeSubmits.db")

MinnLifeSubmits.to_sql("MinnLifeSubmits", con, if_exists='replace')

q3  = """SELECT AdvisorName, ContactID1, IMOName, count(SubmitDate) as SubmitCnt2022, sum(SubmitAmount) as SubmitAmt2022, max(SubmitDate) as LastSubmitDate, MarketerName FROM MinnLifeSubmits group by ContactID1, AdvisorName;"""     

MinnLifeSubmitsGrBy =  pysqldf(q3)   

con = sqlite3.connect("MinnLifeSubmitsGrBy.db")

MinnLifeSubmitsGrBy.to_sql("MinnLifeSubmitsGrBy", con, if_exists='replace')

MinnLifeSubmitsGrBy.info()

vm1 = """SELECT a.* FROM NHIUL_SarQueryGrBy a INNER JOIN MinnLifeSubmitsGrBy b on ((a.AdvisorName =b.AdvisorName));"""      
 
Crossover5=  pysqldf(vm1)

Out4 = Crossover5.to_csv(r'C:/Users/test/Documents/LifeAnalysis/Crossover5.csv', index = None, header=True)



#####use SarQuery

