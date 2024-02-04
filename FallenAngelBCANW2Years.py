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

### Nationwide

NWLast2Years = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/NWLast2Years.csv',encoding= 'iso-8859-1')

NWLast2Years.columns = NWLast2Years.columns.str.replace(' ', '')

NWLast2Years.columns = NWLast2Years.columns.str.lstrip()

NWLast2Years.columns = NWLast2Years.columns.str.rstrip()

NWLast2Years.columns = NWLast2Years.columns.str.strip()

con = sqlite3.connect("NWLast2Years.db")

NWLast2Years.to_sql("NWLast2Years", con, if_exists='replace')

NWLast2Years.info()


NWLast3Months = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/NWLast3Months.csv',encoding= 'iso-8859-1')

NWLast3Months.columns = NWLast3Months.columns.str.replace(' ', '')

NWLast3Months.columns = NWLast3Months.columns.str.lstrip()

NWLast3Months.columns = NWLast3Months.columns.str.rstrip()

NWLast3Months.columns = NWLast3Months.columns.str.strip()

con = sqlite3.connect("NWLast3Months.db")

NWLast3Months.to_sql("NWLast3Months", con, if_exists='replace')

NWLast3Months.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM NWLast2Years group by AdvisorContactIDText, AdvisorName;"""
      
NWLast2YearsGrBy =  pysqldf(q3)  

con = sqlite3.connect("NWLast2YearsGrBy.db")

NWLast2YearsGrBy.to_sql("NWLast2YearsGrBy", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM NWLast3Months group by  AdvisorContactIDText, AdvisorName;"""
      
NWLast3MonthsGrBy =  pysqldf(q3)  

con = sqlite3.connect("NWLast3MonthsGrBy.db")

NWLast3MonthsGrBy.to_sql("NWLast3MonthsGrBy", con, if_exists='replace')

vm1 = """SELECT a.* FROM NWLast2YearsGrBy a INNER JOIN NWLast3MonthsGrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

Commonvmb =  pysqldf(vm1)

######

con = sqlite3.connect("Commonvmb.db")

Commonvmb.to_sql("Commonvmb", con, if_exists='replace')

p2  = """SELECT * FROM NWLast2YearsGrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Commonvmb);"""
        
FallenNWAdvisor =  pysqldf(p2) 

Out4 = FallenNWAdvisor.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/FallenNWAdvisor.csv', index = None, header=True)


#### Athene

AtheneLast2Years = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/AtheneLast2YearsBCA.csv',encoding= 'iso-8859-1')

AtheneLast2Years.columns = AtheneLast2Years.columns.str.replace(' ', '')

AtheneLast2Years.columns = AtheneLast2Years.columns.str.lstrip()

AtheneLast2Years.columns = AtheneLast2Years.columns.str.rstrip()

AtheneLast2Years.columns = AtheneLast2Years.columns.str.strip()

con = sqlite3.connect("AtheneLast2Years.db")

AtheneLast2Years.to_sql("AtheneLast2Years", con, if_exists='replace')

AtheneLast2Years.info()

AtheneLast3Months = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/AtheneLast3MonthsBCA.csv',encoding= 'iso-8859-1')

AtheneLast3Months.columns = AtheneLast3Months.columns.str.replace(' ', '')

AtheneLast3Months.columns = AtheneLast3Months.columns.str.lstrip()

AtheneLast3Months.columns = AtheneLast3Months.columns.str.rstrip()

AtheneLast3Months.columns = AtheneLast3Months.columns.str.strip()

con = sqlite3.connect("AtheneLast3Months.db")

AtheneLast3Months.to_sql("AtheneLast3Months", con, if_exists='replace')

AtheneLast3Months.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM AtheneLast2Years group by AdvisorContactIDText, AdvisorName;"""
      
AtheneLast2YearsGrBy =  pysqldf(q3)  

con = sqlite3.connect("AtheneLast2YearsGrBy.db")

AtheneLast2YearsGrBy.to_sql("AtheneLast2YearsGrBy", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM AtheneLast3Months group by AdvisorContactIDText, AdvisorName;"""
      
AtheneLast3MonthsGrBy =  pysqldf(q3)  

con = sqlite3.connect("AtheneLast3MonthsGrBy.db")

AtheneLast3MonthsGrBy.to_sql("AtheneLast3MonthsGrBy", con, if_exists='replace')

vm1 = """SELECT a.* FROM AtheneLast2YearsGrBy a INNER JOIN AtheneLast3MonthsGrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

Commonvmb1 =  pysqldf(vm1)

######

con = sqlite3.connect("Commonvmb1.db")

Commonvmb1.to_sql("Commonvmb1", con, if_exists='replace')

p2  = """SELECT * FROM AtheneLast2YearsGrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Commonvmb1);"""
        

FallenAtheneAdvisor =  pysqldf(p2) 

Out4 = FallenAtheneAdvisor.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/FallenAtheneAdvisor.csv', index = None, header=True)

########################

#### Ron's Request For 2021 Planning

#################################

### Nationwide

Submit2019_2020 = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/Submit2019_2020.csv',encoding= 'iso-8859-1')

Submit2019_2020.columns = Submit2019_2020.columns.str.replace(' ', '')

Submit2019_2020.columns = Submit2019_2020.columns.str.lstrip()

Submit2019_2020.columns = Submit2019_2020.columns.str.rstrip()

Submit2019_2020.columns.str.strip()

con = sqlite3.connect("Submit2019_2020.db")

Submit2019_2020.to_sql("Submit2019_2020", con, if_exists='replace')

Submit2019_2020.info()


SubmitMar_Dec2020 = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/SubmitMar_Dec2020.csv',encoding= 'iso-8859-1')

SubmitMar_Dec2020.columns = SubmitMar_Dec2020.columns.str.replace(' ', '')

SubmitMar_Dec2020.columns = SubmitMar_Dec2020.columns.str.lstrip()

SubmitMar_Dec2020.columns = SubmitMar_Dec2020.columns.str.rstrip()

SubmitMar_Dec2020.columns = SubmitMar_Dec2020.columns.str.strip()

con = sqlite3.connect("SubmitMar_Dec2020.db")

SubmitMar_Dec2020.to_sql("SubmitMar_Dec2020", con, if_exists='replace')

SubmitMar_Dec2020.info()

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM Submit2019_2020 group by AdvisorContactIDText, AdvisorName;"""
      
Submit2019_2020GrBy =  pysqldf(q3)  

con = sqlite3.connect("Submit2019_2020GrBy.db")

Submit2019_2020GrBy.to_sql("Submit2019_2020GrBy", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitMar_Dec2020 group by  AdvisorContactIDText, AdvisorName;"""
      
SubmitMar_Dec2020GrBy =  pysqldf(q3)  

con = sqlite3.connect("SubmitMar_Dec2020GrBy.db")

SubmitMar_Dec2020GrBy.to_sql("SubmitMar_Dec2020GrBy", con, if_exists='replace')

vm1 = """SELECT a.* FROM Submit2019_2020GrBy a INNER JOIN SubmitMar_Dec2020GrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

Combet2019Q12020_Rem2020 =  pysqldf(vm1)

con = sqlite3.connect("Combet2019Q12020_Rem2020.db")

Combet2019Q12020_Rem2020.to_sql("Combet2019Q12020_Rem2020", con, if_exists='replace')

p2  = """SELECT * FROM Submit2019_2020GrBy WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Combet2019Q12020_Rem2020);"""
        
FallenNWAdvisorfrom2019Q12020 =  pysqldf(p2) 

con = sqlite3.connect("FallenNWAdvisorfrom2019Q12020.db")

FallenNWAdvisorfrom2019Q12020.to_sql("FallenNWAdvisorfrom2019Q12020", con, if_exists='replace')

Out4 = FallenNWAdvisor.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/FallenNWAdvisor.csv', index = None, header=True)

######

NWAppointedAdv = pd.read_csv('C:/Users/test/Documents/NWFallenAngel2021planning/EmailSegmentNWApp.csv',encoding= 'iso-8859-1')

NWAppointedAdv.columns = NWAppointedAdv.columns.str.replace(' ', '')

NWAppointedAdv.columns = NWAppointedAdv.columns.str.lstrip()

NWAppointedAdv.columns = NWAppointedAdv.columns.str.rstrip()

NWAppointedAdv.columns = NWAppointedAdv.columns.str.strip()

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table

con = sqlite3.connect("NWAppointedAdv.db")

NWAppointedAdv.to_sql("NWAppointedAdv", con, if_exists='replace')

qqq2  = """SELECT a.*, b.* FROM FallenNWAdvisorfrom2019Q12020 a INNER JOIN NWAppointedAdv b on a.AdvisorContactIDText = b.ContactID18 ;"""
        
NWFinalFallenAngelsApp =  pysqldf(qqq2) 

Out4 = NWFinalFallenAngelsApp.to_csv (r'C:\Users\test\Documents\NWFallenAngel2021planning\NWFinalFallenAngelsApp.csv', index = None, header=True)

##### 2020 Committed and Latent Committed


SubmitRankOrdering = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrdering.csv',encoding= 'iso-8859-1')

SubmitRankOrdering.columns = SubmitRankOrdering.columns.str.replace(' ', '')

SubmitRankOrdering.columns = SubmitRankOrdering.columns.str.lstrip()

SubmitRankOrdering.columns = SubmitRankOrdering.columns.str.rstrip()

SubmitRankOrdering.columns = SubmitRankOrdering.columns.str.strip()

con = sqlite3.connect("SubmitRankOrdering.db")

SubmitRankOrdering.to_sql("SubmitRankOrdering", con, if_exists='replace')

SubmitRankOrdering.info()

Submit2020 = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/Submit2020.csv',encoding= 'iso-8859-1')

Submit2020.columns = Submit2020.columns.str.replace(' ', '')

Submit2020.columns = Submit2020.columns.str.lstrip()

Submit2020.columns = Submit2020.columns.str.rstrip()

Submit2020.columns = Submit2020.columns.str.strip()

con = sqlite3.connect("Submit2020.db")

Submit2020.to_sql("Submit2020", con, if_exists='replace')

Submit2020.info()

qqq2  = """SELECT a.*, b.AdvisorName FROM SubmitRankOrdering a INNER JOIN Submit2020 b on a.AdvisorContactIDText = b.AdvisorContactIDText ;"""
        
SubmitRankOrderingDataAppend =  pysqldf(qqq2) 

Out4 =SubmitRankOrderingDataAppend.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrderingDataAppend.csv', index = None, header=True)

SubmitRankOrderingDataAppend1 = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrderingDataAppend1.csv',encoding= 'iso-8859-1')


Submit2020FromDW = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/Submit2020FromDW.csv',encoding= 'iso-8859-1')

Submit2020FromDW.columns = Submit2020FromDW.columns.str.replace(' ', '')

Submit2020FromDW.columns = Submit2020FromDW.columns.str.lstrip()

Submit2020FromDW.columns = Submit2020FromDW.columns.str.rstrip()

Submit2020FromDW.columns = Submit2020FromDW.columns.str.strip()

con = sqlite3.connect("Submit2020FromDW.db")

Submit2020FromDW.to_sql("Submit2020FromDW", con, if_exists='replace')

Submit2020FromDW.info()

qqq2  = """SELECT a.*,  b.Gender FROM SubmitRankOrderingDataAppend1 a INNER JOIN Submit2020FromDW b on a.AdvisorContactIDText = b.ContactID1 ;"""
        
SubmitRankOrderingDataAppend2 =  pysqldf(qqq2) 

Out4 =SubmitRankOrderingDataAppend2.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrderingDataAppend2.csv', index = None, header=True)

### Let's bring the ticket distributions

SubmitRankOrderingDataAppend3 = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrderingDataAppend3.csv',encoding= 'iso-8859-1')

TicketsCumRanks2020 = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/TicketsCumRanks2020.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("SubmitRankOrderingDataAppend3.db")

SubmitRankOrderingDataAppend3.to_sql("SubmitRankOrderingDataAppend3", con, if_exists='replace')

con = sqlite3.connect("TicketsCumRanks2020.db")

TicketsCumRanks2020.to_sql("TicketsCumRanks2020", con, if_exists='replace')

TicketsCumRanks2020.info()

qqq2  = """SELECT a.*, b.Tickets, b.Percent, b.CumPercent FROM SubmitRankOrderingDataAppend3 a INNER JOIN TicketsCumRanks2020 b on a.AdvisorContactIDText = b.AdvisorContactIDText ;"""
        
SubmitRankOrderingDataAppend3PlusTickets =  pysqldf(qqq2) 

Out4 =SubmitRankOrderingDataAppend3PlusTickets.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitRankOrderingDataAppend3PlusTickets.csv', index = None, header=True)

#### Check the Fall off producers

TopAdvisors2020Segement_JoeShareFinal = pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/TopAdvisors2020Segement_JoeShareFinal.csv',encoding= 'iso-8859-1')

SubmitReport01012021_19022021= pd.read_csv('C:/Users/test/Documents/JoesRequestFallenAngesl/SubmitReport01012021_19022021.csv',encoding= 'iso-8859-1')

SubmitReport01012021_19022021.columns = SubmitReport01012021_19022021.columns.str.replace(' ', '')

SubmitReport01012021_19022021.columns = SubmitReport01012021_19022021.columns.str.lstrip()

SubmitReport01012021_19022021.columns = SubmitReport01012021_19022021.columns.str.rstrip()

SubmitReport01012021_19022021.columns = SubmitReport01012021_19022021.columns.str.strip()

con = sqlite3.connect("SubmitReport01012021_19022021.db")

SubmitReport01012021_19022021.to_sql("SubmitReport01012021_19022021", con, if_exists='replace')

q3  = """SELECT AdvisorName, AdvisorContactIDText, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt 
      FROM SubmitReport01012021_19022021 group by  AdvisorContactIDText, AdvisorName;"""
      
SubmitReport01012021_19022021GrBy =  pysqldf(q3)  

vm1 = """SELECT a.* FROM TopAdvisors2020Segement_JoeShareFinal a INNER JOIN SubmitReport01012021_19022021GrBy b on ((a.AdvisorContactIDText =b.AdvisorContactIDText));"""      

CommonAdv2020_2021 =  pysqldf(vm1)

con = sqlite3.connect("CommonAdv2020_2021.db")

CommonAdv2020_2021.to_sql("CommonAdv2020_2021", con, if_exists='replace')

p2  = """SELECT * FROM TopAdvisors2020Segement_JoeShareFinal WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM CommonAdv2020_2021);"""
        
FallenAngels2021 =  pysqldf(p2) 

Out4 =FallenAngels2021.to_csv(r'C:/Users/test/Documents/JoesRequestFallenAngesl/FallenAngels2021.csv', index = None, header=True)

### Athene P
