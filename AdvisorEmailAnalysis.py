
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
import sklearn
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

### Advisor Data
AdvEmail_DF = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/AdvisorDataDateProcessed.csv',encoding= 'iso-8859-1')

AdvEmail_DF['SendDate']=pd.to_datetime(AdvEmail_DF['SendDate'])

AdvEmail_DF['OpenDate']=pd.to_datetime(AdvEmail_DF['OpenDate'])

AdvEmail_DF['ClickDate']=pd.to_datetime(AdvEmail_DF['ClickDate'])

### Let's create the rank to make sure that there is a rank for the name based on the email send

### This is similar to SQL equivalent of rank() and partition by

AdvEmail_DF['rank_Email_Send_date'] = AdvEmail_DF.groupby('Name')['EmailID'].rank(method='first')

### Quick validation if you do this with EmailID and Name and Export that then do a pivot with Name along with count of 'rank_Email_Send_date' column and  it should match the total # of records 463,454

export_csv = AdvEmail_DF.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\VVV.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

AdvEmail_DF.info()

con = sqlite3.connect("AdvEmail_DF.db")

AdvEmail_DF.to_sql("AdvEmail_DF", con, if_exists='replace')

Adv_SendbyName = pd.read_sql("SELECT distinct(Name), sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged FROM AdvEmail_DF group by Name",con)

AdvEmailSummary = pd.read_sql("SELECT distinct(Name), SendDate, rank_Email_Send_date, sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged  FROM AdvEmail_DF group by Name",con)


##AdvEmailSummar2 = pd.read_sql("SELECT t4.ID, t4.AccountNumber, t4.AcDate, (SELECT TOP 1 AcDate FROM t4 b WHERE b.AccountNumber=t4.AccountNumber And b.AcDate>t4.AcDate ORDER BY AcDate DESC, ID) AS NextDate, [NextDate]-[AcDate] AS Diff FROM t4 ORDER BY t4.AcDate);

AvdEmailSummaryNonUni = pd.read_sql("SELECT (Name), SendDate, sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged  FROM AdvEmail_DF group by Name",con)


# AdvEmailSummary_2019 = pd.read_sql("SELECT distinct(Name), SendDate, sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged  FROM AdvEmail_DF where SendMonth_Year >='2019-01'  group by Name",con)

Adv_SendbyName = pd.read_sql("SELECT * FROM AdvEmail_DF group by Name",con)

con = sqlite3.connect("AdvEmailSummary.db")

AdvEmailSummary.to_sql("AdvEmailSummary", con, if_exists='replace')

Adv_SendbyName1 = pd.read_sql("SELECT distinct(Name), sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged FROM AdvEmail_DF group by Name",con)

Adv_SendbyName1['EnagagementRank'] = Adv_SendbyName1['TotalEngaged']/Adv_SendbyName1['TotalSend']

AdvEmail_DF.info()

con = sqlite3.connect("Adv_SendbyName1.db")

Adv_SendbyName1.to_sql("Adv_SendbyName1", con, if_exists='replace')

Adv_SendbyName10orMore = pd.read_sql("SELECT * FROM Adv_SendbyName1 where TotalSend >= 10",con)

export_csv = Adv_SendbyName10orMore.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\Adv_SendbyName10orMore.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

AdvEmail_DF.info()

SendandClick = pd.read_sql("SELECT * FROM Adv_SendbyName1 where TotalSend >= 10",con)

### Consecutive Engagement Making sure Open Date and Click Date not NULL

Engage  = pd.read_sql("SELECT * FROM AdvEmail_DF where ((julianday(SendDate)-julianday(OpenDate) < 15) or (julianday(SendDate)-julianday(ClickDate) < 15) and ((OpenDate !='naT') or (ClickDate !='naT'))) group by SubscriberKey ", con);


### Advisor with 100% Engagement

Avd_Cont_Engagement = pd.read_sql("SELECT * FROM AdvEmail_DF where ((julianday(SendDate)-julianday(OpenDate) < 15) or (julianday(SendDate)-julianday(ClickDate) < 15) and ((OpenDate !='naT') or (ClickDate !='naT'))) group by Name, EmailSubject ", con);

export_csv = Avd_Cont_Engagement.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\Avd_Cont_Engagement.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### SQl to find continuous vs. Sporadic Engagement 
### This can be found out based on the existance in the rank values from the col we created



eng_100per = pd.read_sql("SELECT count(distinct(Name)), rank_Email_Send_date FROM Avd_Cont_Engagement where (rank_Email_Send_date=1 and rank_Email_Send_date!= 2) group by Name", con);

eng_100per_12 = pd.read_sql("SELECT count(distinct(Name)), rank_Email_Send_date FROM Avd_Cont_Engagement where (rank_Email_Send_date= 2 and rank_Email_Send_date=2) group by Name", con);

eng_100per_123 = pd.read_sql("SELECT count(distinct(Name)), rank_Email_Send_date FROM Avd_Cont_Engagement where ((rank_Email_Send_date=1) and (rank_Email_Send_date= 2) and (rank_Email_Send_date!= 3)) group by Name", con);

##############

AdvEmailSummary.info()

con = sqlite3.connect("AdvEmailSummary.db")

AdvEmailSummary.to_sql("AdvEmailSummary", con, if_exists='replace')


conunt1ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date, SendDate, Name FROM AdvEmailSummary where rank_Email_Send_date= 1 and TotalEngaged=1 group by Name,SendDate", con);

conunt2ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date, SendDate, Name FROM AdvEmailSummary where (rank_Email_Send_date= 2 and TotalEngaged= 2) group by Name,SendDate", con);

conunt3ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date,SendDate, Name FROM AdvEmailSummary where (rank_Email_Send_date= 3 and TotalEngaged= 3) group by Name,SendDate", con);

conunt4ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date, SendDate,Name FROM AdvEmailSummary where (rank_Email_Send_date= 4 and TotalEngaged= 4) group by Name,SendDate", con);

conunt5ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date,SendDate, Name FROM AdvEmailSummary where (rank_Email_Send_date= 5 and TotalEngaged= 5) group by Name,SendDate", con);

conunt6ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date,SendDate, Name FROM AdvEmailSummary where (rank_Email_Send_date= 6 and TotalEngaged= 6) group by Name,SendDate", con);

conunt7ConEmails = pd.read_sql("SELECT TotalEngaged, rank_Email_Send_date,SendDate, Name FROM AdvEmailSummary where (rank_Email_Send_date= 7 and TotalEngaged= 7) group by Name,SendDate", con);


df1= Avd_Cont_Engagement[['Name', 'rank_Email_Send_date']]

df1['counts'] =1

cs =df1.groupby('rank_Email_Send_date')['counts'].cumsum()
# set series name
cs.name = 'Occ_number'

del df1['counts']
df2= df1.join(cs)

### Lets look into Submit Data

### Submit Data
Submit_DF= pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/Submits05142018-11052019.csv',encoding= 'iso-8859-1')

Submit_DF.info()

Submit_DF['SubmitDate'] = pd.to_datetime(Submit_DF['SubmitDate'])

Submit_DF.info()

con = sqlite3.connect("Submit_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit_DF.to_sql("Submit_DF", con, if_exists='replace')

SubmitSummaryAdv = pd.read_sql("SELECT AdvisorName, AdvisorContactIDText, Account_AccountName, SubmitDate, sum(SubmitDate) as SubmitCount, SubmitAmount FROM Submit_DF group by AdvisorName, Account_AccountName",con)


q  = """SELECT * FROM AdvEmailSummary a
        JOIN SubmitSummaryAdv b on a.Name = b.AdvisorName where ((julianday(b.SubmitDate)-julianday(a.SendDate) >= 0) and (julianday(b.SubmitDate)-julianday(a.SendDate) < 366));"""
        
Adv_Sub_Email =  pysqldf(q)  

## b = """SELECT * FROM AdvEmailSummary a
##        JOIN SubmitSummaryAdv b on a.Name = b.AdvisorName where b.SubmitDate=a.SendDate;"""
        
###Adv_Sub_Email1 =  pysqldf(b) 

### Lets look into Illustration Data

### Illustration data

Illu_DF= pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/Illustrationdata.csv',encoding= 'iso-8859-1')

Illu_DF.info()

Illu_DF['PreparationDate'] = pd.to_datetime(Illu_DF['PreparationDate'])

con = sqlite3.connect("Illu_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Illu_DF.to_sql("Illu_DF", con, if_exists='replace')

AdvIllus_Count = pd.read_sql("SELECT PreparationDate, count(PreparationDate) as Illus_Count, Name, PreparedBy, Imo, Role from Illu_DF where ((PreparationDate >= '2018-05-08') and (PreparationDate >= '2019-10-15') and Role='Agent') group by Name",con)

q2  = """SELECT * FROM AdvEmailSummary c
        JOIN AdvIllus_Count d on (c.Name = d.Name) where ((julianday(d.PreparationDate)-julianday(c.SendDate) >= 0) and (julianday(d.PreparationDate)-julianday(c.SendDate) < 366));"""
        
Adv_Illus_Email =  pysqldf(q2) 

l3  = """SELECT * FROM AdvEmailSummary c
        INNER JOIN AdvIllus_Count d on (c.Name = d.Name);"""
        
Adv_Illus_Email1 =  pysqldf(l3) 

### Lets' pull out the data-sets for correlation

## Submits

Adv_Sub_Email.info()

DF_Adv_Sub_Email = Adv_Sub_Email[['TotalSend','TotalOpen','TotalClicks','TotalEngaged', 'SubmitAmount']]

myBasicCorr = DF_Adv_Sub_Email.corr()
sns.heatmap(myBasicCorr, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')

## Illustrations

Adv_Illus_Email.info()

DF_Adv_Illus_Email = Adv_Illus_Email[['TotalSend','TotalOpen','TotalClicks','TotalEngaged','Illus_Count']]

myBasicCorr1 = DF_Adv_Illus_Email.corr()
sns.heatmap(myBasicCorr1, annot = True)

###

 
q9  = """SELECT * FROM Adv_Illus_Email e
        JOIN SubmitSummaryAdv f on (e.Name = f.AdvisorName);"""

Adv_Illus_Email_Sub =  pysqldf(q9) 


### Submit Rank Data
SubmitRank_DF = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/SubmitRank.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("SubmitRank_DF.db")

SubmitRank_DF.to_sql("SubmitRank_DF", con, if_exists='replace')

vm  = """SELECT * FROM SubmitRank_DF d 
        LEFT JOIN Adv_SendbyName1 c on (d.AdvisorName= c.Name);"""
        
Adv_EngagRank_SubRank =  pysqldf(vm) 

Adv_EngagRank_SubRank.info()

DF11= Adv_EngagRank_SubRank[['Name', 'TotalSend','TotalOpen', 'TotalClicks', 'TotalEngaged','SubmitCount',' SubmitAmount']]

DF11.info()

myBasicCorr2 = DF11.corr()
sns.heatmap(myBasicCorr2, annot = True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')



export_csv = Adv_EngagRank_SubRank.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\Adv_EngagRank_SubRank.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Let's take this Submit Data and Validate that Against

SubmitRank_DF_Jason = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/MissingEmailHistoryFromSubmits.csv',encoding= 'iso-8859-1')

AnnexusLOPMailing_DF = pd.read_csv(r'C:\Users\test\Documents\Schillar\AnnexusLOPMailing.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("SubmitRank_DF_Jason.db")

SubmitRank_DF_Jason.to_sql("SubmitRank_DF_Jason", con, if_exists='replace')


con = sqlite3.connect("AnnexusLOPMailing_DF.db")

AnnexusLOPMailing_DF.to_sql("AnnexusLOPMailing_DF", con, if_exists='replace')

l2  = """SELECT * FROM SubmitRank_DF_Jason c
        INNER JOIN AnnexusLOPMailing_DF d on ( c.AdvisorName= d.AgentName);"""
        
Common =  pysqldf(l2) 

con = sqlite3.connect("Common.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Common.to_sql("Common", con, if_exists='replace')

w2  = """SELECT * FROM SubmitRank_DF_Jason WHERE AdvisorName NOT IN (SELECT AdvisorName FROM Common);"""
        
Remaining =  pysqldf(w2) 

export_csv = Remaining.to_csv (r'C:\Users\test\Documents\Schillar\Remaining.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
       

export_csv = Remaining.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\Remaining.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

con = sqlite3.connect("Remaining.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Remaining.to_sql("Remaining", con, if_exists='replace')

### Let's see if there is any overlap with Marketer Data

Email_MarketerDF = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/ListSupression/IDCDeliverabilityResearch20191014b-Master.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Email_MarketerDF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Email_MarketerDF.to_sql("Email_MarketerDF", con, if_exists='replace')


Mar_Sum_EmailSub = pd.read_sql("SELECT distinct(Name), sum(Sent) as TotalSent, sum(EngagedFinal) as TotalEngaged FROM Email_MarketerDF group by Name",con)



d2  = """SELECT * FROM Remaining c
        INNER JOIN Mar_Sum_EmailSub d on ( c.AdvisorName= d.Name);"""
        
CommonMatch_Mar_Adv =  pysqldf(d2) 

CommonMatch_Mar_Adv.to_sql("CommonMatch_Mar_Adv", con, if_exists='replace')

w3  = """SELECT * FROM Remaining WHERE AdvisorName NOT IN (SELECT AdvisorName FROM CommonMatch_Mar_Adv);"""
        
Remaining1 =  pysqldf(w3) 

### Let's take the new dataset 

Email_DDF1 = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/AdvisorData01022020/AdvisorDataNewDumpProcessed.csv',encoding='iso-8859-1')


con = sqlite3.connect("Email_DDF1.db")

Email_DDF1.to_sql("Email_DDF1", con, if_exists='replace')


Email_DDF1Summary = pd.read_sql("SELECT distinct(Name), SendDate, sum(Sent) as TotalSend, sum(OpenExist) as TotalOpen, sum(ClickExist) as TotalClicks, sum(EngagedFinal) as TotalEngaged, SubscriberKey  FROM Email_DDF1 group by Name",con)


     
       
Submit_DFNew= pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/Submits05142018-11052019.csv',encoding= 'iso-8859-1')

Submit_DFNew.info()

Submit_DFNew['SubmitDate'] = pd.to_datetime(Submit_DFNew['SubmitDate'])

Submit_DFNew.info()

con = sqlite3.connect("Submit_DFNew.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Submit_DFNew.to_sql("Submit_DFNew", con, if_exists='replace')

SubmitSummaryNewAdv = pd.read_sql("SELECT distinct(AdvisorName1), AdvisorContactIDText, count(SubmitDate) as SubmitCount, sum(SubmitAmount) as SubmitTotal FROM Submit_DFNew group by AdvisorName1",con)


vm1  = """SELECT * FROM SubmitSummaryNewAdv c
        LEFT JOIN Email_DDF1Summary d on (c.AdvisorName1 = d.Name);"""

               
NewMissing_SubID =  pysqldf(vm1) 

export_csv = NewMissing_SubID.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\NewMissing_SubID.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

AdvEmail_DF.info()

Domain= AdvEmail_DF['Email'].str.split('@', expand=True)

##Email_DF['Domain'] = Domain

##domain = re.search("@[\w.]+", str)
## print(Domain)

### Note dataframe Domain has two columns separateed by key and values; hence shape is 2

### Pull out the column 1 get the domain

Domain1=Domain[[1]]


### Append the domain with the appropriate dataset
AdvEmail_DF['Domain']=Domain1

con = sqlite3.connect("AdvEmail_DF.db")

AdvEmail_DF.to_sql("AdvEmail_DF", con, if_exists='replace')

AdvEmail_DF.info()

DomainCount2 = pd.read_sql("SELECT count(Domain) as DomainCount, Domain FROM AdvEmail_DF",con)

##export_csv = DomainCount.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\ListSupression\\CorrelationData\DomainCount.csv', index = None, header=True) 


Domain= AdvEmail_DF['Email'].str.split('@', expand=True)

##Email_DF['Domain'] = Domain

##domain = re.search("@[\w.]+", str)
## print(Domain)

### Note dataframe Domain has two columns separateed by key and values; hence shape is 2

### Pull out the column 1 get the domain

Domain1=Domain[[1]]

### Append the domain with the appropriate dataset
AdvEmail_DF['Domain']=Domain1

con = sqlite3.connect("AdvEmail_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AdvEmail_DF.to_sql("AdvEmail_DF", con, if_exists='replace')

export_csv = AdvEmail_DF.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\AdvEmail_DFwithDomain.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

AdvEmail_DF.info()

DomainCount = pd.read_sql("SELECT count(Domain) as DomainCount, Domain, sum(sent) as TotalSent, sum(EngagedFinal) as TotalEngaged  FROM AdvEmail_DF group by Domain order by DomainCount desc",con)

export_csv = DomainCount.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\DomainCount.csv', index = None, header=True) 

AdvisorEmailDomain_DF = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/RawFiles/DomainCount.csv',encoding= 'iso-8859-1')

MarketerEmailDomain_DF = pd.read_csv('C:/Users/test/Documents/AdvisorData-12192019/MarketerDomains.csv',encoding= 'iso-8859-1')

AdvisorEmailDomain_DF.info()

MarketerEmailDomain_DF.info()

con = sqlite3.connect("AdvisorEmailDomain_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
AdvisorEmailDomain_DF.to_sql("AdvisorEmailDomain_DF", con, if_exists='replace')

con = sqlite3.connect("MarketerEmailDomain_DF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
MarketerEmailDomain_DF.to_sql("MarketerEmailDomain_DF", con, if_exists='replace')

j1  = """SELECT * FROM AdvisorEmailDomain_DF e
        JOIN MarketerEmailDomain_DF f on (e.AdvisorsDomain  = f.MarketerDomain);"""

DomainMatch_Mar_Adv =  pysqldf(j1) 

export_csv = DomainMatch_Mar_Adv.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\DomainMatch_Mar_Adv.csv', index = None, header=True) 

con = sqlite3.connect("DomainMatch_Mar_Adv.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
DomainMatch_Mar_Adv.to_sql("DomainMatch_Mar_Adv", con, if_exists='replace')

DomainMatch_Mar_Adv.info()

w2  = """SELECT * FROM MarketerEmailDomain_DF WHERE MarketerDomain NOT IN (SELECT MarketerDomain FROM DomainMatch_Mar_Adv);"""

NameNotExist =  pysqldf(w2) 

export_csv = NameNotExist.to_csv (r'C:\Users\test\Documents\AdvisorData-12192019\RawFiles\NameNotExist.csv', index = None, header=True) 





