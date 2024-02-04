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

## Dataset 1 : Bring the Insurance Discovery Data

#### This part was not used what was finally used. 

InsuranceDiscoveryDF= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/InsuranceDataFeed-Jan2020.csv',encoding= 'iso-8859-1')

InsuranceDiscoveryDF.info()

Annaexu_IDMatch= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Annaexu_IDMatchUpdated.csv',encoding= 'iso-8859-1')

Annaexu_IDMatch.info()

con = sqlite3.connect("InsuranceDiscoveryDF.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
InsuranceDiscoveryDF.to_sql("InsuranceDiscoveryDF", con, if_exists='replace')

con = sqlite3.connect("Annaexu_IDMatch.db")

q2  = """SELECT FirstName, LastName, NPN, CarrierName1, CarrierName2, CarrierName3, CarrierName4, CarrierName5 from InsuranceDiscoveryDF;"""
      
InsuranceCarrier =  pysqldf(q2)  

con = sqlite3.connect("InsuranceCarrier.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
InsuranceCarrier.to_sql("InsuranceCarrier", con, if_exists='replace')

con = sqlite3.connect("InsuranceCarrier.db")

#q2  = """SELECT FirstName, LastName, NPN FROM Annaexu_IDMatch;"""
      
#Annaexu_Sub=  pysqldf(q2)  

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
Annaexu_IDMatch.to_sql("Annaexu_IDMatch", con, if_exists='replace')

##Annaexu_IDMatch_NoNull  = """SELECT * FROM Annaexu_IDMatch where b.NPN__c != 'nan' ;"""
      
###Annex =  pysqldf(Annaexu_IDMatch_NoNull)  

q5 = """SELECT *
      FROM InsuranceCarrier a Inner JOIN Annaexu_IDMatch b on (a.NPN=b.NPN__c);"""
      
Dis_Annex_Car =  pysqldf(q5) 

con = sqlite3.connect("Dis_Annex_Car.db")

#q2  = """SELECT FirstName, LastName, NPN FROM Annaexu_IDMatch;"""
      
#Annaexu_Sub=  pysqldf(q2)  

# Advisors that connected with Allianze and Athene and 
Dis_Annex_Car.to_sql("Dis_Annex_Car", con, if_exists='replace')

q8  = """SELECT * from Dis_Annex_Car where (CarrierName1 like '%Allianz%') or (CarrierName2 like '%Allianz%') or (CarrierName3 like '%Allianz%') or (CarrierName4 like '%Allianz%') or (CarrierName5 like '%Allianz%');"""


##q7  = """SELECT * from Dis_Annex_Car where (CarrierName1 like '%Athene%') or (CarrierName2 like '%Athene%') or (CarrierName3 like '%Athene%') or (CarrierName4 like '%Athene%') or (CarrierName5 like '%Athene%') or (CarrierName1 like '%Allianz%')
 ##     or (CarrierName2 like '%Allianz%') or (CarrierName3 like '%Allianz%') or (CarrierName4 like '%Allianz%') or (CarrierName5 like '%Allianz%');"""

Dis_Annex_Allianze =  pysqldf(q8) 

### This contains a lot of duplicates

Dis_Annex_Allianze = Dis_Annex_Allianze.sort_values("NPN", ascending=False)

Dis_Annex_Allianze2 = Dis_Annex_Allianze.drop_duplicates(["NPN"])

### Last 48Month producer is going to 02/15/2015.. It is mislabeled as 48Month it is more than 48 months

######
#################
#######################################
####################################################

### True Start Begins from here:  What I shared with J and Tony

### Also, in this case, we used Submit Data, however, these submitted advisors' appointments were not verified. 
### As they submitted, it is assumed that the appointments were in place

Submits48M= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Last48monthsProducers.csv',encoding= 'iso-8859-1')

Submits48M.info()

Submits12M= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Last12monthsProducers1.csv',encoding= 'iso-8859-1')

Submits12M.info()

##Submits9M= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Submitslast9months.csv',encoding= 'iso-8859-1')

## Submits9M.info()

con = sqlite3.connect("Submits48M.db")

#q2  = """SELECT FirstName, LastName, NPN FROM Annaexu_IDMatch;"""
      
#Annaexu_Sub=  pysqldf(q2)  

# Advisors that connected with Allianze and Athene and 
Submits48M.to_sql("Submits48M", con, if_exists='replace')

con = sqlite3.connect("Submits12M.db")

Submits12M.to_sql("Submits12M", con, if_exists='replace')

#q2  = """SELECT FirstName, LastName, NPN FROM Annaexu_IDMatch;"""
      
#Annaexu_Sub=  pysqldf(q2)  

# Advisors that connected with Allianze and Athene and 
Submits12M.to_sql("Submits12M", con, if_exists='replace')

v2  = """SELECT AdvisorContactIDText, AdvisorContactNPN, AdvisorName, AdvisorContactEmail, AdvisorContactMailingAddress, AdvisorContactEMail1, max(SubmitDate) as LastDate_Prior12Month, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt FROM Submits48M group by AdvisorContactIDText;"""
      
GrBy48M =  pysqldf(v2)  


v3  = """SELECT AdvisorContactIDText, AdvisorContactNPN, AdvisorName, AdvisorContactEmail, AdvisorContactMailingAddress, AdvisorContactEMail1, max(SubmitDate) as LastDate_12Months, count(SubmitDate) as SubmitCnt, sum(SubmitAmount) as SubmitAmt FROM Submits12M group by AdvisorContactIDText;"""
      
GrBy12M =  pysqldf(v3)  

con = sqlite3.connect("GrBy12M.db")

GrBy12M.to_sql("GrBy12M", con, if_exists='replace')

con = sqlite3.connect("GrBy48M.db")

GrBy48M.to_sql("GrBy48M", con, if_exists='replace')

v6 = """SELECT a.AdvisorContactIDText, a.AdvisorContactEmail, a.AdvisorContactMailingAddress, a.AdvisorContactEMail1, a.AdvisorContactNPN, a.AdvisorName, a.LastDate_Prior12Month, a.SubmitCnt as SubmitCnt_24m, a.SubmitAmt as SubmitAmt_24M, b.LastDate_12Months, b.SubmitCnt as SubmitCnt_12M, b.SubmitAmt as SubmitAmt_12M
      FROM GrBy48M a INNER JOIN GrBy12M b on ((a.AdvisorContactIDText= b.AdvisorContactIDText));"""
      
Comm48M_12M =  pysqldf(v6) 

Comm48M_12M.info()

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table

con = sqlite3.connect("Comm48M_12M.db")

Comm48M_12M.to_sql("Comm48M_12M", con, if_exists='replace')

w2  = """SELECT * FROM GrBy48M WHERE AdvisorContactIDText NOT IN (SELECT AdvisorContactIDText FROM Comm48M_12M);"""
        
Remaining1 =  pysqldf(w2)  

##export_csv = Remaining1.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Campaign\Remaining1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

##Remaining= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Remaining.csv',encoding= 'iso-8859-1')

### Lets' Try to match between 3 Months Fallen Angles and Dis_Annex_Athene_Allianze

# Advisors that connected with Allianze and Athene and 
##Remaining.to_sql("Remaining1", con, if_exists='replace')

##con = sqlite3.connect("Remaining.db")

##Remaining.info()

Allianze= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AllianzAdvisorsNotTerminated.csv',encoding= 'iso-8859-1')

Allianze.to_sql("Allianze", con, if_exists='replace')

con = sqlite3.connect("Allianze.db")

Check1 = GrBy48M[~GrBy48M['AdvisorContactIDText'].isin(Comm48M_12M['AdvisorContactIDText'])]

##Dis_Annex_Allianze2.to_sql("Dis_Annex_Allianze2", con, if_exists='replace')

###con = sqlite3.connect("Dis_Annex_Allianze2.db")

Remaining1.to_sql("Remaining1", con, if_exists='replace')

con = sqlite3.connect("Remaining1.db")

v9 = """SELECT *
      FROM Remaining1 a Inner JOIN Allianze b on (a.AdvisorContactNPN=b.NPN);"""
      
###  Match12Month is a mislebel showes the Annexus FallenAngels Appointed with Allianz  
Match12Month =  pysqldf(v9)

export_csv = Match12Month.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Match12Month.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### End of 2019 Analysis

Annex_AppData= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Annex_AppData.csv',encoding= 'iso-8859-1')

#### List for Non-producers....
######################
######################
### Start here


Annexus_AdvisorDF= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Annex_AllAdvisorsData.csv',encoding= 'iso-8859-1')




Annexus_AdvisorDF.to_sql("Annexus_AdvisorDF", con, if_exists='replace')

Annexus_AdvisorDF.info()



con = sqlite3.connect("Annexus_AdvisorDF.db")

Annexus_AdvisorDF.to_sql("Annexus_AdvisorDF", con, if_exists='replace')

Annex_AndrewH =  pd.read_sql("SELECT * FROM Annexus_AdvisorDF where FirstName='Andrew' and LastName='Hauch'",con)

Uni_Annexus_AdvisorDF = pd.read_sql("SELECT distinct(SFContactId), AdvisorKey, FirstName, LastName, FullName, DeletedFlag, SFEmail,SFAlternateEmail FROM Annexus_AdvisorDF",con)

con = sqlite3.connect("Uni_Annexus_AdvisorDF.db")

Uni_Annexus_AdvisorDF.to_sql("Uni_Annexus_AdvisorDF", con, if_exists='replace')

Annex_AndrewH1 =  pd.read_sql("SELECT * FROM Uni_Annexus_AdvisorDF where FirstName='Andrew' and LastName='Hauch'",con)

GrBy48M.info()

w1 = """SELECT a.*, b.AdvisorContactIDText
      FROM Uni_Annexus_AdvisorDF a INNER JOIN GrBy48M b on (a.SFContactId = b.AdvisorContactIDText);"""
      
CommonMatch=  pysqldf(w1) 

con = sqlite3.connect("CommonMatch.db")

CommonMatch.to_sql("CommonMatch", con, if_exists='replace')

w9  = """SELECT * FROM Uni_Annexus_AdvisorDF WHERE SFContactId NOT IN (SELECT SFContactId FROM CommonMatch);"""
        
Annex_NonProducer =  pysqldf(w9)  

Annex_NonProducer.info()

con = sqlite3.connect("Annex_NonProducer.db")

Annex_NonProducer.to_sql("Annex_NonProducer", con, if_exists='replace')

Annex_AndrewH2 =  pd.read_sql("SELECT * FROM Annex_NonProducer where FirstName='Andrew' and LastName='Hauch'",con)

#### There is no NPN to match with the Allianze

### Let's look into the connecttion with Allianz

Annaexu_IDMatch= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Annaexu_IDMatch.csv',encoding= 'iso-8859-1')


### This IDMatch has duplicate rows ..i.e., 2 Andrew Hauch exists......

con = sqlite3.connect("Annaexu_IDMatch.db")

Annaexu_IDMatch.to_sql("Annaexu_IDMatch", con, if_exists='replace')

Annaexu_IDMatch.info()

Annex_AndrewH3 =  pd.read_sql("SELECT * FROM Annaexu_IDMatch where FirstName='Andrew' and LastName='Hauch'",con)


### Let's join ContactID and ID

w16 = """SELECT a.*, b.NPN__c
      FROM Annex_NonProducer a INNER JOIN Annaexu_IDMatch b on (a.SFContactId = b.Id);"""
      
Annex_NonProducer_Up=  pysqldf(w16) 

Annex_NonProducer_Up.info()

con = sqlite3.connect("Annex_NonProducer_Up.db")

Annex_NonProducer_Up.to_sql("Annex_NonProducer_Up", con, if_exists='replace')

Annex_AndrewH4 =  pd.read_sql("SELECT * FROM Annex_NonProducer_Up where FirstName='Andrew' and LastName='Hauch'",con)

Annex_NonProducer_Final= pd.read_sql("select distinct(SFContactId), AdvisorKey, FirstName, LastName, FullName, DeletedFlag, SFEmail,SFAlternateEmail, NPN__c from Annex_NonProducer_Up ",con)

con = sqlite3.connect("Annex_NonProducer_Final.db")

Annex_NonProducer_Final.to_sql("Annex_NonProducer_Final", con, if_exists='replace')

Annex_AndrewH6 =  pd.read_sql("SELECT * FROM Annex_NonProducer_Final where FirstName='Andrew' and LastName='Hauch'",con)


## Join with Allianz

con = sqlite3.connect("Annex_NonProducer_Final.db")

Annex_NonProducer_Final.to_sql("Annex_NonProducer_Final", con, if_exists='replace')

Allianze.to_sql("Allianze", con, if_exists='replace')

con = sqlite3.connect("Allianze.db")



v29 = """SELECT *
      FROM Annex_NonProducer_Final a Inner JOIN Allianze b on (a.NPN__c=b.NPN);"""
      
###  Match12Month is a mislebel showes the Annexus FallenAngels Appointed with Allianz  
Annex_NonProducer_FAllianz =  pysqldf(v29)

con = sqlite3.connect("Annex_NonProducer_FAllianz.db")

Annex_NonProducer_FAllianz.to_sql("Annex_NonProducer_FAllianz", con, if_exists='replace')

Annex_AndrewH7 =  pd.read_sql("SELECT * FROM Annex_NonProducer_FAllianz where FirstName='Andrew' and LastName='Hauch'",con)

export_csv = Annex_NonProducer_FAllianz.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Annex_NonProducer_FAllianz03022020.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


Annex_NonProducer_Final2= pd.read_sql("select distinct(SFContactId), AdvisorKey, FirstName, LastName, FullName, DeletedFlag, SFEmail,SFAlternateEmail, NPN__c from Annex_NonProducer_FAllianz",con)

Annex_NonProducer_Final3= pd.read_sql("select distinct(AdvisorKey), SFContactId, FirstName, LastName, FullName, DeletedFlag, SFEmail,SFAlternateEmail, NPN__c from Annex_NonProducer_FAllianz",con)

##Annex_NonProducer_Final3= Annex_NonProducer_Final3.sort_values('AdvisorKey', inplace = True) 
  
# dropping ALL duplicte values 
#Annex_NonProducer_Final34= Annex_NonProducer_Final3.drop_duplicates(subset=None, keep="first", inplace=False)
Annex_NonProducer_Final34 = Annex_NonProducer_Final3.drop_duplicates(subset='SFContactId', keep="first")

export_csv = Annex_NonProducer_Final34.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Annex_NonProducer_Final34.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

# displaying data 



export_csv = Annex_NonProducer_FAllianz.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Annex_NonProducer_FAllianz.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#### Bring the missing email from Discovery


MissingEmail= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/NP_MissingEmails03022020.csv',encoding= 'iso-8859-1')

InsuranceDataFeedJan = pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/InsuranceDataFeed-Jan2020.csv',encoding= 'iso-8859-1')

#### Create Data Table

con = sqlite3.connect("MissingEmail.db")

MissingEmail.to_sql("MissingEmail", con, if_exists='replace')

con = sqlite3.connect("InsuranceDataFeedJan.db")

InsuranceDataFeedJan.to_sql("InsuranceDataFeedJan", con, if_exists='replace')

v49 = """SELECT a.*, b.Email_BusinessType, b.Email_PersonalType
      FROM MissingEmail a Inner JOIN InsuranceDataFeedJan b on (a.NPN__c=b.NPN);"""

Annex_NonProducer_EmailApp =  pysqldf(v49)



con = sqlite3.connect("Annex_NonProducer_EmailApp.db")

Annex_NonProducer_EmailApp.to_sql("Annex_NonProducer_EmailApp", con, if_exists='replace')

Annex_AndrewH8 =  pd.read_sql("SELECT * FROM Annex_NonProducer_EmailApp where FirstName='Andrew' and LastName='Hauch'",con)

### Until this point it is correct

BDRepDataFeed20200102_Jason = pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/BDRepDataFeed20200102_Jason.csv',encoding= 'iso-8859-1')

#### Create Data Table

con = sqlite3.connect("BDRepDataFeed20200102_Jason.db")

BDRepDataFeed20200102_Jason.to_sql("BDRepDataFeed20200102_Jason", con, if_exists='replace')

BDRepDataFeed20200102_Jason.info()

v59 = """SELECT a.*, b.Email_BusinessType as Bemail1, b.Email_Business2Type as Bemail2, b.Email_PersonalType as Bemail3
      FROM Annex_NonProducer_EmailApp a LEFT JOIN BDRepDataFeed20200102_Jason b on (a.NPN__c=b.NPN);"""
      
v60 = """SELECT a.*, b.Email_BusinessType as Bemail1, b.Email_Business2Type as Bemail2, b.Email_PersonalType as Bemail3
      FROM Annex_NonProducer_EmailApp a LEFT JOIN BDRepDataFeed20200102_Jason b on (a.NPN__c=b.NPN);"""      

Annex_NonProducer_EmailApp2 =  pysqldf(v59)

con = sqlite3.connect("Annex_NonProducer_EmailApp2.db")

Annex_NonProducer_EmailApp2.to_sql("Annex_NonProducer_EmailApp2", con, if_exists='replace')

Annex_AndrewH9 =  pd.read_sql("SELECT * FROM Annex_NonProducer_EmailApp2 where FirstName='Andrew' and LastName='Hauch'",con)

export_csv = Annex_NonProducer_EmailApp2.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Annex_NonProducer_EmailApp2_03022020.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#### Lets validate the appointment

Appointment= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/Appointment.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Appointment.db")

Appointment.to_sql("Appointment", con, if_exists='replace')

con = sqlite3.connect("Annex_NonProducer_Final34.db")

Annex_NonProducer_Final34.to_sql("Annex_NonProducer_Final34", con, if_exists='replace')

v89 = """SELECT * FROM  Annex_NonProducer_Final34 a LEFT JOIN Appointment b on (a.AdvisorKey=b.AdvisorKey);"""
      
NonProducer_AppMatch =  pysqldf(v89)      

##### Validation between 12MonthMatch and Annex_NonProducer_Final34

con = sqlite3.connect("Annex_NonProducer_Final34.db")

Annex_NonProducer_Final34.to_sql("Annex_NonProducer_Final34", con, if_exists='replace')

con = sqlite3.connect("Match12Month.db")

Match12Month.to_sql("Match12Month", con, if_exists='replace')

Annex_NonProducer_Final34.info()

Match12Month.info()

v98 = """SELECT * FROM  Annex_NonProducer_Final34 a INNER JOIN Match12Month b on (a.SFContactId=b.AdvisorContactIDText);"""
      
NonProducer_Match12month =  pysqldf(v98) 

IMO= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/IMOInfo.csv',encoding= 'iso-8859-1')

IMO.info()

con = sqlite3.connect("IMO.db")

IMO.to_sql("IMO", con, if_exists='replace')

v99 = """SELECT a.*, b.IMOName FROM  Annex_NonProducer_Final34 a INNER JOIN IMO b on (a.SFContactId=b.SFAccountId);"""
      
IMOAppendNonProducer_Match12month =  pysqldf(v99) 


##### Athene active advisors 
#########################################
##########################################################
###########################################################################
#############################################################################################
#########################################################################################################

Athene= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/AtheneAppointedAdvisorsSellingProducts12MBackedoff.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("Athene.db")

Athene.to_sql("Athene", con, if_exists='replace')

Athene.info()

Athen_GrBy =  pd.read_sql("SELECT NPN, ContactID18, FullName, Email, AgentCRD, AgentKey, NWLastSubmitDate, AtheneLastSubmitDate FROM Athene group by AgentKey,NPN, AgentCRD,Email, FullName ",con)

v16 = """SELECT a.*
      FROM Athene a INNER JOIN Allianze  b on (a.NPN=b.NPN);"""      

Athene_Allianz =  pysqldf(v16)

con = sqlite3.connect("Athene_Allianz.db")

Athene_Allianz.to_sql("Athene_Allianz", con, if_exists='replace')


### Let Substract all the 3 lists

## List 1

AlllianzAppFallenAnnexusAdvisorSharedJay_Tony= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/Sub/AlllianzAppFallenAnnexusAdvisorSharedJay_Tony.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AlllianzAppFallenAnnexusAdvisorSharedJay_Tony.db")

AlllianzAppFallenAnnexusAdvisorSharedJay_Tony.to_sql("AlllianzAppFallenAnnexusAdvisorSharedJay_Tony", con, if_exists='replace')


v17 = """SELECT a.*
      FROM Athene_Allianz a INNER JOIN AlllianzAppFallenAnnexusAdvisorSharedJay_Tony b on (a.NPN=b.NPN);"""      

Athene_Allianz_123 =  pysqldf(v17)

con = sqlite3.connect("Athene_Allianz_123.db")

Athene_Allianz_123.to_sql("Athene_Allianz_123", con, if_exists='replace')

##Need to substract from list1

p1  = """SELECT * FROM Athene_Allianz WHERE NPN NOT IN (SELECT NPN FROM Athene_Allianz_123);"""
        
L1 =  pysqldf(p1)  

con = sqlite3.connect("L1.db")

L1.to_sql("L1", con, if_exists='replace')

### List2

Non_producerslist_JasonShared= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/Sub/Non_producerslist_JasonShared.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("Non_producerslist_JasonShared.db")

Non_producerslist_JasonShared.to_sql("Non_producerslist_JasonShared", con, if_exists='replace')

v18 = """SELECT a.*
      FROM L1 a INNER JOIN Non_producerslist_JasonShared b on (a.NPN=b.NPN);"""      

Athene_Allianz_456 =  pysqldf(v18)


################# ***************************************************************#########################
AFinal= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/Sub/AFinal.csv',encoding= 'iso-8859-1')

con = sqlite3.connect("AFinal.db")

AFinal.to_sql("AFinal", con, if_exists='replace')

v19 = """SELECT a.* FROM Athene a INNER JOIN AFinal b on (a.NPN=b.NPN);"""      

Athene_AFinal =  pysqldf(v19)

con = sqlite3.connect("Athene_AFinal.db")

Athene_AFinal.to_sql("Athene_AFinal", con, if_exists='replace')

p2  = """SELECT * FROM Athene WHERE NPN NOT IN (SELECT NPN FROM Athene_AFinal);"""
        
Rem =  pysqldf(p2)  

con = sqlite3.connect("Rem.db")

Rem.to_sql("Rem", con, if_exists='replace')

Allianze.to_sql("Allianze", con, if_exists='replace')

con = sqlite3.connect("Allianze.db")

v28 = """SELECT a.*
      FROM Rem a INNER JOIN Allianze b on (a.NPN=b.NPN);"""      

Rem_Allianz =  pysqldf(v28)

#####################################################################....................................##################

### Allianz advisors not terminated

Allianz1= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Data/AllianzAdvisorsNotTerminated.csv',encoding= 'iso-8859-1')

BDRepPre= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Data/BDRepDataProcessed03042020.csv',encoding= 'iso-8859-1')

RIPRepPre= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Data/RIARepDataProcessed03042020.csv',encoding= 'iso-8859-1')

BDRepPre.info()

RIPRepPre.info()

con = sqlite3.connect("Allianz1.db")

Allianz1.to_sql("Allianz1", con, if_exists='replace')

con = sqlite3.connect("BDRepPre.db")

BDRepPre.to_sql("BDRepPre", con, if_exists='replace')

con = sqlite3.connect("RIPRepPre.db")

RIPRepPre.to_sql("RIPRepPre", con, if_exists='replace')

bb1 = """SELECT a.*, b.RepCRD, b.NPN as NPN1, b.FullName as BDRepFullName, b.BDFirmName, b.EmailBusinessType, b.EmailBusiness2Type, b.EmailPersonalType
      FROM Allianz1 a LEFT JOIN BDRepPre b on (a.NPN=b.NPN);"""      

Allianz_BDRep =  pysqldf(bb1)

RIPRepPre.info()

bb2 = """SELECT a.*, b.RepCRD, b.NPN as NPN1, b.FullName as BDRepFullName, b.RIAFirmName, b.EmailBusinessType, b.EmailBusiness2Type, b.EmailPersonalType
      FROM Allianz1 a LEFT JOIN RIPRepPre b on (a.NPN=b.NPN);"""      

Allianz_RIARep =  pysqldf(bb2)

Allianz_RIARep.info()

Rem.info()

#### Let's try to do an email match with Rem


con = sqlite3.connect("Rem.db")

Rem.to_sql("Rem", con, if_exists='replace')

con = sqlite3.connect("Allianz_RIARep.db")

Allianz_RIARep.to_sql("Allianz_RIARep", con, if_exists='replace')

bb3 = """SELECT a.NPN, a.Email, b.CarrierName, b.CarrierGroup, b.NPN1 as NPN2, a.ContactID18, a.FullName, a.AgentCRD, b.RepCRD, b.BDRepFullName, b.RIAFirmName, b.EmailBusinessType, b.EmailBusiness2Type, b.EmailPersonalType
      FROM Rem a INNER JOIN Allianz_RIARep b on ((a.NPN= b.NPN) or (a.AgentCRD=b.RepCRD) or (a.Email=b.EmailBusinessType) or (a.Email=b.EmailBusiness2Type) or (a.Email=b.EmailPersonalType));"""      

Email_Rem_AlliRIARep =  pysqldf(bb3)


Allianz_BDRep.to_sql("Allianz_BDRep", con, if_exists='replace')

con = sqlite3.connect("Allianz_BDRep.db")

bb4 = """SELECT a.NPN, a.Email, b.CarrierName, b.CarrierGroup, b.NPN as NPN1, a.ContactID18, a.FullName, a.AgentCRD, b.RepCRD, b.BDRepFullName, b.BDFirmName, b.EmailBusinessType, b.EmailBusiness2Type, b.EmailPersonalType
      FROM Rem a INNER JOIN Allianz_BDRep b on ((a.NPN= b.NPN) or (a.AgentCRD=b.RepCRD) or (a.Email=b.EmailBusinessType) or (a.Email=b.EmailBusiness2Type) or (a.Email=b.EmailPersonalType));"""      

Email_Rem_AlliBDRep =  pysqldf(bb4)


Email_Rem_AlliBDRep.to_sql("Email_Rem_AlliBDRep", con, if_exists='replace')

con = sqlite3.connect("Email_Rem_AlliBDRep.db")

Email_Rem_AlliRIARep.to_sql("Email_Rem_AlliRIARep", con, if_exists='replace')

con = sqlite3.connect("Email_Rem_AlliRIARep.db")

bb5 = """SELECT a.*
      FROM Email_Rem_AlliBDRep a INNER JOIN Email_Rem_AlliRIARep b on ((a.NPN= b.NPN) or (a.AgentCRD=b.RepCRD) or (a.Email=b.EmailBusinessType) or (a.Email=b.EmailBusiness2Type) or (a.Email=b.EmailPersonalType));"""      

### Remove the common one
      
Email_BD_RIA =  pysqldf(bb5)

con = sqlite3.connect("Email_BD_RIA.db")

Email_BD_RIA.to_sql("Email_BD_RIA", con, if_exists='replace')



m1  = """SELECT * FROM Email_Rem_AlliBDRep  WHERE NPN NOT IN (SELECT NPN FROM Email_BD_RIA);"""
        
Remmm =  pysqldf(m1) 

export_csv = Email_Rem_AlliRIARep.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Campaign\AtheneAdvisors\JShare\Email_Rem_AlliRIARep.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

export_csv = Email_Rem_AlliBDRep.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Campaign\AtheneAdvisors\JShare\Email_Rem_AlliBDRep.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Let's look into common matches 

Athene_JaasonEmail= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/Athene_JaasonEmail.csv',encoding= 'iso-8859-1')

NW_JasonEmail= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/NW_JasonEmail.csv',encoding= 'iso-8859-1')

Athene_JaasonEmail.to_sql("Athene_JaasonEmail", con, if_exists='replace')

con = sqlite3.connect("Athene_JaasonEmail.db")

NW_JasonEmail.to_sql("NW_JasonEmail", con, if_exists='replace')

con = sqlite3.connect("NW_JasonEmail.db")



Athene_JaasonEmail.info()

NW_JasonEmail.info()

Athene_Submit =  pd.read_sql("SELECT ContactID18, FullName, count(AtheneLastSubmitDate), Max(AtheneLastSubmitDate) FROM Athene_JaasonEmail group by ContactID18",con)

NW_Submit =  pd.read_sql("SELECT ContactID18, FullName, count(NWLastSubmitDate), Max(NWLastSubmitDate) as NWSubmitDate FROM NW_JasonEmail group by ContactID18",con)

con = sqlite3.connect("Athene_Submit.db")

Athene_Submit.to_sql("Athene_Submit", con, if_exists='replace')

con = sqlite3.connect("NW_Submit.db")

NW_Submit.to_sql("NW_Submit", con, if_exists='replace')


cc5 = """SELECT a.*,b.NWSubmitDate FROM Athene_Submit a LEFT JOIN NW_Submit b on (a.ContactID18= b.ContactID18) ;"""      

Athen_NW =  pysqldf(cc5)

export_csv = Athen_NW.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Campaign\AtheneAdvisors\Athen_NW.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


### Now bring the right files

Athene_LP_NP= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/Sub/AtheneAppAdvisorsNotSellingNWAthene12months_NL_LP.csv',encoding= 'iso-8859-1')


con = sqlite3.connect("Athene_LP_NP.db")

Athene_LP_NP.to_sql("Athene_LP_NP", con, if_exists='replace')

Athene_LP_NP.info()

AFinal.info()

bb6 = """SELECT * FROM Athene_LP_NP a INNER JOIN AFinal b on a.ContactID18= b.AdvisorContactIDText;"""      

### Remove the common one
      
CommKH =  pysqldf(bb6)

con = sqlite3.connect("CommKH.db")

CommKH.to_sql("CommKH", con, if_exists='replace')

m9  = """SELECT * FROM Athene_LP_NP WHERE ContactID18 NOT IN (SELECT ContactID18 FROM CommKH);"""

FinalAthen_NW =  pysqldf(m9)

con = sqlite3.connect("FinalAthen_NW.db")

FinalAthen_NW.to_sql("FinalAthen_NW", con, if_exists='replace')

FinalAthen_NW.info()

### Now Join FinalAthen_NW with Athene_JaasonEmail

am5 = """SELECT a.*, b.Email
      FROM FinalAthen_NW a INNER JOIN Athene_JaasonEmail b on (a.ContactID18= b.ContactID18);"""      

Athene_LP_NP_EmailApp =  pysqldf(am5)


con = sqlite3.connect("Athene_LP_NP_EmailApp.db")

Athene_LP_NP_EmailApp.to_sql("Athene_LP_NP_EmailApp", con, if_exists='replace')

### Bring the file Email_Rem_AlliRIARep_JFinal

ERem_AlliRIARep_JFinal= pd.read_csv('C:/Users/test/Documents/DiscoveryDataFields/Campaign/AtheneAdvisors/JShare/Email_Rem_AlliRIARep_JFinal.csv',encoding= 'iso-8859-1')

ERem_AlliRIARep_JFinal.info()

con = sqlite3.connect("ERem_AlliRIARep_JFinal.db")

ERem_AlliRIARep_JFinal.to_sql("ERem_AlliRIARep_JFinal", con, if_exists='replace')

am9 = """SELECT a.*
      FROM Athene_LP_NP_EmailApp a INNER JOIN ERem_AlliRIARep_JFinal b on (a.ContactID18= b.ContactID18);"""      

RemoveCommon =  pysqldf(am9)

RemoveCommon.info()

con = sqlite3.connect("RemoveCommon.db")

RemoveCommon.to_sql("RemoveCommon", con, if_exists='replace')

### Substract the common ones...

m10  = """SELECT * FROM Athene_LP_NP_EmailApp WHERE ContactID18 NOT IN (SELECT ContactID18 FROM RemoveCommon);"""

Athene_LP_NP_EmailApp_Final =  pysqldf(m10)

export_csv = Athene_LP_NP_EmailApp_Final.to_csv (r'C:\Users\test\Documents\DiscoveryDataFields\Campaign\AtheneAdvisors\Athene_LP_NP_EmailApp_Final.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



