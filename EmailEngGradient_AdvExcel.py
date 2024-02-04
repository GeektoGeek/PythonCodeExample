
"""
Spyder Editor

This code is written by Dan Sarkar

This is a temporary script file.

### Make sure to install pandasql 

pip install -U pandasql

### Make sure to install tensorflow
pip install tensorflow

"""
import tensorflow as ts
import pandas as pd
import seaborn as sns
import numpy as np
import pandasql as ps
import pandas as pd
import sqlite3
import pandas.io.sql as psql
ps = lambda q: sqldf(q, globals())
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


# This line tells the notebook to show plots inside of the notebook

sess = ts.Session()
a = ts.constant(10)
b = ts.constant(32)
print(sess.run(a+b))


data = pd.read_csv('C:/Users/test/Documents/EmailEngagement/EmailAnalysis-Round2/IDCDataPython.csv')

data.info()

#df['event']=df['expiration_date'].isnull().astype('int').

con = sqlite3.connect("IDCDataPy.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
data.to_sql("IDCDataPy", con, if_exists='replace')

AdExcel_Mar_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Advisors Excel' and Category = 'Marketer') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = AdExcel_Mar_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\AdExcel_Mar_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


AdExcel_AssMar_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Advisors Excel' and Category = 'Associate Marketer') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = AdExcel_AssMar_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\AdExcel_AssMar_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


AdExcel_CaseDesign_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Advisors Excel' and Category = 'Case Design') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = AdExcel_CaseDesign_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\AdExcel_CaseDesign_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

### Gradient

Gradient_Mar_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Marketer') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)


export_csv = Gradient_Mar_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\Gradient_Mar_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


Gradient_AssMar_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Associate Marketer') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)


export_csv = Gradient_AssMar_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\Gradient_AssMar_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


Gradient_CaseManager_EmailSub = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Case Manager') group by Name, EmailSubject, SendMonthandYear order by SendMonthandYear desc ",con)


export_csv = Gradient_CaseManager_EmailSub.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat\Grdient_CaseManager_EmailSub.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

##### Engagement Rank Gradient 

##############################################

Grad_Mar_EngRank = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, Account_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Marketer') group by Name, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = Grad_Mar_EngRank.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat_Engagement\Grad_Mar_EngRank.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


Grad_AssMar_EngRank = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, Account_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Associate Marketer') group by Name, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = Grad_AssMar_EngRank.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat_Engagement\Grad_AssMar_EngRank.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


Grad_CaseManager_EngRank = pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, EmailSubject, Account_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Case Manager') group by Name, SendMonthandYear order by SendMonthandYear desc ",con)

##AdExcel_Mar = pd.read_sql("SELECT Name, SendMonthandYear as SendMonth_Year, Category, EmailSubject, Branch_Name__c  FROM IDCDataPy group by Name",con)

export_csv = Grad_CaseManager_EngRank.to_csv (r'C:\Users\test\Documents\EmailEngagement\EmailAnalysis-Round2\Gradient_AdvisorExcel_3Cat_Engagement\Grad_CaseManager_EngRank.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


##### Engagement Rank Advisor Excel 

##############################################

# test= pd.read_sql("SELECT distinct(Name), SendMonthandYear as SendMonth_Year, sum(TotalSent) as TotalSent, sum(TotalEngaged) as TotalEngaged, Category, Account_Name__c  FROM IDCDataPy where (Account_Name__c='Gradient Insurance Brokerage' and Category = 'Associate Marketer') group by Name, SendMonthandYear order by SendMonthandYear desc ",con)

IDC_Sub = pd.read_sql("SELECT SubscriberKey, Account_Name__c, Category, Name, SendDate, SendMonthandYear FROM IDCDataPy ",con)


IDC_Email = pd.read_sql("SELECT SubscriberKey, JobId, EmailSubject, TotalSent, TotalEngaged, Account_Name__c  FROM IDCDataPy ",con)


q  = """SELECT sum(b.TotalEngaged) as SumEngaged,  a.SubscriberKey, a.Name, a.SendDate, a.SendMonthandYear, a.Account_Name__c, a.Category, b.JobId, sum(b.TotalSent) as SumSend, b.EmailSubject, b.TotalSent, b.TotalEngaged
           FROM IDC_Sub a
           LEFT JOIN
           IDC_Email b
           ON a.SubscriberKey = b.SubscriberKey group by b.EmailSubject, a.Name;"""
            
 
merge_df = ps(q)  

con = sqlite3.connect("merge_df.db")

# insert the data from dataframe into database so that if you run multiple times it will be replaced and create a table
merge_df.to_sql("merge_df", con, if_exists='replace')

count_Email = pd.read_sql("SELECT  SumEngaged, SumSend, Name, Account_Name__c, SendMonthandYear  FROM merge_df where SumEngaged =0 group by Name order by SumSend desc ",con)



           